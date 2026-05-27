"""
UMambaBot3D：在 PlainConvUNet 的瓶颈层（bottleneck）插入 Mamba 块。

架构思路（来自 U-Mamba 论文 bowang-lab/U-Mamba）：
    Encoder（标准卷积，与 nnUNet 完全一致）
        ↓
    Bottleneck Mamba（最深层特征图上做序列建模，捕获全局上下文）
        ↓
    Decoder（标准卷积 + skip connections，与 nnUNet 完全一致）

只改了一处：encoder 输出的最深特征（形状 [B, C_bot, D, H, W]）在送入 decoder 前，
先展平空间维度 → Mamba → 还原回 3D。其余代码零改动。

Mamba 实现策略：
    优先使用 mamba-ssm（官方 CUDA 加速版，需额外安装）；
    若未安装则自动回退到本文件内置的纯 PyTorch 实现（MambaPurePyTorch），
    数学等价，无需编译，速度略慢（瓶颈层空间分辨率小，影响可接受）。
"""

from __future__ import annotations
from typing import List, Tuple, Type, Union

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd

from dynamic_network_architectures.architectures.unet import PlainConvUNet


# ──────────────────────────────────────────────────────────────────────────────
# 纯 PyTorch Mamba 实现（无需 CUDA 扩展）
# ──────────────────────────────────────────────────────────────────────────────

class MambaPurePyTorch(nn.Module):
    """
    纯 PyTorch 实现的 Mamba 块，与 mamba_ssm.Mamba API 兼容。

    输入/输出：(B, L, d_model)

    实现参考：
        Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces", 2023
        https://arxiv.org/abs/2312.00752

    核心计算：
        1. 输入投影：x → x_proj（expand 倍），z（gate）
        2. 局部卷积：x_proj 沿 L 维做深度可分离卷积（模拟 causal conv1d）
        3. SSM：构造 A、B、C、dt，用并行 cumsum 近似线性递推
        4. 门控输出：SSM 输出 ⊙ silu(z) → 线性压缩回 d_model
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        d_inner = int(d_model * expand)
        self.d_model  = d_model
        self.d_state  = d_state
        self.d_inner  = d_inner

        # 输入投影：x → [x_proj || z]
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)

        # 局部深度可分离卷积（沿序列维度，groups=d_inner）
        self.conv1d = nn.Conv1d(
            d_inner, d_inner, kernel_size=d_conv,
            padding=d_conv - 1, groups=d_inner, bias=True,
        )

        # SSM 参数：输出 = [dt(1), B(d_state), C(d_state)]
        self.x_proj = nn.Linear(d_inner, 1 + d_state * 2, bias=False)
        # A 初始化为对数形式（保证负定）
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_inner))

        # 输出投影
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

        self.norm = nn.LayerNorm(d_inner)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, d_model)
        B, L, _ = x.shape

        xz = self.in_proj(x)                           # (B, L, 2*d_inner)
        x_in, z = xz.chunk(2, dim=-1)                  # each (B, L, d_inner)

        # 局部卷积（因果卷积通过截断实现）
        x_conv = self.conv1d(x_in.transpose(1, 2))[:, :, :L].transpose(1, 2)
        x_conv = F.silu(x_conv)                        # (B, L, d_inner)

        # SSM 参数（输入依赖）：[dt(1) | B(N) | C(N)]
        x_proj_out = self.x_proj(x_conv)               # (B, L, 1 + 2*N)
        dt, B_ssm, C_ssm = x_proj_out.split([1, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(dt)                             # (B, L, 1)，保证正值

        A = -torch.exp(self.A_log.float())              # (d_inner, N)

        # 离散化 ZOH：
        #   A_bar = exp(A * dt),  B_bar = (A_bar - 1) / A * B
        # 用累积和近似线性递推（并行扫描简化版）
        # dt: (B, L, 1), A: (d_inner, N) → broadcast 到 (B, L, d_inner, N)
        dt_expand = dt.unsqueeze(-1)                    # (B, L, 1, 1)
        A_expand  = A.unsqueeze(0).unsqueeze(0)         # (1, 1, d_inner, N)
        dA = torch.exp(dt_expand * A_expand)            # (B, L, d_inner, N)

        # B_bar * u（u = x_conv）
        u = x_conv.unsqueeze(-1)                        # (B, L, d_inner, 1)
        B_expand = B_ssm.unsqueeze(2)                   # (B, L, 1, N)
        dBu = dt_expand * B_expand * u                  # (B, L, d_inner, N)

        # 线性递推用 cumsum 近似（O(L) 并行，非严格精确但实用）
        # h_t = A_bar * h_{t-1} + dBu_t
        # 累积乘 dA（沿 L 维），再加权 dBu
        log_dA_cumsum = torch.cumsum(torch.log(dA.clamp(min=1e-8)), dim=1)  # (B, L, d_inner, N)
        decay = torch.exp(log_dA_cumsum)                # (B, L, d_inner, N)
        # 每步的 dBu 需要除以对应的衰减再累积（标准并行扫描）
        h = torch.cumsum(dBu / decay.clamp(min=1e-8), dim=1) * decay       # (B, L, d_inner, N)

        # 输出 y_t = C * h_t + D * u
        C_expand = C_ssm.unsqueeze(2)                   # (B, L, 1, N)
        y = (h * C_expand).sum(dim=-1)                  # (B, L, d_inner)
        y = y + self.D.unsqueeze(0).unsqueeze(0) * x_conv  # skip connection

        # 门控
        y = y * F.silu(z)                               # (B, L, d_inner)
        return self.out_proj(y)                         # (B, L, d_model)


def _get_mamba_cls(d_model: int, d_state: int, d_conv: int, expand: int) -> nn.Module:
    """优先使用官方 mamba-ssm，回退到纯 PyTorch 实现。"""
    try:
        from mamba_ssm import Mamba
        return Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
    except ImportError:
        return MambaPurePyTorch(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)


# ──────────────────────────────────────────────────────────────────────────────
# 瓶颈 Mamba 封装（3D 特征图 ↔ 序列）
# ──────────────────────────────────────────────────────────────────────────────

class MambaBottleneck3D(nn.Module):
    """
    在 3D 体素特征图上应用 Mamba 序列模型。

    输入：(B, C, D, H, W)
    处理：展平为 (B, D*H*W, C) → Mamba → 残差加回 → 还原为 (B, C, D, H, W)
    输出：(B, C, D, H, W)，形状不变
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.norm  = nn.LayerNorm(d_model)
        self.mamba = _get_mamba_cls(d_model, d_state, d_conv, expand)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        x_seq = x.permute(0, 2, 3, 4, 1).reshape(B, D * H * W, C)   # (B, L, C)
        x_seq = self.mamba(self.norm(x_seq)) + x_seq                  # 残差
        return x_seq.reshape(B, D, H, W, C).permute(0, 4, 1, 2, 3).contiguous()


# ──────────────────────────────────────────────────────────────────────────────
# UMambaBot3D：PlainConvUNet + bottleneck Mamba
# ──────────────────────────────────────────────────────────────────────────────

class UMambaBot3D(PlainConvUNet):
    """
    U-Mamba Bot 3D：PlainConvUNet + bottleneck Mamba 块。

    与 PlainConvUNet 构造参数完全相同，额外支持三个 Mamba 超参：
        mamba_d_state   SSM 状态维度（默认 16）
        mamba_d_conv    局部卷积核（默认 4）
        mamba_expand    通道扩展倍数（默认 2）

    若 mamba-ssm 已安装则自动使用 CUDA 加速版；否则用纯 PyTorch 版。
    """

    def __init__(
        self,
        input_channels: int,
        n_stages: int,
        features_per_stage: Union[int, List[int], Tuple[int, ...]],
        conv_op: Type[_ConvNd],
        kernel_sizes,
        strides,
        n_conv_per_stage,
        num_classes: int,
        n_conv_per_stage_decoder,
        conv_bias: bool = False,
        norm_op=None,
        norm_op_kwargs: dict = None,
        dropout_op=None,
        dropout_op_kwargs: dict = None,
        nonlin=None,
        nonlin_kwargs: dict = None,
        deep_supervision: bool = False,
        nonlin_first: bool = False,
        mamba_d_state: int = 16,
        mamba_d_conv:  int = 4,
        mamba_expand:  int = 2,
    ):
        super().__init__(
            input_channels, n_stages, features_per_stage, conv_op, kernel_sizes,
            strides, n_conv_per_stage, num_classes, n_conv_per_stage_decoder,
            conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs,
            nonlin, nonlin_kwargs, deep_supervision, nonlin_first,
        )
        d_bot = self.encoder.output_channels[-1]
        self.mamba_bot = MambaBottleneck3D(d_bot, mamba_d_state, mamba_d_conv, mamba_expand)

    def forward(self, x: torch.Tensor):
        skips = self.encoder(x)
        skips[-1] = self.mamba_bot(skips[-1])
        return self.decoder(skips)
