"""
MLAUNetBot3D：在 PlainConvUNet 的瓶颈层插入 Multi-head Latent Attention (MLA) 块。

架构动机（参考 DeepSeek-V2, 2024）：
    标准 self-attention 的 K、V 各需 O(N × d_model) 显存；
    MLA 先把 x 压缩到低秩潜变量 c_kv（维度 d_c << d_model），
    再分别上投影出 K 和 V，KV 存储从 O(N × d_model) 降至 O(N × d_c)。
    attention 矩阵仍是完整的 N×N（full attention，保留全局感受野），
    参数和激活内存均显著减少，允许在瓶颈层用更细粒度 patch。

插入位置与 UMambaBot3D 完全对称：
    Encoder（卷积，与 nnUNet 一致）
        ↓
    Bottleneck MLA（最深层特征展平 → K MLA Transformer 块 → 还原 3D）
        ↓
    Decoder（卷积 + skip，与 nnUNet 一致）
"""

from __future__ import annotations
from typing import List, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd

from dynamic_network_architectures.architectures.unet import PlainConvUNet


# ──────────────────────────────────────────────────────────────────────────────
# Multi-head Latent Attention
# ──────────────────────────────────────────────────────────────────────────────

class MultiHeadLatentAttention(nn.Module):
    """
    Multi-head Latent Attention（MLA）。
低秩KV压缩注意力
    与标准 MHA 的唯一区别：
        K 和 V 不直接从 x 投影，而是先经共享的"KV 压缩"投影到低秩潜变量
        c_kv ∈ R^{d_c}，再分别上投影到 K、V。
        压缩比 compression_ratio 决定 d_c = d_model // compression_ratio。

    参数量对比（d_model=256, h=8, compression_ratio=4, d_c=64）：
        标准 MHA（W_Q, W_K, W_V, W_O）: 4 × 256² ≈ 262k
        MLA（W_Q, W_DKV, W_UK, W_UV, W_O）: 256²×2 + 256×64×3 ≈ 179k  → 少 32%
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        compression_ratio: int = 4,
        attn_drop: float = 0.0,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        self.num_heads = num_heads
        self.d_head    = d_model // num_heads
        self.d_c       = max(d_model // compression_ratio, num_heads)  # 保证至少 num_heads 维
        self.scale     = self.d_head ** -0.5

        # Q：全维直接投影
        self.W_Q   = nn.Linear(d_model, d_model, bias=False)

        # KV 压缩：x → 低秩潜变量 c_kv
        self.W_DKV = nn.Linear(d_model, self.d_c, bias=False)
        self.norm_c = nn.LayerNorm(self.d_c)

        # KV 上投影：c_kv → K, V（各 d_model）
        self.W_UK  = nn.Linear(self.d_c, d_model, bias=False)
        self.W_UV  = nn.Linear(self.d_c, d_model, bias=False)

        # 输出投影
        self.W_O   = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, C)
        B, N, C = x.shape
        h, d = self.num_heads, self.d_head

        # Q
        q = self.W_Q(x).reshape(B, N, h, d).transpose(1, 2)          # (B, h, N, d)

        # KV via low-rank compression
        c_kv = self.norm_c(self.W_DKV(x))                             # (B, N, d_c)
        k = self.W_UK(c_kv).reshape(B, N, h, d).transpose(1, 2)      # (B, h, N, d)
        v = self.W_UV(c_kv).reshape(B, N, h, d).transpose(1, 2)      # (B, h, N, d)

        # Scaled dot-product attention（full attention，保留全局感受野）
        attn = (q @ k.transpose(-2, -1)) * self.scale                 # (B, h, N, N)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)             # (B, N, C)
        return self.W_O(out)


# ──────────────────────────────────────────────────────────────────────────────
# MoE FFN（DeepSeek V3 风格：shared expert + loss-free bias 均衡）
# ──────────────────────────────────────────────────────────────────────────────

class FFNExpert(nn.Module):
    """单个 FFN 专家：Linear → GELU → Dropout → Linear → Dropout。"""

    def __init__(self, d_model: int, d_ff: int, drop: float = 0.0):
        super().__init__()
        self.fc1  = nn.Linear(d_model, d_ff)
        self.fc2  = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.drop(F.gelu(self.fc1(x)))))


class MoEFFN(nn.Module):
    """

    Mixture-of-Experts FFN（DeepSeek V3 风格）。含loss-free bias负载均衡

    结构：1 个 shared expert（始终激活）+ num_routed_experts 个路由专家，
    每次 forward 只激活 top_k 个路由专家。

    负载均衡：loss-free bias 机制：
        - expert_bias: register_buffer（非 Parameter，不进优化器）
        - 每个 train step 的 forward 末尾自动调用 update_expert_bias()
        - bias 只影响路由选择（topk_idx），gate 权重用原始 scores softmax

    专家宽度：d_ff = d_model * mlp_ratio // 2
        shared + top_k=2 路由专家，共 3 个半宽专家，激活容量约等于原单 FFN。
    """

    def __init__(
        self,
        d_model: int,
        mlp_ratio: int = 4,
        num_routed_experts: int = 4,
        top_k: int = 2,
        drop: float = 0.0,
    ):
        super().__init__()
        d_ff = d_model * mlp_ratio // 2

        self.num_routed_experts = num_routed_experts
        self.top_k = top_k

        self.shared_expert = FFNExpert(d_model, d_ff, drop)
        self.experts = nn.ModuleList([FFNExpert(d_model, d_ff, drop) for _ in range(num_routed_experts)])
        self.router  = nn.Linear(d_model, num_routed_experts, bias=False)

        self.register_buffer('expert_bias',     torch.zeros(num_routed_experts))
        self.register_buffer('expert_load_ema', torch.full((num_routed_experts,), 1.0 / num_routed_experts))

    @torch.no_grad()
    def update_expert_bias(self, topk_idx: torch.Tensor):
        """EMA 估计各专家负载，调整 bias：过载专家 bias 下降，欠载专家 bias 上升。"""
        counts = torch.zeros(self.num_routed_experts, device=topk_idx.device, dtype=torch.float32)
        for e in range(self.num_routed_experts):
            counts[e] = (topk_idx == e).sum()
        load = counts / topk_idx.numel()
        self.expert_load_ema.mul_(0.99).add_(load * 0.01)
        target = 1.0 / self.num_routed_experts
        self.expert_bias.add_((target - self.expert_load_ema) * 1e-3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D)
        B, N, D = x.shape
        x_flat = x.reshape(B * N, D)                                           # (BN, D)

        scores         = self.router(x_flat)                                   # (BN, E)
        routing_scores = scores + self.expert_bias.detach()                    # bias 只影响选谁
        _, topk_idx    = routing_scores.topk(self.top_k, dim=-1)              # (BN, top_k)

        # gate 权重用原始 scores（无 bias），保证梯度路径干净
        gate_weights = F.softmax(scores.gather(-1, topk_idx), dim=-1)         # (BN, top_k)

        # 全算所有路由专家输出（bottleneck N 极小），再按 topk_idx gather
        expert_outs = torch.stack([e(x_flat) for e in self.experts], dim=1)   # (BN, E, D)
        selected    = expert_outs.gather(
            1, topk_idx.unsqueeze(-1).expand(-1, -1, D)
        )                                                                       # (BN, top_k, D)
        routed = (gate_weights.unsqueeze(-1) * selected).sum(dim=1)           # (BN, D)

        out = (self.shared_expert(x_flat) + routed).reshape(B, N, D)

        if self.training:
            self.update_expert_bias(topk_idx.detach())

        return out


class MLATransformerBlock(nn.Module):
    """
    Pre-LN Transformer block，attention 替换为 MLA。
    use_moe=True：FFN 替换为 MoE-FFN（DeepSeek V3 风格）。
    use_moe=False：FFN 为简单 MLP，兼容旧 checkpoint（mlp.0/mlp.3 命名）。
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        compression_ratio: int = 4,
        mlp_ratio: int = 4,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        use_moe: bool = True,
    ):
        super().__init__()
        self.norm1    = nn.LayerNorm(d_model)
        self.attn     = MultiHeadLatentAttention(d_model, num_heads, compression_ratio, attn_drop)
        self.norm2    = nn.LayerNorm(d_model)
        self.use_moe  = use_moe
        if use_moe:
            self.ffn  = MoEFFN(d_model, mlp_ratio, drop=proj_drop)
        else:
            self.mlp  = nn.Sequential(
                nn.Linear(d_model, d_model * mlp_ratio),
                nn.GELU(),
                nn.Dropout(proj_drop),
                nn.Linear(d_model * mlp_ratio, d_model),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        if self.use_moe:
            x = x + self.ffn(self.norm2(x))
        else:
            x = x + self.mlp(self.norm2(x))
        return x


# ──────────────────────────────────────────────────────────────────────────────
# 瓶颈 MLA 封装（3D 特征图 ↔ 序列）
# ──────────────────────────────────────────────────────────────────────────────

class MLABottleneck3D(nn.Module):
    """
    在 3D 体素特征图上应用 MLA Transformer。

    输入：(B, C, D, H, W)
    处理：展平为 (B, D*H*W, C) → K 个 MLA block → 还原为 (B, C, D, H, W)
    输出：(B, C, D, H, W)，形状不变
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        num_blocks: int = 2,
        compression_ratio: int = 4,
        mlp_ratio: int = 4,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        use_moe: bool = True,
    ):
        super().__init__()
        # 保证 num_heads 能整除 d_model（某些 plan 瓶颈通道数不是 8 的整倍数）
        while d_model % num_heads != 0 and num_heads > 1:
            num_heads //= 2

        self.blocks = nn.Sequential(*[
            MLATransformerBlock(d_model, num_heads, compression_ratio, mlp_ratio, attn_drop, proj_drop, use_moe=use_moe)
            for _ in range(num_blocks)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, *spatial = x.shape
        N = 1
        for s in spatial:
            N *= s

        x_seq = x.flatten(2).transpose(1, 2)      # (B, N, C)
        x_seq = self.blocks(x_seq)
        x_seq = self.norm(x_seq)
        return x_seq.transpose(1, 2).reshape(B, C, *spatial).contiguous()


# ──────────────────────────────────────────────────────────────────────────────
# MLAUNetBot3D：PlainConvUNet + bottleneck MLA
# ──────────────────────────────────────────────────────────────────────────────

class MLAUNetBot3D(PlainConvUNet):
    """
    最终网络类
    MLA-UNet Bot 3D：PlainConvUNet + 瓶颈 MLA Transformer 块。

    与 PlainConvUNet 构造参数完全相同，额外支持四个 MLA 超参：
        mla_num_heads        多头注意力头数（默认 8，自动向下对齐到能整除 d_model）
        mla_num_blocks       MLA Transformer 层数（默认 2）
        mla_compression_ratio  KV 压缩比 d_c = d_model // ratio（默认 4）
        mla_mlp_ratio        FFN 扩展倍数（默认 4）
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
        norm_op_kwargs: dict = None,  # type: ignore
        dropout_op=None,
        dropout_op_kwargs: dict = None,  # type: ignore
        nonlin=None,
        nonlin_kwargs: dict = None,  # type: ignore
        deep_supervision: bool = False,
        nonlin_first: bool = False,
        mla_num_heads: int = 8,
        mla_num_blocks: int = 2,
        mla_compression_ratio: int = 4,
        mla_mlp_ratio: int = 4,
        mla_use_moe: bool = True,
    ):
        super().__init__(
            input_channels, n_stages, features_per_stage, conv_op, kernel_sizes,
            strides, n_conv_per_stage, num_classes, n_conv_per_stage_decoder,
            conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs,
            nonlin, nonlin_kwargs, deep_supervision, nonlin_first,
        )
        d_bot = self.encoder.output_channels[-1]
        self.mla_bot = MLABottleneck3D(
            d_bot, mla_num_heads, mla_num_blocks, mla_compression_ratio, mla_mlp_ratio,
            use_moe=mla_use_moe,
        )

    def forward(self, x: torch.Tensor):
        skips = self.encoder(x)
        skips[-1] = self.mla_bot(skips[-1])
        return self.decoder(skips)


# ──────────────────────────────────────────────────────────────────────────────
# Depthwise Separable Conv + MLAUNetDWBot3D
# ──────────────────────────────────────────────────────────────────────────────

class DWSepConv3d(nn.Module):
    """
    3D Depthwise Separable Conv：depthwise (groups=C_in) + pointwise (1×1×1)。

    外部接口与 nn.Conv3d 完全相同，可直接替换。
    参数量：k³·C_in + C_in·C_out  vs 标准 k³·C_in·C_out
    C 足够大时参数显著减少，同等预算下支持更大卷积核（更大感受野）。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups: int = 1,        # 忽略，depthwise 固定 groups=in_channels
        bias: bool = False,
        padding_mode: str = 'zeros',
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory = {}
        if device is not None:
            factory['device'] = device
        if dtype is not None:
            factory['dtype'] = dtype

        self.dw = nn.Conv3d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=in_channels, bias=False,
            padding_mode=padding_mode, **factory,
        )
        self.pw = nn.Conv3d(in_channels, out_channels, 1, bias=bias, **factory)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.dw(x))


def _replace_encoder_conv3d_with_dw_sep(module: nn.Module, dw_kernel_size: int = 5) -> None:
    """
    递归把 module 内所有 nn.Conv3d 替换为更大核的 DWSepConv3d（in-place）。

    替换规则：
      - 1×1×1 卷积跳过（无空间混合，没有感受野可扩大）
      - 各维度中原来 >= 3 的维度扩大到 dw_kernel_size；原来 == 1 的维度保持 1
      - 新 padding 自动按 (k-1)//2 计算，保持 stride=1 时空间尺寸不变
      - stride 保持原值（保留下采样语义）
    """
    for name, child in module.named_children():
        if isinstance(child, nn.Conv3d):
            orig_k = child.kernel_size          # tuple, e.g. (3,3,3) 或 (1,3,3)
            if all(k == 1 for k in orig_k):     # 1×1×1：跳过
                continue
            # 只扩大空间维度（原来 >= 3 的维度），保留各向异性中的 1
            new_k       = tuple(dw_kernel_size if k >= 3 else k for k in orig_k)
            new_padding = tuple((k - 1) // 2    for k in new_k)
            setattr(module, name, DWSepConv3d(
                in_channels=child.in_channels,
                out_channels=child.out_channels,
                kernel_size=new_k,
                stride=child.stride,
                padding=new_padding,
                dilation=child.dilation,
                bias=child.bias is not None,
                padding_mode=child.padding_mode,
            ))
        else:
            _replace_encoder_conv3d_with_dw_sep(child, dw_kernel_size)


class IBConvBlock3D(nn.Module):
    """
    倒置瓶颈卷积块（Inverted Bottleneck Conv Block，MedNeXt / ConvNeXt 思路）。

    结构（stride 永远为 1）：
        DW(k×k×k, groups=C_in) → InstanceNorm3d → PW_expand(1×1×1) → GELU
        → PW_compress(1×1×1) → + residual

    设计原则：
      - 大核 ONLY 在 depthwise 空间混合，stride=1，cuDNN 可用 Winograd 加速
      - pointwise 1×1×1 做通道变换，充分利用 Tensor Core
      - stride=2 下采样在本块外由标准 ConvDropoutNormReLU 处理
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 7, exp_r: int = 4, channels_per_group: int = 1,
                 use_checkpoint: bool = False):
        super().__init__()
        # channels_per_group=1 → depthwise (groups=C, M=1)
        # channels_per_group=16 → grouped (groups=C/16, M=16, Tensor Core 可用)
        groups = max(1, in_channels // channels_per_group)
        self.dw   = nn.Conv3d(in_channels, in_channels, kernel_size,
                              padding=kernel_size // 2, groups=groups, bias=False)
        self.norm = nn.InstanceNorm3d(in_channels, affine=True)

        mid = exp_r * min(in_channels, out_channels)
        self.pw_up   = nn.Conv3d(in_channels, mid, 1, bias=False)
        self.act     = nn.GELU()
        self.pw_down = nn.Conv3d(mid, out_channels, 1, bias=False)

        self.proj = (nn.Conv3d(in_channels, out_channels, 1, bias=False)
                     if in_channels != out_channels else nn.Identity())
        self.use_checkpoint = use_checkpoint

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.proj(x)
        x = self.norm(self.dw(x))
        x = self.act(self.pw_up(x))
        x = self.pw_down(x)
        return x + residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(
                self._forward_impl, x, use_reentrant=False
            )
        return self._forward_impl(x)


def _replace_cdnr_with_ib(module: nn.Module,
                           ib_kernel_size: int = 7,
                           exp_r: int = 4,
                           channels_per_group: int = 1,
                           use_checkpoint: bool = False) -> None:
    """
    递归将所有 stride=1 的 ConvDropoutNormReLU 替换为 IBConvBlock3D。
    stride=2（下采样）保持不变——大核步长卷积 cuDNN 无优化。
    """
    for name, child in list(module.named_children()):
        if type(child).__name__ == 'ConvDropoutNormReLU':
            conv = child.conv
            if all(s == 1 for s in conv.stride):
                setattr(module, name, IBConvBlock3D(
                    in_channels=conv.in_channels,
                    out_channels=conv.out_channels,
                    kernel_size=ib_kernel_size,
                    exp_r=exp_r,
                    channels_per_group=channels_per_group,
                    use_checkpoint=use_checkpoint,
                ))
            # stride=2: 保持原样
        else:
            _replace_cdnr_with_ib(child, ib_kernel_size, exp_r, channels_per_group, use_checkpoint)


class MLAUNetIBBot3D(MLAUNetBot3D):
    """
    MLAUNetBot3D + 倒置瓶颈卷积块（Inverted Bottleneck，MedNeXt 架构思路）。

    将 Encoder + Decoder 所有 stride=1 ConvDropoutNormReLU 替换为 IBConvBlock3D：
        DW(k×k×k) → InstanceNorm3d → PW_expand(×exp_r) → GELU → PW_compress → + residual

    stride=2 下采样保持标准 3×3×3 卷积不变。
    MLA Bottleneck 不变。

    与 MLAUNetDWBot3D（简单 DWSep 替换）的核心区别：
      - 大核 ONLY 在 stride=1 depthwise，下采样不受影响
      - expand→GELU→compress 增强非线性（类 ConvNeXt 设计）
      - 每个块有独立 residual connection（等效于残差大核卷积）
    """

    def __init__(self, *args, ib_kernel_size: int = 7, exp_r: int = 4,
                 channels_per_group: int = 1, use_checkpoint: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        _replace_cdnr_with_ib(self.encoder, ib_kernel_size, exp_r, channels_per_group, use_checkpoint)
        _replace_cdnr_with_ib(self.decoder, ib_kernel_size, exp_r, channels_per_group, use_checkpoint)


class DWSepUNet3D(PlainConvUNet):
    """
    PlainConvUNet + DWSepConv3d 替换（纯大核深度可分离卷积，无 MLA，无 MoE）。

    将 Encoder + Decoder 所有 nn.Conv3d（含 stride=2 下采样）替换为 DWSepConv3d：
        DW(k×k×k, groups=C) → PW(1×1×1)

    1×1×1 卷积跳过；各向异性（某维度 k=1）保持不变。
    用于独立消融"大核感受野"的贡献，不引入倒置瓶颈或非线性结构。
    """

    def __init__(self, *args, dw_kernel_size: int = 7, **kwargs):
        super().__init__(*args, **kwargs)
        _replace_encoder_conv3d_with_dw_sep(self.encoder, dw_kernel_size)
        _replace_encoder_conv3d_with_dw_sep(self.decoder, dw_kernel_size)


class IBConvUNet3D(PlainConvUNet):
    """
    PlainConvUNet + IBConvBlock3D（纯大核倒置瓶颈替换，无 MLA，无 MoE）。

    将 Encoder + Decoder 所有 stride=1 ConvDropoutNormReLU 替换为 IBConvBlock3D：
        DW(k×k×k, groups=C) → InstanceNorm3d → PW_expand(×exp_r) → GELU → PW_compress → + residual

    stride=2 下采样保持标准 3×3×3 不变。
    用于独立消融大核卷积的贡献，去除 MLA/MoE 的干扰。
    """

    def __init__(self, *args, ib_kernel_size: int = 7, exp_r: int = 4,
                 channels_per_group: int = 1, use_checkpoint: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        _replace_cdnr_with_ib(self.encoder, ib_kernel_size, exp_r, channels_per_group, use_checkpoint)
        _replace_cdnr_with_ib(self.decoder, ib_kernel_size, exp_r, channels_per_group, use_checkpoint)


class MLAUNetDWBot3D(MLAUNetBot3D):
    """
    MLAUNetBot3D + 全网络大核 Depthwise Separable Conv（MedNeXt 思路）。

    核心思想：
      标准 Conv3d 参数量 = k³ · C_in · C_out
      DW Sep Conv 参数量 = k³ · C_in  +  C_in · C_out    （消去 k³ 与 C_out 的耦合）

      节省出的参数预算用来把 k 从 3 扩大到 7：
        k=3 标准:  27  · C_in · C_out
        k=7 DWSep: 343 · C_in + C_in · C_out  （C=128 时参数量仅约原来的 30%）

      → 更大感受野，参数不增反减。

    dw_kernel_size（默认 7）：depthwise 空间核大小。
    replace_decoder（默认 True）：是否同时替换 Decoder（MedNeXt 全替换策略）。
    MLA Bottleneck 保持标准卷积不变。
    """

    def __init__(self, *args, dw_kernel_size: int = 5, replace_decoder: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        _replace_encoder_conv3d_with_dw_sep(self.encoder, dw_kernel_size)
        if replace_decoder:
            _replace_encoder_conv3d_with_dw_sep(self.decoder, dw_kernel_size)


# ──────────────────────────────────────────────────────────────────────────────
# DWSepResBlock3D：DWSep(k=3) + 残差，仅用于 encoder
# ──────────────────────────────────────────────────────────────────────────────

class DWSepResBlock3D(nn.Module):
    """
    Encoder 卷积块：DWSep(k=3) + 残差连接。

    结构（stride 永远为 1）：
        DW(k×k×k, groups=C_in) → InstanceNorm3d → GELU
        → PW(1×1×1, C_in→C_out) → + proj(x) → output

    设计原则：
      - k=3 保证与原 nnUNet plain conv 感受野一致，参数量降为 k³·C + C·C_out
      - DWSep 节省出的容量用于 "多一层"（n_conv_per_stage +1）
      - 每块内显式残差：稳定深层梯度，允许叠更多层
      - stride=2 下采样在外部由原 ConvDropoutNormReLU 处理，不受影响
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.dw   = nn.Conv3d(in_channels, in_channels, kernel_size,
                              padding=pad, groups=in_channels, bias=False)
        self.norm = nn.InstanceNorm3d(in_channels, affine=True)
        self.act  = nn.GELU()
        self.pw   = nn.Conv3d(in_channels, out_channels, 1, bias=False)
        self.proj = (nn.Conv3d(in_channels, out_channels, 1, bias=False)
                     if in_channels != out_channels else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.proj(x)
        x = self.act(self.norm(self.dw(x)))
        return self.pw(x) + residual


def _replace_encoder_cdnr_with_dwsep_res(module: nn.Module, kernel_size: int = 3) -> None:
    """
    递归将 module 内所有 stride=1 的 ConvDropoutNormReLU 替换为 DWSepResBlock3D。
    stride=2（下采样）保持原样。仅对 encoder 调用，decoder 不传入。
    """
    for name, child in list(module.named_children()):
        if type(child).__name__ == 'ConvDropoutNormReLU':
            conv = child.conv
            if all(s == 1 for s in conv.stride):
                setattr(module, name, DWSepResBlock3D(
                    conv.in_channels, conv.out_channels, kernel_size
                ))
        else:
            _replace_encoder_cdnr_with_dwsep_res(child, kernel_size)


class MLAUNetDWSepResBot3D(MLAUNetBot3D):
    """
    MLAUNetBot3D + DWSep(k=3)+残差 encoder + plain conv decoder + MLA 瓶颈。

    改动范围：
      - Encoder stride=1 块 → DWSepResBlock3D（DWSep k=3 + 残差）
      - Decoder 保持 PlainConvUNet 原始 plain conv 不变
      - MLA Bottleneck 不变

    对比 MLAUNetIBBot3D（IBConvBlock3D k=7 全网络）：
      - 卷积核更小（k=3 vs k=7）：感受野与原 nnUNet 一致，深度换感受野
      - encoder/decoder 不对称：decoder 省参，把容量都留给 encoder 深度
      - 残差同样开启，梯度流一致
    """

    def __init__(self, *args, dw_kernel_size: int = 3, **kwargs):
        super().__init__(*args, **kwargs)
        _replace_encoder_cdnr_with_dwsep_res(self.encoder, dw_kernel_size)
