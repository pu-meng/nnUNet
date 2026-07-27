"""
DeepPlainResGN：深层 Plain Conv U-Net + 残差 + GroupNorm

消融实验目标：
    证明 MedNeXt 的性能提升来自"更深 + 残差 + GroupNorm"，
    而非 depthwise separable conv（倒置瓶颈结构）。

架构设计（对齐 MedNeXt-L）：
    - 9 个处理位置：5 encoder stages + 4 decoder stages（与 MedNeXt block_counts=[3,4,8,8,8,8,8,4,3] 一致）
    - 每个 stage 使用 BasicBlockD（plain 3×3 conv + 残差跳连）
    - 归一化：GroupNorm(num_groups=8)，替换 nnUNet 默认的 InstanceNorm
    - 激活：LeakyReLU（与 nnUNet baseline 一致，排除激活函数影响）
    - 通道数：[32, 64, 128, 256, 384]（对应 encoder stage 0-4）
    - block 数：encoder [3,4,4,4,4]，decoder [4,4,4,3]
    - 参数量：~61.1M，与 MedNeXt-L（61.8M）对齐

对照实验链：
    nnUNet_Baseline（6 stages, InstanceNorm, 无残差, 无深度）     → 0.7941
    ↓ +残差 +GroupNorm +深度（9位置，本架构）                    → ?
    MedNeXt-L（9位置, GroupNorm, 残差, DW sep conv）             → 0.8402

若本架构接近 MedNeXt，说明 DW sep conv 不是关键因素。
"""

import functools
import types
import torch
import torch.nn as nn
import torch.utils.checkpoint as _cp
from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet, UNetDecoder
from dynamic_network_architectures.building_blocks.residual import BasicBlockD, StackedResidualBlocks
from pumengyu.architectures.mla_unetr import MLABottleneck3D

# GroupNorm(8, num_channels) — 与 nnUNet norm_op(num_features) 接口兼容
_GroupNorm8 = functools.partial(torch.nn.GroupNorm, 8)

_ARCH_KWARGS = dict(
    n_stages=5,
    features_per_stage=[32, 64, 128, 256, 384],
    conv_op=torch.nn.Conv3d,
    kernel_sizes=[[3, 3, 3]] * 5,
    strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
    n_blocks_per_stage=[3, 4, 4, 4, 4],
    n_conv_per_stage_decoder=[4, 4, 4, 3],
    conv_bias=True,
    norm_op=_GroupNorm8,
    norm_op_kwargs={},
    dropout_op=None,
    dropout_op_kwargs=None,
    nonlin=torch.nn.LeakyReLU,
    nonlin_kwargs={"inplace": True},
    block=BasicBlockD,
)


def _ckpt_forward(self, x):
    """替换 StackedResidualBlocks.forward，对每个 block 启用 gradient checkpointing。"""
    for block in self.blocks:
        x = _cp.checkpoint(block, x, use_reentrant=False)
    return x


def enable_gradient_checkpointing(net: torch.nn.Module) -> None:
    """对网络中所有 StackedResidualBlocks 启用 gradient checkpointing。"""
    for m in net.modules():
        if isinstance(m, StackedResidualBlocks):
            m.forward = types.MethodType(_ckpt_forward, m)


class DeepResGN_MLABot(ResidualEncoderUNet):
    """
    DeepResGN 骨干 + MLA Bottleneck：你真正的贡献架构。

    底座：DeepPlainResGN（深层 plain conv + 残差 + GroupNorm，61M 参数）
    插件：MLABottleneck3D 插在 encoder 最深层（384 ch, 8³ 分辨率）之后

    forward 逻辑：
        encoder(x) → skips         # skips[-1] 是 bottleneck 特征 (B, 384, 8, 8, 8)
        mla_bot(skips[-1])          # 全局依赖建模，原地增强 bottleneck
        decoder(skips)              # 正常解码
    """

    def __init__(
        self,
        *args,
        mla_num_heads: int = 8,
        mla_num_blocks: int = 2,
        mla_compression_ratio: int = 4,
        mla_mlp_ratio: int = 4,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        bottleneck_ch = self.encoder.output_channels[-1]
        self.mla_bot = MLABottleneck3D(
            d_model=bottleneck_ch,
            num_heads=mla_num_heads,
            num_blocks=mla_num_blocks,
            compression_ratio=mla_compression_ratio,
            mlp_ratio=mla_mlp_ratio,
        )

    def forward(self, x):
        skips = self.encoder(x)
        skips[-1] = self.mla_bot(skips[-1])
        return self.decoder(skips)


def build_deep_res_gn_mla(
    num_input_channels: int,
    num_output_channels: int,
    enable_deep_supervision: bool = True,
    use_checkpoint: bool = True,
    mla_num_heads: int = 8,
    mla_num_blocks: int = 2,
    mla_compression_ratio: int = 4,
    mla_mlp_ratio: int = 4,
) -> DeepResGN_MLABot:
    net = DeepResGN_MLABot(
        input_channels=num_input_channels,
        num_classes=num_output_channels,
        deep_supervision=enable_deep_supervision,
        mla_num_heads=mla_num_heads,
        mla_num_blocks=mla_num_blocks,
        mla_compression_ratio=mla_compression_ratio,
        mla_mlp_ratio=mla_mlp_ratio,
        **_ARCH_KWARGS,
    )
    if use_checkpoint:
        enable_gradient_checkpointing(net)
    return net


def build_deep_plain_res_gn(
    num_input_channels: int,
    num_output_channels: int,
    enable_deep_supervision: bool = True,
    use_checkpoint: bool = False,
) -> ResidualEncoderUNet:
    net = ResidualEncoderUNet(
        input_channels=num_input_channels,
        num_classes=num_output_channels,
        deep_supervision=enable_deep_supervision,
        **_ARCH_KWARGS,
    )
    if use_checkpoint:
        enable_gradient_checkpointing(net)
    return net


# ─────────────────────────────────────────────────────────────────────────────
# DeepDWIBResGN：DW + Inverted Bottleneck 消融架构
#
# 消融目标：
#   DeepPlainResGN（plain 3×3, GN, residual）→ 0.7966
#   DeepDWIBResGN（DW+IB,    GN, residual）→ ?
#   唯一变量 = block 类型（plain vs DW+IB）
#   若 DeepDWIBResGN ≈ MedNeXt → 证明 DW+IB 是可迁移的有效成分
#
# 架构参数：
#   features=[32,64,128,256,512], enc blocks=[3,4,8,8,8], r=8, ~58.12M
# ─────────────────────────────────────────────────────────────────────────────

class DWIBBlock(nn.Module):
    """
    ConvNeXt-v1 风格的 3D DW+IB 残差块。

    结构（无 stride/channel 变化时）：
        Input → DW Conv3d(k, groups=C) → GN → PW(C→C*r) → GELU → PW(C*r→C) → + Input

    结构（有 stride 或 channel 变化时）：
        Main:  Input → [1×1 strided proj] → DW(k, stride=1) → GN → PW expand → GELU → PW compress
        Skip:  Input → [1×1 strided conv + GN]
        Output: main + skip

    接口与 BasicBlockD 完全兼容（StackedResidualBlocks 的 block 签名）。
    nonlin / nonlin_kwargs / stochastic_depth_p / squeeze_excitation 参数接受但忽略。
    """

    def __init__(
        self,
        conv_op,
        input_channels: int,
        output_channels: int,
        kernel_size,
        stride,
        conv_bias: bool = False,
        norm_op=None,
        norm_op_kwargs: dict = None,
        dropout_op=None,
        dropout_op_kwargs=None,
        nonlin=None,
        nonlin_kwargs=None,
        stochastic_depth_p: float = 0.0,
        squeeze_excitation: bool = False,
        squeeze_excitation_reduction_ratio: float = 1 / 16,
        expansion_ratio: int = 8,
    ):
        super().__init__()
        norm_op_kwargs = norm_op_kwargs or {}

        if isinstance(stride, int):
            stride = (stride,) * 3
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        stride = tuple(stride)
        padding = tuple(k // 2 for k in kernel_size)

        needs_change = (input_channels != output_channels) or any(s > 1 for s in stride)

        if needs_change:
            self.proj = conv_op(input_channels, output_channels, 1,
                                stride=stride, bias=conv_bias)
        else:
            self.proj = None

        self.dw   = conv_op(output_channels, output_channels, kernel_size,
                            stride=1, padding=padding,
                            groups=output_channels, bias=conv_bias)
        self.norm = norm_op(output_channels, **norm_op_kwargs)

        hidden = output_channels * expansion_ratio
        self.pw_expand   = conv_op(output_channels, hidden, 1, bias=conv_bias)
        self.act         = nn.GELU()
        self.pw_compress = conv_op(hidden, output_channels, 1, bias=conv_bias)

        if needs_change:
            self.shortcut = nn.Sequential(
                conv_op(input_channels, output_channels, 1, stride=stride, bias=conv_bias),
                norm_op(output_channels, **norm_op_kwargs),
            )
        else:
            self.shortcut = None

    def forward(self, x):
        residual = x
        if self.proj is not None:
            x = self.proj(x)
        x = self.dw(x)
        x = self.norm(x)
        x = self.pw_expand(x)
        x = self.act(x)
        x = self.pw_compress(x)
        if self.shortcut is not None:
            residual = self.shortcut(residual)
        return x + residual


class DWIBEncoder(nn.Module):
    """
    由 DWIBBlock 构成的 3D U-Net encoder，暴露 UNetDecoder 所需的所有属性。

    Attributes（UNetDecoder 读取）：
        output_channels / strides / conv_op / conv_bias /
        norm_op / norm_op_kwargs / dropout_op / dropout_op_kwargs /
        nonlin / nonlin_kwargs / kernel_sizes
    """

    def __init__(
        self,
        input_channels: int,
        features_per_stage: list,
        conv_op,
        kernel_sizes: list,
        strides: list,
        n_blocks_per_stage: list,
        expansion_ratio: int = 8,
        conv_bias: bool = True,
        norm_op=None,
        norm_op_kwargs: dict = None,
    ):
        super().__init__()
        norm_op_kwargs = norm_op_kwargs or {}
        n_stages = len(features_per_stage)

        self.output_channels   = features_per_stage
        self.strides           = strides
        self.conv_op           = conv_op
        self.conv_bias         = conv_bias
        self.norm_op           = norm_op
        self.norm_op_kwargs    = norm_op_kwargs
        self.dropout_op        = None
        self.dropout_op_kwargs = {}
        self.nonlin            = nn.GELU
        self.nonlin_kwargs     = {}
        self.kernel_sizes      = kernel_sizes

        self.stages = nn.ModuleList()
        in_ch = input_channels
        for i in range(n_stages):
            out_ch   = features_per_stage[i]
            stride_i = strides[i]
            ks_i     = kernel_sizes[i]
            nb       = n_blocks_per_stage[i]
            blocks = [
                DWIBBlock(
                    conv_op=conv_op,
                    input_channels=in_ch if j == 0 else out_ch,
                    output_channels=out_ch,
                    kernel_size=ks_i,
                    stride=stride_i if j == 0 else 1,
                    conv_bias=conv_bias,
                    norm_op=norm_op,
                    norm_op_kwargs=norm_op_kwargs,
                    expansion_ratio=expansion_ratio,
                )
                for j in range(nb)
            ]
            self.stages.append(nn.Sequential(*blocks))
            in_ch = out_ch

    def forward(self, x):
        skips = []
        for stage in self.stages:
            x = stage(x)
            skips.append(x)
        return skips


def enable_gradient_checkpointing_dwib(encoder: DWIBEncoder) -> None:
    """对 DWIBEncoder 每个 stage 的每个 DWIBBlock 启用 gradient checkpointing。"""
    for stage in encoder.stages:
        for block in stage:
            if isinstance(block, DWIBBlock):
                orig_fwd = block.forward
                def make_ckpt(f):
                    def ckpt_fwd(x):
                        return _cp.checkpoint(f, x, use_reentrant=False)
                    return ckpt_fwd
                block.forward = make_ckpt(orig_fwd)


class DeepDWIBNet(nn.Module):
    """
    DeepDWIBResGN 完整网络：DWIBEncoder + UNetDecoder（plain-conv decoder）。

    参数量 ≈ 58.12M（encoder DW+IB; decoder plain-conv）。
    """

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        deep_supervision: bool = True,
        expansion_ratio: int = 8,
        n_conv_per_stage_decoder: list = None,
        use_checkpoint: bool = True,
    ):
        super().__init__()
        if n_conv_per_stage_decoder is None:
            n_conv_per_stage_decoder = [4, 4, 4, 3]

        self.encoder = DWIBEncoder(
            input_channels=input_channels,
            features_per_stage=[32, 64, 128, 256, 512],
            conv_op=nn.Conv3d,
            kernel_sizes=[[3, 3, 3]] * 5,
            strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
            n_blocks_per_stage=[3, 4, 8, 8, 8],
            expansion_ratio=expansion_ratio,
            conv_bias=True,
            norm_op=_GroupNorm8,
            norm_op_kwargs={},
        )
        if use_checkpoint:
            enable_gradient_checkpointing_dwib(self.encoder)

        self.decoder = UNetDecoder(
            encoder=self.encoder,
            num_classes=num_classes,
            n_conv_per_stage=n_conv_per_stage_decoder,
            deep_supervision=deep_supervision,
        )

    def forward(self, x):
        skips = self.encoder(x)
        return self.decoder(skips)


def build_deep_dwib_res_gn(
    num_input_channels: int,
    num_output_channels: int,
    enable_deep_supervision: bool = True,
    expansion_ratio: int = 8,
    n_conv_per_stage_decoder: list = None,
    use_checkpoint: bool = True,
) -> DeepDWIBNet:
    """构建 DeepDWIBResGN（DW+IB encoder，~58.12M 参数）。"""
    return DeepDWIBNet(
        input_channels=num_input_channels,
        num_classes=num_output_channels,
        deep_supervision=enable_deep_supervision,
        expansion_ratio=expansion_ratio,
        n_conv_per_stage_decoder=n_conv_per_stage_decoder,
        use_checkpoint=use_checkpoint,
    )


# ─────────────────────────────────────────────────────────────────────────────
# DeepDWIBMedConfig：显式 MedNeXt-L 配置对齐的独立复现
#
# 目的：
#   对齐论文/结构表中显式可见的配置：
#       channels     = 32,64,128,256,512,256,128,64,32
#       block_counts = [3,4,8,8,8,8,8,4,3]
#       exp_r        = [3,4,8,8,8,8,8,4,3]
#   但不复用官方 MedNeXt 代码，也不一比一复刻官方 block/up/down 的内部细节。
#
#   该模型用于检验：只靠显式结构信息 + DW/IB 设计思想，是否能独立复现
#   MedNeXt 级别性能。
# ─────────────────────────────────────────────────────────────────────────────


def _match_spatial_shape(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Pad/crop x so its spatial size matches ref. Keeps implementation robust to odd sizes."""
    target = ref.shape[2:]
    current = x.shape[2:]

    pads = []
    for cur, tar in reversed(list(zip(current, target))):
        diff = tar - cur
        left = max(diff // 2, 0)
        right = max(diff - left, 0)
        pads.extend([left, right])
    if any(pads):
        x = torch.nn.functional.pad(x, pads)

    slices = [slice(None), slice(None)]
    for dim, tar in enumerate(target, start=2):
        cur = x.shape[dim]
        if cur == tar:
            slices.append(slice(None))
        else:
            start = max((cur - tar) // 2, 0)
            slices.append(slice(start, start + tar))
    return x[tuple(slices)]


class DWIBMedConfigNet(nn.Module):
    """
    MedNeXt-L 显式配置对齐版的自写 DW+IB U-Net。

    保留为“独立复现”而非官方复刻：
      - encoder/decoder 都使用本文件 DWIBBlock；
      - 下采样使用 stride=2 的 DWIBBlock；
      - 上采样使用常规 ConvTranspose3d + residual DWIB blocks；
      - skip fusion 用 add；
      - deep supervision 输出顺序对齐 nnUNet: [full, 1/2, 1/4, 1/8, 1/16]。
    """

    CHANNELS = [32, 64, 128, 256, 512]
    BLOCK_COUNTS = [3, 4, 8, 8, 8, 8, 8, 4, 3]
    EXP_R = [3, 4, 8, 8, 8, 8, 8, 4, 3]

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        deep_supervision: bool = True,
        use_checkpoint: bool = True,
    ):
        super().__init__()
        self.do_ds = deep_supervision
        self.use_checkpoint = use_checkpoint

        c = self.CHANNELS
        bc = self.BLOCK_COUNTS
        er = self.EXP_R

        self.stem = nn.Conv3d(input_channels, c[0], kernel_size=1, bias=True)

        self.enc_block_0 = self._make_stage(c[0], c[0], bc[0], er[0], first_stride=1)
        self.down_0 = DWIBBlock(nn.Conv3d, c[0], c[1], 3, 2, True, _GroupNorm8, {}, expansion_ratio=er[1])

        self.enc_block_1 = self._make_stage(c[1], c[1], bc[1], er[1], first_stride=1)
        self.down_1 = DWIBBlock(nn.Conv3d, c[1], c[2], 3, 2, True, _GroupNorm8, {}, expansion_ratio=er[2])

        self.enc_block_2 = self._make_stage(c[2], c[2], bc[2], er[2], first_stride=1)
        self.down_2 = DWIBBlock(nn.Conv3d, c[2], c[3], 3, 2, True, _GroupNorm8, {}, expansion_ratio=er[3])

        self.enc_block_3 = self._make_stage(c[3], c[3], bc[3], er[3], first_stride=1)
        self.down_3 = DWIBBlock(nn.Conv3d, c[3], c[4], 3, 2, True, _GroupNorm8, {}, expansion_ratio=er[4])

        self.bottleneck = self._make_stage(c[4], c[4], bc[4], er[4], first_stride=1)

        self.up_3 = nn.ConvTranspose3d(c[4], c[3], kernel_size=2, stride=2, bias=True)
        self.dec_block_3 = self._make_stage(c[3], c[3], bc[5], er[5], first_stride=1)

        self.up_2 = nn.ConvTranspose3d(c[3], c[2], kernel_size=2, stride=2, bias=True)
        self.dec_block_2 = self._make_stage(c[2], c[2], bc[6], er[6], first_stride=1)

        self.up_1 = nn.ConvTranspose3d(c[2], c[1], kernel_size=2, stride=2, bias=True)
        self.dec_block_1 = self._make_stage(c[1], c[1], bc[7], er[7], first_stride=1)

        self.up_0 = nn.ConvTranspose3d(c[1], c[0], kernel_size=2, stride=2, bias=True)
        self.dec_block_0 = self._make_stage(c[0], c[0], bc[8], er[8], first_stride=1)

        self.out_0 = nn.Conv3d(c[0], num_classes, kernel_size=1, bias=True)
        self.out_1 = nn.Conv3d(c[1], num_classes, kernel_size=1, bias=True)
        self.out_2 = nn.Conv3d(c[2], num_classes, kernel_size=1, bias=True)
        self.out_3 = nn.Conv3d(c[3], num_classes, kernel_size=1, bias=True)
        self.out_4 = nn.Conv3d(c[4], num_classes, kernel_size=1, bias=True)

        if use_checkpoint:
            for module in [
                self.enc_block_0, self.enc_block_1, self.enc_block_2, self.enc_block_3,
                self.bottleneck, self.dec_block_3, self.dec_block_2,
                self.dec_block_1, self.dec_block_0,
            ]:
                self._enable_stage_checkpointing(module)

    @staticmethod
    def _make_stage(in_ch: int, out_ch: int, n_blocks: int, exp_r: int, first_stride: int):
        blocks = []
        for i in range(n_blocks):
            blocks.append(DWIBBlock(
                conv_op=nn.Conv3d,
                input_channels=in_ch if i == 0 else out_ch,
                output_channels=out_ch,
                kernel_size=3,
                stride=first_stride if i == 0 else 1,
                conv_bias=True,
                norm_op=_GroupNorm8,
                norm_op_kwargs={},
                expansion_ratio=exp_r,
            ))
        return nn.Sequential(*blocks)

    @staticmethod
    def _enable_stage_checkpointing(stage: nn.Sequential) -> None:
        for block in stage:
            if isinstance(block, DWIBBlock):
                orig_fwd = block.forward

                def make_ckpt(f):
                    def ckpt_fwd(x):
                        return _cp.checkpoint(f, x, use_reentrant=False)
                    return ckpt_fwd

                block.forward = make_ckpt(orig_fwd)

    def forward(self, x):
        x = self.stem(x)

        x_res_0 = self.enc_block_0(x)
        x = self.down_0(x_res_0)
        x_res_1 = self.enc_block_1(x)
        x = self.down_1(x_res_1)
        x_res_2 = self.enc_block_2(x)
        x = self.down_2(x_res_2)
        x_res_3 = self.enc_block_3(x)
        x = self.down_3(x_res_3)

        x = self.bottleneck(x)
        x_ds_4 = self.out_4(x)

        x = self.up_3(x)
        x = _match_spatial_shape(x, x_res_3)
        x = self.dec_block_3(x + x_res_3)
        x_ds_3 = self.out_3(x)

        x = self.up_2(x)
        x = _match_spatial_shape(x, x_res_2)
        x = self.dec_block_2(x + x_res_2)
        x_ds_2 = self.out_2(x)

        x = self.up_1(x)
        x = _match_spatial_shape(x, x_res_1)
        x = self.dec_block_1(x + x_res_1)
        x_ds_1 = self.out_1(x)

        x = self.up_0(x)
        x = _match_spatial_shape(x, x_res_0)
        x = self.dec_block_0(x + x_res_0)
        x = self.out_0(x)

        if self.do_ds:
            return [x, x_ds_1, x_ds_2, x_ds_3, x_ds_4]
        return x


def build_deep_dwib_med_config(
    num_input_channels: int,
    num_output_channels: int,
    enable_deep_supervision: bool = True,
    use_checkpoint: bool = True,
) -> DWIBMedConfigNet:
    """构建显式 MedNeXt-L 配置对齐版自写 DW+IB U-Net。"""
    return DWIBMedConfigNet(
        input_channels=num_input_channels,
        num_classes=num_output_channels,
        deep_supervision=enable_deep_supervision,
        use_checkpoint=use_checkpoint,
    )
