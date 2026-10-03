"""
MedNeXt-L wrapper：从 nnunet_mednext 提取架构，适配 nnUNetv2 的 build_network_architecture 接口。

架构固定为 MedNeXt-L（Large）：
    n_channels=32, kernel_size=3, exp_r=[3,4,8,8,8,8,8,4,3],
    block_counts=[3,4,8,8,8,8,8,4,3], do_res=True, do_res_up_down=True

Deep supervision 输出顺序（do_ds=True）：
    [full_res, 1/2, 1/4, 1/8, 1/16] — 与 nnUNetv2 期望格式一致
"""

import torch
from torch import nn
from torch.utils import checkpoint as _cp
from nnunet_mednext.network_architecture.mednextv1.MedNextV1 import MedNeXt
from nnunet_mednext.network_architecture.mednextv1.blocks import MedNeXtBlock
from pumengyu.architectures.mla_unetr import (
    MLABottleneck3D, MHABottleneck3D, DistanceBiasedMHABottleneck3D,
)

_MEDNEXT_L_KWARGS = dict(
    n_channels=32,
    exp_r=[3, 4, 8, 8, 8, 8, 8, 4, 3],
    kernel_size=3,
    do_res=True,
    do_res_up_down=True,
    block_counts=[3, 4, 8, 8, 8, 8, 8, 4, 3],
    checkpoint_style='outside_block',
    norm_type='group',
    dim='3d',
)


def _match_spatial_shape(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Pad/crop x so its spatial size matches ref. Robust to odd nnUNet patch sizes."""
    target = ref.shape[2:]#ref:(B,C,D,H,W),target:(D,H,W)
    current = x.shape[2:]
#x需要调整的张量,ref是提供目标尺寸的参考张量

    pads = []
    for cur, tar in reversed(list(zip(current, target))):
        #zip(current,target)把对应位置配成一对,list()是把这些配对收集成列表
        #reversed()是从最后一项往前遍历,cur是当前尺寸,tar是目标尺寸
        diff = tar - cur
        left = max(diff // 2, 0)
        right = max(diff - left, 0)
        pads.extend([left, right])#把这个两个数追加到列表末尾,
    if any(pads):
        x = torch.nn.functional.pad(x, pads)#按照pads指定的数量,在x边缘补0,再把结果赋给x


    slices = [slice(None), slice(None)] #slice(None)表示取所有元素,这里是为了保留batch和channel维度,
    
    for idx, tar in enumerate(target):
        dim = idx + 2#加2是因为前面还有B,C两个维度,
        cur = x.shape[dim]
        if cur == tar:
            slices.append(slice(None))#追加全部保留规则,
        else:
            start = max((cur - tar) // 2, 0)
            slices.append(slice(start, start + tar))#slice(1, 4)表示取索引1到3的元素,不包括4
    return x[tuple(slices)]


def _resize_deep_supervision_outputs(outputs, full_shape, deep_supervision_scales):
    if deep_supervision_scales is None:
        return outputs
    resized = [outputs[0]]
    for out, scale in zip(outputs[1:], deep_supervision_scales[1:]):
        target_shape = tuple(max(1, int(round(s * f))) for s, f in zip(scale, full_shape))
        if tuple(out.shape[2:]) != target_shape:
            out = torch.nn.functional.interpolate(
                out, size=target_shape, mode="trilinear", align_corners=False
            )
        resized.append(out)
    return resized


def build_mednext_large(
    num_input_channels: int,
    num_output_channels: int,
    enable_deep_supervision: bool = True,
) -> MedNeXt:#函数返回一个MedNeXt实例,
    # 始终用 deep_supervision=True 建图，确保 out_1/2/3/4 层存在，
    # 使 checkpoint 无论何时都能完整加载（推理时 do_ds=False 不影响输出）。
    net = MedNeXt(
        in_channels=num_input_channels,
        n_classes=num_output_channels,
        deep_supervision=True,#深监督是指:除了最终分割输出,还让解码器中间阶段产生预测,并在训练时参与损失计算;
        **_MEDNEXT_L_KWARGS, #type:ignore
    )#_MEDNEXT_L_KWARGS是保存MedNeXt-L配置的字典,函数中的**表示,把字典展开成一组带名字的参数
    
    net.do_ds = enable_deep_supervision#设置网络对象的属性do_ds,控制forward()时是否产生,返回深监督的多尺度输出.
    return net


class MedNeXtSkipConvE1(MedNeXt):
    """MedNeXt-L E1: apply an identity-initialized 1x1x1 convolution to each encoder skip."""

    def __init__(self, *args, deep_supervision_scales=None, **kwargs):
        super().__init__(*args, **kwargs)
        n_channels = kwargs.get("n_channels", 32)
        self.deep_supervision_scales = deep_supervision_scales
        self.skip_convs = nn.ModuleList(
            [
                nn.Conv3d(channels, channels, kernel_size=1, bias=True)
                for channels in (n_channels, n_channels * 2, n_channels * 4, n_channels * 8)
            ]
        )
        for conv in self.skip_convs:
            nn.init.dirac_(conv.weight)
            nn.init.zeros_(conv.bias)

    def _fuse_skip(
        self, stage: int, encoder_skip: torch.Tensor, decoder_up: torch.Tensor
    ) -> torch.Tensor:
        """E1 control: D_i + Conv1x1x1(E_i), with Conv initialized as identity."""
        return decoder_up + self.skip_convs[stage](encoder_skip)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        if self.outside_block_checkpointing:
            x_res_0 = self.iterative_checkpoint(self.enc_block_0, x)
            x = _cp.checkpoint(self.down_0, x_res_0, self.dummy_tensor)
            x_res_1 = self.iterative_checkpoint(self.enc_block_1, x)
            x = _cp.checkpoint(self.down_1, x_res_1, self.dummy_tensor)
            x_res_2 = self.iterative_checkpoint(self.enc_block_2, x)
            x = _cp.checkpoint(self.down_2, x_res_2, self.dummy_tensor)
            x_res_3 = self.iterative_checkpoint(self.enc_block_3, x)
            x = _cp.checkpoint(self.down_3, x_res_3, self.dummy_tensor)

            x = self.iterative_checkpoint(self.bottleneck, x)
            if self.do_ds:
                x_ds_4 = _cp.checkpoint(self.out_4, x, self.dummy_tensor)

            x_up_3 = _match_spatial_shape(
                _cp.checkpoint(self.up_3, x, self.dummy_tensor), x_res_3
            )
            x = self.iterative_checkpoint(self.dec_block_3, self._fuse_skip(3, x_res_3, x_up_3))
            if self.do_ds:
                x_ds_3 = _cp.checkpoint(self.out_3, x, self.dummy_tensor)

            x_up_2 = _match_spatial_shape(
                _cp.checkpoint(self.up_2, x, self.dummy_tensor), x_res_2
            )
            x = self.iterative_checkpoint(self.dec_block_2, self._fuse_skip(2, x_res_2, x_up_2))
            if self.do_ds:
                x_ds_2 = _cp.checkpoint(self.out_2, x, self.dummy_tensor)

            x_up_1 = _match_spatial_shape(
                _cp.checkpoint(self.up_1, x, self.dummy_tensor), x_res_1
            )
            x = self.iterative_checkpoint(self.dec_block_1, self._fuse_skip(1, x_res_1, x_up_1))
            if self.do_ds:
                x_ds_1 = _cp.checkpoint(self.out_1, x, self.dummy_tensor)

            x_up_0 = _match_spatial_shape(
                _cp.checkpoint(self.up_0, x, self.dummy_tensor), x_res_0
            )
            x = self.iterative_checkpoint(self.dec_block_0, self._fuse_skip(0, x_res_0, x_up_0))
            x = _cp.checkpoint(self.out_0, x, self.dummy_tensor)
        else:
            x_res_0 = self.enc_block_0(x)
            x = self.down_0(x_res_0)
            x_res_1 = self.enc_block_1(x)
            x = self.down_1(x_res_1)
            x_res_2 = self.enc_block_2(x)
            x = self.down_2(x_res_2)
            x_res_3 = self.enc_block_3(x)
            x = self.down_3(x_res_3)

            x = self.bottleneck(x)
            if self.do_ds:
                x_ds_4 = self.out_4(x)

            x_up_3 = _match_spatial_shape(self.up_3(x), x_res_3)
            x = self.dec_block_3(self._fuse_skip(3, x_res_3, x_up_3))
            if self.do_ds:
                x_ds_3 = self.out_3(x)

            x_up_2 = _match_spatial_shape(self.up_2(x), x_res_2)
            x = self.dec_block_2(self._fuse_skip(2, x_res_2, x_up_2))
            if self.do_ds:
                x_ds_2 = self.out_2(x)

            x_up_1 = _match_spatial_shape(self.up_1(x), x_res_1)
            x = self.dec_block_1(self._fuse_skip(1, x_res_1, x_up_1))
            if self.do_ds:
                x_ds_1 = self.out_1(x)

            x_up_0 = _match_spatial_shape(self.up_0(x), x_res_0)
            x = self.dec_block_0(self._fuse_skip(0, x_res_0, x_up_0))
            x = self.out_0(x)

        if self.do_ds:
            return _resize_deep_supervision_outputs(
                [x, x_ds_1, x_ds_2, x_ds_3, x_ds_4],
                x.shape[2:],
                self.deep_supervision_scales,
            )
        return x


def build_mednext_large_skip_conv_e1(
    num_input_channels: int,
    num_output_channels: int,
    enable_deep_supervision: bool = True,
    deep_supervision_scales=None,
) -> MedNeXtSkipConvE1:
    """Build the E1 Skip Conv control with the same MedNeXt-L backbone as E0."""
    net = MedNeXtSkipConvE1(
        in_channels=num_input_channels,
        n_classes=num_output_channels,
        deep_supervision=True,
        deep_supervision_scales=deep_supervision_scales,
        **_MEDNEXT_L_KWARGS,
    )
    net.do_ds = enable_deep_supervision
    return net


class DecoderConditionedSpatialGate(nn.Module):
    """Legacy E2 pointwise gate: one skip weight per voxel, without neighborhood context."""

    def __init__(self, channels: int):
        super().__init__()
        hidden_channels = max(channels // 4, 8)
        self.project = nn.Conv3d(channels * 2, hidden_channels, kernel_size=1, bias=True)
        self.to_gate = nn.Conv3d(hidden_channels, 1, kernel_size=1, bias=True)
        # Start from the original additive skip: 1 + tanh(0) = 1.
        nn.init.zeros_(self.to_gate.weight)
        nn.init.zeros_(self.to_gate.bias)

    def forward(self, encoder_skip: torch.Tensor, decoder_up: torch.Tensor) -> torch.Tensor:
        if encoder_skip.shape != decoder_up.shape:
            raise RuntimeError(
                "E2 spatial gate requires shape-matched encoder and decoder features; "
                f"got {tuple(encoder_skip.shape)} and {tuple(decoder_up.shape)}."
            )
        logits = self.to_gate(torch.nn.functional.gelu(
            self.project(torch.cat((encoder_skip, decoder_up), dim=1))
        ))
        return (1.0 + torch.tanh(logits)) * encoder_skip


class DecoderConditionedSpatialContextGate(nn.Module):
    """Filter a skip with decoder-conditioned local 3D context at each bridge."""

    def __init__(self, channels: int):
        super().__init__()
        hidden_channels = max(channels // 4, 8)
        # Fuse D and E at each voxel before collecting its 3x3x3 neighborhood.
        self.project = nn.Conv3d(channels * 2, hidden_channels, kernel_size=1, bias=True)
        self.context = nn.Conv3d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=True)
        self.to_gate = nn.Conv3d(hidden_channels, 1, kernel_size=1, bias=True)
        # Start from the original additive skip: 1 + tanh(0) = 1.
        nn.init.zeros_(self.to_gate.weight)
        nn.init.zeros_(self.to_gate.bias)

    def forward(self, encoder_skip: torch.Tensor, decoder_up: torch.Tensor) -> torch.Tensor:
        if encoder_skip.shape != decoder_up.shape:
            raise RuntimeError(
                "E2 spatial-context gate requires shape-matched encoder and decoder features; "
                f"got {tuple(encoder_skip.shape)} and {tuple(decoder_up.shape)}."
            )
        fused = torch.nn.functional.gelu(self.project(torch.cat((encoder_skip, decoder_up), dim=1)))
        logits = self.to_gate(torch.nn.functional.gelu(self.context(fused)))
        return (1.0 + torch.tanh(logits)) * encoder_skip


class MedNeXtSpatialGateE2(MedNeXtSkipConvE1):
    """Legacy E2: decoder-conditioned pointwise skip filtering at each bridge."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        n_channels = kwargs.get("n_channels", 32)
        # E2 must not retain E1's unused 1x1 convolutions as hidden parameters.
        del self.skip_convs
        self.skip_gates = nn.ModuleList(
            [
                DecoderConditionedSpatialGate(channels)
                for channels in (n_channels, n_channels * 2, n_channels * 4, n_channels * 8)
            ]
        )

    def _fuse_skip(
        self, stage: int, encoder_skip: torch.Tensor, decoder_up: torch.Tensor
    ) -> torch.Tensor:
        return decoder_up + self.skip_gates[stage](encoder_skip, decoder_up)


class MedNeXtSpatialContextGateE2(MedNeXtSkipConvE1):
    """E2: decoder-conditioned 3D-neighborhood skip filtering at each bridge."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        n_channels = kwargs.get("n_channels", 32)
        del self.skip_convs
        self.skip_gates = nn.ModuleList(
            [
                DecoderConditionedSpatialContextGate(channels)
                for channels in (n_channels, n_channels * 2, n_channels * 4, n_channels * 8)
            ]
        )

    def _fuse_skip(
        self, stage: int, encoder_skip: torch.Tensor, decoder_up: torch.Tensor
    ) -> torch.Tensor:
        return decoder_up + self.skip_gates[stage](encoder_skip, decoder_up)


def build_mednext_large_spatial_gate_e2(
    num_input_channels: int,
    num_output_channels: int,
    enable_deep_supervision: bool = True,
    deep_supervision_scales=None,
) -> MedNeXtSpatialGateE2:
    """Build E2 with the E0 backbone and decoder-conditioned spatial gates."""
    net = MedNeXtSpatialGateE2(
        in_channels=num_input_channels,
        n_classes=num_output_channels,
        deep_supervision=True,
        deep_supervision_scales=deep_supervision_scales,
        **_MEDNEXT_L_KWARGS,
    )
    net.do_ds = enable_deep_supervision
    return net


def build_mednext_large_spatial_context_gate_e2(
    num_input_channels: int,
    num_output_channels: int,
    enable_deep_supervision: bool = True,
    deep_supervision_scales=None,
) -> MedNeXtSpatialContextGateE2:
    """Build E2 with the E0 backbone and decoder-conditioned 3D-context gates."""
    net = MedNeXtSpatialContextGateE2(
        in_channels=num_input_channels,
        n_classes=num_output_channels,
        deep_supervision=True,
        deep_supervision_scales=deep_supervision_scales,
        **_MEDNEXT_L_KWARGS,
    )
    net.do_ds = enable_deep_supervision
    return net


class MedNeXtMLABot(MedNeXt):#Bot是bottleneck的缩写
    """
    MedNeXt-L + MLA Bottleneck：在 MedNeXt bottleneck 后插入 MLABottleneck3D。

    MedNeXt-L bottleneck 输出通道 = n_channels × 16 = 512。
    MLA 建立最低分辨率特征图内的全局依赖，补充 IB 卷积（k=3）缺失的全局上下文。

    forward 完全复制 MedNeXt，仅在 self.bottleneck(x) 后追加 self.mla_bot(x)。
    """

    def __init__(self, *args,
                 mla_num_heads: int = 8,
                 mla_num_blocks: int = 2,
                 mla_compression_ratio: int = 4,
                 mla_mlp_ratio: int = 4,
                 mla_use_moe: bool = True,
                 enable_mednext_grn: bool = False,
                 deep_supervision_scales=None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        if enable_mednext_grn:
            self._enable_grn_in_all_mednext_blocks()
        n_channels = kwargs.get('n_channels', 32)
        self.deep_supervision_scales = deep_supervision_scales
        self.mla_bot = MLABottleneck3D(
            d_model=n_channels * 16,
            num_heads=mla_num_heads,
            num_blocks=mla_num_blocks,
            compression_ratio=mla_compression_ratio,
            mlp_ratio=mla_mlp_ratio,
            use_moe=mla_use_moe,
        )

    def _enable_grn_in_all_mednext_blocks(self) -> None:
        """Enable the MedNeXt-provided 3D Global Response Normalization.

        GRN is different from the preceding GroupNorm: GroupNorm normalizes
        values within each channel, whereas GRN compares the total response
        of each channel with the mean response of all channels. For
        ``x1.shape == (B, C, D, H, W)`` the block computes::

            gx = ||x1||_2 over (D, H, W)       # (B, C, 1, 1, 1)
            nx = gx / mean(gx over C)
            y  = gamma * (x1 * nx) + beta + x1

        ``gamma`` and ``beta`` are learnable parameters initialized to zero,
        so GRN starts as an identity mapping. The installed MedNeXt already
        implements this operation but leaves it disabled by default. This
        method enables it in every standard, downsampling and upsampling
        block, including ``down_0`` whose constructor does not receive the
        global ``grn`` flag. It is used by
        ``nnUNetTrainer_MedNeXt_MLA_MoE_GRN`` and its
        ``nnUNetTrainer_MedNeXt_MLA_MoE_GRN_MSDHCCPretrain`` variant.
        """
        enabled_blocks = 0
        for module in self.modules():
            if not isinstance(module, MedNeXtBlock):
                continue
            if module.dim != '3d':
                raise RuntimeError("MedNeXt_MLA_MoE_GRN requires 3D MedNeXt blocks")
            expanded_channels = module.conv2.out_channels
            module.grn = True
            if not hasattr(module, 'grn_beta'):
                module.register_parameter(
                    'grn_beta',
                    nn.Parameter(torch.zeros(1, expanded_channels, 1, 1, 1)), #type:ignore
                )
            if not hasattr(module, 'grn_gamma'):
#hasattr是python的内置函数,判断module到底有没有属性grn_gamma,如果没有就注册一个新的参数grn_gamma,它是一个可学习的参数,初始化为全零张量,形状为(1, expanded_channels, 1, 1, 1),其中expanded_channels是该模块卷积层的输出通道数.
                module.register_parameter(
                    'grn_gamma',
                    nn.Parameter(torch.zeros(1, expanded_channels, 1, 1, 1)), #type:ignore
                )
#nn.Parameter是定义可以学习参数的函数,
#register_parameter是把这个parameter挂到神经网络模块名下

            enabled_blocks += 1
        if enabled_blocks == 0:
            raise RuntimeError("GRN was requested but no MedNeXt blocks were found")

    def _apply_bottleneck_context(self, x: torch.Tensor) -> torch.Tensor:
        return self.mla_bot(x)

    def forward(self, x):

#_cp是torch.utils.checkpoint的别名,用于梯度检查点技术,
#_cp.checkpoint()是一个函数,它可以在前向传播时保存中间激活,以减少显存占用,在反向传播时重新计算这些激活.


        x = self.stem(x) #正式通道开始前的输入通道映射层
        if self.outside_block_checkpointing:
            x_res_0 = self.iterative_checkpoint(self.enc_block_0, x)
            #enc_block_o是一串block,
            #down_0是一个block

            x = _cp.checkpoint(self.down_0, x_res_0, self.dummy_tensor)
            x_res_1 = self.iterative_checkpoint(self.enc_block_1, x)
            x = _cp.checkpoint(self.down_1, x_res_1, self.dummy_tensor)
            x_res_2 = self.iterative_checkpoint(self.enc_block_2, x)
            x = _cp.checkpoint(self.down_2, x_res_2, self.dummy_tensor)
            x_res_3 = self.iterative_checkpoint(self.enc_block_3, x)
            x = _cp.checkpoint(self.down_3, x_res_3, self.dummy_tensor)

            x = self.iterative_checkpoint(self.bottleneck, x)
            # MLA/MHA Transformer 也纳入 gradient checkpointing，避免保留整个
            # attention/FFN 中间激活。use_reentrant=False 支持正常的 autograd 输入。
            x = _cp.checkpoint(self._apply_bottleneck_context, x, use_reentrant=False)

            if self.do_ds:
                x_ds_4 = _cp.checkpoint(self.out_4, x, self.dummy_tensor)

            x_up_3 = _cp.checkpoint(self.up_3, x, self.dummy_tensor)
            x_up_3 = _match_spatial_shape(x_up_3, x_res_3) #type:ignore
            dec_x = x_res_3 + x_up_3
            x = self.iterative_checkpoint(self.dec_block_3, dec_x)
            if self.do_ds:
                x_ds_3 = _cp.checkpoint(self.out_3, x, self.dummy_tensor)
            del x_res_3, x_up_3

            x_up_2 = _cp.checkpoint(self.up_2, x, self.dummy_tensor)
            x_up_2 = _match_spatial_shape(x_up_2, x_res_2)#type:ignore
            dec_x = x_res_2 + x_up_2
            x = self.iterative_checkpoint(self.dec_block_2, dec_x)
            if self.do_ds:
                x_ds_2 = _cp.checkpoint(self.out_2, x, self.dummy_tensor)
            del x_res_2, x_up_2

            x_up_1 = _cp.checkpoint(self.up_1, x, self.dummy_tensor)
            x_up_1 = _match_spatial_shape(x_up_1, x_res_1)#type:ignore
            dec_x = x_res_1 + x_up_1
            x = self.iterative_checkpoint(self.dec_block_1, dec_x)
            if self.do_ds:
                x_ds_1 = _cp.checkpoint(self.out_1, x, self.dummy_tensor)
            del x_res_1, x_up_1

            x_up_0 = _cp.checkpoint(self.up_0, x, self.dummy_tensor)
            x_up_0 = _match_spatial_shape(x_up_0, x_res_0)#type:ignore
            dec_x = x_res_0 + x_up_0
            x = self.iterative_checkpoint(self.dec_block_0, dec_x)
            del x_res_0, x_up_0, dec_x

            x = _cp.checkpoint(self.out_0, x, self.dummy_tensor)

        else:
            x_res_0 = self.enc_block_0(x)
            x = self.down_0(x_res_0)
            x_res_1 = self.enc_block_1(x)
            x = self.down_1(x_res_1)
            x_res_2 = self.enc_block_2(x)
            x = self.down_2(x_res_2)
            x_res_3 = self.enc_block_3(x)
            x = self.down_3(x_res_3)

            x = self.bottleneck(x)
            x = self._apply_bottleneck_context(x)

            if self.do_ds:
                x_ds_4 = self.out_4(x)

            x_up_3 = self.up_3(x)
            x_up_3 = _match_spatial_shape(x_up_3, x_res_3)
            dec_x = x_res_3 + x_up_3
            x = self.dec_block_3(dec_x)
            if self.do_ds:
                x_ds_3 = self.out_3(x)
            del x_res_3, x_up_3

            x_up_2 = self.up_2(x)
            x_up_2 = _match_spatial_shape(x_up_2, x_res_2)
            dec_x = x_res_2 + x_up_2
            x = self.dec_block_2(dec_x)
            if self.do_ds:
                x_ds_2 = self.out_2(x)
            del x_res_2, x_up_2

            x_up_1 = self.up_1(x)
            x_up_1 = _match_spatial_shape(x_up_1, x_res_1)
            dec_x = x_res_1 + x_up_1
            x = self.dec_block_1(dec_x)
            if self.do_ds:
                x_ds_1 = self.out_1(x)
            del x_res_1, x_up_1

            x_up_0 = self.up_0(x)
            x_up_0 = _match_spatial_shape(x_up_0, x_res_0)
            dec_x = x_res_0 + x_up_0
            x = self.dec_block_0(dec_x)
            del x_res_0, x_up_0, dec_x

            x = self.out_0(x)

        if self.do_ds:
            return _resize_deep_supervision_outputs(
                [x, x_ds_1, x_ds_2, x_ds_3, x_ds_4],
                x.shape[2:],#type:ignore
                self.deep_supervision_scales,
            )
        else:
            return x


class PlainConvUpBlock3D(nn.Module):
    """Minimal U-Net-style learned 2x upsampling for the decoder."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
            bias=False,
        )

    def forward(self, x: torch.Tensor, dummy_tensor=None) -> torch.Tensor:
        return self.up(x)


class PlainConvDecoderStage3D(nn.Module):
    """One ordinary 3x3x3 convolution after additive skip fusion.

    GroupNorm with one group per channel matches the batch-size-independent
    normalization policy of the MedNeXt control. There is deliberately no
    expansion MLP, depthwise convolution stack or residual block here.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv3d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.norm = nn.GroupNorm(num_groups=channels, num_channels=channels)
        #norm这个会针对每个样本,每个通道单独计算均值和方差
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, dummy_tensor=None) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class MedNeXtMLAPlainConvDecoderBot(MedNeXtMLABot):
    """MedNeXt encoder + MLA/MoE bottleneck + shallow plain-conv decoder.

    The encoder, four downsampling blocks, eight-block MedNeXt bottleneck,
    MLA/MoE context module, additive skip connections and deep-supervision
    heads are identical to ``MedNeXtMLABot``. Only the right-hand decoder is
    replaced: each scale uses a 2x transposed convolution, adds the matching
    encoder skip, then applies one ordinary 3x3x3 Conv-GN-GELU stage.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        n_channels = int(kwargs.get('n_channels', 32))

        self.up_3 = PlainConvUpBlock3D(16 * n_channels, 8 * n_channels)
        self.dec_block_3 = nn.Sequential(PlainConvDecoderStage3D(8 * n_channels))
        self.up_2 = PlainConvUpBlock3D(8 * n_channels, 4 * n_channels)
        self.dec_block_2 = nn.Sequential(PlainConvDecoderStage3D(4 * n_channels))
        self.up_1 = PlainConvUpBlock3D(4 * n_channels, 2 * n_channels)
        self.dec_block_1 = nn.Sequential(PlainConvDecoderStage3D(2 * n_channels))
        self.up_0 = PlainConvUpBlock3D(2 * n_channels, n_channels)
        self.dec_block_0 = nn.Sequential(PlainConvDecoderStage3D(n_channels))


class MedNeXtMHABot(MedNeXtMLABot):
    """MedNeXt-L + 标准 Transformer bottleneck（MHA + MLP）。"""

    def __init__(self, *args,
                 mha_num_heads: int = 8,
                 mha_num_blocks: int = 2,
                 mha_mlp_ratio: int = 4,
                 mha_use_moe: bool = False,
                 **kwargs):
        # 先由父类完成完全相同的 MedNeXt 构建，再仅替换上下文模块。
        super().__init__(
            *args,
            mla_num_heads=mha_num_heads,
            mla_num_blocks=mha_num_blocks,
            mla_mlp_ratio=mha_mlp_ratio,
            mla_use_moe=mha_use_moe,
            **kwargs,
        )
        n_channels = kwargs.get('n_channels', 32)
        self.mla_bot = MHABottleneck3D(
            d_model=n_channels * 16,
            num_heads=mha_num_heads,
            num_blocks=mha_num_blocks,
            mlp_ratio=mha_mlp_ratio,
            use_moe=mha_use_moe,
        )


class MedNeXtMHADistanceBot(MedNeXtMHABot):
    """MedNeXt-L + MHA with a fixed 3D distance weight in the bottleneck."""

    def __init__(self, *args, mha_distance_alpha: float = 0.01, **kwargs):
        super().__init__(*args, **kwargs)
        n_channels = kwargs.get('n_channels', 32)
        self.mla_bot = DistanceBiasedMHABottleneck3D(
            d_model=n_channels * 16,
            num_heads=kwargs.get('mha_num_heads', 8),
            num_blocks=kwargs.get('mha_num_blocks', 2),
            mlp_ratio=kwargs.get('mha_mlp_ratio', 4),
            use_moe=kwargs.get('mha_use_moe', False),
            distance_alpha=mha_distance_alpha,
        )


class HCCBottleneckAdapter3D(nn.Module):
    """
    HCC-specific residual adapter for MedNeXt+MLA bottleneck features.

    The zero-initialized up projection makes the adapter an identity mapping at
    initialization: y = x + 0. This allows loading a Dataset003-trained base and
    learning only the HCC residual correction.
    """

    def __init__(self, channels: int, reduction: int = 16, zero_init: bool = True):
        super().__init__()
        hidden_channels = max(channels // reduction, 1)
        self.down = nn.Conv3d(channels, hidden_channels, kernel_size=1, bias=True)
        self.act = nn.GELU()
        self.up = nn.Conv3d(hidden_channels, channels, kernel_size=1, bias=True)
        if zero_init:
            nn.init.zeros_(self.up.weight)
            nn.init.zeros_(self.up.bias)#type:ignore

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(self.act(self.down(x)))


class MedNeXtMLAHCCAdapterBot(MedNeXtMLABot):
    """
    MedNeXt-L + MLA Bottleneck + HCC-specific residual adapter.

    Adapter position:
        MedNeXt bottleneck -> MLABottleneck3D -> hcc_adapter -> decoder

    The module name is deliberately `hcc_adapter` so it can be isolated for
    freezing, loading and checkpoint inspection.
    """

    def __init__(self, *args,
                 hcc_adapter_reduction: int = 16,
                 hcc_adapter_zero_init: bool = True,
                 **kwargs):
        super().__init__(*args, **kwargs)
        n_channels = kwargs.get('n_channels', 32)
        self.hcc_adapter = HCCBottleneckAdapter3D(
            channels=n_channels * 16,
            reduction=hcc_adapter_reduction,
            zero_init=hcc_adapter_zero_init,
        )

    def _apply_bottleneck_context(self, x: torch.Tensor) -> torch.Tensor:
        x = super()._apply_bottleneck_context(x)
        return self.hcc_adapter(x)

    def freeze_except_hcc_adapter(self) -> None:
        for param in self.parameters():
            param.requires_grad = False
        for param in self.hcc_adapter.parameters():
            param.requires_grad = True


def build_mednext_large_mla(
    num_input_channels: int,
    num_output_channels: int,
    enable_deep_supervision: bool = True,
    mla_num_heads: int = 8,
    mla_num_blocks: int = 2,
    mla_compression_ratio: int = 4,
    mla_mlp_ratio: int = 4,
    mla_use_moe: bool = True,
    enable_mednext_grn: bool = False,
    deep_supervision_scales=None,
) -> MedNeXtMLABot:
    net = MedNeXtMLABot(
        in_channels=num_input_channels,
        n_classes=num_output_channels,
        deep_supervision=True,
        mla_num_heads=mla_num_heads,
        mla_num_blocks=mla_num_blocks,
        mla_compression_ratio=mla_compression_ratio,
        mla_mlp_ratio=mla_mlp_ratio,
        mla_use_moe=mla_use_moe,
        enable_mednext_grn=enable_mednext_grn,
        deep_supervision_scales=deep_supervision_scales,
        **_MEDNEXT_L_KWARGS,
    )
    net.do_ds = enable_deep_supervision
    return net


def build_mednext_large_mla_plain_conv_decoder(
    num_input_channels: int,
    num_output_channels: int,
    enable_deep_supervision: bool = True,
    mla_num_heads: int = 8,
    mla_num_blocks: int = 2,
    mla_compression_ratio: int = 4,
    mla_mlp_ratio: int = 4,
    mla_use_moe: bool = True,
    deep_supervision_scales=None,
) -> MedNeXtMLAPlainConvDecoderBot:
    net = MedNeXtMLAPlainConvDecoderBot(
        in_channels=num_input_channels,
        n_classes=num_output_channels,
        deep_supervision=True,
        mla_num_heads=mla_num_heads,
        mla_num_blocks=mla_num_blocks,
        mla_compression_ratio=mla_compression_ratio,
        mla_mlp_ratio=mla_mlp_ratio,
        mla_use_moe=mla_use_moe,
        deep_supervision_scales=deep_supervision_scales,
        **_MEDNEXT_L_KWARGS,
    )
    net.do_ds = enable_deep_supervision
    return net


def build_mednext_large_mha(
    num_input_channels: int,
    num_output_channels: int,
    enable_deep_supervision: bool = True,
    mha_num_heads: int = 8,
    mha_num_blocks: int = 2,
    mha_mlp_ratio: int = 4,
    mha_use_moe: bool = False,
    deep_supervision_scales=None,
) -> MedNeXtMHABot:
    net = MedNeXtMHABot(
        in_channels=num_input_channels,
        n_classes=num_output_channels,
        deep_supervision=True,
        mha_num_heads=mha_num_heads,
        mha_num_blocks=mha_num_blocks,
        mha_mlp_ratio=mha_mlp_ratio,
        mha_use_moe=mha_use_moe,
        deep_supervision_scales=deep_supervision_scales,
        **_MEDNEXT_L_KWARGS,
    )
    net.do_ds = enable_deep_supervision
    return net


def build_mednext_large_mha_distance(
    num_input_channels: int,
    num_output_channels: int,
    enable_deep_supervision: bool = True,
    mha_num_heads: int = 8,
    mha_num_blocks: int = 2,
    mha_mlp_ratio: int = 4,
    mha_use_moe: bool = False,
    mha_distance_alpha: float = 0.01,
    deep_supervision_scales=None,
) -> MedNeXtMHADistanceBot:
    net = MedNeXtMHADistanceBot(
        in_channels=num_input_channels,
        n_classes=num_output_channels,
        deep_supervision=True,
        mha_num_heads=mha_num_heads,
        mha_num_blocks=mha_num_blocks,
        mha_mlp_ratio=mha_mlp_ratio,
        mha_use_moe=mha_use_moe,
        mha_distance_alpha=mha_distance_alpha,
        deep_supervision_scales=deep_supervision_scales,
        **_MEDNEXT_L_KWARGS,
    )
    net.do_ds = enable_deep_supervision
    return net


def build_mednext_large_mla_hcc_adapter(
    num_input_channels: int,
    num_output_channels: int,
    enable_deep_supervision: bool = True,
    mla_num_heads: int = 8,
    mla_num_blocks: int = 2,
    mla_compression_ratio: int = 4,
    mla_mlp_ratio: int = 4,
    mla_use_moe: bool = True,
    deep_supervision_scales=None,
    hcc_adapter_reduction: int = 16,
    hcc_adapter_zero_init: bool = True,
    freeze_base: bool = True,
) -> MedNeXtMLAHCCAdapterBot:
    net = MedNeXtMLAHCCAdapterBot(
        in_channels=num_input_channels,
        n_classes=num_output_channels,
        deep_supervision=True,
        mla_num_heads=mla_num_heads,
        mla_num_blocks=mla_num_blocks,
        mla_compression_ratio=mla_compression_ratio,
        mla_mlp_ratio=mla_mlp_ratio,
        mla_use_moe=mla_use_moe,
        deep_supervision_scales=deep_supervision_scales,
        hcc_adapter_reduction=hcc_adapter_reduction,
        hcc_adapter_zero_init=hcc_adapter_zero_init,
        **_MEDNEXT_L_KWARGS,
    )
    net.do_ds = enable_deep_supervision
    if freeze_base:
        net.freeze_except_hcc_adapter()
    return net
