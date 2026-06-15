"""
nnFormer wrapper：使用 mednextv1 包中已附带的 nnFormer_tumor 架构，
适配 nnUNetv2 的 build_network_architecture 接口。

固定配置（与 nnFormer 原论文 128³ 设置一致）：
    embedding_dim=96, depths=[2,2,2,2], num_heads=[3,6,12,24]
    patch_size=[4,4,4], window_size=[4,4,8,4]

注意：该版本 DS 被注释掉，始终返回单 tensor，对应 trainer 需关闭 DS。
"""

import torch.nn as nn
import torch.utils.checkpoint as ck
from nnunet_mednext.network_architecture.custom_modules.custom_networks.nnFormer.nnFormer_tumor import nnFormer as _nnFormer
from nnunet_mednext.network_architecture.custom_modules.custom_networks.nnFormer.nnFormer_tumor import (
    SwinTransformerBlock,
    SwinTransformerBlock_kv,
)


def _apply_gradient_checkpointing(model: _nnFormer) -> _nnFormer:
    """对所有 Swin block 应用梯度检查点，节省约 40% 激活显存，代价是多跑一次前向。"""
    for module in model.modules():
        if isinstance(module, SwinTransformerBlock):
            def _make_enc(m):
                orig = m.forward
                def _fwd(x, mask_matrix):
                    # mask_matrix 捕获在闭包里（可以是 None 或 tensor）
                    def fn(x):
                        return orig(x, mask_matrix)
                    return ck.checkpoint(fn, x, use_reentrant=False)
                return _fwd
            module.forward = _make_enc(module)
        elif isinstance(module, SwinTransformerBlock_kv):
            def _make_dec(m):
                orig = m.forward
                def _fwd(x, mask_matrix, skip=None, x_up=None):
                    def fn(x, skip, x_up):
                        return orig(x, mask_matrix, skip=skip, x_up=x_up)
                    return ck.checkpoint(fn, x, skip, x_up, use_reentrant=False)
                return _fwd
            module.forward = _make_dec(module)
    return model


def build_nnformer(
    num_input_channels: int,
    num_output_channels: int,
    patch_size: tuple = (128, 128, 128),
    use_gradient_checkpointing: bool = False,
) -> _nnFormer:
    model = _nnFormer(
        crop_size=list(patch_size),
        embedding_dim=96,
        input_channels=num_input_channels,
        num_classes=num_output_channels,
        conv_op=nn.Conv3d,
        depths=[2, 2, 2, 2],
        num_heads=[3, 6, 12, 24],
        patch_size=[4, 4, 4],
        window_size=[4, 4, 8, 4],
        deep_supervision=False,
        dummy=False,
    )
    if use_gradient_checkpointing:
        _apply_gradient_checkpointing(model)
    return model
