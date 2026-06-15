"""
MedNeXt-L wrapper：从 nnunet_mednext 提取架构，适配 nnUNetv2 的 build_network_architecture 接口。

架构固定为 MedNeXt-L（Large）：
    n_channels=32, kernel_size=3, exp_r=[3,4,8,8,8,8,8,4,3],
    block_counts=[3,4,8,8,8,8,8,4,3], do_res=True, do_res_up_down=True

Deep supervision 输出顺序（do_ds=True）：
    [full_res, 1/2, 1/4, 1/8, 1/16] — 与 nnUNetv2 期望格式一致
"""

from nnunet_mednext.network_architecture.mednextv1.MedNextV1 import MedNeXt


def build_mednext_large(
    num_input_channels: int,
    num_output_channels: int,
    enable_deep_supervision: bool = True,
) -> MedNeXt:
    return MedNeXt(
        in_channels=num_input_channels,
        n_channels=32,
        n_classes=num_output_channels,
        exp_r=[3, 4, 8, 8, 8, 8, 8, 4, 3],
        kernel_size=3,
        deep_supervision=enable_deep_supervision,
        do_res=True,
        do_res_up_down=True,
        block_counts=[3, 4, 8, 8, 8, 8, 8, 4, 3],
        checkpoint_style='outside_block',
        norm_type='group',
        dim='3d',
    )
