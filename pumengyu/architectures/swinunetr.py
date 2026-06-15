"""
SwinUNETR wrapper：使用 MONAI 官方实现，适配 nnUNetv2 的 build_network_architecture 接口。

固定配置（SwinUNETR-B，与论文标准一致）：
    feature_size=48, depths=(2,2,2,2), num_heads=(3,6,12,24), window_size=7

注意：MONAI SwinUNETR 不含 deep supervision，对应 trainer 需关闭 DS。
"""

from monai.networks.nets import SwinUNETR as _SwinUNETR


def build_swinunetr(
    num_input_channels: int,
    num_output_channels: int,
    patch_size: tuple = (128, 128, 128),  # 保留签名兼容性，此版本 MONAI 不需要 img_size
) -> _SwinUNETR:
    return _SwinUNETR(
        in_channels=num_input_channels,
        out_channels=num_output_channels,
        patch_size=2,          # token 化步长，MONAI 1.x 新接口（原 img_size 已废弃）
        feature_size=48,       # SwinUNETR-B 标准配置
        depths=(2, 2, 2, 2),
        num_heads=(3, 6, 12, 24),
        window_size=7,
        norm_name='instance',
        spatial_dims=3,
        use_checkpoint=True,   # 梯度检查点，减少显存~40%（compile已关闭，无冲突）
    )
