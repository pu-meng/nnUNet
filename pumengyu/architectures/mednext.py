"""
MedNeXt-L wrapper：从 nnunet_mednext 提取架构，适配 nnUNetv2 的 build_network_architecture 接口。

架构固定为 MedNeXt-L（Large）：
    n_channels=32, kernel_size=3, exp_r=[3,4,8,8,8,8,8,4,3],
    block_counts=[3,4,8,8,8,8,8,4,3], do_res=True, do_res_up_down=True

Deep supervision 输出顺序（do_ds=True）：
    [full_res, 1/2, 1/4, 1/8, 1/16] — 与 nnUNetv2 期望格式一致
"""

from torch.utils import checkpoint as _cp
from nnunet_mednext.network_architecture.mednextv1.MedNextV1 import MedNeXt
from pumengyu.architectures.mla_unetr import MLABottleneck3D

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


def build_mednext_large(
    num_input_channels: int,
    num_output_channels: int,
    enable_deep_supervision: bool = True,
) -> MedNeXt:
    # 始终用 deep_supervision=True 建图，确保 out_1/2/3/4 层存在，
    # 使 checkpoint 无论何时都能完整加载（推理时 do_ds=False 不影响输出）。
    net = MedNeXt(
        in_channels=num_input_channels,
        n_classes=num_output_channels,
        deep_supervision=True,
        **_MEDNEXT_L_KWARGS,
    )
    net.do_ds = enable_deep_supervision
    return net


class MedNeXtMLABot(MedNeXt):
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
                 **kwargs):
        super().__init__(*args, **kwargs)
        n_channels = kwargs.get('n_channels', 32)
        self.mla_bot = MLABottleneck3D(
            d_model=n_channels * 16,
            num_heads=mla_num_heads,
            num_blocks=mla_num_blocks,
            compression_ratio=mla_compression_ratio,
            mlp_ratio=mla_mlp_ratio,
        )

    def forward(self, x):
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
            x = self.mla_bot(x)

            if self.do_ds:
                x_ds_4 = _cp.checkpoint(self.out_4, x, self.dummy_tensor)

            x_up_3 = _cp.checkpoint(self.up_3, x, self.dummy_tensor)
            dec_x = x_res_3 + x_up_3
            x = self.iterative_checkpoint(self.dec_block_3, dec_x)
            if self.do_ds:
                x_ds_3 = _cp.checkpoint(self.out_3, x, self.dummy_tensor)
            del x_res_3, x_up_3

            x_up_2 = _cp.checkpoint(self.up_2, x, self.dummy_tensor)
            dec_x = x_res_2 + x_up_2
            x = self.iterative_checkpoint(self.dec_block_2, dec_x)
            if self.do_ds:
                x_ds_2 = _cp.checkpoint(self.out_2, x, self.dummy_tensor)
            del x_res_2, x_up_2

            x_up_1 = _cp.checkpoint(self.up_1, x, self.dummy_tensor)
            dec_x = x_res_1 + x_up_1
            x = self.iterative_checkpoint(self.dec_block_1, dec_x)
            if self.do_ds:
                x_ds_1 = _cp.checkpoint(self.out_1, x, self.dummy_tensor)
            del x_res_1, x_up_1

            x_up_0 = _cp.checkpoint(self.up_0, x, self.dummy_tensor)
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
            x = self.mla_bot(x)

            if self.do_ds:
                x_ds_4 = self.out_4(x)

            x_up_3 = self.up_3(x)
            dec_x = x_res_3 + x_up_3
            x = self.dec_block_3(dec_x)
            if self.do_ds:
                x_ds_3 = self.out_3(x)
            del x_res_3, x_up_3

            x_up_2 = self.up_2(x)
            dec_x = x_res_2 + x_up_2
            x = self.dec_block_2(dec_x)
            if self.do_ds:
                x_ds_2 = self.out_2(x)
            del x_res_2, x_up_2

            x_up_1 = self.up_1(x)
            dec_x = x_res_1 + x_up_1
            x = self.dec_block_1(dec_x)
            if self.do_ds:
                x_ds_1 = self.out_1(x)
            del x_res_1, x_up_1

            x_up_0 = self.up_0(x)
            dec_x = x_res_0 + x_up_0
            x = self.dec_block_0(dec_x)
            del x_res_0, x_up_0, dec_x

            x = self.out_0(x)

        if self.do_ds:
            return [x, x_ds_1, x_ds_2, x_ds_3, x_ds_4]
        else:
            return x


def build_mednext_large_mla(
    num_input_channels: int,
    num_output_channels: int,
    enable_deep_supervision: bool = True,
    mla_num_heads: int = 8,
    mla_num_blocks: int = 2,
    mla_compression_ratio: int = 4,
    mla_mlp_ratio: int = 4,
) -> MedNeXtMLABot:
    net = MedNeXtMLABot(
        in_channels=num_input_channels,
        n_classes=num_output_channels,
        deep_supervision=True,
        mla_num_heads=mla_num_heads,
        mla_num_blocks=mla_num_blocks,
        mla_compression_ratio=mla_compression_ratio,
        mla_mlp_ratio=mla_mlp_ratio,
        **_MEDNEXT_L_KWARGS,
    )
    net.do_ds = enable_deep_supervision
    return net
