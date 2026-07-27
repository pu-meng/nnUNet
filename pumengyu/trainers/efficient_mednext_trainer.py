"""Official EfficientMedNeXt-L baseline on the local nnU-Net v2 pipeline."""

from torch._dynamo import OptimizedModule

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import (
    ConfigurationManager,
    PlansManager,
)
from pumengyu.architectures.efficient_mednext_official import (
    build_efficient_mednext_large_official,
)
from pumengyu.mixins import AutoInternalTestMixin, AutoReportMixin


class nnUNetTrainer_EfficientMedNeXt_L_Official(
    AutoInternalTestMixin, AutoReportMixin, nnUNetTrainer
):
    """Fair architecture baseline using the pinned official EfficientMedNeXt-L.

    Network:
      - official SLDGroup/EfficientMedNeXt source at commit 803f7efe...
      - base channels 32 and uniform decoder channels 32
      - DMRFB receptive-field branches [1, 3, 5]
      - block counts [3, 4, 4, 4, 4, 4, 4, 4, 3]

    Everything outside the network (split, preprocessing, augmentation, loss,
    optimizer, schedule, inference, report, and visualization) is inherited from
    the same local nnU-Net v2 pipeline used by the other architecture baselines.
    """

    @classmethod
    def build_network_architecture(
        cls,
        plans_manager: PlansManager,
        configuration_manager: ConfigurationManager,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ):
        return build_efficient_mednext_large_official(
            num_input_channels=num_input_channels,
            num_output_channels=num_output_channels,
            enable_deep_supervision=enable_deep_supervision,
        )

    def set_deep_supervision_enabled(self, enabled: bool):
        mod = self.network.module if self.is_ddp else self.network  # type: ignore
        if isinstance(mod, OptimizedModule):
            mod = mod._orig_mod
        mod.do_ds = enabled  # type: ignore

    def _do_i_compile(self):
        # The pinned official network uses explicit gradient checkpoint calls.
        return False

