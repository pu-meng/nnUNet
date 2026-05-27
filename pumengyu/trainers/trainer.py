import pydoc
import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from pumengyu.mixins import (
    CopyPasteMixin, DifficultyCopyPasteMixin,
    SmallTumorOversampleMixin,
    SizeStratifiedOversampleMixin,
    UnifiedFocalLossMixin, AutoReportMixin,
    NoTumorFPPenaltyMixin,
)
from pumengyu.architectures.umamba import UMambaBot3D


class nnUNetTrainer_UFL(UnifiedFocalLossMixin, AutoReportMixin, nnUNetTrainer):
    """
    仅叠加 AsymmetricUnifiedFocalLoss，不做 CopyPaste / 过采样。

    用途：单独评估 UFL 对 Dataset003_Liver 的贡献，
    与 nnUNetTrainer_CopyPaste（无UFL）形成 2×2 消融对照。

    结果目录：nnUNetTrainer_UFL__nnUNetPlans__3d_fullres/
    """


class nnUNetTrainer_UFL_v2(UnifiedFocalLossMixin, AutoReportMixin, nnUNetTrainer):
    """结果目录：nnUNetTrainer_UFL_v2__nnUNetPlans__3d_fullres/"""


class nnUNetTrainer_UFL_delta06(UnifiedFocalLossMixin, AutoReportMixin, nnUNetTrainer):
    """
    UFL_DELTA=0.6，偏向惩罚漏检（FN），对应原 nnUNetTrainer_UFL 训练时的实际超参。
    用于与 UFL_v2（delta=0.5）做显式对照消融。
    结果目录：nnUNetTrainer_UFL_delta06__nnUNetPlans__3d_fullres/
    """
    UFL_DELTA = 0.6


# ------------------------------------------------------------------ #
# 以下为已放弃方向，仅保留以维持历史结果目录可复现                    #
# ------------------------------------------------------------------ #

class nnUNetTrainer_CopyPaste(CopyPasteMixin, SmallTumorOversampleMixin, AutoReportMixin, nnUNetTrainer):
    """已放弃。结果目录：nnUNetTrainer_CopyPaste__nnUNetPlans__3d_fullres/"""


class nnUNetTrainer_CopyPaste_v2(CopyPasteMixin, SmallTumorOversampleMixin, AutoReportMixin, nnUNetTrainer):
    """已放弃。结果目录：nnUNetTrainer_CopyPaste_v2__nnUNetPlans__3d_fullres/"""


class nnUNetTrainer_CopyPaste_Diff(
    DifficultyCopyPasteMixin, SmallTumorOversampleMixin, AutoReportMixin, nnUNetTrainer
):
    """已放弃。结果目录：nnUNetTrainer_CopyPaste_Diff__nnUNetPlans__3d_fullres/"""


# ------------------------------------------------------------------ #
# 当前主线                                                            #
# ------------------------------------------------------------------ #

class nnUNetTrainer_SizeOversample(SizeStratifiedOversampleMixin, AutoReportMixin, nnUNetTrainer):
    """
    大小分层重复过采样（无 CopyPaste）。

    边界从当前 fold 训练集的肿瘤大小分布中自动计算（百分位数），无硬编码阈值：
        极小（≤ P20） → 3× 重复
        小（P20~P45） → 2× 重复
        中/大（中间段）→ 不额外重复
        极大（> P90） → 3× 重复
        无肿瘤 case   → 3× 重复（抑制误报）

    消融目标：单独评估"纯重复过采样（不粘贴）"对极小/极大肿瘤和无肿瘤误报的贡献。

    结果目录：nnUNetTrainer_SizeOversample__nnUNetPlans__3d_fullres/
    """


class nnUNetTrainer_SizeOversample_UFL(
    SizeStratifiedOversampleMixin, UnifiedFocalLossMixin, AutoReportMixin, nnUNetTrainer
):
    """
    大小分层重复过采样 + UFL（delta=0.6）。

    SizeOversample 负责数据层面的频率均衡，UFL 负责 loss 层面对漏检的惩罚。

    结果目录：nnUNetTrainer_SizeOversample_UFL__nnUNetPlans__3d_fullres/
    """


class nnUNetTrainer_SizeOversampleV2(SizeStratifiedOversampleMixin, AutoReportMixin, nnUNetTrainer):
    """
    大小分层重复过采样 V2（加强版倍数）。

    对比 V1（极小×3, 小×2, 无肿瘤×3），V2 大幅提升小肿瘤和无肿瘤曝光频率：
        极小（≤P20）  → 6× 重复
        小（P20~P45） → 5× 重复
        中/大（中间段）→ 1×（不变）
        极大（>P90）  → 3×（不变）
        无肿瘤 case   → 6× 重复

    消融目标：单独评估过采样倍数提升对小肿瘤召回和无肿瘤误报的影响（不加 loss 惩罚）。
    结果目录：nnUNetTrainer_SizeOversampleV2__nnUNetPlans__3d_fullres/
    """
    SSO_TINY_REPEAT:     int = 6
    SSO_SMALL_REPEAT:    int = 5
    SSO_NO_TUMOR_REPEAT: int = 6


class nnUNetTrainer_SizeOversampleV2_NTFP(
    NoTumorFPPenaltyMixin, SizeStratifiedOversampleMixin, AutoReportMixin, nnUNetTrainer
):
    """
    大小分层重复过采样 V2 + 无肿瘤专项误报惩罚（NoTumorFPPenalty）。

    数据层：极小×6, 小×5, 无肿瘤×6（增加困难 case 曝光频率）。
    Loss 层：GT 无肿瘤的 patch 中，额外惩罚预测出肿瘤的概率均值（lambda=1.0）。

    两路协同：oversample 让模型更多见无肿瘤 case，NTFP 直接在 loss 上惩罚误报；
    有肿瘤 sample 的 loss 路径不受影响，不牺牲小肿瘤召回。

    结果目录：nnUNetTrainer_SizeOversampleV2_NTFP__nnUNetPlans__3d_fullres/
    """
    SSO_TINY_REPEAT:     int = 6
    SSO_SMALL_REPEAT:    int = 5
    SSO_NO_TUMOR_REPEAT: int = 6


class nnUNetTrainer_SizeOversampleV2_Ext25(SizeStratifiedOversampleMixin, AutoReportMixin, nnUNetTrainer):
    """
    SizeOversampleV2 + 外部无肿瘤 25 case（IRCADb×5 + CHAOS CT×20）。

    对照实验：与 SizeOversampleV2 完全相同的过采样倍数，唯一变量是 splits 里多了
    25 个外部无肿瘤 case，用于评估数据层扩充对 FP 率的独立贡献。

    结果目录：nnUNetTrainer_SizeOversampleV2_Ext25__nnUNetPlans__3d_fullres/
    """
    SSO_TINY_REPEAT:     int = 6
    SSO_SMALL_REPEAT:    int = 5
    SSO_NO_TUMOR_REPEAT: int = 6


class nnUNetTrainer_SizeOversampleV2_NTFP_Ext25(
    NoTumorFPPenaltyMixin, SizeStratifiedOversampleMixin, AutoReportMixin, nnUNetTrainer
):
    """
    SizeOversampleV2_NTFP + 外部无肿瘤 25 case（IRCADb×5 + CHAOS CT×20）。

    主实验：过采样 V2 倍数 + NTFP loss + 外部数据三路协同，评估完整方案效果。

    结果目录：nnUNetTrainer_SizeOversampleV2_NTFP_Ext25__nnUNetPlans__3d_fullres/
    """
    SSO_TINY_REPEAT:     int = 6
    SSO_SMALL_REPEAT:    int = 5
    SSO_NO_TUMOR_REPEAT: int = 6


# ------------------------------------------------------------------ #
# U-Mamba 系列                                                        #
# ------------------------------------------------------------------ #

def _build_umamba_bot(
    plans_manager: PlansManager,
    configuration_manager: ConfigurationManager,
    num_input_channels: int,
    num_output_channels: int,
    enable_deep_supervision: bool,
    **mamba_kwargs,
) -> UMambaBot3D:
    """
    从 nnUNet plans 读取网络结构参数，构建 UMambaBot3D。
    与 get_network_from_plans 完全等价，仅把网络类换成 UMambaBot3D。
    """
    arch_kwargs = dict(**configuration_manager.network_arch_init_kwargs)
    for key in configuration_manager.network_arch_init_kwargs_req_import:
        if arch_kwargs.get(key) is not None:
            arch_kwargs[key] = pydoc.locate(arch_kwargs[key])
    return UMambaBot3D(
        input_channels=num_input_channels,
        num_classes=num_output_channels,
        deep_supervision=enable_deep_supervision,
        **arch_kwargs,
        **mamba_kwargs,
    )


class nnUNetTrainer_UMamba(AutoReportMixin, nnUNetTrainer):
    """
    U-Mamba Bot：在 nnUNet 标准 PlainConvUNet 的瓶颈层插入 Mamba 块。

    其余（encoder、decoder、loss、data augmentation）与 nnUNetTrainer 完全一致，
    唯一变量 = 瓶颈处的 Mamba 全局上下文建模。

    消融目标：评估 Mamba 瓶颈对小肿瘤长程依赖建模的贡献（对照 Baseline 和 SizeOversample）。

    依赖（需先安装）：
        pip install mamba-ssm causal-conv1d

    结果目录：nnUNetTrainer_UMamba__nnUNetPlans__3d_fullres/
    """

    @staticmethod
    def build_network_architecture(
        plans_manager: PlansManager,
        configuration_manager: ConfigurationManager,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ):
        return _build_umamba_bot(
            plans_manager, configuration_manager,
            num_input_channels, num_output_channels, enable_deep_supervision,
        )


class nnUNetTrainer_UMamba_SizeOversample(SizeStratifiedOversampleMixin, AutoReportMixin, nnUNetTrainer):
    """
    U-Mamba Bot + 大小分层重复过采样。

    组合：Mamba 瓶颈（全局上下文）+ 数据层过采样（极小/极大/无肿瘤频率均衡）。

    结果目录：nnUNetTrainer_UMamba_SizeOversample__nnUNetPlans__3d_fullres/
    """

    @staticmethod
    def build_network_architecture(
        plans_manager: PlansManager,
        configuration_manager: ConfigurationManager,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ):
        return _build_umamba_bot(
            plans_manager, configuration_manager,
            num_input_channels, num_output_channels, enable_deep_supervision,
        )
