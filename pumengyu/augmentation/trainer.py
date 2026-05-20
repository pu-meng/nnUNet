from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from pumengyu.mixins import (
    CopyPasteMixin, SmallTumorOversampleMixin,
    UnifiedFocalLossMixin, AutoReportMixin,
)


class nnUNetTrainer_UFL(UnifiedFocalLossMixin, AutoReportMixin, nnUNetTrainer):
    """
    仅叠加 AsymmetricUnifiedFocalLoss，不做 CopyPaste / 过采样。

    用途：单独评估 UFL 对 Dataset003_Liver 的贡献，
    与 nnUNetTrainer_CopyPaste（无UFL）形成 2×2 消融对照。

    结果目录：nnUNetTrainer_UFL__nnUNetPlans__3d_fullres/
    """


class nnUNetTrainer_CopyPaste(CopyPasteMixin, SmallTumorOversampleMixin, AutoReportMixin, nnUNetTrainer):
    """
    Copy-Paste 小肿瘤增强 + 小肿瘤过采样。

    相比 baseline (nnUNetTrainer)：
      - SmallTumorOversampleMixin：在 identifiers 层面将小肿瘤 case 重复 3 次，
        提高 batch 中小肿瘤 case 的出现概率。
      - CopyPasteMixin：train_step 内随机将小肿瘤 ROI 粘贴到其他 case 的肝脏区域，
        直接增加极小肿瘤体素的训练曝光。

    结果目录：nnUNetTrainer_CopyPaste__nnUNetPlans__3d_fullres/
    """


class nnUNetTrainer_CopyPasteUFL(
    CopyPasteMixin, UnifiedFocalLossMixin,
    SmallTumorOversampleMixin, AutoReportMixin, nnUNetTrainer
):
    """
    Copy-Paste + AsymmetricUnifiedFocalLoss + 小肿瘤过采样。

    在 nnUNetTrainer_CopyPaste 基础上叠加 UnifiedFocalLossMixin：
      - UnifiedFocalLossMixin：default CE+Dice + λ×AUFL，
        AUFL 自动平衡极小肿瘤体素与背景体素的梯度贡献，无需手调 class weight。

    MRO：CopyPaste → UFL(_build_loss) → Oversample → AutoReport → nnUNetTrainer

    结果目录：nnUNetTrainer_CopyPasteUFL__nnUNetPlans__3d_fullres/
    """
