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


# ------------------------------------------------------------------ #
# v2：修复 CopyPaste 多连通域 bug 后的对照实验版本                    #
# Mixin 代码与上方一致，类名不同 → 结果存新目录，旧 fold_4 保留对比  #
# ------------------------------------------------------------------ #

class nnUNetTrainer_UFL_v2(UnifiedFocalLossMixin, AutoReportMixin, nnUNetTrainer):
    """结果目录：nnUNetTrainer_UFL_v2__nnUNetPlans__3d_fullres/"""


class nnUNetTrainer_CopyPaste_v2(CopyPasteMixin, SmallTumorOversampleMixin, AutoReportMixin, nnUNetTrainer):
    """结果目录：nnUNetTrainer_CopyPaste_v2__nnUNetPlans__3d_fullres/"""


class nnUNetTrainer_OfflineCopyPaste_v2(SmallTumorOversampleMixin, AutoReportMixin, nnUNetTrainer):
    """
    配合 pumengyu/tools/offline_copypaste.py 使用。
    增强数据已离线生成，读 splits_final_cp.json（原 splits_final.json 不受影响）。
    仍保留 SmallTumorOversampleMixin 对原始小肿瘤 case 做过采样。

    结果目录：nnUNetTrainer_OfflineCopyPaste_v2__nnUNetPlans__3d_fullres/
    """

    def do_split(self):
        from batchgenerators.utilities.file_and_folder_operations import load_json, join
        splits_file = join(self.preprocessed_dataset_folder_base, 'splits_final_cp.json')
        splits = load_json(splits_file)
        self.print_to_log_file(f"[OfflineCopyPaste] 使用 splits_final_cp.json")
        self.print_to_log_file(f"The split file contains {len(splits)} splits.")
        self.print_to_log_file(f"Desired fold for training: {self.fold}")
        tr_keys = splits[self.fold]['train']
        val_keys = splits[self.fold]['val']
        self.print_to_log_file(f"This split has {len(tr_keys)} training and {len(val_keys)} validation cases.")
        return tr_keys, val_keys
