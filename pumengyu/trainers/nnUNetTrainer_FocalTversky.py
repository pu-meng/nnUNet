"""
nnUNetTrainer_DCFocalTversky
不修改 nnUNet 源码，继承 nnUNetTrainer，只替换 loss 为：
  DiceCE (weight=0.7) + FocalTversky (weight=0.3)

参数来自 MSD_LiverTumorSeg 实验最优配置：
  alpha=0.3, beta=0.7, gamma=0.75

用法：
  nnUNetv2_train DATASET_ID 3d_fullres FOLD \
    -tr nnUNetTrainer_DCFocalTversky
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn

from nnunetv2.paths import nnUNet_raw
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from batchgenerators.utilities.file_and_folder_operations import join


# ─────────────────────────── FocalTversky Loss ───────────────────────────

class FocalTverskyLoss(nn.Module):
    """
    L = (1 - TI)^gamma
    TI = TP / (TP + alpha·FP + beta·FN)
    beta > alpha → 加重 FN 惩罚 → 提升 recall，适合小肿瘤。
    用 softmax + one-hot 避免 sigmoid 在 AMP fp16 下 NaN。
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.7,
                 gamma: float = 0.75, eps: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma
        self.eps   = eps

    def forward(self, net_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # net_output: [B, C, *spatial]   (logits, NOT softmax)
        # target:     [B, 1, *spatial]   (integer labels)
        net_output = net_output.float()             # 防 AMP fp16 溢出
        probs = torch.softmax(net_output, dim=1)    # [B, C, *]

        # one-hot: [B, C, *]
        target_onehot = torch.zeros_like(probs)
        target_onehot.scatter_(1, target.long(), 1)

        # 跳过背景类（class 0），只算前景
        p = probs[:, 1:]            # [B, C-1, *]
        t = target_onehot[:, 1:]   # [B, C-1, *]

        dims = tuple(range(2, p.ndim))
        tp = (p * t).sum(dim=dims)
        fp = (p * (1 - t)).sum(dim=dims)
        fn = ((1 - p) * t).sum(dim=dims)

        tversky = tp / (tp + self.alpha * fp + self.beta * fn + self.eps)
        loss = ((1 - tversky) ** self.gamma).mean()
        return loss


# ──────────────────── DC_and_FocalTversky compound loss ──────────────────

class DC_and_FocalTversky_loss(nn.Module):
    """
    DiceCE (weight_dce) + FocalTversky (weight_ft)
    接口与 DC_and_CE_loss 相同，可直接被 DeepSupervisionWrapper 包裹。
    """

    def __init__(self,
                 soft_dice_kwargs: dict,
                 ce_kwargs: dict,
                 ft_alpha: float = 0.3,
                 ft_beta:  float = 0.7,
                 ft_gamma: float = 0.75,
                 weight_dce: float = 0.7,
                 weight_ft:  float = 0.3,
                 ignore_label=None):
        super().__init__()
        self.weight_dce = weight_dce
        self.weight_ft  = weight_ft
        self.ignore_label = ignore_label

        self.dce = DC_and_CE_loss(
            soft_dice_kwargs, ce_kwargs,
            weight_ce=1, weight_dice=1,
            ignore_label=ignore_label,
            dice_class=MemoryEfficientSoftDiceLoss,
        )
        self.ft = FocalTverskyLoss(alpha=ft_alpha, beta=ft_beta, gamma=ft_gamma)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dce_loss = self.dce(net_output, target)

        # ignore_label 区域不计入 FocalTversky
        if self.ignore_label is not None:
            mask = (target != self.ignore_label)
            net_out_ft = net_output * mask
            target_ft  = torch.where(mask, target, torch.zeros_like(target))
        else:
            net_out_ft = net_output
            target_ft  = target

        ft_loss = self.ft(net_out_ft, target_ft)
        return self.weight_dce * dce_loss + self.weight_ft * ft_loss


# ────────────────────────────── Trainer ──────────────────────────────────

class nnUNetTrainer_DCFocalTversky(nnUNetTrainer):
    """
    用 DiceCE(0.7) + FocalTversky(0.3) 替换默认 DiceCE。
    其余训练配置（lr, aug, patch size 等）全部继承父类。
    """

    def _build_loss(self):
        if self.label_manager.has_regions:
            # region-based label 暂不支持，回退到父类
            return super()._build_loss()

        loss = DC_and_FocalTversky_loss(
            soft_dice_kwargs={
                'batch_dice': self.configuration_manager.batch_dice,
                'smooth': 1e-5,
                'do_bg': False,
                'ddp': self.is_ddp,
            },
            ce_kwargs={},
            ft_alpha=0.3,
            ft_beta=0.7,
            ft_gamma=0.75,
            weight_dce=0.7,
            weight_ft=0.3,
            ignore_label=self.label_manager.ignore_label,
        )

        # deep supervision：高分辨率输出权重指数衰减
        if self.enable_deep_supervision:
            ds_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(ds_scales))])
            if self.is_ddp and not self._do_i_compile():
                weights[-1] = 1e-6
            else:
                weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)

        return loss

    def perform_actual_validation(self, save_probabilities: bool = False):
        super().perform_actual_validation(save_probabilities)
        if self.local_rank == 0:
            from pumengyu.tools.analyasis.auto_report import run_auto_report
            run_auto_report(
                fold_dir=self.output_folder,
                gt_dir=join(self.preprocessed_dataset_folder_base, "gt_segmentations"),
                img_dir=join(str(nnUNet_raw), self.plans_manager.dataset_name, "imagesTr"),
            )
