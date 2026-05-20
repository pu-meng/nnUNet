"""
nnUNetTrainer_ConvBoundaryLoss

用固定 Laplacian 卷积核提取预测概率图和 GT one-hot 的边界响应，
计算两者差异作为 boundary loss，叠加在标准 CE+Dice 之上。

对比旧方案 (nnUNetTrainer_BoundaryLoss / dist_field.py)：
  - 无需 scipy EDT，无 CPU 计算瓶颈
  - 无需修改 DataLoader
  - 无需额外网络预测头
  - 全在 GPU 上完成，梯度直接从 softmax 流过
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autocast

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.helpers import dummy_context
from pumengyu.mixins import SmallTumorOversampleMixin, AutoReportMixin


# ─────────────────────────────────────────────────────────────────────────────
# Loss module
# ─────────────────────────────────────────────────────────────────────────────

def _make_laplacian_3d(spacing: tuple | None = None) -> torch.Tensor:
    """
    3D 各向异性 Laplacian 核，shape (1, 1, 3, 3, 3)。

    离散 Laplacian 各方向权重 = 1 / spacing_i²，中心 = -2 * Σ(1/spacing_i²)。
    spacing=None 时退化为各向同性（等权重 1/1²）。

    为何要加权：CT 轴向层间距常为 3-5 mm，面内分辨率约 0.77 mm，
    各向同性核会把 z 方向的"一步"与 xy 方向等价对待，严重低估 z 向边界响应。
    """
    if spacing is None:
        spacing = (1.0, 1.0, 1.0)
    sz, sy, sx = float(spacing[0]), float(spacing[1]), float(spacing[2])
    wz, wy, wx = 1.0 / sz**2, 1.0 / sy**2, 1.0 / sx**2
    k = torch.zeros(1, 1, 3, 3, 3)
    k[0, 0, 1, 1, 1] = -2.0 * (wz + wy + wx)
    k[0, 0, 0, 1, 1] = wz;  k[0, 0, 2, 1, 1] = wz
    k[0, 0, 1, 0, 1] = wy;  k[0, 0, 1, 2, 1] = wy
    k[0, 0, 1, 1, 0] = wx;  k[0, 0, 1, 1, 2] = wx
    return k


def _make_laplacian_2d(spacing: tuple | None = None) -> torch.Tensor:
    """
    2D 各向异性 Laplacian 核，shape (1, 1, 3, 3)。
    spacing=None 时各向同性。
    """
    if spacing is None:
        spacing = (1.0, 1.0)
    sy, sx = float(spacing[0]), float(spacing[1])
    wy, wx = 1.0 / sy**2, 1.0 / sx**2
    k = torch.zeros(1, 1, 3, 3)
    k[0, 0, 1, 1] = -2.0 * (wy + wx)
    k[0, 0, 0, 1] = wy;  k[0, 0, 2, 1] = wy
    k[0, 0, 1, 0] = wx;  k[0, 0, 1, 2] = wx
    return k


class ConvBoundaryLoss(nn.Module):
    """
    对每个前景类别分别做：
      1. pred_softmax[:, c] 卷 Laplacian → pred_boundary
      2. gt_onehot[:, c]   卷 Laplacian → gt_boundary
      3. MSE(pred_boundary, gt_boundary)

    Args:
        num_classes: 包含背景的总类别数 K（背景 channel 跳过）
        is_2d:       True 时使用 2D 核，False 时使用 3D 核
        spacing:     物理 voxel 尺寸（mm），如 (3.0, 0.77, 0.77)；
                     None → 各向同性。用于构建各向异性权重核。
    """

    def __init__(self, num_classes: int, is_2d: bool = False,
                 spacing: tuple | None = None):
        super().__init__()
        self.num_classes = num_classes
        self.is_2d = is_2d
        kernel = (_make_laplacian_2d(spacing) if is_2d
                  else _make_laplacian_3d(spacing))
        self.register_buffer('kernel', kernel)  # 随 .to(device) 自动迁移

    def _laplacian(self, x: torch.Tensor) -> torch.Tensor:
        w = self.kernel.to(dtype=x.dtype)          # 匹配输入精度
        if self.is_2d:
            return F.conv2d(x, w, padding=1)
        return F.conv3d(x, w, padding=1)

    def forward(self, pred_logit: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """
        pred_logit : (B, C, [D,] H, W)  网络输出 logit（未经 softmax）
        gt         : (B, 1, [D,] H, W)  整数类别标签，ignore label = -1
        """
        pred_prob = F.softmax(pred_logit.float(), dim=1)   # float32，梯度可传

        # ignore label (-1) clamp 到 0（背景），不影响前景 channel 计算
        gt_long = gt.squeeze(1).long().clamp(min=0)        # (B, [D,] H, W)

        # one-hot: (B, [D,] H, W, C) → (B, C, [D,] H, W)
        gt_onehot = F.one_hot(gt_long, self.num_classes).float()
        n = gt_onehot.dim()                                 # 4（2D）或 5（3D）
        gt_onehot = gt_onehot.permute(0, n - 1, *range(1, n - 1))

        loss = torch.tensor(0.0, device=pred_logit.device)
        for c in range(1, self.num_classes):               # 跳过背景 c=0
            pred_bd = self._laplacian(pred_prob[:, c:c+1])
            gt_bd   = self._laplacian(gt_onehot[:, c:c+1])
            loss    = loss + F.mse_loss(pred_bd, gt_bd)

        return loss / max(self.num_classes - 1, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class nnUNetTrainer_ConvBoundaryLoss(SmallTumorOversampleMixin, AutoReportMixin, nnUNetTrainer):
    """
    标准 nnUNet + Laplacian 卷积边界 loss。

    Loss:
        L = L_seg + λ · L_boundary
        L_seg      = CE + Dice  (nnUNetTrainer 默认)
        L_boundary = mean over fg classes of MSE(∇²pred_c, ∇²gt_c)

    λ 在前 WARMUP_EPOCHS 个 epoch 线性从 0 升到 LAMBDA_BD_MAX，之后固定。
    """

    LAMBDA_BD_MAX  = 0.3
    WARMUP_EPOCHS  = 50

    def initialize(self):
        super().initialize()
        is_2d   = (len(self.configuration_manager.patch_size) == 2)
        num_cls = self.label_manager.num_segmentation_heads
        spacing = tuple(self.configuration_manager.spacing)   # (sz, sy, sx) in mm
        self._conv_bd_loss = ConvBoundaryLoss(
            num_cls, is_2d=is_2d, spacing=spacing).to(self.device)

    def train_step(self, batch: dict) -> dict:
        data   = batch['data'].to(self.device, non_blocking=True)
        target = batch['target']
        if isinstance(target, list):
            target = [t.to(self.device, non_blocking=True) for t in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)  # type: ignore
        ctx = (autocast(self.device.type, enabled=True)
               if self.device.type == 'cuda' else dummy_context())

        with ctx:
            output = self.network(data)
            l_seg  = self.loss(output, target)  # type: ignore

            # deep supervision: output/target 都是 list，取第 0 个（最高分辨率）
            pred_hr = output[0] if isinstance(output, (list, tuple)) else output
            gt_hr   = target[0] if isinstance(target, list)          else target

            l_bd = self._conv_bd_loss(pred_hr, gt_hr)
            lam  = min(self.current_epoch / max(self.WARMUP_EPOCHS, 1), 1.0) * self.LAMBDA_BD_MAX
            l    = l_seg + lam * l_bd

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)              # type: ignore
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)  # type: ignore
            self.grad_scaler.step(self.optimizer)                  # type: ignore
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)  # type: ignore
            self.optimizer.step()                                  # type: ignore

        return {
            'loss':  l.detach().cpu().numpy(),
            'l_seg': l_seg.detach().cpu().numpy(),
            'l_bd':  (lam * l_bd).detach().cpu().numpy(),
        }

