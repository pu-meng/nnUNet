"""
nnUNetTrainer_BATseg
BATseg boundary-aware loss integrated into nnUNet.

Architecture:
  Parallel Conv head on the highest-resolution decoder features predicts
  a K-channel surface distance field alongside the segmentation head.

Loss:
  ℓ = ℓ_CE + ℓ_Dice + ℓ_ba
  ℓ_ba = mean(|f - f̄|³)
  (BATseg Eq.3 sign corrected — the negative sign in the paper is a typo;
   with it, the gradient moves predictions *away* from the target.)

Usage:
  nnUNetv2_train DATASET_ID 3d_fullres FOLD -tr nnUNetTrainer_BATseg

Reference: BATseg, arXiv:2412.06507
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.helpers import dummy_context

from pumengyu.tools.dist_field import compute_batch_distance_field


# ──────────────────────── Network Wrapper ────────────────────────

class BATsegNet(nn.Module):
    """
    Wraps a standard nnUNet backbone and adds a parallel distance field head.

    train mode → returns (seg_out, dist_out)
    eval  mode → returns seg_out only  (inference-compatible)
    """

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

        decoder = backbone.decoder
        in_ch = decoder.encoder.output_channels[0]
        conv_op = decoder.encoder.conv_op
        num_classes = decoder.num_classes

        self.dist_head = conv_op(in_ch, num_classes, 1, 1, 0, bias=True)
        nn.init.zeros_(self.dist_head.weight)
        nn.init.zeros_(self.dist_head.bias)

        self._dec_features: torch.Tensor | None = None
        decoder.stages[-1].register_forward_hook(self._capture_hook)

    def _capture_hook(self, module, inp, out):
        self._dec_features = out

    def forward(self, x: torch.Tensor):
        seg_out = self.backbone(x)
        if self.training:
            return seg_out, self.dist_head(self._dec_features)
        return seg_out


# ──────────────────────── Boundary-Aware Loss ────────────────────────

def boundary_aware_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """
    ℓ_ba = mean(|f - f̄|³)

    Self-weighting: large errors (near boundaries) contribute |e|³,
    small errors (truncated background → GT≈0) vanish automatically.
    No stop-gradient on the weight term (BATseg ablation Table 8: -2% Dice).

    pred: (B, K, H, W, D) float, predicted distance field
    gt:   (B, K, H, W, D) float in [0, 1], ground truth distance field
    """
    pred = pred.float()
    gt = gt.to(pred.device, dtype=pred.dtype)
    err = pred - gt
    return (err.pow(2) * err.abs()).mean()


# ──────────────────────── Trainer ────────────────────────

class nnUNetTrainer_BATseg(nnUNetTrainer):

    def initialize(self):
        super().initialize()
        # Cache spacing once; used every train_step for EDT sampling
        self._spacing = tuple(self.configuration_manager.spacing)

    @staticmethod
    def build_network_architecture(
        plans_manager, configuration_manager,
        num_input_channels, num_output_channels,
        enable_deep_supervision=True,
    ) -> nn.Module:
        backbone = get_network_from_plans(
            configuration_manager.network_arch_class_name,
            configuration_manager.network_arch_init_kwargs,
            configuration_manager.network_arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            allow_init=True,
            deep_supervision=enable_deep_supervision,
        )
        return BATsegNet(backbone)

    def train_step(self, batch: dict) -> dict:
        data = batch['data'].to(self.device, non_blocking=True)
        target = batch['target']
        if isinstance(target, list):
            target = [t.to(self.device, non_blocking=True) for t in target]
        else:
            target = target.to(self.device, non_blocking=True)

        # Distance field is computed from the *augmented* full-resolution target
        # (aug has already happened inside the dataloader), so it stays in sync.
        target_full = target[0] if isinstance(target, list) else target
        gt_dist = compute_batch_distance_field(
            target_full,
            num_classes=self.label_manager.num_segmentation_heads,
            spacing=self._spacing,
        )

        self.optimizer.zero_grad(set_to_none=True)
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            seg_out, dist_out = self.network(data)
            l_seg = self.loss(seg_out, target)
            l_ba  = boundary_aware_loss(dist_out, gt_dist)
            l = l_seg + l_ba

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        return {'loss': l.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict:
        # network.eval() → returns only seg_out; base class handles the rest
        return super().validation_step(batch)
