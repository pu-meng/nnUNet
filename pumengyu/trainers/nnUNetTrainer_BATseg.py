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
from torch import autocast

import os

import nnunetv2.training.nnUNetTrainer.nnUNetTrainer as _trainer_module
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader, crop_and_pad_nd
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.helpers import dummy_context

from pumengyu.tools.dist_field import compute_batch_distance_field


# ──────────────────────── DataLoader with precomputed EDT ────────────────────────

class BATsegDataLoader(nnUNetDataLoader):
    """
    Loads precomputed distance fields (<case_id>_dist.npy) from disk and
    crops them with the same bbox as the image/seg.  No online EDT needed.

    Falls back to online EDT if the precomputed file is missing.
    Run pumengyu/scripts/precompute_dist_fields.py once before training.
    """
    # Set by get_dataloaders before worker processes fork.
    _dist_params: dict = {'num_classes': None, 'spacing': None}

    def generate_train_batch(self) -> dict:
        selected_keys = self.get_indices()
        import torch
        from threadpoolctl import threadpool_limits

        data_all = None
        seg_all = None
        dist_all = None

        num_classes = self._dist_params['num_classes']
        spacing = self._dist_params['spacing']

        with torch.no_grad():
            with threadpool_limits(limits=1, user_api=None):
                for j, i in enumerate(selected_keys):
                    force_fg = self.get_do_oversample(j)
                    data, seg, seg_prev, properties = self._data.load_case(i)
                    shape = data.shape[1:]
                    bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg, properties['class_locations'])
                    bbox = [[lb, ub] for lb, ub in zip(bbox_lbs, bbox_ubs)]

                    data_cropped = torch.from_numpy(crop_and_pad_nd(data, bbox, 0)).float()
                    seg_cropped = torch.from_numpy(
                        crop_and_pad_nd(seg, bbox, -1, cast_cropped_to=np.int16)).to(torch.int16)
                    if seg_prev is not None:
                        seg_prev_cropped = torch.from_numpy(
                            crop_and_pad_nd(seg_prev, bbox, -1, cast_cropped_to=np.int16)).to(torch.int16)
                        seg_cropped = torch.cat((seg_cropped, seg_prev_cropped[None]), dim=0)

                    if self.patch_size_was_2d:
                        data_cropped = data_cropped[:, 0]
                        seg_cropped = seg_cropped[:, 0]

                    if self.transforms is not None:
                        transformed = self.transforms(**{'image': data_cropped, 'segmentation': seg_cropped})
                        data_sample = transformed['image']
                        seg_sample = transformed['segmentation']
                    else:
                        data_sample = data_cropped
                        seg_sample = seg_cropped

                    # ── distance field ──
                    dist_path = os.path.join(
                        self._data.source_folder, f'{i}_dist.npy')
                    if os.path.isfile(dist_path):
                        dist_full = np.load(dist_path)          # (K, H, W, D)
                        dist_cropped = torch.from_numpy(
                            crop_and_pad_nd(dist_full, bbox, 0).copy())
                    else:
                        # precomputed file missing: fall back to online EDT on seg
                        seg_for_edt = (seg_sample[0] if isinstance(seg_sample, list)
                                       else seg_sample)
                        dist_cropped = compute_batch_distance_field(
                            seg_for_edt.unsqueeze(0),
                            num_classes=num_classes, spacing=spacing,
                        )[0]  # (K, H, W, D)

                    if data_all is None:
                        data_all = torch.empty(
                            (self.batch_size, *data_sample.shape), dtype=torch.float32)
                    data_all[j] = data_sample

                    if isinstance(seg_sample, list):
                        if seg_all is None:
                            seg_all = [torch.empty((self.batch_size, *s.shape), dtype=s.dtype)
                                       for s in seg_sample]
                        for s_idx, s in enumerate(seg_sample):
                            seg_all[s_idx][j] = s
                    else:
                        if seg_all is None:
                            seg_all = torch.empty(
                                (self.batch_size, *seg_sample.shape), dtype=seg_sample.dtype)
                        seg_all[j] = seg_sample

                    if dist_all is None:
                        dist_all = torch.empty(
                            (self.batch_size, *dist_cropped.shape), dtype=torch.float32)
                    dist_all[j] = dist_cropped

        return {'data': data_all, 'target': seg_all, 'dist_field': dist_all,
                'keys': selected_keys}


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

    @property
    def decoder(self):
        return self.backbone.decoder

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

    def get_dataloaders(self):
        # Set EDT params on the class BEFORE worker processes fork, so children
        # inherit the populated dict and compute distance fields autonomously.
        BATsegDataLoader._dist_params = {
            'num_classes': self.label_manager.num_segmentation_heads,
            'spacing': self._spacing,
        }

        # Temporarily replace nnUNetDataLoader in the trainer module's namespace
        # so that super().get_dataloaders() constructs BATsegDataLoader instances.
        _original = _trainer_module.nnUNetDataLoader
        _trainer_module.nnUNetDataLoader = BATsegDataLoader
        try:
            mt_gen_train, mt_gen_val = super().get_dataloaders()
        finally:
            _trainer_module.nnUNetDataLoader = _original

        return mt_gen_train, mt_gen_val

    def train_step(self, batch: dict) -> dict:
        data = batch['data'].to(self.device, non_blocking=True)
        target = batch['target']
        if isinstance(target, list):
            target = [t.to(self.device, non_blocking=True) for t in target]
        else:
            target = target.to(self.device, non_blocking=True)

        gt_dist = batch['dist_field']  # precomputed offline and loaded in worker

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
