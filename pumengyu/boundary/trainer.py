"""
nnUNetTrainer_BoundaryLoss
Adds boundary-aware surface distance loss (BATseg, arXiv:2412.06507) to
the standard nnUNet training loop.  Drop-in replacement for nnUNetTrainer.

Architecture
  Standard nnUNet backbone + parallel distance-field prediction head
  tapped from the highest-resolution decoder features.

Loss
  L = L_seg + L_boundary
  L_seg      = CE + Dice  (unchanged from nnUNetTrainer)
  L_boundary = mean(|e|² · |e|)  where e = pred_dist - gt_dist
  (cubic penalty: large errors near boundaries dominate; tiny far-field
  errors vanish automatically — no hard truncation needed in the loss.)

Distance field GT
  Computed online from the *augmented* segmentation mask inside each
  DataLoader worker process.  This is correct: spatial augmentation
  (rotation, scaling, elastic) changes geometry, so the distance field
  must be derived after the transform, not before.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch import autocast

import nnunetv2.training.nnUNetTrainer.nnUNetTrainer as _trainer_module
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader, crop_and_pad_nd
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.helpers import dummy_context

from pumengyu.boundary.dist_field import compute_batch_distance_field


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader
# ─────────────────────────────────────────────────────────────────────────────

class _BoundaryDataLoader(nnUNetDataLoader):
    """
    Extends nnUNetDataLoader to compute a surface distance field for each
    training sample after augmentation.
    Class-level _dist_params must be populated before worker processes fork.
    """
    _dist_params: dict = {'num_classes': None, 'spacing': None}

    def generate_train_batch(self) -> dict:
        selected_keys = self.get_indices()
        from threadpoolctl import threadpool_limits

        data_all = seg_all = dist_all = None
        num_classes = self._dist_params['num_classes']
        spacing     = self._dist_params['spacing']

        with torch.no_grad():
            with threadpool_limits(limits=1, user_api=None):
                for j, i in enumerate(selected_keys):
                    force_fg = self.get_do_oversample(j)
                    data, seg, seg_prev, properties = self._data.load_case(i)
                    shape = data.shape[1:]
                    bbox_lbs, bbox_ubs = self.get_bbox(
                        shape, force_fg, properties['class_locations'])
                    bbox = [[lb, ub] for lb, ub in zip(bbox_lbs, bbox_ubs)]

                    data_c = torch.from_numpy(crop_and_pad_nd(data, bbox, 0)).float()
                    seg_c  = torch.from_numpy(
                        crop_and_pad_nd(seg, bbox, -1,
                                        cast_cropped_to=np.int16)).to(torch.int16)
                    if seg_prev is not None:
                        seg_prev_c = torch.from_numpy(
                            crop_and_pad_nd(seg_prev, bbox, -1,
                                            cast_cropped_to=np.int16)).to(torch.int16)
                        seg_c = torch.cat((seg_c, seg_prev_c[None]), dim=0)

                    if self.patch_size_was_2d:
                        data_c = data_c[:, 0]
                        seg_c  = seg_c[:, 0]

                    if self.transforms is not None:
                        out       = self.transforms(image=data_c, segmentation=seg_c)
                        data_s    = out['image']
                        seg_s     = out['segmentation']
                    else:
                        data_s, seg_s = data_c, seg_c

                    seg_for_edt = seg_s[0] if isinstance(seg_s, list) else seg_s
                    dist_s = compute_batch_distance_field(
                        seg_for_edt.unsqueeze(0),
                        num_classes=num_classes,
                        spacing=spacing,
                    )[0]  # (K, H, W, D)

                    if data_all is None:
                        data_all = torch.empty(
                            (self.batch_size, *data_s.shape), dtype=torch.float32)
                    data_all[j] = data_s

                    if isinstance(seg_s, list):
                        if seg_all is None:
                            seg_all = [
                                torch.empty((self.batch_size, *s.shape), dtype=s.dtype)
                                for s in seg_s]
                        for idx, s in enumerate(seg_s):
                            seg_all[idx][j] = s
                    else:
                        if seg_all is None:
                            seg_all = torch.empty(
                                (self.batch_size, *seg_s.shape), dtype=seg_s.dtype)
                        seg_all[j] = seg_s

                    if dist_all is None:
                        dist_all = torch.empty(
                            (self.batch_size, *dist_s.shape), dtype=torch.float32)
                    dist_all[j] = dist_s

        return {'data': data_all, 'target': seg_all,
                'dist_field': dist_all, 'keys': selected_keys}


# ─────────────────────────────────────────────────────────────────────────────
# Network
# ─────────────────────────────────────────────────────────────────────────────

class _DistHeadNet(nn.Module):
    """
    Wraps a standard nnUNet backbone and adds a parallel distance-field head.
    train → (seg_out, dist_out)
    eval  → seg_out only  (inference-compatible with standard nnUNet pipeline)
    """

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

        decoder   = backbone.decoder
        in_ch     = decoder.encoder.output_channels[0]
        conv_op   = decoder.encoder.conv_op
        num_cls   = decoder.num_classes

        self.dist_head = conv_op(in_ch, num_cls, 1, 1, 0, bias=True)
        nn.init.zeros_(self.dist_head.weight)
        nn.init.zeros_(self.dist_head.bias)

        self._dec_features: torch.Tensor | None = None
        decoder.stages[-1].register_forward_hook(self._hook)

    @property
    def decoder(self):
        return self.backbone.decoder

    def _hook(self, module, inp, out):
        self._dec_features = out

    def forward(self, x: torch.Tensor):
        seg_out = self.backbone(x)
        if self.training:
            return seg_out, self.dist_head(self._dec_features)
        return seg_out


# ─────────────────────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────────────────────

def _boundary_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """L_boundary = mean(|e|² · |e|) = mean(e² · |e|)  where e = pred − gt."""
    pred = pred.float()
    gt   = gt.to(pred.device, dtype=pred.dtype)
    err  = pred - gt
    return (err.pow(2) * err.abs()).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class nnUNetTrainer_BoundaryLoss(nnUNetTrainer):

    def initialize(self):
        super().initialize()
        self._spacing = tuple(self.configuration_manager.spacing)

    @staticmethod
    def build_network_architecture(
        plans_manager,
        configuration_manager,
        num_input_channels,
        num_output_channels,
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
        return _DistHeadNet(backbone)

    def get_dataloaders(self):
        _BoundaryDataLoader._dist_params = {
            'num_classes': self.label_manager.num_segmentation_heads,
            'spacing':     self._spacing,
        }
        _orig = _trainer_module.nnUNetDataLoader
        _trainer_module.nnUNetDataLoader = _BoundaryDataLoader
        try:
            loaders = super().get_dataloaders()
        finally:
            _trainer_module.nnUNetDataLoader = _orig
        return loaders

    def train_step(self, batch: dict) -> dict:
        data   = batch['data'].to(self.device, non_blocking=True)
        target = batch['target']
        if isinstance(target, list):
            target = [t.to(self.device, non_blocking=True) for t in target]
        else:
            target = target.to(self.device, non_blocking=True)
        gt_dist = batch['dist_field']

        self.optimizer.zero_grad(set_to_none=True)  # type: ignore
        ctx = (autocast(self.device.type, enabled=True)
               if self.device.type == 'cuda' else dummy_context())
        with ctx:
            seg_out, dist_out = self.network(data)          # type: ignore
            l_seg = self.loss(seg_out, target)              # type: ignore
            l_ba  = _boundary_loss(dist_out, gt_dist)
            l     = l_seg + l_ba

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)       # type: ignore
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)  # type: ignore
            self.grad_scaler.step(self.optimizer)           # type: ignore
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)  # type: ignore
            self.optimizer.step()                           # type: ignore

        return {'loss': l.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict:
        return super().validation_step(batch)
