"""
Truncated normalized surface distance field (BATseg, Section 4.2).

For each class k:
  - Inside  (p_i ∈ mask):  d_i  = positive distance to boundary   → [0.5, 1]
  - Outside (p_j ∉ mask):  d_j  = negative distance, truncated    → (0, 0.5)
  - Truncated outside (|d_j| > max_d):                             → 0
  - Empty mask:                                                     → all zeros

EDT uses physical spacing so anisotropic voxels are handled correctly.

距离场计算
"""

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt


def compute_surface_distance_field(
    seg: np.ndarray,
    num_classes: int,
    spacing: tuple | list | None = None,
) -> np.ndarray:
    """
    Args:
        seg:         (H, W, D) int array
        num_classes: K (0 = background)
        spacing:     physical voxel size in mm, e.g. (1.0, 0.77, 0.77).
                     None → unit spacing (isotropic).
    Returns:
        (K, H, W, D) float32, values in [0, 1]; background channel = zeros.
    """
    dist_field = np.zeros((num_classes,) + seg.shape, dtype=np.float32)

    for k in range(1, num_classes):
        mask = (seg == k)
        if not mask.any():
            continue

        d_inside  = distance_transform_edt(mask,  sampling=spacing).astype(np.float32)
        d_outside = distance_transform_edt(~mask, sampling=spacing).astype(np.float32)

        max_d = float(d_inside.max())
        if max_d == 0.0:
            continue

        truncated = (~mask) & (d_outside > max_d)
        combined  = np.where(mask, d_inside, -d_outside)
        normalized = (combined / max_d + 1.0) / 2.0
        normalized[truncated] = 0.0

        dist_field[k] = normalized

    return dist_field


def compute_batch_distance_field(
    target: torch.Tensor,
    num_classes: int,
    spacing: tuple | list | None = None,
) -> torch.Tensor:
    """
    Args:
        target:      (B, 1, H, W, D) integer class label tensor
        num_classes: K
        spacing:     physical voxel size in mm.  None → isotropic.
    Returns:
        (B, K, H, W, D) float32 CPU tensor
    """
    target_np = target.detach().cpu().numpy().astype(np.int32)
    B = target_np.shape[0]
    spatial = target_np.shape[2:]

    dist_batch = np.zeros((B, num_classes) + spatial, dtype=np.float32)
    for b in range(B):
        dist_batch[b] = compute_surface_distance_field(
            target_np[b, 0], num_classes, spacing=spacing
        )

    return torch.from_numpy(dist_batch)
