"""
Truncated normalized surface distance field (BATseg, Section 4.2).

For each class k:
  - Inside  (p_i ∈ mask):  d_i  = positive distance to boundary   → [0.5, 1]
  - Outside (p_j ∉ mask):  d_j  = negative distance, truncated    → (0, 0.5)
  - Truncated outside (|d_j| > max_d):                             → 0
  - Empty mask:                                                     → all zeros

EDT uses physical spacing so anisotropic voxels are handled correctly.
"""

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt


def save_dist_npz(dist: np.ndarray, seg: np.ndarray, path: str, pad: int = 20) -> None:
    """Crop dist field to seg bounding box and save as uint8 npz."""
    H, W, D = seg.shape
    nz = np.where(seg > 0)
    if len(nz[0]) == 0:
        np.savez_compressed(path,
                            data=np.zeros((dist.shape[0], 1, 1, 1), dtype=np.uint8),
                            bbox=np.array([0, H, 0, W, 0, D]),
                            shape=np.array([H, W, D]))
        return
    h_min = max(0, int(nz[0].min()) - pad)
    h_max = min(H, int(nz[0].max()) + pad + 1)
    w_min = max(0, int(nz[1].min()) - pad)
    w_max = min(W, int(nz[1].max()) + pad + 1)
    d_min = max(0, int(nz[2].min()) - pad)
    d_max = min(D, int(nz[2].max()) + pad + 1)
    cropped = dist[:, h_min:h_max, w_min:w_max, d_min:d_max]
    uint8_data = (cropped * 255).clip(0, 255).astype(np.uint8)
    np.savez_compressed(path,
                        data=uint8_data,
                        bbox=np.array([h_min, h_max, w_min, w_max, d_min, d_max]),
                        shape=np.array([H, W, D]))


def load_dist_npz(path: str) -> np.ndarray:
    """Load dist npz and reconstruct full-size float32 array."""
    d = np.load(path)
    uint8_data = d['data']
    h_min, h_max, w_min, w_max, d_min, d_max = d['bbox']
    H, W, D = d['shape']
    K = uint8_data.shape[0]
    dist_full = np.zeros((K, H, W, D), dtype=np.float32)
    dist_full[:, h_min:h_max, w_min:w_max, d_min:d_max] = uint8_data.astype(np.float32) / 255.0
    return dist_full


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

        # inside: positive, outside: negative
        combined = np.where(mask, d_inside, -d_outside)

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
        spacing:     physical voxel size in mm (from configuration_manager.spacing).
                     None → isotropic (not recommended for real MRI data).
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
