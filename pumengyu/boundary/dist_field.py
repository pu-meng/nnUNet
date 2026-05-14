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
#scipy是建立在numpy之上得Python科学计算库,提供数学,信号处理,图像处理等算法,
#edt是Euclidean Distance Transform,欧式距离变换
#distance_transform_edt是输入一个bool数组,对每个voxel计算 到最近边界得欧式距离

def compute_surface_distance_field(
    seg: np.ndarray,
    num_classes: int,
    spacing: tuple | list | None = None,
) -> np.ndarray:
    """
    Args:
    seg可以是[D,H,W]也可以是[H,W],但是不可以是四个维度或者5个维度
        seg:         (H, W, D) int array
        num_classes: K (0 = background)
        spacing:     physical voxel size in mm, e.g. (1.0, 0.77, 0.77).
                     None → unit spacing (isotropic).
    Returns:
        (K, H, W, D) float32, values in [0, 1]; background channel = zeros.
    """
    dist_field = np.zeros((num_classes,) + seg.shape, dtype=np.float32)
#(2,)+(256,256,256) = (2,256,256,256)这个是元组拼接技巧
    for k in range(1, num_classes):
        mask = (seg == k)
        if not mask.any():
            continue
#spacing是告诉distance_transform_edt输入的mask是物理坐标,而不是像素坐标,
#要根据spacing来计算距离,如果spacing是None,则默认spacing=(1.0, 1.0, 1.0),即单位间距,等效于像素坐标

        d_inside  = distance_transform_edt(mask,  sampling=spacing).astype(np.float32)#type:ignore
        d_outside = distance_transform_edt(~mask, sampling=spacing).astype(np.float32)#type:ignore
#distance_transform_edt是计算距离变换,True位置有值,False位置为0,
#distance_transform_edt(mask)是真实得3D计算,边界=True/false得交界
        max_d = float(d_inside.max())
        if max_d == 0.0:
            continue

        truncated = (~mask) & (d_outside > max_d)
        #&是逐元素按位与,与bool数组是逻辑AND,
        combined  = np.where(mask, d_inside, -d_outside)
        #np.where(条件,条件为True取这个,条件为False取这个)
        #mask位置取d_inside,非mask位置取-d_outside,得到一个距离场,mask内是正数,mask外是负数
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
    #.detach是把tensor从计算图中分离,不再追踪梯度

    B = target_np.shape[0]
    spatial = target_np.shape[2:]

    dist_batch = np.zeros((B, num_classes) + spatial, dtype=np.float32)
    for b in range(B):
        dist_batch[b] = compute_surface_distance_field(
            target_np[b, 0], num_classes, spacing=spacing
        )
#这里的输入target_np:[B,1,H,W,D],所以target_np[b,0]是[H,W,D],是单个样本的标签图
    return torch.from_numpy(dist_batch)
