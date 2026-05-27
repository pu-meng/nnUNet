"""
nnUNet v2 preprocessing 核心工具

完整复现 nnUNet 的 CTNormalization + resampling + .b2nd/.pkl 写入流程，
不依赖 nnUNetv2_preprocess 命令，可单独对任意 case 运行。

参数来源（Dataset003_Liver）：
  target_spacing  = [1.0, 0.767578125, 0.767578125]  (z, y, x)
  clip_min        = -15.0   (foreground percentile_00_5)
  clip_max        = 197.0   (foreground percentile_99_5)
  norm_mean       = 99.48
  norm_std        = 37.14
"""

from __future__ import annotations
import json
import pickle
from pathlib import Path

import blosc2
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import zoom


# ──────────────────────────────────────────────────────────────────────────────
# 常量（从 Dataset003 的 nnUNetPlans.json + dataset_fingerprint.json 读取）
# ──────────────────────────────────────────────────────────────────────────────

TARGET_SPACING = [1.0, 0.767578125, 0.767578125]   # (z, y, x)
CLIP_MIN  = -15.0
CLIP_MAX  = 197.0
NORM_MEAN = 99.48
NORM_STD  = 37.14

TUMOR_CLS = 2
LIVER_CLS = 1


# ──────────────────────────────────────────────────────────────────────────────
# resampling
# ──────────────────────────────────────────────────────────────────────────────

def _resample_volume(
    arr: np.ndarray,
    orig_spacing_zyx: list[float],
    target_spacing_zyx: list[float],
    order: int,
) -> np.ndarray:
    """用 scipy.ndimage.zoom 做各向异性 resampling。"""
    zoom_factors = [
        orig_spacing_zyx[i] / target_spacing_zyx[i]
        for i in range(3)
    ]
    return zoom(arr, zoom_factors, order=order, prefilter=(order > 1))


# ──────────────────────────────────────────────────────────────────────────────
# normalization
# ──────────────────────────────────────────────────────────────────────────────

def ct_normalize(arr: np.ndarray) -> np.ndarray:
    """CTNormalization：clip → z-score（使用 Dataset003 的 foreground 统计）。"""
    arr = arr.astype(np.float32)
    arr = np.clip(arr, CLIP_MIN, CLIP_MAX)
    arr = (arr - NORM_MEAN) / (NORM_STD + 1e-8)
    return arr


# ──────────────────────────────────────────────────────────────────────────────
# class_locations（nnUNet 前景过采样需要）
# ──────────────────────────────────────────────────────────────────────────────

def get_class_locations(seg: np.ndarray) -> dict:
    """
    seg shape: (1, Z, Y, X)，int16
    返回 {cls: ndarray(N, 3)}，坐标为 (z, y, x)。
    """
    locs = {}
    for cls in [LIVER_CLS, TUMOR_CLS]:
        coords = np.argwhere(seg[0] == cls).astype(np.int64)  # (N, 3): z, y, x
        if len(coords) > 0:
            channel_col = np.zeros((len(coords), 1), dtype=np.int64)
            coords = np.hstack([channel_col, coords])          # (N, 4): channel, z, y, x
        locs[cls] = coords
    return locs


# ──────────────────────────────────────────────────────────────────────────────
# .b2nd 写入（与 nnUNet v2 完全兼容）
# ──────────────────────────────────────────────────────────────────────────────

def save_b2nd(arr: np.ndarray, path: Path) -> None:
    blosc2.asarray(np.ascontiguousarray(arr), urlpath=str(path), mode="w")


# ──────────────────────────────────────────────────────────────────────────────
# 主函数：nii.gz → .b2nd + .pkl
# ──────────────────────────────────────────────────────────────────────────────

def preprocess_case(
    ct_path: Path,
    seg_path: Path,
    case_id: str,
    out_dir: Path,
    verbose: bool = True,
) -> None:
    """
    将一个 case 的 CT + seg nii.gz 预处理成 nnUNet .b2nd + .pkl，
    输出到 out_dir（Dataset003 的 nnUNetPlans_3d_fullres 目录）。

    seg 约定：label 0=background, 1=liver, 2=tumor（无肿瘤 case 全为 0/1）
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 读取 CT ──────────────────────────────────────────────────────────────
    ct_sitk = sitk.ReadImage(str(ct_path))
    ct_arr  = sitk.GetArrayFromImage(ct_sitk).astype(np.float32)  # (Z, Y, X)
    # SimpleITK spacing 是 (x, y, z)，转成 (z, y, x)
    orig_spacing_zyx = list(reversed(ct_sitk.GetSpacing()))

    if verbose:
        print(f"  [{case_id}] orig spacing(zyx)={[round(s,3) for s in orig_spacing_zyx]}"
              f"  shape={ct_arr.shape}")

    # ── 读取 seg ─────────────────────────────────────────────────────────────
    seg_sitk = sitk.ReadImage(str(seg_path))
    seg_arr  = sitk.GetArrayFromImage(seg_sitk).astype(np.int16)  # (Z, Y, X)

    # ── 记录原始信息（用于 pkl）──────────────────────────────────────────────
    shape_before_crop = ct_arr.shape
    bbox = [[0, ct_arr.shape[0]], [0, ct_arr.shape[1]], [0, ct_arr.shape[2]]]

    sitk_stuff = {
        "spacing":   list(ct_sitk.GetSpacing()),
        "origin":    list(ct_sitk.GetOrigin()),
        "direction": list(ct_sitk.GetDirection()),
    }

    # ── Resampling ───────────────────────────────────────────────────────────
    ct_resampled  = _resample_volume(ct_arr,  orig_spacing_zyx, TARGET_SPACING, order=3)
    seg_resampled = _resample_volume(seg_arr, orig_spacing_zyx, TARGET_SPACING, order=0)
    seg_resampled = seg_resampled.astype(np.int16)

    if verbose:
        print(f"  [{case_id}] resampled shape={ct_resampled.shape}")

    # ── CTNormalization ──────────────────────────────────────────────────────
    ct_normed = ct_normalize(ct_resampled)

    # ── 添加 channel 维度 → (1, Z, Y, X) ────────────────────────────────────
    data_out = ct_normed[np.newaxis].astype(np.float32)
    seg_out  = seg_resampled[np.newaxis].astype(np.int16)

    # ── class_locations ──────────────────────────────────────────────────────
    class_locs = get_class_locations(seg_out)

    # ── 写 .b2nd ─────────────────────────────────────────────────────────────
    save_b2nd(data_out, out_dir / f"{case_id}.b2nd")
    save_b2nd(seg_out,  out_dir / f"{case_id}_seg.b2nd")

    # ── 写 .pkl ──────────────────────────────────────────────────────────────
    props = {
        "sitk_stuff":                              sitk_stuff,
        "spacing":                                 orig_spacing_zyx,
        "shape_before_cropping":                   shape_before_crop,
        "bbox_used_for_cropping":                  bbox,
        "shape_after_cropping_and_before_resampling": shape_before_crop,
        "class_locations":                         class_locs,
    }
    with open(out_dir / f"{case_id}.pkl", "wb") as f:
        pickle.dump(props, f)

    if verbose:
        liver_n  = len(class_locs.get(LIVER_CLS, []))
        tumor_n  = len(class_locs.get(TUMOR_CLS, []))
        print(f"  [{case_id}] done  liver_voxels={liver_n}  tumor_voxels={tumor_n}")
