"""
图像特征提取模块。

对每个 case 的每个肿瘤 CC 提取：
  - 体积（体素数）
  - 肿瘤 HU：均值、std、P10、P90
  - 周围肝脏 HU（CC 外扩 N 体素与肝脏的交集）
  - 对比度 = 肿瘤HU均值 - 周围肝脏HU均值
  - CC 数量（per case）

输出 per-case 和 per-CC 两级 dict，供 dataset_profile 汇总。
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage
from tqdm import tqdm


LIVER_LABEL = 1
TUMOR_LABEL = 2
DILATION_RADIUS = 5   # 周围肝脏区域外扩体素数


def _load(path: Path) -> np.ndarray:
    return np.asarray(nib.load(str(path)).dataobj)


def _find(directory: Path, stem: str) -> Path | None:
    # 优先精确匹配，再尝试 nnUNet 多通道命名（_0000 后缀）
    for suf in (".nii.gz", ".nii"):
        p = directory / (stem + suf)
        if p.exists():
            return p
        p = directory / (stem + "_0000" + suf)
        if p.exists():
            return p
    return None


def _stems(directory: Path) -> list[str]:
    result = set()
    for f in directory.iterdir():
        for suf in (".nii.gz", ".nii"):
            if f.name.endswith(suf):
                result.add(f.name[: -len(suf)])
    return sorted(result)


PCTS = [5, 10, 20, 25, 30, 50, 70, 75, 80, 90, 95]


def _hu_stats(hu_values: np.ndarray, prefix: str = "") -> dict:
    if hu_values.size == 0:
        d = {"mean": np.nan, "std": np.nan}
        d.update({f"p{p}": np.nan for p in PCTS})
        return {f"{prefix}{k}": v for k, v in d.items()} if prefix else d
    d = {
        "mean": float(np.mean(hu_values)),
        "std":  float(np.std(hu_values)),
    }
    d.update({f"p{p}": float(np.percentile(hu_values, p)) for p in PCTS})
    return {f"{prefix}{k}": v for k, v in d.items()} if prefix else d


def extract_case(ct: np.ndarray, gt: np.ndarray) -> dict | None:
    """
    提取单个 case 的全部特征。
    返回 None 表示无肿瘤 case。
    """
    tumor_mask = gt == TUMOR_LABEL
    liver_mask = gt == LIVER_LABEL
    if tumor_mask.sum() == 0:
        return None

    labeled, n_cc = ndimage.label(tumor_mask)
    cc_list = []

    for i in range(1, n_cc + 1):
        cc_mask = labeled == i
        size = int(cc_mask.sum())

        # 肿瘤 HU（全百分位）
        tumor_hu = ct[cc_mask].astype(np.float32)
        t_stats = _hu_stats(tumor_hu, prefix="tumor_hu_")

        # 周围肝脏 HU
        struct = ndimage.generate_binary_structure(3, 1)
        dilated = ndimage.binary_dilation(cc_mask, structure=struct,
                                          iterations=DILATION_RADIUS)
        ring = dilated & liver_mask & ~tumor_mask
        if ring.sum() > 0:
            liver_hu = ct[ring].astype(np.float32)
            l_stats = _hu_stats(liver_hu, prefix="liver_hu_")
            contrast = t_stats["tumor_hu_mean"] - l_stats["liver_hu_mean"]
        else:
            l_stats = _hu_stats(np.array([]), prefix="liver_hu_")
            contrast = np.nan

        cc_list.append({
            "size": size,
            **t_stats,
            **l_stats,
            "contrast": contrast,
        })

    return {
        "n_cc":         n_cc,
        "total_voxels": int(tumor_mask.sum()),
        "cc_list":      cc_list,
    }


def extract_dataset(gt_dir: Path, img_dir: Path,
                    tumor_label: int = TUMOR_LABEL) -> dict[str, dict | None]:
    """
    对 gt_dir 下所有 case 提取特征。
    返回 {case_name: case_features or None(无肿瘤)}
    """
    global TUMOR_LABEL
    TUMOR_LABEL = tumor_label

    stems = _stems(gt_dir)
    results: dict[str, dict | None] = {}

    for stem in tqdm(stems, desc="提取图像特征", unit="case"):
        gt_path  = _find(gt_dir,  stem)
        img_path = _find(img_dir, stem)
        if gt_path is None or img_path is None:
            continue
        gt = _load(gt_path).astype(np.int16)
        ct = _load(img_path).astype(np.float32)
        results[stem] = extract_case(ct, gt)

    return results
