"""
CC（连通域）分析核心模块。

提供：
  - GT CC 大小分布统计
  - 预测 CC 分类（TP / FP-in-tumor / FP-in-notumor / FN）
  - Separability Gap 计算
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage
from tqdm import tqdm


# ──────────────────────────── IO ─────────────────────────────────────

def _load(path: Path) -> np.ndarray:
    return np.asarray(nib.load(str(path)).dataobj, dtype=np.int16)


def _find(directory: Path, stem: str) -> Path | None:
    for suf in (".nii.gz", ".nii"):
        p = directory / (stem + suf)
        if p.exists():
            return p
    return None


def _cases(directory: Path) -> list[str]:
    stems = set()
    for f in directory.iterdir():
        name = f.name
        for suf in (".nii.gz", ".nii"):
            if name.endswith(suf):
                stems.add(name[: -len(suf)])
    return sorted(stems)


def _cc_sizes(mask: np.ndarray) -> list[int]:
    if mask.sum() == 0:
        return []
    labeled, n = ndimage.label(mask)
    return [(labeled == i).sum() for i in range(1, n + 1)]


# ──────────────────────────── GT analysis ────────────────────────────

def analyze_gt(gt_dir: Path, tumor_label: int = 2) -> dict:
    """
    分析所有 GT case 的肿瘤 CC 分布（不依赖任何模型预测）。

    返回 dict：
      cases_total, cases_tumor, cases_notumor,
      notumor_list,
      gt_cc_sizes: list[int]（所有有肿瘤 case 的 CC 体素数）
      per_case_cc_count: dict[str, int]
    """
    files = _cases(gt_dir)
    gt_cc_sizes: list[int] = []
    per_case_cc_count: dict[str, int] = {}
    notumor_list: list[str] = []

    for stem in tqdm(files, desc="GT 分析", unit="case"):
        gt_path = _find(gt_dir, stem)
        if gt_path is None:
            continue
        gt = _load(gt_path)
        mask = gt == tumor_label
        if mask.sum() == 0:
            notumor_list.append(stem)
        else:
            sizes = _cc_sizes(mask)
            gt_cc_sizes.extend(sizes)
            per_case_cc_count[stem] = len(sizes)

    return {
        "cases_total": len(files),
        "cases_tumor": len(per_case_cc_count),
        "cases_notumor": len(notumor_list),
        "notumor_list": notumor_list,
        "gt_cc_sizes": gt_cc_sizes,
        "per_case_cc_count": per_case_cc_count,
    }


# ──────────────────────────── Pred analysis ──────────────────────────

def analyze_pred(gt_dir: Path, pred_dir: Path, tumor_label: int = 2) -> dict:
    """
    对有预测文件的 case 进行 CC 分类。

    返回 dict：
      tp_cc, fp_in_tumor_cc, fp_in_notumor_cc : list[int]  体素数列表（聚合用）
      per_cc_pred : list[dict]  每个预测 CC 的详细记录
        {case, cc_idx, size, label: TP/FP_tumor/FP_notumor}
      per_case_pred : list[dict]  每个 case 的汇总
        {case, has_tumor, n_pred_cc, n_tp, n_fp_tumor, n_fp_notumor,
         fp_vox_total, fn_vox_total}
      notumor_fp_detail : dict[str, list[int]]
      missing : list[str]
    """
    gt_stems = _cases(gt_dir)
    tp_cc: list[int] = []
    fp_in_tumor_cc: list[int] = []
    fp_in_notumor_cc: list[int] = []
    notumor_fp_detail: dict[str, list[int]] = {}
    missing: list[str] = []
    per_cc_pred: list[dict] = []
    per_case_pred: list[dict] = []

    for stem in tqdm(gt_stems, desc="预测 CC 分类", unit="case"):
        gt_path   = _find(gt_dir,  stem)
        pred_path = _find(pred_dir, stem)
        if gt_path is None:
            continue
        if pred_path is None:
            missing.append(stem)
            continue

        gt        = _load(gt_path)
        pred      = _load(pred_path)
        gt_mask   = gt   == tumor_label
        pred_mask = pred == tumor_label
        has_tumor = gt_mask.sum() > 0

        fn_vox = int((gt_mask & ~pred_mask).sum())
        fp_vox = int((pred_mask & ~gt_mask).sum())

        case_n_tp = case_n_fp_t = case_n_fp_n = 0

        if pred_mask.sum() > 0:
            labeled, n = ndimage.label(pred_mask)
            for i in range(1, n + 1):
                cc   = labeled == i
                size = int(cc.sum())
                if has_tumor:
                    if (cc & gt_mask).sum() > 0:
                        label = "TP"
                        tp_cc.append(size)
                        case_n_tp += 1
                    else:
                        label = "FP_tumor"
                        fp_in_tumor_cc.append(size)
                        case_n_fp_t += 1
                else:
                    label = "FP_notumor"
                    fp_in_notumor_cc.append(size)
                    case_n_fp_n += 1

                per_cc_pred.append({
                    "case": stem, "cc_idx": i,
                    "size": size, "label": label,
                })

        if not has_tumor:
            sizes = [r["size"] for r in per_cc_pred if r["case"] == stem]
            notumor_fp_detail[stem] = sizes

        per_case_pred.append({
            "case":          stem,
            "has_tumor":     int(has_tumor),
            "n_pred_cc":     case_n_tp + case_n_fp_t + case_n_fp_n,
            "n_tp":          case_n_tp,
            "n_fp_tumor":    case_n_fp_t,
            "n_fp_notumor":  case_n_fp_n,
            "fp_vox_total":  fp_vox,
            "fn_vox_total":  fn_vox,
        })

    return {
        "tp_cc": tp_cc,
        "fp_in_tumor_cc": fp_in_tumor_cc,
        "fp_in_notumor_cc": fp_in_notumor_cc,
        "per_cc_pred": per_cc_pred,
        "per_case_pred": per_case_pred,
        "notumor_fp_detail": notumor_fp_detail,
        "missing": missing,
    }


# ──────────────────────────── Separability ───────────────────────────

def separability_gap(gt_cc_sizes: list[int], fp_notumor_cc: list[int]) -> dict:
    min_tp = int(min(gt_cc_sizes)) if gt_cc_sizes else None
    max_fp = int(max(fp_notumor_cc)) if fp_notumor_cc else None
    if min_tp is not None and max_fp is not None:
        gap = min_tp - max_fp
        feasible = gap > 0
    else:
        gap = None
        feasible = None
    return {"min_tp_cc": min_tp, "max_fp_notumor_cc": max_fp,
            "gap": gap, "feasible": feasible}


# ──────────────────────────── helpers ────────────────────────────────

def pct_str(arr: list[int]) -> str:
    if not arr:
        return "N/A"
    a = np.array(arr)
    return (f"min={a.min()}  P5={int(np.percentile(a,5))}"
            f"  P25={int(np.percentile(a,25))}  P50={int(np.percentile(a,50))}"
            f"  P75={int(np.percentile(a,75))}  P95={int(np.percentile(a,95))}"
            f"  max={a.max()}")


def size_groups(sizes: list[int]) -> dict[str, int]:
    if not sizes:
        return {"极小(<5k)": 0, "小(5k-50k)": 0, "中等(50k-300k)": 0, "大(>=300k)": 0}
    a = np.array(sizes)
    return {
        "极小(<5k)":     int((a < 5000).sum()),
        "小(5k-50k)":   int(((a >= 5000) & (a < 50000)).sum()),
        "中等(50k-300k)": int(((a >= 50000) & (a < 300000)).sum()),
        "大(>=300k)":   int((a >= 300000).sum()),
    }
