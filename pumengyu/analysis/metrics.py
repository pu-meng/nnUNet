"""
从 nnUNet summary.json 读取 per-case 指标，计算大小分层统计。
"""
from __future__ import annotations
import json
import math
from pathlib import Path

import numpy as np
import nibabel as nib
from batchgenerators.utilities.file_and_folder_operations import load_pickle


# ──────────────────────── 读取 summary.json ──────────────────────────

def load_summary(summary_path: Path) -> list[dict]:
    """返回 per-case 列表，每条含 case_name, dice_liver, dice_tumor, FP, FN, n_ref。"""
    data = json.load(open(summary_path))
    records = []
    for c in data["metric_per_case"]:
        pred_file = Path(c["prediction_file"])
        case_name = pred_file.stem.replace(".nii", "")
        m1 = c["metrics"].get("1", {})
        m2 = c["metrics"].get("2", {})
        dice_tumor = m2.get("Dice", float("nan"))
        records.append({
            "case": case_name,
            "dice_liver": m1.get("Dice", float("nan")),
            "dice_tumor": dice_tumor if not math.isnan(dice_tumor) else None,
            "FP_tumor": m2.get("FP", 0),
            "FN_tumor": m2.get("FN", 0),
            "n_ref_tumor": m2.get("n_ref", 0),
            "n_pred_tumor": m2.get("n_pred", 0),
        })
    return records


# ──────────────────────── 肿瘤大小分类 ──────────────────────────────

def classify_size(n_ref_vox: int) -> str:
    if n_ref_vox == 0:
        return "无肿瘤"
    elif n_ref_vox < 5000:
        return "极小(<5k)"
    elif n_ref_vox < 50000:
        return "小(5k-50k)"
    elif n_ref_vox < 300000:
        return "中等(50k-300k)"
    else:
        return "大(>=300k)"


def add_size_category(records: list[dict]) -> list[dict]:
    for r in records:
        r["size_cat"] = classify_size(r["n_ref_tumor"])
    return records


# ──────────────────────── 聚合统计 ───────────────────────────────────

SIZE_ORDER = ["极小(<5k)", "小(5k-50k)", "中等(50k-300k)", "大(>=300k)", "无肿瘤"]

def aggregate(records: list[dict]) -> dict:
    """返回分组 dice 均值、无肿瘤误报率、Overall 等。"""
    tumor_cases = [r for r in records if r["size_cat"] != "无肿瘤"]
    notumor_cases = [r for r in records if r["size_cat"] == "无肿瘤"]

    # 无肿瘤误报
    fp_notumor = [r for r in notumor_cases if (r["n_pred_tumor"] or 0) > 0]

    # 综合 Dice（无肿瘤误报计 0，无肿瘤正确排除不计入）
    scores = []
    for r in records:
        if r["size_cat"] == "无肿瘤":
            if (r["n_pred_tumor"] or 0) > 0:
                scores.append(0.0)   # 误报
            # 正确排除不加入均值（nnUNet 行为）
        else:
            d = r["dice_tumor"]
            if d is not None and not math.isnan(d):
                scores.append(d)
            else:
                scores.append(0.0)

    overall = float(np.mean(scores)) if scores else float("nan")

    # 分组 dice
    by_size: dict[str, list[float]] = {k: [] for k in SIZE_ORDER}
    for r in tumor_cases:
        d = r["dice_tumor"]
        if d is not None and not math.isnan(d):
            by_size[r["size_cat"]].append(d)

    size_stats = {}
    for cat in SIZE_ORDER[:-1]:
        vals = by_size[cat]
        size_stats[cat] = {
            "n": len(vals),
            "mean": float(np.mean(vals)) if vals else float("nan"),
            "std":  float(np.std(vals))  if vals else float("nan"),
        }

    return {
        "overall": overall,
        "n_tumor_cases": len(tumor_cases),
        "n_notumor_cases": len(notumor_cases),
        "n_fp_notumor": len(fp_notumor),
        "notumor_fp_rate": len(fp_notumor) / len(notumor_cases) if notumor_cases else float("nan"),
        "size_stats": size_stats,
        "all_tumor_dice": [r["dice_tumor"] for r in tumor_cases
                           if r["dice_tumor"] is not None and not math.isnan(r["dice_tumor"])],
    }
