"""Refresh paper-facing report metrics from existing artifacts only.

This utility never runs inference and never removes or regenerates visualization
files. It rewrites metric text from existing ``summary.json``/predictions and
prints an artifact audit so incomplete historical runs remain explicit.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from pumengyu.tools.analyasis.metric_standard import aggregate_liver_tumor_metrics


WORKSPACE = Path("/home/PuMengYu/nnUNet_workspace")
INTERNAL_ROOT = WORKSPACE / "results_v2" / "Dataset003_Liver"
IRCADB_ROOT = WORKSPACE / "results_v2" / "IRCADb" / "source_only"
HCC_ROOT = WORKSPACE / "results_v2" / "Dataset013_HCCReferencedCT" / "source_only"


def _rows_from_summary(summary_path: Path) -> tuple[list[dict], list[dict]]:
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    positive, negative = [], []
    for case in data["metric_per_case"]:
        metrics = case["metrics"]
        liver = metrics.get("1", {})
        tumor = metrics.get("2", {})
        tp, fp, fn = (int(tumor.get(k, 0)) for k in ("TP", "FP", "FN"))
        liver_dice = float(liver["Dice"])
        if tp + fn == 0:
            negative.append({"liver_dice": liver_dice, "pred_tumor": tp + fp})
            continue
        denom_p, denom_r = tp + fp, tp + fn
        dice = float(tumor["Dice"])
        positive.append({
            "dice": dice,
            "jaccard": tp / (tp + fp + fn),
            "recall": tp / denom_r,
            "fnr": fn / denom_r,
            "precision": tp / denom_p if denom_p else 0.0,
            "fdr": fp / denom_p if denom_p else 0.0,
            "liver_dice": liver_dice,
        })
    return positive, negative


def _rows_from_legacy_report(text: str) -> tuple[list[dict], list[dict]]:
    """Recover rounded positive rows for the two historical reports lacking summary.json."""
    positive = []
    row_re = re.compile(
        r"^\s+(liver_\d+)\s+(\d\.\d{4})\s+(\d\.\d{4})\s+"
        r"(\d\.\d{4})\s+(\d\.\d{4})\s+[\d,]+\s+[\d,]+\s+[\d,]+\s+",
        re.MULTILINE,
    )
    seen = set()
    for match in row_re.finditer(text):
        case, dice, recall, precision, fdr = match.groups()
        if case in seen:
            continue
        seen.add(case)
        dice_f = float(dice)
        positive.append({
            "dice": dice_f,
            "jaccard": dice_f / (2 - dice_f),
            "recall": float(recall),
            "fnr": 1 - float(recall),
            "precision": float(precision),
            "fdr": float(fdr),
            "liver_dice": None,
        })
    liver_match = re.search(r"Liver\s+Dice: mean=(\d\.\d{4})", text, re.DOTALL)
    if not liver_match:
        raise ValueError("legacy report lacks Liver mean")
    liver_mean = float(liver_match.group(1))
    for row in positive:
        row["liver_dice"] = liver_mean

    negative = []
    neg_re = re.compile(
        r"^\s+(liver_\d+)\s+liver_dice=(\d\.\d{4})\s+pred_tumor=([\d,]+)",
        re.MULTILINE,
    )
    for _, liver, pred in neg_re.findall(text):
        negative.append({"liver_dice": float(liver), "pred_tumor": int(pred.replace(",", ""))})
    if len(positive) != 23 or len(negative) != 3:
        raise ValueError(f"legacy rows mismatch: positive={len(positive)}, negative={len(negative)}")
    # Preserve the exact all-case Liver mean printed by the historical report.
    current_sum = sum(r["liver_dice"] for r in positive + negative)
    correction = (liver_mean * (len(positive) + len(negative)) - current_sum) / len(positive)
    for row in positive:
        row["liver_dice"] += correction
    return positive, negative


def _metric_section(positive: list[dict], negative: list[dict]) -> str:
    agg = aggregate_liver_tumor_metrics(positive, negative)
    p, ref = agg["primary"], agg["nnunet_reference"]
    lines = [
        "Tumor 综合指标（PMY-LT-v1 论文主口径；仅 GT 有肿瘤 case）",
        f"  无肿瘤 case (n={len(negative)}) 的 Tumor Dice/Recall/Precision 统一为 N/A，不进入均值",
    ]
    for label, key in (("Dice", "dice"), ("Jaccard", "jaccard"), ("Recall", "recall"),
                       ("FNR", "fnr"), ("Precision", "precision"), ("FDR", "fdr")):
        lines.append(
            f"  {label:<12}: mean={p[key]['mean']:.4f}  std={p[key]['std']:.4f}"
            f"  (有肿瘤 n={len(positive)})"
        )
    lines += [
        f"  Overall     : (all-case liver {agg['liver']['mean']:.4f}"
        f" + positive-only tumor {p['dice']['mean']:.4f}) / 2 = {p['overall']:.4f}",
        f"  构成        : 全部 n={agg['n_all']} | 有肿瘤 n={len(positive)}"
        f" | 无肿瘤 n={len(negative)} (TN={agg['n_negative_tn']}, FP={agg['n_negative_fp']})",
        "",
        "nnUNet foreground_mean 参考（仅供 summary.json 追溯，不参与论文排名）",
        f"  Tumor Dice : mean={ref['tumor_dice']['mean']:.4f}"
        f"  std={ref['tumor_dice']['std']:.4f}  (valid n={ref['n_tumor_valid']})",
        f"  Overall    : {ref['overall']:.4f}",
        "",
    ]
    return "\n".join(lines) + "\n"


def _replace_section(report_path: Path, positive: list[dict], negative: list[dict]) -> None:
    text = report_path.read_text(encoding="utf-8")
    updated, count = re.subn(
        r"Tumor 综合指标[^\n]*\n.*?(?=FPV / FNV)",
        _metric_section(positive, negative),
        text,
        count=1,
        flags=re.DOTALL,
    )
    if count != 1:
        raise ValueError(f"metric section not found: {report_path}")
    updated = updated.replace("tumor_dice=0，体现为 liver Dice 下降", "tumor_dice=N/A，误报单独报告")
    report_path.write_text(updated, encoding="utf-8")


def refresh_internal() -> list[dict]:
    audit = []
    for report in sorted(INTERNAL_ROOT.glob("*/fold_0/test_report_custom.txt")):
        fold = report.parent
        summary = fold / "test_prediction" / "summary.json"
        source = "summary.json"
        if summary.exists():
            positive, negative = _rows_from_summary(summary)
        else:
            positive, negative = _rows_from_legacy_report(report.read_text(encoding="utf-8"))
            source = "legacy report rows (4-decimal recovery)"
        _replace_section(report, positive, negative)
        audit.append({
            "domain": "internal", "method": fold.parent.name, "pred": len(list((fold / "test_prediction").glob("*.nii.gz"))),
            "summary": summary.exists(), "report": True,
            "png": len(list((fold / "vis_png_custom").glob("*.png"))), "source": source,
        })
    return audit


def refresh_external(root: Path, domain: str) -> list[dict]:
    audit = []
    for method_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        pred_dir = method_dir / "predictions"
        report = method_dir / "report_custom.txt"
        summary = pred_dir / "summary.json"
        if not summary.exists() or not report.exists():
            audit.append({
                "domain": domain, "method": method_dir.name,
                "pred": len(list(pred_dir.glob("*.nii.gz"))), "summary": summary.exists(),
                "report": report.exists(), "png": len(list((method_dir / "test_viz").rglob("*.png"))),
                "source": "SKIP missing summary/report",
            })
            continue
        positive, negative = _rows_from_summary(summary)
        _replace_section(report, positive, negative)
        audit.append({
            "domain": domain, "method": method_dir.name,
            "pred": len(list(pred_dir.glob("*.nii.gz"))), "summary": summary.exists(), "report": True,
            "png": len(list((method_dir / "test_viz").rglob("*.png"))), "source": "summary.json",
        })
    return audit


def main() -> None:
    audit = refresh_internal()
    audit += refresh_external(IRCADB_ROOT, "IRCADb")
    audit += refresh_external(HCC_ROOT, "HCC")
    for row in audit:
        print(
            f"[{row['domain']}] {row['method']} pred={row['pred']} summary={row['summary']}"
            f" report={row['report']} png={row['png']} source={row['source']}"
        )
    print(f"TOTAL refreshed={len(audit)}")


if __name__ == "__main__":
    main()
