"""Shared aggregation rules for liver/tumor evaluation reports.

PMY-LT-v1 is the paper-facing primary standard:

* liver metrics use all fixed-test cases;
* tumor segmentation metrics use GT-positive cases only;
* GT-negative tumor cases are always N/A for tumor segmentation metrics;
* negative-case false positives are reported separately;
* Overall is the arithmetic mean of all-case Liver Dice and positive-only
  Tumor Dice.

The nnUNet ``foreground_mean`` convention is retained as a secondary reference
for traceability only. It excludes true-negative empty tumor cases and counts
false-positive empty tumor cases as zero.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np


STANDARD_ID = "PMY-LT-v1"


def _array_mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(np.mean(values)) if values else float("nan")


def _array_std(values: Iterable[float]) -> float:
    values = list(values)
    return float(np.std(values)) if values else float("nan")


def aggregate_liver_tumor_metrics(
    has_tumor_cases: list[dict[str, Any]],
    no_tumor_cases: list[dict[str, Any]],
    *,
    show_liver: bool = True,
) -> dict[str, Any]:
    """Aggregate one fixed test set under PMY-LT-v1 and nnUNet reference rules."""

    all_cases = has_tumor_cases + no_tumor_cases
    liver_dices = [
        float(row["liver_dice"])
        for row in all_cases
        if row.get("liver_dice") is not None and not np.isnan(row["liver_dice"])
    ]
    false_positive_cases = [row for row in no_tumor_cases if row.get("pred_tumor", 0) > 0]
    n_fp = len(false_positive_cases)
    n_tn = len(no_tumor_cases) - n_fp

    positive_metrics = {
        "dice": [float(row["dice"]) for row in has_tumor_cases],
        "jaccard": [float(row["jaccard"]) for row in has_tumor_cases],
        "recall": [float(row["recall"]) for row in has_tumor_cases],
        "fnr": [float(row["fnr"]) for row in has_tumor_cases],
        "precision": [float(row["precision"]) for row in has_tumor_cases],
        "fdr": [float(row["fdr"]) for row in has_tumor_cases],
    }

    mean_liver = _array_mean(liver_dices)
    primary = {
        name: {"mean": _array_mean(values), "std": _array_std(values)}
        for name, values in positive_metrics.items()
    }
    primary_tumor = primary["dice"]["mean"]
    primary_overall = (
        (mean_liver + primary_tumor) / 2
        if show_liver and not np.isnan(mean_liver) and not np.isnan(primary_tumor)
        else primary_tumor
    )

    # Secondary nnUNet reference: GT-negative FP -> 0, GT-negative TN -> NaN/excluded.
    nnunet_dices = positive_metrics["dice"] + [0.0] * n_fp
    nnunet_jaccards = positive_metrics["jaccard"] + [0.0] * n_fp
    nnunet_tumor = _array_mean(nnunet_dices)
    nnunet_overall = (
        (mean_liver + nnunet_tumor) / 2
        if show_liver and not np.isnan(mean_liver) and not np.isnan(nnunet_tumor)
        else nnunet_tumor
    )

    return {
        "standard_id": STANDARD_ID,
        "n_all": len(all_cases),
        "n_tumor_positive": len(has_tumor_cases),
        "n_tumor_negative": len(no_tumor_cases),
        "n_negative_tn": n_tn,
        "n_negative_fp": n_fp,
        "negative_fp_rate": n_fp / len(no_tumor_cases) if no_tumor_cases else float("nan"),
        "false_positive_cases": false_positive_cases,
        "liver": {"mean": mean_liver, "std": _array_std(liver_dices), "n": len(liver_dices)},
        "primary": {**primary, "overall": primary_overall},
        "nnunet_reference": {
            "tumor_dice": {"mean": nnunet_tumor, "std": _array_std(nnunet_dices)},
            "tumor_jaccard": {"mean": _array_mean(nnunet_jaccards), "std": _array_std(nnunet_jaccards)},
            "overall": nnunet_overall,
            "n_tumor_valid": len(nnunet_dices),
        },
    }
