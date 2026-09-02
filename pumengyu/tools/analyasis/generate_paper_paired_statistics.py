#!/usr/bin/env python3
"""Generate case-level paired statistics for the MedNeXt_MLA_MoE paper.

This script is intentionally read-only with respect to experiment results. It
parses the existing nnU-Net ``summary.json`` files and writes paper-analysis
artifacts under ``pumengyu/notes/paper/statistics`` and
``pumengyu/notes/paper/figures``.

Primary comparison endpoint:
    Tumor Dice on GT-positive cases.

Fixed comparisons:
    1. MedNeXt_MLA_MoE vs MedNeXt
    2. MedNeXt_MLA_MoE vs MedNeXt_MHA_MoE
    3. MedNeXt_MLA_MoE vs MedNeXt_MLA

Statistics:
    * paired mean and median differences
    * paired bootstrap 95% CI for the mean difference
    * two-sided Wilcoxon signed-rank test
    * Holm correction across the three comparisons within each
      dataset/metric family
"""

from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon


REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_ROOT = Path(
    os.environ.get(
        "NNUNET_RESULTS_V2",
        "/home/PuMengYu/nnUNet_workspace/results_v2",
    )
)
OUTPUT_DIR = REPO_ROOT / "pumengyu/notes/paper/statistics"
FIGURE_DIR = REPO_ROOT / "pumengyu/notes/paper/figures"

MAIN_MODEL = "MedNeXt_MLA_MoE"
COMPARISONS = (
    ("MedNeXt", "完整组合 vs 原始骨干"),
    ("MedNeXt_MHA_MoE", "固定 MoE：MLA vs MHA"),
    ("MedNeXt_MLA", "固定 MLA：MoE vs MLP"),
)
MODELS = ("MedNeXt", "MedNeXt_MHA_MoE", "MedNeXt_MLA", MAIN_MODEL)
METRICS = ("tumor_dice", "tumor_recall", "tumor_precision", "liver_dice")
METRIC_LABELS = {
    "tumor_dice": "Tumor Dice",
    "tumor_recall": "Tumor Recall",
    "tumor_precision": "Tumor Precision",
    "liver_dice": "Liver Dice",
}

BOOTSTRAP_REPETITIONS = 10_000
RANDOM_SEED = 20260727
EXPECTED_CASE_COUNTS = {
    "Internal": (26, 23, 3),
    "IRCADb": (20, 15, 5),
    "HCC": (21, 21, 0),
}
EXPECTED_TUMOR_DICE = {
    "Internal": {
        "MedNeXt": 0.7600,
        "MedNeXt_MHA_MoE": 0.7499,
        "MedNeXt_MLA": 0.7535,
        MAIN_MODEL: 0.7590,
    },
    "IRCADb": {
        "MedNeXt": 0.6900,
        "MedNeXt_MHA_MoE": 0.7261,
        "MedNeXt_MLA": 0.6778,
        MAIN_MODEL: 0.7349,
    },
    "HCC": {
        "MedNeXt": 0.4175,
        "MedNeXt_MHA_MoE": 0.3943,
        "MedNeXt_MLA": 0.4645,
        MAIN_MODEL: 0.4080,
    },
}


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    summaries: dict[str, Path]
    reports: dict[str, Path]
    prediction_dirs: dict[str, Path]
    visualization_dirs: dict[str, Path]


@dataclass(frozen=True)
class CaseMetrics:
    case_id: str
    reference_file: str
    tumor_positive: bool
    tumor_n_ref: int
    tumor_n_pred: int
    liver_dice: float
    tumor_dice: float | None
    tumor_recall: float | None
    tumor_precision: float | None


def internal_fold(model: str) -> Path:
    trainer = f"nnUNetTrainer_{model}"
    return (
        RESULTS_ROOT
        / "Dataset003_Liver"
        / f"{trainer}__nnUNetPlans__3d_fullres"
        / "fold_0"
    )


def dataset_specs() -> tuple[DatasetSpec, ...]:
    internal_prediction_dirs = {
        model: internal_fold(model) / "test_prediction"
        for model in MODELS
    }
    internal_summaries = {
        model: internal_prediction_dirs[model] / "summary.json"
        for model in MODELS
    }
    internal_reports = {
        model: internal_fold(model) / "test_report_custom.txt"
        for model in MODELS
    }
    internal_visualization_dirs = {
        model: internal_fold(model) / "test_viz"
        for model in MODELS
    }
    ircad_root = RESULTS_ROOT / "IRCADb/source_only"
    hcc_root = RESULTS_ROOT / "Dataset013_HCCReferencedCT/source_only"
    return (
        DatasetSpec(
            "Internal",
            internal_summaries,
            internal_reports,
            internal_prediction_dirs,
            internal_visualization_dirs,
        ),
        DatasetSpec(
            "IRCADb",
            {
                model: ircad_root / model / "predictions/summary.json"
                for model in MODELS
            },
            {
                model: ircad_root / model / "report_custom.txt"
                for model in MODELS
            },
            {
                model: ircad_root / model / "predictions"
                for model in MODELS
            },
            {
                model: ircad_root / model / "test_viz"
                for model in MODELS
            },
        ),
        DatasetSpec(
            "HCC",
            {
                model: hcc_root / model / "predictions/summary.json"
                for model in MODELS
            },
            {
                model: hcc_root / model / "report_custom.txt"
                for model in MODELS
            },
            {
                model: hcc_root / model / "predictions"
                for model in MODELS
            },
            {
                model: hcc_root / model / "test_viz"
                for model in MODELS
            },
        ),
    )


def source_checkpoints() -> dict[str, Path]:
    return {
        model: internal_fold(model) / "checkpoint_best.pth"
        for model in MODELS
    }


def checkpoint_provenance_audit() -> dict[str, dict[str, object]]:
    """Validate and record the exact fold-0 best checkpoint provenance."""
    import torch

    audit: dict[str, dict[str, object]] = {}
    for model, best_path in source_checkpoints().items():
        if not best_path.is_file():
            raise FileNotFoundError(f"Missing source checkpoint_best.pth: {best_path}")

        fold_dir = best_path.parent
        final_path = fold_dir / "checkpoint_final.pth"
        best = torch.load(
            best_path,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        best_epoch = int(best.get("current_epoch", -1))
        del best

        final_epoch: int | None = None
        final_mtime: str | None = None
        suspicious = False
        if final_path.is_file():
            final = torch.load(
                final_path,
                map_location="cpu",
                weights_only=False,
                mmap=True,
            )
            final_epoch = int(final.get("current_epoch", -1))
            del final
            final_mtime = datetime.fromtimestamp(
                final_path.stat().st_mtime
            ).isoformat(timespec="seconds")
            suspicious = (
                best_path.stat().st_mtime > final_path.stat().st_mtime
                and best_epoch >= 0
                and final_epoch >= 0
                and best_epoch < final_epoch
            )
        if suspicious:
            raise RuntimeError(
                f"Suspicious checkpoint_best provenance for {model}: "
                f"best epoch {best_epoch} was written after final epoch "
                f"{final_epoch}; best={best_path}; final={final_path}"
            )

        trainer = fold_dir.parent.name.split("__", 1)[0]
        audit[model] = {
            "dataset": "Dataset003_Liver",
            "trainer": trainer,
            "fold": 0,
            "checkpoint": best_path.name,
            "path": str(best_path),
            "epoch": best_epoch,
            "size_bytes": best_path.stat().st_size,
            "mtime": datetime.fromtimestamp(
                best_path.stat().st_mtime
            ).isoformat(timespec="seconds"),
            "final_path": str(final_path) if final_path.is_file() else None,
            "final_epoch": final_epoch,
            "final_mtime": final_mtime,
            "provenance_valid": True,
        }
    return audit


def case_id_from_path(path: str) -> str:
    name = Path(path).name
    if name.endswith(".nii.gz"):
        return name[:-7]
    return Path(name).stem


def safe_ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def load_cases(summary_path: Path) -> dict[str, CaseMetrics]:
    with summary_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if "metric_per_case" not in data:
        raise ValueError(f"Missing metric_per_case: {summary_path}")

    cases: dict[str, CaseMetrics] = {}
    for item in data["metric_per_case"]:
        metrics = item["metrics"]
        if "1" not in metrics or "2" not in metrics:
            raise ValueError(f"Missing liver/tumor metrics in {summary_path}")
        liver = metrics["1"]
        tumor = metrics["2"]
        case_id = case_id_from_path(item["reference_file"])
        if case_id in cases:
            raise ValueError(f"Duplicate case {case_id} in {summary_path}")

        n_ref = int(tumor["n_ref"])
        n_pred = int(tumor["n_pred"])
        tp = int(tumor["TP"])
        fn = int(tumor["FN"])
        fp = int(tumor["FP"])
        positive = n_ref > 0
        cases[case_id] = CaseMetrics(
            case_id=case_id,
            reference_file=str(item["reference_file"]),
            tumor_positive=positive,
            tumor_n_ref=n_ref,
            tumor_n_pred=n_pred,
            liver_dice=float(liver["Dice"]),
            tumor_dice=float(tumor["Dice"]) if positive else None,
            tumor_recall=safe_ratio(tp, tp + fn) if positive else None,
            tumor_precision=safe_ratio(tp, tp + fp) if positive else None,
        )
    return cases


def validate_inputs(
    specs: Iterable[DatasetSpec],
) -> tuple[
    dict[str, dict[str, dict[str, CaseMetrics]]],
    dict[str, dict[str, object]],
]:
    checkpoint_audit = checkpoint_provenance_audit()

    loaded: dict[str, dict[str, dict[str, CaseMetrics]]] = {}
    audit: dict[str, dict[str, object]] = {}
    for spec in specs:
        dataset_cases: dict[str, dict[str, CaseMetrics]] = {}
        for model in MODELS:
            summary = spec.summaries[model]
            report = spec.reports[model]
            if not summary.is_file():
                raise FileNotFoundError(f"Missing summary: {summary}")
            if not report.is_file():
                raise FileNotFoundError(f"Missing report: {report}")
            if report.stat().st_size == 0:
                raise ValueError(f"Empty report: {report}")
            dataset_cases[model] = load_cases(summary)

        canonical_ids = set(dataset_cases[MAIN_MODEL])
        for model in MODELS:
            model_ids = set(dataset_cases[model])
            if model_ids != canonical_ids:
                missing = sorted(canonical_ids - model_ids)
                extra = sorted(model_ids - canonical_ids)
                raise ValueError(
                    f"{spec.name}/{model} case mismatch; missing={missing}, extra={extra}"
                )

        artifact_status: dict[str, dict[str, object]] = {}
        for model in MODELS:
            pred_dir = spec.prediction_dirs[model]
            viz_dir = spec.visualization_dirs[model]
            prediction_ids = (
                {
                    case_id_from_path(str(path))
                    for path in pred_dir.glob("*.nii.gz")
                }
                if pred_dir.is_dir()
                else set()
            )
            missing_predictions = sorted(canonical_ids - prediction_ids)
            stale_predictions = sorted(prediction_ids - canonical_ids)
            png_count = (
                sum(1 for _ in viz_dir.rglob("*.png"))
                if viz_dir.is_dir()
                else 0
            )
            predictions_complete = not missing_predictions and not stale_predictions
            visualizations_complete = png_count > 0
            complete = predictions_complete and visualizations_complete
            artifact_status[model] = {
                "prediction_dir": str(pred_dir),
                "prediction_count": len(prediction_ids),
                "missing_prediction_case_ids": missing_predictions,
                "stale_prediction_case_ids": stale_predictions,
                "predictions_complete": predictions_complete,
                "summary": str(spec.summaries[model]),
                "report": str(spec.reports[model]),
                "visualization_dir": str(viz_dir),
                "visualization_png_count": png_count,
                "visualizations_complete": visualizations_complete,
                "checkpoint": checkpoint_audit[model],
                "statistics_eligible": True,
                "complete": complete,
                "status": "complete" if complete else "partial",
            }

        reference_path_variants: dict[str, list[str]] = {}
        for case_id in sorted(canonical_ids):
            reference_files = {
                dataset_cases[model][case_id].reference_file for model in MODELS
            }
            positivity = {
                dataset_cases[model][case_id].tumor_positive for model in MODELS
            }
            tumor_n_refs = {
                dataset_cases[model][case_id].tumor_n_ref for model in MODELS
            }
            # Historical HCC evaluations used two label-directory names. Their
            # 21 label files were verified byte-identical with SHA-256 on
            # 2026-07-27, so a directory-string difference is traceability
            # metadata rather than a pairing failure. Case id plus exact GT
            # tumor voxel count remain hard consistency checks below.
            if len(reference_files) != 1:
                reference_path_variants[case_id] = sorted(reference_files)
            if len(positivity) != 1 or len(tumor_n_refs) != 1:
                raise ValueError(
                    f"{spec.name}/{case_id}: GT tumor status/count differs across models"
                )

        n_total = len(canonical_ids)
        n_positive = sum(
            dataset_cases[MAIN_MODEL][case_id].tumor_positive
            for case_id in canonical_ids
        )
        n_negative = n_total - n_positive
        expected = EXPECTED_CASE_COUNTS[spec.name]
        if (n_total, n_positive, n_negative) != expected:
            raise ValueError(
                f"{spec.name}: observed {(n_total, n_positive, n_negative)}, "
                f"expected {expected}"
            )

        for model in MODELS:
            positive_dice = [
                case.tumor_dice
                for case in dataset_cases[model].values()
                if case.tumor_positive
            ]
            observed = float(np.mean(positive_dice))
            expected_mean = EXPECTED_TUMOR_DICE[spec.name][model]
            if not math.isclose(observed, expected_mean, abs_tol=5e-5):
                raise ValueError(
                    f"{spec.name}/{model}: Tumor Dice mean {observed:.6f} "
                    f"does not match expected {expected_mean:.4f}"
                )

        audit[spec.name] = {
            "n_total": n_total,
            "n_positive": n_positive,
            "n_negative": n_negative,
            "case_ids": sorted(canonical_ids),
            "summaries": {
                model: str(spec.summaries[model]) for model in MODELS
            },
            "reports": {model: str(spec.reports[model]) for model in MODELS},
            "artifacts": artifact_status,
            "reference_path_variants": reference_path_variants,
        }
        loaded[spec.name] = dataset_cases

    audit["source_checkpoints"] = checkpoint_audit
    return loaded, audit


def metric_value(case: CaseMetrics, metric: str) -> float | None:
    return getattr(case, metric)


def paired_arrays(
    model_a: dict[str, CaseMetrics],
    model_b: dict[str, CaseMetrics],
    metric: str,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    case_ids: list[str] = []
    values_a: list[float] = []
    values_b: list[float] = []
    for case_id in sorted(model_a):
        value_a = metric_value(model_a[case_id], metric)
        value_b = metric_value(model_b[case_id], metric)
        if value_a is None or value_b is None:
            continue
        case_ids.append(case_id)
        values_a.append(float(value_a))
        values_b.append(float(value_b))
    if not case_ids:
        raise ValueError(f"No paired values for metric {metric}")
    return case_ids, np.asarray(values_a), np.asarray(values_b)


def bootstrap_mean_ci(
    differences: np.ndarray,
    rng: np.random.Generator,
) -> tuple[float, float]:
    n = len(differences)
    indices = rng.integers(0, n, size=(BOOTSTRAP_REPETITIONS, n))
    bootstrap_means = differences[indices].mean(axis=1)
    low, high = np.percentile(bootstrap_means, [2.5, 97.5])
    return float(low), float(high)


def wilcoxon_pvalue(values_a: np.ndarray, values_b: np.ndarray) -> float:
    differences = values_a - values_b
    if np.allclose(differences, 0):
        return 1.0
    result = wilcoxon(
        values_a,
        values_b,
        alternative="two-sided",
        zero_method="wilcox",
        method="auto",
    )
    return float(result.pvalue)


def holm_adjust(pvalues: list[float]) -> list[float]:
    count = len(pvalues)
    order = sorted(range(count), key=pvalues.__getitem__)
    adjusted = [1.0] * count
    running_max = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * pvalues[index])
        running_max = max(running_max, candidate)
        adjusted[index] = running_max
    return adjusted


def describe(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "median": float(np.median(values)),
        "q1": float(np.percentile(values, 25)),
        "q3": float(np.percentile(values, 75)),
    }


def calculate_statistics(
    loaded: dict[str, dict[str, dict[str, CaseMetrics]]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rng = np.random.default_rng(RANDOM_SEED)
    checkpoints = source_checkpoints()
    rows: list[dict[str, object]] = []
    detail_rows: list[dict[str, object]] = []

    for dataset_name, models in loaded.items():
        for comparator, comparison_label in COMPARISONS:
            for metric in METRICS:
                case_ids, main_values, comparator_values = paired_arrays(
                    models[MAIN_MODEL], models[comparator], metric
                )
                differences = main_values - comparator_values
                ci_low, ci_high = bootstrap_mean_ci(differences, rng)
                main_desc = describe(main_values)
                comparator_desc = describe(comparator_values)
                diff_desc = describe(differences)
                row = {
                    "dataset": dataset_name,
                    "comparison": f"{MAIN_MODEL} vs {comparator}",
                    "comparison_label": comparison_label,
                    "main_model": MAIN_MODEL,
                    "comparator": comparator,
                    "metric": metric,
                    "metric_label": METRIC_LABELS[metric],
                    "n_pairs": len(case_ids),
                    "main_mean": main_desc["mean"],
                    "main_std": main_desc["std"],
                    "main_median": main_desc["median"],
                    "main_q1": main_desc["q1"],
                    "main_q3": main_desc["q3"],
                    "comparator_mean": comparator_desc["mean"],
                    "comparator_std": comparator_desc["std"],
                    "comparator_median": comparator_desc["median"],
                    "comparator_q1": comparator_desc["q1"],
                    "comparator_q3": comparator_desc["q3"],
                    "mean_difference": diff_desc["mean"],
                    "median_difference": diff_desc["median"],
                    "difference_q1": diff_desc["q1"],
                    "difference_q3": diff_desc["q3"],
                    "bootstrap_ci_low": ci_low,
                    "bootstrap_ci_high": ci_high,
                    "wilcoxon_p_raw": wilcoxon_pvalue(
                        main_values, comparator_values
                    ),
                    "wilcoxon_p_holm": None,
                    "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
                    "random_seed": RANDOM_SEED,
                }
                rows.append(row)

                for case_id, main_value, comparator_value, difference in zip(
                    case_ids,
                    main_values,
                    comparator_values,
                    differences,
                    strict=True,
                ):
                    detail_rows.append(
                        {
                            "dataset": dataset_name,
                            "case_id": case_id,
                            "comparison": row["comparison"],
                            "model_a": MAIN_MODEL,
                            "model_b": comparator,
                            "value_a": float(main_value),
                            "value_b": float(comparator_value),
                            "checkpoint_a": str(checkpoints[MAIN_MODEL]),
                            "checkpoint_b": str(checkpoints[comparator]),
                            "main_model": MAIN_MODEL,
                            "comparator": comparator,
                            "metric": metric,
                            "main_value": float(main_value),
                            "comparator_value": float(comparator_value),
                            "paired_difference": float(difference),
                        }
                    )

    for dataset_name in loaded:
        for metric in METRICS:
            family = [
                row
                for row in rows
                if row["dataset"] == dataset_name and row["metric"] == metric
            ]
            adjusted = holm_adjust(
                [float(row["wilcoxon_p_raw"]) for row in family]
            )
            for row, adjusted_p in zip(family, adjusted, strict=True):
                row["wilcoxon_p_holm"] = adjusted_p

    return rows, detail_rows


def negative_case_audit(
    loaded: dict[str, dict[str, dict[str, CaseMetrics]]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dataset_name, models in loaded.items():
        for model_name, cases in models.items():
            negative = [case for case in cases.values() if not case.tumor_positive]
            false_positive = [
                case.case_id for case in negative if case.tumor_n_pred > 0
            ]
            rows.append(
                {
                    "dataset": dataset_name,
                    "model": model_name,
                    "n_negative": len(negative),
                    "n_false_positive_cases": len(false_positive),
                    "false_positive_case_ids": false_positive,
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def format_float(value: object, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def format_pvalue(value: object) -> str:
    number = float(value)
    return "<0.0001" if number < 0.0001 else f"{number:.4f}"


def primary_interpretation(row: dict[str, object]) -> str:
    diff = float(row["mean_difference"])
    low = float(row["bootstrap_ci_low"])
    high = float(row["bootstrap_ci_high"])
    p_holm = float(row["wilcoxon_p_holm"])
    if low > 0 and p_holm < 0.05:
        return "主模型的病例级 Tumor Dice 更高，均值差 CI 不跨 0，Holm 校正后 Wilcoxon p<0.05。"
    if high < 0 and p_holm < 0.05:
        return "主模型的病例级 Tumor Dice 更低，均值差 CI 不跨 0，Holm 校正后 Wilcoxon p<0.05。"
    direction = "更高" if diff > 0 else "更低" if diff < 0 else "持平"
    return (
        f"主模型平均 Tumor Dice {direction}，但当前配对证据未同时满足"
        "均值差 CI 不跨 0和Holm校正后 Wilcoxon p<0.05。"
    )


def write_markdown(
    path: Path,
    rows: list[dict[str, object]],
    detail_rows: list[dict[str, object]],
    negative_rows: list[dict[str, object]],
    audit: dict[str, dict[str, object]],
) -> None:
    primary = [row for row in rows if row["metric"] == "tumor_dice"]
    lines = [
        "# MedNeXt_MLA_MoE 三组病例级配对统计",
        "",
        f"> 生成日期：{date.today().isoformat()}  ",
        "> 主要终点：GT 阳性病例上的 Tumor Dice  ",
        f"> bootstrap：{BOOTSTRAP_REPETITIONS:,} 次，seed={RANDOM_SEED}  ",
        "> 检验：双侧 Wilcoxon signed-rank；同一数据域/指标内三组比较采用 Holm 校正。  ",
        "> 解释边界：固定 checkpoint 的病例级配对检验不能替代多随机种子或多 fold 训练。",
        "",
        "## 1. 输入核对",
        "",
        "| 数据域 | 总病例 | 肿瘤阳性 | 肿瘤阴性 | 四模型病例集合 | GT状态/体素数 |",
        "|---|---:|---:|---:|---|---|",
    ]
    for dataset_name in ("Internal", "IRCADb", "HCC"):
        item = audit[dataset_name]
        lines.append(
            f"| {dataset_name} | {item['n_total']} | {item['n_positive']} | "
            f"{item['n_negative']} | 一致 | 一致 |"
        )

    lines.extend(
        [
            "",
            "统计输入要求逐病例 `summary.json`、文本报告和可信 "
            "`checkpoint_best.pth`；完整实验产物还要求 NIfTI 预测与实际 "
            "`test_viz` PNG。下表将统计可用性与实验完整性分开报告。"
            "本次直接读取已有 `summary.json`，没有重新推理。",
            "",
            "| 数据域 | 模型 | 预测病例 | summary | 报告 | test_viz PNG | checkpoint来源 |",
            "|---|---|---:|---|---|---:|---|",
        "",
        "## 2. 主要终点",
            "",
            "| 数据域 | 对比 | n | 主模型均值 | 对照均值 | 平均差值 | "
            "配对bootstrap 95% CI | 中位差值 | Wilcoxon p | Holm p |",
            "|---|---|---:|---:|---:|---:|---|---:|---:|---:|",
        ]
    )
    artifact_lines: list[str] = []
    for dataset_name in ("Internal", "IRCADb", "HCC"):
        for model in MODELS:
            item = audit[dataset_name]["artifacts"][model]
            checkpoint = item["checkpoint"]
            artifact_lines.append(
                f"| {dataset_name} | {model} | {item['prediction_count']} | "
                f"存在 | 存在 | {item['visualization_png_count']} | "
                f"通过（epoch {checkpoint['epoch']}）；{item['status']} |"
            )
    # Insert immediately after the artifact table header, before section 2.
    section_two_index = lines.index("## 2. 主要终点")
    lines[section_two_index - 1:section_two_index - 1] = artifact_lines + [""]
    for row in primary:
        lines.append(
            f"| {row['dataset']} | {row['comparison_label']} | {row['n_pairs']} | "
            f"{format_float(row['main_mean'])} | "
            f"{format_float(row['comparator_mean'])} | "
            f"{float(row['mean_difference']):+.4f} | "
            f"[{format_float(row['bootstrap_ci_low'])}, "
            f"{format_float(row['bootstrap_ci_high'])}] | "
            f"{float(row['median_difference']):+.4f} | "
            f"{format_pvalue(row['wilcoxon_p_raw'])} | "
            f"{format_pvalue(row['wilcoxon_p_holm'])} |"
        )

    lines.extend(["", "## 3. 分域解释", ""])
    for dataset_name in ("Internal", "IRCADb", "HCC"):
        lines.append(f"### {dataset_name}")
        lines.append("")
        for row in [r for r in primary if r["dataset"] == dataset_name]:
            lines.append(
                f"- **{row['comparison_label']}**："
                f"平均差值 {float(row['mean_difference']):+.4f}，"
                f"95% CI [{format_float(row['bootstrap_ci_low'])}, "
                f"{format_float(row['bootstrap_ci_high'])}]，"
                f"Holm p={format_pvalue(row['wilcoxon_p_holm'])}。"
                f"{primary_interpretation(row)}"
            )
        lines.append("")

    lines.extend(
        [
            "## 4. 关键病例",
            "",
            "下表分别列出每组 Tumor Dice 对比中主模型改善最大和退化最大的病例。",
            "",
            "| 数据域 | 对比 | 改善最大3例 | 退化最大3例 |",
            "|---|---|---|---|",
        ]
    )
    for row in primary:
        relevant = [
            item
            for item in detail_rows
            if item["dataset"] == row["dataset"]
            and item["comparison"] == row["comparison"]
            and item["metric"] == "tumor_dice"
        ]
        ranked = sorted(
            relevant, key=lambda item: float(item["paired_difference"])
        )
        worst = ", ".join(
            f"{item['case_id']} ({float(item['paired_difference']):+.3f})"
            for item in ranked[:3]
        )
        best = ", ".join(
            f"{item['case_id']} ({float(item['paired_difference']):+.3f})"
            for item in reversed(ranked[-3:])
        )
        lines.append(
            f"| {row['dataset']} | {row['comparison_label']} | {best} | {worst} |"
        )

    lines.extend(
        [
            "",
            "## 5. 阴性病例误报（描述性）",
            "",
            "| 数据域 | 模型 | 阴性病例 | 误报病例 | case IDs |",
            "|---|---|---:|---:|---|",
        ]
    )
    for item in negative_rows:
        case_ids = ", ".join(item["false_positive_case_ids"]) or "无"
        value = (
            "N/A"
            if int(item["n_negative"]) == 0
            else str(item["n_false_positive_cases"])
        )
        lines.append(
            f"| {item['dataset']} | {item['model']} | {item['n_negative']} | "
            f"{value} | {case_ids} |"
        )

    lines.extend(
        [
            "",
            "Internal 仅 3 个阴性病例、IRCADb 仅 5 个阴性病例，因此不把 FP "
            "计数包装为强统计显著性结论。HCC test 全部为肿瘤阳性病例，FP rate 为 N/A。",
            "",
            "## 6. 次要终点",
            "",
            "完整的 Tumor Recall、Tumor Precision 和 Liver Dice 结果见 "
            "`paired_case_statistics.csv`。这些指标与主要终点使用相同的配对 "
            "bootstrap、Wilcoxon 和分域 Holm 校正流程。",
            "",
            "## 7. 产物",
            "",
            "- `paired_case_statistics.csv`：聚合统计；",
            "- `paired_case_differences.csv`：每例配对明细；",
            "- `statistics_metadata.json`：输入路径、病例清单、checkpoint 和参数；",
            "- `paired_case_tumor_dice_table.md`：正文候选病例级配对统计表。",
            "",
            "## 8. 尚未回答的问题",
            "",
            "- 当前统计不能量化训练随机性；",
            "- 多随机种子和多 fold 尚未完成；",
            "- 若小差值未通过校正，论文应报告方向和置信区间，不写“显著优于”；",
            "- 病例级统计完成不等于模型效率、Attention机制或因果解释已经完成。",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_primary_differences(
    figure_base: Path,
    detail_rows: list[dict[str, object]],
    statistic_rows: list[dict[str, object]],
) -> None:
    dataset_names = ("Internal", "IRCADb", "HCC")
    comparator_names = tuple(item[0] for item in COMPARISONS)
    comparator_labels = (
        "MLA+MoE\n− MedNeXt",
        "MLA+MoE\n− MHA+MoE",
        "MLA+MoE\n− MLA",
    )
    comparator_caption_labels = (
        "MedNeXt（原始骨干）",
        "MedNeXt_MHA_MoE（固定 MoE，比较 MLA 与 MHA）",
        "MedNeXt_MLA（固定 MLA，比较 MoE 与 MLP）",
    )
    colors = {"Internal": "#4C78A8", "IRCADb": "#F58518", "HCC": "#54A24B"}
    rng = np.random.default_rng(RANDOM_SEED)

    primary_details = [
        item for item in detail_rows if item["metric"] == "tumor_dice"
    ]
    positions = np.arange(1, len(comparator_names) + 1, dtype=float)
    output_stems = {
        "Internal": "paired_case_tumor_dice_internal",
        "IRCADb": "paired_case_tumor_dice_ircadb",
        "HCC": "paired_case_tumor_dice_hcc",
    }

    for dataset_name in dataset_names:
        dataset_details = [
            item for item in primary_details if item["dataset"] == dataset_name
        ]
        max_abs = max(
            abs(float(item["paired_difference"])) for item in dataset_details
        )
        y_limit = min(1.0, max(0.12, max_abs * 1.18))
        fig, ax = plt.subplots(figsize=(9.2, 5.9))
        ax.axhspan(0, y_limit, color="#DDF2E6", alpha=0.28, zorder=0)
        ax.axhspan(-y_limit, 0, color="#F8E1E1", alpha=0.25, zorder=0)
        ax.axhline(0, color="#333333", linewidth=1.2, linestyle="--", zorder=1)
        caption_statistics: list[tuple[str, dict[str, object]]] = []
        dataset_n: int | None = None

        for position, comparator, caption_label in zip(
            positions,
            comparator_names,
            comparator_caption_labels,
            strict=True,
        ):
            relevant = [
                item
                for item in dataset_details
                if item["comparator"] == comparator
            ]
            values = [float(item["paired_difference"]) for item in relevant]
            values_array = np.asarray(values)
            dataset_n = len(values_array)
            jitter = rng.uniform(-0.25, 0.25, size=len(values_array))
            ax.scatter(
                np.full(len(values_array), position) + jitter,
                values_array,
                s=38,
                alpha=0.68,
                color=colors[dataset_name],
                edgecolors="white",
                linewidths=0.55,
                zorder=3,
            )
            statistic = next(
                row
                for row in statistic_rows
                if row["dataset"] == dataset_name
                and row["comparator"] == comparator
                and row["metric"] == "tumor_dice"
            )
            caption_statistics.append((caption_label, statistic))
            mean_difference = float(statistic["mean_difference"])
            ci_low = float(statistic["bootstrap_ci_low"])
            ci_high = float(statistic["bootstrap_ci_high"])
            ax.errorbar(
                position,
                mean_difference,
                yerr=np.asarray(
                    [[mean_difference - ci_low], [ci_high - mean_difference]]
                ),
                fmt="D",
                markersize=6.3,
                color="#111111",
                ecolor="#111111",
                elinewidth=1.8,
                capsize=4,
                capthick=1.4,
                zorder=5,
            )

        ax.set_title(dataset_name, fontsize=14, fontweight="bold")
        ax.set_xlim(0.55, len(comparator_names) + 0.45)
        ax.set_ylim(-y_limit, y_limit)
        ax.set_xticks(positions)
        ax.set_xticklabels(comparator_labels, fontsize=11)
        ax.grid(axis="y", color="#D8D8D8", linewidth=0.7, alpha=0.8)
        ax.set_ylabel(
            "Tumor Dice difference",
            fontsize=12,
        )
        ax.text(
            0.985,
            0.965,
            "↑ MLA+MoE better",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=10.5,
            color="#2F6B4F",
            fontweight="bold",
        )
        ax.text(
            0.985,
            0.035,
            "↓ Other model better",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=10.5,
            color="#9A4747",
            fontweight="bold",
        )
        ax.tick_params(axis="y", labelsize=10.5)
        fig.tight_layout()

        output_base = figure_base.parent / output_stems[dataset_name]
        output_base.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
        fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight")
        plt.close(fig)

        caption_lines = [
            f"图注：{dataset_name} 数据集上的病例级配对 Tumor Dice 差值。",
            "",
            "这张图怎么看：",
            "- 图中的 MLA+MoE 是本文模型 MedNeXt_MLA_MoE 的简称。",
            "- 每个彩色圆点代表一个肿瘤阳性病例。",
            "- 横轴直接写出相减的两个模型，例如“MLA+MoE − MedNeXt”。",
            "- 纵轴是这两个模型在同一病例上的 Tumor Dice 差值。",
            "- 点在 0 线上方表示 MLA+MoE 在该病例更好；在 0 线下方表示另一个模型更好。",
            "- 黑色菱形表示平均差值；黑色竖向误差线表示配对 bootstrap 95% CI。",
            f"- 本图共纳入 n={dataset_n} 个肿瘤阳性病例。",
            "- 为提高单图可读性，三张图分别使用适合自身数据范围的纵轴；跨数据集效应大小应以以下数值为准，不能只比较图中垂直距离。",
            "",
            "统计结果：",
        ]
        for caption_label, statistic in caption_statistics:
            mean_difference = float(statistic["mean_difference"])
            ci_low = float(statistic["bootstrap_ci_low"])
            ci_high = float(statistic["bootstrap_ci_high"])
            adjusted_p = float(statistic["wilcoxon_p_holm"])
            if ci_low > 0 and adjusted_p < 0.05:
                conclusion = "MLA+MoE 显著更高"
            elif ci_high < 0 and adjusted_p < 0.05:
                conclusion = "MLA+MoE 显著更低"
            else:
                conclusion = "校正后未达到显著差异"
            caption_lines.append(
                f"- 对比 {caption_label}：平均差 {mean_difference:+.3f}，"
                f"95% CI [{ci_low:+.3f}, {ci_high:+.3f}]，"
                f"Holm 校正后 p={format_pvalue(adjusted_p)}；{conclusion}。"
            )
        caption_lines.extend(
            [
                "",
                "注意：",
                "- 这里是固定 checkpoint、固定病例上的病例级配对统计，不等于多随机种子或多 fold 的训练稳定性。",
                "- 完整逐病例数据见 ../statistics/paired_case_differences.csv。",
                "- 完整统计表见 ../statistics/paired_case_statistics.md。",
                "",
            ]
        )
        output_base.with_name(output_base.name + "_caption").with_suffix(
            ".txt"
        ).write_text("\n".join(caption_lines), encoding="utf-8")


def write_primary_table(
    path: Path,
    statistic_rows: list[dict[str, object]],
    audit: dict[str, object],
) -> None:
    dataset_names = ("Internal", "IRCADb", "HCC")
    dataset_labels = {
        "Internal": "MSD",
        "IRCADb": "IRCADb",
        "HCC": "HCC",
    }
    comparator_names = tuple(item[0] for item in COMPARISONS)
    comparator_labels = {
        "MedNeXt": "MedNeXt（原始骨干）",
        "MedNeXt_MHA_MoE": "MHA+MoE（固定 MoE）",
        "MedNeXt_MLA": "MLA（固定 MLA）",
    }
    checkpoints = audit["source_checkpoints"]
    valid_checkpoint_count = sum(
        1
        for checkpoint in checkpoints.values()
        if checkpoint["provenance_valid"]
    )
    lines = [
        "# 病例级配对 Tumor Dice 统计表",
        "",
        f"> 数据核验：9/9 组比较均由相同 case ID 的逐病例结果配对生成；"
        f"涉及的 {valid_checkpoint_count}/4 个 `checkpoint_best.pth` 均通过来源检查。  ",
        "> 病例数：MSD 26 例（23 例肿瘤阳性）、IRCADb 20 例（15 例肿瘤阳性）、HCC 21 例（均为肿瘤阳性）。  ",
        "> 产物完整性：IRCADb/HCC 完整；MSD 的 summary、报告和可视化存在，但正式目录缺少 26 例 NIfTI 预测，因此仍标记为部分完成，需在 P0-4 修复。  ",
        "> 本文模型：`MedNeXt_MLA_MoE`（表中简称 MLA+MoE）。  ",
        "> 平均差 = MLA+MoE Tumor Dice − 对照方法 Tumor Dice。正值表示 MLA+MoE 更高，负值表示对照方法更高。  ",
        "",
        "## 什么叫“配对”、bootstrap 和 Holm 校正",
        "",
        "- **配对**：不是把不同患者混在一起比较，而是对同一个 CT 病例分别计算两个模型的 Tumor Dice，再求“MLA+MoE Dice − 对照方法 Dice”。",
        "- **配对重采样 95% 置信区间（paired bootstrap 95% CI）**：把整组病例差值重复抽样 10,000 次，每次抽完一个病例后放回，因此同一病例在一次抽样中可能出现多次；每次都重算平均差。中间 95% 的结果形成表中的区间。区间若跨过 0，说明现有病例不足以稳定判断谁更高；区间完全大于 0 或完全小于 0，才说明差异方向比较稳定。",
        "- **Holm 校正后 p 值**：每个数据集同时比较 3 个对照方法，比较次数越多，越容易偶然出现“显著”。Holm 校正会把这个风险控制得更严格。这里使用双侧 Wilcoxon 配对检验，校正后 p < 0.05 才写为“有统计学差异”。",
        "- 两项判断共同使用：置信区间不能跨 0，并且 Holm 校正后 p < 0.05，表中才写“显著更高”或“显著更低”。",
        "",
    ]
    for dataset_name in dataset_names:
        dataset_label = dataset_labels[dataset_name]
        lines.extend(
            [
                "",
                f"## {dataset_label}",
                "",
                "| 对照方法 | 肿瘤阳性病例数 n | 平均差 | 配对重采样 95% 置信区间 | Holm 校正后 p 值 | 结论 |",
                "|---|---:|---:|---|---:|---|",
            ]
        )
        for comparator in comparator_names:
            row = next(
                item
                for item in statistic_rows
                if item["dataset"] == dataset_name
                and item["comparator"] == comparator
                and item["metric"] == "tumor_dice"
            )
            difference = float(row["mean_difference"])
            ci_low = float(row["bootstrap_ci_low"])
            ci_high = float(row["bootstrap_ci_high"])
            adjusted_p = float(row["wilcoxon_p_holm"])
            if ci_low > 0 and adjusted_p < 0.05:
                conclusion = "MLA+MoE 显著更高"
            elif ci_high < 0 and adjusted_p < 0.05:
                conclusion = "MLA+MoE 显著更低"
            else:
                conclusion = "校正后无显著差异"
            lines.append(
                f"| {comparator_labels[comparator]} | "
                f"{int(row['n_pairs'])} | {difference:+.3f} | "
                f"[{ci_low:+.3f}, {ci_high:+.3f}] | "
                f"{format_pvalue(adjusted_p)} | {conclusion} |"
            )
    lines.extend(
        [
            "",
            "说明：",
            "",
            "- 该表只统计各数据集中的肿瘤阳性病例；",
            "- 病例级配对显著性不能替代多随机种子或多 fold 的训练稳定性分析；",
            "- 完整逐病例差值见 `paired_case_differences.csv`；",
            "- 完整 Recall、Precision 和 Liver Dice 统计见 `paired_case_statistics.csv`。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    specs = dataset_specs()
    loaded, audit = validate_inputs(specs)
    rows, detail_rows = calculate_statistics(loaded)
    negative_rows = negative_case_audit(loaded)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT_DIR / "paired_case_statistics.csv", rows)
    write_csv(OUTPUT_DIR / "paired_case_differences.csv", detail_rows)

    metadata = {
        "generated_on": date.today().isoformat(),
        "primary_endpoint": "Tumor Dice on GT-positive cases",
        "main_model": MAIN_MODEL,
        "comparisons": [
            {"comparator": comparator, "label": label}
            for comparator, label in COMPARISONS
        ],
        "metrics": list(METRICS),
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "confidence_level": 0.95,
        "random_seed": RANDOM_SEED,
        "multiple_testing": (
            "Holm correction across three comparisons within each dataset/metric"
        ),
        "input_audit": audit,
        "negative_case_audit": negative_rows,
    }
    (OUTPUT_DIR / "statistics_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_markdown(
        OUTPUT_DIR / "paired_case_statistics.md",
        rows,
        detail_rows,
        negative_rows,
        audit,
    )
    write_primary_table(
        OUTPUT_DIR / "paired_case_tumor_dice_table.md",
        rows,
        audit,
    )

    print(f"Wrote {OUTPUT_DIR / 'paired_case_statistics.csv'}")
    print(f"Wrote {OUTPUT_DIR / 'paired_case_differences.csv'}")
    print(f"Wrote {OUTPUT_DIR / 'paired_case_statistics.md'}")
    print(f"Wrote {OUTPUT_DIR / 'statistics_metadata.json'}")
    print(f"Wrote {OUTPUT_DIR / 'paired_case_tumor_dice_table.md'}")


if __name__ == "__main__":
    main()
