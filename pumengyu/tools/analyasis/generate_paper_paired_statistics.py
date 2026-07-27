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
    internal_summaries = {
        model: internal_fold(model) / "test_prediction/summary.json"
        for model in MODELS
    }
    internal_reports = {
        model: internal_fold(model) / "test_report_custom.txt"
        for model in MODELS
    }
    ircad_root = RESULTS_ROOT / "IRCADb/source_only"
    hcc_root = RESULTS_ROOT / "Dataset013_HCCReferencedCT/source_only"
    return (
        DatasetSpec("Internal", internal_summaries, internal_reports),
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
        ),
    )


def source_checkpoints() -> dict[str, Path]:
    return {
        model: internal_fold(model) / "checkpoint_best.pth"
        for model in MODELS
    }


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
    checkpoints = source_checkpoints()
    missing_checkpoints = [str(path) for path in checkpoints.values() if not path.is_file()]
    if missing_checkpoints:
        raise FileNotFoundError(
            "Missing source checkpoint_best.pth:\n" + "\n".join(missing_checkpoints)
        )

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
            "reference_path_variants": reference_path_variants,
        }
        loaded[spec.name] = dataset_cases

    audit["source_checkpoints"] = {
        model: {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for model, path in checkpoints.items()
    }
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
        "> 生成日期：2026-07-27  ",
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
            "四个 source model 的 `checkpoint_best.pth` 均存在；统计直接读取已有 "
            "`summary.json`，没有重新推理。",
            "",
            "## 2. 主要终点",
            "",
            "| 数据域 | 对比 | n | 主模型均值 | 对照均值 | 平均差值 | "
            "配对bootstrap 95% CI | 中位差值 | Wilcoxon p | Holm p |",
            "|---|---|---:|---:|---:|---:|---|---:|---:|---:|",
        ]
    )
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
            "- `../figures/paired_case_tumor_dice_differences.png`：配对差值图；",
            "- `../figures/paired_case_tumor_dice_differences.svg`：矢量图。",
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
) -> None:
    dataset_names = ("Internal", "IRCADb", "HCC")
    comparator_names = tuple(item[0] for item in COMPARISONS)
    column_titles = tuple(item[1] for item in COMPARISONS)
    colors = {"Internal": "#4C78A8", "IRCADb": "#F58518", "HCC": "#54A24B"}
    rng = np.random.default_rng(RANDOM_SEED)

    fig, axes = plt.subplots(3, 3, figsize=(13.2, 10.2), sharey=True)
    for row_index, dataset_name in enumerate(dataset_names):
        for column_index, (comparator, title) in enumerate(
            zip(comparator_names, column_titles, strict=True)
        ):
            ax = axes[row_index, column_index]
            values = [
                float(item["paired_difference"])
                for item in detail_rows
                if item["dataset"] == dataset_name
                and item["comparator"] == comparator
                and item["metric"] == "tumor_dice"
            ]
            values_array = np.asarray(values)
            jitter = rng.uniform(-0.10, 0.10, size=len(values_array))
            ax.axhline(0, color="#555555", linewidth=1, linestyle="--", zorder=1)
            ax.scatter(
                np.ones(len(values_array)) + jitter,
                values_array,
                s=25,
                alpha=0.80,
                color=colors[dataset_name],
                edgecolors="white",
                linewidths=0.35,
                zorder=3,
            )
            ax.boxplot(
                values_array,
                positions=[1],
                widths=0.28,
                patch_artist=True,
                showfliers=False,
                boxprops={
                    "facecolor": colors[dataset_name],
                    "alpha": 0.22,
                    "edgecolor": colors[dataset_name],
                },
                medianprops={"color": "#111111", "linewidth": 1.5},
                whiskerprops={"color": colors[dataset_name]},
                capprops={"color": colors[dataset_name]},
            )
            ax.set_xlim(0.55, 1.45)
            ax.set_xticks([])
            ax.grid(axis="y", color="#DDDDDD", linewidth=0.6, alpha=0.8)
            if row_index == 0:
                ax.set_title(title, fontsize=10.5)
            if column_index == 0:
                ax.set_ylabel(
                    f"{dataset_name}\nΔ Tumor Dice\n(main - comparator)",
                    fontsize=9.5,
                )
            ax.text(
                0.97,
                0.04,
                f"n={len(values_array)}\nmean={np.mean(values_array):+.3f}",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=8.5,
            )

    fig.suptitle(
        "Case-level paired Tumor Dice differences for MedNeXt_MLA_MoE",
        fontsize=13,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    figure_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(figure_base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


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
        "generated_on": "2026-07-27",
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
    plot_primary_differences(
        FIGURE_DIR / "paired_case_tumor_dice_differences",
        detail_rows,
    )

    print(f"Wrote {OUTPUT_DIR / 'paired_case_statistics.csv'}")
    print(f"Wrote {OUTPUT_DIR / 'paired_case_differences.csv'}")
    print(f"Wrote {OUTPUT_DIR / 'paired_case_statistics.md'}")
    print(f"Wrote {OUTPUT_DIR / 'statistics_metadata.json'}")
    print(
        "Wrote "
        f"{FIGURE_DIR / 'paired_case_tumor_dice_differences.png'}"
    )
    print(
        "Wrote "
        f"{FIGURE_DIR / 'paired_case_tumor_dice_differences.svg'}"
    )


if __name__ == "__main__":
    main()
