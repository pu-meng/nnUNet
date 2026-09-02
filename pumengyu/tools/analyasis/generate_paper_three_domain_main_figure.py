#!/usr/bin/env python3
"""Generate the paper's three-domain main-results figure.

The script reads existing per-case ``summary.json`` files only. It does not run
inference or alter experiment outputs. Besides PNG/SVG figures, it writes the
exact plotted values and an artifact/provenance audit for the six core methods.
"""

from __future__ import annotations

import csv
import json
import os
from datetime import date, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pumengyu.tools.analyasis.generate_paper_paired_statistics import (
    CaseMetrics,
    case_id_from_path,
    internal_fold,
    load_cases,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_ROOT = Path(
    os.environ.get(
        "NNUNET_RESULTS_V2",
        "/home/PuMengYu/nnUNet_workspace/results_v2",
    )
)
OUTPUT_DIR = REPO_ROOT / "pumengyu/notes/paper/statistics"
FIGURE_DIR = REPO_ROOT / "pumengyu/notes/paper/figures"

METHODS = (
    "MedNeXt",
    "MedNeXt_MHA",
    "MedNeXt_MLA",
    "MedNeXt_MHA_MoE",
    "MedNeXt_MLA_MoE",
    "MedNeXt_MLA_MoE_SizeOV4",
)
DISPLAY_NAMES = {
    "MedNeXt": "MedNeXt",
    "MedNeXt_MHA": "MHA",
    "MedNeXt_MLA": "MLA",
    "MedNeXt_MHA_MoE": "MHA+MoE",
    "MedNeXt_MLA_MoE": "MLA+MoE",
    "MedNeXt_MLA_MoE_SizeOV4": "MLA+MoE\n+SizeOV4",
}
MAIN_MODEL = "MedNeXt_MLA_MoE"
DATASETS = ("Internal", "IRCADb", "HCC")
DATASET_DISPLAY_NAMES = {
    "Internal": "MSD",
    "IRCADb": "IRCADb",
    "HCC": "HCC",
}
EXPECTED_CASE_COUNTS = {
    "Internal": (26, 23, 3),
    "IRCADb": (20, 15, 5),
    "HCC": (21, 21, 0),
}


def dataset_paths(
    dataset: str,
    method: str,
) -> tuple[Path, Path, Path, Path]:
    if dataset == "Internal":
        fold_dir = internal_fold(method)
        prediction_dir = fold_dir / "test_prediction"
        return (
            prediction_dir / "summary.json",
            fold_dir / "test_report_custom.txt",
            prediction_dir,
            fold_dir / "test_viz",
        )
    if dataset == "IRCADb":
        method_dir = RESULTS_ROOT / "IRCADb/source_only" / method
    elif dataset == "HCC":
        method_dir = (
            RESULTS_ROOT
            / "Dataset013_HCCReferencedCT/source_only"
            / method
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    prediction_dir = method_dir / "predictions"
    return (
        prediction_dir / "summary.json",
        method_dir / "report_custom.txt",
        prediction_dir,
        method_dir / "test_viz",
    )


def checkpoint_info(method: str) -> dict[str, object]:
    import torch

    best_path = internal_fold(method) / "checkpoint_best.pth"
    final_path = internal_fold(method) / "checkpoint_final.pth"
    if not best_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {best_path}")

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
                f"Suspicious checkpoint_best provenance for {method}: "
                f"best epoch {best_epoch} was written after final epoch "
                f"{final_epoch}"
            )

    return {
        "dataset": "Dataset003_Liver",
        "trainer": internal_fold(method).parent.name.split("__", 1)[0],
        "fold": 0,
        "checkpoint": best_path.name,
        "path": str(best_path),
        "epoch": best_epoch,
        "mtime": datetime.fromtimestamp(
            best_path.stat().st_mtime
        ).isoformat(timespec="seconds"),
        "final_path": str(final_path) if final_path.is_file() else None,
        "final_epoch": final_epoch,
        "final_mtime": final_mtime,
        "provenance_valid": True,
    }


def summarize(cases: dict[str, CaseMetrics]) -> dict[str, float | int]:
    positive = [case for case in cases.values() if case.tumor_positive]
    negative = [case for case in cases.values() if not case.tumor_positive]
    liver = float(np.mean([case.liver_dice for case in cases.values()]))
    tumor = float(np.mean([case.tumor_dice for case in positive]))
    recall = float(np.mean([case.tumor_recall for case in positive]))
    precision = float(np.mean([case.tumor_precision for case in positive]))
    return {
        "n_total": len(cases),
        "n_positive": len(positive),
        "n_negative": len(negative),
        "liver_dice": liver,
        "tumor_dice": tumor,
        "overall": (liver + tumor) / 2,
        "tumor_recall": recall,
        "tumor_precision": precision,
    }


def load_and_audit() -> tuple[list[dict[str, object]], dict[str, object]]:
    checkpoint_audit = {
        method: checkpoint_info(method)
        for method in METHODS
    }
    rows: list[dict[str, object]] = []
    artifact_audit: dict[str, dict[str, object]] = {}

    for dataset in DATASETS:
        dataset_cases: dict[str, dict[str, CaseMetrics]] = {}
        dataset_artifacts: dict[str, object] = {}
        for method in METHODS:
            summary, report, prediction_dir, visualization_dir = dataset_paths(
                dataset,
                method,
            )
            if not summary.is_file():
                raise FileNotFoundError(f"Missing summary: {summary}")
            if not report.is_file() or report.stat().st_size == 0:
                raise FileNotFoundError(f"Missing or empty report: {report}")
            cases = load_cases(summary)
            dataset_cases[method] = cases

            expected_ids = set(cases)
            prediction_ids = (
                {
                    case_id_from_path(str(path))
                    for path in prediction_dir.glob("*.nii.gz")
                }
                if prediction_dir.is_dir()
                else set()
            )
            missing_predictions = sorted(expected_ids - prediction_ids)
            stale_predictions = sorted(prediction_ids - expected_ids)
            png_count = (
                sum(1 for _ in visualization_dir.rglob("*.png"))
                if visualization_dir.is_dir()
                else 0
            )
            predictions_complete = (
                not missing_predictions and not stale_predictions
            )
            visualizations_complete = png_count > 0
            complete = predictions_complete and visualizations_complete
            dataset_artifacts[method] = {
                "prediction_dir": str(prediction_dir),
                "prediction_count": len(prediction_ids),
                "missing_prediction_case_ids": missing_predictions,
                "stale_prediction_case_ids": stale_predictions,
                "summary": str(summary),
                "report": str(report),
                "visualization_dir": str(visualization_dir),
                "visualization_png_count": png_count,
                "predictions_complete": predictions_complete,
                "visualizations_complete": visualizations_complete,
                "checkpoint": checkpoint_audit[method],
                "status": "complete" if complete else "partial",
            }

        canonical_ids = set(dataset_cases[MAIN_MODEL])
        for method in METHODS:
            method_ids = set(dataset_cases[method])
            if method_ids != canonical_ids:
                raise ValueError(
                    f"{dataset}/{method} case mismatch; "
                    f"missing={sorted(canonical_ids - method_ids)}, "
                    f"extra={sorted(method_ids - canonical_ids)}"
                )
            for case_id in canonical_ids:
                reference = dataset_cases[MAIN_MODEL][case_id]
                candidate = dataset_cases[method][case_id]
                if (
                    reference.tumor_positive != candidate.tumor_positive
                    or reference.tumor_n_ref != candidate.tumor_n_ref
                ):
                    raise ValueError(
                        f"{dataset}/{method}/{case_id}: GT status/count mismatch"
                    )

        observed = summarize(dataset_cases[MAIN_MODEL])
        expected = EXPECTED_CASE_COUNTS[dataset]
        counts = (
            observed["n_total"],
            observed["n_positive"],
            observed["n_negative"],
        )
        if counts != expected:
            raise ValueError(
                f"{dataset}: observed counts {counts}, expected {expected}"
            )

        for method in METHODS:
            metrics = summarize(dataset_cases[method])
            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    **metrics,
                    "trainer": checkpoint_audit[method]["trainer"],
                    "fold": 0,
                    "checkpoint": checkpoint_audit[method]["checkpoint"],
                    "checkpoint_epoch": checkpoint_audit[method]["epoch"],
                    "checkpoint_path": checkpoint_audit[method]["path"],
                    "artifact_status": dataset_artifacts[method]["status"],
                }
            )
        artifact_audit[dataset] = dataset_artifacts

    return rows, {
        "generated_on": date.today().isoformat(),
        "methods": list(METHODS),
        "datasets": list(DATASETS),
        "checkpoint_audit": checkpoint_audit,
        "artifact_audit": artifact_audit,
        "interpretation_boundary": (
            "Datasets are reported separately; no cross-dataset average is used."
        ),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(rows: list[dict[str, object]], output_base: Path) -> None:
    metric_config = {
        "overall": ("Overall", "#355C7D", "o"),
        "tumor_dice": ("Tumor Dice", "#F67280", "s"),
        "tumor_recall": ("Tumor Recall", "#6C9A8B", "^"),
    }
    x = np.arange(len(METHODS), dtype=float)
    main_index = METHODS.index(MAIN_MODEL)
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 5.8), sharey=True)

    for axis, dataset in zip(axes, DATASETS, strict=True):
        dataset_rows = {
            str(row["method"]): row
            for row in rows
            if row["dataset"] == dataset
        }
        metrics = ("overall", "tumor_dice", "tumor_recall")
        offsets = np.linspace(-0.16, 0.16, num=len(metrics))
        axis.axvspan(
            main_index - 0.45,
            main_index + 0.45,
            color="#F6BD60",
            alpha=0.18,
            zorder=0,
        )

        for metric, offset in zip(metrics, offsets, strict=True):
            label, color, marker = metric_config[metric]
            values = np.asarray(
                [float(dataset_rows[method][metric]) for method in METHODS]
            )
            positions = x + offset
            axis.plot(
                positions,
                values,
                color=color,
                marker=marker,
                markersize=6,
                linewidth=1.6,
                label=label,
                zorder=3,
            )
            best_index = int(np.argmax(values))
            axis.scatter(
                [positions[best_index]],
                [values[best_index]],
                marker="*",
                s=150,
                color="#F2C14E",
                edgecolor="#6B4F00",
                linewidth=0.7,
                zorder=5,
            )
            for index, (position, value) in enumerate(
                zip(positions, values, strict=True)
            ):
                axis.annotate(
                    f"{value:.3f}",
                    (position, value),
                    xytext=(0, 7 if metric != "tumor_recall" else -13),
                    textcoords="offset points",
                    ha="center",
                    fontsize=6.3,
                    fontweight="bold" if index == best_index else "normal",
                    color="#222222",
                )

        axis.set_title(
            DATASET_DISPLAY_NAMES[dataset],
            fontsize=12,
            fontweight="bold",
        )
        axis.set_xticks(x)
        axis.set_xticklabels(
            [DISPLAY_NAMES[method] for method in METHODS],
            rotation=28,
            ha="right",
            fontsize=8.5,
        )
        for index, tick in enumerate(axis.get_xticklabels()):
            if index == main_index:
                tick.set_color("#B35300")
                tick.set_fontweight("bold")
        axis.set_xlim(-0.55, len(METHODS) - 0.45)
        axis.set_ylim(0.25, 1.0)
        axis.grid(axis="y", color="#D8D8D8", linewidth=0.7, alpha=0.8)
        axis.text(
            main_index,
            0.275,
            "Proposed",
            ha="center",
            va="bottom",
            fontsize=7.2,
            color="#B35300",
            fontweight="bold",
        )

    axes[0].set_ylabel("Metric value (0–1)", fontsize=10.5)
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.995),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    rows, metadata = load_and_audit()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "three_domain_main_results.csv"
    metadata_path = OUTPUT_DIR / "three_domain_main_results_metadata.json"
    figure_base = FIGURE_DIR / "three_domain_main_results"
    write_csv(csv_path, rows)
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    plot(rows, figure_base)
    print(f"Wrote {csv_path}")
    print(f"Wrote {metadata_path}")
    print(f"Wrote {figure_base.with_suffix('.png')}")
    print(f"Wrote {figure_base.with_suffix('.svg')}")


if __name__ == "__main__":
    main()
