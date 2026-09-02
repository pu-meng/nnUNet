#!/usr/bin/env python3
"""Generate the HCC cross-trainer consensus-failure overview figure."""

from __future__ import annotations

import csv
import json
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
import numpy as np

from pumengyu.tools.analyasis.generate_cross_trainer_case_analysis import (
    FAIR_METHODS,
    _external_paths,
    _load_domain,
    _validate,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = REPO_ROOT / "pumengyu/notes/paper/statistics"
FIGURE_DIR = REPO_ROOT / "pumengyu/notes/paper/figures"
MAIN_MODEL = "MedNeXt_MLA_MoE"
FAILURE_THRESHOLD = 0.3


def build_data() -> tuple[
    list[str],
    list[str],
    np.ndarray,
    list[dict[str, object]],
    dict[str, object],
]:
    domain = _load_domain(
        "HCC",
        "HCCReferencedCT v2 source-only",
        _external_paths("results_v2/Dataset013_HCCReferencedCT/source_only"),
    )
    _validate(domain)
    if len(domain.methods) != len(FAIR_METHODS):
        raise ValueError(
            f"HCC method coverage {len(domain.methods)}/{len(FAIR_METHODS)}"
        )
    if MAIN_MODEL not in domain.methods:
        raise ValueError(f"Missing main model: {MAIN_MODEL}")

    methods = [method for method in FAIR_METHODS if method in domain.methods]
    case_ids = sorted(next(iter(domain.methods.values())))
    raw_matrix = np.asarray(
        [
            [
                bool(
                    domain.methods[method][case_id].dice is not None
                    and domain.methods[method][case_id].dice < FAILURE_THRESHOLD
                )
                for case_id in case_ids
            ]
            for method in methods
        ],
        dtype=np.uint8,
    )
    failure_counts = raw_matrix.sum(axis=0)
    order = sorted(
        range(len(case_ids)),
        key=lambda index: (-int(failure_counts[index]), case_ids[index]),
    )
    ordered_cases = [case_ids[index] for index in order]
    matrix = raw_matrix[:, order]

    case_rows: list[dict[str, object]] = []
    for column, case_id in enumerate(ordered_cases):
        count = int(matrix[:, column].sum())
        case_rows.append(
            {
                "dataset": "HCCReferencedCT v2",
                "case_id": case_id,
                "failed_methods": count,
                "total_methods": len(methods),
                "failure_proportion": count / len(methods),
                "all_methods_failed": count == len(methods),
                "majority_failed": count / len(methods) >= 0.5,
                "failure_threshold": FAILURE_THRESHOLD,
            }
        )

    metadata = {
        "generated_on": date.today().isoformat(),
        "dataset": "HCCReferencedCT v2 source-only held-out test",
        "case_count": len(ordered_cases),
        "method_count": len(methods),
        "methods": methods,
        "case_order": ordered_cases,
        "failure_definition": f"Tumor Dice < {FAILURE_THRESHOLD}",
        "summary_sources": domain.sources,
        "missing_methods": domain.missing,
        "all_method_failure_cases": [
            row["case_id"] for row in case_rows if row["all_methods_failed"]
        ],
        "majority_failure_cases": [
            row["case_id"] for row in case_rows if row["majority_failed"]
        ],
        "checkpoint_provenance": (
            "Not re-audited for all 30 methods by this figure generator; "
            "the figure reads the existing source-only summary.json files."
        ),
    }
    return methods, ordered_cases, matrix, case_rows, metadata


def write_case_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_matrix_csv(
    path: Path,
    methods: list[str],
    case_ids: list[str],
    matrix: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", *case_ids])
        for method, row in zip(methods, matrix, strict=True):
            writer.writerow([method, *[int(value) for value in row]])


def plot(
    methods: list[str],
    case_ids: list[str],
    matrix: np.ndarray,
    output_base: Path,
) -> None:
    proportions = matrix.mean(axis=0)
    all_failed = proportions == 1.0
    majority = proportions >= 0.5
    bar_colors = np.where(
        all_failed,
        "#C0392B",
        np.where(majority, "#E67E22", "#AAB2BD"),
    )

    fig = plt.figure(figsize=(15.8, 11.2))
    grid = fig.add_gridspec(2, 1, height_ratios=(1.4, 6.5), hspace=0.06)
    bar_axis = fig.add_subplot(grid[0])
    heat_axis = fig.add_subplot(grid[1], sharex=bar_axis)
    x = np.arange(len(case_ids))

    bar_axis.bar(x, proportions, color=bar_colors, width=0.82)
    bar_axis.axhline(0.5, color="#555555", linestyle="--", linewidth=1)
    bar_axis.set_ylim(0, 1.08)
    bar_axis.set_ylabel("Severe-failure\nproportion")
    bar_axis.set_yticks([0, 0.5, 1.0])
    bar_axis.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    bar_axis.tick_params(axis="x", labelbottom=False)
    for index, proportion in enumerate(proportions):
        if proportion < 0.5:
            continue
        bar_axis.text(
            index,
            proportion + 0.025,
            f"{int(matrix[:, index].sum())}/{len(methods)}",
            ha="center",
            va="bottom",
            rotation=90,
            fontsize=7,
        )

    heat_axis.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
        cmap=ListedColormap(["#F7F7F7", "#4C78A8"]),
        vmin=0,
        vmax=1,
    )
    heat_axis.set_xticks(x)
    heat_axis.set_xticklabels(case_ids, rotation=65, ha="right", fontsize=8)
    heat_axis.set_yticks(np.arange(len(methods)))
    heat_axis.set_yticklabels(methods, fontsize=7.5)
    heat_axis.set_xlabel("HCC held-out test cases (ordered by consensus failure)")
    heat_axis.set_ylabel("30 Dataset003 source-only methods")
    heat_axis.set_xticks(np.arange(-0.5, len(case_ids), 1), minor=True)
    heat_axis.set_yticks(np.arange(-0.5, len(methods), 1), minor=True)
    heat_axis.grid(which="minor", color="white", linewidth=0.35)
    heat_axis.tick_params(which="minor", bottom=False, left=False)

    main_index = methods.index(MAIN_MODEL)
    heat_axis.add_patch(
        Rectangle(
            (-0.5, main_index - 0.5),
            len(case_ids),
            1,
            fill=False,
            edgecolor="#F2C14E",
            linewidth=2.2,
        )
    )
    heat_axis.get_yticklabels()[main_index].set_color("#B35300")
    heat_axis.get_yticklabels()[main_index].set_fontweight("bold")
    for index, universal in enumerate(all_failed):
        if universal:
            heat_axis.add_patch(
                Rectangle(
                    (index - 0.5, -0.5),
                    1,
                    len(methods),
                    fill=False,
                    edgecolor="#C0392B",
                    linewidth=1.2,
                )
            )

    fig.suptitle(
        "HCC consensus severe failures across 30 source-only methods",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    fig.text(
        0.5,
        0.012,
        "Blue: Tumor Dice < 0.3. Red columns: all 30 methods failed. "
        "Orange bars: at least half failed. Gold outline: MedNeXt_MLA_MoE.",
        ha="center",
        fontsize=9,
    )
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    methods, cases, matrix, case_rows, metadata = build_data()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    case_csv = OUTPUT_DIR / "hcc_cross_trainer_failure_cases.csv"
    matrix_csv = OUTPUT_DIR / "hcc_cross_trainer_failure_matrix.csv"
    metadata_path = OUTPUT_DIR / "hcc_cross_trainer_failure_metadata.json"
    figure_base = FIGURE_DIR / "hcc_cross_trainer_failure_overview"
    write_case_csv(case_csv, case_rows)
    write_matrix_csv(matrix_csv, methods, cases, matrix)
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    plot(methods, cases, matrix, figure_base)
    print(f"Wrote {case_csv}")
    print(f"Wrote {matrix_csv}")
    print(f"Wrote {metadata_path}")
    print(f"Wrote {figure_base.with_suffix('.png')}")
    print(f"Wrote {figure_base.with_suffix('.svg')}")


if __name__ == "__main__":
    main()
