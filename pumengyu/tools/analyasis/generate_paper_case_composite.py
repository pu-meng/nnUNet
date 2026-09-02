#!/usr/bin/env python3
"""Generate a provenance-traceable IRCADb success/failure composite figure.

The figure compares the fixed Dataset003 source-only ``checkpoint_best.pth``
predictions from MedNeXt and MedNeXt_MLA_MoE. It only reads existing NIfTI
files and does not run inference.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import SimpleITK as sitk

from pumengyu.tools.analyasis.generate_paper_paired_statistics import (
    checkpoint_provenance_audit,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE = Path("/home/PuMengYu/nnUNet_workspace")
IRCAD_ROOT = WORKSPACE / "external_val/ircadb_full"
RESULT_ROOT = WORKSPACE / "results_v2/IRCADb/source_only"
FIGURE_DIR = REPO_ROOT / "pumengyu/notes/paper/figures"
STATISTICS_DIR = REPO_ROOT / "pumengyu/notes/paper/statistics"

MODEL_BASELINE = "MedNeXt"
MODEL_MAIN = "MedNeXt_MLA_MoE"
WINDOW_MIN = -200.0
WINDOW_MAX = 250.0


@dataclass(frozen=True)
class CaseVolumes:
    case_id: str
    image: np.ndarray
    label: np.ndarray
    baseline: np.ndarray
    main: np.ndarray
    image_path: Path
    label_path: Path
    baseline_path: Path
    main_path: Path


def read_array(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path)))


def load_case(case_id: str) -> CaseVolumes:
    image_path = IRCAD_ROOT / "images" / f"{case_id}_0000.nii.gz"
    label_path = IRCAD_ROOT / "labels" / f"{case_id}.nii.gz"
    baseline_path = (
        RESULT_ROOT / MODEL_BASELINE / "predictions" / f"{case_id}.nii.gz"
    )
    main_path = (
        RESULT_ROOT / MODEL_MAIN / "predictions" / f"{case_id}.nii.gz"
    )
    volumes = CaseVolumes(
        case_id=case_id,
        image=read_array(image_path),
        label=read_array(label_path),
        baseline=read_array(baseline_path),
        main=read_array(main_path),
        image_path=image_path,
        label_path=label_path,
        baseline_path=baseline_path,
        main_path=main_path,
    )
    shapes = {
        volumes.image.shape,
        volumes.label.shape,
        volumes.baseline.shape,
        volumes.main.shape,
    }
    if len(shapes) != 1:
        raise ValueError(f"{case_id}: shape mismatch {sorted(shapes)}")
    return volumes


def dice(gt: np.ndarray, pred: np.ndarray) -> float:
    gt_count = int(gt.sum())
    pred_count = int(pred.sum())
    if gt_count + pred_count == 0:
        return 1.0
    intersection = int(np.logical_and(gt, pred).sum())
    return 2.0 * intersection / (gt_count + pred_count)


def best_improvement_slice(case: CaseVolumes) -> int:
    """Choose the slice where the main model removes the most baseline FP."""
    gt_tumor = case.label == 2
    baseline_tumor = case.baseline == 2
    main_tumor = case.main == 2
    candidates = np.flatnonzero(
        np.logical_or.reduce((gt_tumor, baseline_tumor, main_tumor))
        .reshape(gt_tumor.shape[0], -1)
        .any(axis=1)
    )
    if len(candidates) == 0:
        raise ValueError(f"{case.case_id}: no GT-positive slices")
    return int(
        max(
            candidates,
            key=lambda z: (
                int(np.logical_and(~gt_tumor[z], baseline_tumor[z]).sum())
                - int(np.logical_and(~gt_tumor[z], main_tumor[z]).sum()),
                int(np.logical_xor(gt_tumor[z], baseline_tumor[z]).sum())
                - int(np.logical_xor(gt_tumor[z], main_tumor[z]).sum()),
                dice(gt_tumor[z], main_tumor[z])
                - dice(gt_tumor[z], baseline_tumor[z]),
                int(gt_tumor[z].sum()),
            ),
        )
    )


def crop_bounds(case: CaseVolumes, slices: list[int]) -> tuple[slice, slice]:
    masks: list[np.ndarray] = []
    for z_index in slices:
        masks.extend(
            [
                case.label[z_index] > 0,
                case.baseline[z_index] > 0,
                case.main[z_index] > 0,
            ]
        )
    union = np.logical_or.reduce(masks)
    coordinates = np.argwhere(union)
    if coordinates.size == 0:
        height, width = union.shape
        return slice(0, height), slice(0, width)
    y_min, x_min = coordinates.min(axis=0)
    y_max, x_max = coordinates.max(axis=0)
    margin = 18
    return (
        slice(max(0, int(y_min) - margin), min(union.shape[0], int(y_max) + margin + 1)),
        slice(max(0, int(x_min) - margin), min(union.shape[1], int(x_max) + margin + 1)),
    )


def show_ct(axis: plt.Axes, image: np.ndarray) -> None:
    axis.imshow(
        image,
        cmap="gray",
        vmin=WINDOW_MIN,
        vmax=WINDOW_MAX,
        interpolation="nearest",
    )
    axis.set_xticks([])
    axis.set_yticks([])


def overlay_mask(
    axis: plt.Axes,
    mask: np.ndarray,
    color: str,
    alpha: float = 0.55,
) -> None:
    rgba = np.zeros((*mask.shape, 4), dtype=float)
    rgba[mask, :3] = matplotlib.colors.to_rgb(color)
    rgba[mask, 3] = alpha
    axis.imshow(rgba, interpolation="nearest")


def slice_record(
    panel: str,
    case: CaseVolumes,
    z_index: int,
) -> dict[str, object]:
    gt = case.label[z_index] == 2
    baseline = case.baseline[z_index] == 2
    main = case.main[z_index] == 2
    return {
        "panel": panel,
        "case_id": case.case_id,
        "slice_z": z_index,
        "gt_tumor_voxels": int(gt.sum()),
        "baseline_tumor_voxels": int(baseline.sum()),
        "main_tumor_voxels": int(main.sum()),
        "baseline_slice_dice": dice(gt, baseline),
        "main_slice_dice": dice(gt, main),
        "main_tp_voxels": int(np.logical_and(gt, main).sum()),
        "main_fp_voxels": int(np.logical_and(~gt, main).sum()),
        "main_fn_voxels": int(np.logical_and(gt, ~main).sum()),
    }


def plot(
    rows: list[tuple[str, CaseVolumes, int]],
    crops: dict[str, tuple[slice, slice]],
    output_base: Path,
) -> None:
    column_titles = (
        "CT",
        "Ground truth",
        "MedNeXt",
        "MedNeXt_MLA_MoE",
        "Main-model error",
    )
    fig, axes = plt.subplots(
        len(rows),
        len(column_titles),
        figsize=(13.8, 16.2),
    )
    for column, title in enumerate(column_titles):
        axes[0, column].set_title(title, fontsize=11, fontweight="bold")

    for row_index, (panel, case, z_index) in enumerate(rows):
        y_slice, x_slice = crops[case.case_id]
        ct = case.image[z_index, y_slice, x_slice]
        gt = case.label[z_index, y_slice, x_slice] == 2
        baseline = case.baseline[z_index, y_slice, x_slice] == 2
        main = case.main[z_index, y_slice, x_slice] == 2
        fp = np.logical_and(~gt, main)
        fn = np.logical_and(gt, ~main)

        for column in range(len(column_titles)):
            show_ct(axes[row_index, column], ct)
        overlay_mask(axes[row_index, 1], gt, "#33A02C")
        overlay_mask(axes[row_index, 2], baseline, "#FFB000")
        overlay_mask(axes[row_index, 3], main, "#FFB000")
        overlay_mask(axes[row_index, 4], fp, "#E31A1C")
        overlay_mask(axes[row_index, 4], fn, "#1F78B4")
        axes[row_index, 0].set_ylabel(
            f"{panel}\n{case.case_id}, z={z_index}",
            fontsize=9.2,
            rotation=0,
            labelpad=58,
            va="center",
        )

    legend = [
        Patch(facecolor="#33A02C", alpha=0.55, label="GT tumor"),
        Patch(facecolor="#FFB000", alpha=0.55, label="Predicted tumor"),
        Patch(facecolor="#E31A1C", alpha=0.55, label="False positive"),
        Patch(facecolor="#1F78B4", alpha=0.55, label="False negative"),
    ]
    fig.legend(
        handles=legend,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.006),
    )
    fig.suptitle(
        "IRCADb case-level improvement and representative failure patterns",
        fontsize=14,
        fontweight="bold",
        y=0.998,
    )
    fig.text(
        0.5,
        0.032,
        "A: case-level improvement through false-positive suppression; "
        "B: persistent negative-case false positive; "
        "C: small-lesion miss; D1-D3: adjacent-slice partial recovery. "
        f"CT window [{WINDOW_MIN:.0f}, {WINDOW_MAX:.0f}] HU.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0.06, 0.055, 1, 0.982), h_pad=0.75, w_pad=0.35)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    case_016 = load_case("ircadb_016")
    case_014 = load_case("ircadb_014")
    case_015 = load_case("ircadb_015")
    improvement_z = best_improvement_slice(case_016)
    rows = [
        ("A", case_016, improvement_z),
        ("B", case_014, 45),
        ("C", case_015, 102),
        ("D1", case_015, 103),
        ("D2", case_015, 104),
        ("D3", case_015, 106),
    ]
    for _, case, z_index in rows:
        if z_index < 0 or z_index >= case.image.shape[0]:
            raise IndexError(f"{case.case_id}: invalid z={z_index}")

    crops = {
        "ircadb_016": crop_bounds(case_016, [improvement_z]),
        "ircadb_014": crop_bounds(case_014, [45]),
        "ircadb_015": crop_bounds(case_015, [102, 103, 104, 106]),
    }
    output_base = FIGURE_DIR / "paper_case_composite_ircadb"
    plot(rows, crops, output_base)

    checkpoints = checkpoint_provenance_audit()
    metadata = {
        "generated_on": date.today().isoformat(),
        "dataset": "3D-IRCADb",
        "source_only": True,
        "models": [MODEL_BASELINE, MODEL_MAIN],
        "checkpoints": {
            model: checkpoints[model]
            for model in (MODEL_BASELINE, MODEL_MAIN)
        },
        "window_hu": [WINDOW_MIN, WINDOW_MAX],
        "rows": [
            {
                **slice_record(panel, case, z_index),
                "image_path": str(case.image_path),
                "label_path": str(case.label_path),
                "baseline_prediction_path": str(case.baseline_path),
                "main_prediction_path": str(case.main_path),
                "crop_y": [crops[case.case_id][0].start, crops[case.case_id][0].stop],
                "crop_x": [crops[case.case_id][1].start, crops[case.case_id][1].stop],
            }
            for panel, case, z_index in rows
        ],
        "interpretation_boundary": (
            "Slice-level panels illustrate spatial error patterns and do not "
            "replace case-level paired statistics."
        ),
    }
    STATISTICS_DIR.mkdir(parents=True, exist_ok=True)
    metadata_path = STATISTICS_DIR / "paper_case_composite_ircadb_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {output_base.with_suffix('.png')}")
    print(f"Wrote {output_base.with_suffix('.svg')}")
    print(f"Wrote {metadata_path}")


if __name__ == "__main__":
    main()
