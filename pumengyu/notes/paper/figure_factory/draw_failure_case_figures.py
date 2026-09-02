from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/pumengyu_matplotlib_cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from PIL import Image
import SimpleITK as sitk


PAPER_DIR = Path(__file__).resolve().parents[1]
FIGURE_DIR = PAPER_DIR / "figures"
ASSET_DIR = PAPER_DIR / "assets"
STATISTICS_DIR = PAPER_DIR / "statistics"

WORKSPACE = Path("/home/PuMengYu/nnUNet_workspace")
IRCAD_ROOT = WORKSPACE / "external_val/ircadb_full"
IRCAD_RESULT = WORKSPACE / "results_v2/IRCADb/source_only/MedNeXt_MLA_MoE"
IRCAD_PREDICTIONS = IRCAD_RESULT / "predictions"

CURRENT_CHECKPOINT = (
    WORKSPACE
    / "results_v2/Dataset003_Liver"
    / "nnUNetTrainer_MedNeXt_MLA_MoE__nnUNetPlans__3d_fullres"
    / "fold_0/checkpoint_best.pth"
)
HISTORICAL_FOLD = (
    WORKSPACE
    / "results_v2/Dataset003_Liver"
    / "nnUNetTrainer_MLAUNet_MoE_SizeOversampleV4__nnUNetPlans__3d_fullres"
    / "fold_0"
)
HISTORICAL_CHECKPOINT = HISTORICAL_FOLD / "checkpoint_best.pth"

WINDOW_MIN = -150.0
WINDOW_MAX = 250.0
TP_COLOR = "#22c55e"
FP_COLOR = "#ef4444"
FN_COLOR = "#2563eb"
FRAME_COLOR = "#475569"
SOFT_COLOR = "#6b7280"


@dataclass(frozen=True)
class ExternalFigure:
    stem: str
    title: str
    case_id: str
    slices: tuple[int, ...]


@dataclass(frozen=True)
class HistoricalRow:
    source: str
    case_id: str
    z_index: int
    gt: int
    tp: int
    fp: int
    fn: int


@dataclass(frozen=True)
class HistoricalFigure:
    stem: str
    title: str
    rows: tuple[HistoricalRow, ...]


EXTERNAL_FIGURES = (
    ExternalFigure(
        "ircadb_014_z45_mednext_mla_fp",
        "No-tumor False Positive",
        "ircadb_014",
        (45,),
    ),
    ExternalFigure(
        "ircadb_015_z102_mednext_mla_fn",
        "Small-lesion False Negative",
        "ircadb_015",
        (102,),
    ),
    ExternalFigure(
        "ircadb_015_adjacent_slice_recovery",
        "Adjacent-slice Partial Recovery",
        "ircadb_015",
        (103, 104, 106),
    ),
    ExternalFigure(
        "ircadb_008_z96_mednext_mla_unexplained_fp",
        "Visually Ambiguous False Positive",
        "ircadb_008",
        (96,),
    ),
)

HISTORICAL_FIGURES = (
    HistoricalFigure(
        "lits_liver_41_false_positive",
        "Diffuse Hypodensity False Positive",
        (
            HistoricalRow(
                "liver_41_z45_full.png",
                "liver_41",
                45,
                0,
                0,
                1090,
                0,
            ),
        ),
    ),
    HistoricalFigure(
        "lits_liver_30_cystic_false_positive",
        "Focal Cystic-appearing False Positive",
        (
            HistoricalRow(
                "liver_30_z152_full.png",
                "liver_30",
                152,
                0,
                0,
                92,
                0,
            ),
        ),
    ),
    HistoricalFigure(
        "lits_liver_33_adjacent_slice_recovery",
        "Low-contrast Adjacent-slice Recovery",
        (
            HistoricalRow(
                "liver_33_z49_full.png",
                "liver_33",
                49,
                755,
                2,
                0,
                753,
            ),
            HistoricalRow(
                "liver_33_z50_full.png",
                "liver_33",
                50,
                1392,
                727,
                0,
                665,
            ),
        ),
    ),
    HistoricalFigure(
        "lits_liver_13_context_spillover",
        "3D Context Spillover across Tumor Boundary",
        (
            HistoricalRow(
                "liver_13_z327_full.png",
                "liver_13",
                327,
                0,
                0,
                58,
                0,
            ),
            HistoricalRow(
                "liver_13_z328_full.png",
                "liver_13",
                328,
                136,
                119,
                64,
                17,
            ),
            HistoricalRow(
                "liver_13_z334_full.png",
                "liver_13",
                334,
                641,
                593,
                63,
                48,
            ),
        ),
    ),
)


def read_volume(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path)))


def overlay_errors(
    axis: plt.Axes,
    image: np.ndarray,
    gt: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, int]:
    show_ct(axis, image)
    tp = np.logical_and(gt, prediction)
    fp = np.logical_and(~gt, prediction)
    fn = np.logical_and(gt, ~prediction)
    rgba = np.zeros((*gt.shape, 4), dtype=float)
    rgba[tp] = (*matplotlib.colors.to_rgb(TP_COLOR), 0.34)
    rgba[fp] = (*matplotlib.colors.to_rgb(FP_COLOR), 0.76)
    rgba[fn] = (*matplotlib.colors.to_rgb(FN_COLOR), 0.72)
    axis.imshow(rgba, interpolation="nearest")
    return {
        "gt": int(gt.sum()),
        "tp": int(tp.sum()),
        "fp": int(fp.sum()),
        "fn": int(fn.sum()),
    }


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
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)
        spine.set_edgecolor(FRAME_COLOR)


def lesion_crop(
    image: np.ndarray,
    gt: np.ndarray,
    prediction: np.ndarray,
    margin: int = 32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lesion = np.logical_or(gt, prediction)
    coordinates = np.argwhere(lesion)
    if coordinates.size == 0:
        return image, gt, prediction
    y_min, x_min = coordinates.min(axis=0)
    y_max, x_max = coordinates.max(axis=0)
    y_slice = slice(
        max(0, int(y_min) - margin),
        min(image.shape[0], int(y_max) + margin + 1),
    )
    x_slice = slice(
        max(0, int(x_min) - margin),
        min(image.shape[1], int(x_max) + margin + 1),
    )
    return image[y_slice, x_slice], gt[y_slice, x_slice], prediction[y_slice, x_slice]


def finish_case_figure(
    fig: plt.Figure,
    title: str,
    output_stem: str,
    rows: int,
    provenance_line: str,
) -> None:
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.995)
    legend = (
        Patch(facecolor=TP_COLOR, alpha=0.34, label="True positive"),
        Patch(facecolor=FP_COLOR, alpha=0.76, label="False positive"),
        Patch(facecolor=FN_COLOR, alpha=0.72, label="False negative"),
    )
    fig.legend(
        handles=legend,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.050),
        fontsize=9,
    )
    fig.text(
        0.5,
        0.014,
        provenance_line,
        ha="center",
        va="bottom",
        fontsize=8,
        color=SOFT_COLOR,
    )
    top = 0.91 if rows == 1 else 0.94
    fig.tight_layout(rect=(0.015, 0.145, 0.985, top), h_pad=0.78, w_pad=0.38)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / f"{output_stem}.svg", bbox_inches="tight")
    fig.savefig(
        FIGURE_DIR / f"{output_stem}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def draw_external_figure(spec: ExternalFigure) -> list[dict[str, object]]:
    image_path = IRCAD_ROOT / "images" / f"{spec.case_id}_0000.nii.gz"
    label_path = IRCAD_ROOT / "labels" / f"{spec.case_id}.nii.gz"
    prediction_path = IRCAD_PREDICTIONS / f"{spec.case_id}.nii.gz"
    image = read_volume(image_path)
    label = read_volume(label_path)
    prediction = read_volume(prediction_path)
    if not (image.shape == label.shape == prediction.shape):
        raise ValueError(
            f"{spec.case_id}: shape mismatch "
            f"{image.shape}, {label.shape}, {prediction.shape}"
        )

    row_count = len(spec.slices)
    fig, axes = plt.subplots(
        row_count,
        3,
        figsize=(9.7, 3.10 * row_count + 1.45),
        squeeze=False,
    )
    column_titles = ("CT", "Prediction Error Map", "Lesion Zoom")
    for column, title in enumerate(column_titles):
        axes[0, column].set_title(title, fontsize=11, fontweight="bold", pad=8)

    records: list[dict[str, object]] = []
    for row_index, z_index in enumerate(spec.slices):
        if not 0 <= z_index < image.shape[0]:
            raise IndexError(f"{spec.case_id}: invalid z={z_index}")
        ct_slice = image[z_index]
        gt_slice = label[z_index] == 2
        pred_slice = prediction[z_index] == 2
        show_ct(axes[row_index, 0], ct_slice)
        counts = overlay_errors(
            axes[row_index, 1],
            ct_slice,
            gt_slice,
            pred_slice,
        )
        zoom_ct, zoom_gt, zoom_pred = lesion_crop(
            ct_slice,
            gt_slice,
            pred_slice,
        )
        overlay_errors(
            axes[row_index, 2],
            zoom_ct,
            zoom_gt,
            zoom_pred,
        )
        axes[row_index, 0].set_ylabel(
            f"{spec.case_id}\nz = {z_index}",
            fontsize=9,
            fontweight="bold",
            rotation=90,
            labelpad=8,
        )
        axes[row_index, 1].text(
            0.5,
            -0.055,
            (
                f"GT {counts['gt']}  |  TP {counts['tp']}  |  "
                f"FP {counts['fp']}  |  FN {counts['fn']}"
            ),
            transform=axes[row_index, 1].transAxes,
            ha="center",
            va="top",
            fontsize=8.5,
            color="#334155",
        )
        records.append(
            {
                "case_id": spec.case_id,
                "slice_z": z_index,
                **counts,
                "image_path": str(image_path),
                "label_path": str(label_path),
                "prediction_path": str(prediction_path),
            }
        )

    finish_case_figure(
        fig,
        spec.title,
        spec.stem,
        row_count,
        "3D-IRCADb | MedNeXt_MLA_MoE source-only | checkpoint_best.pth",
    )
    return records


def longest_true_run(mask: np.ndarray) -> tuple[int, int]:
    indices = np.flatnonzero(mask)
    if len(indices) == 0:
        raise ValueError("No image region detected")
    breaks = np.flatnonzero(np.diff(indices) > 1)
    starts = np.r_[0, breaks + 1]
    stops = np.r_[breaks + 1, len(indices)]
    start, stop = max(
        zip(starts, stops, strict=True),
        key=lambda pair: int(indices[pair[1] - 1] - indices[pair[0]]),
    )
    return int(indices[start]), int(indices[stop - 1]) + 1


def extract_historical_panels(source_path: Path) -> tuple[np.ndarray, ...]:
    """Crop the three rendered panels from a traceable historical PNG.

    The original full-split predictions were deleted after visualization.
    Therefore these panels are intentionally derived from the preserved
    test_viz/train_viz raster instead of pretending to reconstruct masks.
    """
    image = np.asarray(Image.open(source_path).convert("RGB"))
    height, width = image.shape[:2]
    segments = ((0.00, 0.39), (0.37, 0.77), (0.74, 1.00))
    panels: list[np.ndarray] = []
    for start_ratio, stop_ratio in segments:
        x_start = int(width * start_ratio)
        x_stop = int(width * stop_ratio)
        segment = image[:, x_start:x_stop]
        ink = segment.mean(axis=2) < 244
        row_start, row_stop = longest_true_run(ink.mean(axis=1) > 0.12)
        col_start, col_stop = longest_true_run(ink.mean(axis=0) > 0.14)
        pad = 2
        row_start = max(0, row_start - pad)
        row_stop = min(height, row_stop + pad)
        col_start = max(0, col_start - pad)
        col_stop = min(segment.shape[1], col_stop + pad)
        panels.append(segment[row_start:row_stop, col_start:col_stop])
    return tuple(panels)


def draw_historical_figure(spec: HistoricalFigure) -> list[dict[str, object]]:
    row_count = len(spec.rows)
    fig, axes = plt.subplots(
        row_count,
        3,
        figsize=(9.7, 3.08 * row_count + 1.45),
        squeeze=False,
    )
    column_titles = ("CT", "Prediction Error Map", "Lesion Zoom")
    for column, title in enumerate(column_titles):
        axes[0, column].set_title(title, fontsize=11, fontweight="bold", pad=8)

    records: list[dict[str, object]] = []
    for row_index, row in enumerate(spec.rows):
        source_path = ASSET_DIR / row.source
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        panels = extract_historical_panels(source_path)
        for column, panel in enumerate(panels):
            axes[row_index, column].imshow(panel, interpolation="nearest")
            axes[row_index, column].set_xticks([])
            axes[row_index, column].set_yticks([])
            for spine in axes[row_index, column].spines.values():
                spine.set_visible(True)
                spine.set_linewidth(1.2)
                spine.set_edgecolor(FRAME_COLOR)
        axes[row_index, 0].set_ylabel(
            f"{row.case_id}\nz = {row.z_index}",
            fontsize=9,
            fontweight="bold",
            rotation=90,
            labelpad=8,
        )
        axes[row_index, 1].text(
            0.5,
            -0.055,
            (
                f"GT {row.gt}  |  TP {row.tp}  |  "
                f"FP {row.fp}  |  FN {row.fn}"
            ),
            transform=axes[row_index, 1].transAxes,
            ha="center",
            va="top",
            fontsize=8.5,
            color="#334155",
        )
        records.append(
            {
                "case_id": row.case_id,
                "slice_z": row.z_index,
                "gt": row.gt,
                "tp": row.tp,
                "fp": row.fp,
                "fn": row.fn,
                "source_raster": str(source_path),
                "source_sha256": sha256(source_path),
            }
        )

    finish_case_figure(
        fig,
        spec.title,
        spec.stem,
        row_count,
        (
            "Dataset003_Liver | historical MLAUNet_MoE_SizeOversampleV4 | "
            "checkpoint_best.pth"
        ),
    )
    return records


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_record(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    import torch

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    return {
        "path": str(path),
        "checkpoint": path.name,
        "trainer": checkpoint.get("trainer_name"),
        "epoch": int(checkpoint.get("current_epoch", -1)),
        "mtime": datetime.fromtimestamp(path.stat().st_mtime).isoformat(
            timespec="seconds"
        ),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def artifact_record(
    prediction_dir: Path,
    summary_path: Path,
    report_path: Path,
    visualization_dir: Path,
) -> dict[str, object]:
    return {
        "prediction_dir": str(prediction_dir),
        "prediction_count": (
            sum(1 for _ in prediction_dir.glob("*.nii.gz"))
            if prediction_dir.is_dir()
            else 0
        ),
        "summary_path": str(summary_path),
        "summary_exists": summary_path.is_file(),
        "report_path": str(report_path),
        "report_exists": report_path.is_file() and report_path.stat().st_size > 0,
        "visualization_dir": str(visualization_dir),
        "visualization_png_count": (
            sum(1 for _ in visualization_dir.rglob("*.png"))
            if visualization_dir.is_dir()
            else 0
        ),
    }


def main() -> None:
    external_records: dict[str, list[dict[str, object]]] = {}
    historical_records: dict[str, list[dict[str, object]]] = {}
    for spec in EXTERNAL_FIGURES:
        external_records[spec.stem] = draw_external_figure(spec)
    for spec in HISTORICAL_FIGURES:
        historical_records[spec.stem] = draw_historical_figure(spec)

    metadata = {
        "generated_on": datetime.now().isoformat(timespec="seconds"),
        "style_specification": str(
            PAPER_DIR / "figure_factory/论文绘图规范与工具方法.md"
        ),
        "external_ircadb": {
            "dataset": "3D-IRCADb",
            "model_source": "Dataset003_Liver source-only",
            "checkpoint": checkpoint_record(CURRENT_CHECKPOINT),
            "artifacts": artifact_record(
                IRCAD_PREDICTIONS,
                IRCAD_PREDICTIONS / "summary.json",
                IRCAD_RESULT / "report_custom.txt",
                IRCAD_RESULT / "test_viz",
            ),
            "figures": external_records,
        },
        "historical_internal": {
            "dataset": "Dataset003_Liver full-split visualization",
            "checkpoint": checkpoint_record(HISTORICAL_CHECKPOINT),
            "artifacts": artifact_record(
                HISTORICAL_FOLD / "test_prediction",
                HISTORICAL_FOLD / "test_prediction/summary.json",
                HISTORICAL_FOLD / "test_report_custom.txt",
                HISTORICAL_FOLD / "test_viz",
            ),
            "figures": historical_records,
            "provenance_boundary": (
                "The selected historical train-case NIfTI predictions were "
                "deleted after visualization. Figures 10-13 are recomposed "
                "from preserved source PNGs and are not attributed to the "
                "current MedNeXt_MLA_MoE model."
            ),
        },
    }
    STATISTICS_DIR.mkdir(parents=True, exist_ok=True)
    metadata_path = STATISTICS_DIR / "failure_case_figure_provenance.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(EXTERNAL_FIGURES) + len(HISTORICAL_FIGURES)} figures")
    print(f"Wrote {metadata_path}")


if __name__ == "__main__":
    main()
