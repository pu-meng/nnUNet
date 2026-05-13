"""
用stage-1 cv预测裁剪ROI,创建Dataset004
Create Dataset004_LiverTumor from Dataset003_Liver via Stage-1 CV predictions.

Pipeline (no data leakage)
  For each fold k in {0,1,2,3,4}:
    - Predict liver masks for fold k's val cases using the fold-k Stage-1 model.
    - Each case is only predicted by a model that was NOT trained on it.
  Combined predictions cover all 131 cases exactly once.

  Then for every case:
    - Use the liver prediction mask to compute a bounding box + margin.
    - Crop the original image and the original label to that ROI.
    - Remap labels: background=0 liver=1 → 0,  tumor=2 → 1.
    - Save as Dataset004.

  Crop metadata (original shape, crop indices, spacing) is saved to
  Dataset004/crop_meta.json for use at inference time (map-back).

Usage
  python pumengyu/twostage/create_dataset.py \\
      --workspace /home/PuMengYu/nnUNet_workspace \\
      [--dataset003_id 3] \\
      [--margin_mm 20] \\
      [--folds 0 1 2 3 4]

After this script finishes, run:
  nnUNetv2_plan_and_preprocess -d 4 --verify_dataset_integrity
"""

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import SimpleITK as sitk


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_splits(preprocessed_d003: Path) -> list[dict]:
    with open(preprocessed_d003 / "splits_final.json") as f:
        return json.load(f)


def predict_val_cases(
    fold: int,
    val_cases: list[str],
    images_tr: Path,
    pred_dir: Path,
    dataset_id: int,
    trainer: str = "nnUNetTrainer",
    plans:   str = "nnUNetPlans",
    config:  str = "3d_fullres",
) -> None:
    """
    Run nnUNetv2_predict for a specific fold, but only on the val cases of
    that fold (symlinked into a temp directory).
    Predictions are written directly into pred_dir.
    """
    pred_dir.mkdir(parents=True, exist_ok=True)

    # Check which val cases are already predicted
    missing = [c for c in val_cases
               if not (pred_dir / f"{c}.nii.gz").exists()]
    if not missing:
        print(f"  fold {fold}: all {len(val_cases)} predictions already exist, skipping.")
        return

    with tempfile.TemporaryDirectory(prefix=f"d003_fold{fold}_") as tmp:
        tmp_path = Path(tmp)
        for case in missing:
            src = images_tr / f"{case}_0000.nii.gz"
            dst = tmp_path / f"{case}_0000.nii.gz"
            dst.symlink_to(src.resolve())

        cmd = [
            "nnUNetv2_predict",
            "-i", str(tmp_path),
            "-o", str(pred_dir),
            "-d", str(dataset_id),
            "-c", config,
            "-f", str(fold),
            "-tr", trainer,
            "-p", plans,
        ]
        print(f"  fold {fold}: predicting {len(missing)} cases …")
        subprocess.run(cmd, check=True)


def liver_bbox_sitk(
    liver_mask: sitk.Image,
    margin_mm: float,
) -> tuple[list[int], list[int]]:
    """
    Return (start_xyz, size_xyz) for a ROI that covers the liver mask
    with a physical margin in mm on all sides.
    Coordinates are in SimpleITK convention: (x, y, z) = (col, row, slice).
    """
    arr = sitk.GetArrayViewFromImage(liver_mask)   # (z, y, x)
    spacing = liver_mask.GetSpacing()              # (sp_x, sp_y, sp_z)
    full_size = liver_mask.GetSize()               # (sx, sy, sz)

    nz = np.where(arr > 0)
    if len(nz[0]) == 0:
        return [0, 0, 0], list(full_size)

    # Array indices: (z, y, x)
    z_min, z_max = int(nz[0].min()), int(nz[0].max()) + 1
    y_min, y_max = int(nz[1].min()), int(nz[1].max()) + 1
    x_min, x_max = int(nz[2].min()), int(nz[2].max()) + 1

    # Margin in voxels for each axis
    mx = int(np.ceil(margin_mm / spacing[0]))
    my = int(np.ceil(margin_mm / spacing[1]))
    mz = int(np.ceil(margin_mm / spacing[2]))

    x0 = max(0,            x_min - mx)
    y0 = max(0,            y_min - my)
    z0 = max(0,            z_min - mz)
    x1 = min(full_size[0], x_max + mx)
    y1 = min(full_size[1], y_max + my)
    z1 = min(full_size[2], z_max + mz)

    return [x0, y0, z0], [x1 - x0, y1 - y0, z1 - z0]   # start, size


def crop_and_remap(
    case_id:    str,
    images_tr:  Path,
    labels_tr:  Path,
    pred_dir:   Path,
    out_images: Path,
    out_labels: Path,
    margin_mm:  float,
) -> dict:
    """
    Crop one case and return its metadata dict.
    Label remapping: {0→0, 1(liver)→0, 2(tumor)→1}.
    """
    img   = sitk.ReadImage(str(images_tr / f"{case_id}_0000.nii.gz"))
    label = sitk.ReadImage(str(labels_tr / f"{case_id}.nii.gz"))
    pred  = sitk.ReadImage(str(pred_dir  / f"{case_id}.nii.gz"))

    # Liver is class 1 in Dataset003 predictions
    liver_mask = sitk.BinaryThreshold(pred, lowerThreshold=1, upperThreshold=1,
                                      insideValue=1, outsideValue=0)

    start_xyz, size_xyz = liver_bbox_sitk(liver_mask, margin_mm)

    roi = sitk.RegionOfInterestImageFilter()
    roi.SetIndex(start_xyz)
    roi.SetSize(size_xyz)

    img_crop   = roi.Execute(img)
    label_crop = roi.Execute(label)

    # Remap: liver=1→0, tumor=2→1
    label_arr = sitk.GetArrayFromImage(label_crop).astype(np.uint8)
    tumor_arr = (label_arr == 2).astype(np.uint8)
    tumor_img = sitk.GetImageFromArray(tumor_arr)
    tumor_img.CopyInformation(label_crop)

    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    sitk.WriteImage(img_crop,  str(out_images / f"{case_id}_0000.nii.gz"))
    sitk.WriteImage(tumor_img, str(out_labels / f"{case_id}.nii.gz"))

    spacing = img.GetSpacing()   # (sp_x, sp_y, sp_z)
    full_sz = img.GetSize()

    return {
        "original_size_xyz":  list(full_sz),
        "crop_start_xyz":     start_xyz,
        "crop_size_xyz":      size_xyz,
        "spacing_xyz":        list(spacing),
        "origin":             list(img.GetOrigin()),
        "direction":          list(img.GetDirection()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace",    required=True,
                        help="Path to nnUNet_workspace root")
    parser.add_argument("--dataset003_id", type=int, default=3)
    parser.add_argument("--dataset004_id", type=int, default=4)
    parser.add_argument("--margin_mm",    type=float, default=20.0,
                        help="ROI margin around liver prediction (mm)")
    parser.add_argument("--folds",        type=int, nargs="+",
                        default=[0, 1, 2, 3, 4])
    parser.add_argument("--trainer",      default="nnUNetTrainer")
    parser.add_argument("--plans",        default="nnUNetPlans")
    parser.add_argument("--config",       default="3d_fullres")
    args = parser.parse_args()

    ws = Path(args.workspace)
    d3_id = args.dataset003_id
    d4_id = args.dataset004_id

    # ── paths ──
    raw_d3      = ws / "raw"          / f"Dataset{d3_id:03d}_Liver"
    pre_d3      = ws / "preprocessed" / f"Dataset{d3_id:03d}_Liver"
    results_d3  = (ws / "results" / f"Dataset{d3_id:03d}_Liver"
                   / f"{args.trainer}__{args.plans}__{args.config}")
    images_tr   = raw_d3 / "imagesTr"
    labels_tr   = raw_d3 / "labelsTr"

    raw_d4      = ws / "raw" / f"Dataset{d4_id:03d}_LiverTumor"
    out_images  = raw_d4 / "imagesTr"
    out_labels  = raw_d4 / "labelsTr"

    # CV predictions go here (one folder, all 131 cases combined)
    cv_pred_dir = results_d3 / "cv_predictions"

    # ── Step 1: per-fold predictions ──
    print("=== Step 1: Stage-1 CV predictions (no leakage) ===")
    splits = load_splits(pre_d3)
    all_val_cases: list[str] = []
    for fold in args.folds:
        val_cases = splits[fold]["val"]
        all_val_cases.extend(val_cases)
        predict_val_cases(
            fold=fold,
            val_cases=val_cases,
            images_tr=images_tr,
            pred_dir=cv_pred_dir,
            dataset_id=d3_id,
            trainer=args.trainer,
            plans=args.plans,
            config=args.config,
        )

    # Sanity: each case predicted exactly once
    assert len(all_val_cases) == len(set(all_val_cases)), \
        "Duplicate cases across folds — check splits_final.json"
    print(f"  Combined CV predictions: {len(all_val_cases)} cases")

    # ── Step 2: crop to liver ROI ──
    print("\n=== Step 2: Crop liver ROI → Dataset004 ===")
    crop_meta: dict[str, dict] = {}
    for case_id in sorted(all_val_cases):
        meta = crop_and_remap(
            case_id=case_id,
            images_tr=images_tr,
            labels_tr=labels_tr,
            pred_dir=cv_pred_dir,
            out_images=out_images,
            out_labels=out_labels,
            margin_mm=args.margin_mm,
        )
        crop_meta[case_id] = meta
        print(f"  {case_id}: {meta['original_size_xyz']} → {meta['crop_size_xyz']}")

    # ── Step 3: dataset.json ──
    dataset_json = {
        "name":            "LiverTumor",
        "description":     (
            "Tumor segmentation on liver-ROI-cropped CT. "
            "Cropped using Stage-1 CV predictions (no leakage). "
            f"Margin: {args.margin_mm} mm."
        ),
        "reference":       "Derived from MSD Task03_Liver",
        "licence":         "CC-BY-SA 4.0",
        "tensorImageSize": "3D",
        "channel_names":   {"0": "CT"},
        "labels":          {"background": 0, "tumor": 1},
        "numTraining":     len(all_val_cases),
        "file_ending":     ".nii.gz",
    }
    with open(raw_d4 / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=2)

    # ── Step 4: crop metadata (needed at inference for map-back) ──
    with open(raw_d4 / "crop_meta.json", "w") as f:
        json.dump(crop_meta, f, indent=2)

    print(f"\nDataset004 created at: {raw_d4}")
    print("Next step:")
    print(f"  nnUNetv2_plan_and_preprocess -d {d4_id} --verify_dataset_integrity")


if __name__ == "__main__":
    main()
