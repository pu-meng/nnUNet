"""
Convert planned HCC-TACE-Seg cases to a standalone nnU-Net raw dataset.

This converter consumes the CSV produced by plan_multiphase_cases.py. It writes
a physically separate dataset such as Dataset013_HCCMultiPhase and never writes
to Dataset003_Liver.

First-pass channel layout:
  0000 = PRE / non-contrast CT, resampled to the referenced CT space
  0001 = SEG-referenced contrast CT

Labels:
  0 = background
  1 = liver
  2 = tumor (Mass and Necrosis)

Example dry run:
  python -m pumengyu.tools.hcc.convert_multiphase \
    --plan-csv pumengyu/notes/data/hcc_multiphase_case_plan.csv \
    --out-dir /tmp/Dataset013_HCCMultiPhase \
    --cases HCC_001

Real output should go to:
  /home/PuMengYu/nnUNet_workspace/raw/Dataset013_HCCMultiPhase
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pydicom
import SimpleITK as sitk


DEFAULT_PLAN_CSV = Path("pumengyu/notes/data/hcc_multiphase_case_plan.csv")
DEFAULT_DATASET_NAME = "Dataset013_HCCMultiPhase"

LIVER_LABELS = {"liver"}
TUMOR_LABELS = {"mass", "necrosis"}


def read_plan(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_dicom_series(series_dir: Path) -> sitk.Image:
    return read_dicom_stack(series_dir)


def _normal_from_orientation(iop) -> np.ndarray:
    row = np.asarray([float(iop[0]), float(iop[1]), float(iop[2])], dtype=np.float64)
    col = np.asarray([float(iop[3]), float(iop[4]), float(iop[5])], dtype=np.float64)
    normal = np.cross(row, col)
    normal /= np.linalg.norm(normal)
    return normal


def _direction_from_orientation(iop) -> tuple[float, ...]:
    row = np.asarray([float(iop[0]), float(iop[1]), float(iop[2])], dtype=np.float64)
    col = np.asarray([float(iop[3]), float(iop[4]), float(iop[5])], dtype=np.float64)
    normal = np.cross(row, col)
    normal /= np.linalg.norm(normal)
    # ITK direction matrix columns are x-axis, y-axis, z-axis directions.
    mat = np.stack([row, col, normal], axis=1)
    return tuple(float(x) for x in mat.reshape(-1))


def _slice_sort_key(ds: pydicom.Dataset, normal: np.ndarray) -> float:
    ipp = np.asarray([float(x) for x in ds.ImagePositionPatient], dtype=np.float64)
    return float(np.dot(ipp, normal))


def read_dicom_stack(series_dir: Path, sop_instance_uids: set[str] | None = None) -> sitk.Image:
    files = sorted(series_dir.glob("*.dcm"))
    if not files:
        raise FileNotFoundError(f"No DICOM files found: {series_dir}")

    slices = []
    for f in files:
        ds = pydicom.dcmread(str(f), force=True)
        if getattr(ds, "Modality", "") != "CT":
            continue
        if sop_instance_uids is not None and str(getattr(ds, "SOPInstanceUID", "")) not in sop_instance_uids:
            continue
        if not hasattr(ds, "ImagePositionPatient") or not hasattr(ds, "ImageOrientationPatient"):
            continue
        slices.append((f, ds))

    if not slices:
        raise FileNotFoundError(f"No matching CT slices found: {series_dir}")

    iop = slices[0][1].ImageOrientationPatient
    normal = _normal_from_orientation(iop)

    # Some HCC folders contain duplicate z positions inside one SeriesInstanceUID.
    # Keep one slice per physical position, preferring the larger InstanceNumber;
    # this selects the later contiguous reconstruction in observed duplicated stacks.
    by_pos: dict[float, tuple[Path, pydicom.Dataset]] = {}
    for f, ds in slices:
        pos = round(_slice_sort_key(ds, normal), 3)
        old = by_pos.get(pos)
        if old is None:
            by_pos[pos] = (f, ds)
            continue
        old_inst = int(float(getattr(old[1], "InstanceNumber", 0)))
        new_inst = int(float(getattr(ds, "InstanceNumber", 0)))
        if new_inst > old_inst:
            by_pos[pos] = (f, ds)

    ordered = [by_pos[pos] for pos in sorted(by_pos)]
    arrays = []
    positions = []
    for _, ds in ordered:
        arr = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        arrays.append(arr * slope + intercept)
        positions.append(_slice_sort_key(ds, normal))

    volume = np.stack(arrays, axis=0)
    image = sitk.GetImageFromArray(volume)

    first = ordered[0][1]
    pixel_spacing = [float(x) for x in first.PixelSpacing]
    if len(positions) > 1:
        diffs = np.diff(sorted(positions))
        z_spacing = float(np.median(np.abs(diffs)))
    else:
        z_spacing = float(getattr(first, "SliceThickness", 1.0))

    image.SetSpacing((pixel_spacing[1], pixel_spacing[0], z_spacing))
    image.SetOrigin(tuple(float(x) for x in first.ImagePositionPatient))
    image.SetDirection(_direction_from_orientation(first.ImageOrientationPatient))
    return image


def resample_to_reference(moving: sitk.Image, reference: sitk.Image, default_value: float = -1024) -> sitk.Image:
    return sitk.Resample(
        moving,
        reference,
        sitk.Transform(),
        sitk.sitkLinear,
        default_value,
        moving.GetPixelID(),
    )


def _segment_number_to_label(seg_ds: pydicom.Dataset) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for segment in getattr(seg_ds, "SegmentSequence", []):
        number = int(getattr(segment, "SegmentNumber"))
        label = str(getattr(segment, "SegmentLabel", "")).strip().lower()
        if label in LIVER_LABELS:
            mapping[number] = 1
        elif label in TUMOR_LABELS:
            mapping[number] = 2
    return mapping


def _frame_segment_number(frame_group: pydicom.Dataset) -> int | None:
    seq = getattr(frame_group, "SegmentIdentificationSequence", None)
    if not seq:
        return None
    return int(getattr(seq[0], "ReferencedSegmentNumber"))


def _frame_position(frame_group: pydicom.Dataset) -> tuple[float, float, float] | None:
    seq = getattr(frame_group, "PlanePositionSequence", None)
    if not seq:
        return None
    ipp = getattr(seq[0], "ImagePositionPatient", None)
    if ipp is None:
        return None
    return float(ipp[0]), float(ipp[1]), float(ipp[2])


def dicom_seg_to_label(seg_path: Path, reference: sitk.Image) -> sitk.Image:
    seg_file = next(seg_path.glob("*.dcm"), None)
    if seg_file is None:
        raise FileNotFoundError(f"No SEG DICOM found: {seg_path}")

    ds = pydicom.dcmread(str(seg_file), force=True)
    arr = ds.pixel_array.astype(np.uint8)  # (F, Y, X), binary frames
    if arr.ndim != 3:
        raise ValueError(f"Expected SEG pixel array with 3 dims, got {arr.shape}: {seg_file}")

    seg_to_label = _segment_number_to_label(ds)
    if not seg_to_label:
        raise ValueError(f"No usable Liver/Mass/Necrosis segments found in {seg_file}")

    out = np.zeros((reference.GetSize()[2], reference.GetSize()[1], reference.GetSize()[0]), dtype=np.uint8)
    frame_groups = getattr(ds, "PerFrameFunctionalGroupsSequence", None)
    if frame_groups is None:
        raise ValueError(f"SEG has no PerFrameFunctionalGroupsSequence: {seg_file}")
    if len(frame_groups) != arr.shape[0]:
        raise ValueError(f"SEG frame count mismatch: {len(frame_groups)} vs {arr.shape[0]}")

    # Process liver first and tumor second so tumor overwrites liver where they overlap.
    frame_order = []
    for frame_idx, fg in enumerate(frame_groups):
        seg_num = _frame_segment_number(fg)
        label = seg_to_label.get(seg_num) if seg_num is not None else None
        if label is not None:
            frame_order.append((label, frame_idx, fg))
    frame_order.sort(key=lambda x: x[0])

    for label, frame_idx, fg in frame_order:
        pos = _frame_position(fg)
        if pos is None:
            continue
        try:
            _, _, z = reference.TransformPhysicalPointToIndex(pos)
        except RuntimeError:
            continue
        if z < 0 or z >= out.shape[0]:
            continue
        mask = arr[frame_idx] > 0
        if mask.shape != out[z].shape:
            raise ValueError(f"SEG frame shape {mask.shape} does not match reference slice {out[z].shape}")
        out[z][mask] = label

    label_img = sitk.GetImageFromArray(out)
    label_img.CopyInformation(reference)
    return label_img


def source_sop_instance_uids_from_seg(seg_path: Path) -> set[str]:
    seg_file = next(seg_path.glob("*.dcm"), None)
    if seg_file is None:
        raise FileNotFoundError(f"No SEG DICOM found: {seg_path}")
    ds = pydicom.dcmread(str(seg_file), stop_before_pixels=True, force=True)
    uids: set[str] = set()
    for fg in getattr(ds, "PerFrameFunctionalGroupsSequence", []):
        for derivation in getattr(fg, "DerivationImageSequence", []):
            for source in getattr(derivation, "SourceImageSequence", []):
                uid = getattr(source, "ReferencedSOPInstanceUID", None)
                if uid:
                    uids.add(str(uid))
    return uids


def write_dataset_json(out_dir: Path, dataset_name: str, num_training: int) -> None:
    dataset_json = {
        "channel_names": {
            "0": "pre_ct",
            "1": "contrast_ct",
        },
        "labels": {
            "background": 0,
            "liver": 1,
            "tumor": 2,
        },
        "numTraining": num_training,
        "file_ending": ".nii.gz",
        "name": dataset_name,
        "description": "HCC-TACE-Seg first-pass 2-channel multiphase CT dataset",
        "reference": "HCC-TACE-Seg",
        "licence": "See source dataset terms",
        "release": "v1_202201",
    }
    (out_dir / "dataset.json").write_text(json.dumps(dataset_json, indent=2), encoding="utf-8")


def selected_rows(rows: list[dict[str, str]], cases: set[str] | None, include_review: bool) -> Iterable[dict[str, str]]:
    allowed = {"ready_2ch"}
    if include_review:
        allowed.add("review_2ch")
    for row in rows:
        if cases is not None and row["patient_id"] not in cases:
            continue
        if row["status"] in allowed:
            yield row


def convert_case(row: dict[str, str], out_dir: Path) -> None:
    case_id = row["patient_id"]
    images_tr = out_dir / "imagesTr"
    labels_tr = out_dir / "labelsTr"
    images_tr.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)

    source_sops = source_sop_instance_uids_from_seg(Path(row["label_seg_path"]))
    pre_ct = read_dicom_stack(Path(row["pre_ct_path"]))
    ref_ct = read_dicom_stack(Path(row["referenced_ct_path"]), source_sops or None)
    pre_resampled = resample_to_reference(pre_ct, ref_ct)
    label_img = dicom_seg_to_label(Path(row["label_seg_path"]), ref_ct)

    sitk.WriteImage(pre_resampled, str(images_tr / f"{case_id}_0000.nii.gz"), useCompression=True)
    sitk.WriteImage(ref_ct, str(images_tr / f"{case_id}_0001.nii.gz"), useCompression=True)
    sitk.WriteImage(label_img, str(labels_tr / f"{case_id}.nii.gz"), useCompression=True)

    label_arr = sitk.GetArrayFromImage(label_img)
    print(
        f"[{case_id}] wrote shape={label_arr.shape} "
        f"liver={(label_arr == 1).sum():,} tumor={(label_arr == 2).sum():,}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-csv", type=Path, default=DEFAULT_PLAN_CSV)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--cases", nargs="*", default=None)
    parser.add_argument("--include-review", action="store_true")
    parser.add_argument("--max-cases", type=int, default=None)
    args = parser.parse_args()

    rows = read_plan(args.plan_csv)
    cases = set(args.cases) if args.cases else None
    todo = list(selected_rows(rows, cases, args.include_review))
    if args.max_cases is not None:
        todo = todo[:args.max_cases]
    if not todo:
        raise RuntimeError("No cases selected for conversion")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    converted = 0
    failures: list[dict[str, str]] = []
    for row in todo:
        try:
            convert_case(row, args.out_dir)
            converted += 1
        except Exception as e:
            failures.append({"patient_id": row["patient_id"], "error": str(e)})
            print(f"[{row['patient_id']}] FAILED: {e}")

    write_dataset_json(args.out_dir, args.dataset_name, converted)
    if failures:
        (args.out_dir / "conversion_failures.json").write_text(
            json.dumps(failures, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    print(f"Converted {converted}/{len(todo)} cases to {args.out_dir}")
    if failures:
        print(f"Failures: {len(failures)} (see conversion_failures.json)")


if __name__ == "__main__":
    main()
