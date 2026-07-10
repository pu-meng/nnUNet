"""
Convert HCC-TACE-Seg SEG-referenced CT series to a clean nnU-Net dataset.

This is the conservative single-channel conversion:

  0000 = CT series explicitly referenced by the DICOM-SEG object
  label = DICOM-SEG converted to background=0, liver=1, tumor=2

Other baseline CT series, follow-up CT series, portal vein, and abdominal aorta
segments are not used for training labels.

Example smoke test:
  python -m pumengyu.tools.hcc.convert_referenced_ct \
    --inventory pumengyu/notes/data/hcc_series_inventory.csv \
    --out-dir /tmp/Dataset013_HCCReferencedCT_smoke \
    --cases HCC_001

Real output:
  python -m pumengyu.tools.hcc.convert_referenced_ct \
    --inventory pumengyu/notes/data/hcc_series_inventory.csv \
    --out-dir /home/PuMengYu/nnUNet_workspace/raw/Dataset013_HCCReferencedCT
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import pydicom
import SimpleITK as sitk

from pumengyu.tools.hcc.convert_multiphase import (
    dicom_seg_to_label,
    read_dicom_stack,
    source_sop_instance_uids_from_seg,
)


DEFAULT_INVENTORY = Path("pumengyu/notes/data/hcc_series_inventory.csv")
DEFAULT_DATASET_NAME = "Dataset013_HCCReferencedCT"

METADATA_FIELDS = [
    "case_id",
    "status",
    "reason",
    "image_source_path",
    "label_seg_path",
    "study_folder",
    "image_series_folder",
    "label_series_folder",
    "study_date",
    "referenced_ct_description",
    "referenced_ct_series_uid",
    "seg_series_uid",
    "segment_labels",
    "image_shape_zyx",
    "label_values",
    "liver_voxels",
    "tumor_voxels",
]


def read_inventory(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def first_dicom_header(series_path: Path) -> pydicom.Dataset:
    dicom_file = next(series_path.glob("*.dcm"), None)
    if dicom_file is None:
        raise FileNotFoundError(f"No DICOM files found in {series_path}")
    return pydicom.dcmread(str(dicom_file), stop_before_pixels=True, force=True)


def build_case_plan(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    patients = sorted({r["patient_id"] for r in rows})
    by_patient = {
        patient: [r for r in rows if r["patient_id"] == patient]
        for patient in patients
    }

    planned: list[dict[str, str]] = []
    for patient in patients:
        patient_rows = by_patient[patient]
        segs = [r for r in patient_rows if r["modality"] == "SEG"]
        cts_by_uid = {
            r["series_instance_uid"]: r
            for r in patient_rows
            if r["modality"] == "CT" and r["series_instance_uid"]
        }

        row = {field: "" for field in METADATA_FIELDS}
        row["case_id"] = patient

        if not segs:
            row.update({"status": "exclude", "reason": "missing SEG"})
            planned.append(row)
            continue

        seg = sorted(segs, key=lambda r: r["study_folder"])[0]
        ref_uids = [u for u in seg["referenced_series_uids"].split("|") if u]
        ref_ct = next((cts_by_uid[uid] for uid in ref_uids if uid in cts_by_uid), None)
        if ref_ct is None:
            row.update({
                "status": "exclude",
                "reason": "SEG referenced CT not found",
                "label_seg_path": seg["local_path"],
                "study_folder": seg["study_folder"],
                "label_series_folder": seg["series_folder"],
                "seg_series_uid": seg["series_instance_uid"],
                "segment_labels": seg["segment_labels"],
            })
            planned.append(row)
            continue

        row.update({
            "status": "ready",
            "image_source_path": ref_ct["local_path"],
            "label_seg_path": seg["local_path"],
            "study_folder": ref_ct["study_folder"],
            "image_series_folder": ref_ct["series_folder"],
            "label_series_folder": seg["series_folder"],
            "referenced_ct_description": ref_ct["series_description"],
            "referenced_ct_series_uid": ref_ct["series_instance_uid"],
            "seg_series_uid": seg["series_instance_uid"],
            "segment_labels": seg["segment_labels"],
        })
        try:
            ds = first_dicom_header(Path(ref_ct["local_path"]))
            row["study_date"] = str(getattr(ds, "StudyDate", ""))
        except Exception as e:
            row["reason"] = f"study date read failed: {e}"
        planned.append(row)

    return planned


def selected_rows(rows: list[dict[str, str]], cases: set[str] | None) -> Iterable[dict[str, str]]:
    for row in rows:
        if row["status"] != "ready":
            continue
        if cases is not None and row["case_id"] not in cases:
            continue
        yield row


def write_dataset_json(out_dir: Path, dataset_name: str, num_training: int) -> None:
    dataset_json = {
        "channel_names": {
            "0": "CT",
        },
        "labels": {
            "background": 0,
            "liver": 1,
            "tumor": 2,
        },
        "numTraining": num_training,
        "file_ending": ".nii.gz",
        "name": dataset_name,
        "description": "HCC-TACE-Seg single-channel SEG-referenced CT dataset",
        "reference": "HCC-TACE-Seg",
        "licence": "See source dataset terms",
        "release": "v1_202201",
    }
    (out_dir / "dataset.json").write_text(json.dumps(dataset_json, indent=2), encoding="utf-8")


def convert_case(row: dict[str, str], out_dir: Path) -> dict[str, str]:
    case_id = row["case_id"]
    images_tr = out_dir / "imagesTr"
    labels_tr = out_dir / "labelsTr"
    images_tr.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)

    source_sops = source_sop_instance_uids_from_seg(Path(row["label_seg_path"]))
    ref_ct = read_dicom_stack(Path(row["image_source_path"]), source_sops or None)
    label_img = dicom_seg_to_label(Path(row["label_seg_path"]), ref_ct)

    sitk.WriteImage(ref_ct, str(images_tr / f"{case_id}_0000.nii.gz"), useCompression=True)
    sitk.WriteImage(label_img, str(labels_tr / f"{case_id}.nii.gz"), useCompression=True)

    label_arr = sitk.GetArrayFromImage(label_img)
    values = sorted(int(v) for v in set(label_arr.ravel().tolist()))
    row = dict(row)
    row.update({
        "image_shape_zyx": "x".join(str(x) for x in label_arr.shape),
        "label_values": "|".join(str(v) for v in values),
        "liver_voxels": str(int((label_arr == 1).sum())),
        "tumor_voxels": str(int((label_arr == 2).sum())),
    })
    print(
        f"[{case_id}] wrote shape={row['image_shape_zyx']} values={row['label_values']} "
        f"liver={row['liver_voxels']} tumor={row['tumor_voxels']}"
    )
    return row


def write_metadata(rows: list[dict[str, str]], out_dir: Path) -> None:
    csv_path = out_dir / "case_metadata.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=METADATA_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    summary: dict[str, Any] = {
        "num_rows": len(rows),
        "status_counts": dict(Counter(r["status"] for r in rows)),
        "reason_counts": dict(Counter(r["reason"] or "<none>" for r in rows)),
        "cases": rows,
    }
    (out_dir / "case_metadata.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--cases", nargs="*", default=None)
    parser.add_argument("--max-cases", type=int, default=None)
    args = parser.parse_args()

    planned = build_case_plan(read_inventory(args.inventory))
    cases = set(args.cases) if args.cases else None
    todo = list(selected_rows(planned, cases))
    if args.max_cases is not None:
        todo = todo[:args.max_cases]
    if not todo:
        raise RuntimeError("No ready cases selected for conversion")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    converted_rows: list[dict[str, str]] = []
    failures: list[dict[str, str]] = []
    for row in todo:
        try:
            converted_rows.append(convert_case(row, args.out_dir))
        except Exception as e:
            failed = dict(row)
            failed.update({"status": "failed", "reason": str(e)})
            failures.append(failed)
            print(f"[{row['case_id']}] FAILED: {e}")

    all_metadata = converted_rows + failures + [r for r in planned if r["status"] == "exclude"]
    write_dataset_json(args.out_dir, args.dataset_name, len(converted_rows))
    write_metadata(all_metadata, args.out_dir)
    if failures:
        (args.out_dir / "conversion_failures.json").write_text(
            json.dumps(failures, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    print(f"Converted {len(converted_rows)}/{len(todo)} selected cases to {args.out_dir}")
    if failures:
        print(f"Failures: {len(failures)} (see conversion_failures.json)")


if __name__ == "__main__":
    main()
