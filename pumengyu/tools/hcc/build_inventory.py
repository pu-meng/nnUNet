"""
Build a DICOM series inventory for HCC-TACE-Seg.

This is intentionally read-only with respect to the HCC source tree. It scans
patient/study/series folders, reads lightweight DICOM headers, and writes a CSV
plus JSON summary that downstream conversion code can use to decide which CT
series and DICOM SEG objects belong together.

Example:
  python -m pumengyu.tools.hcc.build_inventory \
    --hcc-root /home/PuMengYu/HCC/HCC-TACE-Seg_v1_202201 \
    --out-csv pumengyu/notes/data/hcc_series_inventory.csv \
    --out-json pumengyu/notes/data/hcc_series_inventory_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pydicom


DEFAULT_HCC_ROOT = Path("/home/PuMengYu/HCC/HCC-TACE-Seg_v1_202201")
DEFAULT_OUT_CSV = Path("pumengyu/notes/data/hcc_series_inventory.csv")
DEFAULT_OUT_JSON = Path("pumengyu/notes/data/hcc_series_inventory_summary.json")


CSV_FIELDS = [
    "patient_id",
    "study_folder",
    "series_folder",
    "num_files",
    "modality",
    "series_description",
    "protocol_name",
    "study_instance_uid",
    "series_instance_uid",
    "series_number",
    "image_type",
    "rows",
    "columns",
    "slice_thickness",
    "pixel_spacing",
    "spacing_between_slices",
    "referenced_series_uids",
    "segment_labels",
    "local_path",
]


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return "|".join(_safe_str(v) for v in value)
    return str(value).replace("\n", " ").replace("\r", " ").strip()


def _get_sequence(ds: pydicom.Dataset, name: str):
    return getattr(ds, name, None) or []


def _extract_referenced_series_uids(ds: pydicom.Dataset) -> list[str]:
    uids: set[str] = set()

    for item in _get_sequence(ds, "ReferencedSeriesSequence"):
        uid = getattr(item, "SeriesInstanceUID", None)
        if uid:
            uids.add(str(uid))

    for study_item in _get_sequence(ds, "ReferencedStudySequence"):
        for series_item in _get_sequence(study_item, "ReferencedSeriesSequence"):
            uid = getattr(series_item, "SeriesInstanceUID", None)
            if uid:
                uids.add(str(uid))

    for frame_item in _get_sequence(ds, "PerFrameFunctionalGroupsSequence"):
        for derivation in _get_sequence(frame_item, "DerivationImageSequence"):
            for source in _get_sequence(derivation, "SourceImageSequence"):
                for purpose in _get_sequence(source, "PurposeOfReferenceCodeSequence"):
                    _ = purpose  # keeps pydicom from lazily warning in some edge cases
                uid = getattr(source, "ReferencedSOPInstanceUID", None)
                if uid:
                    # SOP UID is not the series UID, but keeping this branch makes
                    # it clear that frame-level references exist. We do not output
                    # SOP UIDs because downstream series matching should use
                    # SeriesInstanceUID.
                    pass

    return sorted(uids)


def _extract_segment_labels(ds: pydicom.Dataset) -> list[str]:
    labels: list[str] = []
    for segment in _get_sequence(ds, "SegmentSequence"):
        label = getattr(segment, "SegmentLabel", None)
        if label:
            labels.append(str(label))
    return labels


def _read_header(dicom_file: Path) -> pydicom.Dataset:
    return pydicom.dcmread(str(dicom_file), stop_before_pixels=True, force=True)


def iter_series_dirs(hcc_root: Path):
    data_root = hcc_root / "hcc_tace_seg"
    if not data_root.exists():
        raise FileNotFoundError(f"HCC data directory not found: {data_root}")

    for patient_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        for study_dir in sorted(p for p in patient_dir.iterdir() if p.is_dir()):
            for series_dir in sorted(p for p in study_dir.iterdir() if p.is_dir()):
                dicom_files = sorted(series_dir.glob("*.dcm"))
                if dicom_files:
                    yield patient_dir.name, study_dir.name, series_dir.name, series_dir, dicom_files


def build_inventory(hcc_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for patient_id, study_folder, series_folder, series_dir, dicom_files in iter_series_dirs(hcc_root):
        try:
            ds = _read_header(dicom_files[0])
        except Exception as e:
            rows.append({
                "patient_id": patient_id,
                "study_folder": study_folder,
                "series_folder": series_folder,
                "num_files": str(len(dicom_files)),
                "modality": "READ_ERROR",
                "series_description": str(e),
                "protocol_name": "",
                "study_instance_uid": "",
                "series_instance_uid": "",
                "series_number": "",
                "image_type": "",
                "rows": "",
                "columns": "",
                "slice_thickness": "",
                "pixel_spacing": "",
                "spacing_between_slices": "",
                "referenced_series_uids": "",
                "segment_labels": "",
                "local_path": str(series_dir),
            })
            continue

        rows.append({
            "patient_id": patient_id,
            "study_folder": study_folder,
            "series_folder": series_folder,
            "num_files": str(len(dicom_files)),
            "modality": _safe_str(getattr(ds, "Modality", "")),
            "series_description": _safe_str(getattr(ds, "SeriesDescription", "")),
            "protocol_name": _safe_str(getattr(ds, "ProtocolName", "")),
            "study_instance_uid": _safe_str(getattr(ds, "StudyInstanceUID", "")),
            "series_instance_uid": _safe_str(getattr(ds, "SeriesInstanceUID", "")),
            "series_number": _safe_str(getattr(ds, "SeriesNumber", "")),
            "image_type": _safe_str(getattr(ds, "ImageType", "")),
            "rows": _safe_str(getattr(ds, "Rows", "")),
            "columns": _safe_str(getattr(ds, "Columns", "")),
            "slice_thickness": _safe_str(getattr(ds, "SliceThickness", "")),
            "pixel_spacing": _safe_str(getattr(ds, "PixelSpacing", "")),
            "spacing_between_slices": _safe_str(getattr(ds, "SpacingBetweenSlices", "")),
            "referenced_series_uids": "|".join(_extract_referenced_series_uids(ds)),
            "segment_labels": "|".join(_extract_segment_labels(ds)),
            "local_path": str(series_dir),
        })
    return rows


def build_summary(rows: list[dict[str, str]]) -> dict[str, Any]:
    patients = sorted({r["patient_id"] for r in rows})
    modality_counts = Counter(r["modality"] for r in rows)
    patient_counts: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        patient_counts[row["patient_id"]][row["modality"]] += 1

    missing_seg = [
        patient for patient in patients
        if patient_counts[patient].get("SEG", 0) == 0
    ]
    patients_with_multiple_studies = sorted({
        r["patient_id"] for r in rows
        if len({x["study_folder"] for x in rows if x["patient_id"] == r["patient_id"]}) > 1
    })

    return {
        "num_patients": len(patients),
        "num_series": len(rows),
        "modality_counts": dict(sorted(modality_counts.items())),
        "num_patients_missing_seg": len(missing_seg),
        "patients_missing_seg": missing_seg,
        "num_patients_with_multiple_studies": len(patients_with_multiple_studies),
        "patients_with_multiple_studies": patients_with_multiple_studies,
        "per_patient_modality_counts": {
            patient: dict(sorted(counts.items()))
            for patient, counts in sorted(patient_counts.items())
        },
    }


def write_csv(rows: list[dict[str, str]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_json(summary: dict[str, Any], out_json: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hcc-root", type=Path, default=DEFAULT_HCC_ROOT)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    args = parser.parse_args()

    rows = build_inventory(args.hcc_root)
    summary = build_summary(rows)
    write_csv(rows, args.out_csv)
    write_json(summary, args.out_json)

    print(f"Wrote {len(rows)} series rows to {args.out_csv}")
    print(f"Wrote summary to {args.out_json}")
    print(json.dumps({
        "num_patients": summary["num_patients"],
        "num_series": summary["num_series"],
        "modality_counts": summary["modality_counts"],
        "num_patients_missing_seg": summary["num_patients_missing_seg"],
        "num_patients_with_multiple_studies": summary["num_patients_with_multiple_studies"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

