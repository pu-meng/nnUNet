"""
Plan HCC-TACE-Seg cases for a first-pass multiphase nnU-Net dataset.

The inventory step tells us all series and, for SEG objects, which CT series
they reference. This planner creates a conservative 2-channel mapping:

  channel 0: PRE / non-contrast CT from the same study when available
  channel 1: SEG-referenced CT series, used as the label reference space
  label    : DICOM SEG object

No files are converted here. This only writes a CSV/JSON mapping for review.

Example:
  python -m pumengyu.tools.hcc.plan_multiphase_cases \
    --inventory pumengyu/notes/data/hcc_series_inventory.csv \
    --out-csv pumengyu/notes/data/hcc_multiphase_case_plan.csv \
    --out-json pumengyu/notes/data/hcc_multiphase_case_plan_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_INVENTORY = Path("pumengyu/notes/data/hcc_series_inventory.csv")
DEFAULT_OUT_CSV = Path("pumengyu/notes/data/hcc_multiphase_case_plan.csv")
DEFAULT_OUT_JSON = Path("pumengyu/notes/data/hcc_multiphase_case_plan_summary.json")


CSV_FIELDS = [
    "patient_id",
    "status",
    "reason",
    "label_seg_path",
    "label_segment_labels",
    "label_study_folder",
    "label_series_folder",
    "referenced_ct_path",
    "referenced_ct_description",
    "referenced_ct_study_folder",
    "referenced_ct_series_folder",
    "pre_ct_path",
    "pre_ct_description",
    "pre_ct_study_folder",
    "pre_ct_series_folder",
]


def read_inventory(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def is_pre_ct(row: dict[str, str]) -> bool:
    if row["modality"] != "CT":
        return False
    desc = row["series_description"].upper()
    return "PRE" in desc


def choose_pre_ct(patient_cts: list[dict[str, str]], label_study_folder: str) -> dict[str, str] | None:
    same_study_pre = [r for r in patient_cts if r["study_folder"] == label_study_folder and is_pre_ct(r)]
    if same_study_pre:
        return sorted(same_study_pre, key=lambda r: (int_or_large(r["series_number"]), -int_or_zero(r["num_files"])))[0]

    any_pre = [r for r in patient_cts if is_pre_ct(r)]
    if any_pre:
        return sorted(any_pre, key=lambda r: (r["study_folder"], int_or_large(r["series_number"]), -int_or_zero(r["num_files"])))[0]

    return None


def int_or_large(value: str) -> int:
    try:
        return int(float(value))
    except Exception:
        return 10**9


def int_or_zero(value: str) -> int:
    try:
        return int(float(value))
    except Exception:
        return 0


def empty_row(patient_id: str, status: str, reason: str) -> dict[str, str]:
    row = {field: "" for field in CSV_FIELDS}
    row.update({"patient_id": patient_id, "status": status, "reason": reason})
    return row


def plan_cases(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    all_patients = sorted({r["patient_id"] for r in rows})
    by_patient: dict[str, list[dict[str, str]]] = {
        patient: [r for r in rows if r["patient_id"] == patient]
        for patient in all_patients
    }
    ct_by_uid = {
        r["series_instance_uid"]: r
        for r in rows
        if r["modality"] == "CT" and r["series_instance_uid"]
    }

    planned: list[dict[str, str]] = []
    for patient in all_patients:
        patient_rows = by_patient[patient]
        segs = [r for r in patient_rows if r["modality"] == "SEG"]
        cts = [r for r in patient_rows if r["modality"] == "CT"]

        if not segs:
            planned.append(empty_row(patient, "exclude", "missing SEG"))
            continue

        # Most cases have one SEG. If there are multiple, pick the one with the
        # richest segment list and keep the ambiguity visible in the reason.
        seg = sorted(segs, key=lambda r: (-len(r["segment_labels"].split("|")), r["study_folder"]))[0]
        reason_parts = []
        if len(segs) > 1:
            reason_parts.append(f"multiple SEG ({len(segs)}), selected one")

        ref_uids = [u for u in seg["referenced_series_uids"].split("|") if u]
        ref_ct = None
        for uid in ref_uids:
            candidate = ct_by_uid.get(uid)
            if candidate and candidate["patient_id"] == patient:
                ref_ct = candidate
                break

        if ref_ct is None:
            row = empty_row(patient, "exclude", "SEG referenced CT not found")
            row.update({
                "label_seg_path": seg["local_path"],
                "label_segment_labels": seg["segment_labels"],
                "label_study_folder": seg["study_folder"],
                "label_series_folder": seg["series_folder"],
            })
            planned.append(row)
            continue

        pre_ct = choose_pre_ct(cts, seg["study_folder"])
        if pre_ct is None:
            row = empty_row(patient, "exclude", "PRE CT not found")
            row.update({
                "label_seg_path": seg["local_path"],
                "label_segment_labels": seg["segment_labels"],
                "label_study_folder": seg["study_folder"],
                "label_series_folder": seg["series_folder"],
                "referenced_ct_path": ref_ct["local_path"],
                "referenced_ct_description": ref_ct["series_description"],
                "referenced_ct_study_folder": ref_ct["study_folder"],
                "referenced_ct_series_folder": ref_ct["series_folder"],
            })
            planned.append(row)
            continue

        if pre_ct["study_folder"] != seg["study_folder"]:
            reason_parts.append("PRE from different study")
        if "Mass" not in seg["segment_labels"].split("|"):
            reason_parts.append("SEG has no Mass label")

        planned.append({
            "patient_id": patient,
            "status": "ready_2ch" if not reason_parts else "review_2ch",
            "reason": "; ".join(reason_parts),
            "label_seg_path": seg["local_path"],
            "label_segment_labels": seg["segment_labels"],
            "label_study_folder": seg["study_folder"],
            "label_series_folder": seg["series_folder"],
            "referenced_ct_path": ref_ct["local_path"],
            "referenced_ct_description": ref_ct["series_description"],
            "referenced_ct_study_folder": ref_ct["study_folder"],
            "referenced_ct_series_folder": ref_ct["series_folder"],
            "pre_ct_path": pre_ct["local_path"],
            "pre_ct_description": pre_ct["series_description"],
            "pre_ct_study_folder": pre_ct["study_folder"],
            "pre_ct_series_folder": pre_ct["series_folder"],
        })

    return planned


def build_summary(planned: list[dict[str, str]]) -> dict[str, Any]:
    status_counts = Counter(r["status"] for r in planned)
    reason_counts = Counter(r["reason"] or "<none>" for r in planned)
    return {
        "num_patients": len(planned),
        "status_counts": dict(sorted(status_counts.items())),
        "reason_counts": dict(reason_counts.most_common()),
        "ready_or_review_patients": [
            r["patient_id"] for r in planned
            if r["status"] in {"ready_2ch", "review_2ch"}
        ],
        "excluded_patients": [
            {"patient_id": r["patient_id"], "reason": r["reason"]}
            for r in planned
            if r["status"] == "exclude"
        ],
    }


def write_csv(rows: list[dict[str, str]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    args = parser.parse_args()

    rows = read_inventory(args.inventory)
    planned = plan_cases(rows)
    summary = build_summary(planned)

    write_csv(planned, args.out_csv)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote {len(planned)} planned cases to {args.out_csv}")
    print(f"Wrote summary to {args.out_json}")
    print(json.dumps({
        "num_patients": summary["num_patients"],
        "status_counts": summary["status_counts"],
        "excluded_patients": summary["excluded_patients"][:10],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

