"""
Quality-control a converted HCC nnU-Net raw dataset.

Checks labelsTr for basic sanity before plan_and_preprocess:
  - missing image channels
  - empty liver or tumor
  - suspicious tumor/liver volume ratio

Example:
  python -m pumengyu.tools.hcc.qc_converted_dataset \
    --dataset-dir /home/PuMengYu/nnUNet_workspace/raw/Dataset013_HCCMultiPhase \
    --out-csv pumengyu/notes/data/hcc_converted_qc.csv \
    --out-json pumengyu/notes/data/hcc_converted_qc_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import SimpleITK as sitk


DEFAULT_DATASET_DIR = Path("/home/PuMengYu/nnUNet_workspace/raw/Dataset013_HCCMultiPhase")
DEFAULT_OUT_CSV = Path("pumengyu/notes/data/hcc_converted_qc.csv")
DEFAULT_OUT_JSON = Path("pumengyu/notes/data/hcc_converted_qc_summary.json")

CSV_FIELDS = [
    "case_id",
    "status",
    "reason",
    "shape_zyx",
    "liver_voxels",
    "tumor_voxels",
    "tumor_liver_ratio",
    "missing_channels",
]


def expected_channels(dataset_dir: Path) -> list[str]:
    dataset_json = dataset_dir / "dataset.json"
    if not dataset_json.exists():
        return ["0000", "0001"]
    data = json.loads(dataset_json.read_text(encoding="utf-8"))
    channel_names = data.get("channel_names", {})
    return [f"{int(channel):04d}" for channel in sorted(channel_names, key=lambda x: int(x))]


def qc_dataset(dataset_dir: Path) -> list[dict[str, str]]:
    labels_dir = dataset_dir / "labelsTr"
    images_dir = dataset_dir / "imagesTr"
    if not labels_dir.exists():
        raise FileNotFoundError(f"labelsTr not found: {labels_dir}")
    if not images_dir.exists():
        raise FileNotFoundError(f"imagesTr not found: {images_dir}")

    channels = expected_channels(dataset_dir)
    rows: list[dict[str, str]] = []
    for label_file in sorted(labels_dir.glob("*.nii.gz")):
        case_id = label_file.name.removesuffix(".nii.gz")
        missing_channels = [
            channel for channel in channels
            if not (images_dir / f"{case_id}_{channel}.nii.gz").exists()
        ]

        seg = sitk.GetArrayFromImage(sitk.ReadImage(str(label_file)))
        liver = int((seg == 1).sum())
        tumor = int((seg == 2).sum())
        ratio = float(tumor / liver) if liver > 0 else float("inf") if tumor > 0 else 0.0

        reasons = []
        reasons.extend(f"missing_{channel}" for channel in missing_channels)
        if liver == 0:
            reasons.append("empty_liver")
        if tumor == 0:
            reasons.append("empty_tumor")
        if liver > 0 and ratio > 1.5:
            reasons.append("tumor_liver_ratio_gt_1.5")

        rows.append({
            "case_id": case_id,
            "status": "ok" if not reasons else "review",
            "reason": "|".join(reasons),
            "shape_zyx": "x".join(str(x) for x in seg.shape),
            "liver_voxels": str(liver),
            "tumor_voxels": str(tumor),
            "tumor_liver_ratio": f"{ratio:.6f}",
            "missing_channels": "|".join(missing_channels),
        })
    return rows


def build_summary(rows: list[dict[str, str]]) -> dict[str, Any]:
    status_counts = Counter(r["status"] for r in rows)
    reason_counts = Counter()
    for row in rows:
        for reason in row["reason"].split("|"):
            if reason:
                reason_counts[reason] += 1
    return {
        "num_cases": len(rows),
        "status_counts": dict(sorted(status_counts.items())),
        "reason_counts": dict(sorted(reason_counts.items())),
        "review_cases": [
            {
                "case_id": r["case_id"],
                "reason": r["reason"],
                "liver_voxels": int(r["liver_voxels"]),
                "tumor_voxels": int(r["tumor_voxels"]),
                "tumor_liver_ratio": float(r["tumor_liver_ratio"]),
            }
            for r in rows
            if r["status"] == "review"
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
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    args = parser.parse_args()

    rows = qc_dataset(args.dataset_dir)
    summary = build_summary(rows)
    write_csv(rows, args.out_csv)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote QC CSV to {args.out_csv}")
    print(f"Wrote QC summary to {args.out_json}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
