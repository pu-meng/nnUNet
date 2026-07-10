"""
Move selected converted HCC cases out of imagesTr/labelsTr without deleting them.

This is for QC exclusions before nnUNetv2_plan_and_preprocess. Files are moved
to _excluded_qc/imagesTr and _excluded_qc/labelsTr, and dataset.json numTraining
is updated to the remaining number of labelsTr cases.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


DEFAULT_DATASET_DIR = Path("/home/PuMengYu/nnUNet_workspace/raw/Dataset013_HCCMultiPhase")


def move_case(dataset_dir: Path, case_id: str) -> None:
    images_tr = dataset_dir / "imagesTr"
    labels_tr = dataset_dir / "labelsTr"
    excluded_images = dataset_dir / "_excluded_qc" / "imagesTr"
    excluded_labels = dataset_dir / "_excluded_qc" / "labelsTr"
    excluded_images.mkdir(parents=True, exist_ok=True)
    excluded_labels.mkdir(parents=True, exist_ok=True)

    for channel in ("0000", "0001"):
        src = images_tr / f"{case_id}_{channel}.nii.gz"
        if src.exists():
            dst = excluded_images / src.name
            if dst.exists():
                dst.unlink()
            shutil.move(str(src), str(dst))

    src_label = labels_tr / f"{case_id}.nii.gz"
    if src_label.exists():
        dst_label = excluded_labels / src_label.name
        if dst_label.exists():
            dst_label.unlink()
        shutil.move(str(src_label), str(dst_label))


def update_dataset_json(dataset_dir: Path) -> None:
    dataset_json_path = dataset_dir / "dataset.json"
    if not dataset_json_path.exists():
        return
    data = json.loads(dataset_json_path.read_text(encoding="utf-8"))
    data["numTraining"] = len(list((dataset_dir / "labelsTr").glob("*.nii.gz")))
    dataset_json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--cases", nargs="+", required=True)
    args = parser.parse_args()

    for case_id in args.cases:
        move_case(args.dataset_dir, case_id)
        print(f"moved {case_id} to _excluded_qc")
    update_dataset_json(args.dataset_dir)
    print(f"remaining labelsTr: {len(list((args.dataset_dir / 'labelsTr').glob('*.nii.gz')))}")


if __name__ == "__main__":
    main()

