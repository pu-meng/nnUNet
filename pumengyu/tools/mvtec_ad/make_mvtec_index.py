#!/usr/bin/env python3
"""Create a CSV index for an extracted MVTec AD dataset."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
EXPECTED_CATEGORIES = {
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
}


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS


def find_categories(root: Path) -> list[Path]:
    categories = []
    for path in [root, *root.iterdir()] if root.exists() else []:
        if path.is_dir() and path.name in EXPECTED_CATEGORIES and (path / "train").exists():
            categories.append(path)
        elif path.is_dir():
            categories.extend(
                child
                for child in path.iterdir()
                if child.is_dir()
                and child.name in EXPECTED_CATEGORIES
                and (child / "train").exists()
            )
    return sorted(set(categories), key=lambda p: p.name)


def mask_for_image(category: Path, defect_type: str, image_path: Path) -> Path | None:
    if defect_type == "good":
        return None
    mask_name = f"{image_path.stem}_mask.png"
    mask_path = category / "ground_truth" / defect_type / mask_name
    return mask_path if mask_path.exists() else None


def iter_rows(root: Path):
    for category in find_categories(root):
        for image_path in sorted((category / "train" / "good").glob("*")):
            if not is_image(image_path):
                continue
            yield {
                "category": category.name,
                "split": "train",
                "defect_type": "good",
                "label": 0,
                "image_path": str(image_path),
                "mask_path": "",
            }

        test_dir = category / "test"
        for defect_dir in sorted(p for p in test_dir.iterdir() if p.is_dir()):
            label = 0 if defect_dir.name == "good" else 1
            for image_path in sorted(defect_dir.glob("*")):
                if not is_image(image_path):
                    continue
                mask_path = mask_for_image(category, defect_dir.name, image_path)
                yield {
                    "category": category.name,
                    "split": "test",
                    "defect_type": defect_dir.name,
                    "label": label,
                    "image_path": str(image_path),
                    "mask_path": str(mask_path) if mask_path else "",
                }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/home/PuMengYu/MVTec_AD/data/raw"),
        help="Extracted MVTec AD root.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/PuMengYu/MVTec_AD/data/processed/mvtec_index.csv"),
        help="Output CSV path.",
    )
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    rows = list(iter_rows(root))
    if not rows:
        raise SystemExit(f"No images found under {root}")

    fieldnames = ["category", "split", "defect_type", "label", "image_path", "mask_path"]
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    train_count = sum(row["split"] == "train" for row in rows)
    test_count = sum(row["split"] == "test" for row in rows)
    anomaly_count = sum(row["label"] == 1 for row in rows)
    print(f"Wrote: {output}")
    print(f"Rows: {len(rows)} train={train_count} test={test_count} anomaly_test={anomaly_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
