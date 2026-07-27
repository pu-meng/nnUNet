#!/usr/bin/env python3
"""Inspect an extracted MVTec AD dataset directory."""

from __future__ import annotations

import argparse
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


def count_images(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def find_category_dirs(root: Path) -> list[Path]:
    candidates = []
    for path in [root, *root.iterdir()] if root.exists() else []:
        if not path.is_dir():
            continue
        if path.name in EXPECTED_CATEGORIES and (path / "train").exists() and (path / "test").exists():
            candidates.append(path)
        else:
            for child in path.iterdir():
                if child.is_dir() and child.name in EXPECTED_CATEGORIES and (child / "train").exists():
                    candidates.append(child)
    return sorted(set(candidates), key=lambda p: p.name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Path to data/raw or extracted MVTec AD directory")
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    if not root.exists():
        print(f"Missing path: {root}")
        return 2

    categories = find_category_dirs(root)
    if not categories:
        print(f"No MVTec AD category directories found under: {root}")
        print("Expected category folders such as transistor/train and transistor/test.")
        return 1

    print(f"Root: {root}")
    print(f"Found categories: {len(categories)}")
    print()
    print(f"{'category':<14} {'train':>8} {'test':>8} {'masks':>8} status")
    print("-" * 52)

    for category in categories:
        train_count = count_images(category / "train")
        test_count = count_images(category / "test")
        mask_count = count_images(category / "ground_truth")
        status = "ok" if train_count and test_count else "check"
        print(f"{category.name:<14} {train_count:>8} {test_count:>8} {mask_count:>8} {status}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

