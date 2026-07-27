#!/usr/bin/env python3
"""Save a quick MVTec AD image/mask sample sheet."""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
from PIL import Image


def load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def load_mask(path: Path | None, size: tuple[int, int]) -> Image.Image:
    if path is None or not path.exists():
        return Image.new("L", size, 0)
    return Image.open(path).convert("L")


def collect_samples(category_dir: Path, defect_type: str, limit: int, seed: int):
    image_dir = category_dir / "test" / defect_type
    if not image_dir.exists():
        raise SystemExit(f"Missing defect directory: {image_dir}")

    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    random.Random(seed).shuffle(images)
    samples = []
    for image_path in images[:limit]:
        mask_path = None
        if defect_type != "good":
            candidate = category_dir / "ground_truth" / defect_type / f"{image_path.stem}_mask.png"
            mask_path = candidate if candidate.exists() else None
        samples.append((image_path, mask_path))
    return samples


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/home/PuMengYu/MVTec_AD/data/raw"))
    parser.add_argument("--category", default="transistor")
    parser.add_argument("--defect-type", default=None)
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/PuMengYu/MVTec_AD/outputs/sample_sheet.png"),
    )
    args = parser.parse_args()

    category_dir = args.root.expanduser().resolve() / args.category
    if not category_dir.exists():
        raise SystemExit(f"Missing category: {category_dir}")

    defect_type = args.defect_type
    if defect_type is None:
        defect_types = sorted(
            p.name for p in (category_dir / "test").iterdir() if p.is_dir() and p.name != "good"
        )
        if not defect_types:
            raise SystemExit(f"No anomalous defect types found for {args.category}")
        defect_type = defect_types[0]

    samples = collect_samples(category_dir, defect_type, args.limit, args.seed)
    if not samples:
        raise SystemExit(f"No samples found for {args.category}/{defect_type}")

    cols = len(samples)
    fig, axes = plt.subplots(2, cols, figsize=(3 * cols, 6), squeeze=False)
    for col, (image_path, mask_path) in enumerate(samples):
        image = load_rgb(image_path)
        mask = load_mask(mask_path, image.size)

        axes[0][col].imshow(image)
        axes[0][col].set_title(image_path.stem, fontsize=9)
        axes[0][col].axis("off")

        axes[1][col].imshow(image)
        axes[1][col].imshow(mask, cmap="Reds", alpha=0.45)
        axes[1][col].axis("off")

    fig.suptitle(f"{args.category} / {defect_type}", fontsize=12)
    fig.tight_layout()
    args.output.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    print(f"Wrote: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
