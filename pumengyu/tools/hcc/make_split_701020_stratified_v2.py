"""
Create a non-destructive HCCReferencedCT 70/10/21 stratified split v2.

This script does NOT overwrite nnU-Net's active splits_final.json. It writes
sidecar v2 files that can be applied after currently running HCC trainers finish.

Policy:
- Use corrected tumor burden ratio: tumor voxels / all foreground liver+tumor voxels.
- Stratify by corrected ratio into tiny/small/medium.
- HCC_065 and HCC_075 are high-burden review/extreme cases and are excluded from
  test; they are kept in train.
- Test is selected before validation and must not be used for model selection.

Run dry first:
    python -m pumengyu.tools.hcc.make_split_701020_stratified_v2

Write sidecar files:
    python -m pumengyu.tools.hcc.make_split_701020_stratified_v2 --write
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np


SEED = 42
DATASET = "Dataset013_HCCReferencedCT"
ROOT = Path("/home/PuMengYu/nnUNet_workspace")
RAW_DIR = ROOT / "raw" / DATASET
PREPROC_DIR = ROOT / "preprocessed" / DATASET
SPLITS_V2_PATH = PREPROC_DIR / "splits_final_701020_stratified_v2.json"
INFO_V2_PATH = PREPROC_DIR / "split_info_701020_stratified_v2.json"
NOTES_CASES_PATH = Path("/home/PuMengYu/nnUNet/pumengyu/notes/data/hcc_split_701020_stratified_v2_cases.csv")

REVIEW_EXTREME_CASES = {"HCC_065", "HCC_075"}
TEST_QUOTAS = {"tiny": 7, "small": 8, "medium": 6}
VAL_QUOTAS = {"tiny": 3, "small": 4, "medium": 3}


def case_id_from_label(path: Path) -> str:
    return path.name.removesuffix(".nii.gz")


def ratio_bin(ratio: float) -> str:
    if ratio < 0.03:
        return "tiny"
    if ratio < 0.15:
        return "small"
    if ratio < 0.60:
        return "medium"
    return "extreme"


def load_case_stats() -> dict[str, dict[str, object]]:
    stats: dict[str, dict[str, object]] = {}
    for label_path in sorted((RAW_DIR / "labelsTr").glob("HCC_*.nii.gz")):
        case = case_id_from_label(label_path)
        seg = nib.load(str(label_path)).get_fdata().astype(np.uint8)
        tumor_voxels = int((seg == 2).sum())
        foreground_voxels = int((seg > 0).sum())
        label1_liver_voxels = int((seg == 1).sum())
        ratio = tumor_voxels / foreground_voxels if foreground_voxels else 0.0
        stats[case] = {
            "case": case,
            "tumor_voxels": tumor_voxels,
            "foreground_voxels": foreground_voxels,
            "label1_liver_voxels": label1_liver_voxels,
            "corrected_tumor_foreground_ratio": ratio,
            "bin": ratio_bin(ratio),
            "review_extreme": case in REVIEW_EXTREME_CASES,
        }
    return stats


def evenly_pick(cases: list[str], n: int, stats: dict[str, dict[str, object]], seed: int, salt: str) -> list[str]:
    """Pick n cases spread across the within-bin ratio range."""
    if n > len(cases):
        raise RuntimeError(f"Cannot pick {n} from {len(cases)} cases")
    rng = random.Random(f"{seed}:{salt}")
    cases = cases[:]
    rng.shuffle(cases)
    cases.sort(key=lambda c: float(stats[c]["corrected_tumor_foreground_ratio"]))
    if n == 0:
        return []
    if n == len(cases):
        return sorted(cases)
    positions = np.linspace(0, len(cases) - 1, n)
    selected_indices: list[int] = []
    used: set[int] = set()
    for pos in positions:
        idx = int(round(float(pos)))
        if idx in used:
            for delta in range(1, len(cases)):
                for cand in (idx - delta, idx + delta):
                    if 0 <= cand < len(cases) and cand not in used:
                        idx = cand
                        break
                else:
                    continue
                break
        used.add(idx)
        selected_indices.append(idx)
    return sorted(cases[i] for i in selected_indices)


def make_split(stats: dict[str, dict[str, object]], seed: int) -> tuple[list[str], list[str], list[str]]:
    eligible = [c for c, s in stats.items() if not bool(s["review_extreme"])]
    grouped: dict[str, list[str]] = defaultdict(list)
    for case in eligible:
        grouped[str(stats[case]["bin"])].append(case)

    test: list[str] = []
    for bin_name, quota in TEST_QUOTAS.items():
        test.extend(evenly_pick(grouped[bin_name], quota, stats, seed, f"test:{bin_name}"))

    remaining = [c for c in eligible if c not in set(test)]
    grouped_remaining: dict[str, list[str]] = defaultdict(list)
    for case in remaining:
        grouped_remaining[str(stats[case]["bin"])].append(case)

    val: list[str] = []
    for bin_name, quota in VAL_QUOTAS.items():
        val.extend(evenly_pick(grouped_remaining[bin_name], quota, stats, seed, f"val:{bin_name}"))

    train = sorted(c for c in stats if c not in set(test) and c not in set(val))
    return sorted(train), sorted(val), sorted(test)


def subset_summary(name: str, cases: list[str], stats: dict[str, dict[str, object]]) -> dict[str, object]:
    ratios = [float(stats[c]["corrected_tumor_foreground_ratio"]) for c in cases]
    bins = Counter(str(stats[c]["bin"]) for c in cases)
    review = sorted(c for c in cases if bool(stats[c]["review_extreme"]))
    return {
        "name": name,
        "n": len(cases),
        "mean_corrected_tumor_foreground_ratio": round(sum(ratios) / len(ratios), 6),
        "min_corrected_tumor_foreground_ratio": round(min(ratios), 6),
        "max_corrected_tumor_foreground_ratio": round(max(ratios), 6),
        "bins": dict(sorted(bins.items())),
        "review_extreme_cases": review,
        "cases": cases,
    }


def write_case_csv(train: list[str], val: list[str], test: list[str], stats: dict[str, dict[str, object]]) -> None:
    split_by_case = {c: "train" for c in train} | {c: "val" for c in val} | {c: "test" for c in test}
    lines = [
        "case_id,split,bin,corrected_tumor_foreground_ratio,tumor_voxels,foreground_voxels,label1_liver_voxels,review_extreme"
    ]
    for case in sorted(stats):
        s = stats[case]
        lines.append(
            ",".join(
                [
                    case,
                    split_by_case[case],
                    str(s["bin"]),
                    f"{float(s['corrected_tumor_foreground_ratio']):.6f}",
                    str(s["tumor_voxels"]),
                    str(s["foreground_voxels"]),
                    str(s["label1_liver_voxels"]),
                    str(s["review_extreme"]),
                ]
            )
        )
    NOTES_CASES_PATH.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="Write v2 sidecar split files")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    stats = load_case_stats()
    train, val, test = make_split(stats, args.seed)

    if len(train) != 70 or len(val) != 10 or len(test) != 21:
        raise RuntimeError(f"Unexpected sizes: train={len(train)} val={len(val)} test={len(test)}")
    if REVIEW_EXTREME_CASES & set(test):
        raise RuntimeError(f"Review/extreme cases leaked into test: {REVIEW_EXTREME_CASES & set(test)}")

    print("HCCReferencedCT 70/10/21 stratified v2")
    print(f"seed: {args.seed}")
    print("corrected ratio: tumor voxels / all foreground voxels (label > 0)")
    print("review/extreme cases excluded from test:", ", ".join(sorted(REVIEW_EXTREME_CASES)))
    for name, cases in [("train", train), ("val", val), ("test", test)]:
        summary = subset_summary(name, cases, stats)
        print(
            f"{name:>5}: n={summary['n']}, "
            f"mean={summary['mean_corrected_tumor_foreground_ratio']}, "
            f"range=[{summary['min_corrected_tumor_foreground_ratio']}, "
            f"{summary['max_corrected_tumor_foreground_ratio']}], "
            f"bins={summary['bins']}, "
            f"review={summary['review_extreme_cases']}"
        )
        print("       " + ", ".join(cases))

    if not args.write:
        print("\n[dry-run] No files were written. Re-run with --write to create v2 sidecar files.")
        return

    info = {
        "dataset": DATASET,
        "version": "701020_stratified_v2",
        "seed": args.seed,
        "active_splits_final_json_was_not_modified": True,
        "policy": (
            "Fixed 70/10/21 split selected by corrected tumor burden ratio. "
            "HCC_065 and HCC_075 are retained as review/extreme train cases and excluded from test. "
            "Test cases must not be used for checkpoint selection or model tuning."
        ),
        "ratios": {"train": 70, "val": 10, "test": 21},
        "bin_thresholds": {
            "tiny": "ratio < 0.03",
            "small": "0.03 <= ratio < 0.15",
            "medium": "0.15 <= ratio < 0.60",
            "extreme": "ratio >= 0.60; review cases excluded from test",
        },
        "test_quotas": TEST_QUOTAS,
        "val_quotas": VAL_QUOTAS,
        "review_extreme_cases": sorted(REVIEW_EXTREME_CASES),
        "train": subset_summary("train", train, stats),
        "val": subset_summary("val", val, stats),
        "test": subset_summary("test", test, stats),
    }

    SPLITS_V2_PATH.write_text(json.dumps([{"train": train, "val": val}], indent=4) + "\n")
    INFO_V2_PATH.write_text(json.dumps(info, indent=4, ensure_ascii=False) + "\n")
    write_case_csv(train, val, test, stats)

    print(f"\nWrote v2 nnU-Net train/val sidecar: {SPLITS_V2_PATH}")
    print(f"Wrote v2 split info: {INFO_V2_PATH}")
    print(f"Wrote case table: {NOTES_CASES_PATH}")
    print("Active splits_final.json was not modified.")


if __name__ == "__main__":
    main()
