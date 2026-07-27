"""
Create a fixed 70/10/21 HCC split from the existing fold-0 nnU-Net split.

Policy:
- The current fold-0 validation cases are frozen as the held-out test set.
- Ten cases are selected from the current fold-0 training set as validation.
- The remaining 70 cases are used for training.
- nnU-Net's splits_final.json stores only train/val.
- The held-out test list is written to split_info_701020_stratified_v2.json.

Run dry first:
    python -m pumengyu.tools.hcc.make_split_701020_from_fold0

Apply:
    python -m pumengyu.tools.hcc.make_split_701020_from_fold0 --apply
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path


SEED = 42
DATASET = "Dataset013_HCCReferencedCT"
PREPROC_DIR = Path("/home/PuMengYu/nnUNet_workspace/preprocessed") / DATASET
SPLITS_PATH = PREPROC_DIR / "splits_final.json"
INFO_PATH = PREPROC_DIR / "split_info_701020_stratified_v2.json"
QC_PATH = Path("/home/PuMengYu/nnUNet/pumengyu/notes/data/hcc_referenced_ct_qc.csv")


def load_ratio_by_case() -> dict[str, float]:
    ratios: dict[str, float] = {}
    with QC_PATH.open(newline="") as f:
        for row in csv.DictReader(f):
            ratios[row["case_id"]] = float(row["tumor_liver_ratio"])
    return ratios


def ratio_bin(ratio: float) -> str:
    if ratio < 0.03:
        return "tiny"
    if ratio < 0.15:
        return "small"
    if ratio < 0.60:
        return "medium"
    return "large"


def pick_validation_cases(train80: list[str], ratios: dict[str, float], seed: int) -> list[str]:
    """Select 10 validation cases with rough tumor/liver-ratio coverage."""
    grouped: dict[str, list[str]] = defaultdict(list)
    for case in train80:
        grouped[ratio_bin(ratios[case])].append(case)

    # Approximate the fold-0 train distribution while keeping extreme cases visible.
    quotas = {
        "tiny": 3,
        "small": 3,
        "medium": 2,
        "large": 2,
    }

    rng = random.Random(seed)
    selected: list[str] = []
    for bin_name in ["tiny", "small", "medium", "large"]:
        cases = grouped[bin_name][:]
        rng.shuffle(cases)
        selected.extend(cases[: quotas[bin_name]])

    if len(selected) != 10:
        remaining = [c for c in train80 if c not in selected]
        rng.shuffle(remaining)
        selected.extend(remaining[: 10 - len(selected)])

    return sorted(selected)


def subset_summary(name: str, cases: list[str], ratios: dict[str, float]) -> dict[str, object]:
    bins = defaultdict(int)
    for case in cases:
        bins[ratio_bin(ratios[case])] += 1
    return {
        "name": name,
        "n": len(cases),
        "mean_tumor_liver_ratio": round(sum(ratios[c] for c in cases) / len(cases), 6),
        "min_tumor_liver_ratio": round(min(ratios[c] for c in cases), 6),
        "max_tumor_liver_ratio": round(max(ratios[c] for c in cases), 6),
        "ratio_bins": dict(sorted(bins.items())),
        "cases": cases,
    }


def print_summary(train: list[str], val: list[str], test: list[str], ratios: dict[str, float]) -> None:
    print("HCC 70/10/21 split from existing fold-0")
    print(f"seed: {SEED}")
    for cases_name, cases in [("train", train), ("val", val), ("test", test)]:
        summary = subset_summary(cases_name, cases, ratios)
        print(
            f"{cases_name:>5}: n={summary['n']}, "
            f"mean_ratio={summary['mean_tumor_liver_ratio']}, "
            f"range=[{summary['min_tumor_liver_ratio']}, {summary['max_tumor_liver_ratio']}], "
            f"bins={summary['ratio_bins']}"
        )
        print("       " + ", ".join(cases))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Write splits_final.json and split_info json")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    splits = json.loads(SPLITS_PATH.read_text())
    fold0 = splits[0]
    train80 = sorted(fold0["train"])
    test21 = sorted(fold0["val"])
    ratios = load_ratio_by_case()

    val10 = pick_validation_cases(train80, ratios, args.seed)
    train70 = sorted(c for c in train80 if c not in set(val10))

    if len(train70) != 70 or len(val10) != 10 or len(test21) != 21:
        raise RuntimeError(
            f"Unexpected split sizes: train={len(train70)}, val={len(val10)}, test={len(test21)}"
        )

    print_summary(train70, val10, test21, ratios)

    if not args.apply:
        print("\n[dry-run] No files were written. Re-run with --apply to update the split.")
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = SPLITS_PATH.with_name(f"splits_final.json.bak_before_701020_{ts}")
    SPLITS_PATH.rename(backup)

    SPLITS_PATH.write_text(json.dumps([{"train": train70, "val": val10}], indent=4) + "\n")

    info = {
        "dataset": DATASET,
        "created_from": str(backup),
        "seed": args.seed,
        "policy": (
            "Current fold-0 val cases are frozen as held-out test; "
            "10 validation cases are sampled from current fold-0 train by tumor/liver-ratio bins."
        ),
        "ratios": {"train": 70, "val": 10, "test": 21},
        "note": "test cases are not written to splits_final.json and must not be used for checkpoint selection.",
        "train": subset_summary("train", train70, ratios),
        "val": subset_summary("val", val10, ratios),
        "test": subset_summary("test", test21, ratios),
    }
    INFO_PATH.write_text(json.dumps(info, indent=4, ensure_ascii=False) + "\n")

    print(f"\nBacked up old split: {backup}")
    print(f"Wrote nnU-Net train/val split: {SPLITS_PATH}")
    print(f"Wrote held-out test record: {INFO_PATH}")


if __name__ == "__main__":
    main()
