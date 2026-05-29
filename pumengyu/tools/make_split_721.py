"""
生成 7:1:2 固定划分的 splits_final.json（分层抽样）

131 例 → 92 train / 13 val(nnUNet监控) / 26 test(论文报告)
- 分层：按有无肿瘤分别抽样，确保三个子集比例一致
- 写入 preprocessed/<Dataset>/splits_final.json（仅 fold 0，train+val）
- 同时输出 split_info_712.json，记录三个子集 case ID

用法：
    python pumengyu/tools/make_split_721.py --dry-run
    python pumengyu/tools/make_split_721.py
"""

import argparse
import json
import random
import numpy as np
import nibabel as nib
from datetime import datetime
from pathlib import Path

DATASET_MAP = {
    "3": ("Dataset003_Liver",      "Dataset003_Liver"),
    "4": ("Dataset004_LiverTumor", "Dataset004_LiverTumor"),
}
_DEFAULT    = "3"
SEED        = 42
TRAIN_RATIO = 0.7
VAL_RATIO   = 0.1   # nnUNet 训练监控用
# test_ratio = 0.2  → 论文报告用


def get_all_cases(images_dir: Path) -> list[str]:
    return sorted(f.name.replace("_0000.nii.gz", "") for f in images_dir.glob("*_0000.nii.gz"))


def has_tumor(case: str, gt_dir: Path) -> bool:
    lbl = np.asarray(nib.load(str(gt_dir / f"{case}.nii.gz")).dataobj)
    return bool(np.any(lbl == 2))


def stratified_split(cases: list[str], gt_dir: Path, seed: int = SEED):
    """按有无肿瘤分层，再各自 7:1:2 切分，最后合并。"""
    rng = random.Random(seed)

    tumor    = [c for c in cases if has_tumor(c, gt_dir)]
    no_tumor = [c for c in cases if not has_tumor(c, gt_dir)]

    def cut(lst):
        s = lst.copy()
        rng.shuffle(s)
        n = len(s)
        n_tr = round(n * TRAIN_RATIO)
        n_va = round(n * VAL_RATIO)
        return s[:n_tr], s[n_tr:n_tr+n_va], s[n_tr+n_va:]

    tr_t, va_t, te_t = cut(tumor)
    tr_n, va_n, te_n = cut(no_tumor)

    train = sorted(tr_t + tr_n)
    val   = sorted(va_t + va_n)
    test  = sorted(te_t + te_n)

    return train, val, test, len(tumor), len(no_tumor)


def print_summary(train, val, test, gt_dir):
    def count_tumor(lst):
        return sum(1 for c in lst if has_tumor(c, gt_dir))

    print(f"\n{'子集':<8} {'n':>4}  {'有肿瘤':>6}  {'无肿瘤':>6}")
    print("-" * 32)
    for label, subset in [("train", train), ("val", val), ("test", test)]:
        nt = count_tumor(subset)
        print(f"{label:<8} {len(subset):>4}  {nt:>6}  {len(subset)-nt:>6}")
    print(f"\ntest cases ({len(test)}):")
    for c in test:
        print(f"  {c}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="只打印不写文件")
    parser.add_argument("--dataset", choices=["3", "4"], default=_DEFAULT,
                        help="3=Dataset003_Liver (default), 4=Dataset004_LiverTumor")
    args = parser.parse_args()

    raw_name, pre_name = DATASET_MAP[args.dataset]
    dataset_dir = Path("/home/PuMengYu/nnUNet_workspace/raw") / raw_name
    preproc_dir = Path("/home/PuMengYu/nnUNet_workspace/preprocessed") / pre_name
    gt_dir      = preproc_dir / "gt_segmentations"

    cases = get_all_cases(dataset_dir / "imagesTr")
    print(f"总计：{len(cases)} 例，正在分层抽样（seed={SEED}）...")

    train, val, test, n_tumor, n_no_tumor = stratified_split(cases, gt_dir, SEED)
    print_summary(train, val, test, gt_dir)

    if args.dry_run:
        print("\n[dry-run] 未写入任何文件")
        return

    # ── splits_final.json（nnUNet 格式，fold 0 = train + val）──────────────
    splits_path = preproc_dir / "splits_final.json"
    if splits_path.exists():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = splits_path.with_name(f"splits_final.json.bak_{ts}")
        splits_path.rename(backup)
        print(f"\n旧 splits_final.json 已备份为 {backup.name}")

    splits = [{"train": train, "val": val}]
    with open(splits_path, "w") as f:
        json.dump(splits, f, indent=2)
    print(f"已写入 {splits_path}")

    # ── split_info_712.json（完整记录三个子集）────────────────────────────
    info = {
        "seed": SEED,
        "ratios": {"train": 0.7, "val": 0.1, "test": 0.2},
        "note": "val=nnUNet训练监控, test=论文报告internal test set",
        "total": len(cases),
        "train": {"n": len(train), "cases": train},
        "val":   {"n": len(val),   "cases": val},
        "test":  {"n": len(test),  "cases": test},
    }
    info_path = preproc_dir / "split_info_712.json"
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"已写入 {info_path}")


if __name__ == "__main__":
    main()
