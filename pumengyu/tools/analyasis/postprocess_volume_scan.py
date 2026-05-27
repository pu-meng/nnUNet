"""
体积过滤后处理扫描脚本

对已有的 validation nii 预测结果，扫描多个体积阈值，
输出每个阈值下的综合Dice/有肿瘤Dice/FP率/各大小类别Dice，找出最佳阈值。

不需要重新推理，直接对预测 nii 做后处理。

用法：
  python pumengyu/tools/analyasis/postprocess_volume_scan.py \
    --val_dir /home/PuMengYu/nnUNet_workspace/results/Dataset003_Liver/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_4/validation \
    --gt_dir  /home/PuMengYu/nnUNet_workspace/preprocessed/Dataset003_Liver/gt_segmentations

可选参数：
  --thresholds  自定义扫描阈值列表（空格分隔，默认见代码）
  --tumor_cls   肿瘤标签值，Dataset003=2，Dataset004=1（默认自动检测）
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage


THRESHOLDS_DEFAULT = [0, 10, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 50000]

SIZE_BINS = [
    ("极小(<5k)",      lambda g: g < 5_000),
    ("小(5k-50k)",     lambda g: 5_000 <= g < 50_000),
    ("中等(50k-300k)", lambda g: 50_000 <= g < 300_000),
    ("大(>=300k)",     lambda g: g >= 300_000),
]


def detect_tumor_cls(val_dir: Path) -> int:
    """从 summary.json 自动检测肿瘤标签（Dataset003=2, Dataset004=1）。"""
    summary_path = val_dir / "summary.json"
    if summary_path.exists():
        data = json.load(open(summary_path))
        for c in data.get("metric_per_case", []):
            if "2" in c.get("metrics", {}):
                return 2
    return 1


def volume_filter(pred: np.ndarray, tumor_cls: int, min_size: int) -> np.ndarray:
    if min_size <= 0:
        return pred
    out = pred.copy()
    mask = pred == tumor_cls
    if not mask.any():
        return out
    labeled, n = ndimage.label(mask)
    for i in range(1, n + 1):
        if (labeled == i).sum() < min_size:
            out[labeled == i] = 0
    return out


def case_metrics(pred: np.ndarray, gt: np.ndarray, tumor_cls: int) -> dict:
    p = pred == tumor_cls
    g = gt == tumor_cls
    tp = int((p & g).sum())
    fp = int((p & ~g).sum())
    fn = int((~p & g).sum())
    gt_tumor = tp + fn
    pred_tumor = tp + fp
    if gt_tumor == 0:
        return {"gt_tumor": 0, "pred_tumor": pred_tumor,
                "dice": None, "recall": None, "precision": None}
    if tp == 0 and fp == 0 and fn == 0:
        return {"gt_tumor": gt_tumor, "pred_tumor": 0,
                "dice": 1.0, "recall": 1.0, "precision": 1.0}
    dice = 2 * tp / (2 * tp + fp + fn)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    return {"gt_tumor": gt_tumor, "pred_tumor": pred_tumor,
            "dice": dice, "recall": recall, "precision": precision}


def load_cases(val_dir: Path, gt_dir: Path, tumor_cls: int):
    """返回每个 case 的 (pred_arr, gt_arr, gt_tumor_size)。"""
    cases = []
    for pred_path in sorted(val_dir.glob("*.nii.gz")):
        case = pred_path.stem.replace(".nii", "")
        gt_path = gt_dir / f"{case}.nii.gz"
        if not gt_path.exists():
            print(f"  [WARN] 找不到 GT: {gt_path}，跳过")
            continue
        pred = np.asarray(nib.load(pred_path).dataobj, dtype=np.int16)
        gt = np.asarray(nib.load(gt_path).dataobj, dtype=np.int16)
        gt_tumor = int((gt == tumor_cls).sum())
        cases.append({"case": case, "pred": pred, "gt": gt, "gt_tumor": gt_tumor})
    return cases


def scan(cases: list, thresholds: list[int], tumor_cls: int) -> list[dict]:
    results = []
    for thr in thresholds:
        all_dice, tumor_dice = [], []
        fp_count, no_tumor_total = 0, 0
        bin_dices: dict[str, list] = {b[0]: [] for b in SIZE_BINS}

        for c in cases:
            pred_pp = volume_filter(c["pred"], tumor_cls, thr)
            m = case_metrics(pred_pp, c["gt"], tumor_cls)

            if m["gt_tumor"] == 0:
                no_tumor_total += 1
                if m["pred_tumor"] > 0:
                    fp_count += 1
                    all_dice.append(0.0)
                else:
                    all_dice.append(1.0)
            else:
                d = m["dice"] if m["dice"] is not None else 0.0
                all_dice.append(d)
                tumor_dice.append(d)
                for label, cond in SIZE_BINS:
                    if cond(m["gt_tumor"]):
                        bin_dices[label].append(d)
                        break

        row = {
            "threshold":    thr,
            "comp_dice":    float(np.mean(all_dice)),
            "tumor_dice":   float(np.mean(tumor_dice)) if tumor_dice else float("nan"),
            "fp_count":     fp_count,
            "no_tumor_n":   no_tumor_total,
            "fp_rate":      fp_count / no_tumor_total if no_tumor_total > 0 else 0.0,
        }
        for label, _ in SIZE_BINS:
            vals = bin_dices[label]
            row[label] = float(np.mean(vals)) if vals else float("nan")
        results.append(row)
    return results


def build_table(results: list[dict]) -> str:
    lines = []
    size_labels = [b[0] for b in SIZE_BINS]
    header = (f"{'阈值(v)':>10}  {'综合Dice':>9}  {'有肿瘤Dice':>10}  "
              f"{'FP数':>5}  {'FP率':>7}  "
              + "  ".join(f"{lb[:6]:>8}" for lb in size_labels))
    sep = "=" * len(header)
    lines.append("\n" + sep)
    lines.append("体积过滤扫描结果（阈值=0 表示无过滤，即原始预测）")
    lines.append(sep)
    lines.append(header)
    lines.append("-" * len(header))
    for r in results:
        bins_str = "  ".join(
            f"{r.get(lb, float('nan')):>8.4f}" for lb, _ in SIZE_BINS
        )
        lines.append(
            f"{r['threshold']:>10}  {r['comp_dice']:>9.4f}  {r['tumor_dice']:>10.4f}  "
            f"{r['fp_count']:>5}  {r['fp_rate']:>6.1%}  {bins_str}"
        )
    lines.append(sep)
    lines.append("")

    best = max(results, key=lambda r: r["comp_dice"])
    lines.append(f"最佳综合Dice: 阈值={best['threshold']}  综合Dice={best['comp_dice']:.4f}")

    base_tumor_dice = results[0]["tumor_dice"]
    candidates = [r for r in results if r["tumor_dice"] >= base_tumor_dice - 0.005]
    if candidates:
        best_fp = min(candidates, key=lambda r: r["fp_count"])
        lines.append(
            f"最小FP（有肿瘤Dice不低于原始-0.5%）: 阈值={best_fp['threshold']}"
            f"  综合Dice={best_fp['comp_dice']:.4f}  FP={best_fp['fp_count']}"
        )
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--val_dir",    required=True, help="fold_X/validation 目录")
    p.add_argument("--gt_dir",     required=True, help="gt_segmentations 目录")
    p.add_argument("--out_txt",    required=True, help="输出 txt 文件路径")
    p.add_argument("--thresholds", type=int, nargs="+", default=THRESHOLDS_DEFAULT,
                   help="扫描的体积阈值列表（单位：voxels）")
    p.add_argument("--tumor_cls",  type=int, default=None,
                   help="肿瘤标签值（默认自动检测：Dataset003=2，Dataset004=1）")
    args = p.parse_args()

    val_dir  = Path(args.val_dir)
    gt_dir   = Path(args.gt_dir)
    out_path = Path(args.out_txt)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tumor_cls = args.tumor_cls if args.tumor_cls is not None else detect_tumor_cls(val_dir)
    print(f"tumor_cls={tumor_cls}，val_dir={val_dir.name}，加载 case...")

    cases = load_cases(val_dir, gt_dir, tumor_cls)
    print(f"共 {len(cases)} 个 case，扫描 {len(args.thresholds)} 个阈值...")

    results = scan(cases, args.thresholds, tumor_cls)
    table = build_table(results)
    out_path.write_text(table, encoding="utf-8")
    print(f"结果已写入 {out_path}")


if __name__ == "__main__":
    main()
