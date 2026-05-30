"""
一键实验后分析入口。

用法：
  cd /home/PuMengYu/nnUNet

  # 分析验证集（默认）
  python -m pumengyu.analysis.run \\
    --trainer nnUNetTrainer_Baseline \\
    --dataset Dataset003_Liver --fold 0

  # 分析测试集
  python -m pumengyu.analysis.run \\
    --trainer nnUNetTrainer_Baseline \\
    --dataset Dataset003_Liver --fold 0 --split test

  # 分析全集（需先跑推理生成 all_prediction）：
  #   nnUNetv2_predict -i .../imagesTr -o .../fold_0/all_prediction \\
  #     -d Dataset003_Liver -c 3d_fullres -tr nnUNetTrainer_Baseline -f 0
  python -m pumengyu.analysis.run \\
    --trainer nnUNetTrainer_Baseline \\
    --dataset Dataset003_Liver --fold 0 --split all

  # 指定任意 pred_dir（最灵活）
  python -m pumengyu.analysis.run \\
    --trainer nnUNetTrainer_Baseline \\
    --dataset Dataset003_Liver --fold 0 \\
    --pred_dir /path/to/any/prediction_dir

  # 困难 case 切片可视化（dice < 0.5 的 case 生成 GT/Pred 对比图）
  python -m pumengyu.analysis.run \\
    --trainer nnUNetTrainer_Baseline \\
    --dataset Dataset003_Liver --fold 0 \\
    --hard_vis --hard_thresh 0.5

输出目录：
  pumengyu/notes/实验结果分析/{trainer}_fold{N}_{split}/
    report.txt
    plots/
      01_cc_distribution.png
      02_per_case_dice.png
      03_size_stratified_boxplot.png
      04_threshold_curve.png
      05_notumor_fp.png
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path

import csv

from pumengyu.analysis.cc import analyze_gt, analyze_pred, separability_gap
from pumengyu.analysis.metrics import load_summary, add_size_category, aggregate
from pumengyu.analysis.threshold import scan_thresholds
from pumengyu.analysis.report import build_report


# ──────────────────────── 路径自动推导 ───────────────────────────────

WORKSPACE = Path("/home/PuMengYu/nnUNet_workspace")
NOTES_DIR = Path(__file__).parent.parent / "notes" / "实验结果分析"


SPLIT_SUBDIR = {
    "validation": "validation",
    "test":       "test_prediction",
    "all":        "all_prediction",   # 需先手动跑推理
}


def derive_paths(trainer: str, dataset: str, fold: int, split: str,
                 pred_dir_override: Path | None = None) -> dict:
    results_root = WORKSPACE / "results_v2" / dataset
    fold_dir = results_root / f"{trainer}__nnUNetPlans__3d_fullres" / f"fold_{fold}"

    if pred_dir_override is not None:
        pred_dir = pred_dir_override
    else:
        subdir = SPLIT_SUBDIR.get(split, split)
        pred_dir = fold_dir / subdir

    summary = pred_dir / "summary.json"
    gt_dir  = WORKSPACE / "preprocessed" / dataset / "gt_segmentations"
    img_dir = WORKSPACE / "raw" / dataset / "imagesTr"

    return {
        "fold_dir":   fold_dir,
        "pred_dir":   pred_dir,
        "summary":    summary,
        "gt_dir":     gt_dir,
        "img_dir":    img_dir,
    }


# ──────────────────────── 主流程 ─────────────────────────────────────

def run(trainer: str, dataset: str, fold: int, split: str,
        no_plots: bool = False, tumor_label: int = 2,
        pred_dir_override: Path | None = None,
        hard_vis: bool = False, hard_thresh: float = 0.5) -> None:

    paths = derive_paths(trainer, dataset, fold, split, pred_dir_override)
    out_dir = NOTES_DIR / f"{trainer}_fold{fold}_{split}"
    plot_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(exist_ok=True)

    title = f"{trainer}  fold={fold}  [{split}]"
    print(f"\n{'='*60}")
    print(f"分析目标: {title}")
    print(f"输出目录: {out_dir}")
    print(f"{'='*60}")

    # ── GT 分析（全集，不依赖预测）──
    print("\n[1/5] GT CC 分布分析...")
    gt_result = analyze_gt(paths["gt_dir"], tumor_label=tumor_label)

    # ── 预测 CC 分析 ──
    pred_result = None
    if paths["pred_dir"].exists():
        print(f"\n[2/5] 预测 CC 分析（{paths['pred_dir'].name}）...")
        pred_result = analyze_pred(paths["gt_dir"], paths["pred_dir"],
                                   tumor_label=tumor_label)
    else:
        print(f"\n[2/5] 预测目录不存在，跳过: {paths['pred_dir']}")

    # ── Separability Gap ──
    sep = None
    if pred_result:
        sep = separability_gap(
            gt_result["gt_cc_sizes"],
            pred_result["fp_in_notumor_cc"],
        )
        print(f"\n[3/5] Separability Gap = {sep['gap']}  "
              f"({'❌ 失效' if sep['feasible'] is False else '✅ 可行' if sep['feasible'] else '—'})")
    else:
        print("\n[3/5] 跳过 Separability Gap（无预测数据）")

    # ── 指标摘要（summary.json）──
    records = None
    agg = None
    if paths["summary"].exists():
        print("\n[4/5] 读取 summary.json 指标...")
        records = load_summary(paths["summary"])
        records = add_size_category(records)
        agg = aggregate(records)
        print(f"  Overall={agg['overall']:.4f}  "
              f"无肿瘤误报={agg['n_fp_notumor']}/{agg['n_notumor_cases']}")
    else:
        print(f"\n[4/5] 未找到 summary.json，跳过: {paths['summary']}")

    # ── 图表 ──
    if not no_plots:
        print("\n[5/5] 生成图表...")
        from pumengyu.analysis import plots

        gt_cc   = gt_result["gt_cc_sizes"]
        tp_cc   = pred_result["tp_cc"]            if pred_result else []
        fp_t_cc = pred_result["fp_in_tumor_cc"]   if pred_result else []
        fp_n_cc = pred_result["fp_in_notumor_cc"] if pred_result else []

        plots.plot_cc_distribution(gt_cc, tp_cc, fp_t_cc, fp_n_cc, plot_dir, title)

        if records:
            plots.plot_per_case_dice(records, plot_dir, title)
            plots.plot_size_stratified(records, plot_dir, title)

            notumor_records = [r for r in records if r["size_cat"] == "无肿瘤"]
            if notumor_records:
                plots.plot_notumor_fp(notumor_records, plot_dir, title)

            # 困难 case 切片可视化
            if hard_vis:
                hard_dir = plot_dir / "hard_cases"
                hard_dir.mkdir(exist_ok=True)
                plots.plot_hard_cases(
                    records=records,
                    pred_dir=paths["pred_dir"],
                    gt_dir=paths["gt_dir"],
                    img_dir=paths["img_dir"],
                    out_dir=hard_dir,
                    dice_thresh=hard_thresh,
                    title=title,
                )

        if pred_result and (pred_result["tp_cc"] or pred_result["fp_in_notumor_cc"]):
            thresh_result = scan_thresholds(
                pred_result["tp_cc"], pred_result["fp_in_notumor_cc"]
            )
            plots.plot_threshold_curve(thresh_result, plot_dir, title)
        else:
            thresh_result = None
    else:
        thresh_result = None
        print("\n[5/5] 跳过图表（--no_plots）")

    # ── 预测 CSV（GT 来源明确标注，预测来源明确标注）──
    if pred_result and pred_result.get("per_cc_pred"):
        # per_cc_pred.csv — 每行一个预测 CC
        cc_fields = ["case", "cc_idx", "size", "label"]
        cc_path = out_dir / "per_cc_pred.csv"
        with open(cc_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cc_fields)
            w.writeheader()
            w.writerows(pred_result["per_cc_pred"])
        print(f"\nper_cc_pred.csv  ({len(pred_result['per_cc_pred'])} 行)")

    if pred_result and pred_result.get("per_case_pred"):
        # per_case_pred.csv — 每行一个 case（预测维度）
        case_fields = ["case", "has_tumor", "n_pred_cc",
                       "n_tp", "n_fp_tumor", "n_fp_notumor",
                       "fp_vox_total", "fn_vox_total"]
        case_path = out_dir / "per_case_pred.csv"
        with open(case_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=case_fields)
            w.writeheader()
            w.writerows(sorted(pred_result["per_case_pred"],
                               key=lambda x: x["case"]))
        print(f"per_case_pred.csv ({len(pred_result['per_case_pred'])} 行)")

    # summary.json 指标 CSV（含 dice，预测来源）
    if records:
        csv_path = out_dir / "per_case_metrics.csv"
        fields = ["case", "size_cat", "n_ref_tumor", "n_pred_tumor",
                  "dice_tumor", "dice_liver", "FP_tumor", "FN_tumor"]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in sorted(records, key=lambda x: x["case"]):
                w.writerow({k: r.get(k, "") for k in fields})
        print(f"per_case_metrics.csv ({len(records)} 行)")

    # ── 文本报告 ──
    report_text = build_report(
        trainer=trainer, fold=fold, split=split,
        gt_result=gt_result,
        pred_result=pred_result,
        metrics_records=records,
        agg=agg,
        sep=sep,
        thresh_result=thresh_result,
    )
    report_path = out_dir / "report.txt"
    report_path.write_text(report_text, encoding="utf-8")
    print(f"\n文本报告: {report_path}")
    print(report_text)
    print(f"\n✅ 完成  输出目录: {out_dir}")


# ──────────────────────── CLI ─────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="nnUNet 实验后深度分析")
    p.add_argument("--trainer",  required=True, help="Trainer 类名")
    p.add_argument("--dataset",  default="Dataset003_Liver")
    p.add_argument("--fold",     default=0, type=int)
    p.add_argument("--split",    default="validation",
                   choices=["validation", "test", "all"],
                   help="validation/test/all（all 需先跑推理到 all_prediction/）")
    p.add_argument("--pred_dir", default=None, type=Path,
                   help="手动指定预测目录，覆盖 --split 的自动推导")
    p.add_argument("--no_plots",    action="store_true", help="跳过图表生成")
    p.add_argument("--hard_vis",    action="store_true", help="生成困难 case 切片可视化")
    p.add_argument("--hard_thresh", default=0.5, type=float,
                   help="Dice 低于此值视为困难 case（默认 0.5）")
    p.add_argument("--tumor_label", default=2, type=int)
    args = p.parse_args()
    run(args.trainer, args.dataset, args.fold, args.split,
        no_plots=args.no_plots, tumor_label=args.tumor_label,
        pred_dir_override=args.pred_dir,
        hard_vis=args.hard_vis, hard_thresh=args.hard_thresh)


if __name__ == "__main__":
    main()
