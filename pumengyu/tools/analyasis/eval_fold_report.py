"""
nnUNet fold 评估报告 + 可视化
从 summary.json 读指标（无需重新推理），生成：
  - report_custom.txt  (对齐参考格式，有/无肿瘤分开，按 cancer_dice 分级)
  - vis_png_custom/    (每 case 若干轴向切片：GT / Pred / Diff)

用法：
  python pumengyu/eval_fold_report.py \
    --val_dir  <results/.../fold_X/validation> \
    --gt_dir   <preprocessed/DatasetXXX/gt_segmentations> \
    --img_dir  <raw/DatasetXXX/imagesTr> \
    [--vis_slices 5] [--no_vis]
"""

from __future__ import annotations
import argparse, json, os
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")#用这个"Agg"是给只想要保存图片,不需要弹窗的就用这个
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

#nibabel是读写神经影像文件的库,支持格式包括.nii/.nii.gz,.mgz(FreeSurfer),.dcm(DICOM)
#我们直接用img=nib.load(img_path)
#data=img.get_fdata()
#scipy.ndimage是图像处理库,包括连通域标记,缩放,高斯模糊,形态学操作,求质心
#matplotlib.patches是在图上绘制集合图像,
#nibabel是读写医学扫描图像文件,包括MRI,CT,PET等
#scipy.ndimage是同一套函数,处理2D/3D/4D都可以用
#matplotlib.patches这个是画的是几何色块,贴在图上
# ───────────────────────────── postprocessing ────────────────────────────

def apply_min_size_filter(pred: np.ndarray, min_size: int, cls: int = 2) -> np.ndarray:
    """
    pred是要预测的数组
    min_size是连通域的最小大小
    cls是要过滤的类别
    """
    out = pred.copy()
    mask = pred == cls
    #ndimage.label的mask里面是True或者False,然后这个.label吧互相连结的同类型体素归为一个连通域,编号
    #n是连通域的数量,labeld是标记后0为背景,1是第一个连通域,2是第二个连通域
    labeled, n = ndimage.label(mask)#type:ignore
    for i in range(1, n + 1):
        if (labeled == i).sum() < min_size:
            out[labeled == i] = 0
    return out


def compute_case_metrics(pred: np.ndarray, gt: np.ndarray, cls: int = 2) -> dict:
    """
    如果x:[B,C,D,H,W]
    y=x.sum()
    那么y是一个标量.,所有元素加在一起
    y=x.sum(axis=0),y:[C,D,H,W]
    y=x.sum(axis=1),y:[B,D,H,W]

    """
    p = pred == cls
    g = gt == cls
    tp = int((p & g).sum())
    fp = int((p & ~g).sum())
    fn = int((~p & g).sum())
    gt_t = tp + fn
    pred_t = tp + fp
    #gt无肿瘤且预测也无肿瘤
    if tp==0 and fp==0 and fn==0:
        return dict(dice=1.0, recall=float('nan'), precision=float('nan'), fdr=0.0,
                    pred_tumor=0, gt_tumor=0, tp=0, fp=0, fn=0)
    denom_r = tp + fn
    denom_p = tp + fp
    recall    = tp / denom_r if denom_r > 0 else float('nan')
    precision = tp / denom_p if denom_p > 0 else float('nan')
    fdr       = fp / denom_p if denom_p > 0 else 0.0
    dice      = 2 * tp / (2 * tp + fp + fn) 
    return dict(dice=dice, recall=recall, precision=precision, fdr=fdr,
                pred_tumor=pred_t, gt_tumor=gt_t, tp=tp, fp=fp, fn=fn)


# ───────────────────────────── size category ─────────────────────────────

def size_cat(n: int) -> str:
    if n < 5_000:    return "极小(<5k)"
    if n < 50_000:   return "小(5k-50k)"
    if n < 300_000:  return "中等(50k-300k)"
    return "大(>=300k)"


# ───────────────────────────── visualization ─────────────────────────────

OVERLAY = {
    1: (0.2, 0.8, 0.2, 0.35),   # liver — 绿
    2: (1.0, 0.2, 0.2, 0.5),    # tumor — 红
}


def _overlay(ax, img_s, seg_s, title):
    """
    img_s:[H,W],
    img_s.shape=[H,W]
    *img_s.shape解包成H,W
    (*img_s.shape,4)先解包再拼接,最后等价于[H,W,4]
    ax.imshow得vmin和vmax是把小于vmin得映射成纯黑
    把大于vmax得映射成纯白
    ax.imshow只接受[H,W],[3,H,W]这种RGB彩色图
    [H,W,4]这种RGBA彩色图.带者透明度
    """
    ax.imshow(img_s, cmap="gray", vmin=-200, vmax=250)
    for cls_id, color in OVERLAY.items():
        mask = seg_s == cls_id
        if mask.any():
            rgba = np.zeros((*mask.shape, 4))
            rgba[mask] = color
            ax.imshow(rgba)
    ax.set_title(title, fontsize=7)
    ax.axis("off")


def save_visualization(case, img, gt, pred, vis_dir, n_slices):
    fg = np.where((gt > 0).any(axis=(0, 1)))[0]
    if len(fg) == 0:
        fg = np.arange(img.shape[2])
    indices = np.linspace(fg[0], fg[-1], n_slices, dtype=int)

    fig, axes = plt.subplots(n_slices, 3, figsize=(9, 3 * n_slices))
    if n_slices == 1:
        axes = axes[np.newaxis, :]

    for row, z in enumerate(indices):
        img_s = img[:, :, z].T
        gt_s  = gt[:, :, z].T
        pr_s  = pred[:, :, z].T
        _overlay(axes[row, 0], img_s, gt_s,  f"GT   z={z}")
        _overlay(axes[row, 1], img_s, pr_s,  f"Pred z={z}")
        diff = np.zeros((*img_s.shape, 4))
        diff[(pr_s > 0) & (gt_s == 0)] = (1.0, 0.5, 0.0, 0.6)   # FP orange
        diff[(gt_s > 0) & (pr_s == 0)] = (0.0, 0.4, 1.0, 0.6)   # FN blue
        axes[row, 2].imshow(img_s, cmap="gray", vmin=-200, vmax=250)
        axes[row, 2].imshow(diff)
        axes[row, 2].set_title(f"Diff z={z}  orange=FP  blue=FN", fontsize=7)
        axes[row, 2].axis("off")

    patches = [
        mpatches.Patch(color=(0.2, 0.8, 0.2), label="liver"),
        mpatches.Patch(color=(1.0, 0.2, 0.2), label="tumor"),
    ]
    fig.legend(handles=patches, loc="lower center", ncol=2, fontsize=8)
    fig.suptitle(case, fontsize=9)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(os.path.join(vis_dir, f"{case}.png"), dpi=100, bbox_inches="tight")
    plt.close(fig)


# ───────────────────────────── report helpers ────────────────────────────

def fmt_n(n) -> str:
    return f"{int(n):,}" if n is not None else "N/A"


def _section_header(title: str) -> str:
    return "\n" + "=" * 80 + f"\n{title}\n" + "=" * 80


def _row(case, dice, recall, precision, fdr, pred_t, gt_t, gt_l):
    sc = size_cat(gt_t) if gt_t else "—"
    return (f"  {case:<20} {dice:>10.4f} {recall:>8.4f} {precision:>10.4f}"
            f" {fdr:>8.4f} {fmt_n(pred_t):>12} {fmt_n(gt_t):>10}"
            f" {fmt_n(gt_l):>10} {sc:<18}")


COL_HDR = (f"  {'case':<20} {'tumor_dice':>10} {'recall':>8} {'precision':>10}"
           f" {'FDR':>8} {'pred_tumor':>12} {'gt_tumor':>10}"
           f" {'gt_liver':>10} {'size_cat':<18}")
SEP = "-" * 100


# ───────────────────────────── core function ─────────────────────────────

def run_eval_report(val_dir, gt_dir, img_dir,
                    vis_slices: int = 5,
                    no_vis: bool = False,
                    min_tumor_size: int = 0):
    """
    生成 report_custom.txt 和（可选）vis_png_custom/。

    参数：
      val_dir        fold_X/validation 目录
      gt_dir         preprocessed/DatasetXXX/gt_segmentations 目录
      img_dir        raw/DatasetXXX/imagesTr 目录
      vis_slices     每个 case 可视化切片数
      no_vis         True 时跳过可视化
      min_tumor_size 后处理：去掉体素数 < 该值的 cancer 连通域（0=关闭）
    """
    val_dir = Path(val_dir)
    gt_dir  = Path(gt_dir)
    img_dir = Path(img_dir)
    out_dir = val_dir.parent

    summary_path = val_dir / "summary.json"
    if not summary_path.exists():
        print(f"[eval_fold_report] 找不到 {summary_path}")
        return

    summary = json.load(open(summary_path))
    cases_raw = summary["metric_per_case"]

    has_tumor, no_tumor = [], []

    for c in cases_raw:
        case = Path(c["reference_file"]).stem.replace(".nii", "")
        m1 = c["metrics"].get("1", {})
        m2 = c["metrics"].get("2", {})

        gt_liver  = int(m1.get("TP", 0) + m1.get("FN", 0))
        gt_tumor  = int(m2.get("TP", 0) + m2.get("FN", 0))
        pred_tumor = int(m2.get("TP", 0) + m2.get("FP", 0))

        liver_dice = m1.get("Dice")
        if liver_dice is not None and np.isnan(liver_dice):
            liver_dice = None

        if gt_tumor == 0:
            no_tumor.append({
                "case":       case,
                "liver_dice": liver_dice,
                "pred_tumor": pred_tumor,
                "gt_liver":   gt_liver,
            })
        else:
            tp2 = m2.get("TP", 0)
            fp2 = m2.get("FP", 0)
            fn2 = m2.get("FN", 0)
            denom_r = tp2 + fn2
            denom_p = tp2 + fp2
            recall    = tp2 / denom_r if denom_r > 0 else 0.0
            precision = tp2 / denom_p if denom_p > 0 else 0.0
            fdr       = fp2 / denom_p if denom_p > 0 else 0.0
            fnr       = fn2 / denom_r if denom_r > 0 else 0.0
            jaccard   = tp2 / (tp2 + fp2 + fn2) if (tp2 + fp2 + fn2) > 0 else 0.0
            dice2     = m2.get("Dice")
            if dice2 is None or np.isnan(dice2):
                dice2 = 0.0

            has_tumor.append({
                "case":       case,
                "liver_dice": liver_dice,
                "dice":       float(dice2),
                "recall":     recall,
                "precision":  precision,
                "fdr":        fdr,
                "fnr":        fnr,
                "jaccard":    jaccard,
                "pred_tumor": pred_tumor,
                "gt_tumor":   gt_tumor,
                "gt_liver":   gt_liver,
            })

    has_tumor.sort(key=lambda x: x["dice"])

    def mean_std(key):
        vals = [r[key] for r in has_tumor]
        return np.mean(vals), np.std(vals)

    liver_dices = [r["liver_dice"] for r in has_tumor + no_tumor if r["liver_dice"] is not None]
    false_pos_cases = [r for r in no_tumor if r["pred_tumor"] > 0]

    lines = [
        "nnUNet Validation Report",
        "=" * 40,
        f"fold_dir : {val_dir}",
        f"n_cases  : {len(has_tumor) + len(no_tumor)}",
        "",
    ]

    if liver_dices:
        lines.append(f"Liver")
        lines.append(f"  Dice: mean={np.mean(liver_dices):.4f}  std={np.std(liver_dices):.4f}")
        lines.append("")

    lines.append(f"Tumor (有肿瘤 case, n={len(has_tumor)})")
    if has_tumor:
        for metric, key in [("Dice", "dice"), ("Jaccard", "jaccard"),
                             ("Recall", "recall"), ("FDR", "fdr"),
                             ("FNR", "fnr"), ("Precision", "precision")]:
            m, s = mean_std(key)
            lines.append(f"  {metric:<12}: mean={m:.4f}  std={s:.4f}")
    lines.append("")

    lines.append(f"Tumor (无肿瘤 case, n={len(no_tumor)})")
    if no_tumor:
        fp_rate = len(false_pos_cases) / len(no_tumor)
        lines.append(f"  误报率(预测出肿瘤但GT无肿瘤): {fp_rate:.2%}  ({len(false_pos_cases)}/{len(no_tumor)} cases)")
        if false_pos_cases:
            lines.append(f"  误报 cases:")
            for r in false_pos_cases:
                lines.append(f"    {r['case']:<20}  pred_tumor={fmt_n(r['pred_tumor'])}")
        else:
            lines.append(f"  所有无肿瘤 case 均正确预测为阴性")
        lines.append("")
        lines.append(f"  无肿瘤 case 列表:")
        for r in no_tumor:
            flag = "  [误报]" if r["pred_tumor"] > 0 else ""
            lines.append(f"    {r['case']:<20}  liver_dice={r['liver_dice']:.4f}"
                         f"  pred_tumor={fmt_n(r['pred_tumor'])}{flag}")
    else:
        lines.append("  （无此类 case）")
    lines.append("")

    lines.append(_section_header("Per-Case 分级(按 tumor_dice 从低到高)"))

    thresholds = [
        ("[严重失败] tumor_dice < 0.3",        lambda r: r["dice"] < 0.3),
        ("[需要改进] 0.3 <= tumor_dice < 0.7",  lambda r: 0.3 <= r["dice"] < 0.7),
        ("[没问题]   tumor_dice >= 0.7",         lambda r: r["dice"] >= 0.7),
    ]
    for label, cond in thresholds:
        subset = [r for r in has_tumor if cond(r)]
        lines.append(f"\n{label}  (n={len(subset)})")
        lines.append(SEP)
        lines.append(COL_HDR)
        lines.append(SEP)
        for r in subset:
            lines.append(_row(r["case"], r["dice"], r["recall"], r["precision"],
                              r["fdr"], r["pred_tumor"], r["gt_tumor"], r["gt_liver"]))

    if min_tumor_size > 0:
        lines.append(_section_header(
            f"后处理对比 — min_tumor_size={min_tumor_size}"))
        lines.append(f"（去掉预测中体素数 < {min_tumor_size} 的 cancer 连通域）\n")

        col = (f"  {'case':<20} {'dice_before':>11} {'dice_after':>10}"
               f" {'recall_b':>9} {'recall_a':>9}"
               f" {'fdr_b':>7} {'fdr_a':>7}"
               f" {'pred_b':>10} {'pred_a':>10}")
        lines.append(col)
        lines.append("-" * 100)

        delta_dice, delta_recall, delta_fdr = [], [], []

        for pred_path in sorted(val_dir.glob("*.nii.gz")):
            case = pred_path.stem.replace(".nii", "")
            gt_path = gt_dir / f"{case}.nii.gz"
            if not gt_path.exists():
                continue
            pred_arr = np.asarray(nib.load(pred_path).dataobj, dtype=np.int16)
            gt_arr   = np.asarray(nib.load(gt_path).dataobj,   dtype=np.int16)

            before = compute_case_metrics(pred_arr, gt_arr, cls=2)
            pred_pp = apply_min_size_filter(pred_arr, min_tumor_size, cls=2)
            after  = compute_case_metrics(pred_pp,   gt_arr, cls=2)

            gt_t = before["gt_tumor"]
            if gt_t == 0:
                fp_before = before["pred_tumor"]
                fp_after  = after["pred_tumor"]
                flag = "  ← 误报已清除" if fp_before > 0 and fp_after == 0 else (
                       "  ← 误报减少"  if fp_after < fp_before else "")
                lines.append(f"  {case:<20} [无肿瘤]  "
                             f"pred_before={fmt_n(fp_before):>10}  "
                             f"pred_after={fmt_n(fp_after):>10}{flag}")
            else:
                d_dice   = after["dice"]    - before["dice"]
                d_recall = after["recall"]  - before["recall"]
                d_fdr    = after["fdr"]     - before["fdr"]
                delta_dice.append(d_dice)
                delta_recall.append(d_recall)
                delta_fdr.append(d_fdr)
                lines.append(
                    f"  {case:<20} {before['dice']:>11.4f} {after['dice']:>10.4f}"
                    f" {before['recall']:>9.4f} {after['recall']:>9.4f}"
                    f" {before['fdr']:>7.4f} {after['fdr']:>7.4f}"
                    f" {fmt_n(before['pred_tumor']):>10} {fmt_n(after['pred_tumor']):>10}")

        if delta_dice:
            before_dices = [r["dice"] for r in has_tumor]
            lines.append("")
            lines.append(f"  汇总（有肿瘤 cases, n={len(delta_dice)}）:")
            lines.append(f"  Dice     before={np.mean(before_dices):.4f}"
                         f"  after={np.mean(before_dices) + np.mean(delta_dice):.4f}"
                         f"  Δ={np.mean(delta_dice):+.4f}")
            lines.append(f"  Recall   Δ={np.mean(delta_recall):+.4f}")
            lines.append(f"  FDR      Δ={np.mean(delta_fdr):+.4f}")

    report_path = out_dir / "report_custom.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[eval_fold_report] report -> {report_path}")

    if not no_vis:
        vis_dir = out_dir / "vis_png_custom"
        vis_dir.mkdir(exist_ok=True)
        pred_files = sorted(val_dir.glob("*.nii.gz"))
        for pred_path in pred_files:
            case = pred_path.stem.replace(".nii", "")
            gt_path  = gt_dir  / f"{case}.nii.gz"
            img_path = img_dir / f"{case}_0000.nii.gz"
            if not img_path.exists():
                img_path = img_dir / f"{case}.nii.gz"
            if not gt_path.exists() or not img_path.exists():
                print(f"  [WARN] 跳过可视化 {case}（缺 GT 或 CT）")
                continue
            print(f"  vis {case}")
            pred_arr = np.asarray(nib.load(pred_path).dataobj,  dtype=np.int16)
            gt_arr   = np.asarray(nib.load(gt_path).dataobj,    dtype=np.int16)
            img_arr  = np.asarray(nib.load(img_path).dataobj,   dtype=np.float32)
            save_visualization(case, img_arr, gt_arr, pred_arr, str(vis_dir), vis_slices)
        print(f"[eval_fold_report] vis   -> {vis_dir}/")


# ───────────────────────────── main ──────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--val_dir",         required=True)
    p.add_argument("--gt_dir",          required=True)
    p.add_argument("--img_dir",         required=True)
    p.add_argument("--vis_slices",      type=int, default=5)
    p.add_argument("--no_vis",          action="store_true")
    p.add_argument("--min_tumor_size",  type=int, default=0,
                   help="过滤小于该体素数的 cancer 连通域（0=关闭，推荐 100）")
    args = p.parse_args()
    run_eval_report(
        val_dir=args.val_dir,
        gt_dir=args.gt_dir,
        img_dir=args.img_dir,
        vis_slices=args.vis_slices,
        no_vis=args.no_vis,
        min_tumor_size=args.min_tumor_size,
    )


if __name__ == "__main__":
    main()
