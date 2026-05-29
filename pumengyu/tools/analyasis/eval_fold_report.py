"""
nnUNet fold 评估报告 + 可视化
从 summary.json 读指标（无需重新推理），生成：
  - report_custom.txt  (无肿瘤误报 + 综合指标 + 按 cancer_dice 分级)
  - vis_png_custom/    (每 case 若干轴向切片：GT / Pred / Diff)

自动检测数据集模式：
  liver_tumor  — label 1=肝脏, label 2=肿瘤（Dataset003）
  tumor_only   — label 1=肿瘤（Dataset004，已裁剪肝脏 ROI）

用法：
  python pumengyu/tools/analyasis/eval_fold_report.py \
    --val_dir  <results/.../fold_X/validation> \
    --gt_dir   <preprocessed/DatasetXXX/gt_segmentations> \
    --img_dir  <raw/DatasetXXX/imagesTr> \
    [--vis_slices 5] [--no_vis] [--min_tumor_size 0]
"""

from __future__ import annotations
import argparse, json, os
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ───────────────────────────── mode detection ────────────────────────────

def _detect_mode(summary_path: Path) -> str:
    """'liver_tumor' 若存在 label 2，否则 'tumor_only'。"""
    data = json.load(open(summary_path))
    for c in data.get("metric_per_case", []):
        if "2" in c.get("metrics", {}):
            return "liver_tumor"
    return "tumor_only"


# ───────────────────────────── postprocessing ────────────────────────────

def apply_min_size_filter(pred: np.ndarray, min_size: int, cls: int) -> np.ndarray:
    out = pred.copy()
    mask = pred == cls
    labeled, n = ndimage.label(mask)  # type: ignore
    for i in range(1, n + 1):
        if (labeled == i).sum() < min_size:
            out[labeled == i] = 0
    return out


def compute_case_metrics(pred: np.ndarray, gt: np.ndarray, cls: int) -> dict:
    p = pred == cls
    g = gt == cls
    tp = int((p & g).sum())
    fp = int((p & ~g).sum())
    fn = int((~p & g).sum())
    if tp == 0 and fp == 0 and fn == 0:
        return dict(dice=1.0, recall=float('nan'), precision=float('nan'),
                    fdr=0.0, pred_tumor=0, gt_tumor=0, tp=0, fp=0, fn=0)
    denom_r = tp + fn
    denom_p = tp + fp
    recall    = tp / denom_r if denom_r > 0 else float('nan')
    precision = tp / denom_p if denom_p > 0 else float('nan')
    fdr       = fp / denom_p if denom_p > 0 else 0.0
    dice      = 2 * tp / (2 * tp + fp + fn)
    return dict(dice=dice, recall=recall, precision=precision, fdr=fdr,
                pred_tumor=tp + fp, gt_tumor=tp + fn, tp=tp, fp=fp, fn=fn)


# ───────────────────────────── size category ─────────────────────────────

def size_cat(n: int) -> str:
    if n < 5_000:    return "极小(<5k)"
    if n < 50_000:   return "小(5k-50k)"
    if n < 300_000:  return "中等(50k-300k)"
    return "大(>=300k)"


# ───────────────────────────── visualization ─────────────────────────────

_OVERLAY_LIVER_TUMOR = {
    1: (0.2, 0.8, 0.2, 0.35),   # liver — 绿
    2: (1.0, 0.2, 0.2, 0.50),   # tumor — 红
}
_OVERLAY_TUMOR_ONLY = {
    1: (1.0, 0.2, 0.2, 0.50),   # tumor — 红
}


def _overlay(ax, img_s, seg_s, title, overlay_map):
    """
    vmin和vmax是CT灰度窗宽窗位设置,cls_id是分割标签的类别编号
    mask.shape=(H,W),
    """
    ax.imshow(img_s, cmap="gray", vmin=-200, vmax=250)
    for cls_id, color in overlay_map.items():
        mask = seg_s == cls_id
        if mask.any():
            rgba = np.zeros((*mask.shape, 4))
            rgba[mask] = color
            ax.imshow(rgba)
    ax.set_title(title, fontsize=7)
    ax.axis("off")


def save_visualization(case, img, gt, pred, vis_dir, n_slices, mode):
    overlay_map = _OVERLAY_LIVER_TUMOR if mode == "liver_tumor" else _OVERLAY_TUMOR_ONLY
    fg = np.where((gt > 0).any(axis=(0, 1)))[0]
    #gt>0是(H,W,D),.any(axis=(0,1))是(D,),np.where()返回非零元素的索引
    
    if len(fg) == 0:
        fg = np.arange(img.shape[2])
        #如果没有任何的切片含前景(全是背景)
        #fg:[0,1,2,...,D-1]
    indices = np.linspace(fg[0], fg[-1], n_slices, dtype=int)
    

    fig, axes = plt.subplots(n_slices, 3, figsize=(9, 3 * n_slices))
    if n_slices == 1:
        axes = axes[np.newaxis, :]
        #np.newaxis的作用类似None,在指定位置插入一个新轴,把数组维度加一

#enumerate可以作用域任何的可迭代对象,比如列表,字典,元组,字符串,numpy数组
#还有字典,range(10),enumerate会返回一个迭代器,每次迭代返回一个元组(索引,元素)
    for row, z in enumerate(indices):
        img_s = img[:, :, z].T
        gt_s  = gt[:, :, z].T
        pr_s  = pred[:, :, z].T
        _overlay(axes[row, 0], img_s, gt_s,  f"GT   z={z}", overlay_map)
        _overlay(axes[row, 1], img_s, pr_s,  f"Pred z={z}", overlay_map)
        diff = np.zeros((*img_s.shape, 4))
        diff[(pr_s > 0) & (gt_s == 0)] = (1.0, 0.5, 0.0, 0.6)   # FP orange,假阳性
        diff[(gt_s > 0) & (pr_s == 0)] = (0.0, 0.4, 1.0, 0.6)   # FN blue,假阴性
        axes[row, 2].imshow(img_s, cmap="gray", vmin=-200, vmax=250)
        axes[row, 2].imshow(diff)
        axes[row, 2].set_title(f"Diff z={z}  orange=FP  blue=FN", fontsize=7)
        axes[row, 2].axis("off")

    if mode == "liver_tumor":
        #liver_tumor有两种颜色,观看者需要知道每种颜色代表什么,所以需要这个图例说明
        #mpatches.Path是创建一个色块图例条目
        patches = [
            mpatches.Patch(color=(0.2, 0.8, 0.2), label="liver"),
            mpatches.Patch(color=(1.0, 0.2, 0.2), label="tumor"),
        ]
    else:
        patches = [mpatches.Patch(color=(1.0, 0.2, 0.2), label="tumor")]
    fig.legend(handles=patches, loc="lower center", ncol=len(patches), fontsize=8)
    fig.suptitle(case, fontsize=9)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])#type:ignore
    plt.savefig(os.path.join(vis_dir, f"{case}.png"), dpi=100, bbox_inches="tight")
    plt.close(fig)


# ───────────────────────────── report helpers ────────────────────────────

def fmt_n(n) -> str:
    return f"{int(n):,}" if n is not None else "N/A"


def _section_header(title: str) -> str:
    return "\n" + "=" * 80 + f"\n{title}\n" + "=" * 80


def _col_hdr(show_liver: bool) -> str:
    base = (f"  {'case':<20} {'tumor_dice':>10} {'recall':>8} {'precision':>10}"
            f" {'FDR':>8} {'pred_tumor':>12} {'gt_tumor':>10}")
    if show_liver:
        base += f" {'gt_liver':>10} {'size_cat':<18}"
    else:
        base += f" {'size_cat':<18}"
    return base


def _row(case, dice, recall, precision, fdr, pred_t, gt_t, gt_l, show_liver: bool):
    sc = size_cat(gt_t) if gt_t else "—"
    base = (f"  {case:<20} {dice:>10.4f} {recall:>8.4f} {precision:>10.4f}"
            f" {fdr:>8.4f} {fmt_n(pred_t):>12} {fmt_n(gt_t):>10}")
    if show_liver:
        base += f" {fmt_n(gt_l):>10} {sc:<18}"
    else:
        base += f" {sc:<18}"
    return base


SEP = "-" * 100


# ───────────────────────────── core function ─────────────────────────────

def run_eval_report(val_dir, gt_dir, img_dir,
                    vis_slices: int = 5,
                    no_vis: bool = False,
                    min_tumor_size: int = 0,
                    out_dir=None,
                    report_name: str = "report_custom.txt"):
    """
    生成 report_custom.txt 和（可选）vis_png_custom/。

    参数：
      val_dir        fold_X/validation 目录
      gt_dir         preprocessed/DatasetXXX/gt_segmentations 目录
      img_dir        raw/DatasetXXX/imagesTr 目录
      vis_slices     每个 case 可视化切片数
      no_vis         True 时跳过可视化
      min_tumor_size 后处理：去掉体素数 < 该值的 cancer 连通域（0=关闭）
      out_dir        输出目录，None 时默认写到 val_dir.parent（向后兼容）
    """
    val_dir = Path(val_dir)
    gt_dir  = Path(gt_dir)
    img_dir = Path(img_dir)
    out_dir = Path(out_dir) if out_dir is not None else val_dir.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = val_dir / "summary.json"
    if not summary_path.exists():
        print(f"[eval_fold_report] 找不到 {summary_path}")
        return

    mode = _detect_mode(summary_path)
    tumor_label = "2" if mode == "liver_tumor" else "1"
    liver_label = "1" if mode == "liver_tumor" else None
    show_liver  = (mode == "liver_tumor")
    tumor_cls   = int(tumor_label)   # 用于后处理时的 cls 参数
    print(f"[eval_fold_report] 模式: {mode}  (tumor=label{tumor_label})")

    summary    = json.load(open(summary_path))
    cases_raw  = summary["metric_per_case"]

    has_tumor, no_tumor = [], []

    for c in cases_raw:
        case    = Path(c["reference_file"]).stem.replace(".nii", "")
        m_tumor = c["metrics"].get(tumor_label, {})
        m_liver = c["metrics"].get(liver_label, {}) if liver_label else {}

        gt_tumor   = int(m_tumor.get("TP", 0) + m_tumor.get("FN", 0))
        pred_tumor = int(m_tumor.get("TP", 0) + m_tumor.get("FP", 0))
        gt_liver   = int(m_liver.get("TP", 0) + m_liver.get("FN", 0)) if liver_label else None

        liver_dice = m_liver.get("Dice") if liver_label else None
        if liver_dice is not None and np.isnan(liver_dice):
            liver_dice = None

        if gt_tumor == 0:
            no_tumor.append({
                "case":       case,
                "liver_dice": liver_dice,
                "pred_tumor": pred_tumor,
                "gt_liver":   gt_liver,
                "tumor_fp":   int(m_tumor.get("FP", 0)),
                "tumor_fn":   0,
                "liver_fp":   int(m_liver.get("FP", 0)) if liver_label else None,
                "liver_fn":   int(m_liver.get("FN", 0)) if liver_label else None,
            })
        else:
            tp2 = m_tumor.get("TP", 0)
            fp2 = m_tumor.get("FP", 0)
            fn2 = m_tumor.get("FN", 0)
            denom_r   = tp2 + fn2
            denom_p   = tp2 + fp2
            recall    = tp2 / denom_r if denom_r > 0 else 0.0
            precision = tp2 / denom_p if denom_p > 0 else 0.0
            fdr       = fp2 / denom_p if denom_p > 0 else 0.0
            fnr       = fn2 / denom_r if denom_r > 0 else 0.0
            jaccard   = tp2 / (tp2 + fp2 + fn2) if (tp2 + fp2 + fn2) > 0 else 0.0
            dice2     = m_tumor.get("Dice")
            if dice2 is None or np.isnan(dice2):
                dice2 = 0.0

            liver_fp_v = int(m_liver.get("FP", 0)) if liver_label else None
            liver_fn_v = int(m_liver.get("FN", 0)) if liver_label else None

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
                "tumor_fp":   int(fp2),
                "tumor_fn":   int(fn2),
                "liver_fp":   liver_fp_v,
                "liver_fn":   liver_fn_v,
            })

    has_tumor.sort(key=lambda x: x["dice"])

    liver_dices    = [r["liver_dice"] for r in has_tumor + no_tumor if r["liver_dice"] is not None]
    false_pos_cases = [r for r in no_tumor if r["pred_tumor"] > 0]

    # ── 汇总统计 ────────────────────────────────────────────────────────────
    lines = [
        "nnUNet Validation Report",
        "=" * 40,
        f"mode     : {mode}",
        f"fold_dir : {val_dir}",
        f"n_cases  : {len(has_tumor) + len(no_tumor)}",
        "",
    ]

    # 肝脏（仅 liver_tumor 模式）
    if show_liver and liver_dices:
        lines.append("Liver")
        lines.append(f"  Dice: mean={np.mean(liver_dices):.4f}  std={np.std(liver_dices):.4f}")
        lines.append("")

    # 肿瘤（无肿瘤 case）
    lines.append(f"Tumor (无肿瘤 case, n={len(no_tumor)})")
    if no_tumor:
        fp_rate = len(false_pos_cases) / len(no_tumor)
        no_tumor_vols = [r["pred_tumor"] for r in no_tumor]
        lines.append(f"  误报率(预测出肿瘤但GT无肿瘤): {fp_rate:.2%}  ({len(false_pos_cases)}/{len(no_tumor)} cases)")
        lines.append(f"  FP pred_tumor : mean={np.mean(no_tumor_vols):.1f}  std={np.std(no_tumor_vols):.1f}")
        if false_pos_cases:
            lines.append("  误报 cases:")
            for r in false_pos_cases:
                lines.append(f"    {r['case']:<20}  pred_tumor={fmt_n(r['pred_tumor'])}")
        else:
            lines.append("  所有无肿瘤 case 均正确预测为阴性")
        lines.append("")
        lines.append("  无肿瘤 case 列表:")
        for r in no_tumor:
            flag = "  [误报]" if r["pred_tumor"] > 0 else ""
            if show_liver and r["liver_dice"] is not None:
                lines.append(f"    {r['case']:<20}  liver_dice={r['liver_dice']:.4f}"
                             f"  pred_tumor={fmt_n(r['pred_tumor'])}{flag}")
            else:
                lines.append(f"    {r['case']:<20}  pred_tumor={fmt_n(r['pred_tumor'])}{flag}")
    else:
        lines.append("  （无此类 case）")
    lines.append("")

    # ── 综合指标（与 nnUNet summary.json 完全一致）──────────────────────────
    # 无肿瘤正确(GT=0,pred=0) → Dice=NaN → 从均值中排除（nanmean）
    # 无肿瘤误报(GT=0,pred>0) → Dice=0   → 计入均值
    # Overall = (liver_mean + tumor_mean) / 2，与 nnUNet foreground_mean 一致
    n_tn = sum(1 for r in no_tumor if r["pred_tumor"] == 0)
    n_fp = len(false_pos_cases)

    # NaN for TN, 0.0 for FP — 与 nnUNet compute_metrics_on_folder 行为一致
    no_tumor_dices    = [float("nan") if r["pred_tumor"] == 0 else 0.0 for r in no_tumor]
    no_tumor_jaccards = [float("nan") if r["pred_tumor"] == 0 else 0.0 for r in no_tumor]
    all_dices    = [r["dice"]    for r in has_tumor] + no_tumor_dices
    all_jaccards = [r["jaccard"] for r in has_tumor] + no_tumor_jaccards

    recalls    = [r["recall"]    for r in has_tumor]
    fnrs       = [r["fnr"]       for r in has_tumor]
    precisions = [r["precision"] for r in has_tumor] + [0.0] * n_fp
    fdrs       = [r["fdr"]       for r in has_tumor] + [1.0] * n_fp

    mean_tumor_dice = float(np.nanmean(all_dices))
    std_tumor_dice  = float(np.nanstd(all_dices))
    mean_liver_dice = np.mean(liver_dices) if liver_dices else float("nan")
    overall         = (mean_liver_dice + mean_tumor_dice) / 2 if show_liver else mean_tumor_dice
    n_valid         = sum(1 for d in all_dices if not np.isnan(d))

    lines.append(f"Tumor 综合指标（与 nnUNet foreground_mean 一致）")
    lines.append(f"  无肿瘤正确(TN, n={n_tn}) → Dice=NaN 排除；无肿瘤误报(FP, n={n_fp}) → Dice=0 计入")
    lines.append(f"  Dice        : mean={mean_tumor_dice:.4f}  std={std_tumor_dice:.4f}"
                 f"  (参与计算 n={n_valid}，排除 TN n={n_tn})")
    lines.append(f"  Jaccard     : mean={float(np.nanmean(all_jaccards)):.4f}"
                 f"  std={float(np.nanstd(all_jaccards)):.4f}")
    lines.append(f"  Recall      : mean={np.mean(recalls):.4f}  std={np.std(recalls):.4f}"
                 f"  (有肿瘤 n={len(recalls)})")
    lines.append(f"  FNR         : mean={np.mean(fnrs):.4f}  std={np.std(fnrs):.4f}")
    lines.append(f"  Precision   : mean={np.mean(precisions):.4f}  std={np.std(precisions):.4f}"
                 f"  (有肿瘤+误报 n={len(precisions)})")
    lines.append(f"  FDR         : mean={np.mean(fdrs):.4f}  std={np.std(fdrs):.4f}")
    if show_liver:
        lines.append(f"  Overall     : (liver {mean_liver_dice:.4f} + tumor {mean_tumor_dice:.4f}) / 2"
                     f" = {overall:.4f}  ← 与 nnUNet foreground_mean 一致")
    lines.append(f"  构成        : 有肿瘤 n={len(has_tumor)}"
                 f"  |  无肿瘤TN(排除) n={n_tn}"
                 f"  |  无肿瘤FP(计0) n={n_fp}")
    lines.append("")

    # ── 按肿瘤大小分组统计 ───────────────────────────────────────────────
    SIZE_BINS = [
        ("极小(<5k)",       lambda g: g < 5_000),
        ("小(5k-50k)",      lambda g: 5_000 <= g < 50_000),
        ("中等(50k-300k)",  lambda g: 50_000 <= g < 300_000),
        ("大(>=300k)",      lambda g: g >= 300_000),
    ]
    # ── 每个 case 的 voxel volume（mm³），从预测 nii.gz 读 spacing ────────
    vox_vol: dict[str, float] = {}
    for r in has_tumor + no_tumor:
        case = r["case"]
        nii_path = val_dir / f"{case}.nii.gz"
        if nii_path.exists():
            hdr = nib.load(str(nii_path)).header
            zooms = hdr.get_zooms()[:3]
            vox_vol[case] = float(zooms[0] * zooms[1] * zooms[2])
        else:
            vox_vol[case] = 1.0  # fallback: 体素数

    def to_mm3(case, voxels):
        return voxels * vox_vol.get(case, 1.0)

    # ── FPV / FNV 体积误差（mm³）────────────────────────────────────────
    all_cases = has_tumor + no_tumor
    t_fpv_mm3 = [to_mm3(r["case"], r["tumor_fp"]) for r in all_cases]
    t_fnv_mm3 = [to_mm3(r["case"], r["tumor_fn"]) for r in all_cases]

    lines.append("FPV / FNV 体积误差（mm³，体素数 × spacing）")
    lines.append(f"  {'':20} {'FPV总量(mm³)':>14} {'FPV均值/case':>14} {'FNV总量(mm³)':>14} {'FNV均值/case':>14}")
    lines.append(f"  {'Tumor':20} {int(sum(t_fpv_mm3)):>14,} {np.mean(t_fpv_mm3):>14,.1f}"
                 f" {int(sum(t_fnv_mm3)):>14,} {np.mean(t_fnv_mm3):>14,.1f}")
    if show_liver:
        l_fpv_mm3 = [to_mm3(r["case"], r["liver_fp"]) for r in all_cases if r.get("liver_fp") is not None]
        l_fnv_mm3 = [to_mm3(r["case"], r["liver_fn"]) for r in all_cases if r.get("liver_fn") is not None]
        lines.append(f"  {'Liver':20} {int(sum(l_fpv_mm3)):>14,} {np.mean(l_fpv_mm3):>14,.1f}"
                     f" {int(sum(l_fnv_mm3)):>14,} {np.mean(l_fnv_mm3):>14,.1f}")

    lines.append(f"\n  Per-case Tumor FPV（从高到低，前10）")
    lines.append(f"  {'case':<20} {'FPV(mm³)':>12} {'FNV(mm³)':>12} {'gt_tumor(vox)':>14} {'size_cat':<16}")
    sorted_fp = sorted(all_cases, key=lambda r: to_mm3(r["case"], r["tumor_fp"]), reverse=True)[:10]
    for r in sorted_fp:
        sc = size_cat(r.get("gt_tumor", 0)) if r.get("gt_tumor", 0) > 0 else "无肿瘤"
        lines.append(f"  {r['case']:<20} {int(to_mm3(r['case'],r['tumor_fp'])):>12,}"
                     f" {int(to_mm3(r['case'],r['tumor_fn'])):>12,}"
                     f" {fmt_n(r.get('gt_tumor',0)):>14} {sc:<16}")
    lines.append("")

    # ── 病灶级检出率（lesion-wise detection rate）───────────────────────
    from scipy.ndimage import label as _cc_label
    LESION_THRESHOLDS = [0, 10, 100, 500]
    lesion_stats: dict[int, dict] = {t: {"total": 0, "detected": 0} for t in LESION_THRESHOLDS}

    for r in all_cases:
        case = r["case"]
        gt_path   = gt_dir  / f"{case}.nii.gz"
        pred_path = val_dir / f"{case}.nii.gz"
        if not gt_path.exists() or not pred_path.exists():
            continue
        gt_arr   = nib.load(str(gt_path)).get_fdata().astype("uint8")
        pred_arr = nib.load(str(pred_path)).get_fdata().astype("uint8")
        labeled, n_comp = _cc_label(gt_arr == int(tumor_cls))
        pred_tumor_mask = (pred_arr == int(tumor_cls))
        for i in range(1, n_comp + 1):
            comp = (labeled == i)
            vol  = int(comp.sum())
            hit  = bool((comp & pred_tumor_mask).sum() > 0)
            for t in LESION_THRESHOLDS:
                if vol >= t:
                    lesion_stats[t]["total"] += 1
                    if hit:
                        lesion_stats[t]["detected"] += 1

    lines.append("病灶级检出率（lesion-wise detection rate）")
    lines.append(f"  {'最小体素阈值':>10}  {'GT病灶数':>8}  {'检出数':>8}  {'检出率':>8}  {'漏检数':>8}")
    for t in LESION_THRESHOLDS:
        st = lesion_stats[t]
        n, d = st["total"], st["detected"]
        if n == 0:
            continue
        lines.append(f"  >= {t:>4} 体素  {n:>8}  {d:>8}  {d/n*100:>7.1f}%  {n-d:>8}")
    lines.append(f"  注：>=100体素为临床有意义阈值，>=0含标注噪声碎片")
    lines.append("")

    lines.append("Tumor Dice 按大小分组 (有肿瘤 case)")
    lines.append(f"  {'大小分类':<16} {'n':>4}  {'Dice mean':>10}  {'Dice std':>9}  {'Recall':>8}  {'Precision':>10}")
    lines.append("  " + "-" * 68)
    for label, cond in SIZE_BINS:
        subset = [r for r in has_tumor if cond(r["gt_tumor"])]
        if subset:
            d_m = np.mean([r["dice"]      for r in subset])
            d_s = np.std( [r["dice"]      for r in subset])
            r_m = np.mean([r["recall"]    for r in subset])
            p_m = np.mean([r["precision"] for r in subset])
            lines.append(f"  {label:<16} {len(subset):>4}  {d_m:>10.4f}  {d_s:>9.4f}  {r_m:>8.4f}  {p_m:>10.4f}")
        else:
            lines.append(f"  {label:<16} {0:>4}  {'—':>10}")
    lines.append("")

    # ── Per-Case 分级（有肿瘤 case）────────────────────────────────────────
    lines.append(_section_header("Per-Case 分级(按 tumor_dice 从低到高)"))

    col_hdr = _col_hdr(show_liver)
    thresholds = [
        ("[严重失败] tumor_dice < 0.3",        lambda r: r["dice"] < 0.3),
        ("[需要改进] 0.3 <= tumor_dice < 0.7",  lambda r: 0.3 <= r["dice"] < 0.7),
        ("[没问题]   tumor_dice >= 0.7",         lambda r: r["dice"] >= 0.7),
    ]
    for label, cond in thresholds:
        subset = [r for r in has_tumor if cond(r)]
        lines.append(f"\n{label}  (n={len(subset)})")
        lines.append(SEP)
        lines.append(col_hdr)
        lines.append(SEP)
        for r in subset:
            lines.append(_row(r["case"], r["dice"], r["recall"], r["precision"],
                              r["fdr"], r["pred_tumor"], r["gt_tumor"],
                              r["gt_liver"], show_liver))

    # ── 无肿瘤误报 case 详情 ─────────────────────────────────────────────
    # 误报越少 → liver FN 越少 → liver Dice 越高 → Overall 越高
    if false_pos_cases:
        fp_col = (f"  {'case':<20} {'liver_dice':>10} {'pred_tumor':>12}"
                  f" {'gt_liver':>10}  说明")
        lines.append(f"\n[无肿瘤误报] tumor_dice=0，体现为 liver Dice 下降  (n={len(false_pos_cases)})")
        lines.append(SEP)
        lines.append(fp_col)
        lines.append(SEP)
        for r in sorted(false_pos_cases, key=lambda x: x["pred_tumor"], reverse=True):
            ld = f"{r['liver_dice']:.4f}" if r["liver_dice"] is not None else "N/A"
            lines.append(f"  {r['case']:<20} {ld:>10} {fmt_n(r['pred_tumor']):>12}"
                         f" {fmt_n(r['gt_liver']):>10}  GT无肿瘤,预测出{fmt_n(r['pred_tumor'])}体素")
    else:
        lines.append(f"\n[无肿瘤误报] 无  (全部 {len(no_tumor)} 个无肿瘤 case 均正确预测为阴性 ✓)")

    # ── 后处理对比（可选）────────────────────────────────────────────────
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

            before  = compute_case_metrics(pred_arr, gt_arr, cls=tumor_cls)
            pred_pp = apply_min_size_filter(pred_arr, min_tumor_size, cls=tumor_cls)
            after   = compute_case_metrics(pred_pp,   gt_arr, cls=tumor_cls)

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
                d_dice   = after["dice"]   - before["dice"]
                d_recall = after["recall"] - before["recall"]
                d_fdr    = after["fdr"]    - before["fdr"]
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

    report_path = out_dir / report_name
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[eval_fold_report] report -> {report_path}")

    # ── 可视化 ───────────────────────────────────────────────────────────
    if not no_vis:
        vis_dir = out_dir / "vis_png_custom"
        vis_dir.mkdir(exist_ok=True)
        for pred_path in sorted(val_dir.glob("*.nii.gz")):
            case = pred_path.stem.replace(".nii", "")
            gt_path  = gt_dir  / f"{case}.nii.gz"
            img_path = img_dir / f"{case}_0000.nii.gz"
            if not img_path.exists():
                img_path = img_dir / f"{case}.nii.gz"
            if not gt_path.exists() or not img_path.exists():
                print(f"  [WARN] 跳过可视化 {case}（缺 GT 或 CT）")
                continue
            print(f"  vis {case}")
            pred_arr = np.asarray(nib.load(pred_path).dataobj,  dtype=np.int16)  #type:ignore
            gt_arr   = np.asarray(nib.load(gt_path).dataobj,    dtype=np.int16) #type:ignore
            img_arr  = np.asarray(nib.load(img_path).dataobj,   dtype=np.float32) #type:ignore
            save_visualization(case, img_arr, gt_arr, pred_arr, str(vis_dir), vis_slices, mode)
        print(f"[eval_fold_report] vis   -> {vis_dir}/")


# ───────────────────────────── main ──────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--val_dir",        required=True)
    p.add_argument("--gt_dir",         required=True)
    p.add_argument("--img_dir",        required=True)
    p.add_argument("--vis_slices",     type=int, default=5)
    p.add_argument("--no_vis",         action="store_true")
    p.add_argument("--min_tumor_size", type=int, default=0,
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
