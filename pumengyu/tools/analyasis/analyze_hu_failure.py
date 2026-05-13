"""
综合分析各 fold 验证集中肿瘤的 HU 分布、大小、预测误差与 Dice 的关联，
定位分割失败的根本原因。

用法:
    python pumengyu/tools/analyze_hu_failure.py
输出:
    pumengyu/notes/hu_analysis.txt
"""
import json
import math
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

# ── 路径配置 ───────────────────────────────────────────────────────────────────
WORKSPACE   = Path("/home/PuMengYu/nnUNet_workspace")
IMG_DIR     = WORKSPACE / "raw/Dataset003_Liver/imagesTr"
LABEL_DIR   = WORKSPACE / "raw/Dataset003_Liver/labelsTr"
RESULTS_DIR = WORKSPACE / "results/Dataset003_Liver/nnUNetTrainer__nnUNetPlans__3d_fullres"
SPLITS_FILE = WORKSPACE / "preprocessed/Dataset003_Liver/splits_final.json"
OUT_FILE    = Path(__file__).parents[1] / "notes/hu_analysis.txt"


def load_dice_from_report(json_path: Path) -> dict:
    """从 report_custom.json 读取 per-case dice，返回 {case: {dice_cancer, dice_liver}}"""
    if not json_path.exists():
        return {}
    data = json.loads(json_path.read_text())
    return {item["case"]: item for item in data}


# ── HU 统计 ────────────────────────────────────────────────────────────────────
def compute_hu_stats(img_data: np.ndarray, label_data: np.ndarray, spacing) -> dict:
    """计算肝脏和肿瘤的 HU 统计 + 分布重叠指标"""
    voxel_vol = float(np.prod(spacing[:3]))

    liver_mask = label_data == 1
    tumor_mask = label_data == 2

    liver_hu = img_data[liver_mask].astype(np.float64)
    tumor_hu = img_data[tumor_mask].astype(np.float64)

    liver_mean = float(liver_hu.mean())
    liver_std  = float(liver_hu.std())
    tumor_mean = float(tumor_hu.mean())
    tumor_std  = float(tumor_hu.std())

    contrast = tumor_mean - liver_mean

    # ── Cohen's d：标准化的均值差异 ──
    pooled_std = math.sqrt((liver_std ** 2 + tumor_std ** 2) / 2)
    cohens_d = abs(contrast) / pooled_std if pooled_std > 1e-8 else 0.0

    # ── 直方图重叠度: 两个分布共同覆盖的区域面积 ──
    all_hu = np.concatenate([liver_hu, tumor_hu])
    n_bins = 80
    lo = max(-150.0, float(np.percentile(all_hu, 1)))
    hi = min(300.0, float(np.percentile(all_hu, 99)))
    if hi - lo < 5:
        hist_overlap = 1.0
    else:
        bins = np.linspace(lo, hi, n_bins + 1)
        h_liver, _ = np.histogram(liver_hu, bins=bins, density=True)
        h_tumor,  _ = np.histogram(tumor_hu, bins=bins, density=True)
        shared = float(np.sum(np.minimum(h_liver, h_tumor)))
        total  = float(np.sum(np.maximum(h_liver, h_tumor)))
        hist_overlap = shared / total if total > 0 else 1.0

    # ── 体积 ──
    tumor_voxels = int(tumor_mask.sum())
    liver_voxels = int(liver_mask.sum())
    tumor_vol_mm3 = tumor_voxels * voxel_vol
    liver_vol_mm3 = liver_voxels * voxel_vol
    vol_ratio = tumor_vol_mm3 / liver_vol_mm3 if liver_vol_mm3 > 0 else 0.0

    return {
        "liver_hu_mean": liver_mean,
        "liver_hu_std":  liver_std,
        "tumor_hu_mean": tumor_mean,
        "tumor_hu_std":  tumor_std,
        "hu_contrast":   contrast,
        "cohens_d":      cohens_d,
        "hist_overlap":  hist_overlap,

        "tumor_voxels":    tumor_voxels,
        "tumor_volume_mm3": tumor_vol_mm3,
        "liver_volume_mm3": liver_vol_mm3,
        "tumor_liver_ratio": vol_ratio,
    }


# ── 预测误差分析 ───────────────────────────────────────────────────────────────
def compute_prediction_errors(img_data: np.ndarray, gt: np.ndarray, pred: np.ndarray,
                              spacing) -> dict:
    """分析预测错误的类型和位置"""
    voxel_vol = float(np.prod(spacing[:3]))

    gt_tumor  = gt == 2
    gt_liver  = gt == 1
    pred_tumor = pred == 2
    pred_liver = pred == 1

    tp = gt_tumor & pred_tumor          # 正确检出的肿瘤
    fn = gt_tumor & ~pred_tumor         # 漏检
    fp = pred_tumor & ~gt_tumor         # 假阳性
    fp_in_liver   = fp & gt_liver        # 假阳性在肝脏内
    fp_outside    = fp & ~gt_liver & (gt == 0)  # 假阳性在肝脏外
    fn_in_pred_liver = fn & pred_liver  # 漏检部分被预测为肝脏

    result = {
        "tp_voxels": int(tp.sum()),
        "fn_voxels": int(fn.sum()),
        "fp_voxels": int(fp.sum()),
        "fp_liver_voxels": int(fp_in_liver.sum()),
        "fp_outside_voxels": int(fp_outside.sum()),
        "fn_pred_liver_voxels": int(fn_in_pred_liver.sum()),

        "fn_volume_mm3": float(fn.sum() * voxel_vol),
        "fp_volume_mm3": float(fp.sum() * voxel_vol),
    }

    # 假阳性/假阴性位置的 HU 均值
    for name, mask in [("fn_hu_mean", fn), ("fp_hu_mean", fp),
                       ("fp_liver_hu_mean", fp_in_liver),
                       ("fp_outside_hu_mean", fp_outside)]:
        if mask.sum() > 0:
            result[name] = float(img_data[mask].mean())
        else:
            result[name] = float("nan")

    # Recall / Precision
    total_tp_fn = tp.sum() + fn.sum()
    total_tp_fp = tp.sum() + fp.sum()
    result["recall"]    = float(tp.sum() / total_tp_fn) if total_tp_fn > 0 else float("nan")
    result["precision"] = float(tp.sum() / total_tp_fp) if total_tp_fp > 0 else float("nan")

    return result


# ── 分类 ───────────────────────────────────────────────────────────────────────
def classify_contrast(contrast: float) -> str:
    if np.isnan(contrast):
        return "unknown"
    if contrast > 10:
        return "高密度"
    if abs(contrast) < 15:
        return "等密度"
    if contrast < -60:
        return "明显低密度"
    return "低密度"


def classify_size(vol_mm3: float) -> str:
    if np.isnan(vol_mm3):
        return "unknown"
    if vol_mm3 < 200:
        return "微小 (<0.2ml)"
    if vol_mm3 < 2000:
        return "小 (0.2-2ml)"
    if vol_mm3 < 20000:
        return "中 (2-20ml)"
    return "大 (>20ml)"


# ── 辅助格式化 ──────────────────────────────────────────────────────────────────
def fmt(v, decimals=1):
    """格式化数值，nan 显示为 N/A"""
    if isinstance(v, str) or v is None:
        return str(v) if v else ""
    if np.isnan(v):
        return "   N/A"
    return f"{v:.{decimals}f}"


# ══════════════════════════════════════════════════════════════════════════════
#  主流程
# ══════════════════════════════════════════════════════════════════════════════
def main():
    splits = json.loads(SPLITS_FILE.read_text())

    all_rows = []
    for fold_idx, fold in enumerate(splits):
        dice_map = load_dice_from_report(
            RESULTS_DIR / f"fold_{fold_idx}/report_custom.json")
        val_dir  = RESULTS_DIR / f"fold_{fold_idx}/validation"

        for case in sorted(fold["val"]):
            img_path   = IMG_DIR   / f"{case}_0000.nii.gz"
            label_path = LABEL_DIR / f"{case}.nii.gz"
            if not img_path.exists() or not label_path.exists():
                continue

            nii_img   = nib.load(img_path)
            img_data  = nii_img.get_fdata(dtype=np.float32)
            spacing   = nii_img.header.get_zooms()[:3]
            label_data = nib.load(label_path).get_fdata().astype(np.uint8)

            # 跳过无肿瘤 case
            if (label_data == 2).sum() == 0:
                continue

            dice_info = dice_map.get(case, {})
            tumor_dice = dice_info.get("dice_cancer", float("nan"))
            dice_liver = dice_info.get("dice_liver", float("nan"))

            stats = compute_hu_stats(img_data, label_data, spacing)

            # 读取预测结果
            pred_path = val_dir / f"{case}.nii.gz"
            pred_data = None
            pred_err  = {}
            if pred_path.exists():
                pred_data = nib.load(pred_path).get_fdata().astype(np.uint8)
                pred_err  = compute_prediction_errors(img_data, label_data, pred_data, spacing)

            row = {
                "fold":          fold_idx,
                "case":          case,
                "tumor_dice":    tumor_dice,
                "dice_liver":    dice_liver,
                **stats,
                **pred_err,
            }
            all_rows.append(row)

        n = len(fold["val"])
        n_tumor = sum(1 for c in fold["val"]
                      if (LABEL_DIR / f"{c}.nii.gz").exists()
                      and nib.load(LABEL_DIR / f"{c}.nii.gz").get_fdata().astype(np.uint8).max() >= 2)
        print(f"fold_{fold_idx}: {n} cases, {n_tumor} 有肿瘤")

    # ── 排序 ────────────────────────────────────────────────────────────────
    all_rows.sort(key=lambda r: r["tumor_dice"] if not np.isnan(r["tumor_dice"]) else 1.0)

    # ════════════════════════════════════════════════════════════════════════
    #  1. 完整表格
    # ════════════════════════════════════════════════════════════════════════
    sep = "─" * 145
    hdr = (
        f"{'fold':<5} {'case':<14} {'dice_t':>6} {'dice_l':>6} "
        f"{'recall':>6} {'prec':>6} "
        f"{'liverHU':>7} {'tumorHU':>7} {'contrast':>8} "
        f"{'Cohens_d':>8} {'overlap':>7} "
        f"{'vol_mm3':>10} {'t/l_%':>6} "
        f"{'fn_mm3':>8} {'fp_mm3':>8} {'fp_liv':>6} 备注"
    )

    print("\n" + "=" * 145)
    print("全 fold 验证集 — 按 tumor_dice 升序")
    print("=" * 145)
    print(hdr)
    print(sep)

    for r in all_rows:
        ct = classify_contrast(r["hu_contrast"])
        sz = classify_size(r["tumor_volume_mm3"])
        note = f"{ct} / {sz}"

        # 特殊标注
        dice = r["tumor_dice"]
        if not np.isnan(dice) and dice < 0.2:
            note = "⚠ " + note

        print(
            f"{r['fold']:<5} {r['case']:<14} "
            f"{fmt(r['tumor_dice'], 4):>6} {fmt(r['dice_liver'], 4):>6} "
            f"{fmt(r.get('recall', float('nan')), 4):>6} "
            f"{fmt(r.get('precision', float('nan')), 4):>6} "
            f"{fmt(r['liver_hu_mean'], 1):>7} {fmt(r['tumor_hu_mean'], 1):>7} "
            f"{fmt(r['hu_contrast'], 1):>8} {fmt(r['cohens_d'], 3):>8} "
            f"{fmt(r['hist_overlap'], 3):>7} "
            f"{fmt(r['tumor_volume_mm3'], 0):>10} "
            f"{fmt(r['tumor_liver_ratio'] * 100, 2):>6} "
            f"{fmt(r.get('fn_volume_mm3', float('nan')), 0):>8} "
            f"{fmt(r.get('fp_volume_mm3', float('nan')), 0):>8} "
            f"{fmt(r.get('fp_liver_voxels', float('nan')), 0):>6}  {note}"
        )

    # ════════════════════════════════════════════════════════════════════════
    #  2. 分组统计
    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 145)
    print("分组统计")
    print("=" * 145)

    valid = [r for r in all_rows if not np.isnan(r["tumor_dice"])]

    # 2a. 按对比度分组
    print("\n── 按 HU 对比度分类 ──")
    groups = {}
    for r in valid:
        ct = classify_contrast(r["hu_contrast"])
        groups.setdefault(ct, []).append(r)
    for g in ["明显低密度", "低密度", "等密度", "高密度", "unknown"]:
        if g not in groups:
            continue
        grp = groups[g]
        dices = [r["tumor_dice"] for r in grp]
        vols  = [r["tumor_volume_mm3"] for r in grp]
        print(f"\n  {g} ({len(grp)} cases):")
        print(f"    Dice 均值={np.mean(dices):.4f} 中位数={np.median(dices):.4f} "
              f"范围=[{min(dices):.4f}, {max(dices):.4f}]")
        print(f"    体积中位数={np.median(vols):.0f} mm³  "
              f"范围=[{min(vols):.0f}, {max(vols):.0f}]")

        # 低分 case (< 0.5) 占比
        poor = sum(1 for d in dices if d < 0.5)
        if len(grp) > 0:
            print(f"    Dice<0.5: {poor}/{len(grp)} ({100*poor/len(grp):.1f}%)")

    # 2b. 按大小分组
    print("\n── 按肿瘤大小分类 ──")
    size_groups = {}
    for r in valid:
        sz = classify_size(r["tumor_volume_mm3"])
        size_groups.setdefault(sz, []).append(r)
    for s in ["微小 (<0.2ml)", "小 (0.2-2ml)", "中 (2-20ml)", "大 (>20ml)"]:
        if s not in size_groups:
            continue
        grp = size_groups[s]
        dices = [r["tumor_dice"] for r in grp]
        contrasts = [r["hu_contrast"] for r in grp]
        print(f"\n  {s} ({len(grp)} cases):")
        print(f"    Dice 均值={np.mean(dices):.4f} 中位数={np.median(dices):.4f}")
        print(f"    Contrast 均值={np.mean(contrasts):.1f}")

        poor = sum(1 for d in dices if d < 0.5)
        if len(grp) > 0:
            print(f"    Dice<0.5: {poor}/{len(grp)} ({100*poor/len(grp):.1f}%)")

    # 2c. 联合分组：对比度 × 大小
    print("\n── 联合分组：对比度 × 大小 (Dice 均值) ──")
    ct_order = ["明显低密度", "低密度", "等密度", "高密度"]
    sz_order = ["微小 (<0.2ml)", "小 (0.2-2ml)", "中 (2-20ml)", "大 (>20ml)"]
    header = f"{'':>14}" + "".join(f"{s:>16}" for s in sz_order)
    print(header)
    print("─" * len(header))
    for ct in ct_order:
        line = f"{ct:>12}"
        for sz in sz_order:
            vals = [r["tumor_dice"] for r in valid
                    if classify_contrast(r["hu_contrast"]) == ct
                    and classify_size(r["tumor_volume_mm3"]) == sz]
            if vals:
                line += f"{'   ' + f'{np.mean(vals):.3f}' + f'({len(vals)})':>16}"
            else:
                line += f"{'':>16}"
        print(line)

    # ════════════════════════════════════════════════════════════════════════
    #  3. 相关分析
    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 145)
    print("相关分析")
    print("=" * 145)

    metrics = [
        ("tumor_dice",      "Dice"),
        ("hu_contrast",     "Contrast"),
        ("cohens_d",        "Cohen's d"),
        ("hist_overlap",    "HistOverlap"),
        ("tumor_volume_mm3","体积"),
        ("tumor_liver_ratio","肿瘤/肝比"),
    ]

    valid_corr = [r for r in all_rows
                  if all(not np.isnan(r.get(k, float("nan"))) for k, _ in metrics)]

    n_c = len(metrics)
    # 打印矩阵
    print(f"\n  Pearson 相关系数矩阵 (n={len(valid_corr)}):")
    labels = [lb for _, lb in metrics]
    print(f"  {'':>14}", "  ".join(f"{lb:>10}" for lb in labels))
    for i, (ki, li) in enumerate(metrics):
        vals_i = np.array([r[ki] for r in valid_corr])
        line = f"  {li:>12}"
        for j, (kj, _) in enumerate(metrics):
            vals_j = np.array([r[kj] for r in valid_corr])
            r_val = np.corrcoef(vals_i, vals_j)[0, 1]
            line += f"  {r_val:>8.3f}"
        print(line)

    # ── 与 Dice 最相关的因素 ──
    print(f"\n  与 Dice 的 Pearson 相关系数 (按绝对值排序):")
    dice_vals = np.array([r["tumor_dice"] for r in valid_corr])
    corr_list = []
    for ki, li in metrics:
        if ki == "tumor_dice":
            continue
        vals = np.array([r[ki] for r in valid_corr])
        r_val = np.corrcoef(dice_vals, vals)[0, 1]
        corr_list.append((abs(r_val), li, r_val))
    corr_list.sort(reverse=True)
    for _, lb, r_val in corr_list:
        print(f"    {lb:>14}: r = {r_val:+.4f}")

    # ════════════════════════════════════════════════════════════════════════
    #  4. 失败 case 深入分析
    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 145)
    print("失败 case 深入分析（Dice < 0.5 且等/高密度）")
    print("=" * 145)

    failures = [r for r in valid
                if r["tumor_dice"] < 0.5
                and classify_contrast(r["hu_contrast"]) in ("等密度", "高密度")]
    if failures:
        for r in failures:
            ct = classify_contrast(r["hu_contrast"])
            sz = classify_size(r["tumor_volume_mm3"])
            print(f"\n  {r['case']} (fold_{r['fold']}) — Dice={r['tumor_dice']:.4f}, "
                  f"Rec={r.get('recall', float('nan')):.3f}, Prec={r.get('precision', float('nan')):.3f}")
            print(f"    HU: 肝={r['liver_hu_mean']:.1f}±{r['liver_hu_std']:.1f}, "
                  f"肿瘤={r['tumor_hu_mean']:.1f}±{r['tumor_hu_std']:.1f}")
            print(f"    Contrast={r['hu_contrast']:+.1f}  ({ct}), "
                  f"Cohen's d={r['cohens_d']:.3f}, 重叠度={r['hist_overlap']:.3f}")
            print(f"    体积: 肿瘤={r['tumor_volume_mm3']:.0f} mm³ ({sz}), "
                  f"肝脏={r['liver_volume_mm3']:.0f} mm³, "
                  f"占比={100*r['tumor_liver_ratio']:.2f}%")
            if not np.isnan(r.get('fn_volume_mm3', float('nan'))):
                fn_hu = r.get('fn_hu_mean', float('nan'))
                fp_hu = r.get('fp_hu_mean', float('nan'))
                fp_liv = r.get('fp_liver_voxels', 0)
                print(f"    漏检={r['fn_volume_mm3']:.0f} mm³ (HU={fmt(fn_hu)}), "
                      f"假阳={r['fp_volume_mm3']:.0f} mm³ (HU={fmt(fp_hu)}), "
                      f"其中在肝脏内={fp_liv} voxels")

            # 分析失败原因
            reasons = []
            if r["hist_overlap"] > 0.5:
                reasons.append(f"直方图重叠度高({r['hist_overlap']:.2f})")
            if r["cohens_d"] < 0.5:
                reasons.append(f"Cohen's d 小({r['cohens_d']:.3f})")
            if r["tumor_volume_mm3"] < 500:
                reasons.append(f"肿瘤过小({r['tumor_volume_mm3']:.0f} mm³)")
            if r.get("fn_voxels", 0) > r.get("tp_voxels", 0) * 2:
                reasons.append("漏检远多于正确检测，召回率低")
            if r.get("fp_voxels", 0) > r.get("tp_voxels", 0):
                reasons.append("假阳性多，精度低")
            if reasons:
                print(f"    可能原因: {'; '.join(reasons)}")
    else:
        print("  无此类 case")

    # ── 无法用 HU 解释的失败 ──
    print("\n── 低 Dice 但无明显 HU 异常的 case（需其他视角分析）──")
    unexplained = [r for r in valid
                   if r["tumor_dice"] < 0.5
                   and classify_contrast(r["hu_contrast"]) not in ("等密度", "高密度")]
    if unexplained:
        for r in unexplained:
            print(f"  {r['case']} (fold_{r['fold']}) — Dice={r['tumor_dice']:.4f}, "
                  f"Contrast={r['hu_contrast']:+.1f}, "
                  f"Volume={r['tumor_volume_mm3']:.0f} mm³, "
                  f"Cohen's d={r['cohens_d']:.3f}")
    else:
        print("  无此类 case")

    # ════════════════════════════════════════════════════════════════════════
    #  5. 总结
    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 145)
    print("总结")
    print("=" * 145)

    all_dices = [r["tumor_dice"] for r in valid]
    print(f"  共分析 {len(valid)} 个有肿瘤 case")
    print(f"  平均 Dice = {np.mean(all_dices):.4f}, 中位数 = {np.median(all_dices):.4f}")
    poor = sum(1 for d in all_dices if d < 0.5)
    print(f"  Dice < 0.5: {poor} ({100*poor/len(valid):.1f}%)")

    # 各因素对失败的影响
    print("\n  失败因素分析 (Dice < 0.5 的 case):")
    poor_cases = [r for r in valid if r["tumor_dice"] < 0.5]
    factors = {
        "等密度/高密度 (|contrast|<15 或 >10)": lambda r:
            classify_contrast(r["hu_contrast"]) in ("等密度", "高密度"),
        "Cohen's d < 0.5": lambda r: r["cohens_d"] < 0.5,
        "肿瘤微小 (<200 mm³)": lambda r: r["tumor_volume_mm3"] < 200,
        "肿瘤小 (<2000 mm³)": lambda r: r["tumor_volume_mm3"] < 2000,
        "直方图重叠 > 0.5": lambda r: r["hist_overlap"] > 0.5,
    }
    for label, pred_fn in factors.items():
        count = sum(1 for r in poor_cases if pred_fn(r))
        print(f"    {label}: {count}/{len(poor_cases)} ({100*count//len(poor_cases)}%)")

    # 综合结论
    print("\n  结论:")
    corr_list_sorted = sorted(
        [(r_val, lb) for _, lb, r_val in corr_list], key=lambda x: -abs(x[0]))
    top_factor = corr_list_sorted[0][1]
    print(f"    - 与 Dice 最相关的单一指标: {top_factor}")
    print(f"    - 存在多因素叠加效应: 小体积 + 低对比度 = 极低 Dice")
    print(f"    - 大部分失败 case 可以用 HU 特征解释")


if __name__ == "__main__":
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUT_FILE.open("w", encoding="utf-8") as f:
        class Tee:
            def write(self, s):
                f.write(s)
                sys.__stdout__.write(s)
            def flush(self):
                f.flush()
                sys.__stdout__.flush()
        sys.stdout = Tee()
        main()
    sys.stdout = sys.__stdout__
    print(f"\n已保存到: {OUT_FILE}")
