"""
分析难搞case的HU强度分布，找出指标低的原因。
覆盖全部5个fold的验证集，逐case深度分析。

运行: python pumengyu/tools/analyze_hard_cases.py
输出: pumengyu/notes/hard_cases_analysis.txt
"""

import sys
import numpy as np
import nibabel as nib
from pathlib import Path

RAW_DIR   = Path("/home/PuMengYu/nnUNet_workspace/raw/Dataset003_Liver")
RESULTS_DIR = Path("/home/PuMengYu/nnUNet_workspace/results/Dataset003_Liver/"
                   "nnUNetTrainer__nnUNetPlans__3d_fullres")
IMAGE_DIR = RAW_DIR / "imagesTr"
LABEL_DIR = RAW_DIR / "labelsTr"
OUT_FILE  = Path(__file__).parent.parent / "notes" / "hard_cases_analysis.txt"

# 每个fold的验证集（来自 splits_final.json）
FOLD_VAL = {
    0: ['liver_101','liver_11','liver_112','liver_115','liver_12','liver_120',
        'liver_128','liver_17','liver_19','liver_24','liver_25','liver_27',
        'liver_3','liver_38','liver_40','liver_41','liver_42','liver_44',
        'liver_5','liver_51','liver_52','liver_58','liver_64','liver_70',
        'liver_75','liver_77','liver_82'],
    1: ['liver_10','liver_103','liver_105','liver_106','liver_116','liver_123',
        'liver_127','liver_14','liver_16','liver_21','liver_23','liver_28',
        'liver_4','liver_45','liver_59','liver_66','liver_67','liver_68',
        'liver_7','liver_71','liver_73','liver_85','liver_86','liver_91',
        'liver_93','liver_96'],
    2: ['liver_0','liver_113','liver_114','liver_117','liver_118','liver_121',
        'liver_125','liver_13','liver_18','liver_20','liver_22','liver_26',
        'liver_32','liver_33','liver_35','liver_37','liver_43','liver_48',
        'liver_56','liver_61','liver_78','liver_79','liver_8','liver_80',
        'liver_87','liver_89'],
    3: ['liver_100','liver_102','liver_109','liver_111','liver_122','liver_126',
        'liver_130','liver_29','liver_30','liver_31','liver_36','liver_39',
        'liver_46','liver_49','liver_57','liver_6','liver_60','liver_62',
        'liver_65','liver_69','liver_81','liver_9','liver_90','liver_92',
        'liver_94','liver_97'],
    4: ['liver_1','liver_104','liver_107','liver_108','liver_110','liver_119',
        'liver_124','liver_129','liver_15','liver_2','liver_34','liver_47',
        'liver_50','liver_53','liver_54','liver_55','liver_63','liver_72',
        'liver_74','liver_76','liver_83','liver_84','liver_88','liver_95',
        'liver_98','liver_99'],
}

# 重点关注的困难case（跨fold）
FOCUS_CASES = {"liver_121", "liver_43"}


def load_case(case_id, fold):
    img_path  = IMAGE_DIR / f"{case_id}_0000.nii.gz"
    lbl_path  = LABEL_DIR / f"{case_id}.nii.gz"
    pred_dir  = RESULTS_DIR / f"fold_{fold}" / "validation"
    pred_path = pred_dir / f"{case_id}.nii.gz"

    img_nib = nib.load(img_path)
    img = img_nib.get_fdata().astype(np.float32)
    lbl = np.round(nib.load(lbl_path).get_fdata()).astype(np.uint8)
    pred = np.round(nib.load(pred_path).get_fdata()).astype(np.uint8) if pred_path.exists() else None
    voxel_vol = float(np.prod(img_nib.header.get_zooms()))
    return img, lbl, pred, voxel_vol


def hu_stats_str(values, label=""):
    if len(values) == 0:
        return f"{label}: 无数据"
    return (f"{label}: n={len(values):,}  "
            f"mean={values.mean():.1f}  median={float(np.median(values)):.1f}  "
            f"std={values.std():.1f}  "
            f"p5={float(np.percentile(values,5)):.1f}  p95={float(np.percentile(values,95)):.1f}  "
            f"[{values.min():.0f}, {values.max():.0f}]")


def analyze_case(case_id, fold, f):
    f.write(f"\n{'='*72}\n")
    star = "  ★ 重点关注" if case_id in FOCUS_CASES else ""
    f.write(f"  Case: {case_id}  (fold_{fold} val){star}\n")
    f.write(f"{'='*72}\n")

    try:
        img, lbl, pred, voxel_vol = load_case(case_id, fold)
    except FileNotFoundError as e:
        f.write(f"  [跳过] 文件不存在: {e}\n")
        return None

    liver_mask = lbl == 1
    tumor_mask = lbl == 2

    liver_hu = img[liver_mask]
    tumor_hu = img[tumor_mask]
    bg_hu    = img[~liver_mask & ~tumor_mask]

    f.write(f"\n【体积信息】\n")
    f.write(f"  图像尺寸   : {img.shape}\n")
    f.write(f"  体素大小   : {voxel_vol:.3f} mm³\n")
    f.write(f"  GT 肝脏    : {liver_mask.sum():,} 体素  ≈ {liver_mask.sum()*voxel_vol/1000:.1f} cm³\n")
    f.write(f"  GT 肿瘤    : {tumor_mask.sum():,} 体素  ≈ {tumor_mask.sum()*voxel_vol/1000:.1f} cm³\n")

    f.write(f"\n【HU 强度统计】\n")
    f.write(f"  {hu_stats_str(liver_hu, 'Liver HU')}\n")
    f.write(f"  {hu_stats_str(tumor_hu, 'Tumor HU')}\n")
    f.write(f"  {hu_stats_str(bg_hu,    'BG HU   ')}\n")

    tumor_stats = {}
    if tumor_mask.sum() > 0:
        liver_mean = float(liver_hu.mean())
        tumor_mean = float(tumor_hu.mean())
        diff = tumor_mean - liver_mean
        liver_p5,  liver_p95  = float(np.percentile(liver_hu, 5)),  float(np.percentile(liver_hu, 95))
        tumor_p5,  tumor_p95  = float(np.percentile(tumor_hu, 5)),  float(np.percentile(tumor_hu, 95))
        overlap_lo = max(liver_p5, tumor_p5)
        overlap_hi = min(liver_p95, tumor_p95)
        overlap_range = max(0.0, overlap_hi - overlap_lo)
        tumor_range   = max(tumor_p95 - tumor_p5, 1.0)
        overlap_ratio = overlap_range / tumor_range
        in_liver_range = float(((tumor_hu >= liver_p5) & (tumor_hu <= liver_p95)).mean())

        warn = "  ⚠ 高重叠，等密度病灶，难以区分" if overlap_ratio > 0.5 else ""
        f.write(f"\n【肿瘤 vs 肝脏 对比】\n")
        f.write(f"  HU 差值 (tumor-liver mean): {diff:+.1f} HU\n")
        f.write(f"  肝脏 [p5,p95]: [{liver_p5:.0f}, {liver_p95:.0f}]\n")
        f.write(f"  肿瘤 [p5,p95]: [{tumor_p5:.0f}, {tumor_p95:.0f}]\n")
        f.write(f"  重叠区间     : [{overlap_lo:.0f}, {overlap_hi:.0f}]  宽度={overlap_range:.0f} HU\n")
        f.write(f"  重叠率(对tumor范围): {overlap_ratio*100:.1f}%{warn}\n")
        f.write(f"  肿瘤体素落入肝脏HU范围内: {in_liver_range*100:.1f}%\n")

        tumor_stats = dict(tumor_mean=tumor_mean, liver_mean=liver_mean,
                           diff=diff, overlap_ratio=overlap_ratio,
                           in_liver_range=in_liver_range)

    if pred is not None:
        pred_tumor = pred == 2
        f.write(f"\n【预测结果】\n")
        f.write(f"  Pred 肿瘤  : {pred_tumor.sum():,} 体素\n")

        if tumor_mask.sum() > 0 and pred_tumor.sum() > 0:
            inter     = int((pred_tumor & tumor_mask).sum())
            dice      = 2*inter / (int(pred_tumor.sum()) + int(tumor_mask.sum()))
            recall    = inter / int(tumor_mask.sum())
            precision = inter / int(pred_tumor.sum())
            f.write(f"  Dice={dice:.4f}  Recall={recall:.4f}  Precision={precision:.4f}\n")

            fn_mask = tumor_mask & ~pred_tumor
            tp_mask = tumor_mask & pred_tumor
            if fn_mask.sum() > 0:
                fn_hu = img[fn_mask]
                tp_hu = img[tp_mask] if tp_mask.sum() > 0 else np.array([], dtype=np.float32)
                f.write(f"\n  漏检(FN)肿瘤体素 HU: {hu_stats_str(fn_hu, 'FN')}\n")
                if len(tp_hu) > 0:
                    f.write(f"  命中(TP)肿瘤体素 HU: {hu_stats_str(tp_hu, 'TP')}\n")
        elif tumor_mask.sum() > 0 and pred_tumor.sum() == 0:
            f.write(f"  ⚠ 预测完全未检出肿瘤（pred_tumor=0）\n")

    return tumor_stats


def dataset_overview(all_records, f):
    """all_records: list of dict with case_id, fold, tumor_mean, liver_mean, diff,
                    gt_tumor_vox, pred_tumor_vox, dice"""
    f.write("\n" + "="*72 + "\n")
    f.write("  全数据集（5 fold）HU 统计概览\n")
    f.write("="*72 + "\n")

    has_tumor = [r for r in all_records if r.get('gt_tumor_vox', 0) > 0]
    if not has_tumor:
        return

    liver_means  = [r['liver_mean']  for r in has_tumor]
    tumor_means  = [r['tumor_mean']  for r in has_tumor]
    diffs        = [r['diff']        for r in has_tumor]
    overlaps     = [r['overlap_ratio'] for r in has_tumor]

    f.write(f"\n  [有肿瘤 case 数量: {len(has_tumor)}]\n")
    f.write(f"  肝脏 HU mean: {np.mean(liver_means):.1f} ± {np.std(liver_means):.1f}  "
            f"[{min(liver_means):.0f}, {max(liver_means):.0f}]\n")
    f.write(f"  肿瘤 HU mean: {np.mean(tumor_means):.1f} ± {np.std(tumor_means):.1f}  "
            f"[{min(tumor_means):.0f}, {max(tumor_means):.0f}]\n")
    f.write(f"  Tumor-Liver HU差: {np.mean(diffs):.1f} ± {np.std(diffs):.1f}  "
            f"[{min(diffs):.0f}, {max(diffs):.0f}]\n")
    f.write(f"  HU重叠率: {np.mean(overlaps)*100:.1f}% ± {np.std(overlaps)*100:.1f}%  "
            f"[{min(overlaps)*100:.1f}%, {max(overlaps)*100:.1f}%]\n")

    # 按 tumor_dice 分级统计
    for label, lo, hi in [("严重失败(dice<0.3)", 0, 0.3),
                           ("需要改进(0.3-0.7)",  0.3, 0.7),
                           ("没问题(dice>=0.7)",  0.7, 1.01)]:
        grp = [r for r in has_tumor if lo <= r.get('dice', 0) < hi]
        if grp:
            g_diffs = [r['diff'] for r in grp]
            g_olap  = [r['overlap_ratio'] for r in grp]
            g_size  = [r['gt_tumor_vox'] for r in grp]
            f.write(f"\n  [{label}]  n={len(grp)}\n")
            f.write(f"    HU差: {np.mean(g_diffs):.1f} ± {np.std(g_diffs):.1f}\n")
            f.write(f"    重叠率: {np.mean(g_olap)*100:.1f}%\n")
            f.write(f"    GT肿瘤体素: {np.mean(g_size):.0f} (中位 {np.median(g_size):.0f})\n")

    f.write(f"\n\n  各case详情 (按 tumor_dice 从低到高):\n")
    header = f"  {'Case':<15} {'Fold':>5} {'tumor_dice':>10} {'HU_diff':>8} {'重叠率':>7} {'GT_tumor':>10} {'Pred_tumor':>11} {'Liver_HU':>9} {'Tumor_HU':>9}\n"
    f.write(header)
    f.write(f"  {'-'*95}\n")

    sorted_recs = sorted(all_records, key=lambda r: r.get('dice', 1.0))
    for r in sorted_recs:
        if r.get('gt_tumor_vox', 0) == 0:
            continue
        flag = " ★" if r['case_id'] in FOCUS_CASES else ""
        dice_str = f"{r.get('dice', float('nan')):.4f}"
        f.write(f"  {r['case_id']:<15} {r['fold']:>5} {dice_str:>10} "
                f"{r.get('diff', 0):>+8.1f} {r.get('overlap_ratio', 0)*100:>6.1f}% "
                f"{r.get('gt_tumor_vox', 0):>10,} {r.get('pred_tumor_vox', 0):>11,} "
                f"{r.get('liver_mean', 0):>9.1f} {r.get('tumor_mean', 0):>9.1f}{flag}\n")


def get_dice_from_pred(lbl, pred, tumor_only=True):
    if pred is None:
        return float('nan')
    if tumor_only:
        gt_m   = lbl == 2
        pred_m = pred == 2
    else:
        gt_m   = lbl > 0
        pred_m = pred > 0
    if gt_m.sum() == 0:
        return float('nan')
    inter = int((gt_m & pred_m).sum())
    return 2*inter / (int(gt_m.sum()) + int(pred_m.sum()))


if __name__ == "__main__":
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        f.write("nnUNet 难搞Case HU强度分析 — 全5 fold\n")
        f.write("重点: liver_121 (漏检), liver_43 (误报)\n")
        f.write(f"输出文件: {OUT_FILE}\n")
        f.write("="*72 + "\n")

        all_records = []

        # ── 逐fold、逐case 分析 ──────────────────────────────────────────
        for fold_idx in range(5):
            f.write(f"\n\n{'#'*72}\n")
            f.write(f"# FOLD {fold_idx}\n")
            f.write(f"{'#'*72}\n")

            cases = FOLD_VAL[fold_idx]
            for case_id in sorted(cases):
                # 收集基础信息用于汇总表
                try:
                    img, lbl, pred, voxel_vol = load_case(case_id, fold_idx)
                except FileNotFoundError:
                    continue

                gt_tumor_vox  = int((lbl == 2).sum())
                pred_tumor_vox = int((pred == 2).sum()) if pred is not None else 0
                dice = get_dice_from_pred(lbl, pred)

                rec = dict(case_id=case_id, fold=fold_idx,
                           gt_tumor_vox=gt_tumor_vox, pred_tumor_vox=pred_tumor_vox,
                           dice=dice)

                # 深度分析（写入文件）
                tstats = analyze_case(case_id, fold_idx, f)
                if tstats:
                    rec.update(tstats)
                elif gt_tumor_vox == 0:
                    # 无肿瘤case，仍记录基础肝脏HU
                    liver_hu = img[lbl == 1]
                    if len(liver_hu) > 0:
                        rec['liver_mean'] = float(liver_hu.mean())

                all_records.append(rec)
                # 实时刷出，方便观察进度
                f.flush()
                sys.stdout.write(f"\r处理: fold_{fold_idx} {case_id}   ")
                sys.stdout.flush()

        sys.stdout.write("\n")

        # ── 全局汇总 ────────────────────────────────────────────────────
        f.write("\n\n")
        dataset_overview(all_records, f)

    print(f"\n完成！结果已写入: {OUT_FILE}")
