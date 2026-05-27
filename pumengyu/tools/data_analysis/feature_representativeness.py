#!/usr/bin/env python3
"""
特征代表性对比
────────────────────────────────────────────────────────────
读 intrinsic_features.json（原始体素算的 ~20 个分布级特征）+ OOF Dice，
对所有特征做 Spearman ρ 排名，找出最能预测分割失败的指标。

逻辑：
  ρ 越负 → 该特征越高时 Dice 越低 → 越能代表"难"
  |ρ| 越大 → 预测力越强
  95% CI 用于判断特征间差异是否显著

用法: python pumengyu/tools/data_analysis/feature_representativeness.py
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

INTRINSIC_FILE = Path('/home/PuMengYu/nnUNet/pumengyu/notes/实验结果分析/intrinsic_features.json')
RESULT_BASE    = Path('/home/PuMengYu/nnUNet_workspace/results/Dataset003_Liver/nnUNetTrainer__nnUNetPlans__3d_fullres')
OUT_FILE       = Path('/home/PuMengYu/nnUNet/pumengyu/notes/实验结果分析/feature_representativeness.txt')

# 按语义分组，控制输出顺序；换指标在这里增删
FEATURE_GROUPS = [
    ('可分性·分布级',  ['hist_overlap', 'bhattacharyya', 'auc', 'auc_sep']),
    ('可分性·参数化',  ['contrast', 'cohens_d', 'cnr']),
    ('分布形状',       ['skew_tumor', 'kurt_tumor', 'bimodality']),
    ('几何',           ['vol_tumor_mm3', 't_l_ratio_pct']),
    ('位置/离散',      ['mu_tumor', 'median_tumor', 'std_tumor', 'iqr_tumor']),
]


def load_intrinsic(path):
    data = json.loads(path.read_text(encoding='utf-8'))
    return {case: feats for case, feats in data.items()
            if feats.get('n_tumor_vox', 0) > 0}


def load_oof_dice(result_base, folds=range(5)):
    dice = {}
    for fold in folds:
        rpt = result_base / f'fold_{fold}' / 'report_custom.json'
        if not rpt.exists():
            continue
        for item in json.loads(rpt.read_text(encoding='utf-8')):
            d = item.get('dice_cancer')
            if d is not None:
                dice[item['case']] = float(d)
    return dice


def spear_ci(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = ~(np.isnan(x) | np.isnan(y))
    n = int(m.sum())
    if n < 4:
        return float('nan'), float('nan'), n, float('nan'), float('nan')
    res = spearmanr(x[m], y[m])
    rho_f, p_f = float(res[0]), float(res[1]) #type: ignore
    z  = math.atanh(rho_f)
    se = 1.0 / math.sqrt(n - 3)
    zc = 1.959964
    return rho_f, p_f, n, math.tanh(z - zc * se), math.tanh(z + zc * se)


def fmt_row(feat, rho, p, n, lo, hi):
    ci  = f"[{lo:+.2f},{hi:+.2f}]" if not math.isnan(lo) else '  —  '
    p_s = f'{p:.2e}' if not math.isnan(p) else '—'
    return f"  {feat:<20}{abs(rho):>7.3f}{rho:>+9.3f}{ci:>18}{p_s:>12}{n:>6}"


def main():
    meta = load_intrinsic(INTRINSIC_FILE)
    oof  = load_oof_dice(RESULT_BASE)

    common = sorted(set(meta) & set(oof))
    print(f'有肿瘤 case：{len(meta)}，有 OOF Dice：{len(oof)}，交集：{len(common)}')

    dice_arr    = np.array([oof[c] for c in common])
    all_results = []
    out         = []

    out.append('特征代表性对比 — Spearman ρ vs OOF Tumor Dice')
    out.append(f'  数据集：Dataset003_Liver，n={len(common)} 个有肿瘤 case')
    out.append('')

    hdr = f"  {'特征':<20}{'|ρ|':>7}{'ρ':>9}{'95%CI':>18}{'p':>12}{'n':>6}"
    sep = '  ' + '-' * 72

    for group_name, feats in FEATURE_GROUPS:
        out.append(f'【{group_name}】')
        out.append(hdr)
        out.append(sep)
        group_res = []
        for feat in feats:
            x = np.array([meta[c].get(feat, float('nan')) for c in common])
            rho, p, n, lo, hi = spear_ci(x, dice_arr)
            group_res.append((feat, rho, p, n, lo, hi))
            all_results.append((feat, group_name, rho, p, n, lo, hi))
        group_res.sort(key=lambda r: -abs(r[1]))
        for row in group_res:
            out.append(fmt_row(*row))
        out.append('')

    all_results.sort(key=lambda r: -abs(r[2]))
    out.append('═' * 74)
    out.append('全局排名（所有特征按 |ρ| 降序）')
    out.append(hdr)
    out.append(sep)
    for rank, (feat, _, rho, p, n, lo, hi) in enumerate(all_results, 1):
        out.append(f"  {rank:2}. {fmt_row(feat, rho, p, n, lo, hi).strip()}")

    out.append('')
    best = all_results[0]
    worst = all_results[-1]
    spread = abs(best[2]) - abs(worst[2])
    ci_half = (best[5] - best[4]) / 2.0 if not math.isnan(best[4]) else float('nan')
    out.append(f'  最强：{best[0]}（|ρ|={abs(best[2]):.3f}），最弱：{worst[0]}（|ρ|={abs(worst[2]):.3f}），差距={spread:.3f}')
    if not math.isnan(ci_half):
        out.append(f'  单个 ρ 的 95% CI 半宽 ≈ {ci_half:.2f}，差距{">" if spread > ci_half else "<"}CI半宽'
                   f' → 特征间差异{"显著" if spread > ci_half else "统计上无法区分"}')

    text = '\n'.join(out)
    print(text)
    OUT_FILE.write_text(text + '\n', encoding='utf-8')
    print(f'\n已写出：{OUT_FILE}')


if __name__ == '__main__':
    main()
