#!/usr/bin/env python3
"""
内在难度计算（per-case，纯内在、零泄露）
────────────────────────────────────
读 intrinsic_features.json（由 feature_extraction/extract_intrinsic_features.py 生成），
给每个有肿瘤 case 算一个连续难度分，供在线 CopyPasteMixin 做难度加权采样。

难度指标由 DIFFICULTY_FEAT 控制，修改这一行即可换指标：
  难度 = rank_norm(DIFFICULTY_FEAT)
  weight = FLOOR + (1 - FLOOR) × difficulty  → [FLOOR, 1]

用法:
    python pumengyu/tools/data_analysis/compute_difficulty.py
    python pumengyu/tools/data_analysis/compute_difficulty.py --check   # 用 OOF Dice 体检
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

INTRINSIC_FILE  = Path('/home/PuMengYu/nnUNet/pumengyu/notes/实验结果分析/intrinsic_features.json')
OUT_FILE        = Path('/home/PuMengYu/nnUNet/pumengyu/notes/实验结果分析/difficulty.json')
RESULT_BASE     = Path('/home/PuMengYu/nnUNet_workspace/results/Dataset003_Liver/nnUNetTrainer__nnUNetPlans__3d_fullres')

DIFFICULTY_FEAT = 'hist_overlap'   # ← 换指标只改这里
FLOOR = 0.05


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


def rank_norm(x):
    x = np.asarray(x, dtype=float)
    r = rankdata(x, method='average')
    return (r - 1.0) / (len(x) - 1.0) if len(x) > 1 else np.zeros_like(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true',
                    help='用 OOF Dice 做相关性体检（只打印，不影响 difficulty.json）')
    args = ap.parse_args()

    meta = load_intrinsic(INTRINSIC_FILE)
    print(f'读到 {len(meta)} 个有肿瘤 case')

    cases, feat_vals = [], []
    for c, feats in meta.items():
        v = feats.get(DIFFICULTY_FEAT, float('nan'))
        if math.isnan(v):
            continue
        cases.append(c)
        feat_vals.append(v)

    n = len(cases)
    print(f'特征 [{DIFFICULTY_FEAT}] 有效 case：{n}')

    feat_arr   = np.array(feat_vals)
    difficulty = rank_norm(feat_arr)
    weight     = FLOOR + (1.0 - FLOOR) * difficulty

    out = {c: float(w) for c, w in zip(cases, weight)}
    OUT_FILE.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'已写出：{OUT_FILE}')

    vol_arr = np.array([meta[c].get('vol_tumor_mm3', float('nan')) for c in cases])
    order   = np.argsort(-weight)
    print(f'\n最难 Top 10（{DIFFICULTY_FEAT} 越高越难）：')
    print(f"  {'case':14}{'weight':>8}{DIFFICULTY_FEAT:>14}{'vol_mm3':>10}")
    for i in order[:10]:
        print(f"  {cases[i]:14}{weight[i]:>8.3f}{feat_arr[i]:>14.3f}{vol_arr[i]:>10.0f}")
    print('\n最易 Bottom 10：')
    for i in order[-10:]:
        print(f"  {cases[i]:14}{weight[i]:>8.3f}{feat_arr[i]:>14.3f}{vol_arr[i]:>10.0f}")

    if args.check:
        oof = load_oof_dice(RESULT_BASE)
        shared = [(c, weight[i], oof[c]) for i, c in enumerate(cases) if c in oof]
        if shared:
            _, ws, ds = zip(*shared)
            rho, p = spearmanr(ws, ds)
            print(f'\n── 体检（OOF Dice，n={len(shared)}）──')
            print(f'  难度权重 vs Dice  ρ = {rho:+.3f} (p={p:.2e})')
            print('  期望：显著负相关（难度越高，Dice 越低）')


if __name__ == '__main__':
    main()
