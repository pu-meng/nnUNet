#!/usr/bin/env python3
"""
内在难度计算（per-case，纯内在、零泄露）
────────────────────────────────────
只读 hu_analysis.txt 里**图像+GT 算出来的**列，给每个有肿瘤 case 算一个连续难度分，
供 offline_copypaste.py 做难度加权采样。全程不碰任何预测列（dice/recall/prec/fn/fp）。

难度 = sqrt( 可分性难度 × 稀有度 )
  · 可分性难度 = rank_norm(overlap)            # overlap 是 Dice 最强单一预测子，越高越难分辨
  · 稀有度     = rank_norm(-KDE密度)           # 在 (log体积, |contrast|) 平面越冷门 = 越长尾
所有旋钮 a priori 写死（秩归一 / Scott 带宽 / 几何平均），不在 CV 上调。

用法:
    python pumengyu/tools/compute_difficulty.py            # 算难度 → 写 difficulty.json
    python pumengyu/tools/compute_difficulty.py --check    # 额外用 OOF Dice 做一次性"体检"(只打印,不影响输出)

输出:
    pumengyu/notes/实验结果分析/difficulty.json   {case: weight}
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import gaussian_kde, rankdata, spearmanr

HU_FILE  = Path('/home/PuMengYu/nnUNet/pumengyu/notes/实验结果分析/hu_analysis.txt')
OUT_FILE = Path('/home/PuMengYu/nnUNet/pumengyu/notes/实验结果分析/difficulty.json')

# a priori 固定：保证最简单的样本也有极小概率被粘，避免权重塌成 0、保留多样性
FLOOR = 0.05


def safe_float(s):
    try:
        return float(s)
    except (ValueError, TypeError):
        return float('nan')


def parse_hu_file(path):
    """解析主表，返回 {case: {overlap, contrast, vol_mm3, cohens_d, dice_t}}。
    dice_t 仅供 --check 体检用，绝不进入难度计算。"""
    meta = {}
    in_table = False
    with open(path, encoding='utf-8') as f:
        for line in f:
            if 'fold  case' in line:
                in_table = True
                continue
            if in_table and line.startswith('─'):
                continue
            if in_table and line.strip() == '':
                break
            if in_table:
                parts = line.split()
                if len(parts) < 12:
                    continue
                try:
                    int(parts[0])
                except ValueError:
                    continue
                case = parts[1]
                meta[case] = {
                    'dice_t':   safe_float(parts[2]),   # 仅体检用
                    'contrast': safe_float(parts[8]),
                    'cohens_d': safe_float(parts[9]),
                    'overlap':  safe_float(parts[10]),
                    'vol_mm3':  safe_float(parts[11]),
                }
    return meta


def rank_norm(x):
    """秩归一到 [0,1]；可复现、对量纲鲁棒。NaN 暂记 0，调用方已过滤。"""
    x = np.asarray(x, dtype=float)
    r = rankdata(x, method='average')          # 1..n，并列取平均秩
    return (r - 1.0) / (len(x) - 1.0) if len(x) > 1 else np.zeros_like(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true',
                    help='用 OOF Dice 做一次性相关性体检（只打印，不影响 difficulty.json）')
    args = ap.parse_args()

    meta = parse_hu_file(HU_FILE)
    print(f'解析到 {len(meta)} 个有肿瘤 case')

    # 只保留三个内在特征都有效的 case（NaN 不参与）
    cases, overlap, contrast, vol = [], [], [], []
    for c, m in meta.items():
        if any(math.isnan(m[k]) for k in ('overlap', 'contrast', 'vol_mm3')):
            continue
        cases.append(c)
        overlap.append(m['overlap'])
        contrast.append(m['contrast'])
        vol.append(m['vol_mm3'])
    n = len(cases)
    print(f'特征完整、参与难度计算的 case：{n}')

    overlap  = np.array(overlap)
    abs_ct   = np.abs(np.array(contrast))
    log_vol  = np.log10(np.maximum(np.array(vol), 1.0))

    # 1) 可分性难度：overlap 越高越难
    sep_hard = rank_norm(overlap)

    # 2) 稀有度：(log体积, |contrast|) 平面上 KDE 密度低 = 长尾
    #    两轴先 z-score，避免量纲差异让某一轴主导；带宽用 scipy 默认 Scott 规则
    def z(a):
        s = a.std()
        return (a - a.mean()) / s if s > 0 else np.zeros_like(a)
    pts = np.vstack([z(log_vol), z(abs_ct)])           # (2, n)
    density = gaussian_kde(pts)(pts)                    # 每个 case 处的密度
    rarity = rank_norm(-density)                        # 密度越低 → rarity 越高

    # 3) 几何平均合成 + floor
    difficulty = np.sqrt(sep_hard * rarity)
    weight = FLOOR + (1.0 - FLOOR) * difficulty        # 缩放到 [FLOOR, 1]

    out = {c: float(w) for c, w in zip(cases, weight)}
    OUT_FILE.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'已写出：{OUT_FILE}')

    # 排序展示，便于人工 sanity check
    order = np.argsort(-weight)
    print('\n最难(权重最高) Top 10：')
    print(f"  {'case':14}{'weight':>8}{'overlap':>9}{'|contrast|':>11}{'vol_mm3':>10}")
    for i in order[:10]:
        print(f"  {cases[i]:14}{weight[i]:>8.3f}{overlap[i]:>9.3f}{abs_ct[i]:>11.1f}{vol[i]:>10.0f}")
    print('\n最易(权重最低) Bottom 10：')
    for i in order[-10:]:
        print(f"  {cases[i]:14}{weight[i]:>8.3f}{overlap[i]:>9.3f}{abs_ct[i]:>11.1f}{vol[i]:>10.0f}")

    # ── 一次性"体检"：难度分 vs 真实 OOF Dice 的相关性（仅诊断，不驱动训练）──
    if args.check:
        dice = np.array([meta[c]['dice_t'] for c in cases])
        mask = ~np.isnan(dice)
        rho, p = spearmanr(weight[mask], dice[mask])
        print('\n── 体检（OOF Dice，仅诊断，不影响输出）──')
        print(f'  难度权重 vs Dice 的 Spearman ρ = {rho:+.3f} (p={p:.2e}, n={mask.sum()})')
        print('  期望为显著负相关：难度越高，真实 Dice 越低 → 内在难度是合理代理')


if __name__ == '__main__':
    main()
