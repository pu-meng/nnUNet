#!/usr/bin/env python3
"""
原始体素 内在特征提取器（per-case，纯内在、零泄露）
──────────────────────────────────────────────────
直接读【原始 CT(原始 HU) + GT 标签】，对每个 case 用**完整的 HU 分布**算一整套
分布级特征 —— 而不是像 hu_analysis.txt 那样把每个特征压成一个均值。
这是后续所有分析（代表性对比 / 难度 / size_density）的统一地基。

数据来源（都不碰任何模型预测 → 零泄露）：
  CT : raw/Dataset003_Liver/imagesTr/{case}_0000.nii.gz   (原始 HU)
  GT : preprocessed/Dataset003_Liver/gt_segmentations/{case}.nii.gz  (1=肝, 2=肿瘤)

算的特征（per-case，肿瘤 vs 肝 的 HU 分布）：
  位置      mu / median (肿瘤、肝)
  离散      std / iqr  (肿瘤、肝)
  均值差    contrast = mu_肿瘤 - mu_肝            ← 最朴素(老师说"不具代表性")
  方差感知  cohens_d = contrast/合并std ; cnr = contrast/std_肝   ← 放射学标准
  分布距离  hist_overlap ; bhattacharyya ; auc(=P(肿瘤体素更亮)) ; auc_sep=|auc-0.5|*2
  形状      skew / kurt(肿瘤) ; bimodality(Sarle 系数,>0.555 疑双峰→坏死核+强化环)
  几何      vol_mm3(肿瘤、肝) ; t_l_ratio_pct

输出：
  notes/实验结果分析/intrinsic_features.csv    (全特征,可直接 Excel 看)
  notes/实验结果分析/intrinsic_features.json   ({case:{feat:val}},喂下游脚本)

用法:
  python pumengyu/tools/feature_extraction/extract_intrinsic_features.py
  python pumengyu/tools/feature_extraction/extract_intrinsic_features.py --limit 5   # 快速试跑
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy.stats import skew, kurtosis

# ── 路径与常量 ────────────────────────────────────────────────────────────────
CT_DIR  = Path('/home/PuMengYu/nnUNet_workspace/raw/Dataset003_Liver/imagesTr')
GT_DIR  = Path('/home/PuMengYu/nnUNet_workspace/preprocessed/Dataset003_Liver/gt_segmentations')
OUT_CSV  = Path('/home/PuMengYu/nnUNet/pumengyu/notes/实验结果分析/intrinsic_features.csv')
OUT_JSON = Path('/home/PuMengYu/nnUNet/pumengyu/notes/实验结果分析/intrinsic_features.json')

LIVER_CLS = 1
TUMOR_CLS = 2
N_BINS    = 100   # 直方图 bin 数（overlap / bhattacharyya / auc 用）

# 输出列顺序
COLUMNS = [
    'case', 'n_tumor_vox', 'n_liver_vox', 'vol_tumor_mm3', 'vol_liver_mm3', 't_l_ratio_pct',
    'mu_tumor', 'mu_liver', 'median_tumor', 'median_liver',
    'std_tumor', 'std_liver', 'iqr_tumor', 'iqr_liver',
    'contrast', 'cohens_d', 'cnr',
    'hist_overlap', 'bhattacharyya', 'auc', 'auc_sep',
    'skew_tumor', 'kurt_tumor', 'bimodality',
]


def read_volume(path):
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)            # (z, y, x)
    sx, sy, sz = img.GetSpacing()                # (x, y, z) mm
    return arr, sx * sy * sz                      # 数组 + 单体素体积(mm³)


def find_ct(case):
    p = CT_DIR / f'{case}_0000.nii.gz'
    if p.exists():
        return p
    hits = sorted(CT_DIR.glob(f'{case}_*.nii.gz'))
    return hits[0] if hits else None


def _hist_separability(t, l, nbins):
    """在共同 HU 区间上做归一化直方图，算 overlap / Bhattacharyya / AUC。"""
    lo = min(t.min(), l.min())
    hi = max(t.max(), l.max())
    if hi <= lo:
        return float('nan'), float('nan'), float('nan')
    edges = np.linspace(lo, hi, nbins + 1)
    pt, _ = np.histogram(t, bins=edges, density=False)
    pl, _ = np.histogram(l, bins=edges, density=False)
    pt = pt / pt.sum()
    pl = pl / pl.sum()
    overlap = float(np.minimum(pt, pl).sum())                 # 直方图交集 [0,1]
    bc      = float(np.sqrt(pt * pl).sum())                   # Bhattacharyya 系数 [0,1]
    # AUC = P(肿瘤体素 > 肝体素) + 0.5 P(相等)，由直方图累积算
    cum_l_below = np.concatenate([[0.0], np.cumsum(pl)[:-1]])  # 每个 bin 之前的肝累积
    auc = float((pt * (cum_l_below + 0.5 * pl)).sum())
    return overlap, bc, auc


def _basic(vals):
    """单分布的位置/离散/形状统计。"""
    if vals.size < 2:
        nan = float('nan')
        return dict(mu=nan, median=nan, std=nan, iqr=nan, skew=nan, kurt=nan)
    q25, q75 = np.percentile(vals, [25, 75])
    return dict(
        mu=float(vals.mean()), median=float(np.median(vals)),
        std=float(vals.std()), iqr=float(q75 - q25),
        skew=float(skew(vals)), kurt=float(kurtosis(vals, fisher=False)),
    )


def compute_features(case):
    gt, vox_vol = read_volume(GT_DIR / f'{case}.nii.gz')
    ct_path = find_ct(case)
    if ct_path is None:
        return None
    ct, _ = read_volume(ct_path)
    if ct.shape != gt.shape:
        print(f'  [跳过] {case}: CT{ct.shape} 与 GT{gt.shape} 形状不一致')
        return None

    tumor = ct[gt == TUMOR_CLS].astype(np.float64)
    liver = ct[gt == LIVER_CLS].astype(np.float64)
    nan = float('nan')

    f = {c: nan for c in COLUMNS}
    f['case'] = case
    f['n_tumor_vox'] = int(tumor.size)
    f['n_liver_vox'] = int(liver.size)
    f['vol_tumor_mm3'] = float(tumor.size * vox_vol)
    f['vol_liver_mm3'] = float(liver.size * vox_vol)
    f['t_l_ratio_pct'] = float(tumor.size / liver.size * 100) if liver.size else nan

    bl = _basic(liver)
    f['mu_liver'], f['median_liver'] = bl['mu'], bl['median']
    f['std_liver'], f['iqr_liver'] = bl['std'], bl['iqr']

    if tumor.size == 0:        # 无肿瘤 case：肿瘤/可分性特征留 NaN（顺带得到无肿瘤名单）
        return f

    bt = _basic(tumor)
    f['mu_tumor'], f['median_tumor'] = bt['mu'], bt['median']
    f['std_tumor'], f['iqr_tumor'] = bt['std'], bt['iqr']
    f['skew_tumor'], f['kurt_tumor'] = bt['skew'], bt['kurt']
    # Sarle 双峰系数 = (skew² + 1) / kurt；>0.555 提示双峰
    f['bimodality'] = (bt['skew'] ** 2 + 1) / bt['kurt'] if bt['kurt'] and not math.isnan(bt['kurt']) else nan

    f['contrast'] = bt['mu'] - bl['mu']
    pooled = math.sqrt((bt['std'] ** 2 + bl['std'] ** 2) / 2) if (bt['std'] or bl['std']) else nan
    f['cohens_d'] = f['contrast'] / pooled if pooled else nan
    f['cnr'] = f['contrast'] / bl['std'] if bl['std'] else nan

    overlap, bc, auc = _hist_separability(tumor, liver, N_BINS)
    f['hist_overlap'] = overlap
    f['bhattacharyya'] = bc
    f['auc'] = auc
    f['auc_sep'] = abs(auc - 0.5) * 2 if not math.isnan(auc) else nan
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0, help='只处理前 N 个 case(调试用,0=全部)')
    args = ap.parse_args()

    cases = sorted(p.name[:-len('.nii.gz')] for p in GT_DIR.glob('*.nii.gz'))
    if args.limit:
        cases = cases[:args.limit]
    print(f'共 {len(cases)} 个 case，开始提取...')

    rows, n_tumor, n_notumor = [], 0, 0
    for i, case in enumerate(cases, 1):
        f = compute_features(case)
        if f is None:
            continue
        rows.append(f)
        if f['n_tumor_vox'] > 0:
            n_tumor += 1
        else:
            n_notumor += 1
        if i % 10 == 0 or i == len(cases):
            print(f'  [{i}/{len(cases)}] {case}')

    # 写 CSV
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, 'w', encoding='utf-8') as fp:
        fp.write(','.join(COLUMNS) + '\n')
        for r in rows:
            fp.write(','.join(
                ('' if (isinstance(r[c], float) and math.isnan(r[c])) else str(r[c]))
                for c in COLUMNS) + '\n')

    # 写 JSON（去掉 case 键放外层）
    out_json = {r['case']: {c: r[c] for c in COLUMNS if c != 'case'} for r in rows}
    OUT_JSON.write_text(json.dumps(out_json, ensure_ascii=False, indent=2), encoding='utf-8')

    print(f'\n完成：{len(rows)} 个 case（有肿瘤 {n_tumor}，无肿瘤 {n_notumor}）')
    print(f'  CSV : {OUT_CSV}')
    print(f'  JSON: {OUT_JSON}')


if __name__ == '__main__':
    main()
