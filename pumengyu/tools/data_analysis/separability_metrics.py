#!/usr/bin/env python3
"""
两总体可分性度量大全
════════════════════════════════════════════════════════════════
把肿瘤体素和肝脏体素各看作一个总体，计算所有主流"两分布有多难区分"的指标，
再与 OOF Dice 算 Spearman ρ，找出最强难度预测子。

指标分类
─────────────────────────────────────────────────────────────
参数化（假设正态）:
  cohen_d       Cohen's d = (μt-μl) / σ_pooled
  cnr           CNR = |μt-μl| / σ_liver
  mahal         Mahalanobis D = |μt-μl| / σ_pooled  （一维时=|d|）

非参数——分布距离:
  overlap       Σ min(pt,pl)   ← 你已有，即 Bayes 最大误判率
  tv            Total Variation = 1 - overlap = ½Σ|pt-pl|
  hellinger     Hellinger H = √(1-BC)，[0,1]
  js_div        Jensen-Shannon 散度（对称KL）
  ks_stat       KS统计量 = max|CDF_t - CDF_l|
  wasserstein   Wasserstein-1 距离（Earth Mover's Distance）

非参数——判别力:
  auc           AUC = P(Xt > Xl)，等价于 Mann-Whitney U / ntnl

分布形状:
  bimodal_coef  双峰系数 BC = (skew²+1)/kurt，>5/9 提示双峰
  kurtosis_diff 峰度差 |kurt_t - kurt_l|
  skew_diff     偏度差 |skew_t - skew_l|

用法:
    cd /home/PuMengYu/nnUNet
    python pumengyu/tools/data_analysis/separability_metrics.py
    python pumengyu/tools/data_analysis/separability_metrics.py --topk 5
"""

import argparse
import json
import math
import warnings
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.stats import spearmanr
#spearmanr专门算秩相关系数
import blosc2

warnings.filterwarnings('ignore')

# ── 路径 ────────────────────────────────────────────────────────────────────
PREPROCESSED = Path('/home/PuMengYu/nnUNet_workspace/preprocessed/Dataset003_Liver/nnUNetPlans_3d_fullres')
HU_FILE      = Path('/home/PuMengYu/nnUNet/pumengyu/notes/实验结果分析/hu_analysis.txt')
OUT_FILE     = Path('/home/PuMengYu/nnUNet/pumengyu/notes/实验结果分析/separability_metrics.txt')
OUT_JSON     = Path('/home/PuMengYu/nnUNet/pumengyu/notes/实验结果分析/separability_metrics.json')

LIVER_CLS, TUMOR_CLS = 1, 2
N_BINS = 200   # 直方图 bin 数（b2nd 存 z-score 值，不是原始 HU，用自适应 range）


# ── 从 HU 分析文件读 OOF Dice ───────────────────────────────────────────────
def load_oof_dice(path):
    """返回 {case: dice}，只用于最终体检，不进入特征计算。"""
    dice_map = {}
    in_table = False
    with open(path, encoding='utf-8') as f:
        for line in f:
            if 'fold  case' in line:
                in_table = True; continue
            if in_table and line.startswith('─'): continue
            if in_table and line.strip() == '': break
            if in_table:
                parts = line.split()
                if len(parts) < 3: continue
                try: int(parts[0])
                except ValueError: continue
                try:
                    dice_map[parts[1]] = float(parts[2])
                except (ValueError, IndexError):
                    pass
    return dice_map


# ── 从预处理文件读体素 ───────────────────────────────────────────────────────
def load_voxels(case_key):
    """读 CT 和 seg，返回 (tumor_hu, liver_hu) numpy 数组。"""
    ct_path  = PREPROCESSED / f'{case_key}.b2nd'
    seg_path = PREPROCESSED / f'{case_key}_seg.b2nd'
    if not ct_path.exists() or not seg_path.exists():
        return None, None
    ct  = blosc2.open(str(ct_path),  mode='r')[:] #type:ignore      # (1,Z,Y,X)
    seg = blosc2.open(str(seg_path), mode='r')[0]  #type:ignore       # (Z,Y,X) int
    ct  = ct[0]  #type:ignore                                         # (Z,Y,X)
    tumor_hu = ct[seg == TUMOR_CLS].astype(np.float32)#type:ignore  
    liver_hu = ct[seg == LIVER_CLS].astype(np.float32)#type:ignore  
    return tumor_hu, liver_hu


# ── 直方图工具 ───────────────────────────────────────────────────────────────
def make_hist(x, y, bins=N_BINS):
    """
    把肿瘤体素和肝脏体素各自变成一个直方图(概率分布),供后续计算overlap,Bhattacharyya,JS散度
    自适应 range：取两组数据合并后的 [p0.5, p99.5]，避免 z-score 值范围未知问题。
    """
    combined = np.concatenate([x, y])#把两个数组首尾拼接成一个
    lo, hi = np.percentile(combined, [0.5, 99.5])#lo是数据combined的第0.5%的值，hi是数据combined的第99.5%的值
    if hi <= lo: hi = lo + 1e-6
    hx, _ = np.histogram(x, bins=bins, range=(lo, hi), density=False)
    #bins是直方图的柱子数量，range是直方图的范围，density是是否归一化
    #hx是每一个区间的落入的数量,_是区间边界,
    hy, _ = np.histogram(y, bins=bins, range=(lo, hi), density=False)
    hx, hy = hx.astype(np.float64), hy.astype(np.float64)
    sx, sy = hx.sum(), hy.sum()
    return (hx / sx if sx > 0 else hx), (hy / sy if sy > 0 else hy)


# ══════════════════════════════════════════════════════════════════════════════
# 各指标计算函数
# ══════════════════════════════════════════════════════════════════════════════

def cohen_d(t, l):
    """
    Cohen's d = (μt-μl) / σ_pooled，越接近 0 越难分。
    t:该case所有的肿瘤体素的z-score值,一维数组
    l:该case所有的肝脏体素的z-score值,一维数组
    
    """
    nt, nl = len(t), len(l)
    if nt < 2 or nl < 2: return np.nan
    #.var是求方差,ddof=Degree of freedom,是自由度校正
    #ddof=1是除以n-1,无偏估计
    sp = math.sqrt(((nt-1)*t.var(ddof=1) + (nl-1)*l.var(ddof=1)) / (nt+nl-2))
    return (t.mean() - l.mean()) / sp if sp > 0 else np.nan


def cnr(t, l):
    """
    CNR = |μt-μl| / σ_liver，放射学标准可见性指标。
    CNR:Contrast-to-Noise Ratio,是对比噪声比,放射科医生用来衡量"病灶在图像中有多显眼"的标准指标
    Cohen's d 分母=两组合并标准差,肝脏和肿瘤一起
    CNR是只用肝脏标准差,只用背景噪声
    CNR越大肝脏和肿瘤对比越明显,越容易分割,Dice越高
    """
    sl = l.std(ddof=1)#求肝脏体素的标准差
    return abs(t.mean() - l.mean()) / sl if sl > 0 else np.nan


def overlap_tv(ht, hl):
    """overlap = Σmin(pt,pl)；TV = 1-overlap。"""
    ov = np.minimum(ht, hl).sum()
    #np.min(a)取一个数组a的全局最小值
    #np.minimum(a,b)逐元素比较两个数组的较小值
    return float(ov), float(1 - ov)


def bhattacharyya(ht, hl):
    """Hellinger H = √(1-BC)，BC = Σ√(pt·pl)，值域 [0,1]。"""
    bc = float(np.sqrt(ht * hl).sum())
    bc = min(max(bc, 1e-15), 1.0)
    return math.sqrt(max(0, 1 - bc))


def js_divergence(ht, hl):
    """Jensen-Shannon 散度（对称 KL），单位 nats，范围 [0, ln2]。"""
    m = (ht + hl) / 2.0
    def kl(p, q):
        mask = (p > 0) & (q > 0)
        return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))
    return 0.5 * kl(ht, m) + 0.5 * kl(hl, m)


def ks_statistic(t, l):
    """
    ks_2samp,是Kolmogorov-Smirov 双样本检验,
    KS 统计量 = max|CDF_t - CDF_l|，不需要直方图。
    """
    stat, _ = stats.ks_2samp(t, l)
    return float(stat)#type:ignore  


def wasserstein1(t, l):
    """Wasserstein-1 / Earth Mover's Distance（HU 单位）。"""
    return float(stats.wasserstein_distance(t, l))


def auc_score(t, l):
    """AUC = P(Xt > Xl)，等价于 Mann-Whitney U / (nt*nl)。
    AUC=0.5 → 等密度不可分；AUC→0或1 → 可分性高。
    取 max(auc, 1-auc) 折叠到 [0.5,1]，表示"可分程度"。"""
    if len(t) > 50000: t = np.random.choice(t, 50000, replace=False)
    if len(l) > 50000: l = np.random.choice(l, 50000, replace=False)
    u_stat, _ = stats.mannwhitneyu(t, l, alternative='two-sided')
    #U_stat=\sum_{i,j}\mathbf{1}[t_i>l_j]
    auc = u_stat / (len(t) * len(l))
    return float(max(auc, 1 - auc))   # 折叠到 [0.5,1]


def bimodality_coef(x):
    """双峰系数 = (skew²+1) / kurt。
    > 5/9 ≈ 0.556 提示双峰/重尾（均匀分布=5/9，正态=3/9=0.333）。"""
    n = len(x)
    if n < 4: return np.nan
    sk = float(stats.skew(x))#计算数组的偏度,返回一个数
    ku = float(stats.kurtosis(x, fisher=False))  # Pearson 峰度（正态=3）
    return (sk**2 + 1) / ku if ku > 0 else np.nan


# ══════════════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════════════

def compute_all(case_key):
    """对单个 case 计算所有指标，返回 dict。"""
    t, l = load_voxels(case_key)
    if t is None or len(t) < 10 or len(l) < 100:#type:ignore  
        return None

    ht, hl      = make_hist(t, l)   # 共享自适应 range

    ov, tv      = overlap_tv(ht, hl)
    hel         = bhattacharyya(ht, hl)
    auc         = auc_score(t, l)

    return {
        # 参数化
        'cohen_d':      abs(cohen_d(t, l)),   # 取绝对，大=可分
        'cnr':          cnr(t, l),
        # 非参数-分布距离（越大=越可分）
        'overlap':      ov,                   # 越小=越可分
        'tv':           tv,                   # Total Variation
        'hellinger':    hel,
        'js_div':       js_divergence(ht, hl),
        'ks_stat':      ks_statistic(t, l),
        'wasserstein':  wasserstein1(t, l),
        # 非参数-判别力（越大=越可分）
        'auc':          auc,                  # 折叠到[0.5,1]
        # 形状
        'bimodal_t':    bimodality_coef(t),
        'skew_t':       abs(float(stats.skew(t))),
    }


def spearman_with_ci(x, y, invert=False):
    """计算 Spearman ρ 及 95% CI（Fisher z 近似）。
    invert=True 时对 x 取负（把"越小越难"转换为"越大越难"方向统一）。"""
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = ~(np.isnan(x) | np.isnan(y))
    if m.sum() < 4: return np.nan, np.nan, np.nan, np.nan
    xi, yi = x[m], y[m]
    if invert: xi = -xi
    rho, p = spearmanr(xi, yi)
    n = int(m.sum())
    z = math.atanh(rho)#type:ignore  
    se = 1.0 / math.sqrt(n - 3)
    zc = 1.959964
    lo, hi = math.tanh(z - zc * se), math.tanh(z + zc * se)
    return float(rho), float(p), float(lo), float(hi)#type:ignore  


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--topk', type=int, default=999, help='只打印前k个特征')
    args = parser.parse_args()

    print('读取 OOF Dice ...')
    oof = load_oof_dice(HU_FILE)
    cases = sorted(oof.keys())
    print(f'  共 {len(cases)} 个有肿瘤 case')

    print('计算各 case 的可分性指标 ...')
    records = {}
    from tqdm import tqdm
    for c in tqdm(cases):
        r = compute_all(c)
        if r is not None:
            records[c] = r

    print(f'  成功计算 {len(records)} / {len(cases)} 个 case')

    # ── 收集特征和 dice ────────────────────────────────────────────────────
    feat_names = list(next(iter(records.values())).keys())
    dice_arr = np.array([oof[c] for c in records])

    # 指标方向说明：哪些"越大=越难"（与 dice 负相关），哪些"越小=越难"（与 dice 正相关）
    # 统一转成"越大=越难"方向，再和 dice 算相关（应为负）
    harder_when_larger = {   # True=越大越难，False=越大越容易
        'overlap':      True,   # overlap 大=分布重合=难
        'cohen_d':      False,
        'cnr':          False,
        'tv':           False,
        'hellinger':    False,
        'js_div':       False,
        'ks_stat':      False,
        'wasserstein':  False,
        'auc':          False,
        'bimodal_t':    None,   # 形状特征，方向不定
        'skew_t':       None,
    }

    results = []
    for fn in feat_names:
        #feat_names是前面提取的所有指标名字的列表
        feat_arr = np.array([records[c][fn] for c in records])
        rho, p, lo, hi = spearman_with_ci(feat_arr, dice_arr)
        results.append((fn, rho, p, lo, hi, len(records)))

    # 按 |ρ| 排序
    results.sort(key=lambda r: -abs(r[1]) if not math.isnan(r[1]) else 0)

    # ── 打印结果 ──────────────────────────────────────────────────────────
    header = f"\n{'特征':<16}{'ρ':>8}{'|ρ|':>7}{'95%CI':>18}{'p':>12}{'n':>6}"
    sep    = '  ' + '-' * 70
    lines  = ['两总体可分性指标 × OOF Dice 的 Spearman 相关（按 |ρ| 降序）', '', header, sep]

    for fn, rho, p, lo, hi in [r[:5] for r in results[:args.topk]]:
        if math.isnan(rho):
            lines.append(f"  {fn:<16}{'NaN':>8}")
            continue
        ci = f"[{lo:+.2f},{hi:+.2f}]"
        lines.append(f"  {fn:<16}{rho:>+8.3f}{abs(rho):>7.3f}{ci:>18}{p:>12.2e}{len(records):>6}")

    # CI 半宽说明
    #CI=Confidence Interval,(置信区间),95%CI是一个范围
    n_eff = len(records)
    ci_half = (math.tanh(math.atanh(0.3) + 1.96/math.sqrt(n_eff-3)) -
               math.tanh(math.atanh(0.3) - 1.96/math.sqrt(n_eff-3))) / 2
    lines += ['', f"  n={n_eff} 时 ρ=0.3 的 95%CI 半宽 ≈ {ci_half:.3f}",
              '  若最高和最低 |ρ| 之差 < CI 半宽，则统计上无法区分。', '']

    text = '\n'.join(lines)
    print(text)
    OUT_FILE.write_text(text + '\n', encoding='utf-8')

    # 保存 JSON 供后续分析（numpy 标量统一转 Python float）
    def to_py(v):
        if isinstance(v, (np.floating, np.integer)): return float(v)
        if isinstance(v, float) and math.isnan(v): return None
        return v

    out = {c: {k: to_py(v) for k, v in {'dice': oof[c], **records[c]}.items()} for c in records}
    OUT_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding='utf-8')

    print(f'\n结果已写出：\n  {OUT_FILE}\n  {OUT_JSON}')


if __name__ == '__main__':
    main()
