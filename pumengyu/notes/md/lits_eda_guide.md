# LiTS 肝脏肿瘤 EDA 分析手册

> 目标：用统计分布替代HU均值，系统性地理解数据、发现问题、找到改进方向。

---

## 环境准备

```bash
pip install nibabel scipy matplotlib scikit-learn tqdm pandas
```

---

## 第一层：数据集层 — 扫描仪差异

**要回答的问题**：不同病例的肝脏HU是否可比？是否存在扫描仪漂移？

```python
import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm
import glob, os

def load_pair(vol_path, seg_path):
    vol = nib.load(vol_path).get_fdata().astype(np.float32)
    seg = nib.load(seg_path).get_fdata().astype(np.uint8)
    return np.clip(vol, -200, 300), seg

records = []
for vol_path in tqdm(sorted(glob.glob('/your/lits/path/volume-*.nii*'))):
    idx = os.path.basename(vol_path).replace('volume-','').split('.')[0]
    seg_path = vol_path.replace(f'volume-{idx}', f'segmentation-{idx}')
    vol, seg = load_pair(vol_path, seg_path)
    
    liver_hu = vol[(seg == 1) & (seg != 2)]   # 纯肝脏，排除肿瘤
    if len(liver_hu) < 100:
        continue
    
    records.append({
        'case_id':      idx,
        'liver_median': np.median(liver_hu),
        'liver_iqr':    np.percentile(liver_hu,75) - np.percentile(liver_hu,25),
        'liver_p5':     np.percentile(liver_hu, 5),
        'liver_p95':    np.percentile(liver_hu,95),
    })

df = pd.DataFrame(records)
print(df['liver_median'].describe())
```

**判读标准**：
- 正常肝脏中位数应在 **50–70 HU**
- 如果跨病例标准差 > 15 HU → 存在扫描仪漂移，需要 HU 标准化
- IQR 过大（> 40 HU）→ 该病例肝脏本身有病变（脂肪肝、肝硬化）

---

## 第二层：病例层 — 肿瘤负荷分布

**要回答的问题**：肿瘤大小、数量的分布是什么样的？极端病例在哪里？

```python
from scipy.ndimage import label as cc_label

for vol, seg, case_id in cases:
    tumor_mask = seg == 2
    
    # 连通域分析：找出独立肿瘤个数
    labeled, n_tumors = cc_label(tumor_mask)
    
    # 每个肿瘤的体积
    zooms = nib.load(vol_path).header.get_zooms()[:3]
    voxel_vol_cc = np.prod(zooms) / 1000  # mm³ → cc
    
    tumor_sizes = [
        (labeled == i).sum() * voxel_vol_cc
        for i in range(1, n_tumors + 1)
    ]
    
    records.append({
        'case_id':       case_id,
        'n_tumors':      n_tumors,
        'total_vol_cc':  sum(tumor_sizes),
        'max_vol_cc':    max(tumor_sizes) if tumor_sizes else 0,
        'min_vol_cc':    min(tumor_sizes) if tumor_sizes else 0,
    })
```

**关注点**：
- 小肿瘤（< 1 cc）是分割最难的群体，单独统计
- 多发肿瘤病例（n_tumors > 3）往往是转移瘤，HU模式不同

---

## 第三层：肿瘤层 — HU分布指纹

**核心思路**：用 7 个统计量替代 1 个均值，描述每个肿瘤的"分布形状"。

```python
from scipy import stats

def tumor_fingerprint(hu_values):
    """
    替代HU均值的完整分布描述
    hu_values: 肿瘤区域所有体素的HU值数组
    """
    if len(hu_values) < 10:
        return None
    
    hist, _ = np.histogram(hu_values, bins=50, range=(-200, 300))
    
    # 峰值计数（双峰 = 坏死区 + 活性区共存）
    from scipy.ndimage import uniform_filter1d
    h_smooth = uniform_filter1d(hist.astype(float), size=3)
    peaks = np.where(
        (h_smooth[1:-1] > h_smooth[:-2]) &
        (h_smooth[1:-1] > h_smooth[2:]) &
        (h_smooth[1:-1] > h_smooth.max() * 0.05)
    )[0]
    
    return {
        # 鲁棒集中趋势（比均值更稳定）
        'median':   np.median(hu_values),
        'p25':      np.percentile(hu_values, 25),
        'p75':      np.percentile(hu_values, 75),
        
        # 分散程度
        'iqr':      np.percentile(hu_values,75) - np.percentile(hu_values,25),
        'std':      np.std(hu_values),
        
        # 分布形状（均值完全丢失这些）
        'skewness': stats.skew(hu_values),    # 负偏 → 坏死低HU拖左尾
        'kurtosis': stats.kurtosis(hu_values), # 高峰度 → 单一成分
        'entropy':  stats.entropy(hist + 1e-10), # 越高 → 异质性越强
        'n_peaks':  len(peaks),               # ≥2 → 双峰，肿瘤内部混合
    }
```

**判读指南**：

| 特征 | 低值含义 | 高值含义 |
|------|---------|---------|
| IQR | HU集中，成分单一 | HU分散，内部混合 |
| 偏度 < 0 | 左尾长，坏死低HU区 | — |
| 熵 | 均匀组织 | 异质性高（难分割）|
| 峰值数 ≥ 2 | — | 双峰：活性+坏死共存 |
| 中位数 ≈ 肝脏中位数 | — | **低对比度**，最难 |

---

## 第四层：边界层 — 困难度量化

**要回答的问题**：边界处肿瘤与肝脏的HU有多难区分？这直接决定你的boundary-aware loss要多重。

```python
from scipy.ndimage import binary_erosion, binary_dilation

def boundary_difficulty(vol, tumor_mask, dilation_r=3):
    """
    返回边界困难度指标
    overlap 越接近 1 → 边界越模糊 → 越难分割
    """
    if tumor_mask.sum() < 20:
        return None
    
    # 边界体素 = 肿瘤mask - 腐蚀后的mask
    eroded   = binary_erosion(tumor_mask, iterations=2)
    boundary = tumor_mask & (~eroded)
    
    # 周围肝脏 = 膨胀后的环形区域
    dilated  = binary_dilation(tumor_mask, iterations=dilation_r)
    surround = dilated & (~tumor_mask)
    
    if boundary.sum() < 5 or surround.sum() < 5:
        return None
    
    b_hu = vol[boundary]
    s_hu = vol[surround]
    
    # HU分布重叠面积（核心困难度指标）
    bins  = np.linspace(-200, 300, 81)
    ha, _ = np.histogram(b_hu, bins=bins, density=True)
    hs, _ = np.histogram(s_hu, bins=bins, density=True)
    overlap = float(np.sum(np.minimum(ha, hs)) * (bins[1] - bins[0]))
    
    return {
        'boundary_median': np.median(b_hu),
        'surround_median': np.median(s_hu),
        'hu_contrast':     abs(np.median(b_hu) - np.median(s_hu)),  # 越大越好
        'overlap':         overlap,  # 越小越好（0=完全可分，1=完全重叠）
        'n_boundary_vox':  int(boundary.sum()),
    }
```

**与Dice的关联验证**：

```python
# 如果你已经有baseline nnUNet的预测结果
# 把每个病例的 overlap 和 Dice 做散点图
# 预期：overlap 越高 → Dice 越低
# 这就是你改进loss的定量依据

import matplotlib.pyplot as plt

plt.scatter(df['bd_overlap'], df['dice_score'], alpha=0.6)
plt.xlabel('边界HU重叠度（困难度）')
plt.ylabel('Dice score')
plt.title('困难度 vs 分割质量')
plt.savefig('difficulty_vs_dice.png', dpi=150)
```

---

## 第五步：找出困难病例的规律

把四层特征汇总，通过相关分析定位问题根源：

```python
# 假设你有baseline Dice
df['dice'] = [...]  # 填入每个病例的nnUNet baseline dice

# 与困难度的相关性
feature_cols = [
    'tumor_iqr',        # 肿瘤内部异质性
    'tumor_entropy',    # 分布复杂度
    'bd_overlap',       # 边界模糊程度
    'bd_hu_contrast',   # 肿瘤-肝脏对比度
    'tumor_median',     # HU绝对值（是否接近肝脏）
    'min_vol_cc',       # 最小肿瘤体积
]

corr = df[feature_cols + ['dice']].corr()['dice'].drop('dice').sort_values()
print(corr)
# 负相关最强的特征 → 导致低Dice的主要因素
```

**预期发现**：

```
bd_overlap       -0.65   ← 边界模糊是最主要问题
tumor_iqr        -0.48   ← 肿瘤异质性
min_vol_cc        0.52   ← 小肿瘤更难（正相关）
bd_hu_contrast    0.44   ← 对比度高更容易
```

---

## 输出整理建议

跑完后整理成这张表，每行一个病例：

| case_id | liver_median | tumor_median | tumor_iqr | bd_overlap | n_tumors | min_vol_cc | dice |
|---------|-------------|-------------|----------|-----------|---------|-----------|------|
| 0       | 58.2        | 45.1        | 38.4     | 0.71      | 2       | 0.4       | 0.62 |
| 1       | 62.1        | 12.3        | 89.2     | 0.31      | 1       | 8.2       | 0.89 |

`bd_overlap` 高 + `tumor_iqr` 高 + `min_vol_cc` 小 的病例 → 你的boundary-aware loss重点优化对象。

---

## 下一步决策树

```
bd_overlap > 0.5 (多数病例)
    → 边界模糊是主因
    → boundary-aware loss 权重加大，focus区域在边界体素

tumor_n_peaks >= 2 (多峰，异质性强)
    → 肿瘤内部成分复杂
    → 考虑在loss中区分核心区 vs 边界区

min_vol_cc < 1cc 的病例 Dice 明显低
    → 小目标问题
    → patch sampling策略调整，确保小肿瘤被采样到

liver_median 跨病例标准差 > 15
    → 扫描仪差异显著
    → 预处理加 z-score normalization per case
```
