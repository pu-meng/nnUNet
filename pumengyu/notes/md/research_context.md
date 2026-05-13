# 研究背景与现状（论文辅助参考）

## 任务定义

CT 图像中的**肝脏 + 肝脏肿瘤**联合分割（两类标签：liver=1, tumor=2）。

---

## 数据集

### 训练数据：MSD Task03_Liver（LiTS）
- 131 个 case，5-fold 交叉验证
- 肿瘤阳性：96 个 case；无肿瘤（仅肝脏）：35 个 case
- 门脉期增强 CT，512×512，层厚不一

### 外部测试集（待处理）：3D-IRCADb
- 20 个 case，原始 DICOM 格式，尚未预处理
- 来自不同机构/扫描仪，用于验证跨数据集泛化能力

---

## 当前基线：nnUNet 3d_fullres，5-fold 结果

| fold | liver Dice | tumor Dice | tumor case数 |
|------|-----------|-----------|------------|
| fold_0 | 0.9652 | **0.7530** | 24 |
| fold_1 | 0.9602 | 0.6257 | 23 |
| fold_2 | 0.9509 | 0.6791 | 26 |
| fold_3 | 0.9572 | 0.6933 | 26 |
| fold_4 | 0.9505 | **0.6021** | 23 |
| **avg** | **0.9568** | **0.6706** | 96 |

- fold_0 最好（0.753），fold_4 最差（0.602），差距 0.15，variance 大
- **开发策略**：日常迭代只跑 fold_0 + fold_4（共20小时），确认方向后再跑完整5-fold

---

## 失败案例分析摘要

### 完全失败（Dice=0）的 case
| case | 原因 |
|------|------|
| liver_39 | contrast=0.0 HU，肿瘤与肝脏 HU 完全一致，物理无解 |
| liver_116 | contrast=-8.9 HU，等密度大肿瘤（61万体素），完全漏检 |
| liver_43 | 肝脏 HU=28.7（正常50-80），疑似非增强CT或严重脂肪肝，数据质量异常 |
| liver_127 | 肿瘤只有298体素，体积极小 |

### 统计发现
- 对比度（contrast = tumor_mean - liver_mean）与 Dice 有弱相关（Spearman r=0.274, p=0.007）
- 但低对比度组样本量极少（|contrast|<15 只有8个case），统计结论脆弱
- 全局均值对比不能代表局部对比度，是统计上的主要局限

---

## 核心研究方向

**目标**：写一篇以"创新解决问题"为核心的论文，指标提升是次要的。

**已排除的方向**：
- 单纯加 Transformer/Mamba 模块（已有大量工作，无明确问题驱动）
- 量子卷积（不实用）
- 靠数据量取胜（全球公开有肿瘤标签数据约1000-1500条，增量有限）

**当前寻找方向**：
- 医学图像分割中**已被文献承认**的问题，无需从自己数据证明
- 候选问题（文献中反复出现）：
  1. 肿瘤与背景的**类别极度不平衡**（小肿瘤体素远少于肝脏体素）
  2. **边界区域**分割准确率系统性低于内部区域
  3. 模型在**跨数据集/跨机构**上泛化能力差

---

## 论文写作原则（师哥建议）

> 论文最重要的是**创新解决问题**，指标没那么重要。

正确逻辑链：
```
看论文找到已被承认的问题
    → 分析现有方法在该问题上的不足
    → 提出针对性解决方案
    → 用实验证明有效
    → 自己的数据分析作为动机佐证
```

**不要**从自己数据里发现问题再找方案——样本量不够，统计上无法自证。

---

## 实验基础设施

- 框架：nnUNet v2（`/home/PuMengYu/nnUNet/`）
- 预处理数据：`/home/PuMengYu/nnUNet_workspace/preprocessed/Dataset003_Liver/`
- 训练结果：`/home/PuMengYu/nnUNet_workspace/results/Dataset003_Liver/`
- 自定义 Trainer 位置：`/home/PuMengYu/nnUNet/pumengyu/trainers/`
- GPU：单卡，训练一折约10小时

---

## 当前待办

1. 搜索论文（关键词：hard sample mining、boundary-aware loss、curriculum learning、class imbalance、liver tumor segmentation）
2. 找到一个问题方向后，在 fold_0 上快速实验验证
3. 预处理 3D-IRCADb 作为外部测试集（DICOM→NIfTI，合并多个 tumor mask）
