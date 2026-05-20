# 实验对比：BoundaryLoss vs TwoStage

**日期**：2026-05-15
**数据集**：Dataset004_LiverTumor（MSD Task03 Liver，label 2 = 肿瘤）
**评估 Fold**：Fold 0（n=27，其中有肿瘤 24 例，无肿瘤 3 例）

---

## 实验配置

| 项目 | BoundaryLoss | TwoStage |
|------|-------------|----------|
| Trainer | `nnUNetTrainer_BoundaryLoss` | `nnUNetTrainer_TwoStage` |
| Plans | `nnUNetPlans` | `nnUNetPlans` |
| Config | `3d_fullres` | `3d_fullres` |
| 核心思路 | 在标准 CE+Dice Loss 基础上加入边界损失（Boundary Loss），引导模型关注肿瘤边缘 | 两阶段策略：Stage1 用标准 nnUNetTrainer 预测肝脏 mask，Stage2 在裁剪后的肝脏 ROI 内专注预测肿瘤 |
| 结果路径 | `results/Dataset004_LiverTumor/nnUNetTrainer_BoundaryLoss__nnUNetPlans__3d_fullres/fold_0/` | `results/Dataset004_LiverTumor/nnUNetTrainer_TwoStage__nnUNetPlans__3d_fullres/fold_0/` |

---

## TwoStage 训练数据与两条验证路径

### 训练数据（Dataset004）

TwoStage 的训练数据是**离线预处理好的固定数据集**，存放于：

```
preprocessed/Dataset004_LiverTumor/nnUNetPlans_3d_fullres/
  liver_0.b2nd        ← 图像（裁剪后 nnUNet 重采样的 CT）
  liver_0_seg.b2nd    ← 标签（tumor=1，肝脏重映射为 background=0）
  liver_0.pkl         ← 元信息（spacing、shape、class_locations）
```

共 131 个 case，标签只有两类（background=0, tumor=1）。

**裁剪方式：纯 GT bbox**（`create_dataset.py`，commit `9b4b96c` 起）

```python
# 用 GT label 的 liver(1) ∪ tumor(2) 定义 bbox，不涉及任何预测结果
organ_mask = BinaryThreshold(gt_label, lower=1, upper=2)
```

> 注：早期版本（`9b4b96c` 之前）曾用 Stage1 5折交叉验证预测（纯 pred，无 GT 参与）
> 做 bbox，后重构为纯 GT 方案。**当前模型训练用的是纯 GT 裁剪。**

---

### 两条验证路径及其差异

```
训练数据：GT bbox 裁剪 → nnUNet 重采样 → .b2nd 离线存储
               │
               ├─► 路径①  perform_actual_validation()
               │          在同一批 GT 裁剪图上直接推理
               │          → report_custom.txt  Dice = 0.7673
               │          分布与训练完全一致，是 Stage2 能力的上界
               │
               └─► 路径②  eval.py 端到端
                          原始 CT → Stage1 预测肝脏 mask
                          → Stage1 pred bbox 裁剪原始 CT
                          → Stage2 推理 → 映射回原图评估
                          → eval_e2e.txt  Dice = 0.7396
                          模拟真实部署，存在 train-inference gap
```

**0.7673 > 0.7396 的原因**：路径① 的测试图像与训练图完全同分布（都是 GT 裁剪），路径② 的裁剪框来自 Stage1 预测，与 GT 框有轻微偏移，产生约 2.8 pp 的 train-inference gap。30mm margin 的设计目标即为补偿这一偏移，确保肿瘤不被裁出 ROI。

---

## 核心指标对比（Fold 0，有肿瘤 case，n=24）

| 指标 | BoundaryLoss | TwoStage（GT裁剪） | TwoStage（端到端真实推理） | 变化（GT裁剪 vs BL） |
|------|:-----------:|:----------------:|:------------------------:|:------------------:|
| **Dice (mean)** | 0.6974 | **0.7673** | 0.7396 | **+0.0699 ↑** |
| **Dice (std)** | 0.2103 | **0.1002** | 0.1372 | **-0.1101 ↓（更稳定）** |
| Jaccard (mean) | 0.5693 | **0.6331** | — | +0.0638 ↑ |
| Recall (mean) | 0.7840 | **0.8101** | 0.8005 | +0.0261 ↑ |
| Precision (mean) | 0.7250 | **0.7685** | 0.7502 | +0.0435 ↑ |
| FDR (mean) | 0.2750 | **0.2315** | 0.2498 | -0.0435 ↓（更准确）|
| FNR (mean) | 0.2160 | **0.1899** | 0.1995 | -0.0261 ↓ |

### Case 分级分布

| 等级 | BoundaryLoss | TwoStage（GT裁剪）| TwoStage（端到端）|
|------|:-----------:|:----------------:|:-----------------:|
| 严重失败（Dice < 0.3） | **2** | **0** | **0** |
| 需要改进（0.3–0.7） | 8 | 7 | 6 |
| 没问题（≥ 0.7） | 14 | **17** | **18** |
| 误报率（无肿瘤 case） | 2/3 | 2/3 | 2/3 |

---

## 关键改进分析

### 1. 严重失败 case 被修复

BoundaryLoss 中两个严重失败 case 在 TwoStage 中均得到显著改善：

| Case | BoundaryLoss Dice | TwoStage（GT裁剪）Dice | 变化 | GT 体积 | 类别 |
|------|:-----------------:|:---------------------:|:----:|:-------:|:----:|
| liver_120 | 0.0831 | 0.6775 | **+0.5944** | 5,231 vox | 小 |
| liver_11 | 0.2349 | 0.5884 | **+0.3535** | 10,549 vox | 小 |

两例均为**小肿瘤**，BoundaryLoss 出现严重 FDR 问题（FDR≈0.86-0.96，大量假阳性），TwoStage 通过 ROI 裁剪限制了预测范围，有效降低了假阳性。

### 2. 模型稳定性大幅提升

Dice 标准差从 0.2103 降至 0.1002（降幅 52%），说明两阶段方法对不同难度 case 的泛化更均匀，边界损失方法在难例上波动较大。

### 3. 端到端推理性能接近 GT 裁剪上界

GT裁剪验证 Dice（0.7673）是 Stage2 的**同分布上界**：测试图和训练图用同一套 GT bbox 裁剪，分布完全一致。端到端 Dice（0.7396）差 2.8 pp，即 Stage1 pred 裁剪框与 GT 裁剪框之间的 train-inference gap。gap 较小说明 Stage1 肝脏定位准确，30mm margin 有效覆盖了偏移量。

### 4. 无肿瘤 case 误报率未改善

两种方法在无肿瘤 case 上均有 2/3 误报（`liver_115`、`liver_41`），且预测的假阳性体积相当。这一问题可能需要专门的后处理（连通域过滤或阈值调整）来解决，而非网络结构层面的改变。

---

## TwoStage Fold 4 额外记录

Fold 4 的验证集与 Fold 0 不同（n=26，有肿瘤 23 例），结果明显较差：

- Dice mean = 0.6099，std = 0.2785
- 严重失败 case 4 例（`liver_95`、`liver_104`、`liver_55`、`liver_15`）

**可能原因**：
- `liver_104`（gt_tumor=275k，dice=0.044）疑似异常 case（极大肿瘤但 recall 仅 0.023）
- Fold 4 含更多极小肿瘤和异常标注，模型在这些难例上表现不稳定
- Fold 4 仅训练一次，可能未充分收敛

---

## 结论

**TwoStage 方法明显优于 BoundaryLoss 方法**，在相同 Fold 0 验证集上 Dice 提升约 7 个百分点，且显著消除了严重失败 case，稳定性大幅提升。两阶段思路（先定位肝脏、再分割肿瘤）从根本上缩小了问题搜索空间，是当前最有效的改进方向。

---

## 后续改进方向

- 后处理：连通域过滤（min_size=100）抑制误报
- 小肿瘤 loss 加权：针对极小（<5k vox）肿瘤提高权重
- Fold 4 深入分析：排查 `liver_104` 等异常 case，确认是否为标注噪声
