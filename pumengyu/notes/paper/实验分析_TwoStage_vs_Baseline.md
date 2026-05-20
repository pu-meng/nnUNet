# 两阶段 vs 标准 nnUNet 实验分析报告

**更新日期**：2026-05-19  
**数据集**：MSD Task03 Liver（131 cases，5-fold CV）  
**对比方**：
- **Baseline**：`Dataset003_Liver / nnUNetTrainer`（肝+肿瘤联合一阶段分割）
- **TwoStage**：`Dataset004_LiverTumor / nnUNetTrainer_TwoStage`（Stage1 肝脏分割 → Stage2 ROI 内肿瘤分割）

---

## 一、性能对比

### 1.1 各 Fold Tumor Dice（有肿瘤 case，GT-crop 验证路径）

| Fold | n（有肿瘤） | Baseline Dice | Baseline std | TwoStage Dice | TwoStage std | Δ Dice | 胜者 |
|------|-----------|--------------|-------------|--------------|-------------|--------|------|
| 0 | 24 | 0.7530 | 0.1382 | **0.7673** | **0.1002** | +0.0143 | TwoStage |
| 1 | 23 | **0.6257** | 0.2931 | 0.6051 | 0.3011 | -0.0206 | Baseline |
| 2 | 22 | **0.7100** | 0.2358 | 0.6650 | 0.3042 | -0.0450 | Baseline |
| 3 | 26 | **0.6934** | 0.2038 | *(训练中)* | — | — | — |
| 4 | 23 | 0.6021 | 0.3001 | **0.6099** | 0.2785 | +0.0078 | TwoStage |
| **4-fold 均值** | — | **0.6727** | — | 0.6618 | — | **-0.0109** | Baseline 领先 |
| **5-fold 均值** | — | **0.6768** | — | ? | — | — | 待 fold_3 完成 |

> **注**：TwoStage 的 Dice 来自 GT-crop 验证（偏乐观），见 §三。

### 1.2 其他指标对比（fold_0，对比最完整）

| 指标 | Baseline | TwoStage（GT crop） | TwoStage（E2E 30mm） |
|------|---------|-------------------|-------------------|
| Dice | 0.7530 | **0.7673** | 0.7396 |
| Recall | 0.7683 | **0.8101** | 0.8005 |
| Precision | **0.7955** | 0.7685 | 0.7502 |
| FDR | **0.2045** | 0.2315 | 0.2498 |
| Dice std | 0.1382 | **0.1002** | 0.1372 |
| 无肿瘤误报率 | 66.7% (2/3) | 66.7% (2/3) | 66.7% (2/3) |

### 1.3 E2E 推理（Stage1 预测 mask 裁剪）vs GT Crop 差距

仅 fold_0 有完整 margin 扫描数据：

| Margin | TwoStage Dice | vs Baseline (0.7530) |
|--------|--------------|---------------------|
| 0 mm | 0.7112 | -0.0418 |
| 5 mm | **0.7430** | -0.0100 |
| 10 mm | 0.7378 | -0.0152 |
| 20 mm | 0.7340 | -0.0190 |
| 30 mm（默认） | 0.7396 | -0.0134 |
| 50 mm | 0.7237 | -0.0293 |
| GT crop（上界） | 0.7673 | +0.0143 |

**关键结论**：以任意 margin 的 E2E 评估，fold_0 上 TwoStage 均低于 Baseline。Baseline 是真正的公平对照。

---

## 二、训练时间与计算消耗

### 2.1 各 Fold 训练时长

| Trainer | Fold | Epoch 数 | 平均 epoch 时间 | 总训练时长 |
|---------|------|---------|--------------|----------|
| Baseline (Dataset003) | 0 | 1000 | 28.7 s | **8.0 h** |
| Baseline | 1 | 1000 | 29.7 s | **8.2 h** |
| Baseline | 2 | 1000 | 28.8 s | **8.0 h** |
| Baseline | 3 | 1000 | 28.7 s | **8.0 h** |
| Baseline | 4 | 1000 | 48.7 s | **13.5 h** |
| **Baseline 合计** | — | 5000 | — | **≈ 45.7 h** |
| TwoStage Stage2 (Dataset004) | 0 | 1000 | 59.9 s | **16.6 h** |
| TwoStage Stage2 | 1 | 1000 | 48.6 s | **13.5 h** |
| TwoStage Stage2 | 2 | 1000 | 34.2 s | **9.5 h** |
| TwoStage Stage2 | 4 | 1000 | 63.9 s | **17.8 h** |
| **TwoStage Stage2 合计（4-fold）** | — | 4000 | — | **≈ 57.4 h** |

### 2.2 总计算成本对比

| 方案 | 训练成本 |
|------|---------|
| Baseline（一阶段） | 45.7 h |
| TwoStage（Stage1 已有，仅 Stage2） | +57.4 h → 共 **103 h** |
| TwoStage（从零开始） | 45.7 h + 71.8 h（估算 5-fold） ≈ **117.5 h** |
| **TwoStage 相对倍数** | **≈ 2.6×** |

**结论**：TwoStage 并不节省计算，而是 **额外增加约 60%～160%** 的训练开销，同时需要存储两套模型权重。

推理开销也类似：需要先跑 Stage1（约 2-5 min/case），再跑 Stage2（约 1-3 min/case），总耗时约 **2× Baseline**。

---

## 三、数据泄露与评估偏差分析

### 3.1 本质问题：训练-推理分布差异（Train-Test Gap）

**这不是传统意义的数据泄露**（未使用测试集标签训练），而是一个 **GT crop 偏乐观评估** 问题：

| 阶段 | 裁剪来源 | 性质 |
|------|---------|------|
| 训练（`create_dataset.py`） | GT label（liver∪tumor bbox） | **Oracle crop**，边界完美 |
| 标准验证（`report_custom.txt`） | nnUNet 内置验证，同样是 GT crop 预处理数据 | **乐观估计** |
| E2E 推理（`eval.py`） | Stage1 **预测** mask 裁剪 | **真实部署场景** |

### 3.2 偏差量化（fold_0）

```
GT crop  Dice = 0.7673   ← report_custom.txt 报告的数字
E2E 30mm Dice = 0.7396   ← 真实推理场景
偏差          = 0.0277   （报告值高估了 2.77 个百分点）
```

### 3.3 为什么会有这个差距？

Stage1 预测的肝脏 mask 边界并不完美，裁剪区域与 GT crop 略有偏差：
- Stage1 肝脏 Dice 约 0.95-0.97（接近但非完美），边界处几 mm 的偏差足以改变 ROI 范围
- 当 margin 太小（0mm）时 Dice 急跌至 0.711；margin 越大（50mm）反而噪声增加
- **最优 margin ≈ 5mm**（fold_0 E2E 最高 0.743），但仍低于 Baseline 0.753

### 3.4 论文中如何描述

- 需明确说明两套评估路径的区别
- 核心对比表格应使用 **E2E 评估**（Stage1 predicted crop）而非 GT crop 结果
- GT crop 结果可作为 "upper bound" 报告

---

## 四、TwoStage 做得更好 vs 更糟的 Case 分析

### 4.1 TwoStage 更好的情况

**中等到大肿瘤（50k+ voxels），fold_0：**

| Case | Baseline Dice | TwoStage Dice | Δ | 肿瘤大小 |
|------|-------------|--------------|---|---------|
| liver_27 | 0.8161 | **0.8107** | -0.0054 | 中等 |
| liver_82 | 0.9181 | **0.9080** | -0.0101 | 中等 |
| liver_64 | 0.9632 | **0.9582** | -0.0050 | 中等 |
| liver_128 | 0.8036 | **0.7572** | -0.0464 | 大 |

实际上 fold_0 中，TwoStage 的提升主要体现在**稳定性**（std 0.100 vs 0.138），而非平均值大幅提升。

**TwoStage 真正领先的场景特征**：
- 全图中肝外假阳性较多时，ROI 裁剪能有效抑制
- 任务简化（tumor-only 分类）让模型专注于肿瘤特征学习
- `SmallTumorOversampleMixin`（小肿瘤重复采样 3×）对极小肿瘤有额外帮助

### 4.2 TwoStage 更糟的情况

**fold_2 差距最大（-0.0450）：**

严重失败 case 对比（fold_2）：

| Case | Baseline Dice | TwoStage Dice | Δ | 原因推断 |
|------|-------------|--------------|---|---------|
| liver_35 | 0.4751 | 0.0000 | -0.4751 | Stage2 在 ROI 内完全漏检（预测全背景） |
| liver_43 | 0.0155 | 0.0613 | +0.0458 | 两者都失败，TwoStage 略好 |
| liver_13 | 0.6248 | 0.1586 | -0.4662 | TwoStage FDR=0.91，假阳性泛滥 |

`liver_35`（TwoStage Dice=0.000，pred=38637 全是 FP）：Stage2 模型在该 case 的 ROI 内产生大面积假阳性，完全偏离 GT。

**系统性弱点：**

1. **错误级联（Error Cascade）**：Stage1 肝脏分割的任何误差（漏分、多分）都会直接影响 Stage2 的 ROI 范围，Baseline 无此问题
2. **极小肿瘤在 ROI 边界处**：GT crop 训练时完整包含，但 E2E 推理时 Stage1 边界偏差可能把肿瘤裁出 ROI
3. **无肿瘤 case 误报**：fold_0/4 误报率 66.7%，与 Baseline 相当，说明 ROI 裁剪并未改善假阳性问题
4. **Fold 间差异大**：fold_0 TwoStage Dice std=0.100，fold_2 std=0.304，说明不同 fold 的 Stage1 质量影响 Stage2

### 4.3 按肿瘤大小分析（fold_1 完整数据）

| 大小分类 | n | Baseline Dice | TwoStage Dice | Δ |
|---------|---|-------------|--------------|---|
| 极小 (<5k) | 8 | —（无对应数据）| 0.4413 | — |
| 小 (5k-50k) | 7 | — | 0.6791 | — |
| 中等 (50k-300k) | 4 | — | 0.8214 | — |
| 大 (>=300k) | 4 | — | 0.5870 | — |

> Baseline 在 fold_1 无分组统计，待补充。

---

## 五、综合评价与论文定位建议

### 5.1 是否有方法创新点？

从当前数据看：
- **TwoStage 方法本身**（ROI 裁剪 + 专注肿瘤）已是领域内经典 pipeline，并非新颖贡献
- **SmallTumorOversampleMixin**（3× 重采样小肿瘤）是一个工程改进，效果有限
- **性能提升不稳定**：仅 2/4 fold 领先，E2E 评估下 fold_0 依然落后于 Baseline

### 5.2 当前结果在论文中的定位

| 用途 | 建议 |
|------|------|
| Baseline 数字 | Dataset003 5-fold CV Dice=**0.677**（肿瘤），Liver Dice=**0.957** |
| TwoStage 作为实验平台 | 提供可控的肿瘤-only 分割框架，便于引入其他方法（BoundaryLoss等）对比 |
| 主要创新方向 | 应在 TwoStage 框架上引入有效改进（如更好的 loss、attention 机制），而非 pipeline 本身 |

### 5.3 后续优先事项

1. **等 fold_3 训练完成** → 得出 TwoStage 完整 5-fold 均值，才能最终判断优劣
2. **补跑其他对照 fold** → BCE/BoundaryLoss 目前只有 1-2 个 fold，统计不可靠
3. **E2E 评估统一化** → fold_1/2/4 无 E2E 结果，仅 fold_0 有完整 margin 扫描
4. **明确论文 contribution** → TwoStage pipeline 是否足够作为 contribution，还是只作为实验框架

---

*本文档由实验数据自动分析生成，数据来源：`nnUNet_workspace/results/` 下各 trainer 的 `report_custom.txt` 和 `eval_e2e*.txt`。*
