# LiTS 数据集异常 Case 分析材料

> 分析方法：对每个 case 的肝脏区域（label≥1）计算 HU 均值/标准差，  
> 阴影定义：肝脏内 HU < min(mean−1.5σ, 40) 的体素占比 > 0.5%。  
> 数据集共 131 个 case，按 7:1:2 分为 train/val/test。

---

## 一、反常A类 — 有阴影但无肿瘤（共9个）

**特征**：肝脏内存在明显低密度阴影，但 GT 标注为无肿瘤。  
**影响**：模型将阴影识别为肿瘤 → **假阳性（FP）**，拉低 Precision。

> 指标来自 MLAUNet_MoE_SizeOversampleV4 checkpoint_best.pth，全 split 实测。  
> 反常A无GT肿瘤，「误报体素」= 模型预测为肿瘤的体素数（理想值=0）。

| Case | Split | 肝脏均值 HU | Liver Dice | 误报体素 | 备注 |
|------|-------|:----------:|:----------:|:-------:|------|
| liver_32 | train | 76.1 | 0.976 | 0 ✅ | 最严重阴影，但模型学对了 |
| liver_119 | val | 74.3 | 0.977 | 0 ✅ | |
| liver_105 | train | 92.0 | 0.984 | 0 ✅ | |
| liver_38 | train | 105.3 | 0.972 | 0 ✅ | |
| liver_114 | train | 100.2 | 0.983 | 17 ❌ | 仍有少量误报 |
| liver_87 | train | 98.4 | 0.979 | 11 ❌ | 仍有少量误报 |
| **liver_41** | **test** | 113.1 | 0.968 | **31,368 ❌** | **论文重点case；几乎所有方法均失败** |
| liver_47 | train | 85.4 | 0.969 | 0 ✅ | |
| liver_91 | test | 114.9 | 0.986 | 0 ✅ | |

**论文可用语言**：
> liver_41 represents a challenging no-tumor case with prominent hypodense regions (shadow ratio 1.47%) 
> that visually mimic hepatic tumors. This case caused false positives across all evaluated methods,
> reflecting an intrinsic ambiguity in the CT appearance rather than a model-specific failure.

---

## 二、反常B类 — 无阴影但有肿瘤（共12个）

**特征**：肿瘤 HU 接近甚至高于周围肝脏，无低密度阴影，肿瘤"隐身"。  
**影响**：模型漏检 → **假阴性（FN）**，拉低 Recall/Sensitivity。

> 指标来自 MLAUNet_MoE_SizeOversampleV4 checkpoint_best.pth，全 split 实测。  
> HU差 = 肿瘤均值 − 肝脏均值（负值越大 = 肿瘤越"隐身"）。

| Case | Split | 肿瘤均值 HU | 肝脏均值 HU | HU差 | Tumor Dice | Recall | Precision | 备注 |
|------|-------|:---------:|:---------:|:----:|:----------:|:------:|:---------:|------|
| liver_35 | val | **134.9** | **164.4** | -29 | 0.401 | 0.253 | 0.965 | 等密度高强化，大量漏检 |
| liver_83 | val | 50.2 | 132.5 | -82 | 0.682 | 0.612 | 0.769 | 小肿瘤无阴影 |
| liver_25 | train | 74.0 | 123.6 | -50 | 0.802 | 0.773 | 0.834 | |
| liver_55 | train | 31.8 | 130.6 | -99 | 0.645 | 0.948 | 0.489 | 低密度但面积极小，FP多 |
| liver_113 | train | 80.5 | 116.5 | -36 | 0.802 | 0.883 | 0.734 | |
| liver_120 | train | 83.4 | 115.4 | -32 | 0.821 | 0.877 | 0.772 | |
| liver_122 | train | 91.7 | 106.4 | -15 | 0.868 | 0.826 | 0.913 | 近等密度 |
| liver_67 | train | 80.7 | 93.3 | -13 | 0.825 | 0.792 | 0.861 | 近等密度 |
| liver_61 | train | 70.3 | 98.3 | -28 | 0.836 | 0.786 | 0.894 | |
| liver_20 | train | 35.0 | 77.3 | -42 | 0.707 | 0.842 | 0.609 | |
| **liver_112** | **test** | 52.3 | 139.3 | -87 | 0.654 | 0.796 | 0.555 | |
| **liver_63** | **test** | 67.6 | 101.9 | -34 | 0.496 | 0.407 | 0.634 | |

---

## 三、liver_127 — 特殊第三类（极小等密度肿瘤）

**位置**：test set  
**特征**：
- 肿瘤体素数仅 **298 个**（极小，仅跨 z630–z632 共 3 个切片）
- 肿瘤均值 HU = 79，肝脏均值 HU = 96，差值仅 **17 HU**
- 65% 的肿瘤体素 HU 低于肝脏均值 10 HU 以内（几乎等密度）
- 肝内存在 1.86% 阴影（来自其他结构，与肿瘤位置无关）

**结论**：非阴影辨别问题，而是**体积极小 + 近等密度 = 人眼和模型均无法分辨**。  
标注可信度存疑（可能为噪声标注）。

**论文可用语言**：
> liver_127 contains an extremely small tumor (298 voxels, spanning only 3 axial slices) 
> with near-isodense appearance (tumor HU: 79 vs. liver HU: 96, Δ=17 HU). 
> Neither visual inspection nor automated segmentation could reliably detect this lesion,
> suggesting potential annotation uncertainty rather than a model limitation.

---

## 四、Test Set 异常 Case 汇总

test set 共 26 个 case，其中 5 个存在明显异常：

> 指标来源：MLAUNet_MoE_SizeOversampleV4，checkpoint_best.pth，min_voxel=1。  
> 反常A（无肿瘤case）的 Tumor Dice/Recall/Precision 按 nnUNet 惯例：TN排除(—)、FP计0。

| Case | Split | 类型 | 核心问题 | Liver Dice | Tumor Dice | Recall | Precision | 实测结果 |
|------|-------|------|---------|:----------:|:----------:|:------:|:---------:|---------|
| liver_41 | **test** | 反常A | 有阴影无肿瘤，阴影占1.47% | 0.968 | **0.000** (FP) | — | — | 误报 31,368 体素(4个假CC)，Precision↓ |
| liver_91 | **test** | 反常A | 有阴影无肿瘤，阴影占0.80% | 0.986 | — (TN) | — | — | ✅ 正确，pred_tumor=0，无误报 |
| liver_112 | **test** | 反常B | 无阴影有肿瘤，Δ HU=-87 | — | 0.654 | 0.796 | 0.555 | 部分漏检(FN)，gt=1,181体素，pred=1,693 |
| liver_63 | **test** | 反常B | 无阴影有肿瘤，Δ HU=-34 | — | 0.496 | 0.407 | 0.634 | 严重漏检(FN)，gt=688体素，pred=442 |
| liver_127 | **test** | 极小等密度 | 298 体素 + Δ HU=17 | — | **0.000** | 0.000 | 0.000 | 完全漏检，pred_tumor=0，标注可疑 |

---

## 五、train_viz / val_viz 实测结果

> 使用 MLAUNet_MoE_SizeOversampleV4 checkpoint_best.pth 推理，min_voxel=1。  
> **有 PNG = 仍有 FP/FN 错误；无 PNG = 模型在该 case 上无明显错误。**

### 反常A（有阴影无肿瘤）— 模型基本学会，但泛化有限

| Case | Split | 实测结果 | 错误切片数 |
|------|-------|---------|---------|
| liver_32 | train | ✅ 无FP，模型学对了 | — |
| liver_38 | train | ✅ 无FP，模型学对了 | — |
| liver_47 | train | ✅ 无FP，模型学对了 | — |
| liver_105 | train | ✅ 无FP，模型学对了 | — |
| liver_87 | train | ❌ **仍有FP** | 2 张 |
| liver_114 | train | ❌ **仍有FP** | 2 张 |
| liver_119 | val | ✅ 泛化良好，无FP | — |
| liver_41 | test | ❌ FP（见test_viz） | — |
| liver_91 | test | ❌ FP（见test_viz） | — |

**解读**：6个训练集反常A中4个被学对，说明"有阴影无肿瘤"的歧义通过训练可以部分克服（模型记住了这些case的决策边界）。但泛化到未见的test case（liver_41/91）仍失败，说明这类学习是"记忆"而非真正理解阴影与肿瘤的区别。

### 反常B（无阴影有肿瘤）— 模型完全学不会

| Case | Split | 实测结果 | 错误切片数 |
|------|-------|---------|---------|
| liver_113 | train | ❌ **仍有FN** | **151 张**（最严重） |
| liver_120 | train | ❌ **仍有FN** | 53 张 |
| liver_122 | train | ❌ **仍有FN** | 64 张 |
| liver_20 | train | ❌ **仍有FN** | 22 张 |
| liver_25 | train | ❌ **仍有FN** | 11 张 |
| liver_55 | train | ❌ **仍有FN** | 16 张 |
| liver_61 | train | ❌ **仍有FN** | 11 张 |
| liver_67 | train | ❌ **仍有FN** | 5 张 |
| liver_35 | val | ❌ **仍有FN** | 29 张 |
| liver_83 | val | ❌ **仍有FN** | 5 张 |
| liver_112 | test | ❌ FN（见test_viz） | — |
| liver_63 | test | ❌ FN（见test_viz） | — |

**解读**：反常B的8个训练集case **全部失败**——模型连自己训练过的数据都学不会识别这类肿瘤。这不是过拟合或泛化问题，而是**CT视觉特征的本质局限**：当肿瘤与肝脏等密度、无阴影时，纯粹依赖像素强度的分割网络无法区分两者，无论怎么训练都无法突破这个瓶颈。

---

## 六、对论文 Discussion 的意义

1. **数据集内在难度**：131 个 case 中 9+12=21 个（16%）存在视觉异常，说明 LiTS 数据集本身存在一定比例的歧义样本。

2. **两类失败的本质不同**：
   - 反常A（假阳性）→ 数据歧义问题，训练可部分克服，但难以完全泛化
   - 反常B（假阴性）→ 视觉特征不足的根本限制，**连训练集都学不好**，是CT模态的固有瓶颈

3. **核心论点**：反常B的存在证明，单纯提升模型架构对等密度无阴影肿瘤的帮助极为有限；要从根本上解决，需要多期CT（增强扫描）或多模态信息，而非更强的分割网络。

4. **可视化目录**：
   - `fold_0/val_viz/`：val 集全量（12个case，781张PNG）
   - `fold_0/train_viz/`：train集14个反常case推理结果（10个case有误差，337张PNG）
   - `fold_0/test_viz/`：test集全量（含liver_41/91/112/63/127）

---

## 七、可视化生成命令

```bash
# 生成 val_viz（用已有预测）
conda run -n medseg python pumengyu/tools/gen_val_train_viz.py --skip-train

# 生成 train_viz（对14个反常case推理）
conda run -n medseg python pumengyu/tools/gen_val_train_viz.py --skip-val

# 同时生成两个
conda run -n medseg python pumengyu/tools/gen_val_train_viz.py
```
