# results_v2 实验结果汇总分析报告

> 分析日期：2026-06-28
> 数据来源：[`results_v2`](../../../../../nnUNet_workspace/results_v2)

> 最新汇总请优先看：[`实验汇总分析_v3.md`](实验汇总分析_v3.md)；MedNeXt 专项消融见：[`MedNeXt探究消融实验.md`](MedNeXt探究消融实验.md)。本文件保留较完整的历史分析过程。

## 一、目录结构

```
results_v2/
├── Dataset003_Liver/          ← 内部测试集（23个Trainer，n=26）
│   ├── nnUNetTrainer_Baseline__nnUNetPlans__3d_fullres/
│   ├── nnUNetTrainer_DeepPlainResGN__nnUNetPlans__3d_fullres/
│   ├── nnUNetTrainer_DeepPlainResGN_SizeOV4__nnUNetPlans__3d_fullres/
│   ├── nnUNetTrainer_DeepResGN_MLA__nnUNetPlans__3d_fullres/
│   ├── nnUNetTrainer_DeepDWIBResGN__nnUNetPlans__3d_fullres/
│   ├── nnUNetTrainer_MedNeXt__nnUNetPlans__3d_fullres/
│   ├── nnUNetTrainer_MedNeXt_SizeOV4__nnUNetPlans__3d_fullres/
│   ├── nnUNetTrainer_MedNeXt_MLA_SizeOV4__nnUNetPlans__3d_fullres/
│   ├── nnUNetTrainer_MLAUNet__nnUNetPlans__3d_fullres/
│   ├── nnUNetTrainer_MLAUNet_MoE__nnUNetPlans__3d_fullres/
│   ├── nnUNetTrainer_MLAUNet_MoE_SizeOversampleV2__nnUNetPlans__3d_fullres/
│   ├── nnUNetTrainer_MLAUNet_MoE_SizeOversampleV4__nnUNetPlans__3d_fullres/
│   ├── nnUNetTrainer_MLAUNet_MoE_SizeOversampleV5__nnUNetPlans__3d_fullres/
│   ├── nnUNetTrainer_MLAUNet_MoE_IB7_SizeOversampleV4__nnUNetPlans__3d_fullres/
│   ├── nnUNetTrainer_MLAUNet_1500__nnUNetPlans__3d_fullres/
│   ├── nnUNetTrainer_MLA_GK5_V4__nnUNetPlans__3d_fullres/
│   ├── nnUNetTrainer_nnFormer__nnUNetPlans__3d_fullres/
│   ├── nnUNetTrainer_SwinUNETR__nnUNetPlans__3d_fullres/
│   ├── nnUNetTrainer_NoMirror__nnUNetPlans__3d_fullres/
│   ├── nnUNetTrainer_SizeOversampleV2__nnUNetPlans__3d_fullres/
│   ├── nnUNetTrainer_SizeOversampleV3__nnUNetPlans__3d_fullres/
│   ├── nnUNetTrainer_SizeOversampleV3_NoMirror__nnUNetPlans__3d_fullres/
│   ├── Tr_Stage1_TumorOnly__nnUNetPlans__3d_fullres/
│   ├── Tr_Stage2_FPSup__nnUNetPlans__3d_fullres/
│   └── DWSepRes4_MoE_SizeOV4__nnUNetPlans__3d_fullres/
│
└── ExternalVal_IRCADb/        ← 外部验证集（19个模型，n=20）
    ├── Baseline/
    ├── DeepPlainResGN/
    ├── DeepPlainResGN_SizeOV4/
    ├── DeepResGN_MLA/
    ├── MedNeXt/
    ├── MedNeXt_SizeOV4/
    ├── MedNeXt_MLA_SizeOV4/
    ├── MLAUNet/
    ├── MLA_GK5_V4/
    ├── MLAUNet_MoE_IB7_SizeOV4/
    ├── MoE/                          (MLAUNet_MoE)
    ├── MoE_SizeOV2/                  (MLAUNet_MoE_SizeOversampleV2)
    ├── MoE_SizeOV4/                  (MLAUNet_MoE_SizeOversampleV4)
    ├── MoE_SizeOV5/                  (MLAUNet_MoE_SizeOversampleV5)
    ├── nnFormer/
    ├── NoMirror/
    ├── SizeOV2/                      (SizeOversampleV2)
    ├── SizeOV3/                      (SizeOversampleV3)
    └── SwinUNETR/
```

---

## 二、内部测试集结果（Dataset003_Liver, n=26）

测试集包含 **26 cases**，其中 **23例有肿瘤**、**3例无肿瘤**（liver_41, liver_89, liver_91）。

### 2.1 综合排行榜（按 Overall Score 降序）

| 排名 | Trainer | Liver Dice | Tumor Dice | Overall | Recall | Precision | 误报率 |
|:---:|:--------|:----------:|:----------:|:-------:|:------:|:---------:|:-----:|
| 1 | **MedNeXt_SizeOV4** | 0.9545 | **0.7317** | **0.8431** | 0.7361 | 0.8187 | 33% |
| 2 | **MedNeXt** | 0.9521 | **0.7283** | **0.8402** | 0.7334 | 0.8128 | 33% |
| 3 | **MLAUNet_MoE_SizeOV4** | 0.9514 | **0.7146** | **0.8330** | 0.7140 | 0.7776 | 33% |
| 4 | MedNeXt_MLA_SizeOV4 | 0.9529 | 0.7040 | 0.8285 | 0.7459 | 0.7775 | 67% |
| 5 | MLAUNet_MoE_IB7_SizeOV4 | 0.9528 | 0.6857 | 0.8192 | 0.7148 | 0.7509 | 67% |
| 6 | SizeOversampleV2 | 0.9516 | 0.6858 | 0.8187 | 0.7105 | 0.7490 | 67% |
| 7 | **MLA_GK5_V4** | **0.9535** | 0.6811 | 0.8173 | 0.7084 | 0.7622 | 67% |
| 8 | MLAUNet_MoE_SizeOV5 | 0.9499 | 0.6835 | 0.8167 | 0.7141 | 0.7816 | 67% |
| 9 | MLAUNet_MoE | 0.9506 | 0.6826 | 0.8166 | 0.7043 | 0.7554 | 67% |
| 10 | MLAUNet_MoE_SizeOV2 | 0.9516 | 0.6788 | 0.8152 | 0.7113 | 0.7723 | 67% |
| 11 | **MLAUNet** | 0.9503 | 0.6793 | 0.8148 | 0.7104 | 0.7427 | 67% |
| 12 | SizeOversampleV3 | 0.9513 | 0.6774 | 0.8143 | 0.7228 | 0.7641 | 67% |
| 13 | **NoMirror** | **0.9581** | 0.6685 | 0.8133 | 0.6769 | 0.7685 | 67% |
| 14 | SizeOversampleV3_NoMirror | **0.9591** | 0.6649 | 0.8120 | 0.6705 | 0.7570 | 67% |
| 15 | Tr_Stage1_TumorOnly | 0.9502 | 0.6592 | 0.8047 | 0.7066 | 0.7250 | 100% |
| 16 | MLAUNet_1500 | 0.9525 | 0.6532 | 0.8028 | 0.7123 | 0.7121 | 100% |
| 17 | **DeepResGN_MLA** | 0.9458 | 0.6480 | 0.7969 | 0.7118 | 0.7225 | 67% |
| 18 | DeepPlainResGN | 0.9455 | 0.6477 | 0.7966 | 0.7219 | 0.6989 | 67% |
| 19 | **Baseline** (nnUNet原始) | 0.9340 | 0.6542 | 0.7941 | **0.7853** | 0.6451 | 100% |
| 20 | DeepPlainResGN_SizeOV4 | 0.9407 | 0.6409 | 0.7908 | 0.7235 | 0.6891 | 100% |
| 21 | **SwinUNETR** | 0.9392 | 0.6301 | 0.7846 | 0.6960 | 0.6752 | 33% |
| 22 | Tr_Stage2_FPSup | 0.1044 | 0.6782 | 0.3913 | 0.7136 | 0.7348 | 67% |
| 23 | **nnFormer** | 0.9448 | 0.6015 | 0.7732 | 0.7049 | 0.6435 | 100% |

> 注：Tr_Stage2_FPSup Liver Dice=0.1044 表明肝脏分割训练有问题（仅关注肿瘤），其Overall无意义。
> SwinUNETR 误报率仅33%但 Liver/Tumor Dice 均偏低。

### 2.2 关键发现

1. **MedNeXt系列统治测试集**：`MedNeXt_SizeOV4` (0.8431) 和 `MedNeXt` (0.8402) 分别领先，且**误报率仅33%**（无肿瘤case仅1/3误报）
2. **MLA系列紧随其后**：`MLAUNet_MoE_SizeOV4` (0.8330) 表现亮眼，误报也仅33%
3. **`DeepResGN_MLA` 处于中游**：Overall 0.7969，Tumor Dice 0.6480，与 Baseline (0.7941) 接近，优于 nnFormer (0.7732) 和 SwinUNETR (0.7846)
4. **Baseline Recall最高（0.7853）但误报率100%**——说明原始 nnUNet 倾向于"宁可错杀"的保守策略

---

## 三、外部验证集结果（IRCADb, n=20）

外部验证集包含 **20 cases**，其中 **15例有肿瘤**、**5例无肿瘤**，来自 IRCADb 公开数据集。

### 3.1 综合排行榜（按 Overall Score 降序）

| 排名 | Trainer | Liver Dice | Tumor Dice | Overall | Recall | Precision | 误报率 |
|:---:|:--------|:----------:|:----------:|:-------:|:------:|:---------:|:-----:|
| 1 | **MoE_SizeOV5** (MLAUNet_MoE_SizeOV5) | **0.9679** | **0.6371** | **0.8025** | 0.6437 | 0.7464 | **40%** |
| 2 | **MLAUNet** | 0.9675 | **0.6341** | **0.8008** | 0.6320 | **0.7580** | **40%** |
| 3 | **SizeOV2** (SizeOversampleV2) | 0.9676 | 0.6307 | 0.7992 | 0.6352 | 0.7547 | **40%** |
| 4 | MLA_GK5_V4 | 0.9656 | 0.6258 | 0.7957 | 0.6472 | 0.7291 | **40%** |
| 5 | MoE (MLAUNet_MoE) | 0.9669 | 0.6175 | 0.7922 | 0.6413 | 0.7140 | **40%** |
| 6 | MedNeXt_MLA_SizeOV4 | 0.9650 | 0.6091 | 0.7870 | 0.6580 | 0.7019 | 60% |
| 7 | MoE_SizeOV4 (MLAUNet_MoE_SizeOV4) | 0.9674 | 0.6068 | 0.7871 | 0.6371 | 0.7081 | **40%** |
| 8 | SizeOV3 (SizeOversampleV3) | 0.9675 | 0.6034 | 0.7855 | 0.6565 | 0.6970 | 60% |
| 9 | **nnFormer** | **0.9600** | 0.6052 | 0.7826 | 0.6199 | 0.7461 | **40%** |
| 10 | MedNeXt_SizeOV4 | 0.9651 | 0.5943 | 0.7797 | 0.6500 | 0.6795 | 60% |
| 11 | Baseline | 0.9673 | 0.5781 | 0.7727 | 0.6253 | 0.6814 | 60% |
| 12 | MedNeXt | 0.9660 | 0.5750 | 0.7705 | 0.6554 | 0.6564 | 60% |
| 13 | MLAUNet_MoE_IB7_SizeOV4 | 0.9675 | 0.5664 | 0.7670 | 0.6251 | 0.6796 | 60% |
| 14 | DeepPlainResGN_SizeOV4 | 0.9536 | 0.5710 | 0.7623 | 0.6359 | 0.6933 | 60% |
| 15 | MoE_SizeOV2 (MLAUNet_MoE_SizeOV2) | 0.9668 | 0.5523 | 0.7596 | 0.6308 | 0.6470 | **80%** |
| 16 | **DeepResGN_MLA** | 0.9577 | 0.5526 | 0.7551 | 0.5933 | 0.7088 | 60% |
| 17 | DeepPlainResGN | 0.9584 | 0.5300 | 0.7442 | 0.5775 | 0.6646 | 60% |
| 18 | **SwinUNETR** | 0.9543 | 0.5226 | 0.7385 | 0.6034 | 0.6052 | **80%** |
| 19 | **NoMirror** | 0.7766 | 0.3257 | 0.5512 | 0.3541 | 0.4616 | **80%** |

### 3.2 关键发现

1. **冠军易主**：内测冠军 MedNeXt_SizeOV4 在外验降至第10（0.7797），而外验冠军是 **MoE_SizeOV5** (0.8025)
2. **MLAUNet系列泛化优异**：多个MLA变体进入外验Top 5，且误报率仅40%
3. **nnFormer外验表现稳定**：内测垫底（0.7732）但在外验排第9（0.7826），是唯一外验反超内测的模型
4. **NoMirror在外验崩溃**：Liver Dice仅0.7766（其他模型约0.96），说明数据增强对IRCADb泛化至关重要
5. **Liver Dice普遍高达0.95+**（除NoMirror），说明肝脏分割任务已接近饱和

---

## 四、内外验证对比分析

### 4.1 泛化差异排名（按差值从小到大）

| Trainer | 内测 Overall | 外验 Overall | **差值** | 泛化评估 |
|:--------|:-----------:|:-----------:|:--------:|:--------:|
| **nnFormer** | 0.7732 | 0.7826 | **+0.0094** | ✅ **唯一正向泛化** |
| **MLAUNet** | 0.8148 | 0.8008 | **-0.0140** | ✅ **最佳泛化** |
| MoE_SizeOV5 | 0.8167 | 0.8025 | **-0.0142** | ✅ **最佳泛化** |
| MLA_GK5_V4 | 0.8173 | 0.7957 | **-0.0216** | ✅ 泛化好 |
| MLAUNet_MoE_SizeOV2 | 0.8152 | 0.7596 | **-0.0556** | ⚠️ 中等 |
| MLAUNet_MoE_IB7_SizeOV4 | 0.8192 | 0.7670 | **-0.0522** | ⚠️ 中等 |
| SwinUNETR | 0.7846 | 0.7385 | **-0.0461** | ⚠️ 中等 |
| **DeepResGN_MLA** | 0.7969 | 0.7551 | **-0.0418** | ⚠️ 中等 |
| Baseline | 0.7941 | 0.7727 | **-0.0214** | ✅ 泛化好 |
| MedNeXt_SizeOV4 🥇内测 | **0.8431** | 0.7797 | **-0.0634** | ⚠️ 过拟合倾向 |
| MedNeXt 🥈内测 | **0.8402** | 0.7705 | **-0.0697** | ❌ **过拟合严重** |
| NoMirror | 0.8133 | 0.5512 | **-0.2621** | ❌ **完全崩溃** |

### 4.2 泛化分析总结

1. **内测与外验排名差异大**：
   - 内测第1的 MedNeXt_SizeOV4 在外验掉到第10
   - 内测中游的 `MoE_SizeOV5`、`MLAUNet` 在外验夺冠
   - 说明精度的排名不能直接外推到泛化

2. **`nnFormer` 是唯一正向泛化的模型**（+0.0094）
   - 说明 Transformer 结构在小数据集上有更好的迁移能力
   - 但绝对精度仍然偏低

3. **MedNeXt 系列过拟合现象明显**
   - MedNeXt 内测0.8402 → 外验0.7705（降幅 **0.0697**）
   - MedNeXt_SizeOV4 内测0.8431 → 外验0.7797（降幅 **0.0634**）
   - 说明 MedNeXt 在大肝肿瘤数据上可能过度拟合了特定分布

4. **MLAUNet 系列内外验证表现均衡**
   - 泛化差值仅 -0.014 ~ -0.024
   - 在内测和外验均稳定在前列

---

## 五、DeepResGN_MLA 详细分析

### 5.1 测试集结果（数据集003）

[`test_report_custom.txt`](../../../../../nnUNet_workspace/results_v2/Dataset003_Liver/nnUNetTrainer_DeepResGN_MLA__nnUNetPlans__3d_fullres/fold_0/test_report_custom.txt)

#### 总体指标
| 指标 | 值 |
|:-----|:----:|
| n_cases | 26 |
| Liver Dice | **0.9458** |
| Tumor Dice | **0.6480** |
| Jaccard | 0.5319 |
| Recall | 0.7118 |
| Precision | 0.7225 |
| FNR | 0.2882 |
| FDR | 0.2775 |
| Overall | **0.7969** |
| 误报率 | 66.67% (2/3) |
| 有肿瘤 | n=23 |
| 无肿瘤 TN(排除) | n=1 |
| 无肿瘤 FP(计0) | n=2 |

#### 按肿瘤大小分层的 Tumor Dice
| 大小分类 | n | Dice mean | Dice std | Recall | Precision |
|:--------|:-:|:---------:|:--------:|:------:|:---------:|
| 极小(<5k) | 6 | 0.5606 | 0.2711 | 0.5471 | 0.7742 |
| 小(5k-50k) | 8 | 0.7428 | 0.1348 | 0.7397 | 0.7674 |
| 中等(50k-300k) | 1 | 0.8003 | 0.0000 | 0.7273 | 0.8895 |
| 大(>=300k) | 8 | 0.7618 | 0.1705 | 0.8055 | 0.7986 |

**极小肿瘤最弱**（Dice 0.5606），**中等及大肿瘤表现良好**（Dice 0.76-0.80）。

#### Per-Case 分级
| 等级 | n | 范围 | 说明 |
|:---|:-:|:----|:-----|
| ❌ 严重失败 | 1 | tumor_dice < 0.3 | liver_127 (Dice=0.0586, 极小肿瘤, recall仅3%) |
| ⚠️ 需要改进 | 9 | 0.3 <= dice < 0.7 | 包括多种大小的肿瘤 |
| ✅ 没问题 | 13 | tumor_dice >= 0.7 | 大case表现稳定 |
| 无肿瘤误报 | 2 | dice=0 | liver_41 (26,202体素), liver_89 (1体素) |

#### FPV/FNV 体积误差
| 器官 | FPV总量(mm³) | FPV均值/case | FNV总量(mm³) | FNV均值/case |
|:----|:-----------:|:-----------:|:-----------:|:-----------:|
| Tumor | 508,932 | 19,574.3 | 684,411 | 26,323.5 |
| Liver | 3,816,867 | 146,802.6 | 1,564,633 | 60,178.2 |

**最大FPV case**：liver_101 (141,772mm³), liver_128 (107,022mm³)

#### 连通域分析
- 假连通域：min=1, max=24,288, mean=5,241
- 真阳性CC最小体素数：**1**（共213个TP CC）
- → **后处理阈值上限 < 1 体素**，无法通过后处理滤除假阳性而不漏掉真实小肿瘤

### 5.2 外部验证结果（IRCADb）

[`report_custom.txt`](../../../../../nnUNet_workspace/results_v2/ExternalVal_IRCADb/DeepResGN_MLA/report_custom.txt)

#### 总体指标
| 指标 | 值 | vs 测试集 |
|:-----|:----:|:---------:|
| n_cases | 20 | - |
| Liver Dice | 0.9577 | ✅ 比测试集高0.0119 |
| Tumor Dice | **0.5526** | ❌ 下降0.0954 |
| Jaccard | 0.4561 | ❌ |
| Recall | 0.5933 | ❌ 降幅大 |
| Precision | 0.7088 | ≈ 接近 |
| Overall | **0.7551** | ❌ 下降0.0418 |
| 误报率 | 60% (3/5) | ≈ 接近 |
| 有肿瘤 | n=15 | - |
| 无肿瘤 TN(排除) | n=2 | - |
| 无肿瘤 FP(计0) | n=3 | - |

#### 按肿瘤大小分层的 Tumor Dice
| 大小分类 | n | Dice mean | Dice std | Recall | Precision |
|:--------|:-:|:---------:|:--------:|:------:|:---------:|
| 极小(<5k) | 2 | **0.3547** | 0.3547 | 0.2753 | 0.4983 |
| 小(5k-50k) | 8 | **0.6506** | 0.2630 | 0.6043 | 0.8646 |
| 中等(50k-300k) | 3 | **0.7712** | 0.0793 | 0.6526 | **0.9637** |
| 大(>=300k) | 2 | **0.8591** | 0.0801 | 0.7781 | **0.9773** |

- **中等和大肿瘤精度极高**（0.9637, 0.9773）
- **极小肿瘤近乎失败**（Dice 0.3547）

#### Per-Case 分级
| 等级 | n | 范围 |
|:---|:-:|:----|
| ❌ 严重失败 | 2 | ircadb_018 (Dice=0.0, 极小肿瘤, recall=0%) |
| | | ircadb_008 (Dice=0.0528) |
| ⚠️ 需要改进 | 3 | dice: 0.44~0.70 |
| ✅ 没问题 | 10 | dice >= 0.70 |
| 无肿瘤误报 | 3 | ircadb_014 (20,779体素), ircadb_007 (544), ircadb_005 (16) |

#### FPV/FNV 体积误差
| 器官 | FPV总量(mm³) | FPV均值/case | FNV总量(mm³) | FNV均值/case |
|:----|:-----------:|:-----------:|:-----------:|:-----------:|
| Tumor | 136,547 | 6,827.4 | 373,644 | 18,682.2 |
| Liver | 1,155,539 | 57,777.0 | 1,385,735 | 69,286.8 |

**最大FPV case**：ircadb_018 (86,195mm³) — 该case对应极小肿瘤(4,539体素)但预测出62,592体素，典型的假阳性大爆发

---

## 六、Tr_Stage系列分析

### Tr_Stage1_TumorOnly (Overall: 0.8047)
- 仅用肿瘤标注训练的Stage 1
- Liver Dice 0.9502（正常），但误报率100%
- 整体精度中等，与Baseline接近

### Tr_Stage2_FPSup (Overall: 0.3913)
- Liver Dice仅**0.1044**（肝脏分割完全失败）
- 说明第二阶段伪标签训练严重损害了肝脏分割能力
- Tumor Dice 0.6782（仅看肿瘤指标尚可）
- **不建议使用该模型**

---

## 七、总结与建议

### 7.1 DeepResGN_MLA 评估

| 维度 | 评估 | 说明 |
|:----|:----|:------|
| 内测精度 | ⭐⭐⭐ | Overall 0.7969，中游，与Baseline持平 |
| 外验精度 | ⭐⭐ | Overall 0.7551，下降明显 |
| 泛化能力 | ⭐⭐⭐ | 差值-0.0418，中等水平 |
| 极小肿瘤 | ⭐ | Dice 0.35-0.56，核心短板 |
| 大肿瘤 | ⭐⭐⭐⭐ | Dice 0.76-0.86，表现可靠 |
| 误报控制 | ⭐⭐⭐ | 误报率60-67%，中规中矩 |

### 7.2 提升方向建议

1. **极小肿瘤检测**（核心瓶颈）
   - 考虑 Focal Loss 或 Dice + Focal 组合损失
   - 加大极小肿瘤的采样权重
   - 引入 Hard Negative Mining

2. **后处理限制**
   - 连通域分析显示：真CC最小1体素，假CC最大24,288体素
   - **无法简单通过体素阈值后处理**，必须从模型层面改进

3. **架构改进方向**
   - MLA机制本身OK（同类MLA模型泛化更好）
   - 可参考 `MLAUNet_MoE_SizeOV4` 的做法，引入 MoE 和 Size Oversample

### 7.3 全局最佳模型推荐

| 场景 | 推荐模型 | 理由 |
|:----|:--------|:------|
| **内测精度优先** | `MedNeXt_SizeOV4` | 内测 Overall 0.8431, Tumor 0.7317 |
| **外验证泛化优先** | `MoE_SizeOV5` 或 `MLAUNet` | 外验 Overall 0.80+, 差值仅-0.014 |
| **综合最佳** | `MLAUNet` | 内测0.8148 + 外验0.8008，均衡且泛化好 |
| **参考 Baseline** | nnUNet Baseline | Recall最高0.7853但精度低、误报100% |
| **若需无肿瘤抑制** | SwinUNETR | 误报率最低33%，但精度偏低 |

### 7.4 误报情况说明

所有模型都存在**liver_41的严重误报**（pred_tumor 18k~33k体素），这是一个"干净"的无肿瘤case但几乎所有模型都预测出大量假阳性。这可能是因为训练数据中该CT的影像特征与肿瘤相似，需要：
- 检查该case的原始图像特征
- 考虑在训练中增加对该case的负样本权重
- 或者确认GT标注是否存在遗漏

---

## 附录：文件夹命名对照

| 外验证文件夹名 | 完整Trainer名 |
|:------------|:------------|
| Baseline | nnUNetTrainer_Baseline__nnUNetPlans__3d_fullres |
| DeepPlainResGN | nnUNetTrainer_DeepPlainResGN__nnUNetPlans__3d_fullres |
| DeepPlainResGN_SizeOV4 | nnUNetTrainer_DeepPlainResGN_SizeOV4__nnUNetPlans__3d_fullres |
| DeepResGN_MLA | nnUNetTrainer_DeepResGN_MLA__nnUNetPlans__3d_fullres |
| MedNeXt | nnUNetTrainer_MedNeXt__nnUNetPlans__3d_fullres |
| MedNeXt_SizeOV4 | nnUNetTrainer_MedNeXt_SizeOV4__nnUNetPlans__3d_fullres |
| MedNeXt_MLA_SizeOV4 | nnUNetTrainer_MedNeXt_MLA_SizeOV4__nnUNetPlans__3d_fullres |
| MLAUNet | nnUNetTrainer_MLAUNet__nnUNetPlans__3d_fullres |
| MLA_GK5_V4 | nnUNetTrainer_MLA_GK5_V4__nnUNetPlans__3d_fullres |
| MLAUNet_MoE_IB7_SizeOV4 | nnUNetTrainer_MLAUNet_MoE_IB7_SizeOversampleV4__nnUNetPlans__3d_fullres |
| MoE | nnUNetTrainer_MLAUNet_MoE__nnUNetPlans__3d_fullres |
| MoE_SizeOV2 | nnUNetTrainer_MLAUNet_MoE_SizeOversampleV2__nnUNetPlans__3d_fullres |
| MoE_SizeOV4 | nnUNetTrainer_MLAUNet_MoE_SizeOversampleV4__nnUNetPlans__3d_fullres |
| MoE_SizeOV5 | nnUNetTrainer_MLAUNet_MoE_SizeOversampleV5__nnUNetPlans__3d_fullres |
| nnFormer | nnUNetTrainer_nnFormer__nnUNetPlans__3d_fullres |
| NoMirror | nnUNetTrainer_NoMirror__nnUNetPlans__3d_fullres |
| SizeOV2 | nnUNetTrainer_SizeOversampleV2__nnUNetPlans__3d_fullres |
| SizeOV3 | nnUNetTrainer_SizeOversampleV3__nnUNetPlans__3d_fullres |
| SwinUNETR | nnUNetTrainer_SwinUNETR__nnUNetPlans__3d_fullres |
