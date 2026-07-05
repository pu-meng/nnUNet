# 实验设计与数据集划分说明

## 一、数据集划分（固定 7:1:2，seed=42）

数据来源：LiTS 内部数据集，共 **131 cases**。

| 子集 | 数量 | 用途 | 是否影响模型 |
|------|------|------|-------------|
| train | 92 | 梯度更新、权重学习 | 是 |
| val   | 13 | 训练过程监控、checkpoint_best 选择 | 是（间接） |
| test  | 26 | **最终性能评估** | **否** |

划分文件：`preprocessed/Dataset003_Liver/split_info_712.json`

### Val 13 cases（监控用，不用于对比）
```
liver_4  liver_35  liver_54  liver_65  liver_70  liver_72
liver_80  liver_83  liver_85  liver_93  liver_95  liver_119  liver_125
```

### Test 26 cases（评估用，完全隔绝）
```
liver_6   liver_11  liver_17  liver_36  liver_41  liver_48  liver_52
liver_58  liver_63  liver_68  liver_75  liver_84  liver_89  liver_90
liver_91  liver_96  liver_97  liver_100 liver_101 liver_111 liver_112
liver_117 liver_127 liver_128 liver_129 liver_130
```

---

## 二、为什么不用 val 指标做实验对比

`checkpoint_best.pth` 是根据 val loss 选出的，val set 间接参与了模型选择。
用 val 指标横向比较各实验，相当于在"选手打过的题"上评分，存在数据泄露风险。

**唯一可信的对比数字来源：`test_report_custom.txt`（26-case test set）。**

---

## 三、Pipeline 设计原则

### 训练结束后自动触发的流程

```
perform_actual_validation()
  │
  ├─ [nnUNet 内部] 在 val 13 cases 上推理 → validation/summary.json
  │                                          用于 checkpoint 选择记录
  │
  └─ [AutoInternalTestMixin] 在 test 26 cases 上推理
       ├─ test_prediction/   （推理完自动删除 nii.gz，节省磁盘）
       ├─ test_report_custom.txt   ← 实验对比的唯一数字来源
       └─ test_viz/                ← TP/FP/FN 可视化（纯 CT 对照 + 叠加图）
```

### 刻意不生成的内容

| 产物 | 废弃原因 |
|------|---------|
| `report_custom.txt` | val 13 cases 参与了 ckpt 选择，对比有泄露嫌疑 |
| `val_viz/` | val 和 test 完全不重叠，val 图没有分析价值 |
| `vis_png_custom/` | 旧风格冗余，被 test_viz 取代 |

---

## 四、可视化设计（test_viz）

每张图两栏：

| 左栏 | 右栏 |
|------|------|
| 原始 CT（无叠加），标题显示 `GT=xxx体素`（TP+FN 算得） | CT + TP/FP/FN 叠加 |

颜色：绿=TP，红=FP，蓝=FN。GT 黄色已移除（TP+FN 已能还原 GT 范围）。

切片选择标准：**FP 或 FN 体素数 ≥ 20**，只展示有误差的切片，TP-only 切片跳过。

---

## 五、各实验状态

| 实验 | test_report | 说明 |
|------|------------|------|
| nnUNetTrainer_Baseline | ✓ | 基线 |
| nnUNetTrainer_MLAUNet | ✗ | 需重跑 --val 补生成 |
| nnUNetTrainer_NoMirror | ✓ | |
| nnUNetTrainer_SizeOversampleV2 | ✓ | |
| nnUNetTrainer_SizeOversampleV3 | ✓ | |
| nnUNetTrainer_SizeOversampleV3_NoMirror | ✓ | |
| Tr_Stage1_TumorOnly | ✓ | |
| Tr_Stage2_FPSup | ✓ | |
| Tr_Stage2_FPSup_v2 | ✗ | 需重跑 --val 补生成 |

MLAUNet 补跑命令：
```bash
RESULTS_FOLDER=/home/PuMengYu/nnUNet_workspace/results_v2 \
  nnUNetv2_train 3 3d_fullres 0 -tr nnUNetTrainer_MLAUNet --val
```

---

## 六、全实验对比结果（2026-06-15，test 26 cases）

| 排名 | 实验 | Liver Dice | Tumor Dice | **Overall** | 极小肿瘤(<5k) |
|------|------|-----------|-----------|-------------|-------------|
| 🥇 1 | MedNeXt | 0.9521 | **0.7283** | **0.8402** | **0.5835** |
| 2 | MLAUNet_MoE_V4 | 0.9514 | 0.7146 | 0.8330 | 0.5726 |
| 3 | SizeOversampleV2 | 0.9516 | 0.6858 | 0.8187 | 0.5582 |
| 4 | MLAUNet_MoE_V5 | 0.9499 | 0.6835 | 0.8167 | 0.5601 |
| 5 | MLAUNet_MoE | 0.9506 | 0.6826 | 0.8166 | 0.5637 |
| 6 | MLAUNet_MoE_V2 | 0.9516 | 0.6788 | 0.8152 | 0.5409 |
| 7 | MLAUNet | 0.9503 | 0.6793 | 0.8148 | 0.5256 |
| 8 | SizeOversampleV3 | 0.9513 | 0.6774 | 0.8143 | 0.5137 |
| 9 | NoMirror | 0.9581 | 0.6685 | 0.8133 | 0.5151 |
| 10 | SizeOversampleV3_NoMirror | 0.9591 | 0.6649 | 0.8120 | 0.5170 |
| 11 | MLAUNet_1500 | 0.9525 | 0.6532 | 0.8028 | 0.5491 |
| 12 | Stage1_TumorOnly | 0.9502 | 0.6592 | 0.8047 | 0.5667 |
| 13 | Baseline | 0.9340 | 0.6542 | 0.7941 | 0.5386 |
| ❌ | Stage2_FPSup | 0.1044 | 0.6782 | 0.3913 | 0.5523 |

Overall = (Liver Dice + Tumor Dice) / 2，与 nnUNet foreground_mean 口径一致。
Stage2_FPSup Liver Dice 崩塌（0.10），训练存在问题，结果无效。

### 关键观察

- MedNeXt Overall 0.8402，比第二名 MLAUNet_MoE_V4（0.8330）高 **+0.007**
- MedNeXt Tumor Dice 0.7283，比 Baseline（0.6542）高 **+0.074**，是提升最大的方向
- Liver Dice 各实验差距很小（0.93~0.96），瓶颈始终在 Tumor
- 极小肿瘤(<5k) 全实验都在 0.51~0.58，改进空间有限，不是靠架构或过采样能根本解决的

---

## 七、为什么 MedNeXt 最强

### 核心原因：大卷积核带来更大感受野

标准 nnUNet 用 3×3×3 卷积，每个体素只能"看到"周围 3 个体素的信息。
MedNeXt 用**深度可分离大卷积核**（最大到 7×7×7），感受野直接扩大 8 倍以上。

对肝脏肿瘤分割的意义：
- 肿瘤边界模糊，需要参考更大范围的上下文才能判断是否是肿瘤
- 大核让模型在每个体素处就能"看到"更远的周边组织，判断更准确

### ConvNeXt 设计带来的工程优势

MedNeXt 把自然图像领域验证过的 ConvNeXt 架构适配到 3D 医学图像：

| 设计 | 作用 |
|------|------|
| 倒置瓶颈（先升维再大核卷积） | 在高维特征空间用大核，表达能力更强 |
| LayerNorm 替代 BatchNorm | 小 batch size 下更稳定（3D 医学图像显存限制 batch 很小） |
| GELU 激活函数 | 梯度更平滑，训练更稳定 |
| 深度可分离卷积 | 大核下参数量可控，不会过拟合 |

### 对比 MLAUNet 系列

MLAUNet 加了 Attention 机制，理论上也能捕获全局关系，但：
- Attention 是在特征图层面做全局加权，计算开销大，实际能用的分辨率受限
- MedNeXt 的大核卷积在**局部大范围**内效果更直接，且计算效率更高
- MLAUNet_MoE_V4 加了混合专家（MoE）后提升到 0.8330，接近 MedNeXt 但仍有差距

### 对比 SizeOversample 系列

SizeOversample 只改了数据采样策略（多采小肿瘤），骨干网络还是标准 nnUNet。
极小肿瘤 Dice 只比 Baseline 高 0.02 左右，骨干网络的特征提取能力才是关键瓶颈。
