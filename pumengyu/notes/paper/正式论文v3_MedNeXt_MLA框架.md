# 面向跨域泛化的 MedNeXt 瓶颈潜在注意力肝肿瘤分割

**作者**：（占位）  
**单位**：（占位）

---

## 写作定位

本文不再以“MLAUNet + MoE + SizeOversample 刷内部 Dice”为主线，而以 **外部验证泛化** 为核心问题。

当前最稳的论文故事是：

> 内部测试集 Dice 最优的模型不一定外部泛化最好。MedNeXt 在 LiTS 内部测试中表现最强，但在 IRCADb 外部验证中出现明显 drop。本文在 MedNeXt bottleneck 处引入低秩潜在注意力，用全局上下文补充 MedNeXt 的局部卷积归纳偏置。实验显示，MedNeXt_MLA 虽然内部 Overall 略低于 MedNeXt/MedNeXt_SizeOV4，但外部 Overall 排名第一，internal-external drop 显著减小，并降低无肿瘤误报。

旧稿中 MLAUNet、MoE、SizeOversample 的结果可以作为对照和补充实验，但不再作为主贡献。

---

## 摘要（草稿）

肝肿瘤三维分割模型在内部测试集上取得高 Dice，并不必然意味着其在外部临床数据上可靠。本文围绕肝肿瘤分割中的跨域泛化问题展开研究，系统比较 nnU-Net、SwinUNETR、nnFormer、MedNeXt 以及多种瓶颈注意力与采样策略。实验发现，MedNeXt-L 在 LiTS 内部测试集上取得最高 Overall，但在 3D-IRCADb 外部验证中出现明显性能下降，提示强局部卷积骨干仍可能对源域外观分布过拟合。为此，本文提出 MedNeXt_MLA，在 MedNeXt 最低分辨率 bottleneck 特征上引入低秩 Multi-head Latent Attention，通过全局 liver-tumor context 建模补充局部倒置瓶颈卷积的感受野限制。结果显示，MedNeXt_MLA 在内部测试集 Overall 为 0.8259，低于 MedNeXt_SizeOV4 的 0.8431；但在 IRCADb 外部验证集中达到最高 Overall 0.8079，Tumor Dice 0.6484，internal-external drop 仅 -0.0180，显著小于 MedNeXt 的 -0.0697 和 MedNeXt_SizeOV4 的 -0.0634。同时，MedNeXt_MLA 将外部无肿瘤误报率从 60% 降至 40%。消融实验表明，均匀 SizeOV4 采样仅带来有限外部收益，而 bottleneck latent attention 是改善跨域泛化的主要因素。本文结果说明，在肝肿瘤分割中，面向外部验证的全局上下文建模比单纯追求内部 Dice 更能提升模型临床可靠性。

**关键词**：肝肿瘤分割；MedNeXt；潜在注意力；跨域泛化；外部验证；假阳性

---

## 1. 引言

### 1.1 背景

肝细胞癌与肝转移瘤是临床中常见的恶性肿瘤类型。基于 CT 的肝脏与肿瘤三维分割对术前规划、疗效评估、消融靶区设计和随访体积变化分析具有重要价值。近年来，nnU-Net 通过自配置预处理、网络结构和训练策略，在多种医学图像分割任务中成为强基线。MedNeXt 进一步借鉴 ConvNeXt 的倒置瓶颈、深度可分离卷积、GroupNorm 和残差设计，在三维医学分割中取得强性能。

然而，肝肿瘤分割的临床难点并不只在内部测试集 Dice。真实应用中，模型需要面对不同医院、扫描协议、病灶大小和图像对比度。一个在内部数据上表现优异的模型，可能在外部数据上出现小病灶漏检、无肿瘤误报或整体 tumor Dice drop。

### 1.2 问题观察

本文在 results_v2 的系统实验中观察到：内部 Overall 排名前两位分别为 MedNeXt_SizeOV4 和 MedNeXt，内部 Overall 分别达到 0.8431 和 0.8402。然而在 3D-IRCADb 外部验证中，二者 Overall 分别下降到 0.7797 和 0.7705，drop 分别为 -0.0634 和 -0.0697。相反，MedNeXt_MLA 内部 Overall 仅为 0.8259，但外部 Overall 达到 0.8079，成为所有有效方法中的外部第一。

这一现象说明：**内部 Dice 最优不等于外部泛化最优**。肝肿瘤分割方法需要从“内部指标优化”转向“跨域可靠性优化”。

### 1.3 方法动机

MedNeXt 的优势来自强局部表征能力：深层 encoder-decoder、倒置瓶颈块、GroupNorm 和残差连接使其能够在源域内部学习稳定的肝脏与肿瘤局部纹理。但局部卷积归纳偏置也可能导致模型对源域纹理、边界和局部对比度模式过拟合。

为补充这一不足，本文在 MedNeXt bottleneck 引入 Multi-head Latent Attention。bottleneck 特征具有最低空间分辨率和最高语义抽象程度，适合以可控计算成本建模全局依赖。MLA 通过低秩 KV 压缩减少注意力开销，同时保留完整 token-token 全局交互。

### 1.4 贡献

本文贡献如下：

1. 从内部测试与外部验证的差异出发，系统揭示肝肿瘤分割中“内部最优不等于外部最优”的泛化问题。
2. 提出 MedNeXt_MLA，在 MedNeXt bottleneck 后插入低秩潜在注意力模块，使强局部卷积骨干获得全局 liver-tumor context。
3. 通过 MedNeXt、MedNeXt_SizeOV4、MedNeXt_MLA、MedNeXt_MLA_SizeOV4 的消融证明：SizeOV4 采样不是主要因素，bottleneck latent attention 才是改善外部泛化的关键。
4. 在 LiTS 内部测试和 3D-IRCADb 外部验证上评估多种 CNN、Transformer 和自研 trainer，MedNeXt_MLA 在外部验证中取得最高 Overall 0.8079，并显著减小 internal-external drop。
5. 通过无肿瘤误报率、size-group analysis 和典型 case 可视化，分析模型跨域失败与成功模式。

---

## 2. 相关工作

### 2.1 自配置医学图像分割

介绍 U-Net、nnU-Net、nnU-Net v2。强调 nnU-Net 是强基线，但其默认 PlainConvUNet 仍主要依赖局部卷积。

### 2.2 MedNeXt 与 ConvNeXt-style 医学分割

介绍 MedNeXt 的倒置瓶颈、depthwise convolution、GroupNorm、residual、deep encoder-decoder。注意措辞：本文不能声称已经拆出 MedNeXt 的全部秘诀，只能说 MedNeXt-style 强局部骨干内部表现强，但外部 drop 明显。

### 2.3 Transformer 与瓶颈全局上下文

介绍 UNETR、SwinUNETR、nnFormer、TransBTS 等。突出 3D 全局 attention 的计算代价，以及 bottleneck attention 是折中方案。

### 2.4 Multi-head Latent Attention

介绍 MLA 低秩 KV 压缩思想。本文不是做 LLM KV cache，而是迁移其低秩全局交互思想到 3D segmentation bottleneck。

### 2.5 跨域泛化与外部验证

强调医学图像分割论文不能只看内部测试集，应重视 external validation、domain shift、false positive safety。

---

## 3. 方法

### 3.1 总体结构

MedNeXt_MLA 以 MedNeXt-L 为基础，保持 encoder、decoder、deep supervision、训练配置不变，仅在 bottleneck 后加入 MLABottleneck3D。

```text
Input CT
  -> MedNeXt stem
  -> MedNeXt encoder blocks
  -> MedNeXt bottleneck
  -> MLABottleneck3D
  -> MedNeXt decoder blocks
  -> segmentation logits
```

核心设计原则：

- 不改变 MedNeXt 局部骨干。
- 不改变 loss。
- 不依赖额外标注。
- 只在最低分辨率加入全局上下文模块。

### 3.2 MedNeXt-L 骨干

使用配置：

```text
n_channels = 32
kernel_size = 3
exp_r = [3,4,8,8,8,8,8,4,3]
block_counts = [3,4,8,8,8,8,8,4,3]
do_res = True
do_res_up_down = True
norm_type = group
```

说明：本文使用的是 MedNeXt-L kernel=3 配置。其强性能主要来自更深的网络、倒置瓶颈和残差结构，而不是大核卷积。

### 3.3 MLABottleneck3D

MLABottleneck3D 接收 bottleneck 特征：

```text
x: (B, C, D, H, W), C=512
flatten -> (B, N, C)
MLA blocks
reshape -> (B, C, D, H, W)
```

MLA 计算：

```text
c_kv = x W_DKV
K = c_kv W_UK
V = c_kv W_UV
Q = x W_Q
Attention = softmax(QK^T / sqrt(d)) V
```

默认参数：

```text
num_heads = 8
num_blocks = 2
compression_ratio = 4
mlp_ratio = 4
```

### 3.4 为什么放在 bottleneck

放在 bottleneck 的理由：

1. 空间分辨率最低，计算可控。
2. 语义最抽象，适合建模肝脏-肿瘤全局关系。
3. 避免在高分辨率层引入过多显存开销。
4. 与 MedNeXt 的局部归纳偏置互补。

### 3.5 SizeOV4 采样作为对照

SizeOV4 并不是本文最终主贡献，而是用于验证采样因素是否解释外部收益。其实现是将所有类别 case 在训练 identifier 列表中均匀重复 2 次：

```text
tiny = 2
small = 2
mid = 2
huge = 2
no_tumor = 2
```

由于所有 case 等比例重复，它不改变训练集大小分布。实验显示 SizeOV4 对 MedNeXt 外部收益有限，且与 MLA 组合后反而下降。

---

## 4. 实验设计

### 4.1 数据集

内部数据：Dataset003_Liver / LiTS 风格数据，fold_0 固定划分。  
外部数据：3D-IRCADb，20 cases，其中 5 个无肿瘤 case，15 个有肿瘤 case。

注意：旧稿里写“IRCADb 全部有肿瘤”已经不对。当前报告中 IRCADb 有 5 个无肿瘤 case。

### 4.2 指标

核心指标：

```text
Liver Dice
Tumor Dice
Overall = (Liver Dice + Tumor Dice) / 2
Recall
Precision
No-tumor FP rate
Internal-external drop = External Overall - Internal Overall
```

本文主排序使用 Overall。

### 4.3 对比方法

主表保留：

- nnU-Net Baseline
- SizeOV2 / SizeOV3
- MLAUNet
- MoE_SizeOV5
- MedNeXt
- MedNeXt_SizeOV4
- MedNeXt_MLA
- MedNeXt_MLA_SizeOV4
- SwinUNETR
- nnFormer

不放主表：

- DWSepRes4_MoE_SizeOV4：无效/未完成，外部 tumor 全 0。
- 太多探索 trainer：可放 supplement 或不放。

---

## 5. 实验结果

### 5.1 外部验证主表

| Rank | Method | External Overall | Liver | Tumor | Recall | Precision | FP rate | Internal Overall | Drop |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | MedNeXt_MLA | 0.8079 | 0.9673 | 0.6484 | 0.6665 | 0.7437 | 40% | 0.8259 | -0.0180 |
| 2 | MoE_SizeOV5 | 0.8025 | 0.9679 | 0.6371 | 0.6437 | 0.7464 | 40% | 0.8167 | -0.0142 |
| 3 | MLAUNet | 0.8008 | 0.9675 | 0.6341 | 0.6320 | 0.7580 | 40% | 0.8148 | -0.0140 |
| 4 | SizeOV2 | 0.7992 | 0.9676 | 0.6307 | 0.6352 | 0.7547 | 40% | 0.8187 | -0.0195 |
| 5 | MLA_GK5_V4 | 0.7957 | 0.9656 | 0.6258 | 0.6472 | 0.7291 | 40% | 0.8173 | -0.0216 |
| 13 | MedNeXt_SizeOV4 | 0.7797 | 0.9651 | 0.5943 | 0.6500 | 0.6795 | 60% | 0.8431 | -0.0634 |
| 15 | MedNeXt | 0.7705 | 0.9660 | 0.5750 | 0.6554 | 0.6564 | 60% | 0.8402 | -0.0697 |

写作重点：

- MedNeXt 内部强但外部不强。
- MedNeXt_MLA 外部第一。
- MedNeXt_MLA drop 明显小。

### 5.2 MedNeXt 消融

| Method | Internal Overall | External Overall | Drop | Internal Tumor | External Tumor | External Precision | External FP |
|---|---:|---:|---:|---:|---:|---:|---:|
| MedNeXt | 0.8402 | 0.7705 | -0.0697 | 0.7283 | 0.5750 | 0.6564 | 60% |
| MedNeXt_SizeOV4 | 0.8431 | 0.7797 | -0.0634 | 0.7317 | 0.5943 | 0.6795 | 60% |
| MedNeXt_MLA | 0.8259 | 0.8079 | -0.0180 | 0.6982 | 0.6484 | 0.7437 | 40% |
| MedNeXt_MLA_SizeOV4 | 0.8285 | 0.7870 | -0.0415 | 0.7040 | 0.6091 | 0.7019 | 60% |

差值解读：

```text
+SizeOV4 on MedNeXt:
  external overall +0.0092
  external tumor +0.0193

+MLA on MedNeXt:
  external overall +0.0374
  external tumor +0.0734
  precision +0.0873
  FP 60% -> 40%

+SizeOV4 on MedNeXt_MLA:
  external overall -0.0209
  external tumor -0.0393
  FP 40% -> 60%
```

结论：MLA 是主因，SizeOV4 不是主因。

### 5.3 Size-group analysis

待补表。需要从 report 中整理：

- tiny <5k
- small 5k-50k
- mid 50k-300k
- large >=300k

目前已知 MedNeXt_MLA 在 small group 上改善明显，`ircadb_016` 是关键 case。

### 5.4 No-tumor FP analysis

重点写：

- MedNeXt 和 MedNeXt_SizeOV4 外部 FP 率 60%。
- MedNeXt_MLA 外部 FP 率 40%。
- `ircadb_007` 和 `ircadb_014` 是固定难例，很多方法都会误报。

### 5.5 Case visualization

必选 case：

1. `ircadb_016`：MedNeXt_MLA 明显修复。
2. `ircadb_008`：小病灶普遍困难。
3. `ircadb_018`：极小病灶失败，用作 limitation。
4. `ircadb_014`：无肿瘤误报案例。

`ircadb_016` 数据：

| Method | Tumor Dice |
|---|---:|
| MedNeXt | 0.2546 |
| MedNeXt_SizeOV4 | 0.5515 |
| MedNeXt_MLA | 0.8121 |
| MedNeXt_MLA_SizeOV4 | 0.7932 |

---

## 6. 讨论

### 6.1 内部最优与外部最优不一致

MedNeXt_SizeOV4 内部 Overall 第一，但外部不如 MedNeXt_MLA。说明单纯以内部分数选模型会误导实际部署。

### 6.2 为什么 MLA 有助于外部泛化

推测机制：

- MedNeXt 强局部骨干容易学习源域局部纹理。
- MLA 在 bottleneck 引入全局上下文，有助于根据肝脏整体结构和病灶上下文判断肿瘤。
- 低秩 attention 避免高分辨率全局 attention 的显存开销。

措辞必须谨慎：这是基于消融和 case-level 结果的解释，不声称已经完全证明机制。

### 6.3 为什么 SizeOV4 不够

SizeOV4 等比例重复所有 case，不改变数据分布。它在 MedNeXt 上只有小幅收益，在 MedNeXt_MLA 上反而降低外部表现。因此它不是本文主贡献。

### 6.4 临床安全性

无肿瘤误报很重要。MedNeXt_MLA 降低 FP 率，但仍有 40% 外部无肿瘤 case 误报。后续 FP-Safe 可以作为进一步改进方向。

### 6.5 局限性

- 当前为单 fold 实验。
- IRCADb 外部验证只有 20 cases，规模有限。
- 仍存在极小病灶失败，如 `ircadb_018`。
- FP-Safe 目前尚未完整纳入主结果。
- 不能严格拆解 MedNeXt 每个内部组件的贡献。

---

## 7. 结论

本文围绕肝肿瘤分割的跨域泛化问题，系统比较了内部测试和外部验证结果。实验表明，内部 Dice 最优的 MedNeXt/MedNeXt_SizeOV4 在外部验证中出现明显 drop，而在 MedNeXt bottleneck 加入低秩潜在注意力后，MedNeXt_MLA 取得最高外部 Overall 0.8079，并将 internal-external drop 从 MedNeXt 的 -0.0697 缩小到 -0.0180。消融结果进一步显示，均匀 SizeOV4 采样不是主要收益来源，bottleneck latent attention 才是改善外部泛化的关键因素。该结果提示，在面向临床部署的肝肿瘤分割中，应从单纯追求内部 Dice 转向关注外部验证、无肿瘤误报和跨域稳健性。

---

## 待补内容清单

- [ ] 根据最新结果补完整内部主表。
- [ ] 根据最新结果补完整外部主表。
- [ ] 从 report 中整理 MedNeXt 系列 size-group 表。
- [ ] 生成 `ircadb_016` 对比图。
- [ ] 生成 `ircadb_014` 无肿瘤误报图。
- [ ] 确认所有实验使用 checkpoint_best 还是 checkpoint_final，文中统一说明。
- [ ] 如果 MedNeXt_MLA_FPSafe 跑出结果，考虑加入主线升级为 FP-safe cross-domain framework。

