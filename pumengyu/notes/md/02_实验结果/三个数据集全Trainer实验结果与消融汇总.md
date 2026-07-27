# 三个数据集全 Trainer 实验结果与消融汇总

> 快照日期：2026-07-22  
> 内部测试：Dataset003_Liver 固定 26 例 test  
> 外部测试 1：3D-IRCADb 20 例 source-only  
> 外部测试 2：HCCReferencedCT v2 固定 21 例 source-only held-out test  
> 主排名指标：各数据域内的 Overall Dice，不计算跨数据集平均或 drop。
> Overall 口径：按 [PMY-LT-v1](指标统计口径.md) 在各数据域内计算 `(Liver Dice(all cases) + Tumor Dice(GT-positive cases)) / 2`；两个外部数据集从不求平均。

## 1. 口径与范围

本文档从当前磁盘中的 `test_report_custom.txt` 和 `report_custom.txt` 重新扫描指标，并统一了内部 trainer 名与外部 method 名。

- 公平主比较池现有 30 种方法；IRCADb/HCC 为 30/30，Dataset003 internal 为 29/30（`DWSepRes4_MoE_SizeOV4` 缺 internal 结果）。关键 MHA/MLA × MLP/MoE 2×2 方法均具备三域指标。
- Dataset003 internal 有 29 种方法的可解析报告；`DWSepRes4_MoE_SizeOV4` 缺 internal 报告。
- IRCADb source-only 有 30 种方法的可解析报告。
- HCCReferencedCT v2 source-only 有 30 种方法的可解析报告；30 种方法均有 21/21 预测、`summary.json`、报告和实际 PNG。
- `HCCAdapter701020`、`HCCRefOnly701020` 和 `MSDHCCMix` 改变了训练数据或适配路径，不进入 source-only 公平排名，在文末单列。

## 2. 总体结论

在公平比较池中，`MedNeXt_MLA_MoE_SizeOV4` 在 Dataset003 internal、3D-IRCADb 和 HCCReferencedCT v2 上分别排名第 1、3、2，是当前跨数据域竞争力最稳定的单一配置。不带 SizeOV4 的 `MedNeXt_MLA_MoE` 分别排名第 4、1、4，其中 IRCADb Overall 从原始 `MedNeXt` 的 0.8280（第 18）提高到 0.8511（第 1）。这些结果足以说明 **MedNeXt_MLA_MoE 是有效且具有跨域竞争力的组合配置**，SizeOV4 版本则表现出更好的三域排名稳定性。

这一结论不等于 MLA、MoE 和 SizeOV4 每个组件都在所有数据域上独立有效。`MedNeXt_MHA` 在 internal、IRCADb 和 HCC 分别排名第 6、2、5，说明在 MedNeXt bottleneck 中引入 attention 明显改善了 IRCADb 表现；纯 `MedNeXt_MLA` 则分别排名第 7、23、1，说明 MLA 的收益具有显著的数据域依赖性。因此，现有证据支持“MLA+MoE 组合配置有效”，但不支持“MLA 或 MoE 在三个数据域上均能独立带来统一收益”。

`DeepDWIBResGN` 在三个数据域分别排名第 10、16、12：其 internal 表现仍有竞争力，且三域均优于 `DeepPlainResGN`，因此不应笼统定义为无效；更准确的结论是，仅依靠 depthwise convolution 和 inverted bottleneck 尚不足以获得与 `MedNeXt_MLA_MoE` 相当的跨域稳定性。相比之下，`MedNeXt_MLA_MoE_FPSafe` 的 Overall 在三个数据域全部下降，且 IRCADb 无肿瘤病例误报率由 40% 升至 60%，因此 FP-Safe 是目前证据最明确的负结果，不应进入最终推荐配置。

`MedNeXt_MHA_MoE` 在 internal/IRCADb/HCC 的 Overall 为 0.8508/0.8463/0.6158，排名第 9/4/7。它在三域的重叠指标都略低于 `MedNeXt_MHA`，仅 IRCADb 误报率从 60% 降至 40%。`EfficientMedNeXt_L_Official` 三域 Overall 为 0.8518/0.8315/0.5921，排名第 8/14/14；它在 IRCADb 略高于原始 MedNeXt，但 internal 和 HCC 均更低。

| 数据域 | 可比方法数 | Overall 第 1 | 最强非 MedNeXt 方法 | 关键观察 |
|---|---:|---|---|---|
| Dataset003 internal | 29 | MedNeXt_MLA_MoE_SizeOV4, **0.8591** | MoE, 0.8585 | 前 2 名为 MedNeXt 系列，与 MoE 差距很小 |
| 3D-IRCADb | 30 | MedNeXt_MLA_MoE, **0.8511** | SizeOV3, 0.8458 | MedNeXt 系列占据前 4，MHA+MoE 为第 4 |
| HCCReferencedCT v2 | 30 | MedNeXt_MLA, **0.6532** | DeepDWIBMedConfig, 0.6190 | 前 5 名不变；MHA+MoE 第 7 |

这一全体比较支持以下判断：

> MedNeXt 是本任务中的一线强卷积主干，选择 MedNeXt 并在其 bottleneck 上比较 MHA、MLA 和 MLA+MoE，是由全部实验而非只由 MedNeXt 家族内部排名支持的研究路线。

但不能扩张为“任意 MedNeXt 变体在所有数据域都领先”。原始 MedNeXt 在 internal/HCC 排名第 5/3，在 IRCADb 排名第 18；纯 MedNeXt_MLA 在 HCC 排名第 1，在 IRCADb 仅排名第 23。真正成立的结论是：MedNeXt 主干强，但外部表现取决于瓶颈组合和数据域。

## 3. 三个数据域全量结果

表格单元格为 `Liver / Tumor / Overall（Overall 排名）`。IRCADb FP 为 5 个无肿瘤病例的 case-level 误报率。HCCReferencedCT v2 的 21 个 test 病例全部有肿瘤，因此不报告 FP 率。

> **2026-07-22 新增方法：** `EfficientMedNeXt_L_Official` 和 `MedNeXt_MHA_MoE` 的 internal、IRCADb 和 HCC 三域指标已经齐全，下表已纳入全部结果。两者的 IRCADb/HCC 交付产物完整；internal test 预测 NIfTI 未保留，因此 internal 按完整交付标准记为部分完成。

| Method | Dataset003 internal L/T/O | IRCADb L/T/O | IRCADb FP | HCC v2 L/T/O |
|---|---|---|---:|---|
| MedNeXt_MLA_MoE_SizeOV4 | 0.9529 / 0.7653 / **0.8591 (#1)** | 0.9650 / 0.7309 / 0.8479 (#3) | 60% | 0.8442 / 0.4269 / **0.6356 (#2)** |
| MedNeXt_SizeOV4 | 0.9545 / 0.7635 / **0.8590 (#2)** | 0.9651 / 0.7131 / 0.8391 (#9) | 60% | 0.8146 / 0.3405 / 0.5775 (#21) |
| MoE | 0.9506 / 0.7768 / 0.8585 (#3) | 0.9669 / 0.6998 / 0.8333 (#11) | 40% | 0.8282 / 0.3438 / 0.5860 (#17) |
| MedNeXt_MLA_MoE | 0.9535 / 0.7590 / 0.8562 (#4) | 0.9673 / 0.7349 / **0.8511 (#1)** | 40% | 0.8441 / 0.4080 / 0.6261 (#4) |
| MedNeXt | 0.9521 / 0.7600 / 0.8561 (#5) | 0.9660 / 0.6900 / 0.8280 (#18) | 60% | 0.8383 / 0.4175 / **0.6279 (#3)** |
| MedNeXt_MHA | 0.9533 / 0.7544 / 0.8538 (#6) | 0.9672 / 0.7303 / **0.8487 (#2)** | 60% | 0.8435 / 0.4035 / 0.6235 (#5) |
| MedNeXt_MLA | 0.9513 / 0.7535 / 0.8524 (#7) | 0.9660 / 0.6778 / 0.8219 (#23) | 60% | 0.8418 / 0.4645 / **0.6532 (#1)** |
| EfficientMedNeXt_L_Official | 0.9520 / 0.7515 / 0.8518 (#8) | 0.9656 / 0.6973 / 0.8315 (#14) | 60% | 0.8375 / 0.3468 / 0.5921 (#14) |
| MedNeXt_MHA_MoE | 0.9517 / 0.7499 / 0.8508 (#9) | 0.9664 / 0.7261 / 0.8463 (#4) | 40% | 0.8373 / 0.3943 / 0.6158 (#7) |
| DeepDWIBResGN | 0.9494 / 0.7502 / 0.8498 (#10) | 0.9630 / 0.6961 / 0.8295 (#16) | 40% | 0.8265 / 0.3582 / 0.5924 (#13) |
| MLAUNet_MoE_IB7_SizeOV4 | 0.9528 / 0.7453 / 0.8490 (#11) | 0.9675 / 0.6797 / 0.8236 (#21) | 60% | 0.8405 / 0.3560 / 0.5983 (#11) |
| MoE_SizeOV4 | 0.9514 / 0.7457 / 0.8485 (#12) | 0.9674 / 0.6877 / 0.8275 (#19) | 40% | 0.8224 / 0.3431 / 0.5827 (#18) |
| SizeOV2 | 0.9516 / 0.7455 / 0.8485 (#13) | 0.9676 / 0.7148 / 0.8412 (#8) | 40% | 0.8382 / 0.3653 / 0.6018 (#9) |
| MedNeXt_MLA_MoE_FPSafe | 0.9510 / 0.7453 / 0.8481 (#14) | 0.9637 / 0.7022 / 0.8329 (#13) | 60% | 0.8335 / 0.3807 / 0.6071 (#8) |
| MLA_GK5_V4 | 0.9535 / 0.7403 / 0.8469 (#15) | 0.9656 / 0.7092 / 0.8374 (#10) | 40% | 0.8416 / 0.3399 / 0.5908 (#15) |
| MoE_SizeOV5 | 0.9499 / 0.7429 / 0.8464 (#16) | 0.9679 / 0.7221 / 0.8450 (#6) | 40% | 0.8237 / 0.3696 / 0.5967 (#12) |
| MLAUNet_1500 | 0.9525 / 0.7384 / 0.8454 (#17) | 0.9673 / 0.6912 / 0.8293 (#17) | 40% | 0.8340 / 0.3657 / 0.5999 (#10) |
| DeepDWIBMedConfig | 0.9525 / 0.7368 / 0.8447 (#18) | 0.9633 / 0.6913 / 0.8273 (#20) | 40% | 0.8381 / 0.3999 / 0.6190 (#6) |
| MoE_SizeOV2 | 0.9516 / 0.7378 / 0.8447 (#19) | 0.9668 / 0.6996 / 0.8332 (#12) | 80% | 0.8327 / 0.3433 / 0.5880 (#16) |
| MLAUNet | 0.9503 / 0.7384 / 0.8443 (#20) | 0.9675 / 0.7186 / 0.8431 (#7) | 40% | 0.8139 / 0.3508 / 0.5823 (#19) |
| SizeOV3 | 0.9513 / 0.7363 / 0.8438 (#21) | 0.9675 / 0.7241 / 0.8458 (#5) | 60% | 0.7009 / 0.0843 / 0.3926 (#27) |
| NoMirror | 0.9581 / 0.7267 / 0.8424 (#22) | 0.7766 / 0.4125 / 0.5946 (#28) | 80% | 0.6133 / 0.2388 / 0.4261 (#26) |
| SizeOV3_NoMirror | 0.9591 / 0.7227 / 0.8409 (#23) | 0.7668 / 0.3865 / 0.5766 (#29) | 60% | 0.5761 / 0.2067 / 0.3914 (#28) |
| Baseline | 0.9340 / 0.7395 / 0.8368 (#24) | 0.9673 / 0.6937 / 0.8305 (#15) | 60% | 0.5022 / 0.0000 / 0.2511 (#30) |
| DeepPlainResGN_SizeOV4 | 0.9407 / 0.7244 / 0.8326 (#25) | 0.9536 / 0.6852 / 0.8194 (#24) | 60% | 0.7936 / 0.2517 / 0.5227 (#24) |
| DeepResGN_MLA | 0.9458 / 0.7044 / 0.8251 (#26) | 0.9577 / 0.6631 / 0.8104 (#25) | 60% | 0.7968 / 0.2995 / 0.5482 (#22) |
| DeepPlainResGN | 0.9455 / 0.7040 / 0.8248 (#27) | 0.9584 / 0.6360 / 0.7972 (#27) | 60% | 0.8227 / 0.3347 / 0.5787 (#20) |
| nnFormer | 0.9448 / 0.6800 / 0.8124 (#28) | 0.9600 / 0.6859 / 0.8230 (#22) | 40% | 0.8207 / 0.2755 / 0.5481 (#23) |
| SwinUNETR | 0.9392 / 0.6575 / 0.7983 (#29) | 0.9543 / 0.6619 / 0.8081 (#26) | 80% | 0.7927 / 0.1949 / 0.4938 (#25) |
| DWSepRes4_MoE_SizeOV4 | — | 0.7290 / 0.0000 / 0.3645 (#30) | 0% | 0.5887 / 0.0000 / 0.2944 (#29) |

## 4. 各数据域排名解读

### 4.1 Dataset003 internal

- `MedNeXt_MLA_MoE_SizeOV4` 以 0.8591 排名第 1，但与 `MedNeXt_SizeOV4` 只差 0.0001，与非 MedNeXt 的 `MoE` 也只差 0.0006。
- 原始 MedNeXt 在没有 attention 和 MoE 时排名第 5，说明主干本身仍处于第一梯队，但不能写成“包揽前列”。
- `SwinUNETR` 和 `nnFormer` 排名第 29/28，本任务中完整 Transformer 基线没有超过强卷积主干。
- `EfficientMedNeXt_L_Official` 以 0.8518 排名第 8，`MedNeXt_MHA_MoE` 以 0.8508 排名第 9；两者均处于第一梯队边缘，但未超过原始 MedNeXt。

### 4.2 3D-IRCADb

- `MedNeXt_MLA_MoE` 以 Overall 0.8511、Tumor Dice 0.7349 排名第 1，也将无肿瘤误报率从 MedNeXt 的 60% 降至 40%。
- `MedNeXt_MHA` 以 0.8487 排名第 2，`MedNeXt_MHA_MoE` 以 0.8463 排名第 4；第 1–5 仅相差 0.0053，外部竞争很接近。
- 原始 MedNeXt 排名第 18，纯 MedNeXt_MLA 排名第 23，说明 attention/FFN 组合对该域的影响很大。
- `EfficientMedNeXt_L_Official` 以 0.8315 排名第 14，高于原始 MedNeXt 0.0035，但低于最优 MLA+MoE 0.0196。
- 原始 MedNeXt 并非 IRCADb 一线结果；是 MLA+MoE 组合将该家族提升到第 1。

MSD、IRCADb 与 HCC 中多个或全部 Trainer 共同失败的病例见 [跨 Trainer 失败病例分析](三个数据集失败案例分析.md)。该分析不再将共性难例归因于单一 FP-Safe 模块。

### 4.3 HCCReferencedCT v2

- Overall 前 5 名全部是 MedNeXt 系列：纯 `MedNeXt_MLA`、`MedNeXt_MLA_MoE_SizeOV4`、`MedNeXt`、`MedNeXt_MLA_MoE` 和 `MedNeXt_MHA`。
- 纯 `MedNeXt_MLA` 以 Tumor Dice 0.4645、Overall 0.6532 排名第 1，相比 MHA 分别提高 0.0610/0.0297；但这一顺序与 IRCADb 相反。
- `MedNeXt_MHA_MoE` 的 Tumor Dice/Overall 为 0.3943/0.6158（第 7），均略低于 MHA+MLP 的 0.4035/0.6235；MoE 在 MHA 路径下没有改善 HCC。
- `EfficientMedNeXt_L_Official` 的 Tumor Dice/Overall 为 0.3468/0.5921（第 14），比原始 MedNeXt 低 0.0707/0.0358，其主要问题是 Recall 从 0.3582 降至 0.2697。
- HCC 第 1 的 Recall 仍仅 0.3937，说明即使 MLA 提高了 HCC 表现，系统性跨域漏检仍然明显。
- HCC 中大多数方法的 Tumor Dice 明显低于 IRCADb，数据域偏移强于模型排名差异。

## 5. 主要消融对比

### 5.1 MedNeXt 主干与 attention/FFN

| 对比 | Internal Overall | IRCADb Overall | HCC Overall | 结论 |
|---|---:|---:|---:|---|
| MedNeXt | 0.8561 | 0.8280 | 0.6279 | 强 internal/HCC 主干，IRCADb 第 18 |
| MedNeXt_MHA | 0.8538 | 0.8487 | 0.6235 | Internal 基本持平，IRCADb 第 2 |
| MedNeXt_MLA | 0.8524 | 0.8219 | 0.6532 | IRCADb 低于 MHA，但 HCC 排名第 1 |
| MedNeXt_MLA_MoE | 0.8562 | 0.8511 | 0.6261 | internal 与原始 MedNeXt 持平，IRCADb 第 1 |
| MedNeXt_MHA_MoE | 0.8508 | 0.8463 | 0.6158 | 三域均略低于 MHA+MLP |

MHA 与纯 MLA 使用相同 MedNeXt-L 主干、2 个 bottleneck block、8 heads 和标准 MLP，仅 attention 路径不同：

| MHA - MLA | Internal Tumor | Internal Overall | IRCADb Tumor | IRCADb Overall | IRCADb FP | HCC Tumor | HCC Overall |
|---|---:|---:|---:|---:|---:|---:|---:|
| 差值 | +0.0009 | +0.0014 | **+0.0525** | **+0.0268** | 0 pp | **-0.0610** | **-0.0297** |

因此，MHA 在 internal 基本持平、在 IRCADb 更好，但 MLA 在 HCC 更好。现有结果不支持任一 attention 路径在三个数据域上统一占优。

固定 MLA attention，将标准 MLP 换成 MoE：

| MoE - MLP | Internal Tumor | Internal Overall | IRCADb Tumor | IRCADb Overall | IRCADb FP | HCC Tumor | HCC Overall |
|---|---:|---:|---:|---:|---:|---:|---:|
| 差值 | +0.0055 | +0.0038 | **+0.0571** | **+0.0292** | -20 pp | **-0.0565** | **-0.0271** |

该对比说明，在 MLA attention 固定时，MoE 在 internal 小幅上升，在 IRCADb 同时改善 Tumor Dice、Overall 和假阳性，但在 HCC 下降。MoE 的收益不能从 IRCADb 泛化为所有外部数据域的统一收益。

固定 MHA attention，将标准 MLP 换成 MoE：

| MoE - MLP | Internal Tumor | Internal Overall | IRCADb Tumor | IRCADb Overall | IRCADb FP | HCC Tumor | HCC Overall |
|---|---:|---:|---:|---:|---:|---:|---:|
| 差值 | **-0.0045** | **-0.0030** | **-0.0042** | **-0.0024** | -20 pp | **-0.0092** | **-0.0077** |

该对比表明 MHA+MoE 在 internal、IRCADb 和 HCC 的重叠指标均小幅下降，仅 IRCADb 误报率改善。MoE 在 MHA 路径下没有带来分割精度收益；这与 MLA 路径上 internal/IRCADb 为正、HCC 为负的结果共同证明 attention–FFN 存在数据域依赖的交互效应。

### 5.2 SizeOV 采样

| 固定架构 | 变化 | Internal Overall | IRCADb Overall | HCC Overall |
|---|---|---:|---:|---:|
| Baseline → SizeOV2 | +SizeOV2 | +0.0117 | +0.0107 | +0.3507 |
| Baseline → SizeOV3 | +SizeOV3 | +0.0070 | +0.0153 | +0.1415 |
| MoE → MoE_SizeOV2 | +SizeOV2 | -0.0138 | -0.0001 | +0.0020 |
| MoE → MoE_SizeOV4 | +SizeOV4 | -0.0100 | -0.0058 | -0.0033 |
| MoE → MoE_SizeOV5 | +SizeOV5 | -0.0121 | +0.0117 | +0.0107 |
| MedNeXt → MedNeXt_SizeOV4 | +SizeOV4 | +0.0029 | +0.0111 | -0.0504 |
| MedNeXt_MLA_MoE → +SizeOV4 | +SizeOV4 | +0.0029 | -0.0032 | +0.0095 |

SizeOV 的作用不稳定：`SizeOV2` 在基线上三个域都有改善，`MoE_SizeOV5` 更适合 IRCADb，而 `MedNeXt_MLA_MoE_SizeOV4` 更适合 HCC。不能把任一 SizeOV 版本写成普遍有效的采样策略。

### 5.3 FP-Safe

| 对比 | Internal Overall | Internal FP | IRCADb Overall | IRCADb FP | HCC Overall |
|---|---:|---:|---:|---:|---:|
| MedNeXt_MLA_MoE | 0.8562 | 66.67% | 0.8511 | 40% | 0.6261 |
| +FP-Safe | 0.8481 | 33.33% | 0.8329 | 60% | 0.6071 |

FP-Safe 改善了 internal 阴性病例误报，但三个数据域的 Overall 都下降，属于明确的负结果。

### 5.4 卷积主干与 Transformer 基线

| Method | Internal Overall | IRCADb Overall | HCC Overall | 观察 |
|---|---:|---:|---:|---|
| DeepPlainResGN | 0.8248 | 0.7972 | 0.5787 | 单纯加深 plain residual 不足 |
| DeepDWIBResGN | 0.8498 | 0.8295 | 0.5924 | DW+IB 在三个域均优于 DeepPlainResGN |
| DeepDWIBMedConfig | 0.8447 | 0.8273 | 0.6190 | HCC 第 6，支持 MedNeXt 式 DW+IB 主干 |
| EfficientMedNeXt_L_Official | 0.8518 | 0.8315 | 0.5921 | IRCADb 略高于 MedNeXt，internal/HCC 更低 |
| MedNeXt | 0.8561 | 0.8280 | 0.6279 | Internal/HCC 一线，IRCADb 需瓶颈改进 |
| SwinUNETR | 0.7983 | 0.8081 | 0.4938 | 三个域均未超过 MedNeXt |
| nnFormer | 0.8124 | 0.8230 | 0.5481 | IRCADb 接近原始 MedNeXt，但 internal/HCC 较弱 |

`DeepDWIBResGN` 和 `DeepDWIBMedConfig` 相对 plain residual 结构的改善，说明 MedNeXt 系列的优势不只是参数量，而与 depthwise convolution、inverted bottleneck 和深层残差主干有关。

EfficientMedNeXt-L 比原始 MedNeXt 的 internal Overall 低 0.0043，IRCADb Overall 高 0.0035，HCC Overall 低 0.0358。它在 IRCADb 与 MedNeXt 处于同一量级，但 HCC Tumor Dice/Recall 明显更低，不支持“EfficientMedNeXt-L 三域统一优于 MedNeXt”。该模型应定位为官方架构/效率基线，不能作为 MLA 或 MoE 模块贡献的直接证据。

#### EfficientMedNeXt-L 与原始 MedNeXt 的直接对比

| 数据域 | EfficientMedNeXt-L Liver | Tumor | Overall | 相对 MedNeXt Liver | Tumor | Overall | Recall | Precision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Dataset003 internal | 0.9520 | 0.7515 | 0.8518 | -0.0001 | -0.0085 | -0.0043 | — | — |
| 3D-IRCADb | 0.9656 | 0.6973 | 0.8315 | -0.0004 | +0.0073 | +0.0035 | — | — |
| HCCReferencedCT v2 | 0.8375 | 0.3468 | 0.5921 | -0.0008 | -0.0707 | -0.0358 | -0.0885 | +0.0599 |

HCC 上 EfficientMedNeXt-L 的 Precision 为 0.6894，高于 MedNeXt 的 0.6295，但 Recall 由 0.3582 降至 0.2697。也就是说，它预测肿瘤时更保守，却漏掉了更多真实肿瘤体素；这解释了 Tumor Dice 下降 0.0707，并说明其 HCC 退化主要来自敏感性不足，而不是肝脏分割变化。

### 5.5 Mirror 对照

`NoMirror` 和 `SizeOV3_NoMirror` 在 internal 仍有 0.8424/0.8409，但 IRCADb 分别降至 0.5946/0.5766，HCC 降至 0.4261/0.3914。这是目前最清晰的跨域崩溃对照之一，表明 mirror 相关训练/推理口径对外部可靠性很重要。

Baseline vs NoMirror 的大小分组、FPV/FNV、连通域和具体病例证据见 [Mirror 消融分析](Mirror消融分析.md)。

## 6. 外部数据域训练/适配结果（不进入 source-only 排名）

| Method | IRCADb Overall | IRCADb Tumor | 为何不进入主排名 |
|---|---:|---:|---|
| MedNeXt_MLA_MoE_HCCAdapter701020 | 0.4408 | 0.1058 | 加入 HCC Adapter/改变模型路由 |
| MedNeXt_MLA_MoE_HCCRefOnly701020 | 0.6483 | 0.3481 | 使用 HCCReferencedCT 训练 |
| MedNeXt_MLA_MoE_MSDHCCMix | 0.7785 | 0.6009 | 使用 MSD/HCC 混合训练 |

这三项回答的是数据域训练或适配问题，不是纯架构消融，不应与 Dataset003 source-only 方法直接混合排名。

## 7. 结果覆盖与产物状态

| 数据域 | 预期公平方法数 | 有指标报告 | 预测/summary/report/PNG 形态 | checkpoint 追溯 |
|---|---:|---:|---|---|
| Dataset003 internal | 30 | 29 | 新增两方法均有 summary/report/PNG（MHA+MoE 2475，EfficientMedNeXt-L 2503），但预测 NIfTI 未保留；历史缺口仍存在 | 新增两方法均使用 Dataset003/fold 0/`checkpoint_best.pth`，旧 report 未全部显式记录实际 checkpoint |
| 3D-IRCADb | 30 | 30 | 29/30 预测、summary、report、PNG 齐全；FP-Safe PNG=0；MHA+MoE/EfficientMedNeXt-L 均为 20/20 | 新增两方法的训练日志明确记录 `checkpoint_best.pth`，但两份 IRCADb 报告头本身未写 dataset/trainer/fold/checkpoint |
| HCCReferencedCT v2 | 30 | 30 | 30 种方法均有 21/21 预测、summary、report 和实际 PNG；MHA+MoE 706 PNG，EfficientMedNeXt-L 674 PNG；当前无运行进程 | 新增两方法均核对为 Dataset003 source-only/fold 0/`checkpoint_best.pth`，报告已记录 dataset、split、checkpoint 和 model_source |

关键 attention 对照与 EfficientMedNeXt-L checkpoint 来源核对如下：

| Method | Dataset / trainer / fold | checkpoint | checkpoint 内部记录 | 文件修改时间 | 来源判定 |
|---|---|---|---|---|---|
| MedNeXt_MHA | Dataset003_Liver / `nnUNetTrainer_MedNeXt_MHA` / fold 0 | `checkpoint_best.pth` | epoch 1000, best EMA 0.91572237 | 2026-07-18 14:42:22 +08:00 | 通过；训练日志在 epoch 999 记录新 best，best 与 final 为同一组网络权重；修改时间来自 Transformer→MHA 名称迁移，迁移 manifest 校验 network SHA256 不变 |
| MedNeXt_MLA | Dataset003_Liver / `nnUNetTrainer_MedNeXt_MLA` / fold 0 | `checkpoint_best.pth` | epoch 946, best EMA 0.9072642 | 2026-07-18 05:03:06 +08:00 | 通过；训练日志在 epoch 945 记录新 best，而 final 为 epoch 1000，best 未被 final 覆盖 |
| MedNeXt_MHA_MoE | Dataset003_Liver / `nnUNetTrainer_MedNeXt_MHA_MoE` / fold 0 | `checkpoint_best.pth` | current_epoch 957, best EMA 0.901722 | 2026-07-22 09:09:19 +08:00 | 通过；checkpoint 内 trainer/dataset 一致，用 best 完成 internal、IRCADb 和 HCC；HCC 报告已记录 source-only 来源 |
| EfficientMedNeXt_L_Official | Dataset003_Liver / `nnUNetTrainer_EfficientMedNeXt_L_Official` / fold 0 | `checkpoint_best.pth` | current_epoch 971, best EMA 0.9035133 | 2026-07-20 12:36:51 +08:00 | 通过；checkpoint 内 trainer/dataset 一致，用 best 完成 internal、IRCADb 和 HCC；HCC 报告已记录 source-only 来源 |

因此，本文档可用于当前指标比较和消融判断，但不应把所有历史目录统一宣称为按现行 `AGENTS.md` 标准完整交付。

## 8. 当前可以采用的最终判断

1. **主干选择成立**：MedNeXt 系列在三个数据域都产生 Overall 第 1，是合理的主干研究空间。
2. **MHA/MLA 关系具有域依赖性**：MHA 在 internal 基本持平、在 IRCADb 更好，纯 MLA 则在 HCC 排名第 1。
3. **纯 MLA 不能被概括为无效**：它在 IRCADb 低于 MHA 和原始 MedNeXt，但在 HCC 取得最优 Overall 0.6532。
4. **MLA+MoE 组合在 internal 小幅上升、在 IRCADb 收益明确，但 HCC 下降**：不能概括为统一外部收益。
5. **不存在一个三域通吃的配置**：Internal 最优为 MedNeXt_MLA_MoE_SizeOV4，IRCADb 最优为 MedNeXt_MLA_MoE，HCC 最优为纯 MedNeXt_MLA。
6. **HCC 的主要问题仍是系统性漏检**：即使排名第 1，Tumor Dice 和 Recall 仍然只有 0.4645/0.3937。
7. **MHA/MLA × MLP/MoE 三域指标矩阵已闭环**：MoE 在 MLA 下提高 internal/IRCADb 但降低 HCC，在 MHA 下三域重叠指标均下降；attention–FFN 作用具有数据域依赖性。产物状态需与指标闭环分开：新增 MHA+MoE 的 internal test 预测 NIfTI 未保留。
8. **EfficientMedNeXt-L 是强基线而非贡献对照**：它仅在 IRCADb 略高于 MedNeXt，internal/HCC 更低，不支持其三域统一优于 MedNeXt，也不能直接证明 MLA 或 MoE 有效。
