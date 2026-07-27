# MedNeXt 系列消融实验结果汇总

> 审计日期：2026-07-22  
> 内部数据：Dataset003_Liver 固定 26 例 test  
> 外部数据：3D-IRCADb 20 例；HCCReferencedCT v2 固定 21 例 held-out test  
> 默认 checkpoint：`checkpoint_best.pth`
> Overall 口径：按 [PMY-LT-v1](指标统计口径.md) 在各数据域内计算 `(Liver Dice(all cases) + Tumor Dice(GT-positive cases)) / 2`；IRCADb Overall 与 HCC Overall 不互相平均。

## 1. MHA/MLA × MLP/MoE 三域 2×2 指标矩阵已闭环

MHA/MLA × MLP/MoE 的 2×2 控制变量矩阵原本缺少 `MHA + MoE` 这一格。`MedNeXt_MHA_MoE` 已于 2026-07-22 补齐 Dataset003 internal、IRCADb 和 HCCReferencedCT v2 三域指标，现在可以完成全部严格对照。这里的“指标闭环”不等于三域产物全部完整：新增方法的 internal test 预测 NIfTI 未保留，internal 按仓库交付标准仍为部分完成。

| Attention / FFN | 标准 MLP | MoE-FFN |
|---|---|---|
| MHA | `MedNeXt_MHA`（三域指标齐全） | `MedNeXt_MHA_MoE`（三域指标齐全；internal 产物部分完成） |
| MLA | `MedNeXt_MLA`（已完成） | `MedNeXt_MLA_MoE`（已完成） |

现在可以在三个数据域上完成以下严格对照：

1. **MLP 固定，比较 attention**：`MedNeXt_MHA` 对 `MedNeXt_MLA`，判断 MHA 与 MLA 在标准 MLP 下的差异。
2. **MoE 固定，比较 attention**：`MedNeXt_MHA_MoE` 对 `MedNeXt_MLA_MoE`，判断 MHA 与 MLA 在 MoE 下的差异。
3. **MHA 固定，比较 FFN**：`MedNeXt_MHA` 对 `MedNeXt_MHA_MoE`，判断 MoE 在 MHA 下是否有效。
4. **MLA 固定，比较 FFN**：`MedNeXt_MLA` 对 `MedNeXt_MLA_MoE`，判断 MoE 在 MLA 下是否有效。

三域结果显示明确交互效应：MoE 在 MHA 下使 internal/IRCADb/HCC Overall 分别下降 0.0030/0.0024/0.0077；在 MLA 下则分别变化 +0.0038/+0.0292/-0.0271。因此 MoE 不是可以脱离 attention 路径和数据域独立宣称有效的组件。

## 2. 结论先行

1. `MedNeXt_MHA` 是在 MedNeXt 瓶颈后加入标准 MHA + 标准 MLP 的 Transformer block，不是将整个 MedNeXt 替换成 Transformer。
2. `MedNeXt_MHA` 与纯 `MedNeXt_MLA` 保持相同主干、block 数、head 数、MLP 和训练配置，唯一主要区别是标准全维 K/V 投影与低秩 K/V 路径，因此这是当前最严格的 attention 控制变量实验。
3. 在 IRCADb 上，`MedNeXt_MHA` 相比 MedNeXt 将 Tumor Dice 从 0.6900 提高到 0.7303，Overall 从 0.8280 提高到 0.8487，说明标准 MHA 瓶颈在该外部集上有效。
4. 纯 `MedNeXt_MLA` 在内部测试与 MHA 接近，在 IRCADb 上低于 MHA，但在 HCC 上以 Tumor Dice 0.4645、Overall 0.6532 排名第 1，分别高于 MHA 0.0610/0.0297。因此低秩 K/V 路径的收益具有明显数据域依赖性，不能概括为统一提升或统一无效。
5. `MedNeXt_MLA_MoE` 在 IRCADb 上取得最高 Tumor Dice 0.7349 和 Overall 0.8511，但在 HCC 上比纯 MLA 分别低 0.0565/0.0271。这说明 MLA + MoE 的收益主要体现在 IRCADb，并没有统一复制到 HCC。
6. 在全部具有同口径可解析报告的方法中，MedNeXt 系列分别取得 Dataset003 internal、IRCADb 和 HCCReferencedCT v2 的最高 Overall，因此选择 MedNeXt 作为主干并在其上寻找最优瓶颈组合有明确的实验依据。
7. `MedNeXt_MHA_MoE` 在 internal/IRCADb/HCC 的 Overall 为 0.8508/0.8463/0.6158，三域均低于对应的 MHA+MLP；但 IRCADb 误报率从 60% 降至 40%。这是“误报改善但重叠指标未改善”的混合结果，不应概括为 MHA+MoE 整体有效。
8. `EfficientMedNeXt_L_Official` 三域 Overall 为 0.8518/0.8315/0.5921。它比原始 MedNeXt 在 internal/HCC 分别低 0.0043/0.0358，仅 IRCADb 高 0.0035；因此它是效率/官方架构对照，不是 MLA 贡献的直接证据，也不支持三域统一优于 MedNeXt。

EfficientMedNeXt-L 与原始 MedNeXt 的三域差值为：internal Liver/Tumor/Overall `-0.0001/-0.0085/-0.0043`，IRCADb `-0.0004/+0.0073/+0.0035`，HCC `-0.0008/-0.0707/-0.0358`。HCC 上它的 Precision 提高 0.0599，但 Recall 降低 0.0885，说明主要代价是更严重的肿瘤漏检，而不是肝脏轮廓退化。

## 3. MedNeXt 在全部实验中的位置

为验证“以 MedNeXt 为主干继续改进”是否合理，这里不只比较 MedNeXt 家族内部，而是将其放回全部具有同口径可解析报告、使用同一 Dataset003 源域训练口径的方法中排名。IRCADb 排名排除 HCCAdapter、HCCRefOnly 和 MSDHCCMix 等改变训练数据或适配路径的方法。

| 数据域 | 可比方法数 | Overall 第1 | 最强非 MedNeXt 方法 | MedNeXt 家族位置 |
|---|---:|---|---|---|
| Dataset003 internal | 29 | MedNeXt_MLA_MoE_SizeOV4, **0.8591** | MoE, 0.8585 | 家族最优第1；原始 MedNeXt 第5 |
| 3D-IRCADb source-only | 30 | MedNeXt_MLA_MoE, **0.8511** | SizeOV3, 0.8458 | 家族最优第1；MHA 第2；MHA+MoE 第4；原始 MedNeXt 第18 |
| HCCReferencedCT v2 source-only | 30 | MedNeXt_MLA, **0.6532** | DeepDWIBMedConfig, 0.6190 | 第1–5名不变；MHA+MoE 第7 |

### 3.1 Dataset003 internal

| Rank | Method | Tumor Dice | Overall |
|---:|---|---:|---:|
| 1 | MedNeXt_MLA_MoE_SizeOV4 | 0.7653 | **0.8591** |
| 2 | MedNeXt_SizeOV4 | 0.7635 | 0.8590 |
| 3 | MoE | 0.7768 | 0.8585 |
| 4 | MedNeXt_MLA_MoE | 0.7590 | 0.8562 |
| 5 | MedNeXt | 0.7600 | 0.8561 |

Dataset003 internal 上，MedNeXt 家族取得第 1、2、4、5 名，原始 MedNeXt 排名第 5。这说明 MedNeXt 主干本身处于第一梯队，但非 MedNeXt 的 MoE 也达到第 3。

### 3.2 3D-IRCADb source-only

| Rank | Method | Tumor Dice | Overall |
|---:|---|---:|---:|
| 1 | MedNeXt_MLA_MoE | **0.7349** | **0.8511** |
| 2 | MedNeXt_MHA | 0.7303 | 0.8487 |
| 3 | MedNeXt_MLA_MoE_SizeOV4 | 0.7309 | 0.8479 |
| 4 | MedNeXt_MHA_MoE | 0.7261 | 0.8463 |
| 5 | SizeOV3 | 0.7241 | 0.8458 |

IRCADb 上，MedNeXt 家族占据第 1–4，说明合适瓶颈组合可以获得一线外部表现。但原始 MedNeXt 仅排名第 18，纯 MedNeXt_MLA 排名第 23，因此不能写成“所有 MedNeXt 变体都领先”。

### 3.3 HCCReferencedCT v2 source-only

| Rank | Method | Tumor Dice | Overall |
|---:|---|---:|---:|
| 1 | MedNeXt_MLA | **0.4645** | **0.6532** |
| 2 | MedNeXt_MLA_MoE_SizeOV4 | 0.4269 | 0.6356 |
| 3 | MedNeXt | 0.4175 | 0.6279 |
| 4 | MedNeXt_MLA_MoE | 0.4080 | 0.6261 |
| 5 | MedNeXt_MHA | 0.4035 | 0.6235 |

HCCReferencedCT v2 上，MedNeXt 系列包揽前 5 名，纯 `MedNeXt_MLA` 超过原始 MedNeXt、MHA 和 MLA+MoE 取得第 1。这进一步说明 MedNeXt 主干在更强域偏移下仍保持一线竞争力，但最优 bottleneck 组合会随数据域改变。

### 3.4 对主干选择的最终判断

可以支持的判断是：

> MedNeXt 在 Dataset003 internal 上处于第一梯队，在 HCCReferencedCT v2 上包揽前 5 名，并且 MedNeXt 系列改进方法分别取得三个数据域的最高 Overall。因此，选择 MedNeXt 作为强卷积主干，再在其 bottleneck 上比较 MHA、MLA 和 MLA+MoE，是由全体实验结果支持的研究路线。

不应扩张为：

> 任意 MedNeXt 变体在所有外部数据集上都优于其他方法。

## 4. 所有已有主要结果

表中数字均为 Dice，`Overall = (all-case Liver Dice + positive-only Tumor Dice) / 2`。HCCReferencedCT v2 没有无肿瘤阴性病例，因此 FP 率为 N/A。

| Method | Internal Tumor | Internal Overall | IRCADb Tumor | IRCADb Overall | IRCADb FP | HCC v2 Tumor | HCC v2 Overall |
|---|---:|---:|---:|---:|---:|---:|---:|
| MedNeXt | 0.7600 | 0.8561 | 0.6900 | 0.8280 | 60% | 0.4175 | 0.6279 |
| MedNeXt_SizeOV4 | 0.7635 | 0.8590 | 0.7131 | 0.8391 | 60% | 0.3405 | 0.5775 |
| MedNeXt_MHA | 0.7544 | 0.8538 | 0.7303 | 0.8487 | 60% | 0.4035 | 0.6235 |
| MedNeXt_MLA | 0.7535 | 0.8524 | 0.6778 | 0.8219 | 60% | **0.4645** | **0.6532** |
| MedNeXt_MLA_MoE | 0.7590 | 0.8562 | **0.7349** | **0.8511** | **40%** | 0.4080 | 0.6261 |
| MedNeXt_MLA_MoE_SizeOV4 | **0.7653** | **0.8591** | 0.7309 | 0.8479 | 60% | 0.4269 | 0.6356 |
| MedNeXt_MLA_MoE_FPSafe | 0.7453 | 0.8481 | 0.7022 | 0.8329 | 60% | 0.3807 | 0.6071 |
| MedNeXt_MHA_MoE | 0.7499 | 0.8508 | 0.7261 | 0.8463 | 40% | 0.3943 | 0.6158 |

## 5. 严格控制变量矩阵

### 5.1 标准 MLP 固定：MHA 对 MLA

| 对比 | Internal Tumor | Internal Overall | IRCADb Tumor | IRCADb Overall | IRCADb FP | HCC Tumor | HCC Overall |
|---|---:|---:|---:|---:|---:|---:|---:|
| MedNeXt_MHA | 0.7544 | 0.8538 | 0.7303 | 0.8487 | 60% | 0.4035 | 0.6235 |
| MedNeXt_MLA | 0.7535 | 0.8524 | 0.6778 | 0.8219 | 60% | 0.4645 | 0.6532 |
| MHA - MLA | +0.0009 | +0.0014 | **+0.0525** | **+0.0268** | 0 | **-0.0610** | **-0.0297** |

结论：MHA 在 internal 基本持平、在 IRCADb 更好，但纯 MLA 在 HCC 明显更好。当前结果支持“attention 路径存在数据域依赖”，不支持任一方在三个数据域上统一占优。

### 5.2 MLA attention 固定：MLP 对 MoE

| 对比 | Internal Tumor | Internal Overall | IRCADb Tumor | IRCADb Overall | IRCADb FP | HCC Tumor | HCC Overall |
|---|---:|---:|---:|---:|---:|---:|---:|
| MLA + MLP | 0.7535 | 0.8524 | 0.6778 | 0.8219 | 60% | 0.4645 | 0.6532 |
| MLA + MoE | 0.7590 | 0.8562 | 0.7349 | 0.8511 | 40% | 0.4080 | 0.6261 |
| MoE - MLP | +0.0055 | +0.0038 | **+0.0571** | **+0.0292** | -20 pp | **-0.0565** | **-0.0271** |

结论：在 MLA attention 固定时，MoE 在 internal 小幅上升，在 IRCADb 同时改善肿瘤 Dice、Overall 和假阳性，但 HCC 下降。MoE 的收益具有数据域依赖性。

### 5.3 MHA attention 固定：MLP 对 MoE

| 对比 | Internal Tumor | Internal Overall | IRCADb Tumor | IRCADb Overall | IRCADb FP | HCC Tumor | HCC Overall |
|---|---:|---:|---:|---:|---:|---:|---:|
| MHA + MLP | 0.7544 | 0.8538 | 0.7303 | 0.8487 | 60% | 0.4035 | 0.6235 |
| MHA + MoE | 0.7499 | 0.8508 | 0.7261 | 0.8463 | 40% | 0.3943 | 0.6158 |
| MoE - MLP | **-0.0045** | **-0.0030** | **-0.0042** | **-0.0024** | -20 pp | **-0.0092** | **-0.0077** |

结论：在 MHA attention 固定时，MoE 在 internal、IRCADb 和 HCC 的 Tumor Dice/Overall 均小幅下降，仅 IRCADb 无肿瘤病例误报率改善 20 个百分点。因此 MoE 在 MHA 路径下没有提高三域分割精度。

### 5.4 MoE 固定：MHA 对 MLA

| 对比 | Internal Tumor | Internal Overall | IRCADb Tumor | IRCADb Overall | IRCADb FP | HCC Tumor | HCC Overall |
|---|---:|---:|---:|---:|---:|---:|---:|
| MHA + MoE | 0.7499 | 0.8508 | 0.7261 | 0.8463 | 40% | 0.3943 | 0.6158 |
| MLA + MoE | 0.7590 | 0.8562 | 0.7349 | 0.8511 | 40% | 0.4080 | 0.6261 |
| MHA - MLA | **-0.0091** | **-0.0054** | **-0.0088** | **-0.0048** | 0 pp | **-0.0137** | **-0.0103** |

结论：在 MoE-FFN 固定时，MLA 在 internal、IRCADb 和 HCC 三域都优于 MHA，Overall 分别高 0.0054/0.0048/0.0103。虽然差值较小，方向在三域一致。

## 6. 训练策略消融

### 6.1 SizeOV4

| 固定架构 | 对照 | Internal Overall 变化 | IRCADb Overall 变化 | HCC Overall 变化 | 判断 |
|---|---|---:|---:|---:|---|
| MedNeXt | Base → SizeOV4 | +0.0029 | +0.0111 | -0.0504 | 收益小且跨域不稳定 |
| MLA + MoE | Base → SizeOV4 | +0.0029 | -0.0032 | +0.0095 | IRCADb 小幅下降，HCC 小幅上升 |

SizeOV4 改变的是病例采样/曝光节奏，不是 attention 结构。它不能解释 `MedNeXt_MLA_MoE` 的 IRCADb 收益。

### 6.2 FP-Safe

| 固定架构 | Internal Overall | Internal FP | IRCADb Overall | IRCADb FP | HCC Overall |
|---|---:|---:|---:|---:|---:|
| MLA + MoE | 0.8562 | 66.67% | 0.8511 | 40% | 0.6261 |
| MLA + MoE + FP-Safe | 0.8481 | 33.33% | 0.8329 | 60% | 0.6071 |

FP-Safe 改善了内部阴性误报，但三个数据域的 Overall 都下降，因此应当作为负结果报告，不应作为主方法贡献。
