# MedNeXt_MLA 贡献成立条件与补强计划

## 1. 这项工作能不能算方法贡献

可以。MedNeXt_MLA 与 ConvNeXt 到 MedNeXt 属于相似的研究范式：将其他领域或通用视觉中的有效思想，经过任务导向的结构适配，迁移到三维医学图像分割。

可主张的核心思路是：

```text
三维卷积擅长局部纹理与边界建模
→ 对远距离解剖关系的显式建模有限
→ 标准 Attention 可以补充全局交互
→ 三维高分辨率 Attention 的计算与显存开销较高
→ 在 token 数最少的 MedNeXt bottleneck 使用低秩 MLA
→ 保留强局部卷积骨干，以最小结构增量补充全局上下文
```

但贡献不能只依赖“第一次将 MLA 加到 MedNeXt”的描述。贡献强度取决于三个方面：

1. 三维任务适配是否有清晰技术动机；
2. 是否通过严格消融隔离 MLA 的作用；
3. 是否在多个内外部数据上获得可重复的证据。

## 2. 为什么 MedNeXt 的贡献更容易成立

MedNeXt 并不只是将 ConvNeXt 的二维卷积替换为三维卷积。它系统完成了：

- 面向三维 U-Net 的 depthwise inverted bottleneck 适配；
- 对应的三维下采样、上采样和残差路径；
- kernel size、深度、宽度和 expansion ratio 的配置与扩展；
- 与医学分割预处理、patch 训练和滑窗推理的集成；
- 多个数据集上的精度、效率与消融实验。

因此，MedNeXt 的贡献是“面向三维医学分割的系统性架构重构与实证”，而不是简单的维度替换。

MedNeXt_MLA 也需要建立同等完整的“问题—设计—消融—效率—外部验证”证据链。

## 3. 当前已经具备的贡献基础

### 3.1 问题是明确的

- MedNeXt 具有强局部卷积表征，但可能更依赖源域纹理与对比度模式。
- 肝肿瘤需要结合肝脏整体形态、病灶位置和跨切片上下文。
- 内部高分不等于外部可靠，现有 IRCADb 结果已显示明显的模型排名反转。

### 3.2 结构选择是克制的

- 保留 MedNeXt encoder-decoder 主体；
- 仅在最低空间分辨率引入全局交互；
- 使用低秩 KV 投影，而不在高分辨率 stage 堆叠多层 Attention；
- 可以与原 MedNeXt 进行直接结构对照。

### 3.3 已有外部收益线索

当前 source-only IRCADb 结果中：

```text
MedNeXt      External Overall = 0.7705
MedNeXt_MLA  External Overall = 0.8079
```

MedNeXt_MLA_MoE 还提高了 Tumor Dice 和 Precision，并将无肿瘤病例误报率从 60% 降到 40%。纯 MLA 已完成的结果未复现这一收益，因此当前证据只支持 MLA+MoE 组合与特定外部域上的可靠性改善相关。

## 4. 当前缺少的关键证据

### 4.1 缺少 Attention 的严格单变量消融

当前已完成的三个实现为：

```text
MedNeXt_MLA     = MLA + standard MLP
MedNeXt_MHA     = standard MHA + standard MLP
MedNeXt_MLA_MoE = MLA + MoE FFN
```

MedNeXt_MLA 与 MedNeXt_MHA 已形成同为标准 MLP 的严格 attention 对照；当前结果不支持 MLA 优于标准 MHA。MLA+MoE 的外部收益仍不能区分是 MoE 独立作用还是 MLA×MoE 交互作用。

需要增加严格对照：

```text
MedNeXt_MHA_MoE = standard MHA + same MoE FFN
MedNeXt_MLA_MoE = MLA + same MoE FFN
```

`MedNeXt_MHA = standard MHA + standard MLP` 已完成；历史结果名 `MedNeXt_Transformer` 统一迁移为 `MedNeXt_MHA`。

### 4.2 缺少三维设计消融

需要回答：

- 为什么放在 bottleneck，而不是 encoder 高层或多尺度位置？
- 1、2、4 个 MLA block 有什么差异？
- compression ratio 为 2、4、8 时，精度与效率如何变化？
- 是否需要三维位置信息？
- MLA 与卷积 bottleneck 使用直接顺序连接、残差融合或门控融合时有何差异？

不必堆叠更多模块，但必须证明当前简单设计不是随意选择。

### 4.3 缺少效率证据

当前 MLA 投影的参数复杂度约为：

$$
2C^2 + \frac{3C^2}{r},
$$

标准 MHA 投影的参数复杂度约为：

$$
4C^2.
$$

当 $r=4$ 时，MLA Attention 投影约为 $2.75C^2$，低于标准 MHA。但当前实现仍显式构造完整的 $N\times N$ Attention 矩阵，因此 token 交互复杂度仍为 $O(N^2)$。

需要报告：

- 整体参数量与 Attention 子模块参数量；
- FLOPs 或 MACs；
- 训练峰值显存；
- 单个体数据推理时间；
- 在相同 FFN 下 MLA 与 MHA 的效率对照。

当前不应笼统声称“整个 MedNeXt_MLA 比普通 Transformer 更轻”，因为两个模型的 FFN 结构尚不相同。

### 4.4 缺少更强的重复性与外部证据

当前主要证据来自 20 例 IRCADb，同时存在：

- 内部 Overall 下降；
- HCC source-only 上未形成稳定领先；
- HCC Adapter 混合验证尚未完成正式分析；
- 标准 Transformer 消融尚未完成；
- 当前主要是单 fold、单 seed 证据。

理想情况下应增加多 seed 或重复试验、置信区间或配对病例统计。在此之前，论文应写成“MLA 改善了 IRCADb 外部域上的可靠性”，不应扩大为“MLA 普遍改善三维肝肿瘤分割的跨域泛化”。

### 4.5 缺少机制证据

需要进一步分析：

- 收益主要来自 FP 减少还是 FN 减少？
- MLA 主要改善哪个肿瘤大小组？
- 哪些病例由失败变为成功？
- 无肿瘤病例的肿瘤预测概率分布是否下降？
- Attention 或 bottleneck feature 是否更关注肝脏内部及病灶相关区域？
- `ircadb_016` 等关键病例的改善能否由连续切片和特征可视化支持？

## 5. 建议的最小完整消融矩阵

| Experiment | 目的 | 必要性 |
|---|---|---|
| MedNeXt | 无 Attention 基线 | 必须 |
| MedNeXt_MHA | 普通 MHA + MLP | 已完成 |
| MedNeXt_MHA_MoE | 与 MLA 保持相同 FFN | 必须 |
| MedNeXt_MLA | 主方法 | 必须 |
| MLA ratio=2/4/8 | 低秩压缩比 | 建议 |
| MLA blocks=1/2/4 | 模块深度 | 建议 |
| encoder-high vs bottleneck | 三维插入位置 | 建议，可缩小实验规模 |
| SizeOV4 + MLA | 采样与结构交互 | 已有 |

核心主表至少需要：

```text
Method
Parameters
Peak memory
Internal Overall / Tumor Dice
IRCADb Overall / Tumor Dice / Precision / FP rate
HCC Overall / Tumor Dice / Recall
```

## 6. 论文中可以如何定义贡献

建议表述：

> 本文提出一种面向三维肝肿瘤分割的 bottleneck latent-attention 架构。该方法保留 MedNeXt 的局部卷积编码器—解码器，仅在最低空间分辨率上对三维特征进行低秩 KV 压缩与全局 token 交互，以最小结构改动补充全局肝脏—肿瘤上下文。通过与无 Attention MedNeXt、标准 MHA 及不同采样策略的对照，本文研究低秩全局上下文对内部分割性能、外部泛化和无肿瘤误报的影响。

在证据尚未补齐前，建议保守表述：

> MedNeXt_MLA 在 3D-IRCADb 上显示出更好的外部肿瘤分割和假阳性控制，但该收益尚未在所有外部数据域上稳定重现。

不建议表述：

> MLA 普遍提高三维肝肿瘤分割的跨域泛化能力。

## 7. 问题驱动而非模块堆叠

本文应坚持以下方法论：

> 先将复杂问题提炼为明确矛盾，再用最简单、最可验证的机制解决。

对应关系为：

```text
深层优化困难       → 残差恒等路径
局部卷积长程建模不足 → Attention
三维 Attention 开销较高 → bottleneck + low-rank MLA
```

Mamba、Transformer、MoE 或其他复杂模块本身不是“好”或“坏”。问题在于：如果仅因为某个结构流行就将其加入网络，却无法说明它解决了肝肿瘤分割中的哪个可定义问题，也没有独立消融证明收益，那么这种设计就只是复杂度堆叠，不是有说服力的方法贡献。

## 8. 执行清单

- [x] MedNeXt 基线结果
- [x] MedNeXt_MLA 内部与 IRCADb 结果
- [x] MedNeXt_MLA_SizeOV4 采样交互对照
- [x] MedNeXt_MHA（MHA + MLP；历史名 MedNeXt_Transformer）
- [ ] MedNeXt_MHA_MoE 严格单变量对照
- [ ] 压缩比消融
- [ ] MLA block 数消融
- [ ] 插入位置消融
- [ ] 参数量、FLOPs、峰值显存和推理时间
- [ ] 三域 best-checkpoint 统一报告
- [ ] FP/FN、肿瘤大小组和关键病例机制分析
- [ ] 根据新结果更新论文摘要、消融表和结论
