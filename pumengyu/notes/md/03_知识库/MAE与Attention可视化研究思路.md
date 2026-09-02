# MAE 预训练与 ViT Attention 可视化研究思路

> 记录日期：2026-07-23
> 定位：当前 MedNeXt_MLA/MoE 三域论文完成后的下一阶段探索，不直接混入现有主结果。
> 核心目标：一方面用 MAE 检验无标签 CT 预训练能否改善跨域肿瘤检出，另一方面通过多层 attention/feature 可视化观察 Transformer 表征从局部纹理到全局语义的变化。

## 1. 两条研究线

### 1.1 3D MAE 自监督预训练

基本设想：把三维 CT 切分为 3D patches，随机遮挡大部分 patch，只将可见 token 输入 encoder，再用轻量 decoder 重建被遮挡区域。预训练结束后丢弃 decoder，将 encoder 用于肝脏与肿瘤分割微调。

希望回答的问题：

1. 在不增加人工标签的情况下，MAE 是否能改善肿瘤分割，尤其是 HCC 外部域的 Recall？
2. MAE 的收益主要来自肝脏整体结构、肿瘤局部纹理，还是更稳定的跨域强度表征？
3. MAE 对小病灶、低对比度病灶和共识失败病例是否有效，还是只提高容易病例的平均 Dice？
4. MAE 预训练后的 ViT/Transformer 特征是否呈现更清晰的肝脏—肿瘤空间分离？

### 1.2 ViT/Attention 多层可视化

基本设想：从 Transformer 的浅层、中层和深层提取 attention 权重与 token features，将 token 恢复为空间网格并映射回 CT，观察不同层关注区域和语义结构的变化。

希望回答的问题：

1. 浅层是否更关注边缘、纹理和局部强度变化？
2. 中层是否逐步形成肝脏范围、血管或病灶候选区域？
3. 深层是否更集中于肿瘤及肝脏—肿瘤全局关系？
4. 正确病例、漏检病例和假阳性病例的 attention/feature 演化是否存在稳定差异？
5. MAE 预训练前后，多层表征是否发生可重复的变化？

## 2. MAE 最小可行实验

第一阶段只做严格、可解释的最小对照，不同时改变 backbone、训练数据和微调策略。

| 组别 | Encoder 初始化 | 分割微调 | 目的 |
|---|---|---|---|
| A | 随机初始化 | 全量微调 | 基线 |
| B | 3D MAE 预训练 | 全量微调 | 判断 MAE 的总体收益 |
| C（可选） | 3D MAE 预训练 | 先冻结、再解冻 | 观察预训练表征是否稳定 |

建议先选择一个明确的 ViT/UNETR 类 encoder 完成 A/B 对照，再考虑把 MAE 思路迁移到 MedNeXt 或当前 bottleneck 模块。否则同时更换主干和预训练方法，无法判断提升来源。

### 2.1 数据边界

- 主实验预训练默认只使用 Dataset003 的训练划分，不使用 validation/test 标签。
- 最严格设置下，连 internal test、IRCADb 和 HCC test 的无标签图像也不进入预训练。
- 如果后续使用外部无标签 CT，必须单列为“额外无标签数据预训练”或 transductive 设置，不能与 source-only 主表混排。
- 所有微调继续使用相同的 92/13/26 划分、训练轮数、checkpoint 选择和 PMY-LT-v1 指标。

### 2.2 初始超参数范围

- 3D patch size：先根据显存和 token 数选择，如 `16×16×16` 或更小 patch。
- mask ratio：从 `75%` 起步，再比较 `50%/75%/90%`，但第一轮只固定一个值。
- 重建目标：先使用归一化 CT voxel/patch；后续再考虑重建梯度、局部统计量或离散 token。
- decoder：保持轻量，避免重建 decoder 变成主要容量来源。
- 微调：与随机初始化组保持完全相同的数据增强、优化器和训练轮数。

### 2.3 评价内容

除 Liver/Tumor/Overall 外，重点看：

- HCC Tumor Recall 是否提高，避免只看 Precision 或 Overall；
- tiny/small tumor Dice 与 Recall；
- 严重失败病例数和完全漏检数；
- internal、IRCADb、HCC 三个数据域是否同向；
- 预训练是否只是加快收敛，还是最终 `checkpoint_best` 确实更好；
- 至少重复多个随机种子或给出 bootstrap 置信区间，避免把单次小差值解释为 MAE 有效。

## 3. 多层 Attention/Feature 提取方案

### 3.1 建议提取位置

对于标准 12-block ViT，可先取：

```text
浅层：block 0 / 2
中层：block 5 / 7
深层：block 10 / 11
```

对于当前只有 2 个 bottleneck attention blocks 的 MedNeXt_MLA/MHA 系列，则提取：

```text
输入 bottleneck feature
attention block 1 输出
attention block 2 输出
最终 LayerNorm 输出
```

同时保存每层的：

- 每个 head 的 attention matrix；
- head-average attention；
- attention rollout（跨层累积）；
- token feature，用 PCA 降到 3 个通道形成伪彩色特征图；
- token norm 或 activation magnitude；
- 最终 prediction、GT、FP、FN mask。

### 3.2 空间恢复

1. 将 patch/token 顺序恢复为 `D×H×W` 网格。
2. 对选定 query token，显示其对所有 key token 的注意力分布。
3. 对肿瘤区域 query、肝脏区域 query 和背景 query 分别取样，避免只看单个任意 token。
4. 将低分辨率热图插值回原 CT 空间，并与 CT、GT、预测轮廓叠加。
5. 除二维轴向切片外，可补充冠状位、矢状位或 3D MIP，防止单切片选择偏差。

### 3.3 推荐图版

每个病例使用统一布局：

```text
CT | GT/Prediction | 浅层 | 中层 | 深层 | Attention Rollout | FP/FN
```

病例至少包括：

- 正确分割的代表病例；
- 小病灶或低对比度漏检病例；
- IRCADb 无肿瘤误报病例，如 `ircadb_014`；
- IRCADb 共识失败病例，如 `ircadb_018`；
- HCC 全部方法严重失败病例，如 `HCC_003`、`HCC_068`；
- HCC 相对成功病例，作为困难病例的匹配对照。

## 4. 实现原则

- 优先使用 forward hook 或显式 `return_attention/return_features` 接口，不改变模型输出和 checkpoint 参数名。
- 在 `model.eval()` 与 `torch.no_grad()` 下提取，固定预处理、patch 位置和随机种子。
- 对滑窗推理需记录 patch 的原图坐标；若要得到整幅 attention 图，需要设计重叠 patch 的聚合方式。
- attention matrix 可能很大，默认只保存选定层、选定 head、选定 query 或降采样结果，不直接保存所有病例的完整矩阵。
- 先验证加入 hook 前后分割输出逐元素一致，确保可视化代码没有改变推理结果。
- 所有图保存模型、checkpoint、case、slice、layer、head、query token 和归一化方法等元数据。

## 5. 解释边界

Attention map 可以说明模型内部的信息路由或相关性，但不能直接等同于因果解释，也不能把高 attention 区域自动称为模型“做出判断的原因”。

因此论文中更稳妥的写法是：

```text
多层 attention/feature 可视化显示模型表征与肝脏、肿瘤或错误区域之间存在空间对应关系；
该结果用于辅助解释指标和病例差异，不单独作为机制因果证据。
```

如需加强机制证据，可进一步加入：

- attention/feature 与 GT tumor mask 的重叠或 point-biserial correlation；
- 对高 attention token 做遮挡/扰动实验；
- layer/head ablation；
- linear probe，检验不同层的肝脏/肿瘤可分性；
- MAE 前后同一病例、同一层和同一 query 的配对比较。

## 6. 推荐推进顺序

1. 先在一个已训练 ViT/Transformer 模型上完成多层特征提取 demo，确认 token 能正确恢复到三维空间。
2. 再实现 attention map、head-average 和 rollout，并用一个成功病例与一个失败病例验证图是否可解释。
3. 建立 3D MAE 最小预训练与随机初始化 A/B 对照。
4. 完成三域微调评估，重点检查 HCC Recall、严重失败数和小病灶结果。
5. 对同一批病例比较 MAE 前后多层特征与 attention，形成定量指标与可视化相互支撑的分析。
6. 只有当重复实验显示稳定收益时，才将 MAE 纳入下一篇论文的方法贡献；否则作为预训练负结果或表征分析记录。

## 7. 预期产物

```text
MAE checkpoint + 预训练日志
随机初始化/MAE 微调的三域完整评估产物
逐层 feature 与 attention 提取脚本
成功/失败病例的统一可视化图版
layer/head 定量分析表
MAE 前后配对结果与统计检验
```

当前阶段先记录思路，不启动训练，也不把 MAE 或 attention 可视化写成现有论文已经完成的贡献。
