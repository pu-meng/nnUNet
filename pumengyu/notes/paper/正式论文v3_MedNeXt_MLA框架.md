# 面向外部可靠性的肝肿瘤 CT 分割：基于 MedNeXt_MLA 的跨数据集验证与视觉歧义分析

**作者**：普孟玉  
**单位**：数学学院

---

## 摘要

肝肿瘤三维分割模型在内部测试集上取得较高 Dice，并不必然意味着其在外部临床数据上具有同等可靠性。针对这一问题，本文围绕肝肿瘤 CT 分割中的外部可靠性展开研究，系统比较 nnU-Net、SwinUNETR、nnFormer、MedNeXt 及多种注意力和采样改进方法在内部测试集与外部验证集上的表现。实验发现，MedNeXt 和 MedNeXt_SizeOV4 在 Dataset003/LiTS 风格内部测试集上取得较高 Overall，分别为 0.8402 和 0.8431；但在 3D-IRCADb 外部验证中分别下降到 0.7705 和 0.7797，表现出明显 internal-external drop。为改善这一问题，本文提出 MedNeXt_MLA，在 MedNeXt-L 最低分辨率 bottleneck 后引入低秩 Multi-head Latent Attention，以可控计算成本补充全局 liver-tumor context。结果显示，MedNeXt_MLA 内部 Overall 为 0.8259，低于 MedNeXt_SizeOV4，但在 3D-IRCADb 外部验证中达到最高 Overall 0.8079 和 Tumor Dice 0.6484，internal-external drop 仅为 -0.0180，并将外部无肿瘤误报率由 MedNeXt 系列的 60% 降至 40%。在额外 HCCReferencedCT held-out test set 上，MedNeXt_MLA_SizeOV4 取得最高 Overall 0.7038 和 Tumor Dice 0.5150，提示不同外部数据集上的最优配置可能存在差异。本文还从 CT 视觉歧义角度分析假阳性、漏检和 3D 上下文误差来源，提示肝肿瘤分割模型评估应从单一内部 Dice 转向外部验证、无肿瘤误报和失败模式解释。

**关键词**：肝肿瘤分割；MedNeXt；潜在注意力；跨数据集验证；外部可靠性；视觉歧义

---

## 1. 引言

肝细胞癌、肝转移瘤等恶性肿瘤是临床影像诊断和治疗决策中的重要对象。基于 CT 的肝脏与肝肿瘤三维分割可用于术前规划、疗效评估、放疗或消融靶区设计以及随访体积变化分析。近年来，nnU-Net 通过自适应预处理、网络配置和训练策略，成为医学图像分割任务中的强基线。MedNeXt 进一步引入 ConvNeXt 风格的倒置瓶颈、深度可分离卷积、GroupNorm 和残差结构，在三维医学图像分割中表现出较强的局部表征能力。

然而，肝肿瘤分割的临床难点并不只体现在内部测试集平均 Dice 上。真实部署中，模型需要面对来自不同中心、不同扫描协议、不同病灶负荷和不同图像对比度的数据。一个在内部测试集上得分最高的模型，可能在外部数据上出现小病灶漏检、无肿瘤病例误报或肿瘤边界过分割。对于肝肿瘤这类边界模糊、强化模式复杂且病灶大小差异显著的任务，仅以内部分数选择模型可能高估其临床可靠性。

本文的实验观察进一步支持这一问题。在 Dataset003/LiTS 风格内部测试集中，MedNeXt_SizeOV4 与 MedNeXt 的 Overall 分别达到 0.8431 和 0.8402，位于内部结果前列；但在 3D-IRCADb 外部验证集中，二者 Overall 分别下降至 0.7797 和 0.7705，drop 分别为 -0.0634 和 -0.0697。相反，MedNeXt_MLA 内部 Overall 为 0.8259，并非内部最高，但其外部 Overall 达到 0.8079，成为外部验证中表现最好的方法。这种内部排名与外部排名不一致的现象说明，肝肿瘤分割研究需要从“内部高 Dice”进一步转向“外部可靠性”。

为此，本文在强 MedNeXt-L 骨干基础上提出 MedNeXt_MLA。该方法保持 MedNeXt 编码器、解码器和训练配置不变，仅在最低分辨率 bottleneck 后加入 MLABottleneck3D。该模块通过低秩 Multi-head Latent Attention 在语义最抽象、空间分辨率最低的位置建模全局依赖，用全局 liver-tumor context 补充局部卷积骨干的归纳偏置。与在高分辨率特征上直接加入全局注意力相比，bottleneck 位置具有更低计算开销，也更适合聚合整体解剖结构与病灶上下文。

本文主要贡献如下：

1. 从内部测试与外部验证的差异出发，系统揭示肝肿瘤 CT 分割中“内部最优不等于外部最可靠”的问题。
2. 提出 MedNeXt_MLA，在 MedNeXt-L bottleneck 后引入低秩潜在注意力模块，使强局部卷积骨干获得全局 liver-tumor context。
3. 在 Dataset003/LiTS 风格内部测试集和 3D-IRCADb 外部验证集上比较 CNN、Transformer、MedNeXt 及多种自研改进方法，证明 MedNeXt_MLA 在外部验证中取得最高 Overall 并显著减小 internal-external drop。
4. 通过 MedNeXt、MedNeXt_SizeOV4、MedNeXt_MLA 和 MedNeXt_MLA_SizeOV4 的消融实验说明，均匀 SizeOV4 采样不是外部收益主因，bottleneck latent attention 对外部可靠性改善更关键。
5. 从 CT 视觉歧义、无肿瘤误报、小病灶漏检和 3D 上下文误差角度分析模型失败模式，为肝肿瘤分割模型的临床安全性评估提供补充依据。

## 2. 相关工作

### 2.1 医学图像分割与 nnU-Net

U-Net 及其三维扩展长期是医学图像分割的基础架构。nnU-Net 进一步将预处理、网络结构、patch size、batch size、数据增强、训练策略和后处理流程自动配置化，在多种医学分割任务中形成稳定强基线。对于肝脏和肝肿瘤分割任务，nnU-Net 的优势在于训练流程稳健、超参数依赖较少，但其默认 PlainConvUNet 仍主要依赖局部卷积特征，面对跨中心外观差异时可能出现泛化下降。

### 2.2 MedNeXt 与 ConvNeXt 风格三维分割

MedNeXt 将 ConvNeXt 风格设计迁移到三维医学图像分割中，通过倒置瓶颈、深度可分离卷积、GroupNorm、残差连接和更深层 encoder-decoder 强化局部表征能力。本文实验中，MedNeXt-L 在内部测试集上表现强于多数 CNN 和 Transformer 对照方法，说明其结构适合学习肝脏与肿瘤局部纹理和边界特征。但内部高分并不能自动保证外部可靠性，MedNeXt 在 3D-IRCADb 上出现明显 drop，提示强局部骨干仍可能学习到源域特异的外观模式。

### 2.3 Transformer 与全局上下文建模

Transformer 及其医学图像变体，如 UNETR、SwinUNETR、nnFormer 等，通过自注意力或窗口注意力建模长程依赖，为三维分割提供了全局上下文表达能力。然而，三维医学图像体素数量大，直接在高分辨率特征上进行全局注意力会带来较高显存和计算成本。窗口注意力和层级结构可以缓解这一问题，但也会引入复杂结构设计。本文采用 bottleneck attention 的折中思路：在最低空间分辨率处引入全局交互，以较低成本补充卷积骨干的上下文建模能力。

### 2.4 低秩潜在注意力

Multi-head Latent Attention 通过低秩压缩键和值表示，降低注意力模块的参数和计算开销。本文并不关注语言模型中的 KV cache 场景，而是借鉴其低秩全局交互思想，将其迁移到三维肝肿瘤分割的 bottleneck 特征上。对于 MedNeXt-L 的最低分辨率特征，MLA 可以在不改变主干结构和训练目标的情况下，引入全局 liver-tumor context。

### 2.5 外部验证与临床可靠性

医学图像分割模型常在内部交叉验证或单一测试集上报告平均 Dice，但临床应用更关心跨中心、跨扫描协议和跨病灶分布的稳定性。肝肿瘤分割还具有特殊风险：无肿瘤病例误报可能增加不必要的临床复核，小病灶漏检可能影响治疗决策，边界过分割可能影响体积评估。因此，外部验证、无肿瘤误报率、召回率、精确率和典型失败案例分析应与平均 Dice 一同构成可靠性评估。

## 3. 方法

### 3.1 总体结构

MedNeXt_MLA 以 MedNeXt-L 为基础，保持其 encoder、decoder、deep supervision 和训练配置不变，仅在最低分辨率 bottleneck 后插入 MLABottleneck3D。整体流程如下：

```text
Input CT
  -> MedNeXt stem
  -> MedNeXt encoder blocks
  -> MedNeXt bottleneck
  -> MLABottleneck3D
  -> MedNeXt decoder blocks
  -> segmentation logits
```

该设计遵循三个原则：第一，不改变 MedNeXt 的局部卷积主干，使结果可直接与 MedNeXt 对照；第二，不引入额外标注或额外监督，避免将收益归因于数据条件变化；第三，只在最低分辨率特征处加入全局上下文模块，以控制显存和计算成本。

### 3.2 MedNeXt-L 骨干

本文使用 MedNeXt-L kernel=3 配置作为主干网络。主要配置如下：

```text
n_channels = 32
kernel_size = 3
exp_r = [3, 4, 8, 8, 8, 8, 8, 4, 3]
block_counts = [3, 4, 8, 8, 8, 8, 8, 4, 3]
do_res = True
do_res_up_down = True
norm_type = group
```

该骨干通过更深的编码器-解码器结构、倒置瓶颈、深度可分离卷积和残差连接增强局部特征学习能力。本文不试图逐一拆解 MedNeXt 内部每个组件的独立贡献，而是将其作为强局部卷积骨干，研究在其 bottleneck 处补充全局上下文是否能改善外部可靠性。

### 3.3 MLABottleneck3D

MLABottleneck3D 接收 MedNeXt bottleneck 输出特征：

```text
x: (B, C, D, H, W), C = 512
flatten -> (B, N, C)
MLA blocks
reshape -> (B, C, D, H, W)
```

其中，`N = D x H x W` 为空间 token 数。MLA 首先将输入特征投影到低秩潜在空间，再从压缩表示中恢复 key 和 value，同时直接由输入生成 query。其基本计算形式为：

```text
c_kv = x W_DKV
K = c_kv W_UK
V = c_kv W_UV
Q = x W_Q
Attention = softmax(QK^T / sqrt(d)) V
```

默认设置为：

```text
num_heads = 8
num_blocks = 2
compression_ratio = 4
mlp_ratio = 4
```

通过低秩 key-value 表示，MLA 在保持 token 间全局交互的同时降低额外开销。由于模块位于 bottleneck，空间 token 数已经显著减少，因此相比高分辨率注意力更适合三维分割任务。

### 3.4 Bottleneck 位置的动机

本文将 MLA 放置在 bottleneck 后，主要基于以下考虑。首先，bottleneck 特征具有最低空间分辨率，全局注意力的计算成本相对可控。其次，该位置语义抽象程度最高，更适合整合肝脏整体形态、病灶位置和周围组织上下文。再次，该设计保留了 MedNeXt 编码器和解码器中的局部卷积归纳偏置，避免在高分辨率层引入过多结构扰动。最后，bottleneck attention 可以作为强卷积骨干的轻量上下文补充，使消融结果更容易解释。

### 3.5 SizeOV4 采样对照

为区分全局上下文建模和采样策略的作用，本文将 SizeOV4 作为对照实验。SizeOV4 将不同肿瘤大小组和无肿瘤病例在训练 identifier 列表中等比例重复：

```text
tiny = 2
small = 2
mid = 2
huge = 2
no_tumor = 2
```

由于所有分组均等比例重复，SizeOV4 并不实质改变训练集大小分布。实验结果显示，SizeOV4 在 MedNeXt 上仅带来有限外部收益，而与 MedNeXt_MLA 组合后外部表现下降。因此，SizeOV4 在本文中作为采样因素对照，而不是最终主贡献。

## 4. 实验设计

### 4.1 数据集

内部数据使用 Dataset003_Liver / LiTS 风格数据，采用固定 fold_0 划分进行训练、验证和内部测试。该数据集作为源域，用于训练各类模型并评估内部测试性能。

外部验证使用 3D-IRCADb，共 20 个病例，其中 15 个为有肿瘤病例，5 个为无肿瘤病例。3D-IRCADb 与内部数据在病例来源、扫描条件和病灶表现上存在差异，适合作为跨数据集外部验证。本文特别保留无肿瘤病例分析，用于评估模型在外部数据上的假阳性风险。

此外，本文使用 HCCReferencedCT 作为额外 held-out test set。该数据集采用固定 70/10/21 划分，其中 70 例用于训练划分，10 例用于验证划分，21 例作为 test set。本文仅在 21 例 test 病例上进行推理和有标签评估，HCC 的 train/val 病例不参与该外部测试结果的模型选择。该 test set 的 21 例均为有肿瘤病例，因此适合评估 HCC 临床数据上的肿瘤分割性能，但不能用于分析无肿瘤误报率。

### 4.2 评价指标

本文报告以下指标：

```text
Liver Dice
Tumor Dice
Overall = (Liver Dice + Tumor Dice) / 2
Recall
Precision
No-tumor FP rate
Internal-external drop = External Overall - Internal Overall
```

主排序指标为 Overall。由于肝肿瘤临床安全性不仅取决于平均 Dice，本文同时关注 Tumor Dice、Recall、Precision 和无肿瘤误报率。对于跨数据集泛化，本文使用 internal-external drop 衡量外部性能相对内部性能的下降幅度。需要注意的是，无肿瘤误报率仅在包含无肿瘤病例的外部集上报告；HCCReferencedCT test set 全部为有肿瘤病例，因此不计算该指标。

### 4.3 对比方法

本文比较的方法包括 nnU-Net Baseline、SizeOV 系列、MLAUNet、MoE_SizeOV5、MedNeXt、MedNeXt_SizeOV4、MedNeXt_MLA、MedNeXt_MLA_SizeOV4、SwinUNETR 和 nnFormer 等。正式主线重点分析 MedNeXt 系列及 MedNeXt_MLA，因为它们共享强 MedNeXt-L 骨干，更适合判断 bottleneck latent attention 与采样策略的作用。

未完成或外部结果明显无效的方法不纳入主表结论，仅作为内部探索记录保留。例如，外部 tumor 结果全 0 的探索 trainer 不用于正式比较。

## 5. 实验结果

### 5.1 外部验证结果

表 1 给出 3D-IRCADb 外部验证中表现靠前的方法及关键 MedNeXt 对照。MedNeXt_MLA 取得最高 External Overall 0.8079，Liver Dice 为 0.9673，Tumor Dice 为 0.6484，Recall 为 0.6665，Precision 为 0.7437，无肿瘤误报率为 40%。相比之下，MedNeXt_SizeOV4 和 MedNeXt 虽然内部 Overall 更高，但外部 Overall 分别为 0.7797 和 0.7705，无肿瘤误报率均为 60%。

**表 1  外部验证结果与 internal-external drop**

| Rank | Method | External Overall | Liver | Tumor | Recall | Precision | FP rate | Internal Overall | Drop |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | MedNeXt_MLA | 0.8079 | 0.9673 | 0.6484 | 0.6665 | 0.7437 | 40% | 0.8259 | -0.0180 |
| 2 | MoE_SizeOV5 | 0.8025 | 0.9679 | 0.6371 | 0.6437 | 0.7464 | 40% | 0.8167 | -0.0142 |
| 3 | MLAUNet | 0.8008 | 0.9675 | 0.6341 | 0.6320 | 0.7580 | 40% | 0.8148 | -0.0140 |
| 4 | SizeOV2 | 0.7992 | 0.9676 | 0.6307 | 0.6352 | 0.7547 | 40% | 0.8187 | -0.0195 |
| 5 | MLA_GK5_V4 | 0.7957 | 0.9656 | 0.6258 | 0.6472 | 0.7291 | 40% | 0.8173 | -0.0216 |
| 13 | MedNeXt_SizeOV4 | 0.7797 | 0.9651 | 0.5943 | 0.6500 | 0.6795 | 60% | 0.8431 | -0.0634 |
| 15 | MedNeXt | 0.7705 | 0.9660 | 0.5750 | 0.6554 | 0.6564 | 60% | 0.8402 | -0.0697 |

该结果显示，内部测试中最强的 MedNeXt 系列模型在外部验证中并非最优。MedNeXt_MLA 的外部 Tumor Dice 比 MedNeXt 提高 0.0734，External Overall 提高 0.0374，Precision 提高 0.0873，同时无肿瘤误报率从 60% 降至 40%。这说明 bottleneck latent attention 主要改善了外部肿瘤分割和假阳性控制，而不是肝脏 Dice；各方法 Liver Dice 均接近 0.965 以上，差异主要来自肿瘤类别。

### 5.2 MedNeXt 系列消融

为明确 MLA 和 SizeOV4 的相对作用，本文进一步比较 MedNeXt、MedNeXt_SizeOV4、MedNeXt_MLA 和 MedNeXt_MLA_SizeOV4。结果见表 2。

**表 2  MedNeXt 系列消融结果**

| Method | Internal Overall | External Overall | Drop | Internal Tumor | External Tumor | External Precision | External FP |
|---|---:|---:|---:|---:|---:|---:|---:|
| MedNeXt | 0.8402 | 0.7705 | -0.0697 | 0.7283 | 0.5750 | 0.6564 | 60% |
| MedNeXt_SizeOV4 | 0.8431 | 0.7797 | -0.0634 | 0.7317 | 0.5943 | 0.6795 | 60% |
| MedNeXt_MLA | 0.8259 | 0.8079 | -0.0180 | 0.6982 | 0.6484 | 0.7437 | 40% |
| MedNeXt_MLA_SizeOV4 | 0.8285 | 0.7870 | -0.0415 | 0.7040 | 0.6091 | 0.7019 | 60% |

在 MedNeXt 上加入 SizeOV4 后，External Overall 从 0.7705 上升到 0.7797，External Tumor Dice 从 0.5750 上升到 0.5943，改善幅度有限，且无肿瘤误报率仍为 60%。在 MedNeXt 上加入 MLA 后，External Overall 上升到 0.8079，External Tumor Dice 上升到 0.6484，Precision 上升到 0.7437，无肿瘤误报率下降至 40%。相反，将 SizeOV4 叠加到 MedNeXt_MLA 后，External Overall 下降到 0.7870，External Tumor Dice 下降到 0.6091，无肿瘤误报率回升至 60%。

该消融说明，MedNeXt_MLA 的外部收益主要来自 bottleneck latent attention，而不是 SizeOV4 采样。SizeOV4 可以略微提高 MedNeXt 内部与外部肿瘤 Dice，但不能解释 MedNeXt_MLA 的主要外部提升。

### 5.3 内部最高分与外部最高分的排名反转

内部测试结果显示，MedNeXt_SizeOV4 和 MedNeXt 的 Overall 分别达到 0.8431 和 0.8402，高于 MedNeXt_MLA 的 0.8259。如果仅按内部测试集选型，MedNeXt_SizeOV4 会成为首选模型。然而，在 3D-IRCADb 外部验证中，MedNeXt_MLA 反而取得最高 External Overall，且 drop 明显小于 MedNeXt 和 MedNeXt_SizeOV4。

这一排名反转是本文的核心发现。它说明在肝肿瘤 CT 分割中，源域内部性能并不能充分代表模型的外部临床可靠性。特别是对于肿瘤类别，外部数据中的病灶大小、对比度、边界清晰度和无肿瘤病例比例均可能改变模型表现。因此，模型选择应同时考虑内部测试、外部验证、误报风险和典型失败模式，而不是只依据内部 Overall 或 Tumor Dice。

### 5.4 无肿瘤误报分析

3D-IRCADb 中包含 5 个无肿瘤病例，为评估外部假阳性提供了直接依据。MedNeXt 和 MedNeXt_SizeOV4 的无肿瘤误报率均为 60%，说明其在外部无肿瘤病例上容易预测出伪肿瘤区域。MedNeXt_MLA 将该指标降至 40%，表明全局上下文有助于抑制部分外部假阳性。

需要注意的是，40% 的无肿瘤误报率仍然不能满足严格临床安全需求。`ircadb_007` 和 `ircadb_014` 等病例在多种方法中均出现误报，提示这类错误可能不仅来自单一模型结构，也与单期 CT 上低密度影、血管结构、局部噪声或肝实质异质性有关。因此，无肿瘤误报应作为后续模型改进和后处理策略的重要方向。

### 5.5 典型病例分析

病例级分析显示，MedNeXt_MLA 的外部收益并非均匀来自所有病例，而是在部分困难病例上表现突出。`ircadb_016` 是一个关键例子，MedNeXt 的 Tumor Dice 为 0.2546，MedNeXt_SizeOV4 提高到 0.5515，而 MedNeXt_MLA 达到 0.8121，MedNeXt_MLA_SizeOV4 为 0.7932。该病例说明，bottleneck latent attention 可能有助于模型利用整体上下文修复局部分割失败。

同时，仍有部分病例显示出明显局限。`ircadb_008` 的小病灶对多种方法均较困难；`ircadb_018` 的极小病灶可作为漏检或低 Dice 的 limitation；`ircadb_014` 可作为无肿瘤误报示例。这些病例应在后续图示中与原始 CT、标注和不同模型预测共同展示，以说明模型成功与失败的具体来源。

### 5.6 HCCReferencedCT held-out test 结果

为进一步观察模型在另一类临床 HCC 数据上的表现，本文在 HCCReferencedCT 固定 70/10/21 划分的 21 例 held-out test set 上进行评估。所有方法均使用 Dataset003_Liver 训练得到的 checkpoint 进行推理，HCC train/val 病例不用于该外部测试结果的模型选择。与 3D-IRCADb 不同，该 test set 全部为有肿瘤病例，因此本节只分析肿瘤分割性能，不讨论无肿瘤误报率。

**表 3  HCCReferencedCT held-out test 上的主要结果**

| Rank | Method | Overall | Liver Dice | Tumor Dice | Recall | Precision | 严重失败 | Dice>=0.7 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | MedNeXt_MLA_SizeOV4 | 0.7038 | 0.8927 | 0.5150 | 0.4327 | 0.7466 | 4 | 9 |
| 2 | MedNeXt | 0.6980 | 0.8905 | 0.5055 | 0.4153 | 0.8103 | 6 | 8 |
| 3 | MedNeXt_MLA | 0.6825 | 0.8865 | 0.4786 | 0.3865 | 0.8269 | 7 | 8 |
| 4 | DeepDWIBMedConfig | 0.6778 | 0.8850 | 0.4706 | 0.3736 | 0.7699 | 6 | 5 |
| 5 | MedNeXt_MLA_FPSafe | 0.6652 | 0.8757 | 0.4546 | 0.3576 | 0.8302 | 7 | 6 |

HCCReferencedCT 上的结果与 3D-IRCADb 存在差异。MedNeXt_MLA 在 3D-IRCADb 上取得最高外部 Overall，但在 HCCReferencedCT test set 上排名第三；MedNeXt_MLA_SizeOV4 则取得最高 Overall 0.7038 和 Tumor Dice 0.5150。该结果说明，HCCReferencedCT 与 3D-IRCADb 呈现不同的外部数据特征：在 3D-IRCADb 上，单独 MLA 更有利于外部可靠性和无肿瘤误报控制；而在 HCCReferencedCT 上，MLA 与 SizeOV4 组合更有利于提高有肿瘤病例的召回和整体 Dice。

从 MedNeXt 系列内部看，MedNeXt_MLA_SizeOV4 的严重失败病例数最少，为 4 例，Dice>=0.7 的病例数最多，为 9 例。MedNeXt_MLA 的 Precision 最高，为 0.8269，但 Recall 低于 MedNeXt 和 MedNeXt_MLA_SizeOV4，因此 Tumor Dice 未达到最优。MedNeXt_MLA_FPSafe 虽然 Precision 达到 0.8302，但 Recall 下降到 0.3576，导致 Overall 和 Tumor Dice 均低于 MedNeXt_MLA，说明当前 FP-safe 策略未在该 HCC test set 上带来收益。

按肿瘤大小分组观察，MedNeXt_MLA_SizeOV4 的优势主要来自中等肿瘤和大肿瘤。其小肿瘤 Dice 为 0.5088，接近 MedNeXt 的 0.5200 和 MedNeXt_MLA 的 0.5181；中等肿瘤 Dice 为 0.4352，高于 MedNeXt 的 0.3877 和 MedNeXt_MLA 的 0.3784；大肿瘤 Dice 为 0.6699，接近 MedNeXt 的 0.6753，且高于 MedNeXt_MLA 的 0.5550。这提示 SizeOV4 与 MLA 的组合可能更有利于 HCC test set 中中等和大病灶的检出，但该结论仍需更多外部病例验证。

## 6. CT 视觉歧义与失败模式分析

肝肿瘤 CT 分割的错误不仅来自网络结构，也来自影像证据本身的歧义。单期 CT 中，肿瘤可见性受增强时相、病灶类型、病灶大小、周围肝实质状态和成像噪声影响。本文将常见失败模式归纳为三类。

第一类是“有阴影但无肿瘤”。部分无肿瘤病例中可见低密度区、血管截面、伪影或局部肝实质不均匀，这些区域在局部外观上可能接近肿瘤，导致模型产生假阳性。MedNeXt 系列在外部无肿瘤病例中的较高误报率说明，强局部纹理学习可能放大这类风险。

第二类是“无明显阴影但有肿瘤”。部分真实肿瘤在单期 CT 上与周围肝实质对比度较低，边界不清或体积极小，局部视觉证据不足，模型容易漏检或只分割出部分区域。对于这类病例，单纯增加局部卷积能力未必足够，需要结合更大范围的解剖和上下文信息。

第三类是 3D 上下文带来的双重影响。三维 patch 推理可以利用相邻切片信息提升连续病灶分割，但也可能将邻近切片中的 tumor response 延续到当前无肿瘤切片，造成边界过分割或邻近切片误报。因此，3D 上下文既是帮助模型识别隐匿病灶的重要信息，也可能成为局部误差传播的来源。

MedNeXt_MLA 的结果提示，全局上下文建模可以缓解部分视觉歧义，尤其是在外部困难病例和无肿瘤误报方面。但该结论仍应谨慎理解：本文的消融和病例分析支持 MLA 与外部可靠性改善相关，但尚不能完全证明其内部机制。后续可结合 attention map、特征可视化和更大规模外部数据进一步验证。

## 7. 讨论

### 7.1 内部 Dice 与外部可靠性

本文最重要的发现是内部最优与外部最优不一致。MedNeXt_SizeOV4 在内部测试集中 Overall 最高，但外部验证并非最佳；MedNeXt_MLA 内部分数略低，却在外部验证中表现最稳。该结果对模型选择具有直接意义：如果研究只报告内部测试结果，可能会选择外部误报更高、drop 更大的模型。对于临床应用导向的肝肿瘤分割，外部验证应成为模型评价的必要环节。

### 7.2 MLA 改善外部表现的可能原因

MedNeXt 的优势来自局部卷积表征，但这也可能使模型更依赖源域中的局部纹理、对比度和边界模式。MLA 位于 bottleneck，能够在高语义层面整合全局 token 交互，使模型判断肿瘤时不仅依赖局部低密度外观，也参考肝脏整体结构、病灶空间位置和跨区域上下文。这可能解释其在 `ircadb_016` 等病例上明显改善，并降低部分无肿瘤误报。

不过，本文不将该机制解释为定论。当前证据主要来自模型消融、外部指标和病例对比。HCCReferencedCT 的结果进一步提示，单独 MLA 并非在所有外部数据集上都最优，不同数据集上的病灶分布和标注特征可能影响 MLA 与采样策略的相对收益。更严格的机制证明需要进一步特征分析和多外部数据集验证。

### 7.3 SizeOV4 的作用边界

SizeOV4 的设计初衷是缓解不同肿瘤大小组训练机会不均衡。但当前实现对所有大小组和无肿瘤病例等比例重复，因此并未实质改变训练分布。在 3D-IRCADb 上，SizeOV4 在 MedNeXt 上带来小幅外部收益，但不能降低无肿瘤误报率；与 MedNeXt_MLA 结合后反而降低外部表现。相反，在 HCCReferencedCT held-out test 上，MedNeXt_MLA_SizeOV4 取得最高 Overall 和 Tumor Dice。该差异说明，采样策略的作用受外部数据分布影响，不能简单概括为稳定增益或稳定负增益。

### 7.4 临床安全性意义

肝肿瘤分割模型的临床安全性不仅取决于平均 Dice。无肿瘤误报可能造成额外人工复核和不必要警报，小病灶漏检可能影响治疗时机，边界过分割可能干扰体积测量。本文将无肿瘤误报率、Precision、Recall 和典型病例可视化纳入分析，目的是避免高 Overall 掩盖具体临床风险。MedNeXt_MLA 已降低部分外部误报，但仍存在固定难例，后续需要结合 FP-safe 训练、后处理或不确定性估计进一步改进。

### 7.5 局限性

本文仍存在若干局限。第一，当前主要结果基于固定 fold_0，尚未完成多 fold 统计验证。第二，3D-IRCADb 外部验证集仅包含 20 个病例，HCCReferencedCT held-out test set 也只有 21 个病例，规模有限，结论仍需更多外部中心验证。第三，HCCReferencedCT test set 全部为有肿瘤病例，不能用于评估无肿瘤误报风险。第四，本文主要证明 MedNeXt_MLA 与外部可靠性改善相关，尚未通过可解释性实验完全揭示其机制。第五，极小病灶和低对比度病灶仍然困难，如 `ircadb_018` 等病例说明模型仍存在漏检风险。第六，FP-safe 等降低假阳性的进一步策略尚未纳入正式主结果。

## 8. 结论

本文围绕肝肿瘤 CT 分割中的外部可靠性问题，系统比较内部测试与 3D-IRCADb 外部验证结果。实验表明，内部 Overall 最高的 MedNeXt_SizeOV4 和 MedNeXt 在外部验证中出现明显 drop，而在 MedNeXt-L bottleneck 后引入低秩潜在注意力的 MedNeXt_MLA 虽然内部并非最高，却取得最高外部 Overall 0.8079，并将 internal-external drop 从 MedNeXt 的 -0.0697 缩小到 -0.0180。额外 HCCReferencedCT held-out test 结果显示，MedNeXt 系列仍保持较强表现，其中 MedNeXt_MLA_SizeOV4 取得最高 Overall 0.7038 和 Tumor Dice 0.5150。两组外部结果共同提示，肝肿瘤分割研究应从单一内部 Dice 评价转向外部验证、临床安全指标和视觉歧义失败模式分析，同时避免将单一外部数据集上的最优配置过度推广到所有临床场景。

---

## 后续待补

- 补完整内部测试主表，并统一 checkpoint_best / checkpoint_final 口径。
- 补完整外部验证主表，确认所有方法的 rank、drop 和 FP rate。
- 从 report 中整理 size-group analysis，重点补 tiny、small、mid、large 分组。
- 生成 `ircadb_016`、`ircadb_014`、`ircadb_018` 等典型病例图。
- 根据导师意见决定 HCCReferencedCT 是否作为第二外部验证或补充材料。
