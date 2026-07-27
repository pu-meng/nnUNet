# 面向 CVPR 论文目标的导师汇报 PPT 大纲

> 目的：这份材料不是毕业答辩稿，也不是普通论文进度汇报，而是以“写成一篇有 CVPR 目标感的论文”为导向，请老师判断选题高度、核心问题、贡献表述、证据链和后续补强方向。PPT 生成时建议控制在 14 页左右，重点讲清楚：本文不只是做一个肝肿瘤分割网络，而是试图把“内部 Dice 与外部可靠性不一致”作为核心问题，结合 MedNeXt_MLA_MoE、跨数据集验证和视觉失败模式分析，形成一个更有普适意义的医学图像分割可靠性故事。

---

## 1. 标题页

**标题建议**

从内部 Dice 到外部可靠性：面向 CVPR 的肝肿瘤 CT 分割泛化研究

**副标题**

基于 MedNeXt_MLA_MoE、跨数据集验证和视觉失败模式分析的论文选题汇报

**页面目的**

让老师一开始就知道：这次不是请老师帮我把一篇普通论文改顺，而是请老师判断这项工作怎样提升到 CVPR 级论文的选题和叙事高度。

**讲稿要点**

- 目前论文已经形成完整素材：方法、内部测试、3D-IRCADb 外部验证、HCCReferencedCT v2 外部测试、结构消融和视觉失败模式分析。
- 我最想请老师判断的是：这篇工作如果目标定到 CVPR，应该把核心贡献放在“一个新结构 MedNeXt_MLA_MoE”，还是放在“医学分割模型内部指标与外部可靠性不一致这一更大问题”。
- 我希望这篇文章最后不是一篇只报告 Dice 的实验论文，而是能提出一个清晰问题、给出结构性改进、并解释模型失败模式的完整故事。

---

## 2. 这次想请老师按 CVPR 标准帮我看的问题

**核心句**

当前最需要老师帮忙判断的是：这项工作怎样从“结果还不错的医学分割实验”提升为“问题重要、贡献清楚、证据充分、叙事有野心的 CVPR 论文”。

**建议上屏内容**

1. 选题高度是否足够：能否从肝肿瘤任务上升到医学分割模型外部可靠性的普遍问题？
2. 核心贡献如何定义：MedNeXt_MLA_MoE 是主贡献，还是“外部可靠性导向的结构改进 + 评价框架 + 失败模式解释”共同构成贡献？
3. 论文故事是否有 CVPR 感：内部 Dice 排名和外部可靠性排名反转，能否作为全文核心问题来讲？
4. 证据链是否足够强：IRCADb、HCC、多 trainer、结构消融、视觉病例，哪些是主证据，哪些是补充证据？
5. 还缺什么实验或分析：size-group、attention/feature 可视化、多 fold、更多外部数据，哪些最值得优先补？
6. 写法是否要更大胆：题目、摘要和引言是否应该直接强调“external reliability / clinical generalization”，而不是保守写成一个改进网络？

**讲稿要点**

老师，我希望这篇文章的目标定得高一点，不只是完成一篇普通医学图像分割论文。现在素材已经比较多，我想请您帮我判断怎样把它组织成一个更有冲击力的 CVPR 故事。

---

## 3. 研究问题：内部 Dice 高不等于外部可靠

**核心句**

肝肿瘤分割模型在内部测试集上 Dice 高，不代表换到外部临床数据后仍然可靠。

**建议上屏内容**

- 内部测试：Dataset003/LiTS 风格，同源数据。
- 外部验证：3D-IRCADb，含 15 例有肿瘤和 5 例无肿瘤病例。
- 额外外部测试：HCCReferencedCT v2 held-out test，21 例 HCC 有肿瘤病例。
- 临床风险：
  - 小病灶和低对比度病灶漏检。
  - 无肿瘤病例或非肿瘤低密度影误报。
  - 边界过分割影响体积估计和治疗规划。

**讲稿要点**

本文不是追求内部测试集再提高一点 Dice，而是关注模型换到不同外部数据后是否仍然可用，尤其是肿瘤漏检和误报风险。

---

## 4. 数据集和评价口径

**核心句**

论文把内部测试和外部验证分开报告，避免把源域性能和跨数据集性能混在一起。

**建议上屏表格**

| 数据集 | 用途 | 病例构成 | 重点评价 |
|---|---|---|---|
| Dataset003_Liver | 内部测试 | 131 例按 92/13/26 固定划分 | 源域内部表现 |
| 3D-IRCADb | 外部验证 | 20 例，15 例有肿瘤、5 例无肿瘤 | 外部泛化、无肿瘤误报 |
| HCCReferencedCT v2 | 额外 held-out test | 101 例按 70/10/21 固定划分，test 21 例均有肿瘤 | HCC 域偏移、Recall 和严重漏检 |

**评价指标**

```text
Liver Dice
Tumor Dice
Overall = (Liver Dice + Tumor Dice) / 2
Recall / Precision
No-tumor FP rate
Internal-external drop = External Overall - Internal Overall
```

**讲稿要点**

- 26 例 internal test set 只用于最终内部评估，不是训练 validation。
- HCC 的 train/val 病例不参与外部测试结果选择。
- HCC test 均为有肿瘤病例，因此不能评估无肿瘤 FP rate。

---

## 5. 方法位置：MedNeXt_MLA_MoE 不是整体 Transformer 化

**核心句**

MedNeXt_MLA_MoE 保留 MedNeXt-L 强卷积骨干，只在最低分辨率 bottleneck 后加入低秩 Multi-head Latent Attention。

**建议放图**

- `figures/mednext_mla_architecture.svg`
- `figures/mla_bottleneck3d.svg`

**建议上屏内容**

- Backbone：MedNeXt-L，保留 encoder、decoder、deep supervision 和训练配置。
- 插入位置：最低分辨率 bottleneck 后。
- 模块：MLABottleneck3D，使用低秩 key/value 表示。
- 目标：以较小结构扰动补充全局 liver-tumor context。

**讲稿要点**

本文没有用完整 Transformer 替换卷积网络，而是在强卷积 U-Net 的最低空间尺度上加入轻量全局上下文模块，这样计算成本可控，也更容易解释消融结果。

**请老师按高水平论文标准判断**

这一部分是否能支撑一个 CVPR 论文中的方法贡献？如果不够，是否应把方法贡献和外部可靠性问题绑定起来讲，而不是单独夸大 MedNeXt_MLA_MoE。

---

## 6. 关键结果 1：内部最高分和外部最高分不一致

**核心句**

如果只看内部测试，会选择 MedNeXt_SizeOV4 或 MedNeXt；但在 3D-IRCADb 外部验证上，MedNeXt_MLA_MoE 最好。

**建议上屏表格**

| Method | Internal Overall | IRCAD Overall | Drop |
|---|---:|---:|---:|
| MedNeXt_SizeOV4 | 0.8431 | 0.7797 | -0.0634 |
| MedNeXt | 0.8402 | 0.7705 | -0.0697 |
| MedNeXt_MLA_MoE | 0.8259 | 0.8079 | -0.0180 |
| Baseline | 0.7941 | 0.7727 | -0.0214 |

**讲稿要点**

- MedNeXt_MLA_MoE 内部 Overall 不是最高，低于 MedNeXt 和 MedNeXt_SizeOV4。
- 但 MedNeXt_MLA_MoE 在 IRCADb 上 External Overall 最高，且 drop 最小。
- 这页是全文最核心的评价现象：模型选择不能只依赖内部 Dice。

---

## 7. 关键结果 2：IRCADb 外部收益主要来自肿瘤分割和假阳性控制

**核心句**

MedNeXt_MLA_MoE 的提升主要体现在外部 Tumor Dice、Precision 和无肿瘤误报，而不是 Liver Dice。

**建议上屏表格**

| Method | IRCAD Liver | IRCAD Tumor | IRCAD Overall | Precision | No-tumor FP |
|---|---:|---:|---:|---:|---:|
| MedNeXt | 0.9660 | 0.5750 | 0.7705 | 0.6564 | 60% |
| MedNeXt_SizeOV4 | 0.9651 | 0.5943 | 0.7797 | 0.6795 | 60% |
| MedNeXt_MLA_MoE | 0.9673 | 0.6484 | 0.8079 | 0.7437 | 40% |
| Baseline | 0.9673 | 0.5781 | 0.7727 | 0.6814 | 60% |

**讲稿要点**

- 各方法 Liver Dice 都在 0.965 左右，差异不大。
- 真正拉开差距的是肿瘤类别：MedNeXt_MLA_MoE 比 MedNeXt 的 External Tumor Dice 高 0.0734。
- 无肿瘤误报率从 60% 降到 40%，说明全局上下文可能帮助抑制一部分外部假阳性。
- 但 40% 仍然不够临床安全，后面需要结合失败模式分析解释风险。

---

## 8. MedNeXt 系列消融：MLA 不是 SizeOV4 的替代说法

**核心句**

MedNeXt_MLA_MoE 的 IRCADb 外部收益主要来自 bottleneck MLA，而不是 SizeOV4 采样。

**建议上屏表格**

| Method | Internal Overall | IRCAD Overall | Drop | IRCAD Tumor | Precision | No-tumor FP |
|---|---:|---:|---:|---:|---:|---:|
| MedNeXt | 0.8402 | 0.7705 | -0.0697 | 0.5750 | 0.6564 | 60% |
| MedNeXt_SizeOV4 | 0.8431 | 0.7797 | -0.0634 | 0.5943 | 0.6795 | 60% |
| MedNeXt_MLA_MoE | 0.8259 | 0.8079 | -0.0180 | 0.6484 | 0.7437 | 40% |
| MedNeXt_MLA_MoE_SizeOV4 | 0.8285 | 0.7870 | -0.0415 | 0.6091 | 0.7019 | 60% |

**讲稿要点**

- SizeOV4 对 MedNeXt 有小幅提升，但无肿瘤 FP rate 仍是 60%。
- MLA 带来更明显的外部 Overall、Tumor Dice 和 Precision 提升。
- MLA + SizeOV4 在 IRCADb 上不是简单相加，反而低于单独 MedNeXt_MLA_MoE。
- 因此论文应谨慎写：MLA 有助于 3D-IRCADb 外部可靠性，但不是所有数据集和采样策略下都稳定叠加。

**请老师按高水平论文标准判断**

这组消融是否足够支撑“bottleneck latent attention 改善外部可靠性”的说法？如果目标是 CVPR，还需要补哪类分析才能让机制解释更有说服力？

---

## 9. HCCReferencedCT v2：另一个外部域的困难不同

**核心句**

HCC 结果说明外部可靠性不是在一个外部集上固定成立的结论，不同外部域的主要困难可能不同。

**建议上屏表格**

| Method | HCC Overall | HCC Tumor | HCC Recall | 严重失败 | Dice>=0.7 |
|---|---:|---:|---:|---:|---:|
| MedNeXt_MLA_MoE_SizeOV4 | 0.6356 | 0.4269 | 0.3575 | 10 | 8 |
| MedNeXt | 0.6279 | 0.4175 | 0.3582 | 10 | 8 |
| MedNeXt_MLA_MoE | 0.6261 | 0.4080 | 0.3369 | 10 | 7 |
| MedNeXt_SizeOV4 | 0.5775 | 0.3405 | 0.2782 | 11 | 5 |
| Baseline | 0.2511 | 0.0000 | 0.0000 | 21 | 0 |

**讲稿要点**

- HCC 上所有方法 Recall 都明显偏低，最好也只有 0.3575 左右。
- MedNeXt_MLA_MoE_SizeOV4 在 HCC 上最高，但仍有 10 例严重失败。
- 这说明 HCC 的主要问题是 HCC-specific domain shift 下的系统性漏检。
- HCC 不应被讲成“方法全面成功”，更适合作为额外外部挑战和局限证据。

**请老师按高水平论文标准判断**

HCC 结果应该放主结果中，作为第二外部域证明问题更大，还是放到“额外外部验证/讨论”中避免冲淡 MedNeXt_MLA_MoE 在 IRCADb 上的主线？

---

## 10. 多 trainer 结构证据：不是参数量越大越好

**核心句**

结构消融支持的不是“模型越大越好”，而是 DW+IB 强卷积骨干和 bottleneck MLA 的组合更符合本任务。

**建议上屏压缩表**

| Method | 结构类型 | Internal Overall | IRCAD Overall | HCC Overall | 主要结论 |
|---|---|---:|---:|---:|---|
| Baseline | PlainConvUNet | 0.7941 | 0.7727 | 0.2511 | nnU-Net 流程强，但跨 HCC 完全漏检 |
| DeepPlainResGN | plain residual + GN | 0.7966 | 0.7442 | 0.5787 | 参数接近 MedNeXt，但没有对应收益 |
| DeepDWIBResGN | DW+IB residual | 0.8198 | 0.7886 | 0.5924 | DW+IB 比 plain residual 有效 |
| DeepDWIBMedConfig | DW+IB MedNeXt-like | 0.8293 | 0.7867 | 0.6190 | 非官方复刻仍有效 |
| SwinUNETR | Transformer encoder | 0.7846 | 0.7385 | 0.4938 | 整体 Transformer 化未稳定优于 CNN |
| MedNeXt | MedNeXt DW+IB | 0.8402 | 0.7705 | 0.6279 | 内部强，但 IRCAD drop 明显 |
| MedNeXt_MLA_MoE | MedNeXt + bottleneck MLA | 0.8259 | 0.8079 | 0.6261 | 牺牲少量内部，换取 IRCAD 外部最高 |

**讲稿要点**

- DeepPlainResGN 参数接近 MedNeXt，但外部表现没有变好，说明不是单纯参数量问题。
- DW+IB 类模型整体强于 plain residual，对应 MedNeXt 结构本身有价值。
- SwinUNETR 和 nnFormer 没有形成稳定优势，说明整体 Transformer 化不是本任务最稳路线。
- MedNeXt_MLA_MoE 的贡献更像是在强 DW+IB 骨干上补充低分辨率全局上下文。

**请老师按高水平论文标准判断**

这部分是否应该提升为论文的关键证据：说明不是参数量、不是简单 Transformer 化，而是强 DW+IB 骨干 + bottleneck 全局上下文更适合该任务。

---

## 11. 典型病例：MedNeXt_MLA_MoE 有收益，但不是所有病例都解决

**核心句**

病例级分析显示，MedNeXt_MLA_MoE 的收益集中在部分困难病例，但仍存在小病灶漏检和无肿瘤误报。

**建议上屏内容**

- 成功例：`ircadb_016`
  - MedNeXt Tumor Dice：0.2546
  - MedNeXt_SizeOV4：0.5515
  - MedNeXt_MLA_MoE：0.8121
  - MedNeXt_MLA_MoE_SizeOV4：0.7932
- 局限例：
  - `ircadb_014`：无肿瘤误报。
  - `ircadb_015`：小病灶漏检/部分分割。
  - `ircadb_008`：视觉歧义导致小范围误报。

**建议放图**

- `figures/ircadb_014_z45_mednext_mla_fp.png`
- `figures/ircadb_015_z102_mednext_mla_fn.png`
- `figures/ircadb_015_z103_mednext_mla_partial.png`
- `figures/ircadb_008_z96_mednext_mla_unexplained_fp.png`

**讲稿要点**

这页用来避免把结果说得过满：MedNeXt_MLA_MoE 在 IRCADb 上整体最好，但仍有固定难例和临床风险。

---

## 12. CT 视觉歧义：为什么 Dice 背后还有风险

**核心句**

模型错误不只是网络结构问题，也来自 CT 影像证据本身的歧义。

**建议上屏三类失败模式**

| 视觉证据 | 模型行为 | 临床风险 |
|---|---|---|
| 有阴影但该切片无肿瘤 | 容易把低密度结构当成肿瘤 | 假阳性 |
| 无明显阴影但有肿瘤 | 模型难以检出 | 低 Recall / 漏检 |
| 3D 上下文牵连 | 连续切片可帮助分割，也可能提前预测 | 边界外溢 / 邻近切片误报 |

**建议放图**

- `assets/liver_41_z45_full.png`
- `assets/liver_30_z152_full.png`
- `assets/liver_33_z49_full.png`
- `assets/liver_33_z50_full.png`
- `assets/liver_13_z327_full.png`
- `assets/liver_13_z328_full.png`
- `assets/liver_13_z334_full.png`

**讲稿要点**

这部分不是提出新指标，而是解释为什么同一个 Overall 数值背后可能对应不同的临床风险。它也能支撑论文从“内部 Dice”转向“外部验证 + 失败模式解释”的主线。

**请老师判断**

视觉歧义分析是否适合单独作为正文第 6 节，还是压缩后放讨论？

---

## 13. 距离 CVPR 目标还缺什么

**核心句**

目前结果已经能形成论文主线，但如果目标定到 CVPR，还需要补强证据链、机制解释和问题外延，不能只停留在单任务性能比较。

**建议上屏内容**

- 当前主要结果基于固定 fold_0 和固定 internal test set，尚未完成多 fold 或重复划分统计。
- 3D-IRCADb 只有 20 例，HCCReferencedCT v2 held-out test 只有 21 例，外部规模有限。
- HCC test 全部为有肿瘤病例，不能评估无肿瘤假阳性风险。
- MedNeXt_MLA_MoE 的机制解释主要来自指标消融和病例分析，尚无 attention map 或特征可视化直接证明。
- 极小病灶、极淡低密度影和边界不清病灶仍然困难。
- FP-safe 训练、连通域后处理、阈值校准和不确定性筛查尚未纳入正式主结果。
- 目前任务集中在肝肿瘤 CT，若要提高外延，需要考虑是否补充更多数据中心、更多外部域，或把方法和分析抽象成更通用的医学分割可靠性框架。

**讲稿要点**

如果老师认为 CVPR 目标可尝试，我下一步会优先补最能提高论文说服力的实验，而不是盲目堆模型。这里需要老师帮我判断哪些缺口是必须补的，哪些可以作为局限。

---

## 14. 面向 CVPR 的后续补强计划和需要老师定夺的取舍

**核心句**

下一步重点不是继续堆零散实验，而是围绕 CVPR 论文标准补强：问题定义更大、方法贡献更清楚、跨域证据更完整、失败模式解释更可信。

**后续计划**

1. 用 `checkpoint_best.pth` 统一重跑 internal、IRCADb、HCC reports。
2. 按 best-only 结果复核表 1、表 2、表 5、表 6 和表 8。
3. 整理 size-group analysis，优先比较 Baseline、MedNeXt、MedNeXt_MLA_MoE、MedNeXt_MLA_MoE_SizeOV4。
4. 视觉病例保留最能说明问题的图，删除重复或解释成本过高的图。
5. 根据老师意见决定 HCC 和视觉歧义分析在正文中的篇幅。
6. 如果老师认为目标应冲 CVPR，进一步补 mechanism analysis：attention/feature map、case-level failure taxonomy、外部域差异统计或不确定性分析。

**需要老师定夺**

1. 这篇文章是否值得按 CVPR 目标来组织，而不是按普通医学图像论文收束？
2. 题目和摘要是否应直接突出 external reliability / clinical generalization？
3. MedNeXt_MLA_MoE 写成单独方法主贡献，还是写成可靠性问题驱动下的结构改进？
4. HCCReferencedCT v2 放主结果还是补充验证？
5. 结构消融表、多 trainer 表和视觉歧义分析，哪些最能支撑 CVPR 级故事？
6. 如果时间有限，最值得补的一个实验或分析是什么？

**讲稿要点**

老师，我现在最想请您帮我判断的是：如果这篇文章目标定到 CVPR，主线应该怎样拔高，哪些证据必须补，哪些内容应该删掉或放到补充材料。

---

## PPT 总体讲述顺序

**推荐顺序**

问题动机 -> 数据口径 -> 方法位置 -> 内外部排名反转 -> IRCADb 外部收益 -> MedNeXt 系列消融 -> HCC 外部挑战 -> 多 trainer 结构证据 -> 病例和视觉歧义 -> 局限 -> 请老师给取舍意见。

**建议页数**

14 页左右。给老师看论文思路时，不建议超过 15 页。

**面向 CVPR 的主线**

本文不是单纯追求内部 Dice 最高，而是把医学分割模型中“内部高分不等于外部可靠”作为核心问题。MedNeXt_MLA_MoE 在强 MedNeXt 骨干的 bottleneck 处补充低秩全局上下文，使模型在 3D-IRCADb 上获得更好的外部 Tumor Dice、Precision 和无肿瘤误报控制；HCCReferencedCT v2 进一步揭示不同外部域上的系统性漏检；视觉歧义分析则解释了低对比度病灶、非肿瘤低密度影和 3D 上下文牵连如何影响模型行为。面向 CVPR，论文应努力从“一个肝肿瘤模型改进”提升为“外部可靠性导向的医学分割建模、评估与失败解释框架”。

**不建议这样讲**

- 不要从 MedNeXt 结构细节讲太久，方法页只说明插入位置和动机。
- 不要把所有 trainer 的完整表格都放上 PPT。
- 不要把 HCC 讲成全面成功，它更像是揭示另一个外部域的困难。
- 不要把视觉歧义讲成新方法，它是解释模型风险和结果差异的证据。
- 不要把 MedNeXt_MLA_MoE 写成所有外部数据集上的稳定最优方法；更稳的说法是它显著改善 3D-IRCADb 外部可靠性，并在 HCC 上显示出仍需解决的跨域漏检问题。
- 不要把目标说得太小。如果目标是 CVPR，开场就要把问题上升到医学分割外部可靠性、跨域泛化和临床失败模式，而不是只说“我改了一个 trainer”。
