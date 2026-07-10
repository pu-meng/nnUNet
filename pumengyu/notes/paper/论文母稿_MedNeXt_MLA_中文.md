# 论文母稿：MedNeXt_MLA 肝肿瘤分割

> 创建日期：2026-07-08  
> 角色：中文母稿 / 材料总仓 / 思路展开稿。  
> 写法：先尽量完整，不急着精炼；原始路径先保留，定稿前再删除或移到内部记录。  
> 正式稿来源：后续从本文档剪裁、改写到 `正式论文v3_MedNeXt_MLA框架.md`。

---

## 目录

- [论文母稿：MedNeXt\_MLA 肝肿瘤分割](#论文母稿mednext_mla-肝肿瘤分割)
  - [目录](#目录)
  - [0. 这个文件怎么用](#0-这个文件怎么用)
  - [1. 当前一句话故事](#1-当前一句话故事)
  - [2. 当前论文定位](#2-当前论文定位)
    - [2.1 不再这么写](#21-不再这么写)
    - [2.2 现在这么写](#22-现在这么写)
    - [2.3 当前选定题目](#23-当前选定题目)
  - [3. 主要贡献草案](#3-主要贡献草案)
    - [3.1 给导师看的贡献、创新点与发现汇总](#31-给导师看的贡献创新点与发现汇总)
    - [3.2 希望导师重点给建议的问题](#32-希望导师重点给建议的问题)
  - [4. 数据集与任务](#4-数据集与任务)
    - [4.1 内部数据：Dataset003\_Liver / LiTS 风格](#41-内部数据dataset003_liver--lits-风格)
    - [4.2 外部数据：3D-IRCADb](#42-外部数据3d-ircadb)
    - [4.3 HCC-TACE Referenced CT 数据线](#43-hcc-tace-referenced-ct-数据线)
  - [5. 指标口径](#5-指标口径)
  - [6. 主模型结果与排名](#6-主模型结果与排名)
    - [6.1 MedNeXt\_MLA 结果](#61-mednext_mla-结果)
    - [6.2 内部 Dataset003 Test 前几名](#62-内部-dataset003-test-前几名)
    - [6.3 外部 IRCADb 前几名](#63-外部-ircadb-前几名)
    - [6.4 内部-外部排名反转与 Drop 分析](#64-内部-外部排名反转与-drop-分析)
  - [7. MedNeXt\_MLA 方法](#7-mednext_mla-方法)
    - [7.1 MedNeXt-L 骨干](#71-mednext-l-骨干)
    - [7.2 MLABottleneck3D](#72-mlabottleneck3d)
    - [7.3 为什么放在 bottleneck](#73-为什么放在-bottleneck)
    - [7.4 MLA / MoE / SizeOV 的当前定位](#74-mla--moe--sizeov-的当前定位)
  - [8. MedNeXt 系列消融](#8-mednext-系列消融)
    - [8.1 实验指标突然上升的关键节点复盘](#81-实验指标突然上升的关键节点复盘)
      - [8.1.1 Baseline -\> NoMirror：内部 Overall 明显上升，但外部崩溃](#811-baseline---nomirror内部-overall-明显上升但外部崩溃)
      - [8.1.2 DeepPlainResGN -\> DeepDWIBResGN：DW+IB 组合与参数预算重分配](#812-deepplainresgn---deepdwibresgndwib-组合与参数预算重分配)
      - [8.1.3 DWIB -\> MedNeXt / MedNeXt\_SizeOV4：内部同域性能达到最高](#813-dwib---mednext--mednext_sizeov4内部同域性能达到最高)
      - [8.1.4 MedNeXt -\> MedNeXt\_MLA：外部验证指标明显上升](#814-mednext---mednext_mla外部验证指标明显上升)
      - [8.1.5 SizeOV4：内部上分有效，但不是外部收益主因](#815-sizeov4内部上分有效但不是外部收益主因)
      - [8.1.6 MedNeXt\_MLA\_FPSafe：内部 FP 改善，但外部退化](#816-mednext_mla_fpsafe内部-fp-改善但外部退化)
      - [8.1.7 总结](#817-总结)
  - [9. CT 视觉歧义与失败模式分析](#9-ct-视觉歧义与失败模式分析)
    - [9.1 工作定义](#91-工作定义)
    - [9.2 三类错误来源](#92-三类错误来源)
    - [9.3 Case/Slice 混合粒度视觉歧义框架](#93-caseslice-混合粒度视觉歧义框架)
    - [9.4 反常 A：有阴影但无肿瘤](#94-反常-a有阴影但无肿瘤)
    - [9.5 反常 B：无阴影但有肿瘤](#95-反常-b无阴影但有肿瘤)
    - [9.6 3D 上下文的帮助与限制](#96-3d-上下文的帮助与限制)
    - [9.7 为什么这部分重要](#97-为什么这部分重要)
  - [10. NoMirror 和数据增强的启示](#10-nomirror-和数据增强的启示)
  - [11. FPSafe 与 MSDHCCMix 负结果](#11-fpsafe-与-msdhccmix-负结果)
    - [11.1 MedNeXt\_MLA\_FPSafe](#111-mednext_mla_fpsafe)
    - [11.2 MedNeXt\_MLA\_MSDHCCMix](#112-mednext_mla_msdhccmix)
  - [12. 主表计划](#12-主表计划)
    - [12.1 表 1：内部测试集](#121-表-1内部测试集)
    - [12.2 表 2：外部 IRCADb](#122-表-2外部-ircadb)
    - [12.3 表 3：MedNeXt 系列消融](#123-表-3mednext-系列消融)
    - [12.4 表 4：失败模式 / case analysis](#124-表-4失败模式--case-analysis)
  - [13. 图计划](#13-图计划)
    - [13.1 方法图](#131-方法图)
    - [13.2 排名图](#132-排名图)
    - [13.3 Drop 条形图](#133-drop-条形图)
    - [13.4 失败案例图](#134-失败案例图)
  - [14. 当前还没想清楚的问题](#14-当前还没想清楚的问题)
    - [14.1 题目是否要继续用“跨域泛化”](#141-题目是否要继续用跨域泛化)
    - [14.2 MLA/MoE 旧创新是否还要](#142-mlamoe-旧创新是否还要)
    - [14.3 HCC 是否进入正式主表](#143-hcc-是否进入正式主表)
    - [14.3.1 HCC 固定划分怎么处理](#1431-hcc-固定划分怎么处理)
    - [14.4 统计检验怎么做](#144-统计检验怎么做)
    - [14.5 结果是否需要重新核对](#145-结果是否需要重新核对)
  - [15. 可以直接搬进正式论文的段落草稿](#15-可以直接搬进正式论文的段落草稿)
    - [15.1 引言段落草稿](#151-引言段落草稿)
    - [15.2 方法动机段落草稿](#152-方法动机段落草稿)
    - [15.3 结果段落草稿](#153-结果段落草稿)
    - [15.4 CT 视觉歧义段落草稿](#154-ct-视觉歧义段落草稿)
    - [15.5 负结果段落草稿](#155-负结果段落草稿)
  - [16. 原始材料索引](#16-原始材料索引)
  - [17. 下一步任务](#17-下一步任务)

---

## 0. 这个文件怎么用

这个文件不是最终论文，也不是单纯的实验记录。它是把论文相关的所有材料先集中起来的“母稿”：

1. 可以写正式段落。
2. 可以写困惑和判断。
3. 可以保留原始路径，方便核对。
4. 可以写负结果和暂时不进主表的实验。
5. 可以把旧稿中有价值的内容搬进来，但要按当前主线重组。

当前流程：

```text
论文写作状态.md -> 决策板，记录主线、结论、下一步
论文母稿_MedNeXt_MLA_中文.md -> 材料总仓，把所有可用内容先写全
正式论文v3_MedNeXt_MLA框架.md -> 后续从母稿剪裁出的正式论文
```

当前不要追求精简，先追求“不丢材料、不丢思考、不丢证据”。

---

## 1. 当前一句话故事

内部测试集 Dice 最高的模型，不一定是外部临床数据上最可靠的模型。MedNeXt 和 MedNeXt_SizeOV4 在 Dataset003/LiTS 风格内部测试集上取得最高 Overall，但在 IRCADb 外部验证中出现明显性能下降；相反，MedNeXt_MLA 内部并非第一，却在 IRCADb 上取得外部 Overall 第一，并显著减小 internal-external drop。本文进一步从 CT 视觉歧义角度分析模型错误来源：可见低密度阴影可能导致假阳性，等密度/无明显阴影肿瘤可能导致漏检，3D patch 推理还可能把邻近切片的肿瘤上下文延续到当前无肿瘤切片，引起边界过分割或邻近切片误判。

当前主模型：

```text
nnUNetTrainer_MedNeXt_MLA
```

当前主张：

```text
MedNeXt_MLA 的价值不是内部刷榜，而是外部泛化稳定性和临床可靠性分析。
```

---

## 2. 当前论文定位

### 2.1 不再这么写

不再以“MLAUNet + MoE + SizeOversample 刷内部 Dice”为主线。

原因：

1. MedNeXt / MedNeXt_SizeOV4 是更强的 published baseline，内部 Overall 分别达到 0.8402 / 0.8431。
2. MLAUNet / MoE 系列有价值，但内部并没有稳定超过 MedNeXt。
3. 如果只说“我提出新架构，内部指标更高”，证据不稳。
4. MoE 容易被审稿人认为是复杂组件堆叠，且不是最终外部冠军的必要组件。

### 2.2 现在这么写

当前论文主线：

```text
从内部高 Dice 到外部可靠性：基于 MedNeXt_MLA 的肝肿瘤分割跨数据集泛化与 CT 视觉歧义分析
```

核心问题：

1. 内部测试集最高分不等于外部泛化最优。
2. 肝肿瘤 CT 分割不能只看平均 Dice，还要关注无肿瘤误报和小/隐匿肿瘤漏检。
3. 单期 CT 中存在视觉歧义：看起来像肿瘤的区域不一定是肿瘤，真正的肿瘤有时又没有明显阴影。
4. 3D 分割不是逐 2D 切片独立判断，相邻切片的信息会影响当前切片，可能造成边界溢出和邻近切片假阳性。

### 2.3 当前选定题目

当前选定题目：

```text
面向外部可靠性的肝肿瘤 CT 分割：基于 MedNeXt_MLA 的跨数据集验证与视觉歧义分析
```

这个题目比“面向跨域泛化的 MedNeXt 瓶颈潜在注意力肝肿瘤分割”更稳，原因是它把论文价值放在外部可靠性、跨数据集验证和失败模式解释上，而不是只强调模型模块本身。

需要注意：题目中有“跨数据集验证”，因此正文证据不能只像普通消融那样写一个内部 test。IRCADb 是第一外部验证集，可以支撑“外部验证”和“内部-外部排名反转”；但如果只依赖 IRCADb，一个外部集的证据厚度偏薄。HCCReferencedCT 最好作为第二外部/补充临床验证线，至少用于 Discussion 或 Supplementary。

当前写法原则：

1. 主表仍以 Dataset003 内部测试 + IRCADb 外部验证为核心。
2. HCC 不急着强行并入主表，先完成 HCCRefOnly fold 0 sanity。
3. 如果 HCC 结果合理，可新增“第二外部数据集观察”或放 Supplementary/Discussion。
4. 如果要把 HCC 写成严格第二外部验证，不能只报 HCCRefOnly 内部 val，需要做 Dataset003/LiTS 训练模型直接推理 HCCReferencedCT，或建立 HCC held-out test。

---

## 3. 主要贡献草案

当前贡献先写大，后续正式稿再压缩：

1. **问题视角贡献**  
   本文不只报告内部测试集 Dice，而是系统比较内部测试与外部验证的排名差异，指出肝肿瘤分割中“同域最高分”不等于“外部最可靠”。

2. **方法贡献**  
   在 MedNeXt-L 的最低分辨率 bottleneck 后插入低秩 Multi-head Latent Attention，构建 MedNeXt_MLA。该设计不改变 MedNeXt 局部卷积骨干、不引入额外标注，只在语义最抽象、空间分辨率最低的位置加入全局 liver-tumor context。

3. **实验证据贡献**  
   在 Dataset003/LiTS 风格内部测试集和 3D-IRCADb 外部验证集上系统比较 nnU-Net、MedNeXt、SwinUNETR、nnFormer、MLAUNet、MoE、SizeOV 等方法。结果显示 MedNeXt_SizeOV4 内部 Overall 最高 0.8431，但外部下降到 0.7797；MedNeXt_MLA 内部 Overall 0.8259，外部 Overall 0.8079，成为外部第一，drop 仅 -0.0180。

4. **失败模式分析贡献**  
   本文提出 Case/Slice 混合粒度视觉歧义框架，从阴影可见性、case-level 肿瘤状态、slice-level 视觉证据和 3D 上下文帮助与限制四个角度解释假阳性、漏检和边界误差来源。该框架说明模型错误不仅来自网络结构，还来自单期 CT 视觉证据不足和 3D patch 推理机制。

5. **临床安全贡献**  
   本文将无肿瘤误报率、Recall、Precision、FDR、FPV/FNV 体积误差和典型 case 可视化作为辅助指标，避免高 Overall Dice 掩盖临床安全风险。

### 3.1 给导师看的贡献、创新点与发现汇总

这一节用于和导师沟通，不是正式论文正文。目标是让老师快速判断：这篇论文的贡献是否清楚、创新是否够、主线是否稳、还需要补什么实验。

#### 3.1.1 一句话总结

```text
本文不是单纯提出一个分割网络刷内部 Dice，而是围绕“内部高分模型是否真的外部可靠”这一问题，
提出 MedNeXt_MLA，并结合跨数据集验证、无肿瘤误报、小肿瘤漏检和 CT 视觉歧义分析，
说明肝肿瘤 CT 分割需要从内部平均 Dice 转向外部可靠性和临床安全错误分析。
```

#### 3.1.2 我的主要贡献

1. **重新定义论文问题：从内部刷榜转向外部可靠性。**  
   论文主线不是“我做了一个新模块所以内部 Dice 更高”，而是证明内部测试集表现最好的模型不一定是外部临床数据上最可靠的模型。MedNeXt 和 MedNeXt_SizeOV4 在内部 Dataset003 上更高，但外部 IRCADb drop 更大；MedNeXt_MLA 内部不是最高，却在外部验证中最稳。

2. **提出 MedNeXt_MLA：在强 MedNeXt 骨干上加入 bottleneck latent attention。**  
   方法不是重写整个 U-Net，而是在 MedNeXt-L 最低分辨率 bottleneck 后插入低秩 Multi-head Latent Attention。这样保留 MedNeXt 的局部卷积强表征，同时在语义最抽象、计算成本最低的位置补充全局 liver-tumor context。

3. **完成内部测试 + 外部验证的系统比较。**  
   对比包括 nnU-Net Baseline、SizeOV、MLAUNet、MoE、SwinUNETR、nnFormer、MedNeXt、MedNeXt_SizeOV4、MedNeXt_MLA 等。核心结果是：MedNeXt_SizeOV4 内部 Overall 0.8431，但外部 0.7797；MedNeXt_MLA 内部 0.8259，但外部 0.8079，外部第一，drop 只有 -0.0180。

4. **提出 CT 视觉歧义与失败模式分析。**  
   论文不只报平均 Dice，而是解释模型为什么错：有低密度阴影但不是肿瘤会导致假阳性；真实肿瘤在单期 CT 上可能接近等密度、无明显阴影，导致漏检；3D patch 上下文既能帮助连续切片判断，也可能把邻近切片的 tumor response 延续到当前无肿瘤切片。

5. **把临床安全指标引入结果解释。**  
   除 Liver/Tumor Dice 和 Overall 外，加入 no-tumor FP rate、Recall、Precision、FDR、FPV/FNV 体积误差和 case-level 可视化，避免平均 Dice 掩盖无肿瘤误报和真实肿瘤漏检。

6. **保留负结果，形成更可信的消融逻辑。**  
   SizeOV4 能提高内部表现，但不能解释 MedNeXt_MLA 的外部收益；FPSafe 能改善内部 FP，但外部退化；NoMirror 内部上升但外部崩溃。这些结果共同说明：外部可靠性不是简单采样、关闭增强或 FP loss 就能解决。

#### 3.1.3 创新点怎么表述更稳

可以向老师这样表述：

```text
创新点 1：问题视角创新
从“内部测试集 Dice 最优”转为“外部可靠性 + 临床安全错误”的评价视角。

创新点 2：方法设计创新
在 MedNeXt-L bottleneck 处加入低秩 MLA，用全局上下文补充局部卷积骨干，而不改变 nnU-Net/MedNeXt 主训练范式。

创新点 3：证据链创新
不仅比较内部测试结果，还比较外部 IRCADb、internal-external drop、无肿瘤 FP 率和 case-level 失败模式。

创新点 4：解释框架创新
提出 CT 视觉歧义分析框架，把假阳性、漏检和 3D 上下文误差与单期 CT 的视觉证据不足联系起来。
```

注意：不要把创新点说成“我内部 Dice 最高”。当前数据不支持这个说法。更稳的说法是：

```text
MedNeXt_MLA 的价值在于外部验证稳定性和错误模式解释，而不是内部测试集刷榜。
```

#### 3.1.4 目前最重要的发现

1. **内部第一和外部第一不一致。**  
   MedNeXt_SizeOV4 内部 Overall 最高 0.8431，但外部 0.7797；MedNeXt_MLA 内部 0.8259，但外部 0.8079。

2. **MedNeXt_MLA 的收益主要体现在外部稳定性。**  
   MedNeXt 外部 drop 为 -0.0697，MedNeXt_SizeOV4 为 -0.0634，MedNeXt_MLA 只有 -0.0180。

3. **SizeOV4 不是外部收益主因。**  
   MedNeXt_MLA_SizeOV4 外部 0.7870，低于 MedNeXt_MLA 0.8079，说明采样不能解释 MLA 的外部优势。

4. **显式 FP 抑制不是跨域安全的充分条件。**  
   MedNeXt_MLA_FPSafe 内部 Overall 0.8326，高于 MedNeXt_MLA 的 0.8259，但外部降到 0.7744，说明源域 FP 控制可能不具备跨域稳定性。

5. **CT 视觉歧义是论文深度来源。**  
   题目可以保留“跨数据集泛化”，但正文必须加入 CT 视觉歧义和临床安全错误分析。论文深度不只来自外部验证数字，而来自对失败模式的解释。

6. **HCC 线暂时不能混入主结论。**  
   HCCReferencedCT 可以作为后续临床扩展线，但 HCCRefOnly 是 HCC 内部 train/val sanity，不等于跨数据集外部验证；旧 MSDHCCMix 已确认剔除。

#### 3.1.5 当前最稳的论文主张

```text
本文提出的 MedNeXt_MLA 并不是在内部测试集上追求最高 Dice 的模型，
而是一个面向外部可靠性的 MedNeXt 变体。通过在 bottleneck 引入低秩 latent attention，
模型在保持强局部卷积骨干的同时获得全局上下文建模能力。实验显示，
MedNeXt_MLA 在 Dataset003 内部测试中不是最高分，但在 IRCADb 外部验证中取得最高 Overall，
并显著减小 internal-external drop。进一步的 case-level 分析表明，
肝肿瘤 CT 分割错误与单期 CT 视觉歧义、无肿瘤误报、小/隐匿肿瘤漏检以及 3D 上下文传播有关。
因此，肝肿瘤分割评价应从内部平均 Dice 扩展到外部验证、临床安全指标和失败模式解释。
```

### 3.2 希望导师重点给建议的问题

问导师时可以直接带这几个问题：

1. **题目是否合适？**  
   当前题目是“面向外部可靠性的肝肿瘤 CT 分割：基于 MedNeXt_MLA 的跨数据集验证与视觉歧义分析”。请老师判断“外部可靠性 / 跨数据集验证 / 视觉歧义分析”这个组合是否适合作为论文标题主线。

2. **创新点是否够集中？**  
   当前把创新分成方法创新、外部验证证据、视觉歧义解释和临床安全分析。请老师判断是否太散，还是应该压缩成 2-3 个核心贡献。

3. **CT 视觉歧义分析能否作为论文亮点？**  
   目前我认为论文深度不只来自外部验证数字，而来自解释为什么模型会 FP/FN。请老师判断这部分是否适合放 Results/Discussion，是否需要更多病例图。

4. **HCCReferencedCT 是否要进入正文？**  
   当前建议 HCC 暂不进主表，只作为后续临床扩展线。请老师判断是否需要补一个 HCC held-out 或直接外推实验来增强“跨数据集”证据。

5. **负结果是否保留？**  
   FPSafe、NoMirror、SizeOV4 的负/中性结果能支撑“内部提升不等于外部可靠”。请老师判断正式论文里应该放多少，哪些放 Supplementary。

6. **统计检验和图表需要补到什么程度？**  
   当前已有主表、drop 分析和 case-level 分析计划。请老师判断是否必须补 case-level 统计检验、bootstrap CI、FLOPs/参数量或更多外部 case。

---

## 4. 数据集与任务

### 4.1 内部数据：Dataset003_Liver / LiTS 风格

当前内部测试集口径：

```text
Dataset003_Liver
fold_0
test cases: 26
有肿瘤: 23
无肿瘤: 3
```

内部 report 来源：

```text
/home/PuMengYu/nnUNet_workspace/results_v2/Dataset003_Liver/*/fold_0/test_report_custom.txt
```

当前主模型内部 report：

```text
/home/PuMengYu/nnUNet_workspace/results_v2/Dataset003_Liver/nnUNetTrainer_MedNeXt_MLA__nnUNetPlans__3d_fullres/fold_0/test_report_custom.txt
```

### 4.2 外部数据：3D-IRCADb

当前外部验证口径：

```text
IRCADb external validation
n = 20
有肿瘤: 15
无肿瘤: 5
```

外部 report 来源：

```text
/home/PuMengYu/nnUNet_workspace/results_v2/ExternalVal_IRCADb/*/report_custom.txt
```

当前主模型外部 report：

```text
/home/PuMengYu/nnUNet_workspace/results_v2/ExternalVal_IRCADb/MedNeXt_MLA/report_custom.txt
```

IRCADb 转换脚本：

```text
pumengyu/ext_val/prepare_ircadb.py
```

注意：旧稿中曾写“IRCADb 全部有肿瘤”，这是旧信息，现在不对。当前报告中 IRCADb 有 5 个无肿瘤 case。

### 4.3 HCC-TACE Referenced CT 数据线

HCC 数据线目前不作为主表核心，而是外部临床数据扩展线。

当前状态：

```text
旧数据集已删除:
/home/PuMengYu/nnUNet_workspace/raw/Dataset013_HCCMultiPhase

新数据集:
/home/PuMengYu/nnUNet_workspace/raw/Dataset013_HCCReferencedCT
/home/PuMengYu/nnUNet_workspace/preprocessed/Dataset013_HCCReferencedCT
```

新标准：

```text
只使用 DICOM-SEG 明确引用的 CT series。
不使用 PRE CT、follow-up CT、未被 SEG 引用的其他 CT。
```

标签：

```text
0 = background
1 = liver
2 = tumor
```

segment 映射：

```text
Liver -> 1
Mass -> 2
Necrosis -> 2
Portal vein -> 0
Abdominal aorta -> 0
```

病例：

```text
原始 HCC: 105 cases
排除:
  HCC_103: 没有 SEG
  HCC_048: SEG 引用的 CT 找不到
  HCC_089: 转换后 label 全 0
  HCC_099: liver 体素异常少，QC 异常

最终训练数据: 101 cases
QC: 99 ok, 2 review
review: HCC_065, HCC_075
```

预处理：

```text
nnUNetv2_plan_and_preprocess -d 13 --verify_dataset_integrity
已通过。

3d_fullres:
spacing = [2.5, 0.78125, 0.78125]
patch = [40, 224, 224]
batch = 2
normalization = CTNormalization
```

sanity trainer：

```text
nnUNetTrainer_MedNeXt_MLA_HCCRefOnly
```

命令：

```bash
nnUNetv2_train 13 3d_fullres 0 -tr nnUNetTrainer_MedNeXt_MLA_HCCRefOnly
```

HCC 线写作定位：

1. 作为外部临床数据扩展，不替代当前主线。
2. 若 fold 0 训练稳定，可作为 Supplementary / Discussion 证据。
3. 若结果差，也可支持 HCC-TACE 与 LiTS/IRCADb 分布差异明显、治疗相关外观增加泛化困难。

---

## 5. 指标口径

核心指标：

```text
Liver Dice
Tumor Dice
Overall = (Liver Dice + Tumor Dice) / 2
Recall
Precision
FDR = 1 - Precision
FNR = 1 - Recall
No-tumor FP rate
Overall internal-external drop = External Overall - Internal Overall
FPV / FNV volume error
```

排序默认使用：

```text
Overall = (Liver Dice + Tumor Dice) / 2
```

无肿瘤 case 的 tumor dice 口径：

```text
  无肿瘤正确 TN: tumor Dice = NaN，仅从 Tumor Dice 均值中排除；该病例仍计入 Liver Dice。
  无肿瘤误报 FP: tumor Dice = 0，计入 Tumor Dice 均值；该病例也计入 Liver Dice。
```

这个口径和当前 report 中的 nnUNet foreground_mean 一致。

---

## 6. 主模型结果与排名

### 6.1 MedNeXt_MLA 结果

主模型：

```text
nnUNetTrainer_MedNeXt_MLA
```

| 指标 | 内部 Dataset003 Test | 外部 IRCADb |
|---|---:|---:|
| Liver Dice | 0.9535 | 0.9673 |
| Tumor Dice | 0.6982 | 0.6484 |
| Overall | 0.8259 | 0.8079 |
| Recall | 0.7323 | 0.6665 |
| Precision | 0.7921 | 0.7437 |
| 无肿瘤 FP 率 | 66.67% (2/3) | 40.00% (2/5) |
| Overall 外-内 Drop | - | -0.0180 |

排名：

```text
内部 Dataset003 Test: Overall 第 6
外部 IRCADb: Overall 第 1
```

写作时必须强调：

```text
MedNeXt_MLA 不是内部最优模型。
它的优势是外部验证第一和 internal-external drop 小。
```

### 6.2 内部 Dataset003 Test 前几名

| Rank | Method | Overall | Liver | Tumor | Recall | Precision | FP率 | 定位 |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 1 | MedNeXt_SizeOV4 | 0.8431 | 0.9545 | 0.7317 | 0.7361 | 0.8187 | 33% (1/3) | 内部最强 |
| 2 | MedNeXt | 0.8402 | 0.9521 | 0.7283 | 0.7334 | 0.8128 | 33% (1/3) | 强 published baseline |
| 3 | MLAUNet_MoE_SizeOV4 | 0.8330 | 0.9514 | 0.7146 | 0.7140 | 0.7776 | 33% (1/3) | 旧自研主线 |
| 4 | MedNeXt_MLA_FPSafe | 0.8326 | 0.9510 | 0.7143 | 0.7171 | 0.7741 | 33% (1/3) | FP loss 负/中性消融 |
| 5 | MedNeXt_MLA_SizeOV4 | 0.8285 | 0.9529 | 0.7040 | 0.7459 | 0.7775 | 67% (2/3) | MedNeXt_MLA + sampling 消融 |
| 6 | MedNeXt_MLA | 0.8259 | 0.9535 | 0.6982 | 0.7323 | 0.7921 | 67% (2/3) | 当前主模型，内部非最优 |

内部结论：

1. MedNeXt 系列同域能力最强。
2. MedNeXt_SizeOV4 内部 Overall 最高。
3. MedNeXt_MLA 内部不是第一，因此本文不能写“内部最强”。
4. 内部结果只能说明 MedNeXt_MLA 保持了较强性能，不是它的主要卖点。

### 6.3 外部 IRCADb 前几名

| Rank | Method | Overall | Liver | Tumor | Recall | Precision | FP率 | 内部 Overall | Overall 外-内 Drop | 定位 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | MedNeXt_MLA | 0.8079 | 0.9673 | 0.6484 | 0.6665 | 0.7437 | 40% (2/5) | 0.8259 | -0.0180 | 外部 Overall 第一，当前主模型 |
| 2 | MoE_SizeOV5 | 0.8025 | 0.9679 | 0.6371 | 0.6437 | 0.7464 | 40% (2/5) | 0.8167 | -0.0142 | 外部稳定自研对照 |
| 3 | MLAUNet | 0.8008 | 0.9675 | 0.6341 | 0.6320 | 0.7580 | 40% (2/5) | 0.8148 | -0.0140 | 外部稳定自研对照 |
| 4 | SizeOV2 | 0.7992 | 0.9676 | 0.6307 | 0.6352 | 0.7547 | 40% (2/5) | 0.8187 | -0.0195 | 采样策略对照 |
| 5 | MLA_GK5_V4 | 0.7957 | 0.9656 | 0.6258 | 0.6472 | 0.7291 | 40% (2/5) | 0.8173 | -0.0216 | MLA 变体对照 |
| 13 | MedNeXt_SizeOV4 | 0.7797 | 0.9651 | 0.5943 | 0.6500 | 0.6795 | 60% (3/5) | 0.8431 | -0.0634 | 内部最高但外部 drop 大 |
| 16 | MedNeXt | 0.7705 | 0.9660 | 0.5750 | 0.6554 | 0.6564 | 60% (3/5) | 0.8402 | -0.0697 | 内部第二但外部 drop 大 |

外部结论：

1. MedNeXt_MLA 是当前外部 Overall 第一。
2. 外部 Top 5 里多个方法来自 MLA/MoE/SizeOV 探索族，说明全局上下文和相关策略对外部泛化有价值。
3. MedNeXt 和 MedNeXt_SizeOV4 内部强，但外部 drop 大，形成内部-外部排名反转。
4. 这是本文最重要的证据：高同域 Dice 不等于外部临床可靠。

### 6.4 内部-外部排名反转与 Drop 分析

这一节是主结果的关键写法。MedNeXt_MLA 不应被包装成“内部 Dice 最高”的模型，而应被定位为：

```text
内部同域性能保持较强，但真正优势体现在外部 IRCADb 上的排名、Tumor Dice、Precision 和稳定性。
```

Drop 口径：

```text
Overall Drop = External Overall - Internal Overall
Tumor Drop   = External Tumor Dice - Internal Tumor Dice
Liver Drop   = External Liver Dice - Internal Liver Dice
```

因此 Drop 越接近 0，说明从内部 Dataset003 到外部 IRCADb 的性能损失越小。这里的 Drop 不是单独肝脏，也不是单独肿瘤；主文表格中的 `Overall Drop` 指的是：

```text
Overall_external - Overall_internal
```

内部-外部排名反转表：

| Method | Internal Overall | Internal Rank | External Overall | External Rank | Rank Change | Overall Drop | Tumor Drop | Liver Drop | 解释 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| MedNeXt_SizeOV4 | 0.8431 | 1 | 0.7797 | 13 | -12 | -0.0634 | -0.1374 | +0.0106 | 内部最高，但外部明显下降 |
| MedNeXt | 0.8402 | 2 | 0.7705 | 16 | -14 | -0.0697 | -0.1533 | +0.0139 | 强 published baseline，但跨数据集损失最大 |
| MedNeXt_MLA | 0.8259 | 6 | 0.8079 | 1 | +5 | -0.0180 | -0.0498 | +0.0138 | 内部非最优，但外部第一，drop 明显更小 |

主结果解释：

```text
MedNeXt_SizeOV4 和 MedNeXt 在内部 Dataset003 上分别排名第 1 和第 2，
但在 IRCADb 外部验证中下降到第 13 和第 16。
相反，MedNeXt_MLA 内部 Overall 仅排名第 6，
但外部 Overall 排名第 1，并且 Overall Drop 只有 -0.0180。
```

这个现象说明：

```text
内部测试集排名不能直接预测外部验证排名。
同域 Dice 最高的模型不一定是外部临床数据上最可靠的模型。
```

更具体地看，MedNeXt_MLA 的 liver Dice 在外部并没有下降，反而从 0.9535 到 0.9673，小幅上升；真正发生跨数据集损失的是 tumor Dice。相比 MedNeXt 和 MedNeXt_SizeOV4，MedNeXt_MLA 的 tumor Dice drop 更小：

```text
MedNeXt tumor drop:          0.5750 - 0.7283 = -0.1533
MedNeXt_SizeOV4 tumor drop:  0.5943 - 0.7317 = -0.1374
MedNeXt_MLA tumor drop:      0.6484 - 0.6982 = -0.0498
```

因此，bottleneck latent attention 的结果不能简单表述为“提高内部 Dice”。更准确的论文表述是：

```text
Bottleneck latent attention does not maximize source-domain Dice,
but it substantially reduces the tumor segmentation degradation under external validation.
```

中文正式稿可写为：

```text
瓶颈 latent attention 并未带来最高的源域内部 Dice，
但显著缓解了外部验证中肿瘤 Dice 的退化。
这表明其主要价值在于提升跨数据集泛化稳定性，
而不是单纯提高内部测试集平均分。
```

---

## 7. MedNeXt_MLA 方法

### 7.1 MedNeXt-L 骨干

使用配置：

```text
n_channels = 32
kernel_size = 3
exp_r = [3,4,8,8,8,8,8,4,3]
block_counts = [3,4,8,8,8,8,8,4,3]
do_res = True
do_res_up_down = True
norm_type = group
checkpoint_style = outside_block
```

说明：

1. 本文使用的是 MedNeXt-L kernel=3 配置。
2. Large 指网络规模，不是卷积核大小。
3. 官方 nnUNet MedNeXt 集成版提供 k=3 和 k=5，没有 3D k=7 trainer。
4. MedNeXt 的强性能主要来自更深的 9 阶段编解码器、倒置瓶颈、DW-Conv、GroupNorm、残差连接和更大的 expand ratio。

MedNeXt block：

```text
DW-Conv3d(k=3, groups=C)
GroupNorm
PW-Conv3d expand
GELU
PW-Conv3d compress
Residual
```

### 7.2 MLABottleneck3D

MedNeXt_MLA 的核心改动：

```text
Input CT
  -> MedNeXt stem
  -> MedNeXt encoder blocks
  -> MedNeXt bottleneck
  -> MLABottleneck3D
  -> MedNeXt decoder blocks
  -> segmentation logits
```

设计原则：

1. 不改变 MedNeXt 局部骨干。
2. 不改变 loss。
3. 不依赖额外标注。
4. 只在最低分辨率 bottleneck 加入全局上下文。

MLA 输入：

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

### 7.3 为什么放在 bottleneck

放在 bottleneck 的理由：

1. 空间分辨率最低，计算和显存可控。
2. 语义最抽象，适合建模肝脏-肿瘤全局关系。
3. 避免在高分辨率层引入大量 attention 开销。
4. 与 MedNeXt 的局部卷积归纳偏置互补。
5. 通过低秩 KV 压缩减小注意力的显存负担。

### 7.4 MLA / MoE / SizeOV 的当前定位

MLA：

```text
保留为核心结构思想，但最终落点从 MLAUNet 转到 MedNeXt_MLA。
```

MoE：

```text
不再作为主创新。
可以作为探索性组件和外部稳定对照。
```

SizeOV：

```text
不写成主创新。
作为采样策略对照，尤其用于证明采样不能解释 MedNeXt_MLA 的外部收益。
```

FPSafe：

```text
负/中性消融。
它改善内部 FP，但外部退化，说明手工 FP 抑制没有带来稳健跨域泛化。
```

---

## 8. MedNeXt 系列消融

| Method | 内部 Overall | 外部 Overall | 外-内 | 内部 Tumor | 外部 Tumor | 外部 Recall | 外部 Precision | 外部 FP率 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MedNeXt | 0.8402 | 0.7705 | -0.0697 | 0.7283 | 0.5750 | 0.6554 | 0.6564 | 60% |
| MedNeXt_SizeOV4 | 0.8431 | 0.7797 | -0.0634 | 0.7317 | 0.5943 | 0.6500 | 0.6795 | 60% |
| MedNeXt_MLA | 0.8259 | 0.8079 | -0.0180 | 0.6982 | 0.6484 | 0.6665 | 0.7437 | 40% |
| MedNeXt_MLA_SizeOV4 | 0.8285 | 0.7870 | -0.0415 | 0.7040 | 0.6091 | 0.6580 | 0.7019 | 60% |
| MedNeXt_MLA_FPSafe | 0.8326 | 0.7744 | -0.0582 | 0.7143 | 0.5852 | 0.6442 | 0.6711 | 60% |

消融结论：

1. MedNeXt 和 MedNeXt_SizeOV4 内部最强，但外部 drop 最大。
2. MedNeXt_MLA 内部下降，但外部上升，是跨域最稳的 MedNeXt 变体。
3. SizeOV4 不能解释外部收益。MedNeXt_MLA_SizeOV4 外部 0.7870，低于 MedNeXt_MLA 0.8079。
4. FPSafe 不能作为最终方法。它内部 0.8326，但外部 0.7744，FP 率仍为 60%。
5. 因此，当前证据支持“bottleneck latent attention 主要改善外部泛化”，而不是“采样或 FP loss 单独带来收益”。

### 8.1 实验指标突然上升的关键节点复盘

这一节用于回忆和解释实验过程中“指标突然上升”的几个节点。需要注意：不同改动提升的是不同目标，有些提升只发生在内部 Dataset003，同步外部验证后并不可靠。因此写论文时不能简单把所有“涨分”都当成有效创新。

#### 8.1.1 Baseline -> NoMirror：内部 Overall 明显上升，但外部崩溃

改动：

```text
关闭镜像增强，mirror_axes = None。
```

最初动机：

```text
肝脏是右侧不对称器官，左右镜像可能生成“肝脏在左侧”的假图像，
因此尝试关闭所有轴的镜像增强。
```

内部 Dataset003 结果：

```text
Baseline Overall: 0.7941
NoMirror Overall: 0.8133
提升: +0.0192
```

主要提升来自：

```text
Liver Dice: 0.9340 -> 0.9581
Precision: 0.6451 -> 0.7685
FDR:       0.3549 -> 0.1915
```

同时代价很明显：

```text
Recall: 0.7853 -> 0.6769
极小肿瘤 Recall: 0.6687 -> 0.4442
liver_127 从部分召回变成完全漏检
```

外部 IRCADb 结果：

```text
NoMirror Overall: 0.5512
SizeOV3_NoMirror Overall: 0.5444
```

结论：

```text
NoMirror 是一次内部指标明显上升的实验，但不是可靠路线。
它说明关闭镜像可能让模型在同域上更保守、Precision 更高，
但也显著损伤 Recall，并在外部验证中崩溃。
```

论文定位：

```text
Discussion / Supplementary。
用于说明内部测试集改进不等于外部可靠，也说明镜像增强虽然解剖上看似不合理，
但对跨机构泛化可能很重要。
```

#### 8.1.2 DeepPlainResGN -> DeepDWIBResGN：DW+IB 组合与参数预算重分配

改动：

```text
从更深的 plain residual + GroupNorm 结构，改为 Depthwise Convolution + Inverted Bottleneck 组合块。
```

结果：

```text
DeepPlainResGN 内部 Overall: 0.7966
DeepDWIBResGN 内部 Overall: 0.8198
内部提升: +0.0232

DeepPlainResGN 外部 Overall: 0.7442
DeepDWIBResGN 外部 Overall: 0.7886
外部提升: +0.0444
```

结论：

```text
当前结果只能证明 DW+IB 组合块有效，不能单独证明提升来自 DW 或来自 IB。
这组实验的合理解释是：DW 降低了 3D spatial conv 的参数和计算成本，
使模型能在相近参数预算下堆叠更多 block；IB 的 expand-compress 结构则增强通道特征变换。
因此提升更像是“低成本空间建模 + 更深 block 堆叠 + 通道扩展压缩”共同作用。
```

关键限制：

```text
这组主线不是大 kernel 带来的收益。当前 MedNeXt / DeepDWIBResGN / DeepDWIBMedConfig
基本都使用 k=3。因此不能写成“大卷积核提升了指标”。
更稳的表述是：在相同 k=3 条件下，DW+IB 使模型以相近参数预算获得更深、更强的局部表征。
```

代码核对：

```text
pumengyu/architectures/mednext.py:
MedNeXt-L 配置为 kernel_size=3，
block_counts=[3,4,8,8,8,8,8,4,3]，
exp_r=[3,4,8,8,8,8,8,4,3]。

pumengyu/architectures/deep_plain_res_gn.py:
DeepPlainResGN 使用普通 3x3 Conv3d residual block，
features_per_stage=[32,64,128,256,384]，
encoder blocks=[3,4,4,4,4]，decoder blocks=[4,4,4,3]，
参数量注释约 61.1M，用于对齐 MedNeXt-L 约 61.8M。

DeepDWIBResGN 同样使用 k=3，
但 encoder block 改为 DWIBBlock，
features_per_stage=[32,64,128,256,512]，
encoder blocks=[3,4,8,8,8]，expansion_ratio=8，
参数量注释约 55.6M/58M。

DeepDWIBMedConfig 进一步对齐 MedNeXt-L 的通道、block_counts 和 exp_r，
仍然是 k=3，使用 DWIBBlock 和 add skip fusion。
```

这说明 DeepPlainResGN 到 DeepDWIBResGN 的变化并不是单一变量：

```text
plain conv residual block -> DW + inverted bottleneck residual block
384 bottleneck channels -> 512 bottleneck channels
encoder [3,4,4,4,4] -> encoder [3,4,8,8,8]
LeakyReLU/plain residual 风格 -> GELU + expand-compress 风格
```

因此当前实验不能把提升拆成“DW 带来多少、IB 带来多少”。可以写成：

```text
DW 降低了 3D 空间卷积的参数和计算成本，使更深 block 堆叠和更高 bottleneck 通道数
在相近参数预算下成为可能；IB 通过 expand-compress 提供额外通道混合能力。
现有结果支持 DW+IB 组合及其参数预算重分配是有效的，但不单独归因于 DW 或 IB。
```

论文定位：

```text
作为 MedNeXt-style 结构有效性的机制背景。
但不要把该结果过度解释为“DW 单独有效”或“IB 单独有效”。
如果后续要严格拆解，需要补 DW-only、IB-without-DW、DW+IB 三组消融。
```

#### 8.1.3 DWIB -> MedNeXt / MedNeXt_SizeOV4：内部同域性能达到最高

改动：

```text
进一步走 MedNeXt-L 风格配置，包括更深的 9 阶段 encoder-decoder、
倒置瓶颈、DW-Conv、GroupNorm、残差连接和更大的 expand ratio。
```

内部结果：

```text
DeepDWIBResGN:     0.8198
DeepDWIBMedConfig: 0.8293
MedNeXt:           0.8402
MedNeXt_SizeOV4:   0.8431
```

这是一条明显的内部指标上升路线：

```text
DWIB 有效 -> MedNeXt-style 配置进一步增强同域拟合 -> MedNeXt_SizeOV4 成为内部 Overall 第一。
```

但外部 IRCADb 并没有同步变好：

```text
MedNeXt 外部 Overall:         0.7705
MedNeXt_SizeOV4 外部 Overall: 0.7797
```

结论：

```text
MedNeXt / MedNeXt_SizeOV4 是内部同域能力最强的路线，
但不是外部泛化最稳的路线。
这正是本文“内部最高 Dice 不等于外部临床可靠”的核心证据之一。
```

#### 8.1.4 MedNeXt -> MedNeXt_MLA：外部验证指标明显上升

改动：

```text
在 MedNeXt bottleneck 后加入低秩 Multi-head Latent Attention。
```

内部结果：

```text
MedNeXt 内部 Overall:     0.8402
MedNeXt_MLA 内部 Overall: 0.8259
```

内部并没有上升，反而下降。因此不能把 MedNeXt_MLA 写成内部最强。

外部 IRCADb 结果：

```text
MedNeXt 外部 Overall:         0.7705
MedNeXt_SizeOV4 外部 Overall: 0.7797
MedNeXt_MLA 外部 Overall:     0.8079
```

相对 MedNeXt 的外部提升：

```text
Overall:   0.7705 -> 0.8079  (+0.0374)
Tumor Dice: 0.5750 -> 0.6484
Precision: 0.6564 -> 0.7437
FP rate:   60% -> 40%
Overall Drop: -0.0697 -> -0.0180
```

结论：

```text
这是本文最重要的“跃升点”。
MedNeXt_MLA 的价值不是内部刷榜，而是外部 IRCADb 上明显提升肿瘤分割和 Precision，
同时显著减小 internal-external drop。
```

论文定位：

```text
主结果 / 主创新。
写作时应强调 bottleneck latent attention 改善的是外部泛化稳定性，
不是单纯提升源域内部 Dice。
```

#### 8.1.5 SizeOV4：内部上分有效，但不是外部收益主因

改动：

```text
均匀重复/过采样训练 case。
```

在 MedNeXt 上：

```text
MedNeXt 内部 Overall:         0.8402
MedNeXt_SizeOV4 内部 Overall: 0.8431
```

内部小幅上升，并成为内部第一。

外部：

```text
MedNeXt 外部 Overall:         0.7705
MedNeXt_SizeOV4 外部 Overall: 0.7797
```

外部只小幅恢复，仍低于 MedNeXt_MLA：

```text
MedNeXt_MLA 外部 Overall: 0.8079
```

在 MedNeXt_MLA 上叠加 SizeOV4：

```text
MedNeXt_MLA 外部 Overall:         0.8079
MedNeXt_MLA_SizeOV4 外部 Overall: 0.7870
```

结论：

```text
SizeOV4 是内部上分技巧或采样对照，不是外部泛化主因。
它不能解释 MedNeXt_MLA 的外部收益。
```

论文定位：

```text
消融对照。
用于证明最终收益不是简单来自采样，而是来自 bottleneck MLA。
```

#### 8.1.6 MedNeXt_MLA_FPSafe：内部 FP 改善，但外部退化

改动：

```text
在 MedNeXt_MLA 上加入 FP-safe / false-positive 控制。
```

内部结果：

```text
MedNeXt_MLA 内部 Overall:        0.8259
MedNeXt_MLA_FPSafe 内部 Overall: 0.8326
内部 FP 率: 67% -> 33%
```

这个实验在内部看起来是有效的：Overall 上升，且无肿瘤误报下降。

外部结果：

```text
MedNeXt_MLA 外部 Overall:        0.8079
MedNeXt_MLA_FPSafe 外部 Overall: 0.7744
外部 FP 率仍为 60%
```

结论：

```text
FPSafe 是典型的“同域指标改善、外部泛化变差”的负结果。
它可能学习到了 Dataset003 的无肿瘤背景模式，但不能保证外部无肿瘤安全。
```

更具体地说，FPSafe 的问题不是“FP 指标本身没意义”，而是源域 FP 抑制容易变成源域背景记忆：

```text
FPSafe 在 Dataset003 内部测试集上有效，说明它确实让模型对源域中的无肿瘤背景、
血管、囊肿、低密度伪影或肝脏边缘区域更加保守。
但这种保守性不一定是跨机构通用的“非肿瘤”判别能力。
```

外部 IRCADb 中图像分布变化后，FPSafe 可能带来两类负效应：

```text
1. 过强抑制真实肿瘤响应：
   外部肿瘤的灰度、边界、对比度和源域不同，
   FPSafe 学到的“不要轻易报 tumor”会把一部分真实肿瘤也压下去，
   导致 Tumor Dice / Recall / Overall 下降。

2. 源域无肿瘤背景规则不能覆盖外部背景：
   外部无肿瘤病例中的血管、囊肿、低密度结构或扫描噪声分布不同，
   因此外部 FP 率仍然可能较高，不能保证无肿瘤安全。
```

因此这组实验应该写成：

```text
FPSafe improves source-domain false-positive behavior,
but the learned suppression is not domain-invariant.
It reduces internal no-tumor false alarms while degrading external tumor segmentation.
```

中文正式稿可写为：

```text
MedNeXt_MLA_FPSafe 在内部测试集上降低了无肿瘤病例误报并提高 Overall，
但在外部 IRCADb 上 Overall 明显下降，且无肿瘤误报并未同步改善。
这说明基于源域负样本学习到的 FP 抑制策略可能具有明显数据集依赖性：
它可以记住 Dataset003 中常见的无肿瘤背景模式，却不一定形成跨机构通用的非肿瘤判别规则。
当外部数据的灰度分布、肝脏纹理和病灶外观发生变化时，
该抑制项可能同时压低真实肿瘤响应，从而损伤 Tumor Dice 和 Recall。
```

论文定位：

```text
负结果 / 中性消融。
说明显式 FP 抑制并不等价于跨域可靠性。
```

#### 8.1.7 总结

可以把实验路线概括成：

```text
内部突然涨：
  NoMirror
  DW+IB 组合 / 参数预算重分配
  MedNeXt / MedNeXt_SizeOV4
  SizeOV4

外部突然涨：
  MedNeXt + bottleneck MLA

看起来涨但不能作为主线：
  NoMirror
  FPSafe
  SizeOV4
```

最终写法：

```text
DW+IB / MedNeXt-style block 在相近参数预算下提升了同域分割能力；
SizeOV4 进一步提高内部测试表现；
但真正改善外部泛化的是 MedNeXt bottleneck 后的 MLA。
NoMirror 和 FPSafe 说明单纯抑制误报会牺牲召回或跨域稳定性。
```

---

## 9. CT 视觉歧义与失败模式分析

### 9.1 工作定义

本文的 CT 视觉歧义不是泛泛说“CT 看不清”，而是指：

```text
在 CT 图像上，看起来像肿瘤的区域不一定是真正肿瘤；
真正的肿瘤有时又没有明显低密度阴影或异常纹理。
```

进一步，3D 分割还存在：

```text
当前 2D 切片本身没有明显阴影、也没有肿瘤，
但如果相邻切片存在明显阴影/肿瘤区域，
模型可能把上下文延续到该切片，
导致边界过分割或邻近切片假阳性。
```

### 9.2 三类错误来源

| 错误来源 | 表现 | 主要风险 |
|---|---|---|
| 可见阴影相似性 | 低密度阴影、术后切除灶、坏死/良性结构被预测为肿瘤 | 假阳性 |
| 隐匿/等密度肿瘤 | 肿瘤区域与正常肝实质接近，单切片无明显异常 | 漏检 |
| 3D 上下文帮助与限制 | 当前切片视觉证据弱时，相邻切片和局部体积上下文可能提供帮助，但不能保证恢复弱证据切片 | 弱证据切片漏检、边界误差或部分难解释 FP |

### 9.3 Case/Slice 混合粒度视觉歧义框架

这是本文用于观看 case、筛选失败案例和解释实验结果的分析框架。它不是模型性能评价表，也不是医学影像诊断标准，而是本文定义的一种 case analysis 方法：把“是否有肿瘤”和“阴影是否可见”拆开，并明确它们在不同情形下应该按 case-level 还是 slice-level 来描述。

核心定义：

```text
case-level tumor status:
  整个病例是否有肿瘤标注。
  例如 ircadb_014 是 case-level 无肿瘤病例；
  ircadb_015 是 case-level 有肿瘤病例。

slice-level shadow visibility:
  某一张 2D 切片上是否存在肿瘤样低密度阴影、极淡阴影或近等密度表现。
  同一个有肿瘤病例内，不同切片的阴影可见性可以不同。
```

本文采用以下观看方式：

```text
对于“有阴影但无肿瘤”：
  肿瘤有无主要按 case-level 定义。
  也就是整例 GT 无肿瘤，但某些 slice 出现肿瘤样低密度阴影。
  这类 case 用于分析无肿瘤病例假阳性。

对于“无阴影/弱阴影但有肿瘤”：
  肿瘤有无按 case-level 定义为有肿瘤；
  阴影可见性主要按 slice-level 描述。
  也就是同一个有肿瘤病例中，部分 slice 的肿瘤近等密度或阴影极淡，
  模型可能在这些切片上漏检。
```

因此，后文 case analysis 必须避免两种混淆：

```text
1. 不能把“某张切片无明显阴影”写成“整个病例无阴影”。
2. 不能把“某张切片无肿瘤标注”写成“整个病例无肿瘤”。
```

推荐分析口径：

| 分析类型 | 肿瘤变量 | 阴影变量 | 推荐粒度 | 主要误差倾向 | 论文用途 |
|---|---|---|---|---|---|
| 有阴影但无肿瘤 | case-level 无肿瘤 | slice-level 有肿瘤样阴影 | case 主导，slice 举证 | 假阳性 | 临床安全风险；说明低密度阴影不等于肿瘤 |
| 无阴影/弱阴影但有肿瘤 | case-level 有肿瘤 | slice-level 无明显阴影、极淡阴影或近等密度 | slice 主导 | 假阴性、局部漏检 | CT 单期模态上限；弱视觉证据肿瘤分析 |
| 有阴影且有肿瘤 | case-level 有肿瘤 | slice-level 有较明显肿瘤影像证据 | slice 主导 | 相对更容易被识别，但仍可能有边界误差 | 成功/部分成功对照；说明模型依赖可见影像证据 |
| 无阴影且无肿瘤 | case-level 无肿瘤 | slice-level 无肿瘤样阴影 | 对照 | 通常作为 TN 对照 | 区分普通阴性样本和真正困难样本 |
| 3D 上下文帮助与限制 | case-level 可有或无肿瘤 | slice-level 相邻切片证据不一致 | 连续 slice 分析 | 弱证据切片 FN、边界误差或部分难解释 FP | 说明 3D patch 推理可能帮助连续性，但不是稳定的弱证据补偿机制 |

这套框架是本文失败模式分析的贡献之一：它把平均 Dice 后面的错误拆成影像证据、标注状态和 3D 连续性三个层面，使外部验证中的 FP/FN 不再只是一个数字，而能对应到可解释的影像条件。

框架边界：

```text
该框架不是把模型简化成“有阴影就预测肿瘤、无阴影就不预测肿瘤”的规则。
模型预测还会受到局部纹理、解剖位置、血管/肝门邻近结构、3D patch 上下文和训练分布影响。
因此，阴影可见性是解释 FP/FN 的重要线索，但不是唯一原因。
更稳的经验性判断是：
  影像证据与标注一致时，模型通常更容易表现稳定；
  影像证据与标注冲突、不充分，或存在 3D 上下文干扰时，模型更容易出错。
```

### 9.4 反常 A：有阴影但无肿瘤

特征：

```text
肝脏内存在明显低密度阴影，但 GT 标注为无肿瘤。
```

风险：

```text
模型可能将阴影识别为肿瘤，产生假阳性，降低 Precision。
```

正文主代表 case：

```text
case: ircadb_014
dataset: External IRCADb
model: MedNeXt_MLA
slice: z=45
GT tumor: 0 voxels
TP: 0
FP: 673
FN: 0
```

![ircadb_014 z45 MedNeXt_MLA no-tumor false positive](figures/ircadb_014_z45_mednext_mla_fp.png)

图像观察：

```text
该切片 GT 无肿瘤标注，但模型在肝脏边缘附近预测出紧凑的红色 FP 区域。
原始 CT 与病灶放大图中可见一个较规则、接近圆形的低密度阴影结构。
该结构在视觉上具有肿瘤样外观，因此容易触发模型的 tumor prediction。
```

可写入正式论文的解释：

```text
在 ircadb_014 的 z=45 切片中，GT 无肿瘤标注，但 MedNeXt_MLA 在肝脏边缘附近预测出
紧凑的假阳性肿瘤区域。该区域在原始 CT 上呈规则圆形低密度阴影，视觉上具有肿瘤样外观。
这一 case 说明，单期 CT 中“低密度阴影”并不必然等于肿瘤，但模型容易将其作为肿瘤候选，
从而在无肿瘤病例中产生假阳性。
```

补充内部 case：

```text
case: liver_41
dataset: Dataset003_Liver internal test
model: MedNeXt_MLA
GT tumor: 0 voxels
pred_tumor: 30,959 voxels
```

旧分析：

```text
liver_41 肝脏内存在大面积弥漫性低密度阴影，GT 无肿瘤。
模型仍产生大体积假阳性。
该现象说明低密度阴影与真实肿瘤在 HU 范围上有重叠，
模型难以区分“低密度但正常”和“低密度且为肿瘤”。
```

定位：

```text
正文优先使用 ircadb_014，因为它来自外部 IRCADb，更贴合外部可靠性主线。
liver_41 可作为 Supplementary 或内部一致性补充，说明类似视觉歧义在内部测试集中也存在。
```

注意：不同模型下 liver_41 的误报体素数略有不同，写论文时要明确对应哪个模型。旧 MLAUNet_MoE_SizeOV4 分析中，liver_41 pred_tumor 约 31,368；当前 MedNeXt_MLA 内部 report 中为 30,959。

边界反例：无明显阴影但仍出现 FP

```text
case: ircadb_008
dataset: External IRCADb
model: MedNeXt_MLA
slice: z=96
GT tumor: 0 voxels
TP: 0
FP: 848
FN: 0
```

![ircadb_008 z96 MedNeXt_MLA no-obvious-shadow false positive](figures/ircadb_008_z96_mednext_mla_unexplained_fp.png)

图像观察：

```text
该切片 GT 无肿瘤标注，且当前 slice 上没有 ircadb_014 那种典型、规则、肿瘤样低密度阴影。
但模型仍在肝门/邻近血管结构附近预测出较大的红色 FP 区域。
检查相邻切片后，该误报也不宜简单解释为相邻肿瘤预测的直接延续。
```

这说明：

```text
无肿瘤 FP 不只来自“明显低密度阴影被误认为肿瘤”。
模型还可能受到局部纹理、解剖位置、血管/肝门邻近结构、训练分布偏差或隐式 3D 上下文影响。
因此，本文的视觉歧义框架应被理解为 case analysis 的解释工具，而不是一个完备的因果规则。
```

### 9.5 反常 B：无阴影但有肿瘤

特征：

```text
肿瘤 HU 接近周围肝实质，单张 CT 切片上没有明显低密度阴影。
```

风险：

```text
模型很容易漏检，即使训练集中出现过，也可能学不会。
```

旧稿中的关键观察：

```text
反常B 的训练集 case 仍然大量失败。
这说明当肿瘤与肝脏等密度、无明显阴影时，
纯粹依赖单期 CT 图像强度和纹理的分割网络难以突破该瓶颈。
```

代表 case：

```text
liver_112
liver_63
liver_127
```

特别是 `liver_127`：

```text
极小肿瘤，298 voxels，仅跨 3 个切片；
近等密度；
多种模型完全或接近完全漏检。
```

写作注意：

```text
不能写成“模型无能”，应写成“单期 CT 视觉证据不足导致的任务上限”。
如果怀疑标注不确定，也只能谨慎写成 annotation uncertainty。
```

外部 IRCADb 连续切片代表 case：

```text
case: ircadb_015
dataset: External IRCADb
model: MedNeXt_MLA
analysis level: slice-level within a case-level tumor-positive patient
```

该 case 的价值在于展示同一病灶在连续切片中的视觉证据变化：部分切片肿瘤近等密度或阴影极淡，模型完全漏检；相邻切片中病灶边界和低密度表现逐渐清楚后，模型开始预测出肿瘤主体。

![ircadb_015 z102 MedNeXt_MLA false negative](figures/ircadb_015_z102_mednext_mla_fn.png)

```text
z=102: GT=113 voxels, TP=0, FP=0, FN=113。
该切片病灶接近等密度/极淡阴影，MedNeXt_MLA 完全漏检。
```

![ircadb_015 z103 MedNeXt_MLA partial detection](figures/ircadb_015_z103_mednext_mla_partial.png)

```text
z=103: GT=193 voxels, TP=100, FP=0, FN=93。
病灶轮廓和低密度表现略增强，模型开始识别一部分肿瘤。
```

![ircadb_015 z104 MedNeXt_MLA partial detection](figures/ircadb_015_z104_mednext_mla_partial.png)

```text
z=104: GT=218 voxels, TP=183, FP=4, FN=35。
视觉证据进一步增强，模型基本覆盖肿瘤主体，但仍有少量边界误差。
```

![ircadb_015 z106 MedNeXt_MLA partial detection](figures/ircadb_015_z106_mednext_mla_partial.png)

```text
z=106: GT=402 voxels, TP=262, FP=0, FN=140。
病灶在连续切片中更明显，模型预测出主要区域，但仍存在部分漏检。
```

可写入正式论文的解释：

```text
ircadb_015 展示了同一病例内肿瘤视觉证据随切片变化的过程。z=102 上病灶接近等密度，
缺乏明显低密度阴影，模型完全漏检；而在 z=103-z106 中，随着病灶边界和低密度表现逐渐清楚，
模型开始预测出肿瘤主体。该现象说明，3D 上下文虽然可能帮助模型利用邻近切片信息，
但不能完全弥补当前切片视觉证据不足的问题。换言之，有肿瘤病例内部也存在 slice-level
“无明显阴影但有肿瘤”的困难切片，不能简单用 case-level 标签概括。
```

### 9.6 3D 上下文的帮助与限制

这一节不再把错误简单归因于“3D 上下文牵连”。当前证据更适合支持一个更谨慎的结论：

```text
3D patch 分割不是逐张 2D 切片独立判断。
相邻切片和局部体积上下文可能帮助模型维持跨切片连续性，
但这种帮助并不稳定，也不能完全弥补当前切片视觉证据不足。
```

当前可用证据主要来自 `ircadb_015` 的连续切片：

```text
z=102: GT=113, TP=0,   FP=0, FN=113  -> 弱视觉证据切片完全漏检
z=103: GT=193, TP=100, FP=0, FN=93   -> 相邻切片开始部分识别
z=104: GT=218, TP=183, FP=4, FN=35   -> 视觉证据增强后基本识别主体
```

![ircadb_015 z102 MedNeXt_MLA false negative](figures/ircadb_015_z102_mednext_mla_fn.png)

![ircadb_015 z103 MedNeXt_MLA partial detection](figures/ircadb_015_z103_mednext_mla_partial.png)

![ircadb_015 z104 MedNeXt_MLA partial detection](figures/ircadb_015_z104_mednext_mla_partial.png)

图注草稿：

```text
Figure X. Consecutive slices from ircadb_015 showing the limitation of 3D contextual compensation.
On z=102, the tumor is near-isodense and completely missed. On adjacent slices z=103 and z=104,
the lesion becomes more visible and is partially segmented. This case shows that 3D context may
support continuity, but it does not guarantee recovery of weak-evidence tumor slices.
```

中文图注草稿：

```text
图 X. ircadb_015 连续切片展示 3D 上下文补偿的局限性。
z=102 上病灶接近等密度，模型完全漏检；相邻 z=103 和 z=104 中病灶逐渐更可见，
模型开始部分分割。该 case 说明 3D 上下文可以提供连续性信息，
但不能保证恢复弱视觉证据切片。
```

这个 case 的解释不是“上下文错误传播”，而是：

```text
同一病灶在连续切片中的视觉证据逐渐增强。
当 z=102 当前切片接近等密度或阴影极淡时，即使相邻 z=103/z=104 已经可部分识别，
模型仍未能把邻近切片信息稳定传播回 z=102，导致完全漏检。
```

因此，3D 上下文在本文中的定位应写成：

```text
3D 上下文可能提供跨切片连续性，但不是可靠的弱证据补偿机制。
当当前切片缺乏足够的局部视觉证据时，相邻切片信息仍可能不足以避免 FN。
```

同时，`ircadb_008 z=96` 说明部分 FP 也不能被简单解释为“当前切片有明显阴影”或“相邻切片直接延续”。这类错误可能来自局部纹理、解剖位置、血管/肝门邻近结构、训练分布偏差或隐式 3D 体积上下文共同作用。

论文图需求：

```text
正文优先使用 ircadb_015 的连续切片图，作为 3D 上下文帮助有限的例子。
如果后续找到更强证据，例如：
  z 上 GT 有肿瘤；
  z+1 上 GT=0；
  但 z+1 仍预测出 tumor FP；
则可作为“3D 上下文错误传播/边界外延续”的补充图。
在没有这种强证据前，不把所有邻近切片 FP 都写成上下文牵连导致。
```

可写入正式论文的解释：

```text
Because the model operates on 3D patches, predictions on a given axial slice are influenced
by neighboring slices and volumetric context. However, the ircadb_015 case shows that such
context does not guarantee recovery of weak-evidence tumor slices. Although adjacent slices
are partially segmented, the near-isodense slice remains completely missed, suggesting that
3D context cannot fully compensate for insufficient local visual evidence.
```

中文正式稿可写为：

```text
由于模型以 3D patch 为输入，单张轴位切片的预测会受到邻近切片和局部体积上下文影响。
然而，ircadb_015 的连续切片结果显示，这种上下文并不能保证恢复弱视觉证据切片：
尽管相邻切片已经能够被部分分割，近等密度切片仍然完全漏检。
这说明 3D 上下文可以提供连续性信息，但不能完全弥补当前切片局部影像证据不足。
```

### 9.7 为什么这部分重要

这部分能让论文不只是“换一个模块，指标略好”，而是说明：

1. 哪些错误是模型可改进的。
2. 哪些错误来自单期 CT 视觉证据不足。
3. 哪些错误来自 3D patch 推理机制。
4. 为什么外部验证和失败案例分析比单一内部 Dice 更重要。

---

## 10. NoMirror 和数据增强的启示

NoMirror 内部结果曾经看起来不错：

```text
Baseline Overall 0.7941
NoMirror Overall 0.8133
```

但外部 IRCADb：

```text
NoMirror Overall 0.5512
SizeOV3_NoMirror Overall 0.5444
```

结论：

1. 关闭镜像增强在内部可能提高 Precision，减少部分误报。
2. 但外部验证显著崩溃，说明模型可能记住了 LiTS 的方向和坐标分布。
3. 镜像增强虽然在解剖上看似不合理，但对跨机构泛化有重要作用。
4. 这进一步支持本文主线：内部改进不等于外部可靠。

写作位置：

```text
Discussion 或 Supplementary。
```

---

## 11. FPSafe 与 MSDHCCMix 负结果

### 11.1 MedNeXt_MLA_FPSafe

代码真实逻辑：

```text
trainer:
  nnUNetTrainer_MedNeXt_MLA_FPSafe

继承关系:
  TopKNoTumorFPPenaltyMixin + nnUNetTrainer_MedNeXt_MLA

代码位置:
  pumengyu/trainers/trainer.py
  pumengyu/mixins.py
```

`FPSafe` 不是后处理，不是采样策略，也不是 case-level 无肿瘤控制。它是在训练 loss 上额外加入一个 patch-level 的 Top-K tumor probability 惩罚：

```text
TKN_TUMOR_FP_LAMBDA = 1.0
TKN_TOPK_PERCENT = 0.01
```

真实 loss 逻辑：

```text
base = base_loss(net_output, target)

probs = softmax(logits)
p_tumor = probs[:, tumor_cls]

has_tumor = 当前 patch 的 target 中是否存在 tumor label
no_tumor_idx = ~has_tumor

如果 batch 中没有 no-tumor patch:
  loss = base

如果 batch 中存在 no-tumor patch:
  no_tumor_probs = p_tumor[no_tumor_idx] 的所有体素
  topk_vals = no_tumor_probs 中最高的 1% tumor probability
  penalty = mean(topk_vals)
  loss = base + 1.0 * penalty
```

因此，`FPSafe` 的真实含义是：

```text
在当前 patch 没有 tumor 标注时，压低模型最自信的前 1% tumor probability。
```

关键限制：

```text
1. 它判断的是 patch-level no tumor，不是 case-level no tumor。
   一个 no-tumor patch 可能来自整例无肿瘤病例，
   也可能来自有肿瘤病例中的非肿瘤区域或肿瘤邻近区域。

2. 它惩罚的是整个 patch 的 tumor probability top 1%，不是 liver-mask 内 top-k。
   因此肝外背景、肝门附近结构、血管邻近区域等都可能参与该惩罚。

3. 它没有 margin。
   只要是 no-tumor patch 中的 top-k tumor probability，都会继续被往低压。
```

结果：

```text
内部 Overall: 0.8326
外部 Overall: 0.7744
外部 FP 率: 60%
```

判断：

```text
FPSafe 内部 FP 控制有用，但没有提升跨域泛化。
它不是最终主线。
```

机制解释：

```text
FPSafe 会让模型在 no-tumor patch 上整体更保守。
在 Dataset003 内部测试集上，这种保守性可能降低无肿瘤病例误报，
因此内部 Overall 和无肿瘤 FP 指标看起来改善。

但外部 IRCADb 中存在弱阴影、近等密度和局部视觉证据不足的肿瘤切片。
这些真实肿瘤本来就需要模型保留较敏感的 tumor response。
patch-level Top-K FP 惩罚可能进一步压低这类响应，
导致 Recall、Tumor Dice 和外部 Overall 下降。
```

写法：

```text
作为 negative / neutral ablation。
说明显式 FP loss 可能学习到同域无肿瘤背景，但不能保证外部无肿瘤安全。
```

可写入正式论文的表述：

```text
MedNeXt_MLA_FPSafe 在 MedNeXt_MLA 的基础上加入 patch-level no-tumor Top-K tumor probability
penalty。该 loss 对当前 patch 中不含肿瘤标注的样本，惩罚预测 tumor 概率最高的 1% 体素。
虽然这一策略在内部测试集上降低了无肿瘤误报并提高 Overall，但外部 IRCADb 上 Overall 明显下降。
这说明基于源域 no-tumor patch 的显式 FP 抑制可能学习到同域背景模式，并使模型过度保守；
它不能直接等价于跨机构外部数据上的临床安全性提升。
```

### 11.2 MedNeXt_MLA_MSDHCCMix

结论更新：

```text
MedNeXt_MLA_MSDHCCMix 不纳入本文结果和 Discussion。
```

原因：

1. 它不是当前 Dataset003/LiTS 主训练口径下的一个同口径 trainer。
2. 代码真实逻辑是旧 `Dataset013_HCCMultiPhase` 作为主数据集，再在运行时追加 `Dataset003_Liver`，不是“Dataset003 主训练 + HCC 外部数据混入”。
3. 旧 `Dataset013_HCCMultiPhase` 已删除，当前 HCC 数据线已改为单通道 `Dataset013_HCCReferencedCT`。
4. 当前论文主线比较的是 Dataset003 内部测试与 IRCADb 外部验证上的 trainer 差异；把这个旧 HCCMultiPhase 主域模型放进来会混淆变量。
5. 即使它在 IRCADb 上有 report，也只能说明一个旧实验产物的表现，不能作为 MedNeXt_MLA、FPSafe、SizeOV 等当前 trainer 的消融证据。

处理方式：

```text
正式论文:
  删除，不写。

母稿:
  只保留这段“剔除原因”，防止后续误把该结果混进主表或负结果。

代码:
  暂不删除 trainer，作为历史结果复现入口保留；
  但不再作为当前论文实验使用。
```

---

## 12. 主表计划

这一节不是正式论文正文，而是正式稿成表前的规划。目标是控制主表数量和口径，避免把探索性实验、旧实验产物和不同训练域的模型混在同一张表里比较。

总原则：

1. 主表只服务当前论文主线：Dataset003/LiTS 内部测试、IRCADb 外部验证、MedNeXt_MLA 外部可靠性。
2. 同一张表内的方法必须训练/验证口径一致。
3. `MedNeXt_MLA_MSDHCCMix` 不进入任何主表或补充负结果，因为它是旧 `Dataset013_HCCMultiPhase` 主域 mix，不是当前 Dataset003 trainer 口径。
4. HCCReferencedCT 只作为后续临床扩展线，除非完成固定 held-out test 和清晰对照，否则不并入当前主表。

### 12.1 表 1：内部测试集

正式表名建议：

```text
Table 1. Internal test performance on Dataset003_Liver.
```

用途：

```text
说明各类模型在同域 Dataset003/LiTS 风格测试集上的表现。
重点不是证明 MedNeXt_MLA 内部最高，而是展示 MedNeXt / MedNeXt_SizeOV4 是更强的同域 baseline。
```

建议列：

```text
Method
Liver Dice
Tumor Dice
Overall
Recall
Precision
No-tumor FP rate
Report path / note
```

建议纳入：

```text
Baseline
SizeOV2
MLAUNet
MoE_SizeOV5
SwinUNETR
nnFormer
MedNeXt
MedNeXt_SizeOV4
MedNeXt_MLA
MedNeXt_MLA_SizeOV4
MedNeXt_MLA_FPSafe
```

可选纳入：

```text
DeepPlainResGN
DeepDWIBResGN
DeepDWIBMedConfig
MLA_GK5_V4
```

不纳入：

```text
MedNeXt_MLA_MSDHCCMix
HCCRefOnly / HCCRefOnly701020
旧 HCCMultiPhase 相关 trainer
```

### 12.2 表 2：外部 IRCADb

正式表名建议：

```text
Table 2. External validation on 3D-IRCADb.
```

用途：

```text
支撑核心论点：内部测试集最高分不等于外部验证最可靠。
```

建议列：

```text
Method
External Liver Dice
External Tumor Dice
External Overall
External Recall
External Precision
No-tumor FP rate
Internal Overall
External - Internal drop
```

必须突出：

```text
MedNeXt_MLA 外部 Overall 第 1。
MedNeXt_MLA drop 小。
MedNeXt / MedNeXt_SizeOV4 内部强，但外部 drop 大。
MoE_SizeOV5 / MLAUNet 外部稳定，可作为支持“不是只有内部排名重要”的对照。
```

不纳入：

```text
MedNeXt_MLA_MSDHCCMix
原因：训练主域不是 Dataset003，不能与当前 Dataset003 trainer 公平比较。
```

### 12.3 表 3：MedNeXt 系列消融

正式表名建议：

```text
Table 3. Ablation study of MedNeXt variants.
```

用途：

```text
只回答 MedNeXt 系列内部的问题：
1. 外部收益是否来自 MLA。
2. SizeOV4 能否解释外部收益。
3. FPSafe 是否能进一步提升外部可靠性。
```

保留：

```text
MedNeXt
MedNeXt_SizeOV4
MedNeXt_MLA
MedNeXt_MLA_SizeOV4
MedNeXt_MLA_FPSafe
```

建议列：

```text
Method
MLA
SizeOV4
FPSafe
Internal Overall
External Overall
Drop
Internal Tumor Dice
External Tumor Dice
External Recall
External Precision
External FP rate
```

表内结论：

```text
MedNeXt_MLA 是外部最稳的 MedNeXt 变体。
SizeOV4 提高 MedNeXt 内部表现，但不能解释 MedNeXt_MLA 的外部收益。
FPSafe 内部 FP 控制有帮助，但外部退化，因此是负/中性消融。
```

### 12.4 表 4：失败模式 / case analysis

正式表名建议：

```text
Table 4. Case-level failure mode analysis.
```

用途：

```text
把平均 Dice 后面的临床风险说清楚：
无肿瘤误报、小/隐匿肿瘤漏检、3D 上下文导致邻近切片误判、边界过分割。
```

可能列：

```text
Case
Dataset
Tumor status
Failure type
GT tumor voxels
Pred tumor voxels
Tumor Dice
Recall
Precision
Explanation
```

候选 case：

```text
liver_41: 有阴影无肿瘤，假阳性
liver_127: 极小等密度，漏检
ircadb_014: 无肿瘤外部误报
ircadb_018: 极小肿瘤外部严重失败
ircadb_008: 小肿瘤外部低 Dice
```

注意：

```text
表 4 不需要纳入所有方法。
它可以只围绕 MedNeXt_MLA、MedNeXt、MedNeXt_SizeOV4 或少数代表方法做 case-level 对照。
重点是解释视觉歧义和 3D 上下文限制，不是再做一次排名。
```

---

## 13. 图计划

### 13.1 方法图

MedNeXt_MLA 结构图：

```text
Input CT
 -> MedNeXt encoder
 -> bottleneck feature
 -> MLA bottleneck
 -> MedNeXt decoder
 -> segmentation
```

需要突出：

1. 只在 bottleneck 加模块。
2. MLA 是低秩 KV 压缩。
3. 不改变 decoder / loss / supervision。

### 13.2 排名图

Internal vs External Overall 散点图：

```text
x = internal Overall
y = external Overall
```

应展示：

1. MedNeXt_SizeOV4 内部高、外部低。
2. MedNeXt_MLA 内部中上、外部最高。
3. MoE_SizeOV5 / MLAUNet 外部稳定。

### 13.3 Drop 条形图

展示：

```text
External Overall - Internal Overall
```

重点：

```text
MedNeXt: -0.0697
MedNeXt_SizeOV4: -0.0634
MedNeXt_MLA: -0.0180
```

### 13.4 失败案例图

至少三类：

1. 有阴影无肿瘤 -> 假阳性。
2. 无阴影有肿瘤 -> 漏检。
3. 3D 上下文帮助与限制 -> 连续切片中弱证据切片仍可能漏检。

---

## 14. 当前还没想清楚的问题

### 14.1 题目是否要继续用“跨域泛化”

困惑：

```text
“跨域泛化”单独看可能有点浅。
```

当前判断：

```text
题目可以保留跨数据集泛化，但正文必须加入 CT 视觉歧义和临床安全错误分析。
论文深度不只来自外部验证数字，而来自对失败模式的解释。
```

### 14.2 MLA/MoE 旧创新是否还要

当前判断：

```text
MLA 要保留，并迁移到 MedNeXt_MLA 作为核心结构。
MoE 不作为主创新，写成探索性对照。
SizeOV 不作为主创新，写成采样对照。
```

### 14.3 HCC 是否进入正式主表

当前判断：

```text
暂时不进 Dataset003/IRCADb 主表。
HCCReferencedCT 训练已经启动，当前跑的是 HCCRefOnly701020 fold 0。
训练完成后优先作为 HCC 专病数据 sanity / held-out test 线，而不是直接混入当前主表。
```

当前正在运行的 HCC trainer：

```text
dataset:
  Dataset013_HCCReferencedCT

trainer:
  nnUNetTrainer_MedNeXt_MLA_HCCRefOnly701020

configuration:
  3d_fullres, fold_0

output:
  /home/PuMengYu/nnUNet_workspace/results_v2/Dataset013_HCCReferencedCT/nnUNetTrainer_MedNeXt_MLA_HCCRefOnly701020__nnUNetPlans__3d_fullres/fold_0

当前进度:
  2026-07-10 已跑到 epoch 600 左右
  checkpoint_latest.pth 已更新
  checkpoint_best.pth 已生成
```

这个 trainer 的意义：

1. 它使用新的单通道 `Dataset013_HCCReferencedCT`，不是旧 `Dataset013_HCCMultiPhase`。
2. 它验证 MedNeXt_MLA 在 HCC-TACE referenced CT 专病数据上能否稳定训练。
3. 当前划分是 70/10/21：train=70，val=10，held-out test=21。
4. test cases 不写入 nnU-Net 的 `splits_final.json`，不能参与 checkpoint selection。
5. 训练完成后，应先在 21 个 held-out HCC test cases 上生成独立报告，再决定是否写进 Supplementary / Discussion。

关于“只有 IRCADb 一个外部集够不够”的判断：

```text
够写，但不够厚。

如果本文题目只写“外部验证”，IRCADb 一个外部集可以支撑；
如果题目写“跨数据集验证”和“外部可靠性”，最好再加一个外部证据。
HCCReferencedCT 是最自然的第二数据线，因为它来自不同来源、病例语义更接近 HCC 临床场景，
并且存在坏死、治疗相关改变、肿瘤/肝脏比例异常等更真实的困难因素。
```

当前最稳实验路线：

1. 不先做大而全的 HCC 多模型竞赛。
2. 先跑 `nnUNetTrainer_MedNeXt_MLA_HCCRefOnly` fold 0，确认数据、标签和 trainer 稳定。
3. 如果 HCCRefOnly Dice 和可视化合理，再考虑做一个“Dataset003/LiTS 训练的 MedNeXt_MLA -> HCCReferencedCT 直接推理”的外部 HCC 测试。
4. 如果 HCC 直接外推结果很差，也可以写成重要发现：HCC-TACE referenced CT 与 LiTS/IRCADb 的分布差异更大，说明外部可靠性问题更严重。
5. 正式主表是否加入 HCC，等 HCC fold 0 sanity 和至少一个跨数据集推理结果出来后再决定。

进一步澄清：

```text
HCCRefOnly 训练不是用来直接证明当前“LiTS/MSD 内部 -> IRCADb 外部”的主结论。
它验证的是：同一个 MedNeXt_MLA 框架在干净的 HCC referenced-CT 专病数据上是否能稳定训练、验证 Dice 是否合理、是否出现治疗相关或坏死相关的典型失败 case。
```

HCC 可以回答的问题：

1. MedNeXt_MLA 能否在 HCC-TACE referenced CT 上稳定训练。
2. HCC 专病数据中的坏死、术后/治疗相关改变、超大 tumor/liver ratio case 是否造成特殊失败。
3. LiTS/IRCADb 上总结出的 CT 视觉歧义，在 HCC 数据中是否也能观察到。
4. 后续是否值得做 MSD+HCC 单通道混合训练。

HCC 暂时不能直接回答的问题：

1. 不能仅凭 HCCRefOnly fold 0 证明 MedNeXt_MLA 的跨数据集泛化更强，因为这是 HCC 内部 train/val。
2. 不能把 HCCRefOnly 的 val Dice 和 LiTS/IRCADb 的 test Dice 直接横向比较，因为数据来源、病例组成和划分口径不同。
3. 不能把 HCC 结果直接混进当前主表，除非完成统一说明、固定划分、可视化和至少一个合理对照。

如果要用 HCC 证明“跨域”，需要另做跨数据集实验：

```text
方案 A：Dataset003/LiTS 训练的 MedNeXt_MLA -> 直接推理 HCCReferencedCT，作为外部 HCC 测试。
方案 B：HCCRefOnly 训练 -> 推理 IRCADb 或 Dataset003 test，测试反向泛化。
方案 C：MSD + HCC 单通道混合训练 -> 分别在 IRCADb 和 HCC held-out 上评估。
```

当前不建议一上来做 A/B/C。先做 HCCRefOnly fold 0 sanity，确认数据和 trainer 没问题。

### 14.3.1 HCC 固定划分怎么处理

当前困惑：

```text
HCC 没有现成 splits_final.json，要不要手动固定划分？
```

当前建议：

```text
第一步不手动造 split，直接让 nnU-Net 第一次训练自动生成 splits_final.json。
nnU-Net 默认使用 KFold(n_splits=5, shuffle=True, random_state=12345)。
Dataset013_HCCReferencedCT 共有 101 cases，因此 fold 0 预计 train=80, val=21。
```

这样做的理由：

1. 这是 nnU-Net 标准流程，最少引入人为变量。
2. `random_state=12345` 固定，因此生成后是可复现的。
3. 第一次训练生成 `splits_final.json` 后必须保留，不要删除；后续所有 HCC 对照都使用同一个 split。

HCC 结果如果要写进论文，必须记录：

```text
split file:
/home/PuMengYu/nnUNet_workspace/preprocessed/Dataset013_HCCReferencedCT/splits_final.json

fold:
fold_0

train/val:
train = 80
val = 21
```

什么时候需要手动划分：

1. 如果要按 tumor/liver ratio 分层，避免 `HCC_065`、`HCC_075` 这类 review case 全落到 train 或 val。
2. 如果要划出独立 held-out test，而不是只做 5-fold val。
3. 如果要和 HCC 临床元数据、治疗状态、扫描日期做分层。

当前阶段不需要手动复杂划分。先自动 fold 0 跑通，再根据结果决定是否需要重做固定 held-out。

### 14.4 统计检验怎么做

待补：

```text
case-level Tumor Dice
MedNeXt_MLA vs MedNeXt
MedNeXt_MLA vs MedNeXt_SizeOV4
MedNeXt_MLA vs Baseline
```

优先：

```text
Wilcoxon signed-rank test
如果 p 不显著，报告 effect size 和外部排名/drop。
```

### 14.5 结果是否需要重新核对

已核对：

```text
MedNeXt_MLA 内部/外部主结果与原始 report 一致。
```

还需核对：

```text
主表中所有方法的最终数字，尤其是新生成的 2026-07-08 HCC/MSD 相关 report 是否不应混入主表。
```

---

## 15. 可以直接搬进正式论文的段落草稿

### 15.1 引言段落草稿

肝脏肿瘤三维分割模型在内部测试集上取得较高 Dice，并不必然意味着其在外部临床数据上可靠。真实应用中，模型需要面对不同机构、扫描协议、肿瘤大小、对比度分布以及无肿瘤病例中的良性低密度结构。本文的系统实验显示，MedNeXt 和 MedNeXt_SizeOV4 在内部 Dataset003/LiTS 风格测试集上取得最高 Overall，但在 3D-IRCADb 外部验证中出现明显性能下降。相反，MedNeXt_MLA 虽然内部 Overall 排名第 6，却在 IRCADb 外部验证中取得 Overall 第 1，并显著减小 internal-external drop。这一现象说明，肝肿瘤分割方法不能只围绕内部平均 Dice 优化，还需要同时评估外部泛化、无肿瘤误报和隐匿肿瘤漏检。

### 15.2 方法动机段落草稿

MedNeXt 的强性能来自深层 ConvNeXt-style 局部卷积骨干，包括深度可分离卷积、倒置瓶颈、GroupNorm 和残差连接。然而，强局部表征也可能使模型更依赖源域纹理和局部对比度模式。为补充这一不足，本文在 MedNeXt 的最低分辨率 bottleneck 后插入低秩 Multi-head Latent Attention。该位置具有最高语义抽象程度和最低空间分辨率，适合以可控计算成本建模全局 liver-tumor context。与直接在高分辨率特征上使用 attention 不同，bottleneck MLA 尽量保持 MedNeXt 局部骨干不变，只在全局上下文层面对其进行补充。

### 15.3 结果段落草稿

在内部 Dataset003 测试集上，MedNeXt_SizeOV4 和 MedNeXt 分别取得 Overall 0.8431 和 0.8402，位列前两名；MedNeXt_MLA 的内部 Overall 为 0.8259，排名第 6。该结果说明 MedNeXt_MLA 并不是内部测试集上的最高分模型。然而，在 3D-IRCADb 外部验证中，MedNeXt_MLA 取得 Overall 0.8079、Tumor Dice 0.6484，位列所有有效方法第一；而 MedNeXt_SizeOV4 和 MedNeXt 的外部 Overall 分别下降到 0.7797 和 0.7705。MedNeXt_MLA 的 internal-external drop 仅为 -0.0180，明显小于 MedNeXt_SizeOV4 的 -0.0634 和 MedNeXt 的 -0.0697。这一排名反转表明，同域最优模型并不必然具备最佳外部泛化能力。

### 15.4 CT 视觉歧义段落草稿

除平均指标外，本文进一步分析模型在典型病例中的错误来源。我们将 CT 视觉歧义定义为：在单期 CT 图像上，看起来像肿瘤的低密度区域不一定是真正肿瘤，而真正的肿瘤有时又缺乏明显低密度阴影或异常纹理。模型通常更容易处理有可见阴影的区域；但对于等密度或无明显阴影的真实肿瘤，模型即使在训练集中见过类似病例，也可能难以稳定识别。此外，由于模型以 3D patch 为输入，当前 2D 切片的预测还会受到邻近切片肿瘤或阴影的影响，导致边界过分割或邻近切片假阳性。因此，肝肿瘤分割的失败并不能完全归因于网络结构不足，还受到 CT 模态视觉证据和 3D 上下文传播机制的共同限制。

### 15.5 负结果段落草稿

本文还观察到，显式无肿瘤 FP 抑制并不必然提升外部可靠性。MedNeXt_MLA_FPSafe 在内部测试集上将 Overall 提升到 0.8326，并将内部无肿瘤误报率降至 33%，但在 IRCADb 外部验证中 Overall 下降到 0.7744，外部无肿瘤误报率仍为 60%。这说明基于源域无肿瘤样本学习到的 FP 抑制策略可能具有明显同域性，不能直接等价于跨域临床安全性。

---

## 16. 原始材料索引

状态与主线：

```text
pumengyu/notes/md/论文写作状态.md
pumengyu/notes/md/论文主线.md
```

实验结果：

```text
pumengyu/notes/md/02_实验结果/README.md
pumengyu/notes/md/02_实验结果/外部验证_IRCADb.md
pumengyu/notes/md/02_实验结果/消融分析.md
```

架构：

```text
pumengyu/notes/md/00_架构设计/MedNeXt架构.md
pumengyu/notes/md/00_架构设计/MLA_MoE架构.md
```

旧稿与可复用材料：

```text
pumengyu/notes/paper/正式论文v3_MedNeXt_MLA框架.md
pumengyu/notes/paper/正式论文v2.md
pumengyu/notes/paper/异常case分析材料.md
```

数据集：

```text
pumengyu/notes/md/03_数据集/HCC_TACE_ReferencedCT_重建记录.md
pumengyu/notes/md/03_数据集/IRCADb标注与后处理.md
pumengyu/notes/md/03_数据集/数据分布度量.md
```

原始 report：

```text
/home/PuMengYu/nnUNet_workspace/results_v2/Dataset003_Liver
/home/PuMengYu/nnUNet_workspace/results_v2/ExternalVal_IRCADb
```

---

## 17. 下一步任务

P0：

1. 从本文第 15 节开始扩写 Introduction。
2. 把第 7 节方法整理成完整 Method。
3. 从第 6、8、12 节整理主表。
4. 做 case-level 统计检验。
5. 选 3 类失败模式图：假阳性、漏检、3D 上下文帮助与限制。

P1：

1. 跑 HCCRefOnly fold 0 sanity。
2. 补参数量/FLOPs/推理时间。
3. 找出 IRCADb `metastasectomi` 相关 case，对应术后切除灶观察。

P2：

1. 全文转英文。
2. 定目标期刊格式。
3. 删除或移动原始路径。
