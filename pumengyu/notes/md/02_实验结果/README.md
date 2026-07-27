# 实验结果

## 当前统一总账（2026-07-22）

- [三个数据集全 Trainer 实验结果与消融汇总](三个数据集全Trainer实验结果与消融汇总.md)：30 种公平 source-only 方法的三域总表、域内排名、消融对比和结果覆盖快照。
- [MedNeXt 系列消融实验结果汇总](MedNeXt系列消融实验结果汇总.md)：只聚焦 MedNeXt、MHA、MLA、MoE、SizeOV4 和 FP-Safe 对照。

> 以上两份文档是当前唯一的结果总账。README 只做索引，不再重复维护排名表和消融结论。
>
> HCCReferencedCT v2 当前完成 30/30 种预期 source-only 方法，均具备 21/21 预测、`summary.json`、报告和实际 PNG。纯 `MedNeXt_MLA` 以 Tumor Dice 0.4645、Overall 0.6532 排名第 1。
>
> 2026-07-22 新增：`MedNeXt_MHA_MoE` 三域 Overall 为 0.8508/0.8463/0.6158；`EfficientMedNeXt_L_Official` 三域 Overall 为 0.8518/0.8315/0.5921。MHA/MLA × MLP/MoE 严格 2×2 **指标矩阵**已闭环；新增两方法的 IRCADb/HCC 产物完整，但 internal test 预测 NIfTI 未保留，按仓库交付标准属于部分完成。
>
> **Overall 口径**：每个数据域内按 PMY-LT-v1 独立计算 `Overall = (Liver Dice(all cases) + Tumor Dice(GT-positive cases)) / 2`。`IRCADb Overall` 和 `HCC Overall` 分开报告，**不是两个外部数据集的平均值**。

## 专题补充分析

- [指标统计口径](指标统计口径.md)：PMY-LT-v1 论文主指标、阴性病例处理、Overall 公式及与 nnUNet `summary.json` 的关系。
- [PMY-LT-v1 修改与重算报告](PMY-LT-v1修改与重算报告.md)：详细记录指标原则、代码修改、88 份报告重算范围、前后数字和仍存的产物缺口。
- [MSD、IRCADb 与 HCC 跨 Trainer 失败病例分析](三个数据集失败案例分析.md)：按当前 MSD=29、IRCADb=30、HCC=30 的实际可用方法统计跨 Trainer 共识失败，区分有肿瘤严重失败与无肿瘤误报。
- [Mirror 消融分析](Mirror消融分析.md)：保留 Baseline vs NoMirror 的大小分组、FPV/FNV、连通域和具体病例分析。
