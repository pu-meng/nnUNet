# 论文图片索引

本目录保存可用于正文或补充材料的正式 SVG 和 300 DPI PNG。当前主稿只引用 SVG；同名 PNG 用于预览和汇报。

## 当前正文六图

1. [`mednext_mla_architecture.svg`](mednext_mla_architecture.svg)：图 1，总体架构。
2. [`mla_bottleneck3d.svg`](mla_bottleneck3d.svg)：图 2，MLABottleneck3D 外层流程。
3. [`mla_low_rank_attention.svg`](mla_low_rank_attention.svg)：图 3，低秩键值自注意力。
4. [`moe_ffn.svg`](moe_ffn.svg)：图 4，MoE 前馈网络。
5. [`three_domain_main_results.svg`](three_domain_main_results.svg)：图 5，三数据域核心结果。
6. [`paper_case_composite_ircadb.svg`](paper_case_composite_ircadb.svg)：图 6，IRCADb 病例级改善与失败模式。

## 保留的补充图片

- `mednext_*`：MedNeXt stage、block、DownBlock、UpBlock 和卷积细节图。
- `mla_moe_block.*`：单个 MLA+MoE block 展开图。
- `ircadb_*`、`lits_*`：精简前的独立病例图。
- `hcc_cross_trainer_failure_overview.*`：HCC 跨 trainer 共同失败总览。

补充图片仍可用于附录、答辩或机制解释，但不能沿用精简前的正文图号。精简前逐图说明已由本索引和 [`../实验与分析/LiTS视觉歧义与失败病例证据.md`](../实验与分析/LiTS视觉歧义与失败病例证据.md) 承接，不再维护重复嵌图预览。图片修改入口见 [`../figure_factory/README.md`](../figure_factory/README.md)。

## 维护边界

- 正式图保留同名 SVG 与 300 DPI PNG：SVG 用于主稿，PNG 用于预览和汇报，两者不视为冗余副本。
- 无同名 SVG、未被脚本或文档引用的单切片中间 PNG 不作为正式成品保留。
- 图形内容只在生成脚本中修改，不直接手改导出文件。
