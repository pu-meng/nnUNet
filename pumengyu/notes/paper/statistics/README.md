# 论文统计与来源索引

本目录保存正文图表背后的机器可读统计、逐病例数据和 provenance。生成文件原则上不手工修改，应回到对应分析脚本重新生成。

## 三域主结果

- `three_domain_main_results.csv`：图 5 使用的数值。
- `three_domain_main_results_metadata.json`：病例数、产物和 checkpoint 来源记录。

## 配对统计

- `paired_case_statistics.md`：完整统计方法、输入审计、配对 bootstrap、Wilcoxon 与 Holm 校正报告；统计解释以此文件为准。
- `paired_case_statistics.csv`：统计检验汇总。
- `paired_case_differences.csv`：逐病例配对差值。
- `paired_case_tumor_dice_table.md`：供正文或审稿快速核对的精简 Tumor Dice 表，不重复展开统计教程。
- `statistics_metadata.json`：统计输入与参数来源。

## 病例图与共同失败

- `paper_case_composite_ircadb_metadata.json`：当前正文图 6 的病例、切片和输入路径。
- `failure_case_figure_provenance.json`：独立失败病例图的来源。
- `hcc_cross_trainer_failure_*`：HCC 跨 trainer 共同失败矩阵及元数据。

## 复杂度

- `core_model_complexity_benchmark.md`：核心模型复杂度可读报告。
- `core_model_complexity_benchmark.json`：机器可读结果。

使用任何表格前仍需核对对应数据域的预测病例数、`summary.json`、文本报告、`test_viz` PNG 和 checkpoint 来源；数值可解析不等同于实验归档完整。
