# Shell 脚本索引

这些脚本属于实验复现与产物修复工具，不是实验完成证明。运行前仍需核对数据集、trainer、fold、checkpoint 和输出目录。

## 当前三域评估与产物修复

- `run_hcc_ext_val_v2_gpu1.sh`：HCC source-only 批量评估。
- `run_hcc_mha_mla_gpu01.sh`：MHA／MLA 的 HCC 对照评估。
- `run_hcc_mha_moe_efficient_gpu1.sh`：MHA-MoE 与 EfficientMedNeXt HCC 评估。
- `run_missing_external_v2*.sh`：补齐 IRCADb 缺失方法。
- `rerun_best_reports_gpu1.sh`：按 best checkpoint 重跑独立报告目录。
- `regen_reports_v2.sh`：重建 results_v2 的验证和测试报告。

## 分析与可视化

- `analysis_workflow.sh`、`volume_filter_scan.sh`：数据与连通域分析。
- `copy_viz_files.sh`、`gen_itksnap_viz.sh`：病例可视化材料。
- `predict_trainset.sh`、`run_test_predict_v2.sh`：指定历史实验的补充推理。

## 历史专用流程

- `01_run_inference.sh`：旧 IRCADb 五折推理流程。
- `regen_reports_003.sh`：旧 `results/` 报告修复流程。
- `stage1_predict.sh`、`train_stage2_v2.sh`：旧两阶段 FP 抑制实验。
- `extract_images.sh`、`run_external_import.sh`：原始数据提取和外部阴性病例导入。
- `kill_orphans.sh`：GPU 孤儿进程检查工具，默认只预览。

历史脚本保留是为了追溯，不能未经路径核对直接执行。
