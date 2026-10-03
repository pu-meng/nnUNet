# MedNeXt+MHA+MoE 模型工作图

这里集中放单独绘制的模型结构与流程图；论文正式图片在相邻的 `paper/figures/`。

| 内容 | 矢量图 |
|---|---|
| MedNeXt-L + MHA + MoE 结构总览 | [model_architecture.svg](model_architecture.svg) |
| 两个 MHA + MoE Transformer Block 的完整串联与残差 | [model_bottleneck_blocks.svg](model_bottleneck_blocks.svg) |
| MHA 投影与 MoE 路由、专家细节 | [model_attention_moe_details.svg](model_attention_moe_details.svg) |
| 病例、训练损失和推理产物流程 | [model_pipeline.svg](model_pipeline.svg) |

[model_architecture.md](model_architecture.md) 是结构说明；
[model_architecture.json](model_architecture.json) 保存静态审计数据。

在仓库根目录用 `./model.sh graph`、`./model.sh pipeline` 或 `./model.sh check`
更新相应产物。旧的 `pumengyu/tools/model_audit/output/` 是指向本目录的软链接。

日常查看只保留 SVG。分析或写论文需要 PNG 时，执行 `./model.sh graph --png` 或 `./model.sh pipeline --png`，同时导出 300 DPI PNG 到 [PNG格式/](PNG格式/)，SVG 仍保留在本目录。
