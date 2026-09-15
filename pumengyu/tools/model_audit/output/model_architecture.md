# MedNeXt-MHA-MoE 模型结构图

这是一张用于代码分析和提出修改思路的结构工作图，不是训练结果。

## 怎么看

- 上半部分：从输入到输出的真实信息流，以及 `E0 → Add0` 等明确 skip 来源和目标。
- 左下：Stage、MedNeXtBlock 主分支和 identity 残差。
- 右下：DownBlock/UpBlock 的重采样 shortcut。
- 最下方：3D feature map 如何变成 token，经过 MHA 和 MoE，再还原。

- MHA/MoE 的投影、注意力矩阵、Router、Top-2 和 expert 汇合见 `model_attention_moe_details.png/svg`。

- 需要看 case、预处理、损失函数和推理后处理：执行 `~/model.sh pipeline`，查看 `model_pipeline.png/svg`。

## 修改位置

- 想改变局部特征：看 MedNeXtBlock。
- 想改变空间尺度：看 DownBlock / UpBlock。
- 想增强全局关系：看 MHA bottleneck。
- 想改变容量或专家选择：看 Router、shared expert 和 routed experts。
- 想研究医院域强度统计：看 GroupNorm / LayerNorm，并结合 feature hook 做验证。

## 当前代码的重要事实

Router 选择 Top-2，但当前实现会先计算全部 4 个 routed experts，再 gather 选中的输出。
因此它不是严格意义上的稀疏计算；这是一个可单独 benchmark 的优化方向。

## 静态审计摘要

- 总参数量：`74,389,680`
- 归一化层：`GroupNorm, LayerNorm`
- Forward shape trace：`未执行`

## 最新静态审计

- 总参数量：`74,389,680`
- 可训练参数量：`74,389,680`
- Forward shape trace：`未执行`

