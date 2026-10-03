# nnU-Net v2 流水线速查

按“数据与规划 → 预处理 → 训练 → 推理 → 评估”查阅；要改具体类或方法，转到 [nnUNet 代码地图](nnUNet代码地图.md)。本篇描述本仓库 nnU-Net v2 的实际路径。具体 patch、spacing、网络和训练变体仍须核对对应 `plans.json` 与 Trainer。

## 数据与规划

1. 固定训练、验证、测试病例清单，并核对影像与标签的 shape、spacing、方向和标签值。测试病例不能参加 checkpoint 选择。
2. `DatasetFingerprintExtractor` 从训练数据统计 spacing、裁剪后形状、前景强度和裁剪比例；`ExperimentPlanner` 据此生成 `plans.json`。入口在 `nnunetv2/experiment_planning/`。
3. `plans.json` 决定配置、目标 spacing、patch size、归一化方案、网络结构和重采样函数；不要把旧项目笔记中的固定比例、固定 patch 或 MONAI 类名当作 nnU-Net 默认值。

## 离线预处理

`DefaultPreprocessor.run_case_npy()` 对每例按以下顺序处理：

```text
轴重排 → 裁剪非零区域 → 归一化 → 图像/标签分别重采样
→ 记录前景采样坐标 → 保存预处理数据与 properties
```

归一化必须在重采样之前；代码在 `nnunetv2/preprocessing/preprocessors/default_preprocessor.py`。裁剪框、裁剪前形状、原始 spacing 等属性用于推理后的空间还原。

| 输入类型 | 常见归一化来源 | 查哪里 |
|---|---|---|
| CT | 训练集前景强度统计；通常先按分位数裁剪再标准化 | `preprocessing/normalization/default_normalization_schemes.py` 与 `plans.json` |
| MRI 等 | 依具体 scheme 按病例标准化，可由 mask 控制统计区域 | 同上 |

输入强度归一化与网络内部的 InstanceNorm、GroupNorm、LayerNorm 是不同阶段。标签重采样要保持离散类别；具体插值函数以配置为准。训练时从预处理缓存读取病例并裁 patch。

## 训练

| 环节 | 当前源码中的默认路径 | 需要改时 |
|---|---|---|
| 网络 | 从 `plans.json` 的 architecture 动态构建 | [网络架构笔记](nnUNet架构笔记.md) |
| 采样 | `nnUNetTrainer` 默认前景过采样比例 `0.33`、每 epoch `250` step；子类可覆盖 | `nnUNetTrainer.py`、`data_loader.py` |
| 增强 | 训练 transform 中执行空间和强度增强，镜像轴由配置决定 | `get_training_transforms()` |
| Loss | `_build_loss()` 构建 Dice + CE，并按 deep supervision 输出加权 | `training/loss/`、[Trainer 架构改动](Trainer架构改动.md) |
| 优化器 | `configure_optimizers()` 构建 SGD 与 PolyLR；子类可覆盖 | `nnUNetTrainer.py` |

验证与训练须使用同一数据定义，但验证不执行随机训练增强。`batch_size` 是每 step 的 patch 数；固定 step 数不等于完整遍历一次训练集。自定义采样、loss、网络和报告应通过 Trainer/Mixin 的实际方法核对，不能从类名推断。

## 推理与评估

推理读取训练时的 plans 和 checkpoint，用相同预处理口径处理原图，做滑窗预测和高斯加权，再把结果重采样、放回原始图像空间。镜像 TTA 是否开启及允许的轴取决于推理配置和 checkpoint 的 `inference_allowed_mirroring_axes`；允许 `n` 个轴时最多有 `2^n` 种组合，不应一律写成 8 次。

正式评估按**固定病例清单与数据域**分别计算。多余的历史预测文件不能混入统计；无阴性病例时，阴性误报指标写 `N/A`。每个方法、每个域交付时须同时核对：

- 全部预期病例的 NIfTI 预测；
- `summary.json` 和人类可读的文本报告；
- `test_viz/` 下实际生成的 PNG 数；
- dataset、trainer、fold、模型来源与可信 checkpoint。

已有完整预测时，优先补缺失的指标、报告或可视化，避免重复推理。checkpoint 来源、epoch、修改时间与训练记录有矛盾时，先停下该方法的正式报告。

## 训练与推理成本

本仓库的 Trainer 在 fold 结果目录写 `resource_usage.json`，记录模型参数、训练时间、样本吞吐和显存等；`pumengyu/tools/analyasis/profile_trainer_flops.py` 可离线生成 `offline_complexity.json`。FLOPs、训练吞吐和端到端推理延迟是不同口径，报告时不要混写。历史成本可查 `recover_historical_costs.py`，缺失的历史峰值显存不能从 checkpoint 反推。

当前实验状态和实际结果根以 [当前项目](../01_当前项目/README.md) 与 [实验档案](../02_实验档案/README.md) 为准。
