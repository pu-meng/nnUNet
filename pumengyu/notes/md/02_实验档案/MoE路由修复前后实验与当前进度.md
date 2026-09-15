# MoE 修复前后实验进度

> 更新：2026-09-10  
> 完成口径：预测 NIfTI、`summary.json`、文本报告、`test_viz` PNG 和 checkpoint 来源同时通过检查。  
> 三域顺序：LiTS internal test / 3D-IRCADb-01 / HCC-TACE-Seg。

## 1. 现在做到哪里

Overall 顺序均为 `LiTS / 3D-IRCADb-01 / HCC-TACE-Seg`。

| 实验 | 状态 | Overall | 产物 | 当前位置 |
|---|---|---|---|---|
| Baseline 从头重训 | **已完成** | `0.8414 / 0.8429 / 0.6013` | 三域预测、summary、报告、PNG 齐全 | 已放入 `results_v2`；重训原件仍在 `results_v2_baseline_retrain_20260902` |
| 修复后 MHA+MoE | **已完成** | `0.8541 / 0.8360 / 0.6368` | `26/26 / 20/20 / 21/21`；PNG `2491 / 692 / 792`；summary、报告齐全 | `results_v2_baseline_retrain_20260902`；**尚未覆盖旧结果** |
| 修复后 MLA+MoE（第一次尝试） | **中断，不再续训** | — | 日志到 epoch 268，best/latest 为 epoch 250；无 final、无三域产物 | `results_v2_baseline_retrain_20260902` |
| 修复后 MLA+MoE（全新两卡 DDP） | **已完成** | `0.8503 / 0.8174 / 0.6458` | `26/26 / 20/20 / 21/21`；PNG `2515 / 726 / 686`；summary、报告齐全 | `8T/nnUNet_result/results_v2_moe_retrain_after_fix_20260907` |
| 修复后 MLA+MoE+SizeOV4 | **待用户在 tmux 重新启动** | — | PTY 试跑于 epoch 3 中止；原目标已清空，三域产物未开始 | 目标：`8T/nnUNet_result/results_v2_moe_sizeov4_retrain_after_fix_20260910` |

当前是 **Baseline、修复后 MHA+MoE、修复后 MLA+MoE 完成；原计划中的 3 个 MoE 重训变体完成 2/3，SizeOV4 等待在 tmux 中重新从头启动**。若只计算主方法 MLA+MoE 的代码修复、两卡重训和三域交付，则已完成。

### 修复后 MLA+MoE checkpoint

- trainer：`nnUNetTrainer_MedNeXt_MLA_MoE`；dataset：`Dataset003_Liver`；fold：0；
- 从随机初始化开始两卡 DDP 训练，`checkpoint_final.pth` 为 epoch 1000；
- 正式评估使用 `checkpoint_best.pth`，记录 epoch 914；
- SHA256：`f12757ea406f80d603bde452f0d8bbfc285dd9e1e0f0e0c83a0d19ede9565bc6`；
- 2026-09-10 核查时无相关训练或评估进程。

## 2. 名称和结果放哪里

| 统一记录名 | 实际 Trainer | 如何处理 | 结果位置 |
|---|---|---|---|
| 纯 MHA（MHA+MLP） | `nnUNetTrainer_MedNeXt_MHA` | 不含 MoE，**不属于本次修复** | `results_v2/Dataset003_Liver/nnUNetTrainer_MedNeXt_MHA__nnUNetPlans__3d_fullres` |
| MHA+MoE（修复前） | `nnUNetTrainer_MedNeXt_MHA_MoE` | 作为旧实验保留；Overall `0.8508 / 0.8463 / 0.6158` | `results_v2`；checkpoint best epoch 957 |
| MHA+MoE（修复后） | `nnUNetTrainer_MedNeXt_MHA_MoE` | 作为新实验保留；Overall `0.8541 / 0.8360 / 0.6368` | `results_v2_baseline_retrain_20260902`；checkpoint best epoch 951 |
| MLA+MoE（修复前） | `nnUNetTrainer_MedNeXt_MLA_MoE` | 保留原有三域数字，只标注为“修复前” | `results_v2` |
| MLA+MoE（修复后旧尝试） | `nnUNetTrainer_MedNeXt_MLA_MoE` | 中断记录保留，不续训、不记三域指标 | `results_v2_baseline_retrain_20260902`；epoch 250 checkpoint |
| MLA+MoE（修复后新 DDP） | `nnUNetTrainer_MedNeXt_MLA_MoE` | 两卡从头训练完成；正式结果使用 best epoch 914 | `8T/nnUNet_result/results_v2_moe_retrain_after_fix_20260907` |
| MLA+MoE+SizeOV4（修复前） | `nnUNetTrainer_MedNeXt_MLA_MoE_SizeOV4` | 保留原有结果，只标注为“修复前” | `results_v2` |
| MLA+MoE+SizeOV4（修复后） | 同上 | PTY 试跑已中止；等待在 tmux 中两卡从头重启 | 目标：`8T/nnUNet_result/results_v2_moe_sizeov4_retrain_after_fix_20260910` |

后续固定用“**方法名 + 修复前/修复后 + 结果根**”区分。旧结果不删除；新 MHA+MoE 在排名、统计和论文同步更新前，不覆盖 `results_v2` 中的修复前版本。

## 3. 修复后病例级配对统计

统计使用修复后 MLA+MoE、修复后 MHA+MoE，以及现有可信 MedNeXt、纯 MLA checkpoint。主要终点为 GT 肿瘤阳性病例的 Tumor Dice；每域进行 3 组比较，使用 10,000 次病例级配对 bootstrap、双侧 Wilcoxon 和域内 Holm 校正。

| 数据域 | 对照 | MLA+MoE 平均差 | 95% CI | Holm p | 判断 |
|---|---|---:|---|---:|---|
| LiTS | MedNeXt | -0.0091 | [-0.0275, +0.0087] | 1.0000 | 无显著差异 |
| LiTS | 修复后 MHA+MoE | -0.0067 | [-0.0185, +0.0045] | 1.0000 | 无显著差异 |
| LiTS | 纯 MLA | -0.0026 | [-0.0240, +0.0195] | 1.0000 | 无显著差异 |
| IRCADb | MedNeXt | -0.0170 | [-0.0675, +0.0292] | 0.5509 | 无显著差异 |
| IRCADb | 修复后 MHA+MoE | -0.0341 | [-0.0769, -0.0055] | 0.1240 | Holm 校正后不显著 |
| IRCADb | 纯 MLA | -0.0049 | [-0.0590, +0.0480] | 0.5439 | 无显著差异 |
| HCC | MedNeXt | +0.0337 | [-0.0057, +0.0742] | 0.2051 | 无显著差异 |
| HCC | 修复后 MHA+MoE | +0.0170 | [-0.0300, +0.0638] | 1.0000 | 无显著差异 |
| HCC | 纯 MLA | -0.0134 | [-0.0990, +0.0495] | 1.0000 | 无显著差异 |

统计没有支持“修复后 MLA+MoE 稳定优于纯 MLA”或“稳定优于 MedNeXt”。它在 HCC 相对 MedNeXt 的均值方向为正，但区间跨 0，不能写成已证实提升。完整统计和输入谱系见 `pumengyu/notes/paper/statistics/`。

## 4. 下一步决策

1. **SizeOV4 已按用户当前决定启动。** 它改变病例采样和曝光节奏，不能单独回答 MoE 是否有效；最终仍按采样对照解释，不作为 MoE 机制证明。
2. **下一项优先设计 parameter-matched dense control。** 固定 MLA、训练设置和参数规模，用不含 router/专家选择的 dense FFN 对照 MoE，先排除额外参数容量解释。当前每个 MoE block 含 5 个半宽专家（每个 `512→1024→512`）和 `512→4` router，共 5,252,608 个可训练 FFN/router 参数；候选 dense FFN 用 `512→5120→512`（`mlp_ratio=10`），为 5,248,512 个参数，每 block 只少 4,096（约 0.078%），可作为近似参数匹配方案。正式实现前仍需独立参数计数和 forward-shape 测试。
3. dense control 若显示 MoE 有稳定增益，再做多 seed 并重新考虑 SizeOV4；若仍无增益，论文主线应回到 MLA/跨域可靠性边界，MoE 作为未获支持的机制或负结果保留。
4. 修复前结果继续保留为历史证据；在论文、统计表和其他材料全部完成谱系切换前，不删除、不覆盖旧目录。

## 5. 修复后 SizeOV4 启动记录

- 目标 Trainer：`nnUNetTrainer_MedNeXt_MLA_MoE_SizeOV4`；`MLA_USE_MOE=True`；五类 SizeOV4 重复系数均为 2。
- 第一次启动于 2026-09-10 16:27 到达 epoch 0，但在首个 forward 因 PyTorch 2.3 DDPOptimizer 不支持路由图中的 higher-order op 而退出；没有生成 checkpoint，不能记为训练完成。
- 已在该 Trainer 中覆盖 `_do_i_compile()` 返回 `False`，与现有 MLA+MoE Trainer 的 eager-mode 策略一致；语法、Trainer 发现和返回值检查通过。
- 第二次启动于 2026-09-10 16:29，仍不带 `--c`，从随机初始化开始；两卡 DDP world size=2，训练标识 `92→184`。epoch 0 已完成，耗时 126.02 秒，随后在 epoch 3 按用户要求中止。
- epoch 0 后的 `checkpoint_best.pth` 记录 trainer 为 `nnUNetTrainer_MedNeXt_MLA_MoE_SizeOV4`、`current_epoch=1`；两个 MLA+MoE block 的 `expert_bias` 和 `expert_load_ema` 均已写入 checkpoint，说明修复后的路由状态提交链实际运行。
- PTY 试跑目录已移至可恢复位置 `.trash/results_v2_moe_sizeov4_retrain_after_fix_20260910_stopped_20260910_1636`；原目标路径已清空。下一次 tmux 启动不带 `--c`，重新从随机初始化开始。该归档中的初期 checkpoint 不能用于正式评估；三域预测、summary、报告和 PNG 均未开始。
