# MedNeXt_MLA_MoE 三组病例级配对统计

> 生成日期：2026-09-10  
> 主要终点：GT 阳性病例上的 Tumor Dice  
> bootstrap：10,000 次，seed=20260727  
> 检验：双侧 Wilcoxon signed-rank；同一数据域/指标内三组比较采用 Holm 校正。  
> 谱系：MHA+MoE 与 MLA+MoE 使用路由修复后重新训练的 checkpoint；MedNeXt 与纯 MLA 使用现有可信对照。  
> 解释边界：固定 checkpoint 的病例级配对检验不能替代多随机种子或多 fold 训练。

## 1. 输入核对

| 数据域 | 总病例 | 肿瘤阳性 | 肿瘤阴性 | 四模型病例集合 | GT状态/体素数 |
|---|---:|---:|---:|---|---|
| LiTS | 26 | 23 | 3 | 一致 | 一致 |
| IRCADb | 20 | 15 | 5 | 一致 | 一致 |
| HCC | 21 | 21 | 0 | 一致 | 一致 |

统计输入要求逐病例 `summary.json`、文本报告和可信 `checkpoint_best.pth`；完整实验产物还要求 NIfTI 预测与实际 `test_viz` PNG。下表将统计可用性与实验完整性分开报告。本次直接读取已有 `summary.json`，没有重新推理。

| 数据域 | 模型 | 预测病例 | summary | 报告 | test_viz PNG | checkpoint来源 |
|---|---|---:|---|---|---:|---|
| LiTS | MedNeXt | 0 | 存在 | 存在 | 2491 | 通过（epoch 987）；partial |
| LiTS | MedNeXt_MHA_MoE | 26 | 存在 | 存在 | 2491 | 通过（epoch 951）；complete |
| LiTS | MedNeXt_MLA | 0 | 存在 | 存在 | 2487 | 通过（epoch 946）；partial |
| LiTS | MedNeXt_MLA_MoE | 26 | 存在 | 存在 | 2515 | 通过（epoch 914）；complete |
| IRCADb | MedNeXt | 20 | 存在 | 存在 | 686 | 通过（epoch 987）；complete |
| IRCADb | MedNeXt_MHA_MoE | 20 | 存在 | 存在 | 692 | 通过（epoch 951）；complete |
| IRCADb | MedNeXt_MLA | 20 | 存在 | 存在 | 695 | 通过（epoch 946）；complete |
| IRCADb | MedNeXt_MLA_MoE | 20 | 存在 | 存在 | 726 | 通过（epoch 914）；complete |
| HCC | MedNeXt | 21 | 存在 | 存在 | 701 | 通过（epoch 987）；complete |
| HCC | MedNeXt_MHA_MoE | 21 | 存在 | 存在 | 792 | 通过（epoch 951）；complete |
| HCC | MedNeXt_MLA | 21 | 存在 | 存在 | 681 | 通过（epoch 946）；complete |
| HCC | MedNeXt_MLA_MoE | 21 | 存在 | 存在 | 686 | 通过（epoch 914）；complete |


## 2. 主要终点

| 数据域 | 对比 | n | 主模型均值 | 对照均值 | 平均差值 | 配对bootstrap 95% CI | 中位差值 | Wilcoxon p | Holm p |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|
| LiTS | 完整组合 vs 原始骨干 | 23 | 0.7509 | 0.7600 | -0.0091 | [-0.0275, 0.0087] | -0.0041 | 0.3931 | 1.0000 |
| LiTS | 固定 MoE：MLA vs MHA | 23 | 0.7509 | 0.7576 | -0.0067 | [-0.0185, 0.0045] | -0.0013 | 0.5202 | 1.0000 |
| LiTS | 固定 MLA：MoE vs MLP | 23 | 0.7509 | 0.7535 | -0.0026 | [-0.0240, 0.0195] | -0.0046 | 0.5009 | 1.0000 |
| IRCADb | 完整组合 vs 原始骨干 | 15 | 0.6729 | 0.6900 | -0.0170 | [-0.0675, 0.0292] | -0.0017 | 0.5509 | 0.5509 |
| IRCADb | 固定 MoE：MLA vs MHA | 15 | 0.6729 | 0.7070 | -0.0341 | [-0.0769, -0.0055] | -0.0096 | 0.0413 | 0.1240 |
| IRCADb | 固定 MLA：MoE vs MLP | 15 | 0.6729 | 0.6778 | -0.0049 | [-0.0590, 0.0480] | -0.0049 | 0.2719 | 0.5439 |
| HCC | 完整组合 vs 原始骨干 | 21 | 0.4511 | 0.4175 | +0.0337 | [-0.0057, 0.0742] | +0.0159 | 0.0684 | 0.2051 |
| HCC | 固定 MoE：MLA vs MHA | 21 | 0.4511 | 0.4342 | +0.0170 | [-0.0300, 0.0638] | +0.0000 | 0.5228 | 1.0000 |
| HCC | 固定 MLA：MoE vs MLP | 21 | 0.4511 | 0.4645 | -0.0134 | [-0.0990, 0.0495] | +0.0000 | 0.6791 | 1.0000 |

## 3. 分域解释

### LiTS

- **完整组合 vs 原始骨干**：平均差值 -0.0091，95% CI [-0.0275, 0.0087]，Holm p=1.0000。主模型平均 Tumor Dice 更低，但当前配对证据未同时满足均值差 CI 不跨 0和Holm校正后 Wilcoxon p<0.05。
- **固定 MoE：MLA vs MHA**：平均差值 -0.0067，95% CI [-0.0185, 0.0045]，Holm p=1.0000。主模型平均 Tumor Dice 更低，但当前配对证据未同时满足均值差 CI 不跨 0和Holm校正后 Wilcoxon p<0.05。
- **固定 MLA：MoE vs MLP**：平均差值 -0.0026，95% CI [-0.0240, 0.0195]，Holm p=1.0000。主模型平均 Tumor Dice 更低，但当前配对证据未同时满足均值差 CI 不跨 0和Holm校正后 Wilcoxon p<0.05。

### IRCADb

- **完整组合 vs 原始骨干**：平均差值 -0.0170，95% CI [-0.0675, 0.0292]，Holm p=0.5509。主模型平均 Tumor Dice 更低，但当前配对证据未同时满足均值差 CI 不跨 0和Holm校正后 Wilcoxon p<0.05。
- **固定 MoE：MLA vs MHA**：平均差值 -0.0341，95% CI [-0.0769, -0.0055]，Holm p=0.1240。主模型平均 Tumor Dice 更低，但当前配对证据未同时满足均值差 CI 不跨 0和Holm校正后 Wilcoxon p<0.05。
- **固定 MLA：MoE vs MLP**：平均差值 -0.0049，95% CI [-0.0590, 0.0480]，Holm p=0.5439。主模型平均 Tumor Dice 更低，但当前配对证据未同时满足均值差 CI 不跨 0和Holm校正后 Wilcoxon p<0.05。

### HCC

- **完整组合 vs 原始骨干**：平均差值 +0.0337，95% CI [-0.0057, 0.0742]，Holm p=0.2051。主模型平均 Tumor Dice 更高，但当前配对证据未同时满足均值差 CI 不跨 0和Holm校正后 Wilcoxon p<0.05。
- **固定 MoE：MLA vs MHA**：平均差值 +0.0170，95% CI [-0.0300, 0.0638]，Holm p=1.0000。主模型平均 Tumor Dice 更高，但当前配对证据未同时满足均值差 CI 不跨 0和Holm校正后 Wilcoxon p<0.05。
- **固定 MLA：MoE vs MLP**：平均差值 -0.0134，95% CI [-0.0990, 0.0495]，Holm p=1.0000。主模型平均 Tumor Dice 更低，但当前配对证据未同时满足均值差 CI 不跨 0和Holm校正后 Wilcoxon p<0.05。

## 4. 关键病例

下表分别列出每组 Tumor Dice 对比中主模型改善最大和退化最大的病例。

| 数据域 | 对比 | 改善最大3例 | 退化最大3例 |
|---|---|---|---|
| LiTS | 完整组合 vs 原始骨干 | liver_58 (+0.102), liver_90 (+0.050), liver_84 (+0.029) | liver_130 (-0.134), liver_101 (-0.075), liver_63 (-0.070) |
| LiTS | 固定 MoE：MLA vs MHA | liver_58 (+0.042), liver_127 (+0.041), liver_97 (+0.022) | liver_101 (-0.068), liver_11 (-0.063), liver_130 (-0.049) |
| LiTS | 固定 MLA：MoE vs MLP | liver_127 (+0.155), liver_58 (+0.067), liver_101 (+0.050) | liver_130 (-0.147), liver_11 (-0.070), liver_52 (-0.052) |
| IRCADb | 完整组合 vs 原始骨干 | ircadb_016 (+0.191), ircadb_017 (+0.035), ircadb_019 (+0.016) | ircadb_012 (-0.286), ircadb_003 (-0.133), ircadb_008 (-0.047) |
| IRCADb | 固定 MoE：MLA vs MHA | ircadb_008 (+0.018), ircadb_015 (+0.011), ircadb_013 (+0.006) | ircadb_012 (-0.277), ircadb_003 (-0.114), ircadb_017 (-0.058) |
| IRCADb | 固定 MLA：MoE vs MLP | ircadb_016 (+0.240), ircadb_003 (+0.146), ircadb_015 (+0.007) | ircadb_012 (-0.284), ircadb_008 (-0.063), ircadb_004 (-0.034) |
| HCC | 完整组合 vs 原始骨干 | HCC_047 (+0.245), HCC_011 (+0.226), HCC_025 (+0.184) | HCC_060 (-0.190), HCC_071 (-0.071), HCC_093 (-0.039) |
| HCC | 固定 MoE：MLA vs MHA | HCC_028 (+0.270), HCC_071 (+0.199), HCC_025 (+0.139) | HCC_020 (-0.248), HCC_060 (-0.161), HCC_052 (-0.068) |
| HCC | 固定 MLA：MoE vs MLP | HCC_025 (+0.192), HCC_011 (+0.182), HCC_047 (+0.172) | HCC_082 (-0.729), HCC_071 (-0.143), HCC_028 (-0.060) |

## 5. 阴性病例误报（描述性）

| 数据域 | 模型 | 阴性病例 | 误报病例 | case IDs |
|---|---|---:|---:|---|
| LiTS | MedNeXt | 3 | 1 | liver_41 |
| LiTS | MedNeXt_MHA_MoE | 3 | 1 | liver_41 |
| LiTS | MedNeXt_MLA | 3 | 1 | liver_41 |
| LiTS | MedNeXt_MLA_MoE | 3 | 2 | liver_41, liver_91 |
| IRCADb | MedNeXt | 5 | 3 | ircadb_005, ircadb_007, ircadb_014 |
| IRCADb | MedNeXt_MHA_MoE | 5 | 3 | ircadb_005, ircadb_007, ircadb_014 |
| IRCADb | MedNeXt_MLA | 5 | 3 | ircadb_005, ircadb_007, ircadb_014 |
| IRCADb | MedNeXt_MLA_MoE | 5 | 3 | ircadb_005, ircadb_007, ircadb_014 |
| HCC | MedNeXt | 0 | N/A | 无 |
| HCC | MedNeXt_MHA_MoE | 0 | N/A | 无 |
| HCC | MedNeXt_MLA | 0 | N/A | 无 |
| HCC | MedNeXt_MLA_MoE | 0 | N/A | 无 |

Internal 仅 3 个阴性病例、IRCADb 仅 5 个阴性病例，因此不把 FP 计数包装为强统计显著性结论。HCC test 全部为肿瘤阳性病例，FP rate 为 N/A。

## 6. 次要终点

完整的 Tumor Recall、Tumor Precision 和 Liver Dice 结果见 `paired_case_statistics.csv`。这些指标与主要终点使用相同的配对 bootstrap、Wilcoxon 和分域 Holm 校正流程。

## 7. 产物

- `paired_case_statistics.csv`：聚合统计；
- `paired_case_differences.csv`：每例配对明细；
- `statistics_metadata.json`：输入路径、病例清单、checkpoint 和参数；
- `paired_case_tumor_dice_table.md`：正文候选病例级配对统计表。

## 8. 尚未回答的问题

- 当前统计不能量化训练随机性；
- 多随机种子和多 fold 尚未完成；
- 若小差值未通过校正，论文应报告方向和置信区间，不写“显著优于”；
- 病例级统计完成不等于模型效率、Attention机制或因果解释已经完成。
