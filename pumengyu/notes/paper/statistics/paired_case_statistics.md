# MedNeXt_MLA_MoE 三组病例级配对统计

> 生成日期：2026-07-30  
> 主要终点：GT 阳性病例上的 Tumor Dice  
> bootstrap：10,000 次，seed=20260727  
> 检验：双侧 Wilcoxon signed-rank；同一数据域/指标内三组比较采用 Holm 校正。  
> 解释边界：固定 checkpoint 的病例级配对检验不能替代多随机种子或多 fold 训练。

## 1. 输入核对

| 数据域 | 总病例 | 肿瘤阳性 | 肿瘤阴性 | 四模型病例集合 | GT状态/体素数 |
|---|---:|---:|---:|---|---|
| Internal | 26 | 23 | 3 | 一致 | 一致 |
| IRCADb | 20 | 15 | 5 | 一致 | 一致 |
| HCC | 21 | 21 | 0 | 一致 | 一致 |

统计输入要求逐病例 `summary.json`、文本报告和可信 `checkpoint_best.pth`；完整实验产物还要求 NIfTI 预测与实际 `test_viz` PNG。下表将统计可用性与实验完整性分开报告。本次直接读取已有 `summary.json`，没有重新推理。

| 数据域 | 模型 | 预测病例 | summary | 报告 | test_viz PNG | checkpoint来源 |
|---|---|---:|---|---|---:|---|
| Internal | MedNeXt | 0 | 存在 | 存在 | 2491 | 通过（epoch 987）；partial |
| Internal | MedNeXt_MHA_MoE | 0 | 存在 | 存在 | 2475 | 通过（epoch 957）；partial |
| Internal | MedNeXt_MLA | 0 | 存在 | 存在 | 2487 | 通过（epoch 946）；partial |
| Internal | MedNeXt_MLA_MoE | 0 | 存在 | 存在 | 2473 | 通过（epoch 998）；partial |
| IRCADb | MedNeXt | 20 | 存在 | 存在 | 686 | 通过（epoch 987）；complete |
| IRCADb | MedNeXt_MHA_MoE | 20 | 存在 | 存在 | 668 | 通过（epoch 957）；complete |
| IRCADb | MedNeXt_MLA | 20 | 存在 | 存在 | 695 | 通过（epoch 946）；complete |
| IRCADb | MedNeXt_MLA_MoE | 20 | 存在 | 存在 | 668 | 通过（epoch 998）；complete |
| HCC | MedNeXt | 21 | 存在 | 存在 | 701 | 通过（epoch 987）；complete |
| HCC | MedNeXt_MHA_MoE | 21 | 存在 | 存在 | 706 | 通过（epoch 957）；complete |
| HCC | MedNeXt_MLA | 21 | 存在 | 存在 | 681 | 通过（epoch 946）；complete |
| HCC | MedNeXt_MLA_MoE | 21 | 存在 | 存在 | 696 | 通过（epoch 998）；complete |


## 2. 主要终点

| 数据域 | 对比 | n | 主模型均值 | 对照均值 | 平均差值 | 配对bootstrap 95% CI | 中位差值 | Wilcoxon p | Holm p |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|
| Internal | 完整组合 vs 原始骨干 | 23 | 0.7590 | 0.7600 | -0.0010 | [-0.0196, 0.0131] | +0.0015 | 0.3765 | 0.7530 |
| Internal | 固定 MoE：MLA vs MHA | 23 | 0.7590 | 0.7499 | +0.0090 | [0.0010, 0.0181] | +0.0045 | 0.0918 | 0.2753 |
| Internal | 固定 MLA：MoE vs MLP | 23 | 0.7590 | 0.7535 | +0.0055 | [-0.0086, 0.0214] | -0.0030 | 1.0000 | 1.0000 |
| IRCADb | 完整组合 vs 原始骨干 | 15 | 0.7349 | 0.6900 | +0.0449 | [0.0048, 0.1206] | +0.0075 | 0.0076 | 0.0229 |
| IRCADb | 固定 MoE：MLA vs MHA | 15 | 0.7349 | 0.7261 | +0.0088 | [-0.0002, 0.0178] | +0.0054 | 0.0640 | 0.1281 |
| IRCADb | 固定 MLA：MoE vs MLP | 15 | 0.7349 | 0.6778 | +0.0571 | [-0.0073, 0.1572] | +0.0000 | 0.9250 | 0.9250 |
| HCC | 完整组合 vs 原始骨干 | 21 | 0.4080 | 0.4175 | -0.0094 | [-0.0446, 0.0267] | +0.0000 | 0.6791 | 0.8409 |
| HCC | 固定 MoE：MLA vs MHA | 21 | 0.4080 | 0.3943 | +0.0137 | [-0.0110, 0.0437] | +0.0028 | 0.4204 | 0.8409 |
| HCC | 固定 MLA：MoE vs MLP | 21 | 0.4080 | 0.4645 | -0.0565 | [-0.1118, -0.0158] | -0.0234 | 0.0050 | 0.0149 |

## 3. 分域解释

### Internal

- **完整组合 vs 原始骨干**：平均差值 -0.0010，95% CI [-0.0196, 0.0131]，Holm p=0.7530。主模型平均 Tumor Dice 更低，但当前配对证据未同时满足均值差 CI 不跨 0和Holm校正后 Wilcoxon p<0.05。
- **固定 MoE：MLA vs MHA**：平均差值 +0.0090，95% CI [0.0010, 0.0181]，Holm p=0.2753。主模型平均 Tumor Dice 更高，但当前配对证据未同时满足均值差 CI 不跨 0和Holm校正后 Wilcoxon p<0.05。
- **固定 MLA：MoE vs MLP**：平均差值 +0.0055，95% CI [-0.0086, 0.0214]，Holm p=1.0000。主模型平均 Tumor Dice 更高，但当前配对证据未同时满足均值差 CI 不跨 0和Holm校正后 Wilcoxon p<0.05。

### IRCADb

- **完整组合 vs 原始骨干**：平均差值 +0.0449，95% CI [0.0048, 0.1206]，Holm p=0.0229。主模型的病例级 Tumor Dice 更高，均值差 CI 不跨 0，Holm 校正后 Wilcoxon p<0.05。
- **固定 MoE：MLA vs MHA**：平均差值 +0.0088，95% CI [-0.0002, 0.0178]，Holm p=0.1281。主模型平均 Tumor Dice 更高，但当前配对证据未同时满足均值差 CI 不跨 0和Holm校正后 Wilcoxon p<0.05。
- **固定 MLA：MoE vs MLP**：平均差值 +0.0571，95% CI [-0.0073, 0.1572]，Holm p=0.9250。主模型平均 Tumor Dice 更高，但当前配对证据未同时满足均值差 CI 不跨 0和Holm校正后 Wilcoxon p<0.05。

### HCC

- **完整组合 vs 原始骨干**：平均差值 -0.0094，95% CI [-0.0446, 0.0267]，Holm p=0.8409。主模型平均 Tumor Dice 更低，但当前配对证据未同时满足均值差 CI 不跨 0和Holm校正后 Wilcoxon p<0.05。
- **固定 MoE：MLA vs MHA**：平均差值 +0.0137，95% CI [-0.0110, 0.0437]，Holm p=0.8409。主模型平均 Tumor Dice 更高，但当前配对证据未同时满足均值差 CI 不跨 0和Holm校正后 Wilcoxon p<0.05。
- **固定 MLA：MoE vs MLP**：平均差值 -0.0565，95% CI [-0.1118, -0.0158]，Holm p=0.0149。主模型的病例级 Tumor Dice 更低，均值差 CI 不跨 0，Holm 校正后 Wilcoxon p<0.05。

## 4. 关键病例

下表分别列出每组 Tumor Dice 对比中主模型改善最大和退化最大的病例。

| 数据域 | 对比 | 改善最大3例 | 退化最大3例 |
|---|---|---|---|
| Internal | 完整组合 vs 原始骨干 | liver_128 (+0.046), liver_84 (+0.045), liver_112 (+0.040) | liver_127 (-0.159), liver_130 (-0.067), liver_36 (-0.011) |
| Internal | 固定 MoE：MLA vs MHA | liver_63 (+0.078), liver_101 (+0.050), liver_127 (+0.026) | liver_58 (-0.018), liver_96 (-0.015), liver_128 (-0.013) |
| Internal | 固定 MLA：MoE vs MLP | liver_101 (+0.124), liver_128 (+0.065), liver_63 (+0.058) | liver_130 (-0.081), liver_96 (-0.023), liver_11 (-0.020) |
| IRCADb | 完整组合 vs 原始骨干 | ircadb_016 (+0.558), ircadb_017 (+0.023), ircadb_019 (+0.021) | ircadb_008 (-0.008), ircadb_004 (-0.003), ircadb_012 (-0.003) |
| IRCADb | 固定 MoE：MLA vs MHA | ircadb_008 (+0.043), ircadb_015 (+0.040), ircadb_016 (+0.026) | ircadb_003 (-0.028), ircadb_013 (-0.012), ircadb_012 (-0.003) |
| IRCADb | 固定 MLA：MoE vs MLP | ircadb_016 (+0.606), ircadb_003 (+0.299), ircadb_015 (+0.040) | ircadb_008 (-0.023), ircadb_009 (-0.019), ircadb_004 (-0.019) |
| HCC | 完整组合 vs 原始骨干 | HCC_082 (+0.228), HCC_020 (+0.069), HCC_026 (+0.049) | HCC_025 (-0.199), HCC_060 (-0.166), HCC_071 (-0.083) |
| HCC | 固定 MoE：MLA vs MHA | HCC_082 (+0.228), HCC_060 (+0.124), HCC_020 (+0.069) | HCC_071 (-0.089), HCC_025 (-0.069), HCC_011 (-0.036) |
| HCC | 固定 MLA：MoE vs MLP | HCC_020 (+0.087), HCC_026 (+0.019), HCC_055 (+0.002) | HCC_082 (-0.501), HCC_025 (-0.191), HCC_071 (-0.154) |

## 5. 阴性病例误报（描述性）

| 数据域 | 模型 | 阴性病例 | 误报病例 | case IDs |
|---|---|---:|---:|---|
| Internal | MedNeXt | 3 | 1 | liver_41 |
| Internal | MedNeXt_MHA_MoE | 3 | 1 | liver_41 |
| Internal | MedNeXt_MLA | 3 | 1 | liver_41 |
| Internal | MedNeXt_MLA_MoE | 3 | 2 | liver_41, liver_91 |
| IRCADb | MedNeXt | 5 | 3 | ircadb_005, ircadb_007, ircadb_014 |
| IRCADb | MedNeXt_MHA_MoE | 5 | 2 | ircadb_007, ircadb_014 |
| IRCADb | MedNeXt_MLA | 5 | 3 | ircadb_005, ircadb_007, ircadb_014 |
| IRCADb | MedNeXt_MLA_MoE | 5 | 2 | ircadb_007, ircadb_014 |
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
