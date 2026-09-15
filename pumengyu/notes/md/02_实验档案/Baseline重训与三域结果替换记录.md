# Baseline 重训与三域结果替换记录

> 状态：2026-09-03 已完成并提升为 canonical 结果。
> 口径：PMY-LT-v1；LiTS 为自定义固定 92/13/26 划分，不是官方隐藏测试榜结果。

## 1. 为什么替换

旧 canonical `checkpoint_best.pth` 曾被短试跑覆盖，无法证明旧主表 Baseline 数字来自可追溯的正式训练。因此旧结果不再作为论文证据。本次从随机初始化重新训练 `nnUNetTrainer_Baseline`，完成三域 source-only 评价后，将新结果替换到 active `results_v2` 与标准化 `results_v2_best`。

## 2. 模型与 checkpoint

| 项目 | 核验结果 |
|---|---|
| dataset / fold | `Dataset003_Liver` / `fold_0` |
| trainer / configuration | `nnUNetTrainer_Baseline` / `3d_fullres` |
| checkpoint | `checkpoint_best.pth` |
| checkpoint epoch | 969 |
| 修改时间 | 2026-09-03 00:20:03 +08:00 |
| SHA256 | `69d6e082f23bc2fbe914548d11c4237bd8874bda676f12d19240f790b8faf208` |
| 模型来源 | LiTS source-only；IRCADb/HCC 数据未用于拟合或模型选择 |

这一路 Baseline 是 nnU-Net v2 按 `nnUNetPlans` 构建的标准 3D full-resolution PlainConvUNet，不包含 MedNeXt、MLA 或 MoE 模块。

## 3. 三域正式结果

| 数据域 | n | Liver Dice | Tumor Dice | Overall | Recall | Precision | FDR | 阴性 FP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LiTS | 26 | 0.9516 | 0.7311 | 0.8414 | 0.7226 | 0.7836 | 0.1729 | 2/3（66.67%） |
| 3D-IRCADb | 20 | 0.9671 | 0.7187 | 0.8429 | 0.6506 | 0.8490 | 0.1510 | 2/5（40.00%） |
| HCCReferencedCT v2 | 21 | 0.8384 | 0.3642 | 0.6013 | 0.2883 | 0.6875 | 0.2649 | N/A（无阴性 case） |

在当前 30-trainer Overall 表中，新 Baseline 的三域名次为 LiTS 第 23、IRCADb 第 8、HCC 第 10。它不改变三个域各自的第 1 名，但显著修正了旧 Baseline 在 HCC 上异常偏低的参照位置。

论文旧表行 `0.8368 / 0.8305 / 0.2511` 已被 `0.8414 / 0.8429 / 0.6013` 取代；三域 Overall 分别变化 `+0.0046 / +0.0124 / +0.3502`。这些差值反映旧结果被可信重训替换，不应解释成某个算法组件的提升。

## 4. 完整产物审计

| 数据域 | 预测 NIfTI | `summary.json` | 文本报告 | `test_viz` PNG | checkpoint 来源 |
|---|---:|---|---|---:|---|
| LiTS | 26/26 | 有 | 有 | 2471 | 通过 |
| 3D-IRCADb | 20/20 | 有 | 有 | 652 | 通过 |
| HCCReferencedCT v2 | 21/21 | 有 | 有 | 677 | 通过 |

因此 Baseline 三域评价满足仓库对“实验完成”的全部要求。HCC 固定测试集 21 例全部为肿瘤阳性，阴性误报指标必须写为 `N/A`，不能写成 0/0 全部正确。

## 5. canonical 路径与旧结果归档

新 active 结果：

- LiTS：`/home/PuMengYu/nnUNet_workspace/results_v2/Dataset003_Liver/nnUNetTrainer_Baseline__nnUNetPlans__3d_fullres`
- 3D-IRCADb：`/home/PuMengYu/nnUNet_workspace/results_v2/IRCADb/source_only/Baseline`
- HCCReferencedCT v2：`/home/PuMengYu/nnUNet_workspace/results_v2/Dataset013_HCCReferencedCT/source_only/Baseline`
- 标准化汇总：`/home/PuMengYu/nnUNet_workspace/results_v2_best/{Dataset003_Liver/Baseline,ExternalVal_IRCADb/Baseline,ExternalVal_HCCReferencedCT_v2/Baseline}`

旧结果没有删除，已合并为 8T 上的单一可恢复归档，并在目录名中明确标记 `INVALID` 与 `DO_NOT_USE`。系统盘不再保留归档软链接或空目录：

- 归档位置：`/home/PuMengYu/8T/nnUNet_workspace_archive/results_archive/INVALID_Baseline_before_retrain_20260903_DO_NOT_USE`
- 清单：物理归档根目录下的 `ARCHIVE_MANIFEST.md`

## 6. 对实验分析的影响

1. Baseline 现在是可用于论文主表的可信参照，而不是待补缺口。
2. Baseline 与 NoMirror 的 Overall 差距从旧记录的 0.0056 缩小为 0.0010；两者阴性病例 FP 率同为 2/3，不能再声称关闭镜像降低了病例级误报率。
3. Baseline 相对 SizeOV2 的三域 Overall 差值为 `+0.0071 / -0.0017 / +0.0005`；SizeOV2 只在 LiTS 明显更高，外部域近乎持平。
4. Baseline 相对 SizeOV3 的三域差值为 `+0.0024 / +0.0029 / -0.2087`；SizeOV3 在 HCC 的明显退化仍成立。
5. HCC 跨 trainer 失败矩阵已按新 Baseline per-case 结果重生成；6 个全方法失败和 11 个多数方法失败病例集合不变，但 Baseline 对应单元格已更新。

## 7. 当前未完成项

- 修复后的 `MedNeXt_MLA_MoE` 重训仍未完成：训练日志到 epoch 268，`checkpoint_latest.pth` 与 `checkpoint_best.pth` 均记录 epoch 250，尚无 `checkpoint_final.pth` 和三域下游产物。
- 其余历史方法的预测/报告/可视化/checkpoint 来源完整度不能由 Baseline 的完成状态代替，仍按各自审计结果标记。
