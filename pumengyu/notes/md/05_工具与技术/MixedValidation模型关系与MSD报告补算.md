# MixedValidation 模型关系与 MSD 报告补算

## 1. 目的

本文说明以下结果目录中三个数据域实际使用的模型，并记录 MSD 已有预测但缺少分析产物时的安全补算命令：

```text
/home/PuMengYu/nnUNet_workspace/results_v2/Dataset013_HCCReferencedCT/mixed_adapter/
```

补算原则：

- 复用已经完成的 26 个 MSD NIfTI 预测；
- 不重新执行 GPU 推理；
- 补齐 `summary.json`、`test_report_custom.txt` 和 `test_viz/*.png`；
- 全部正式结果使用 `checkpoint_best.pth`；
- 不传 `--force`。

## 2. 先分清三个模型名称

### 2.1 MedNeXt_MLA_MoE Base

`MedNeXt_MLA_MoE_Base` 不是新训练的模型。它只是 mixed 结果目录中对原始 Dataset003 模型使用的显示名称。

真实 trainer：

```text
nnUNetTrainer_MedNeXt_MLA_MoE
```

训练数据：

```text
Dataset003_Liver（MSD Liver）
```

正式 checkpoint：

```text
/home/PuMengYu/nnUNet_workspace/results_v2/Dataset003_Liver/nnUNetTrainer_MedNeXt_MLA_MoE__nnUNetPlans__3d_fullres/fold_0/checkpoint_best.pth
```

这个 Base 模型在 mixed 实验中承担两个任务：

```text
Dataset003/MSD 训练得到的 Base
  ├── 测试 MSD 固定 26 例
  └── 外部测试 IRCADb 20 例
```

### 2.2 HCCRefOnly701020

`HCCRefOnly701020` 是独立对照实验，不属于 mixed 路由。

它使用 Dataset013 的 HCC 数据训练整个 MedNeXt+MLA 网络：

```text
HCC train 70 例 → 训练完整 HCCRefOnly 模型
HCC val   10 例 → 训练监控和 checkpoint 选择
IRCADb    20 例 → 跨域外部测试
```

因此，HCCRefOnly 在 IRCADb 上的结果回答的是：只在 HCC 上训练的完整模型能否泛化到 IRCADb。

它不是 adapter，也不是 `checkpoint_best` 的简称。

### 2.3 HCCAdapter701020

HCC Adapter 使用 Dataset003 Base 的 best checkpoint 初始化，然后冻结 Base，只训练 `hcc_adapter.*` 参数：

```text
Dataset003 Base checkpoint_best.pth
  + HCC train 70 例只训练 adapter
  + HCC val 10 例选择 best
  → HCC 固定 test 21 例
```

正式 HCC 测试不能混入 train/val 病例。

## 3. MixedValidation 的实际路由

```text
MSD 固定测试 26 例
  → Dataset003 MedNeXt_MLA_MoE Base
  → MSD/MedNeXt_MLA_MoE_Base/

IRCADb 外部验证 20 例
  → Dataset003 MedNeXt_MLA_MoE Base
  → IRCADb/MedNeXt_MLA_MoE_Base/

HCC 固定测试 21 例
  → Dataset013 HCCAdapter701020
  → HCC/MedNeXt_MLA_MoE_HCCAdapter701020/
```

HCCRefOnly 不在这条 mixed 路由中。

## 4. 已有实验结果汇总

以下结果于 2026-07-14 按实际报告和目录重新核对。`Overall` 定义为 Liver Dice 与 Tumor Dice 的平均值。

### 4.1 正式 MixedValidation 路由

| 测试域 | 实际使用模型 | 模型训练/适配数据 | checkpoint | 测试病例 | Liver Dice | Tumor Dice | Overall | Recall | Precision | 阴性误报 | 当前状态 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| MSD 固定 test | MedNeXt_MLA_MoE Base | Dataset003_Liver | best | 26 | 待补算 | 待补算 | 待补算 | 待补算 | 待补算 | 待补算 | 预测 26/26；报告未完成 |
| IRCADb 外部验证 | MedNeXt_MLA_MoE Base | Dataset003_Liver | best | 20 | 0.9674 | 0.6499 | 0.8086 | 0.6710 | 0.7417 | 2/5（40%） | 完整完成 |
| HCC 固定 test | HCCAdapter701020 | Dataset003 Base 初始化；HCC 70 例只训练 adapter | best | 21 | 0.6571 | 0.1497 | 0.4034 | 0.1244 | 0.3028 | N/A（无阴性病例） | 完整完成 |

正式 mixed 路由的产物检查：

| 测试域 | NIfTI 预测 | `summary.json` | TXT 报告 | PNG | 结果目录 |
|---|---:|---|---|---:|---|
| MSD | 26/26 | 缺失 | 缺失 | 0 | `MixedValidation.../MSD/MedNeXt_MLA_MoE_Base/` |
| IRCADb | 20/20 | 存在，20 例 | 存在 | 669 | `MixedValidation.../IRCADb/MedNeXt_MLA_MoE_Base/` |
| HCC | 21/21 | 存在，21 例 | 存在 | 705 | `MixedValidation.../HCC/MedNeXt_MLA_MoE_HCCAdapter701020/` |

### 4.2 已有对照实验（不属于 mixed 主路由）

| 对照目的 | 测试域 | 使用模型 | checkpoint 口径 | 病例 | Liver Dice | Tumor Dice | Overall | Recall | Precision | 阴性误报 | 产物状态 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Base 直接跨域到 HCC | HCC 固定 test | Dataset003 MedNeXt_MLA_MoE Base | best | 21 | 0.8441 | 0.4080 | 0.6261 | 0.3369 | 0.6895 | N/A | 完整：21 预测、summary、报告、696 PNG |
| 纯 HCC 完整模型跨域到 IRCADb | IRCADb | HCCRefOnly701020 | 推理命令为 final；其权重与 best 逐张量一致 | 20 | 0.9484 | 0.3072 | 0.6278 | 0.3227 | 0.4923 | 2/5（40%） | 完整：20 预测、summary、报告、649 PNG |
| HCC Adapter 直接跨域到 IRCADb（旧单域结果） | IRCADb | HCCAdapter701020 | final；不作为 mixed 正式路由 | 20 | 0.7758 | 0.0794 | 0.4276 | 0.0815 | 0.2658 | 5/5（100%） | 完整：20 预测、summary、报告、775 PNG |

对应结果目录：

```text
# Base → HCC 固定 21 例
/home/PuMengYu/nnUNet_workspace/results_v2/Dataset013_HCCReferencedCT/source_only/MedNeXt_MLA_MoE/

# HCCRefOnly → IRCADb 20 例
/home/PuMengYu/nnUNet_workspace/results_v2/IRCADb/source_only/MedNeXt_MLA_MoE_HCCRefOnly701020/

# HCCAdapter → IRCADb 20 例（旧单域结果，不纳入 mixed 正式路由）
/home/PuMengYu/nnUNet_workspace/results_v2/IRCADb/source_only/MedNeXt_MLA_MoE_HCCAdapter701020/
```

### 4.3 当前结果应如何解释

在 IRCADb 上：

- Dataset003 Base 的 Tumor Dice 为 0.6499，是当前三种相关模型中最可靠的路由；
- HCCRefOnly 降至 0.3072，说明纯 HCC 完整模型出现明显跨域遗忘；
- HCCAdapter 旧单域结果仅为 0.0794，并且 5/5 阴性病例全部误报，因此禁止把 HCCAdapter 直接路由到 IRCADb。

在 HCC 固定 21 例上：

- Dataset003 Base 的 Tumor Dice 为 0.4080；
- HCCAdapter 的 Tumor Dice 为 0.1497，比 Base 低 0.2583；
- HCCAdapter 当前没有带来适配增益，不能在论文中描述成优于 Base；
- HCC test 没有阴性病例，阴性误报率必须写为 N/A。

目前没有 HCCRefOnly 在固定 HCC test 21 例上的完整独立报告，因此不能用训练期 10 例 validation 指标代替 HCC test 结果。

## 5. 当前 MSD 缺失产物的原因

当前 MSD 目录已经包含固定测试集的 26/26 个预测：

```text
/home/PuMengYu/nnUNet_workspace/results_v2/Dataset013_HCCReferencedCT/mixed_adapter/MSD/MedNeXt_MLA_MoE_Base/predictions/
```

但是缺少：

```text
predictions/summary.json
test_report_custom.txt
test_viz/*.png
```

这表示“预测完成、完整产物未完成”。不能把该 MSD 项目汇报为完整实验。

## 6. 安全补算命令

### 6.1 进入环境

```bash
cd /home/PuMengYu/nnUNet
conda activate medseg
```

### 6.2 先确认现有预测为 26 例

```bash
find /home/PuMengYu/nnUNet_workspace/results_v2/Dataset013_HCCReferencedCT/mixed_adapter/MSD/MedNeXt_MLA_MoE_Base/predictions \
  -maxdepth 1 -type f -name 'liver_*.nii.gz' | wc -l
```

预期输出：

```text
26
```

### 6.3 只补 summary、报告和可视化

```bash
python pumengyu/tools/run_internal_test_best_report.py \
  --trainer nnUNetTrainer_MedNeXt_MLA_MoE \
  --method MedNeXt_MLA_MoE_Base \
  --gpu 0 \
  --result_root /home/PuMengYu/nnUNet_workspace/results_v2/Dataset013_HCCReferencedCT/mixed_adapter \
  --domain_dir MSD
```

现有 26 例预测齐全时，脚本应打印：

```text
[reuse] MedNeXt_MLA_MoE_Base: reuse 26 existing predictions
```

随后脚本会依次完成：

1. 检查 Dataset003 `checkpoint_best.pth` 来源；
2. 复用 26 个预测；
3. 生成 `predictions/summary.json`；
4. 生成 `test_report_custom.txt`；
5. 生成 `test_viz/` 中的 PNG。

不要添加 `--force`。`--force` 会重新执行 26 例 GPU 推理。

### 6.4 刷新 mixed 总结文件

MSD 报告生成后，用 dry-run 刷新 `mixed_validation_summary.json`。dry-run 只更新 manifest/summary，不运行三个数据域的推理：

```bash
python pumengyu/ext_val/07_run_mixed_domain_val.py \
  --dry_run \
  --result_root /home/PuMengYu/nnUNet_workspace/results_v2/Dataset013_HCCReferencedCT/mixed_adapter
```

## 7. 完成后验收

### 7.1 检查必要文件

```bash
MSD_DIR=/home/PuMengYu/nnUNet_workspace/results_v2/Dataset013_HCCReferencedCT/mixed_adapter/MSD/MedNeXt_MLA_MoE_Base

test -f "$MSD_DIR/predictions/summary.json" && echo 'summary: OK'
test -f "$MSD_DIR/test_report_custom.txt" && echo 'report: OK'
find "$MSD_DIR/predictions" -maxdepth 1 -type f -name 'liver_*.nii.gz' | wc -l
find "$MSD_DIR/test_viz" -type f -name '*.png' | wc -l
```

必须满足：

- NIfTI 预测为 26 例；
- `summary.json` 存在；
- `test_report_custom.txt` 存在；
- `test_viz/` 中实际存在 PNG；
- checkpoint 来源检查通过。

### 7.2 检查 mixed 三域状态

```bash
jq . /home/PuMengYu/nnUNet_workspace/results_v2/Dataset013_HCCReferencedCT/mixed_adapter/mixed_validation_summary.json
```

预期 MSD、IRCADb、HCC 三项的 `status` 均为：

```text
complete
```

## 8. 结果路径

MSD Base：

```text
/home/PuMengYu/nnUNet_workspace/results_v2/Dataset013_HCCReferencedCT/mixed_adapter/MSD/MedNeXt_MLA_MoE_Base/
```

IRCADb Base：

```text
/home/PuMengYu/nnUNet_workspace/results_v2/Dataset013_HCCReferencedCT/mixed_adapter/IRCADb/MedNeXt_MLA_MoE_Base/
```

HCC Adapter：

```text
/home/PuMengYu/nnUNet_workspace/results_v2/Dataset013_HCCReferencedCT/mixed_adapter/HCC/MedNeXt_MLA_MoE_HCCAdapter701020/
```
