# HCC 少样本监督目标中心适配实验

## 当前总状态

固定结果根（所有后续 Task02 实验均写入这里）：

`/home/PuMengYu/nnUNet_workspace/results_task02`

| 项目 | 状态 | 说明 |
|---|---|---|
| Replay K03 冒烟 | 已完成 | 训练/DDP/数据流 smoke，不是论文结果 |
| HCC-only K03 微调 | 已完成 | 单卡，300 epoch，K=3 |
| HCC-only K03 best | 已完成 | `checkpoint_task02_best.pth`，选择 epoch 6 |
| HCC-only K03 三域评估 | 已完成 | HCC、IRCADb、LiTS 均通过完整产物审计 |
| Replay K01 | 未开始 | 旧的未完成尝试已清理；下次从头正式训练 |
| 其他正式训练 | 未开始 | 见下方矩阵 |
| 自动评估 pipeline | 未完成 | 当前仍需手动调用 `final_evaluation` |

### 当前缺口清单

- [ ] Replay K=1：从头正式训练；
- [ ] HCC-only K=1：未训练；
- [ ] Replay K=3：未训练；
- [ ] HCC-only K=5：未训练；
- [ ] Replay K=5：未训练；
- [ ] HCC-only K=10：未训练；
- [ ] Replay K=10：未训练；
- [ ] Replay K=70：未训练；
- [ ] 把 final evaluation 接入训练结束 pipeline，并用下一次正式训练验证自动评估是否触发。

HCC-only K03 已满足“完整实验完成”。剩余项目仍需逐个训练并执行三域正式评估。

### HCC-only K03 本轮结果

正式评估根：

`/home/PuMengYu/nnUNet_workspace/results_task02/Task02_FinalEvaluation/MedNeXt_MHA_MoE_Task02_HCCOnly_K03`

| 域 | 固定测试病例 | Tumor Dice | Overall | 产物审计 |
|---|---:|---:|---:|---|
| HCC | 21 | 0.0373 | 0.0238 | 完整 |
| IRCADb | 20 | 0.0920 | 0.0797 | 完整 |
| LiTS | 26 | 0.0599 | 0.0364 | 完整 |

- HCC：21 个预测、721 个 PNG；无阴性病例，误报率为 `N/A`。
- IRCADb：20 个预测、820 个 PNG；报告和 `summary.json` 均存在。
- LiTS：26 个预测、2545 个 PNG；报告和 `summary.json` 均存在。
- 最终审计：`Task02_FinalEvaluation/MedNeXt_MHA_MoE_Task02_HCCOnly_K03/task02_final_artifact_audit.json`，状态为 `complete`。
- checkpoint：`checkpoint_task02_best.pth`，选择 epoch 6；SHA256 为 `eedfee209d4028a0f8602100583d53a5457fd42ba62b741625c90803b27d6c0d`。

## 命令

### 正式训练命令（只保留未完成项目）

每次只执行一个命令；统一写入固定结果根，不能并行抢 GPU。`TASK02_CLEAR_FAILED=1` 只会回收无 checkpoint、无最终评估产物的失败残留；有 checkpoint 时仍会拒绝覆盖。

#### HCC-only K=1

```bash
TASK02_CLEAR_FAILED=1 \
  TASK02_RESULTS_ROOT=/home/PuMengYu/nnUNet_workspace/results_task02 \
  TASK02_CUDA_VISIBLE_DEVICES=0 \
  TASK02_NUM_GPUS=1 \
  bash pumengyu/task02/run_task02_stage2.sh nnUNetTrainer_MedNeXt_MHA_MoE_Task02_HCCOnly_K01
```

#### HCC-only K=5

```bash
TASK02_CLEAR_FAILED=1 \
  TASK02_RESULTS_ROOT=/home/PuMengYu/nnUNet_workspace/results_task02 \
  TASK02_CUDA_VISIBLE_DEVICES=0 \
  TASK02_NUM_GPUS=1 \
  bash pumengyu/task02/run_task02_stage2.sh nnUNetTrainer_MedNeXt_MHA_MoE_Task02_HCCOnly_K05
```

#### HCC-only K=10

```bash
TASK02_CLEAR_FAILED=1 \
  TASK02_RESULTS_ROOT=/home/PuMengYu/nnUNet_workspace/results_task02 \
  TASK02_CUDA_VISIBLE_DEVICES=0 \
  TASK02_NUM_GPUS=1 \
  bash pumengyu/task02/run_task02_stage2.sh nnUNetTrainer_MedNeXt_MHA_MoE_Task02_HCCOnly_K10
```

#### Replay K=1

```bash
TASK02_CLEAR_FAILED=1 \
  TASK02_RESULTS_ROOT=/home/PuMengYu/nnUNet_workspace/results_task02 \
  TASK02_CUDA_VISIBLE_DEVICES=0 \
  TASK02_NUM_GPUS=1 \
  bash pumengyu/task02/run_task02_stage2.sh nnUNetTrainer_MedNeXt_MHA_MoE_Task02_Replay_K01
```

#### Replay K=3

```bash
TASK02_CLEAR_FAILED=1 \
  TASK02_RESULTS_ROOT=/home/PuMengYu/nnUNet_workspace/results_task02 \
  TASK02_CUDA_VISIBLE_DEVICES=0,1 \
  TASK02_NUM_GPUS=2 \
  bash pumengyu/task02/run_task02_stage2.sh nnUNetTrainer_MedNeXt_MHA_MoE_Task02_Replay_K03
```

#### Replay K=5

```bash
TASK02_CLEAR_FAILED=1 \
  TASK02_RESULTS_ROOT=/home/PuMengYu/nnUNet_workspace/results_task02 \
  TASK02_CUDA_VISIBLE_DEVICES=0 \
  TASK02_NUM_GPUS=1 \
  bash pumengyu/task02/run_task02_stage2.sh nnUNetTrainer_MedNeXt_MHA_MoE_Task02_Replay_K05
```

#### Replay K=10

```bash
TASK02_CLEAR_FAILED=1 \
  TASK02_RESULTS_ROOT=/home/PuMengYu/nnUNet_workspace/results_task02 \
  TASK02_CUDA_VISIBLE_DEVICES=0 \
  TASK02_NUM_GPUS=1 \
  bash pumengyu/task02/run_task02_stage2.sh nnUNetTrainer_MedNeXt_MHA_MoE_Task02_Replay_K10
```

#### Replay K=70

```bash
TASK02_CLEAR_FAILED=1 \
  TASK02_RESULTS_ROOT=/home/PuMengYu/nnUNet_workspace/results_task02 \
  TASK02_CUDA_VISIBLE_DEVICES=0 \
  TASK02_NUM_GPUS=1 \
  bash pumengyu/task02/run_task02_stage2.sh nnUNetTrainer_MedNeXt_MHA_MoE_Task02_Replay_K70
```

## 正式实验矩阵

| 实验 | 目的 | 训练状态 | 正式评估状态 |
|---|---|---|---|
| HCC-only K=1 | 只用少量 HCC 标注适配 | 未开始 | 未开始 |
| HCC-only K=3 | 只用少量 HCC 标注适配 | 已完成 300 epoch；best epoch 6 | 已完成；HCC/IRCADb/LiTS |
| HCC-only K=5 | 只用少量 HCC 标注适配 | 未开始 | 未开始 |
| HCC-only K=10 | 只用少量 HCC 标注适配 | 未开始 | 未开始 |
| Replay K=1 | 适配 HCC 并保持 LiTS | 未开始；旧尝试已清理 | 未开始 |
| Replay K=3 | 适配 HCC 并保持 LiTS | 未开始 | 未开始 |
| Replay K=5 | 适配 HCC 并保持 LiTS | 未开始 | 未开始 |
| Replay K=10 | 适配 HCC 并保持 LiTS | 未开始 | 未开始 |
| Replay K=70 | 70 例 HCC upper bound | 未开始 | 未开始 |

## 补充说明：研究定义

将 Dataset003_Liver 的 source-only 肝肿瘤分割模型适配到新目标中心 HCC-TACE-Seg。目标中心只有少量带标注 CT，因此比较 HCC-only 少样本微调与 HCC+LiTS replay，并在固定 HCC 测试集上检验目标中心效果，同时观察源域保持情况。

这是 few-shot supervised domain adaptation，不是 Dataset003 与 HCC 的直接联合训练，也不用于证明联合训练负迁移。

## 补充说明：固定数据协议

- HCC：70 adaptation / 10 validation / 21 fixed test；test 不进入训练、采样或 checkpoint 选择；
- LiTS：92 train / 13 validation / 26 fixed test；
- 少样本：K=1/3/5/10；K=70 是 replay upper bound；
- source：`nnUNetTrainer_MedNeXt_MHA_MoE`、Dataset003_Liver、`3d_fullres`、fold 0、source `checkpoint_best.pth`；
- 训练：300 epoch × 250 updates；每 3 epoch 做一次轻量 patch validation；
- HCC-only：按 HCC-val 选择 best；
- Replay：先要求 LiTS-val 不低于 source baseline 绝对值 0.02 以内，再按 HCC-val 选择 best；
- IRCADb：外部固定测试域，不参与训练期 checkpoint 选择。

## 补充说明：评估口径

- `Liver Dice`：全部固定测试病例平均；
- `Tumor Dice / Recall / Precision / FDR`：只在 GT 有肿瘤病例上平均；
- `Overall = (Liver Dice(all cases) + Tumor Dice(GT-positive cases)) / 2`；
- LiTS、IRCADb、HCC 分域计算，不把两个外部域再平均；
- 无肿瘤病例单独报告 FP 率、FPV 和误报病例；没有阴性病例时写 `N/A`；
- nnUNet 原始 `summary.json` 保留作为追溯证据，论文表格使用 PMY-LT-v1。

## 补充说明：实验完成定义

每个方法、每个数据域必须同时有：预期病例的预测 NIfTI、`summary.json`、人类可读报告、`test_viz/` 实际 PNG，以及 dataset/trainer/fold/checkpoint/source provenance。只完成训练或只完成推理，都不能写成完整实验完成。
