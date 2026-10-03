# Task 02 Stage-2 代码设计

本目录只实现 Task 02 的 Stage-2 few-shot shared-model adaptation。它不取代一般的
nnU-Net trainer，也不修改任何现有历史 trainer 或结果目录。

## 固定入口与运行模式

训练始终以 `Dataset003_Liver` 的 plans 启动，因为网络结构和 source checkpoint 均来自
该数据集。HCC 的预处理病例在运行时以独立 loader 加入，不能改写 `splits_final.json`。

计划提供 9 个明确 trainer 入口（每个入口对应一个新的独立结果目录）：

| 训练目的 | k | trainer 名称模式 |
|---|---|---|
| HCC-only 遗忘对照 | 1/3/5/10 | `nnUNetTrainer_MedNeXt_MHA_MoE_Task02_HCCOnly_Kxx` |
| HCC + LiTS replay 主方法 | 1/3/5/10 | `nnUNetTrainer_MedNeXt_MHA_MoE_Task02_Replay_Kxx` |
| HCC 70 + LiTS replay upper bound | 70 | `nnUNetTrainer_MedNeXt_MHA_MoE_Task02_Replay_K70` |

`Source-only` 不新建训练；它直接使用已经冻结的 source checkpoint。

## 代码组件

### 1. `splits.py`

职责：读取并严格审计 `notes/data/task02_fewshot_splits/*.json`。

必须检查：

- JSON 中的 k、seed、名额和 case 数一致；
- 所有 case 属于 HCC 70-case adaptation pool；
- 与 HCC val 10、test 21 无交集；
- metadata CSV 的 SHA256 和协议中写入值一致；
- HCC-only/replay 同 k 使用同一个 JSON；
- 运行时复制该 JSON 到结果目录，并写入运行 manifest。

### 2. `stage2_mixin.py`

职责：提供共用的 Stage-2 trainer 行为。

**严格 source 初始化**

- 只接受协议冻结 checkpoint：MHA+MoE、Dataset003、fold 0、epoch 951、指定 SHA256；
- 直接严格加载所有 network weights，包含 segmentation head；不恢复 source optimizer、scheduler、epoch 或 logger；
- 新实验的 optimizer/scheduler 从零开始，`current_epoch=0`；
- 拒绝命令行的 `-pretrained_weights`，避免 nnU-Net 的默认 loader 跳过 segmentation head 且无法审计来源。

**双数据 loader**

- LiTS loader 仅含固定 LiTS train 92 例；
- HCC loader 仅含当前 k JSON 的病例，或 K70 的全部 70 例；
- HCC val 10、HCC test 21、LiTS val 13、LiTS test 26 都有显式集合断言，不能进入 gradient loader；
- 两个 loader 均使用同一 source 3D-fullres patch、增强和深度监督配置；
- DDP 下每个 rank 使用各自的 augmenter，所有 rank 仍消费一对 HCC/LiTS batch。

**训练更新**

```text
HCC-only:  one HCC batch -> L_HCC_seg -> one optimizer update
Replay:    one HCC batch + one LiTS batch
           -> L_HCC_seg + 1.0 * L_LiTS_replay
           -> one optimizer update
```

每个 epoch 固定 250 个 optimizer updates，总共 300 epoch。两个 loss 都使用同一个
nnU-Net Dice+CE 深度监督 loss，且以各自 batch 的 mean loss 参与相加。MoE deferred
routing-state update 必须只在两个 loss 均反向传播后、optimizer step 前提交一次。

两卡 DDP 下先在 `no_sync()` 内完成 HCC forward/backward，再执行 LiTS
forward/backward 并同步累积梯度，最后只执行一次 optimizer step。损失权重和更新次数
不变，避免两个同时保留的 reentrant checkpoint 图重复触发 DDP ready hook。

Replay 的专家计数由 `routing.py` 在 HCC、LiTS 各自首次 forward 后分别保存并累加。
backward 完成后恢复这份合计值，再走原有 DDP 汇总与 deferred commit；这样梯度检查点
重算不会覆盖或重复计入原始统计，也不改变单域训练的 MoE 行为。
CPU 回归入口：`python -m unittest pumengyu.task02.test_replay_routing -v`。
回归覆盖双域计数、梯度一致性、两种 checkpoint 重算、清空与跨 update 隔离、eval 和
单域行为，以及模拟跨 rank 汇总。另有真实 CPU 两进程 Gloo 测试：混合 checkpoint
小模型复现旧路径 ready-twice 错误，新路径连续两次更新的梯度和参数与全局平均参考一致。
这些检查不替代完整模型的两卡 GPU smoke。

**轻量验证与常规 nnU-Net validation 的边界**

Task02 不调用基础 trainer 的“每 epoch、LiTS 单域”的 validation loop，也不会产生它的
`checkpoint_best.pth` 或进度 PNG。每 **3 epoch** 执行一次两个独立的轻量 patch 验证：

- HCC val 固定只来自 canonical 70/10/21 中的 10 个 val ID；
- LiTS val 固定只来自 `split_info_712.json` 中的 13 个 val ID；
- 每域合计跑 50 个固定种子的 validation step（DDP 时各卡均分），汇总肿瘤 Dice；不读任何 test 输入；
- 结果仅追加到 `<fold_0>/task02_validation_history.jsonl`，并写入本次的 case-ID 清单、epoch 和选择结果；
- `<fold_0>/task02_config.json` 会固化实际 patch、每卡 batch、DDP world size、optimizer/scheduler、增强来源与选择规则；
- 它不生成 NIfTI、`summary.json`、文本报告或 `test_viz/` PNG，也不是论文最终数值。

`checkpoint_task02_best.pth` 由这份轻量记录按预注册规则自动更新；它与基础 trainer 的
`checkpoint_best.pth` 完全分离。

最终只对 `checkpoint_task02_best.pth` 进行 LiTS / IRCADb / HCC 三域正式测试；那一次才必须生成
全部预测、`summary.json`、文本报告、`test_viz/` PNG 与 provenance。

### 3. `selection.py`

职责：离线复核训练期的 `task02_validation_history.jsonl` 与 `task02_selection.json` 是否满足
预注册规则；它不训练、不预测、不生成最终评估产物，也不以人工看曲线代替规则。

```text
HCC-only:
    最高 HCC-val mean Tumor Dice；同分取更早 epoch。

Replay / K70 upper bound:
    先筛选 LiTS-val Tumor Dice >= source-val Tumor Dice - 0.02；
    再取 HCC-val mean Tumor Dice 最高；同分取更早 epoch。
    无候选满足时，明确选择 source checkpoint，状态为 no_acceptable_adaptation.
```

训练器只生成/更新 `checkpoint_task02_best.pth` 和 `task02_selection.json`；不覆盖基础 trainer
的 `checkpoint_best.pth`，从而避免 LiTS 单域 pseudo Dice 与 Task02 双域选择混淆。

### 4. `run_task02_stage2.sh`

用户手动执行的单实验入口。脚本不得自动运行 9 次，也不得并行抢占 GPU；它也**不会**在训练结束后
自动运行正式 test。每次运行必须：

- 设置 `nnUNet_raw`、`nnUNet_preprocessed`、`nnUNet_results` 到新的 Task02 结果根；
- 指定一个明确 trainer、fold 0、GPU 和结果标签；
- 只启动一个全新的 Stage-2 run，拒绝覆盖已有 fold 目录；
- 禁止 `--c`、`--val` 与 `-pretrained_weights`；严格 source loader 已在 trainer 内部执行；
- 训练成功后自动检查 `task02_manifest.json`、`task02_validation_history.jsonl`、
  `task02_selection.json`，并对选中的 checkpoint 自动运行三域正式评估。

### 5. `final_evaluation.py`

正式评估入口。它先复核 `task02_manifest.json`、JSONL 选择历史和 selected checkpoint SHA，之后才用
`checkpoint_task02_best.pth` 跑 LiTS 26、IRCADb 固定清单和 HCC 21 三域。每域必须同时有 NIfTI、
`summary.json`、文本报告、`test_viz/` PNG；末尾会写 `task02_final_artifact_audit.json`，任一域不完整
即报错，不得汇报为完成。

在该入口经过用户的短程 smoke run 前，不把任何训练期 lightweight validation 当作正式三域结果。

## 用户执行入口（一次只运行一个）

```bash
# 先做唯一的 smoke check：3 epoch × 2 updates，独立目录，不能报告为实验结果。
TASK02_CUDA_VISIBLE_DEVICES=0 bash pumengyu/task02/run_task02_smoke.sh

# 训练一个独立 Stage-2 run；示例为 replay, k=3。
TASK02_CUDA_VISIBLE_DEVICES=0 \
  bash pumengyu/task02/run_task02_stage2.sh \
  nnUNetTrainer_MedNeXt_MHA_MoE_Task02_Replay_K03

# 训练成功后会自动检查选择记录并运行正式三域评估；如需单独补评估，仍可执行：
python -m pumengyu.task02.final_evaluation \
  --trainer nnUNetTrainer_MedNeXt_MHA_MoE_Task02_Replay_K03 \
  --gpu 0 \
  --model_results_root /home/PuMengYu/nnUNet_workspace/results_task02

# 独立空间距离权重消融（MHA + MoE，复用同一 source checkpoint）
TASK02_CUDA_VISIBLE_DEVICES=1 \
  bash pumengyu/task02/run_task02_stage2.sh \
  nnUNetTrainer_MedNeXt_MHA_MoE_Distance_Task02_Replay_K05
```

先使用 `--dry_run` 只核验 provenance 并打印正式评估命令；它不会启动预测。

## 与旧代码的关系

不能直接复用 `HCCMixTrainingMixin`：它将病例放入同一个随机 loader，既不能保证每 update
的一对 HCC/LiTS batch，也无法完成 Task02 的 loss 定义、双域 validation 和选择规则。

不能使用 nnU-Net 的 `-pretrained_weights`：该函数会有意跳过 segmentation head，且不验证
Task02 source SHA。Task02 应改用严格的全权重 source loader。

## 先后实施顺序（均由用户手动执行）

1. 实现 split 审计和 strict source loader；
2. 实现 HCC-only trainer，验证数据隔离和 3-epoch HCC/LiTS lightweight validation；
3. 实现 replay pair-update，再确认 MoE routing state 一次提交；
4. 实现自动选择与运行 manifest；
5. 用户执行短程 smoke run；
6. smoke 通过后，用户依次启动 9 个正式 Stage-2 run；
7. 用户对唯一选中的 checkpoint 执行 LiTS / IRCADb / HCC 三域最终评估。

本 README 是设计文件，不代表上述实现或任一训练已完成。
