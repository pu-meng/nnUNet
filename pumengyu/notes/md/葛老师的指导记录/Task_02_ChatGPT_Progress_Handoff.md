# Task 02 进展交接：Few-shot HCC Shared-Model Adaptation

> 状态（2026-09-11）：**实验协议已冻结；Stage-2 trainer 尚未实现；正式微调尚未开始。**
> 不得把“协议完成”表述成“实验结果完成”。

## 1. 当前论文问题

目标不是只在 HCC 医院部署一个孤立模型，而是：

> 新医院只有少量 HCC 标注 CT 时，如何适配共享分割模型、提高 HCC 肿瘤分割，同时避免原 LiTS 源域能力大幅下降？

这属于有 source-data access 的 **few-shot supervised target-center adaptation / replay setting**。适配阶段允许使用 LiTS 的训练病例作 replay；不得把它称为“新医院完全没有旧数据”的纯本地微调。

## 2. 已冻结 source checkpoint

- 模型：`nnUNetTrainer_MedNeXt_MHA_MoE`（修复后的 MHA+MoE）
- source：`Dataset003_Liver` / `3d_fullres` / fold 0
- checkpoint：`checkpoint_best.pth`，epoch 951
- 路径：

```text
/home/PuMengYu/nnUNet_workspace/results_v2_baseline_retrain_20260902/Dataset003_Liver/nnUNetTrainer_MedNeXt_MHA_MoE__nnUNetPlans__3d_fullres/fold_0/checkpoint_best.pth
```

- SHA256：`3910ea7375656b84f14055bcd1cc2310ffd85e343d9dc0a33577aaf60eb13b81`

它是当前 Task 02 不改网络结构时的 Stage-1 source checkpoint。若未来改变网络结构，必须先只用 LiTS train 从头完成一次新的 1000-epoch source pretraining，不能复用旧结构 checkpoint。

## 3. HCC 固定划分与 few-shot 病例

- 数据集：HCC-TACE-Seg / 内部 `Dataset013_HCCReferencedCT`；QC 纳入 101 例。
- 固定全局划分：70 adaptation pool / 10 validation / 21 held-out test；无重叠已核验。
- 规范真源：

```text
/home/PuMengYu/nnUNet_workspace/preprocessed/Dataset013_HCCReferencedCT/split_info_701020_stratified_v2.json
SHA256: c09672362f2c75933cc30b926b618d5bcac52ff13aedfa31a050261cb3551c32
```

- 固定 HCC test 21 例从 Task 02 起只允许最终评估；不能用来选 epoch、LR、方法或 few-shot split。
- 这 21 例历史上曾用于 source-only 外部测试，因此应称为“fixed held-out final test”，不能声称从未被研究者看过。

### 肿瘤负荷分层（已冻结）

使用：

```text
corrected tumor foreground ratio = tumor_voxels / (liver_voxels + tumor_voxels)
```

分层阈值：tiny `<0.03`；small `[0.03, 0.15)`；medium `[0.15, 0.60)`；extreme `>=0.60`。

70 例 pool 的层数为 `21/24/23/2`。候选池保留全部 70 例，包含 extreme 的 `HCC_065`、`HCC_075`；它们没有被人为删除。当前小 k 按比例分配时 extreme 名额为 0，表示未抽中，而非排除。

### 已冻结的单次 few-shot split

每个 k 只有一个预注册 split，不做成本过高的 5-fold / 5 repeats；结果不能包装成跨 split 稳定性结论。所有比较方法必须使用同一个 k split。

| k | sampling seed | 病例 ID |
|---:|---:|---|
| 1 | 20260921 | `HCC_005` |
| 3 | 20260922 | `HCC_086, HCC_105, HCC_091` |
| 5 | 20260923 | `HCC_006, HCC_090, HCC_105, HCC_087, HCC_094` |
| 10 | 20260924 | `HCC_006, HCC_050, HCC_104, HCC_038, HCC_043, HCC_053, HCC_090, HCC_007, HCC_029, HCC_049` |

对应 JSON 位于：

```text
pumengyu/notes/data/task02_fewshot_splits/
```

## 4. 两阶段实验设计（已冻结）

### Stage 1：source pretraining

- LiTS train 92 例训练；LiTS val 13 例选择 source checkpoint。
- HCC 70/10/21 在该阶段完全不参与。
- 当前网络未改，直接复用上方冻结 checkpoint，不重新训练。

### Stage 2：few-shot shared-model adaptation

每个 k、每种策略都从完全相同的 source checkpoint **独立新开实验**：网络权重加载 checkpoint，但 optimizer/scheduler 重新建立；不得跨 k 或跨策略续训。

训练与选择：

- 总计 300 epoch，early stopping 关闭；
- 初始 LR `1e-3`；SGD（momentum 0.99、Nesterov）、PolyLR、weight decay `3e-5`、计划 patch `128^3` 沿用 source trainer；
- 每 epoch 250 optimizer updates；
- 每 3 epoch 在 HCC val 10 例和 LiTS val 13 例上各做 50 个 patch 的轻量验证，仅记录 patch-level Tumor Dice 并自动选择 checkpoint；
- LiTS test 不参与训练、调参或 checkpoint 选择。

## 5. 固定方法与损失

1. **Source-only**：不训练，直接评估冻结 source checkpoint。
2. **HCC-only Full Fine-tuning**：每 k 只用 HCC k 例全参数微调；按最高 HCC-val mean Tumor Dice 选 checkpoint。它是“灾难性遗忘”对照。
3. **HCC + LiTS replay Full Fine-tuning（主方法）**：每次 optimizer update 使用一个 HCC batch 和一个 LiTS train replay batch，全部参数更新：

   ```text
   L = L_HCC_seg + 1.0 * L_LiTS_replay
   ```

   先要求 LiTS-val mean Tumor Dice 相对 source checkpoint 的绝对下降 `<= 0.02`，再在满足约束的 checkpoint 中选择 HCC-val mean Tumor Dice 最高者；同分取更早 epoch。若无 checkpoint 满足 LiTS 约束，则退回 source checkpoint，并报告“无可接受的适配 checkpoint”。
4. **Full-target Upper Bound**：HCC 70 例 + LiTS replay，仍从同一 source checkpoint 做全参数微调；不是 HCC 从头训练。它回答“目标域标注充足且仍需保留 LiTS 时可达到什么程度”。

所以固定底座时，正式 Stage-2 训练数为：HCC-only 的 4 次（k=1/3/5/10）+ replay 的 4 次 + 70-case upper bound 的 1 次 = **9 次**。Source-only 不需要训练。

## 6. 最终测试与指标

从每次训练选出的唯一 checkpoint 最终测试三个域：

- LiTS test 26 例：源域保持；
- IRCADb test 20 例：第三域泛化；
- HCC held-out test 21 例：few-shot target adaptation。

HCC 的主要终点：Tumor Dice、Tumor Recall、Severe Failure Rate。

```text
Severe Failure = per-case Tumor Dice < 0.20
```

另报 `Tumor Dice = 0` 的 complete-miss rate。HCC test 21 例全部为肿瘤阳性，no-tumor FP rate 必须为 `N/A`。

正式评估必须具备：全部预期 NIfTI prediction、`summary.json`、人类可读报告、`test_viz/` PNG、checkpoint/dataset/fold/provenance；缺任一项只能称为部分完成。

## 7. 当前实际完成状态与下一步

已完成：研究问题、source checkpoint、70/10/21 划分、四个 k 的病例 ID、分层规则、失败定义、两阶段设计、训练/选择规则。

尚未完成：Stage-2 trainer / mixed HCC-LiTS dataloader、每 3 epoch 双域轻量验证与自动 checkpoint 选择、短程冒烟测试、9 个正式微调、三域正式测试及其完整产物。

下一步应先实现并验证 Stage-2 流水线；短程冒烟测试必须确认：HCC test 从不进入 loader、LiTS test 从不进入 replay、每 3 epoch 的 HCC/LiTS val 均能执行、LiTS 保持约束自动保存正确 checkpoint。

## 8. 关键解释边界

- 这不是 HCC-only 本地模型，而是 source-data-access replay 的共享模型适配；论文必须如实写明。
- HCC-only FT 的用途是显示遗忘代价，不能把它与 replay 方法混为同一目标。
- IRCADb 只做最终第三域测试，不参与训练或 checkpoint 选择。
- 当前不引入 Adapter、冻结策略、额外路由或新 MoE 模块；先跑干净的 baseline / replay 证据，再决定是否需要新网络方法。

## 9. 协议主文件

[Task_02_FewShot_Adaptation_Protocol.md](Task_02_FewShot_Adaptation_Protocol.md)
