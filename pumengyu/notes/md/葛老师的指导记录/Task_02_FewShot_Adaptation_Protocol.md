# Task 02：冻结 Few-shot Adaptation 实验协议

## 任务定位

**论文目标：SCI 二区**

本任务不训练新模型，也不设计新模块。目标只有一个：

> **冻结后续所有 few-shot target-domain adaptation 方法必须遵守的统一实验协议。**

只有协议先固定，后续比较才具有可信度、可复现性和论文解释力。

---

## 1. 研究场景

- **Source domain**：LiTS / Dataset003_Liver
- **Target domain**：HCC-TACE-Seg
- **冻结 source model**：修复后 `MedNeXt_MHA_MoE`；它是固定底座，不是本 Task 的待比较变量
- **目标问题**：当目标域只能提供极少量带标注 CT 时，能否通过少量适配修复目标域肿瘤漏检与严重失败

### 1.1 冻结的 source checkpoint

Task 02 的全部方法必须从下列同一份 source-only checkpoint 出发：

```text
trainer:     nnUNetTrainer_MedNeXt_MHA_MoE
dataset:     Dataset003_Liver
configuration: 3d_fullres
fold:        0
checkpoint:  checkpoint_best.pth (epoch 951)
path:        /home/PuMengYu/nnUNet_workspace/results_v2_baseline_retrain_20260902/Dataset003_Liver/nnUNetTrainer_MedNeXt_MHA_MoE__nnUNetPlans__3d_fullres/fold_0/checkpoint_best.pth
SHA256:      3910ea7375656b84f14055bcd1cc2310ffd85e343d9dc0a33577aaf60eb13b81
```

该 checkpoint 使用修复后的 MoE 路由提交规则：每个 optimizer iteration 在 backward/checkpoint 重算完成后，先聚合 DDP 全局专家负载，再只提交一次路由 bias/EMA 更新。它不是修复前的历史 MoE 权重。

它已完成 source-only 三域完整评价：LiTS `0.8541`、3D-IRCADb-01 `0.8360`、HCC-TACE-Seg `0.6368` Overall；三域预测病例数为 `26/26 / 20/20 / 21/21`，并均具有 summary、文本报告、PNG 与来源记录。

---

## 2. 数据隔离规则

HCC-TACE-Seg 数据必须严格分成：

1. **Adaptation pool**：只用于 few-shot 适配，从中抽取 `k=1/3/5/10` 个带标注病例。
2. **Validation set**：只用于模型选择、early stopping、checkpoint selection，不进入训练。
3. **Fixed test set**：固定 21 例，永远不能参与训练、参数选择或 checkpoint 选择，所有方法都必须在同一批 21 例上最终评估。

> **Test set 一旦固定，不得因为模型结果不好而重新抽样或更换。**

### 2.1 已冻结的 HCC 70/10/21 划分

Task 02 沿用既有的 HCC `70/10/21` 划分；不重新洗牌，不从原 test 集抽取 few-shot 病例。

| 部分 | 病例数 | Task 02 中的用途 |
|---|---:|---|
| train | 70 | adaptation pool；仅从此处抽取 `k=1/3/5/10` 例进行目标域适配训练 |
| val | 10 | 仅用于适配过程的 checkpoint 选择、early stopping 与预先冻结的模型选择规则；不参与梯度更新 |
| test | 21 | 原有固定 test；仅用于最终一次评价，不参与训练、模型选择、超参数选择或 few-shot 抽样 |

唯一的划分真源为：

```text
/home/PuMengYu/nnUNet_workspace/preprocessed/Dataset013_HCCReferencedCT/
split_info_701020_stratified_v2.json
```

- 数据集：`Dataset013_HCCReferencedCT`；版本：`701020_stratified_v2`；原始划分 seed：`42`；
- 已核验：`70 + 10 + 21 = 101`，train/val/test 两两交集均为空；
- 该文件冻结时的 SHA256：`c09672362f2c75933cc30b926b618d5bcac52ff13aedfa31a050261cb3551c32`；
- 固定 test 的 21 例为：`HCC_003, HCC_011, HCC_020, HCC_021, HCC_025, HCC_026, HCC_028, HCC_047, HCC_052, HCC_055, HCC_058, HCC_059, HCC_060, HCC_068, HCC_071, HCC_078, HCC_081, HCC_082, HCC_083, HCC_093, HCC_102`。

该 test 集过去已经用于 source-only 外部评估与失败分析。因此它不应被描述为“从未看过的盲测集”；但从 Task 02 起，任何适配 epoch、学习率、方法取舍和 few-shot 病例选择都不得参考它。它的规范身份是**固定 held-out final test**。

---

## 3. 标注预算

固定四档：

```text
k = 1, 3, 5, 10
```

其中 `k` 表示可用于目标域适配的带标注 3D CT 病例数。

---

## 4. Few-shot 抽样规则

考虑到单次训练成本，Task 02 第一阶段固定每个标注预算只使用 **1 个预先确定的 few-shot split**：

| 标注预算 | 固定 split 数 | sampling seed |
|---|---:|---|
| `k=1` | 1 | `20260921` |
| `k=3` | 1 | `20260922` |
| `k=5` | 1 | `20260923` |
| `k=10` | 1 | `20260924` |

因此，Direct Full Fine-tuning 共运行 4 个固定的 few-shot 训练任务。Source-only 不需要 few-shot 抽样；Full-target Upper Bound 固定使用同一 adaptation pool 的全部 70 例，也不需要 few-shot 抽样。

这里必须区分两类随机性：

- **sampling seed**：决定从 70 例 adaptation pool 中抽到哪些病例；上表中每个 k 的 seed 必须在训练前生成对应的唯一 split 文件并冻结；
- **training seed**：决定微调中的数据顺序、增强与其他训练随机性。Task 02 第一阶段固定为 `20260920`，对所有方法、所有 split 一致；网络权重始终从已冻结 source checkpoint 加载。训练配置还必须记录所有实际影响随机性的 seed/确定性设置。

Task 02 的目标是以可承受的成本先得到 `k=1/3/5/10` 的标注预算曲线，不量化病例抽样或训练随机性。每个 k 的结果只代表其预先固定的病例组合；正式论文必须明确这一边界，不得将单次结果包装为跨 split 稳定性结论。若后续新方法在关键预算上显示明确价值，可另行预注册并只复验关键预算（例如 k=3、k=10），不能根据 test 结果临时挑选有利 split。

### 4.1 肿瘤负荷分层规则（冻结）

few-shot 病例按既有 HCC `70/10/21` 正式划分所使用的 **corrected tumor foreground ratio** 分层，而不是按原始图像体积、病灶数或未校正的 `tumor/liver ratio` 分层：

```text
corrected tumor foreground ratio = tumor_voxels / (liver_voxels + tumor_voxels)
```

该口径的病例级真源为：

```text
pumengyu/notes/data/hcc_split_701020_stratified_v2_cases.csv
```

它与冻结的 `split_info_701020_stratified_v2.json` 使用同一统计口径。分层阈值固定为：

| 层 | corrected tumor foreground ratio |
|---|---|
| tiny | `< 0.03` |
| small | `0.03 ≤ ratio < 0.15` |
| medium | `0.15 ≤ ratio < 0.60` |
| extreme | `≥ 0.60` |

70 例 adaptation pool 中有 tiny/small/medium/extreme 分别为 `21/24/23/2` 例；其中 `HCC_065`、`HCC_075` 是既有质量记录中的 review/extreme 病例。它们过去被排除出固定 21 例 test，是为了避免在小 test 集中由异常高肿瘤负荷病例主导评价；但它们仍是 HCC target-center 的真实训练病例。

因此，Task 02 的 few-shot 候选池固定为**全部 70 例**，包括 `HCC_065`、`HCC_075`。不得根据病例“极端”“困难”或预期提升幅度把病例从 adaptation pool 删除；是否进入某个 k 的训练，只由下表的预先冻结分层名额和固定随机抽样决定。Full-target Upper Bound 同样使用全部 70 例。

各预算的层内名额由 70 例的 `21:24:23:2` 比例按最大余数法分配，固定为：

| 标注预算 | tiny | small | medium | extreme | 合计 |
|---|---:|---:|---:|---:|---:|
| `k=1` | 0 | 1 | 0 | 0 | 1 |
| `k=3` | 1 | 1 | 1 | 0 | 3 |
| `k=5` | 1 | 2 | 2 | 0 | 5 |
| `k=10` | 3 | 4 | 3 | 0 | 10 |

由于 extreme 仅占 2/70，按比例分配后当前 `k=1/3/5/10` 的 extreme 名额均为 0；这表示它们没有被当前小标注预算**抽中**，而不是被从候选池排除。若未来新增更大的 k，必须按同一 `21:24:23:2` 比例重新计算名额，不能人为删除或强行插入 extreme 病例。

生成具体病例时，先按 case ID 升序排列每个层的 eligible cases，再按 `tiny → small → medium → extreme` 的固定顺序，使用本节开头的对应 `sampling seed` 和 `numpy.random.default_rng(seed)` 在每层无放回抽取规定名额。生成后的 JSON 内的 case ID、ratio、层名、seed、候选池来源文件和 SHA256 必须全部写入；一经生成不得因 validation 或 test 结果替换。

要求：

- 使用固定随机种子；
- 严格遵守上表的肿瘤负荷分层名额；
- 保存每次抽样的病例 ID；
- 所有方法使用完全相同的 few-shot split。

每个方法最终报告每个 k 的单次结果、对应病例 ID、tumor burden statistics 和固定 seed；不得因 test 结果更换该 split。

每个 split 文件至少记录：

```text
split_id
seed
k
case_ids
tumor burden statistics
```

Task 02 已冻结的具体名单位于：

```text
pumengyu/notes/data/task02_fewshot_splits/task02_hcc_k01_seed20260921.json
pumengyu/notes/data/task02_fewshot_splits/task02_hcc_k03_seed20260922.json
pumengyu/notes/data/task02_fewshot_splits/task02_hcc_k05_seed20260923.json
pumengyu/notes/data/task02_fewshot_splits/task02_hcc_k10_seed20260924.json
```

---

## 5. 两阶段共享模型适配

Task 02 研究的不是只部署在 HCC 的孤立本地模型，而是一个在新医院适配后仍应保留
LiTS 源域能力的**共享模型**。因此训练分为严格隔离的两个阶段。

### 5.1 Stage 1：source pretraining

对一个确定的网络结构，先只用 LiTS 的 92 例 train 训练；LiTS 13 例 val 用于 source
checkpoint 选择。此阶段 HCC 的 70/10/21 例均不得参与。

当前 Task 02 的网络结构保持已冻结的 MHA+MoE 底座不变，因此第 1 节指定的
`checkpoint_best.pth` 已是可直接使用的 Stage-1 source checkpoint，不重新训练。
如果未来修改网络结构，则该新结构必须独立从头完成一次 LiTS 1000-epoch source
pretraining，并记录新的 checkpoint provenance；不能把旧结构 checkpoint 直接称作它的
source initialization。

### 5.2 Stage 2：few-shot adaptation

每个 `k=1/3/5/10`、每种适配策略都从**同一个**冻结 Stage-1 source checkpoint 独立
启动；不得将 `k=1` 的训练接续为 `k=3`，也不得从一种策略的末尾继续另一种策略。

Stage 2 的 HCC validation 固定使用 10 例、LiTS validation 固定使用 13 例；每 3 epoch
在固定随机种子下每域合计运行 50 个 validation step（DDP 时各卡均分），计算并记录两域的
**patch-level Tumor Dice**，用于自动
更新 best checkpoint。训练期 validation 不生成完整病例预测、报告或 PNG。HCC test 21 例、
LiTS test 26 例和 IRCADb 20 例始终不参与训练或选择。

### 5.3 固定对照与主方法

固定：

```text
Source-only
vs
HCC-only Full Fine-tuning
vs
HCC + LiTS replay Full Fine-tuning
vs
Full-target Upper Bound (HCC 70 + LiTS replay)
```

第一阶段不更换已冻结的 MHA+MoE 底座，也不额外加入 Adapter、参数高效模块、冻结策略
或新的路由机制。先建立 HCC 适配、LiTS 保持和三域最终测试的基线，再讨论新的网络模块。

### 5.3.1 Source-only

不使用任何 HCC 训练数据，直接用冻结的 MHA+MoE source checkpoint 评估目标域。

作用：

> 作为适配前基线，回答“问题有多严重”。

### 5.3.2 HCC-only Full Fine-tuning（遗忘对照）

使用 `k=1/3/5/10` 个目标域标注病例，对冻结 MHA+MoE 的全部可训练参数进行全参数微调。

作用：

> 作为最直接的 target-only adaptation baseline，量化“只追求 HCC 时会遗忘多少 LiTS”。

### 5.3.3 HCC + LiTS replay Full Fine-tuning（主方法）

使用同一个 k-shot HCC split 与 LiTS train replay 病例，对全部可训练参数做全参数微调。
训练目标的总形式固定为：

```text
L = L_HCC_seg + lambda * L_LiTS_replay
```

两项均为有标签的分割损失；HCC 项学习新医院，LiTS replay 项约束源域遗忘。`lambda`、
两域 batch 交替比例和每轮训练步数必须在 Stage-2 开始前共同冻结，且所有 k 和比较方法
保持相同。LiTS test 不得出现在 replay 或 checkpoint 选择中。

Task 02 Stage-2 固定为：每个 optimizer update 依次使用一个 HCC batch 和一个 LiTS
replay batch，分别计算其 mean segmentation loss 后相加，`lambda = 1.0`；即一个 update
的目标为 `L_HCC_seg + 1.0 * L_LiTS_replay`。每个 epoch 固定 250 个 optimizer updates，
不得因为 k 较小而减少 HCC 项的出现频率。

作用：

> 回答“能否用少量 HCC 标注提升目标域，同时保持 LiTS 能力”。

### 5.3.4 Full-target Upper Bound

从同一冻结 MHA+MoE checkpoint 出发，使用目标域 adaptation pool 的全部 70 例与 LiTS
train replay，进行全参数充分适配。

作用：

> 估计“如果目标域标注资源充足，理论上可以恢复到什么程度”。

---

## 6. 主要评价终点

论文主要终点优先关注肿瘤端：

- Tumor Dice
- Tumor Recall
- Severe Failure Rate

严重失败的主定义固定为：

```text
Severe Failure = per-case Tumor Dice < 0.20
```

这对应临床上肿瘤分割几乎不可用的病例，能覆盖 `Dice = 0` 的完全漏检以及
`Dice = 0.1–0.2` 的极低质量预测。另行报告 `Tumor Dice = 0` 的
**complete-miss rate**，但它是 Severe Failure 的组成性次级描述，不替代主定义。
不得根据 validation 或 test 结果改变阈值。

---

## 7. 辅助评价指标

建议记录：

- Tumor Precision
- FPV
- FNV
- Liver Dice
- trainable parameters
- 总参数量
- 训练时间
- GPU memory
- inference cost（如后续有需要）

注意：

> HCC 固定 test 21 例均为肿瘤阳性，因此 no-tumor FP rate 应报告为 `N/A`。

---

## 8. 模型选择规则

已冻结：

- 不使用 early stopping；每次训练均跑满预先固定的最大 epoch；
- 每 **3 epoch** 对固定 HCC validation 10 例和 LiTS validation 13 例在固定随机种子下各做合计 50 个 validation step（DDP 时各卡均分）的轻量 validation，记录两域 patch-level Tumor Dice；
- HCC validation patch-level Tumor Dice 是目标域主指标。对于 HCC-only Full Fine-tuning（遗忘对照），直接选择该指标最高的 checkpoint；同分取较早 epoch；
- 对 HCC + LiTS replay Full Fine-tuning 和 Full-target Upper Bound，LiTS Dice 不与 HCC Dice 混成加权分数。先要求 LiTS validation patch-level Tumor Dice 相对冻结 source checkpoint 的绝对下降不超过 `0.02`，再在所有满足约束的候选中选择 HCC validation patch-level Tumor Dice 最高者；同分取较早 epoch。若没有任何 Stage-2 checkpoint 满足 LiTS 约束，则选择 source checkpoint 并明确报告“无可接受的适配 checkpoint”；
- LiTS test 不得参与 replay、调参或 checkpoint 选择；
- 21 例 HCC held-out test 不参与训练、调参或 checkpoint 选择，只对每次训练选出的唯一 best checkpoint 做最终一次正式评估。

实现与运行记录必须包含：

- Stage-2 总 epoch、学习率、optimizer、scheduler、weight decay、patch size、数据增强和
  两域数据加载器的实际 batch size，必须由实现写入每次运行的 `task02_config.json`；
- Stage-2 的 300 epoch、`LR=1e-3`、每 epoch 250 optimizer updates 已冻结；其余训练项沿用
  当前 MHA+MoE source trainer（SGD with momentum 0.99 and Nesterov、PolyLR、weight decay
  `3e-5`、计划 patch `128×128×128`），除非在正式运行前以新的协议修订明确替换；
- 每 3 epoch 的 HCC/LiTS validation 及上述 checkpoint selection rule 必须由代码执行，
  不得依赖人工查看日志后手动挑选。

绝对禁止：

> 根据 21 例 test set 的结果挑 checkpoint。

---

## 9. 统一训练预算

所有方法必须尽量保持：

- 相同 source checkpoint
- 相同 few-shot split
- 相同 validation set
- 相同数据增强
- 相同训练 epoch 上限
- 相同优化器与 scheduler
- 按第 8 节预注册的 checkpoint 选择规则（HCC-only 遗忘对照与 replay 主方法的规则差异必须保留）
- 相同 inference pipeline
- 相同最终 test set

只有方法本身允许变化。

---

## 10. 最终流程

```text
LiTS source pretraining (92 train / 13 val)
        ↓
fixed source checkpoint
        ↓
for each k: independent Stage-2 run from the same source checkpoint
   ├── k=1
   ├── k=3
   ├── k=5
   └── k=10
        ↓
HCC-only FT  vs  HCC + LiTS replay FT
        ↓
HCC val 10 + LiTS val 13
(checkpoint selection / retention monitoring)
        ↓
one selected checkpoint: LiTS test 26 + IRCADb test 20 + HCC test 21
        ↓
three-domain Dice / Recall / Severe Failure
        ↓
annotation cost / trainable params / training time
```

---

## 11. 本任务交付物

完成 Task 02 时，需要提交：

### A. 数据划分说明

明确：

- adaptation pool 病例数
- validation 病例数
- fixed test 病例数
- 三部分病例 ID
- 是否存在任何重叠

### B. Few-shot split 文件

例如：

```text
task02_hcc_k01_seed20260921.json
task02_hcc_k03_seed20260922.json
task02_hcc_k05_seed20260923.json
task02_hcc_k10_seed20260924.json
```

### C. 训练配置

明确：

- source checkpoint
- epoch
- optimizer
- lr
- scheduler
- augmentation
- checkpoint selection
- early stopping rule

并明确记录：每个 k 的一个 sampling seed、固定 training seed `20260920`，以及各随机库/数据加载器的 seed 设置。

### D. Severe Failure 定义

必须在正式评估前确定。

---

## 12. 验收标准

Task 02 通过必须满足三个条件：

### ① 无数据泄漏

Test set 完全隔离，不参与训练与模型选择。

### ② 比较公平

不同方法使用同一 source model、同一 few-shot split、同一 validation 和 test protocol。

### ③ 能回答论文问题

该协议必须能够真正回答：

> **在有限目标域标注下，直接 fine-tuning 到底能恢复多少性能，以及它还剩下什么缺口？**

---

## 13. 当前禁止事项

在 Task 02 验收前：

- 不更换冻结的 MHA+MoE 底座，不改动其路由状态提交规则
- 不额外训练 Adapter、参数高效模块、冻结策略或新的路由机制
- 不更换 test set
- 不根据测试结果调整 few-shot 样本
- 不先写“某方法一定更优”
- 不把方法假设写成实验结论

---

## 14. Task 02 在论文证据链中的位置

```text
现实问题
→ source-only target-domain failure
→ Task 02：冻结 few-shot adaptation protocol
→ Direct Fine-tuning baseline
→ 找到 baseline 的真实缺口
→ 提出方法假设
→ 设计高针对性的 adaptation method
→ 多次抽样 / 多 seed 验证
→ Failure recovery analysis
→ Thesis
```

---

## 15. 一句话任务定义

> **先把赛道画死，再让所有方法在同一赛道上比赛。**
