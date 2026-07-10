# FP 抑制与 CopyPaste

## 两阶段 FP 抑制方案


#### 两阶段 FP 抑制方案

> **划分**：7/1/2（训练 / 验证 / 测试），Dataset003_Liver，nnUNet v2 3d_fullres  
> **核心问题**：LiTS 训练集 131 case，其中 13 个无肿瘤 case（9.9%），  
> nnUNet 端到端训练对无肿瘤 case 大量误报（Baseline 误报率 100%）。  
> 最后更新：2026-06-01

---

##### 零、Stage2 目标的精确定义

Stage2 的目标是**无肿瘤 case 的整体误报**，不是一般性 FP 降低。

| FP 类型 | 来源 | Stage2 能否解决 |
|---------|------|----------------|
| **无肿瘤 case 误报** | 模型对整个无肿瘤病例乱报（GT 全零但预测有肿瘤） | ✅ Stage2 核心目标 |
| **有肿瘤 case 溢出 FP** | 分割边界不准、血管截面被过分割 | ❌ 不是 Stage2 目标 |

> 这个区别在论文里必须写清楚：Stage2 提供的是对"整体是否有肿瘤"的上下文判断，而非对所有 FP 逐 CC 精细过滤。

---

##### 一、动机数据（已验证）

测试集 3 个无肿瘤 case（n=26），Baseline 全部误报：

| case | FP 体素数 | 判断 |
|------|---------|------|
| liver_41 | **32,055 vox** | 大 FP，体积阈值无法清除 |
| liver_89 | 321 vox | 小 FP |
| liver_91 | 248 vox | 小 FP |

> **体积阈值方案已废弃**：对全部训练集 GT 连通域统计，最小 CC 仅 **1 voxel**。  
> 任何正数阈值均会误删真阳性，纯体积阈值不可用，Stage2 学习上下文特征是唯一出路。

---

##### 二、数据划分与阶段数据流

| 阶段 | 使用数据 | 无肿瘤 case 处理 | 目的 |
|------|---------|----------------|------|
| **Stage1 训练** | 训练集中有肿瘤 case | **过滤，不参与训练** | 故意让模型从未见过"应全黑"输入 |
| **Stage1 推理** | 全部 131 case（训练+验证+测试） | 正常推理，不过滤 | 无肿瘤 case 产生大量 FP 概率图 |
| **Stage2 训练** | 全部训练集（92 case） | 保留，过采样 3× | Stage2 学习区分真肿瘤与 FP |
| **验证/测试** | 验证集 + 测试集 | 完整保留 | 评估 FP 抑制效果 |

> Stage1 过滤无肿瘤的逻辑：过滤不是为了"干净的训练"，而是**故意制造 FP**——Stage1 从未见过应全黑的 case，推理时大量误报；Stage2 的训练数据因此包含大量 FP 概率图，才有足够的负样本可学。

---

##### 三、当前进度

| 步骤 | 描述 | 状态 |
|------|------|------|
| **Step 1** | 实现 `Tr_Stage1_TumorOnly`，训练 1000 epoch | ✅ **已完成** |
| **Step 2** | Stage1 对全部 131 case 推理，保存概率图 | ✅ **已完成** |
| **Step 3** | ~~Stage1 + 体积阈值后处理（消融基线）~~ | ❌ **已废弃**（GT min CC = 1 vox，阈值会误伤 TP） |
| **Step 4** | `Tr_Stage2_FPSup` v1 已训（Loss 有误）→ v2 修复后待训 | ⚠️ **v2 待训** |
| **Step 5** | Stage2 推理 + 生成最终报告 | 🔲 待做 |

---

##### 四、各步骤详细说明

###### Step 1（已完成）：Stage1 训练结果

测试集结果（n=26）：

| 指标 | Stage1_TumorOnly | Baseline | 说明 |
|------|:----------------:|:--------:|------|
| 肝脏 Dice | 0.9502 | 0.9340 | 略好 |
| 肿瘤 Dice | 0.6592 | 0.6542 | 略好 |
| 无肿瘤误报率 | **100% (3/3)** | 100% (3/3) | 符合预期——故意让它误报 |
| Overall | 0.8047 | 0.7941 | |
| Tumor FPV均值 | 9,603 mm³ | 16,139 mm³ | FP 总量比 Baseline 小（有肿瘤 case 的 FP 更少） |

> 无肿瘤误报率 100% 完全符合设计预期，不是问题。

---

###### Step 2（下一步）：Stage1 全量推理，保存概率图

对全部 131 个训练 case 推理，保存 softmax 概率图（`.npz`）：
  bash /home/PuMengYu/nnUNet/pumengyu/notes/sh/stage1_predict.sh
  

```bash

  bash /home/PuMengYu/nnUNet/pumengyu/notes/sh/stage1_predict.sh
  
RESULTS_FOLDER=/home/PuMengYu/nnUNet_workspace/results_v2 \
nnUNetv2_predict \
  -i /home/PuMengYu/nnUNet_workspace/raw/Dataset003_Liver/imagesTr \
  -o /home/PuMengYu/nnUNet_workspace/results_v2/Dataset003_Liver/Tr_Stage1_TumorOnly__nnUNetPlans__3d_fullres/fold_0/stage1_softmax \
  -d 3 -c 3d_fullres -tr Tr_Stage1_TumorOnly -f 0 \
  --save_probabilities
```

输出：每个 case 一个 `.npz`，包含 softmax 概率图（3 channel：背景/肝脏/肿瘤）。

Stage2 使用其中的 channel=2（肿瘤概率）作为额外输入通道。

---

###### ~~Step 3：Stage1 + 体积阈值后处理（消融基线）~~ ❌ 已废弃

**废弃原因**：对 LiTS 训练集全部 GT 肿瘤连通域统计，最小 CC 为 **1 voxel**。  
设置任何 `min_size > 0` 的阈值都会误删真阳性小病灶，消融实验意义不存在。  
liver_41（32,055 vox）的 FP 量级本身也已排除纯阈值方案的可行性。

> 结论：体积阈值不是可用的 FP 抑制手段，Stage2 的必要性无需消融即可确立。

---

###### Step 4：实现 `Tr_Stage2_FPSup`

> 参照 Peng et al., MICCAI 2022（LRM）设计

**输入**：3 通道（CT + Stage1 肿瘤概率图 + Stage1 二值图（thresh=0.5））

**设计理由**：
- 概率图提供置信度渐变
- 二值图给出明确边界
- 两者互补，比只用概率图效果更好（参考 LRM 论文）

**采样策略**：无肿瘤 case 过采样 3×

---

####### ⚠️ v1 问题复盘（2026-06-04）

v1 训完 500 epoch（26.4h），验证集结果：肝脏 Dice=**0.11**（应 ≥0.90），肿瘤 Dice=0.6401（与 Stage1 持平，无提升）。

**根因**：`_Stage2LossWrapper` 只优化肿瘤通道（MSE+BCE on softmax tumor prob），肝脏通道从未被 Loss 约束。肝脏分割对肿瘤有强解剖约束（肿瘤 ⊂ 肝脏），网络不认识肝脏边界，无法通过上下文区分 TP vs FP，Stage2 退化为普通 Stage1。

**修复**（已提交 `pumengyu/mixins.py`）：
- `_build_loss()` → 直接 `super()._build_loss()`，使用标准 Dice+CE，肝脏+肿瘤同时优化
- 新增 `_load_stage1_weights()`：用 Stage1 `checkpoint_final.pth` 热启动，第一层 conv 1 通道 → 3 通道（新通道初始化为 0），网络无需重新学肝脏/肿瘤
- `num_epochs` 500 → **250**（热启动可提前收敛，约 13h）

---

####### v2 训练命令

```bash
#### 单卡（~13h）
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train Dataset003_Liver 3d_fullres 0 -tr Tr_Stage2_FPSup

#### 双卡 DDP（~7h，第二块 4090D 空闲时推荐）
CUDA_VISIBLE_DEVICES=0,1 nnUNetv2_train Dataset003_Liver 3d_fullres 0 -tr Tr_Stage2_FPSup
```

训练开头日志验证（确认修复生效）：
```
[Stage2Init] 扩展 ...: [32, 1, 3, 3, 3] → [32, 3, 3, 3, 3]
[Stage2FPSup] Loss = Dice+CE (liver+tumor，与 nnUNetTrainer 一致)
```

---

###### Step 5：生成最终报告

Stage2 训练完成后，触发 `AutoInternalTestMixin` 自动推理测试集并生成 `test_report_custom.txt`。

手动补跑报告（如需）：

```bash

cd /home/PuMengYu/nnUNet && bash pumengyu/notes/sh/regen_reports_v2.sh

```

---

##### 五、评估指标（分组报告）

| 指标 | 有肿瘤 case | 无肿瘤 case |
|------|-----------|-----------|
| Tumor Dice | ✓ | N/A |
| FPV（mm³） | ✓ | **✓ 最重要** |
| FNV（mm³） | ✓ | N/A |
| 误报率 | N/A | ✓ |
| FP CC 体积分布 | N/A | ✓ |

**最终目标**：Stage2 在不损失有肿瘤 Dice 的前提下，将无肿瘤误报率从 66.7%（SizeOversampleV2）降至 0% 或接近 0%。

---

##### 六、风险与应对

| 风险 | 应对 |
|------|------|
| 13 个无肿瘤 case 太少，Stage2 学习不稳定 | 过采样 3× + 强数据增强 |
| Stage2 过度抑制，损伤有肿瘤 Dice | 监控 Dice 回落，必要时降低无肿瘤过采样倍数 |
| Stage1 概率图文件管理复杂 | 统一存到 `stage1_softmax/` 子目录，路径在 trainer 里固定 |
| liver_41 FP 体积大（32k vox），Stage2 能否区分 | 关键验证点，若失败考虑加入 3D 上下文特征 |

---

##### 七、文件索引

| 文件 | 内容 |
|------|------|
| 本文件 | 两阶段方案计划 |
| `pumengyu/trainers/trainer.py` | Trainer 实现位置 |
| `pumengyu/mixins.py` | 所有 Mixin 实现 |
| `results_v2/.../Tr_Stage1_TumorOnly.../fold_0/` | Stage1 模型权重 |
| `results_v2/.../Tr_Stage1_TumorOnly.../fold_0/stage1_softmax/` | Stage1 概率图（Step 2 完成后） |
| `pumengyu/notes/sh/regen_reports_v2.sh` | 报告批量生成脚本 |
| `pumengyu/notes/md/实验配置记录.md` | 所有实验配置汇总 |

---

*最后更新：2026-06-04（Step 4 v1 问题复盘+修复，v2 待训）*


---

## 内在难度驱动的 CopyPaste 方案（无阈值 / 防泄露版）


#### 内在难度驱动的 CopyPaste 方案（无阈值 / 防泄露版）

> 一句话：不再手设密度/体积阈值，改为给每个肿瘤算一个**只来自图像+标注**的连续"难度分"，越难分辨（overlap 越高）的肿瘤被 CopyPaste 粘得越多。难度信号全程不碰任何模型预测，因此与"用 5-fold CV 当最终指标"不冲突、无泄露。

---

##### 0. 背景与硬约束

- 数据集：MSD Task03 Liver（肝脏肿瘤）。
- **官方测试集无标签** → 论文里所有指标只能靠**训练集上的 5-fold 交叉验证（CV）**。这是 MSD / nnUNet 领域的通行做法，合规、审稿人接受。
- 但由此产生一条铁律：

  > **"驱动训练的信号" 和 "最终汇报的那批折" 是同一批数据。**
  > 没有第二个带标签的测试集兜底。所以只要驱动信号沾了任何预测结果，就等于拿汇报集的答案反推训练 —— 这就是"对着答案改权重"的泄露。

整个方案的设计目标，就是从结构上让这种泄露**不可能发生**。

---

##### 1. 核心决定

**走"纯内在难度"路线**：难度只从**图像 + Ground Truth 标注**计算，永不使用任何模型/验证/CV 的预测输出。

为什么这条恰好破解上面的约束：

> 它把"驱动训练的东西"和"最终汇报的东西"**彻底切断依赖**。
> 模型的 CV 结果从此**只当输出**（最后报一次），永远**不当输入**（不参与决定粘谁）。
> 既然难度的计算里压根没出现过任何验证预测，那"在同一批折上汇报 CV"就不存在答案被喂回去的问题。

附带好处：
- **无手设阈值**，符合导师要的"像 nnUNet 一样自配置"。
- 对**全新图像**也能直接打分（不依赖某个已训练模型）。
- 比多数 MSD 论文更严谨（很多工作是盯着 CV 错误案例去设计方法的，严格说有"方法选择泄露"），这点可作为论文的严谨性卖点。

---

##### 2. ⚠️ 必须避免的（重点）

先把"泄露"钉死一个定义，后面三条都按它判断：

> **泄露 = 用来评判你模型的那批数据（这里就是 5-fold 的各验证折）的信息，偷偷进到了训练里。**

###### 坑 1：素材库隔离 —— 真泄露，实现层面，必须卡死

CopyPaste 训练第 *k* 折时，能被粘贴的肿瘤**素材池，只能来自第 k 折的训练那部分（其余 4 折）**。

- ❌ 错误：把第 k 折**验证集**里的肿瘤外观/标注，粘进了第 k 折的训练图。
- 为什么这是真泄露：验证折肿瘤的像素和标注直接进了训练，模型在评测时等于"见过答案"，CV 分数虚高。
- ✅ 正确：每折训练时，paste 源池 = 该折的训练 case 集合，**显式排除该折验证 case**。
- 落地检查：CopyPaste 的数据加载逻辑里，源肿瘤列表必须按当前 fold 过滤；写完后用一个断言/日志打印"本折素材池 case 数 + 是否与验证集有交集（必须为 0）"。

###### 坑 2：方法选择泄露 —— 旋钮不能盯着 CV 调

就算难度是内在的，只要方案的超参数（粘几次、难度分的指数、KDE 带宽……）是靠"跑一遍 → 看 CV 分 → 改参数 → 再跑 → 调到 CV 最高"得到的，**答案又通过你的手绕回来了**。这等价于在 5-fold 上做了一次隐式的过拟合。

修法（nnUNet 哲学：参数从数据自动推，不手调）：

- **自由参数越少，泄露面越小。** 能写死的就写死。
- 指数固定 = 1（直接取消这个自由度），而不是 0.5/2 之间试。
- 粘贴次数 ∝ 稀有度的倒数，用公式定，而不是手挑。
- KDE 带宽用 **Scott / Silverman 规则**自动算，不手调。
- 论文里如实写："超参数为先验固定（a priori），未在 CV 上调参。"
- 如果**实在**要选某个参数：唯一干净的办法是**嵌套交叉验证**（见坑 3 注），而不是在汇报的那层 CV 上选。

###### 坑 3：模型难度（OOF Dice）的使用边界

OOF Dice（5-fold 交叉验证给每个训练 case 打的分）是**真**难度，但它来自预测，**不能用来驱动训练**（否则违反第 0 节铁律）。它只允许出现在一个地方：

- ✅ **一次性"体检"**：用它验证"内在难度分高的肿瘤，是不是真的 Dice 低"。报一个相关系数（如 Spearman）当**描述性发现**，证明内在难度这个代理靠谱。
- 验证完即**锁死**，真正喂给 CopyPaste 的只有内在难度。
- 关键区别：OOF 在这里用来**检验代理是否合理**，不用来**决定每个样本的权重**。前者是分析，后者才是泄露。

> 注（嵌套交叉验证）：若将来非要用模型难度驱动训练且不泄露，理论上的正解是 nested CV——外层留一折汇报，内层只在外层训练数据里再切 CV 算难度，难度永不碰外层验证折。代价是 5×5≈25 次训练，对 nnUNet 基本跑不起，**本方案不采用**，仅备查。

---

##### 3. 内在难度分怎么算（已敲定 v1）

**决定：`difficulty = rank_norm(overlap)` —— 只用 overlap 一个特征。**

- `overlap` = 肿瘤 HU 直方图与肝脏 HU 直方图的重叠度（[0,1]），衡量"光靠密度能不能把肿瘤从肝里分出来"= 病灶可显著性（conspicuity）。
- overlap 越高 → 越接近等密度 → 越难分辨 → 难度越高。
- `rank_norm` = 按 overlap 在全体 case 里的排名归一到 [0,1]（nnUNet 风格百分位，可复现、对量纲鲁棒）。
- 最终权重 `weight = 0.05 + 0.95 × difficulty`（FLOOR=0.05 防止最易样本概率塌成 0）。

**为什么只用 overlap（而不是之前设想的"可分性 × 稀有度"几何平均）：**
- overlap 是这批数据里**与 Dice 最相关的单一内在特征**（项目早期相关分析已验证）；
- cohen's d / |contrast| 与 overlap 同属"可分性"，**冗余**；
- **体积长尾**已由 CopyPaste 的 `CP_MAX_LOCS`（库只收小肿瘤）+ `SmallTumorOversampleMixin` 结构性处理，**不必再混 rarity**；
- 几何平均的"必须又难又稀有"会把"常见但难分辨"（等密度，占失败约 56%）的 case 砍到 floor —— 正好误伤最该救的那批；
- 旋钮越少、泄露面越小（坑 2）：单特征、零可调权重，按原理选定，不为凑 ρ 反复试。

**体检结果（一次性，仅诊断，不驱动训练 → 坑 3）：** 难度权重 vs OOF Dice 的 Spearman ρ = **-0.295**（p≈0.001，n=118），显著负相关 → 内在难度是合理代理。ρ 不高（解释约 8% 方差）是正常天花板：overlap 只覆盖"可分性"一个维度，抓不到"尺寸 / 标注噪声"维度（后者由别的机制兜底，见第 8 节 backlog）。

> 实现：`pumengyu/tools/data_analysis/compute_difficulty.py` → 产出 `notes/实验结果分析/difficulty.json`（{case: weight}）。

---

##### 4. 接到 CopyPaste —— 难度加权采样

把连续难度分接到粘贴概率，**无阈值**：

```
肿瘤 i 被选作粘贴源的概率  ∝  w_i = difficulty_i
（或更稳的秩版本: w_i ∝ 1 / rank_i）
```

- 难的（如极低密度大肿瘤、极小肿瘤）自动被反复粘，简单的几乎不粘。
- 这就是"粘那些真正难以分辨的"的实现。
- 严格遵守坑 1：候选源池仅限当前 fold 的训练 case。

---

##### 5. 长尾问题

数据有两条长尾，加权采样同时治两条：

- **特征长尾**：体积跨多个数量级（约 38 → 348986 mm³），极低/极高对比度是少数。
- **性能长尾**：多数 case Dice 高，一条尾巴拖到 Dice→0。

学界对长尾的标准解法就是**难度/稀有度加权重采样**，而 CopyPaste 正是实现它的工具。因此本课题可一句话概括：**用"内在难度加权 CopyPaste"对抗肝脏肿瘤数据的长尾**。

---

##### 6. ⚠️ 无肿瘤 / 假阳必须一起评估（别只盯召回）

- 已知坑：CopyPaste 曾把无肿瘤 case 的误报从 33% 推到 100%（3/3）。原因是只往"召回"使劲。
- 当前分析的 `load_dice` 把 `dice_cancer = None` 的无肿瘤 case 直接跳过了 —— **必须纳入**。
- 无肿瘤 case 的指标不是 Dice，而是**预测出的假肿瘤体积 / 是否误报**（specificity）。
- 评估必须**双轴**：尾部召回/Dice ↑ 的同时，健康肝脏上的假阳没有爆。
- 原则性缓解：CopyPaste 也保留/混入"干净"负样本，别让模型觉得"到处都有瘤"。

---

##### 7. 进度

**已完成：**
1. ✅ 难度公式敲定 = `rank_norm(overlap)`（第 3 节）。
2. ✅ 难度脚本 `compute_difficulty.py` → `difficulty.json`；`--check` 做 OOF 体检（ρ=-0.295）。
3. ✅ 在线难度加权 CopyPaste：`DifficultyCopyPasteMixin` + trainer `nnUNetTrainer_CopyPaste_Diff`
   （仅覆盖"抽哪个 ROI"，建库/粘贴/无肿瘤跳过复用父类；素材池仍来自 `do_split` 的 tr_keys → 坑 1 保持）。

**进行中：**
4. ⏳ fold_4 A/B 对照：`nnUNetTrainer_CopyPaste_v2`（均匀）vs `nnUNetTrainer_CopyPaste_Diff`（难度加权），同折同数据，唯一变量 = 粘谁。有效再扩 5 折。
5. ☐ 评估脚本双轴化：尾部 Dice + 无肿瘤假阳（第 6 节）；并核对 `batch['keys']` 非空，否则无肿瘤跳过会静默失效。

> 备注：已弃用离线方案，改在线 CopyPaste。

---

##### 8. 下一阶段 backlog（不打断当前 A/B，做完这轮再动）

这些都属于"继续去手设阈值、向 nnUNet 自配置靠拢"，且全是 intrinsic（只看图像/GT，零泄露）。

###### B1. 小肿瘤阈值自动化
现状有两个对不上的魔法数字：`SMALL_TUMOR_THRESH_LOCS=6000`（重复过采样）、`CP_MAX_LOCS=5000`（粘贴库上限）。
- 直接版：用肿瘤大小分布的**分位数**（如 33%/50%）当切点，数据自适应、无魔法数字；
- 两个阈值用同一套规则统一掉。

###### B2. 尺寸维度并入难度分（去阈值终态）
把"尺寸稀有度"作为难度的**第二个连续维度**，和 overlap（可分性）合成**一个连续重要性分**，同时驱动 oversample 和 copy-paste —— `6000/5000` 等硬阈值彻底消失。这是导师要的"自配置"终态。
- 注意：避免重新引入坑 2 的可调权重 —— 两维如何合成需 a priori 定（或用"秩相加"这种无旋钮方式）。

###### B3. 粘贴 vs 重复 的消融
`CopyPasteMixin`（像素层、增多样性）与 `SmallTumorOversampleMixin`（case 层、增频率）在"提升小肿瘤曝光"上**部分重叠**，叠加会放大无肿瘤误报。做 重复-only / 粘贴-only / 两者 的消融，厘清各自贡献并控制 specificity。

> 纪律：以上都不在当前 A/B 阶段改 —— `6000 / 5000 / 重复次数` 在 v2 与 Diff 两臂相同，是对照公共项，改了会搅浑"均匀 vs 难度"的对比。


---
