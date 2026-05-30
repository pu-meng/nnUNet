# 两阶段 FP 抑制方案

> **划分**：7/1/2（训练 / 挑选 / 测试），Dataset003_Liver，nnUNet v2 3d_fullres  
> **核心问题**：LiTS 训练集 131 case，其中 13 个无肿瘤 case（9.9%），  
> nnUNet 端到端训练严重偏向预测"有肿瘤"，无肿瘤 case 大量 FP。

---

## 零、Stage2 目标的精确定义

Stage2 的目标是**无肿瘤 case 的整体误报**，不是一般性 FP 降低。两类 FP 的区别：

| FP 类型 | 来源 | Stage2 能否解决 |
|---------|------|----------------|
| **无肿瘤 case 误报** | 模型对整个无肿瘤病例乱报（GT 全零但预测有肿瘤） | ✅ Stage2 核心目标 |
| **有肿瘤 case 溢出 FP** | 分割边界不准、血管截面被过分割，属于精度问题 | ❌ 不是 Stage2 目标 |

> 这个区别在论文里必须写清楚，否则审稿人会问"为什么不直接降 FPV"。  
> Stage2 提供的是：对无肿瘤 case 学习"整体无肿瘤"的上下文判断，而非对所有 FP 逐 CC 做精细过滤。

---

## 一、动机数据（已验证）

| case | FP 体素数 | 判断 |
|------|---------|------|
| liver_41 | **32,055** | 大 FP，接近真实小肿瘤体积，体积阈值会误伤 TP |
| liver_89 | 321 | 小 FP，体积阈值可清除 |
| liver_91 | 248 | 小 FP，体积阈值可清除 |

> Test 集 3 个无肿瘤 case 全部误报（100%），Baseline Overall=0.7941。  
> liver_41 的 FP 量级决定了纯体积阈值不够，需要学习上下文特征来区分。

---

## 一·五、数据划分与阶段数据流（可重复性关键）

> **核心原则**：7/1/2 划分在整个实验中固定不变，所有对比实验共享同一划分，保证控制变量、结果可重复。

| 阶段 | 使用数据范围 | 无肿瘤 case 处理 | 目的 |
|------|------------|----------------|------|
| **Stage1 训练** | 7（训练集）中有肿瘤 case | **过滤掉，不参与训练** | 故意让模型从未见过"应全黑"的输入 |
| **Stage1 推理** | 全部 131 case（7+1+2） | 正常推理，**不过滤** | 让 Stage1 对无肿瘤 case 大量误报，产生 FP 概率图 |
| **Stage2 训练** | 7（训练集，全部 case） | 保留，过采样 3x | 让 Stage2 见到 FP 样本，学会区分真肿瘤与 FP |
| **验证/测试报告** | 1（验证集）和 2（测试集） | 完整保留，计 FPV | 评估 FP 抑制效果 |

> **Stage1 过滤无肿瘤的逻辑**：过滤不是为了"干净的训练"，而是**故意制造 FP**——Stage1 从未见过应全黑的 case，推理时对无肿瘤 case 乱报；Stage2 的训练数据因此包含大量 FP 概率图，才有足够的负样本可学。这是两阶段方案成立的数据层面前提，缺少这一步则 Stage2 无从区分。

---

## 二、实验设计（4 个对比）

| 实验 | Trainer | 训练数据 | 输入通道 | 说明 |
|------|---------|---------|---------|------|
| **Baseline** | `nnUNetTrainer_Baseline` | 全 131 case | 1（CT） | 已完成 |
| **Stage1** | `Tr_Stage1_TumorOnly` | 118 有肿瘤 case | 1（CT） | 故意不见无肿瘤，产生最多 FP |
| **Stage1 + 阈值** | Stage1 推理 + 后处理 | — | — | 消融：体积阈值能否解决 |
| **Stage2（proposed）** | `Tr_Stage2_FPSup` | 全 131 case | **3（CT + Stage1 概率图 + Stage1 二值图）** | 学习区分真肿瘤和 FP，Loss=MSE+CE |

---

## 三、实施步骤

### Step 0：生成当前报告（前置）

```bash
cd /home/PuMengYu/nnUNet
bash pumengyu/notes/sh/regen_reports_v2.sh
```

目的：拿到含 CC 分析节的报告，确认 liver_41 FP 连通域体积。

---

### Step 1：实现 Tr_Stage1_TumorOnly

**核心改动**：在 `get_training_transforms` 或 `do_split` 阶段过滤掉无肿瘤 case。

实现位置：`pumengyu/trainers/trainer.py`，新建 `Tr_Stage1_TumorOnly` 类。

关键逻辑：
```python
# 重写 do_split，只保留有肿瘤的 case
def do_split(self):
    super().do_split()
    # 过滤 train_keys：去掉 gt 全为 0 的 case
    self.train_keys = [k for k in self.train_keys if self._has_tumor(k)]
```

`_has_tumor(k)` 读 `gt_segmentations/{k}.nii.gz`，判断是否含 label=2。

训练命令：
```bash
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train Dataset003_Liver 3d_fullres 0 -tr Tr_Stage1_TumorOnly
```

---

### Step 2：Stage1 推理（全 131 case）

```bash
# validation 集推理
CUDA_VISIBLE_DEVICES=0 nnUNetv2_predict \
  -i /home/PuMengYu/nnUNet_workspace/raw/Dataset003_Liver/imagesTr \
  -o /home/PuMengYu/nnUNet_workspace/results_v2/Dataset003_Liver/Tr_Stage1_TumorOnly__nnUNetPlans__3d_fullres/fold_0/stage1_softmax \
  -d Dataset003_Liver -c 3d_fullres -tr Tr_Stage1_TumorOnly -f 0 \
  --save_probabilities   # ← 保存 softmax 概率图，作为 Stage2 输入
```

输出：每个 case 一个 `.npz`（softmax 概率图），取 channel=2（tumor）拼入 Stage2 输入。

---

### Step 3：Stage1 + 体积阈值后处理（消融基线）

运行现有脚本，加 `--min_tumor_size` 参数：

```bash
cd /home/PuMengYu/nnUNet && python pumengyu/tools/analyasis/eval_fold_report.py \
  --val_dir .../Tr_Stage1_TumorOnly.../fold_0/validation \
  --gt_dir $GT --img_dir $IMG --no_vis \
  --min_tumor_size 100
```

结果写入 `report_custom.txt` 的"后处理对比"节，自动对比前后 Dice 和 FPV。

---

### Step 4：实现 Tr_Stage2_FPSup

> **⚡ 改进点（源自参考论文 Peng et al., MICCAI 2022）**
>
> 原始方案：2 通道输入（CT + Stage1 概率图）+ Dice+CE Loss  
> **改进后**：3 通道输入（CT + Stage1 概率图 + Stage1 二值图）+ MSE+CE Loss
>
> 改进理由：
> - **加入二值图**：概率图提供置信度渐变，二值图（阈值 0.5）给出明确边界，两者互补；论文 LRM 同时使用两者，比只用概率图效果更好。
> - **MSE 替代 Dice**：对 FP 区域（预测值高但 GT=0）惩罚更直接——MSE Loss = (pred−0)²，梯度随预测值线性增大；Dice 关注整体重叠，对局部 FP 不敏感。此为论文中 FP 抑制的核心机制。

**核心改动**：输入通道从 1 改为 3（CT + Stage1 tumor 概率图 + Stage1 二值图）。

```python
class Tr_Stage2_FPSup(nnUNetTrainer):

    @property
    def num_input_channels(self):
        return 3  # CT + stage1_prob + stage1_binary

    def get_plain_dataloaders(self, ...):
        # 数据加载时额外读 stage1_softmax/{case}_tumor_prob.nii.gz
        # stage1_binary = (stage1_prob > 0.5).float()
        # 拼接顺序：[ct, stage1_prob, stage1_binary]
        ...
```

采样策略：无肿瘤 case 过采样 3x（在 `do_split` 里重复 key）。

**Loss：MSE + CE**（对齐论文 LRM 的 Loss_LRM，不用 Dice）：

$$\text{Loss}_{Stage2} = \frac{1}{N}\sum_{i=1}^{N}\left\{(g_i - p_i)^2 - [g_i \log p_i + (1-g_i)\log(1-p_i)]\right\}$$

训练命令：
```bash
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train Dataset003_Liver 3d_fullres 0 -tr Tr_Stage2_FPSup
```

#### 消融对比安排

| 版本 | 输入通道 | Loss | 备注 |
|------|---------|------|------|
| Stage2-v1 | CT + prob（2ch） | Dice + CE | 原始方案，作为消融 baseline |
| **Stage2-v2（主版本）** | CT + prob + binary（3ch） | **MSE + CE** | 对齐参考论文，预期 FPV 下降更显著 |

> 先训练 Stage2-v2（主版本），若效果不及预期再训 v1 做消融。

---

### Step 5：生成报告

```bash
# 在 regen_reports_v2.sh 的 TRAINERS 数组里追加：
Tr_Stage1_TumorOnly
Tr_Stage2_FPSup

# 然后执行
bash pumengyu/notes/sh/regen_reports_v2.sh
```

---

## 四、评估指标（分组报告）

| 指标 | 有肿瘤 case | 无肿瘤 case |
|------|-----------|-----------|
| Tumor Dice | ✓ | N/A |
| FPV（mm³） | ✓ | **✓ 最重要** |
| FNV（mm³） | ✓ | N/A |
| 误报率 | N/A | ✓ |
| FP CC 体积分布（Q1/Q2） | N/A | ✓ |

> 目标：Stage2 在不损失有肿瘤 Dice 的前提下，将无肿瘤误报率从 100% 降低。

---

## 五、风险与应对

| 风险 | 应对 |
|------|------|
| 13 个无肿瘤 case 太少，Stage2 不稳定 | 过采样 3x + 强数据增强（翻转 / 强度扰动） |
| Stage2 过度抑制，损伤有肿瘤 Dice | 监控 Dice 回落，必要时降低无肿瘤过采样倍数 |
| Stage1 概率图文件管理复杂 | 统一存到 `stage1_softmax/` 子目录，路径硬编码在 trainer 里 |
| 工程量超预期 | Stage2 先用轻量 2D UNet 验证概念，再换 nnUNet 3D |

---

## 六、文件索引

| 文件 | 内容 |
|------|------|
| 本文件 | 两阶段方案计划 |
| `pumengyu/trainers/trainer.py` | 新 trainer 实现位置 |
| `pumengyu/notes/sh/regen_reports_v2.sh` | 报告批量生成脚本 |
| `pumengyu/notes/md/报告重新生成命令.md` | 单条报告命令参考 |
| `results_v2/Dataset003_Liver/` | 所有实验结果目录 |

---

*更新时间：2026-05-30（v2：参照 Peng et al. MICCAI 2022 改进 Stage2 输入+Loss）*
