# 两阶段 FP 抑制方案

> **划分**：7/1/2（训练 / 验证 / 测试），Dataset003_Liver，nnUNet v2 3d_fullres  
> **核心问题**：LiTS 训练集 131 case，其中 13 个无肿瘤 case（9.9%），  
> nnUNet 端到端训练对无肿瘤 case 大量误报（Baseline 误报率 100%）。  
> 最后更新：2026-06-01

---

## 零、Stage2 目标的精确定义

Stage2 的目标是**无肿瘤 case 的整体误报**，不是一般性 FP 降低。

| FP 类型 | 来源 | Stage2 能否解决 |
|---------|------|----------------|
| **无肿瘤 case 误报** | 模型对整个无肿瘤病例乱报（GT 全零但预测有肿瘤） | ✅ Stage2 核心目标 |
| **有肿瘤 case 溢出 FP** | 分割边界不准、血管截面被过分割 | ❌ 不是 Stage2 目标 |

> 这个区别在论文里必须写清楚：Stage2 提供的是对"整体是否有肿瘤"的上下文判断，而非对所有 FP 逐 CC 精细过滤。

---

## 一、动机数据（已验证）

测试集 3 个无肿瘤 case（n=26），Baseline 全部误报：

| case | FP 体素数 | 判断 |
|------|---------|------|
| liver_41 | **32,055 vox** | 大 FP，体积阈值无法清除 |
| liver_89 | 321 vox | 小 FP |
| liver_91 | 248 vox | 小 FP |

> **体积阈值方案已废弃**：对全部训练集 GT 连通域统计，最小 CC 仅 **1 voxel**。  
> 任何正数阈值均会误删真阳性，纯体积阈值不可用，Stage2 学习上下文特征是唯一出路。

---

## 二、数据划分与阶段数据流

| 阶段 | 使用数据 | 无肿瘤 case 处理 | 目的 |
|------|---------|----------------|------|
| **Stage1 训练** | 训练集中有肿瘤 case | **过滤，不参与训练** | 故意让模型从未见过"应全黑"输入 |
| **Stage1 推理** | 全部 131 case（训练+验证+测试） | 正常推理，不过滤 | 无肿瘤 case 产生大量 FP 概率图 |
| **Stage2 训练** | 全部训练集（92 case） | 保留，过采样 3× | Stage2 学习区分真肿瘤与 FP |
| **验证/测试** | 验证集 + 测试集 | 完整保留 | 评估 FP 抑制效果 |

> Stage1 过滤无肿瘤的逻辑：过滤不是为了"干净的训练"，而是**故意制造 FP**——Stage1 从未见过应全黑的 case，推理时大量误报；Stage2 的训练数据因此包含大量 FP 概率图，才有足够的负样本可学。

---

## 三、当前进度

| 步骤 | 描述 | 状态 |
|------|------|------|
| **Step 1** | 实现 `Tr_Stage1_TumorOnly`，训练 1000 epoch | ✅ **已完成** |
| **Step 2** | Stage1 对全部 131 case 推理，保存概率图 | ✅ **已完成** |
| **Step 3** | ~~Stage1 + 体积阈值后处理（消融基线）~~ | ❌ **已废弃**（GT min CC = 1 vox，阈值会误伤 TP） |
| **Step 4** | `Tr_Stage2_FPSup` v1 已训（Loss 有误）→ v2 修复后待训 | ⚠️ **v2 待训** |
| **Step 5** | Stage2 推理 + 生成最终报告 | 🔲 待做 |

---

## 四、各步骤详细说明

### Step 1（已完成）：Stage1 训练结果

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

### Step 2（下一步）：Stage1 全量推理，保存概率图

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

### ~~Step 3：Stage1 + 体积阈值后处理（消融基线）~~ ❌ 已废弃

**废弃原因**：对 LiTS 训练集全部 GT 肿瘤连通域统计，最小 CC 为 **1 voxel**。  
设置任何 `min_size > 0` 的阈值都会误删真阳性小病灶，消融实验意义不存在。  
liver_41（32,055 vox）的 FP 量级本身也已排除纯阈值方案的可行性。

> 结论：体积阈值不是可用的 FP 抑制手段，Stage2 的必要性无需消融即可确立。

---

### Step 4：实现 `Tr_Stage2_FPSup`

> 参照 Peng et al., MICCAI 2022（LRM）设计

**输入**：3 通道（CT + Stage1 肿瘤概率图 + Stage1 二值图（thresh=0.5））

**设计理由**：
- 概率图提供置信度渐变
- 二值图给出明确边界
- 两者互补，比只用概率图效果更好（参考 LRM 论文）

**采样策略**：无肿瘤 case 过采样 3×

---

#### ⚠️ v1 问题复盘（2026-06-04）

v1 训完 500 epoch（26.4h），验证集结果：肝脏 Dice=**0.11**（应 ≥0.90），肿瘤 Dice=0.6401（与 Stage1 持平，无提升）。

**根因**：`_Stage2LossWrapper` 只优化肿瘤通道（MSE+BCE on softmax tumor prob），肝脏通道从未被 Loss 约束。肝脏分割对肿瘤有强解剖约束（肿瘤 ⊂ 肝脏），网络不认识肝脏边界，无法通过上下文区分 TP vs FP，Stage2 退化为普通 Stage1。

**修复**（已提交 `pumengyu/mixins.py`）：
- `_build_loss()` → 直接 `super()._build_loss()`，使用标准 Dice+CE，肝脏+肿瘤同时优化
- 新增 `_load_stage1_weights()`：用 Stage1 `checkpoint_final.pth` 热启动，第一层 conv 1 通道 → 3 通道（新通道初始化为 0），网络无需重新学肝脏/肿瘤
- `num_epochs` 500 → **250**（热启动可提前收敛，约 13h）

---

#### v2 训练命令

```bash
# 单卡（~13h）
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train Dataset003_Liver 3d_fullres 0 -tr Tr_Stage2_FPSup

# 双卡 DDP（~7h，第二块 4090D 空闲时推荐）
CUDA_VISIBLE_DEVICES=0,1 nnUNetv2_train Dataset003_Liver 3d_fullres 0 -tr Tr_Stage2_FPSup
```

训练开头日志验证（确认修复生效）：
```
[Stage2Init] 扩展 ...: [32, 1, 3, 3, 3] → [32, 3, 3, 3, 3]
[Stage2FPSup] Loss = Dice+CE (liver+tumor，与 nnUNetTrainer 一致)
```

---

### Step 5：生成最终报告

Stage2 训练完成后，触发 `AutoInternalTestMixin` 自动推理测试集并生成 `test_report_custom.txt`。

手动补跑报告（如需）：

```bash

cd /home/PuMengYu/nnUNet && bash pumengyu/notes/sh/regen_reports_v2.sh

```

---

## 五、评估指标（分组报告）

| 指标 | 有肿瘤 case | 无肿瘤 case |
|------|-----------|-----------|
| Tumor Dice | ✓ | N/A |
| FPV（mm³） | ✓ | **✓ 最重要** |
| FNV（mm³） | ✓ | N/A |
| 误报率 | N/A | ✓ |
| FP CC 体积分布 | N/A | ✓ |

**最终目标**：Stage2 在不损失有肿瘤 Dice 的前提下，将无肿瘤误报率从 66.7%（SizeOversampleV2）降至 0% 或接近 0%。

---

## 六、风险与应对

| 风险 | 应对 |
|------|------|
| 13 个无肿瘤 case 太少，Stage2 学习不稳定 | 过采样 3× + 强数据增强 |
| Stage2 过度抑制，损伤有肿瘤 Dice | 监控 Dice 回落，必要时降低无肿瘤过采样倍数 |
| Stage1 概率图文件管理复杂 | 统一存到 `stage1_softmax/` 子目录，路径在 trainer 里固定 |
| liver_41 FP 体积大（32k vox），Stage2 能否区分 | 关键验证点，若失败考虑加入 3D 上下文特征 |

---

## 七、文件索引

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
