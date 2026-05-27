# Trainer 登记册

每次新增 Trainer 类，在此登记一条。  
**格式**：类名 | 数据集 | 核心思路（一句话）| 关键超参 | 实验目录后缀

---

## Dataset003\_Liver（全 CT，联合分割肝脏+肿瘤，label1=肝脏 label2=肿瘤）

### nnUNetTrainer
- **核心**：标准 nnUNet baseline，CE+Dice loss，无任何改动
- **实验目录**：`nnUNetTrainer__nnUNetPlans__3d_fullres/`
- **定义位置**：nnUNet 原生（无需自定义）

---

### nnUNetTrainer\_UFL
- **核心**：在 CE+Dice 基础上叠加 AsymmetricUnifiedFocalLoss（仅针对肿瘤类），自动平衡极小肿瘤体素与大背景的梯度贡献
- **关键超参**：`UFL_LAMBDA=0.5` / **`UFL_DELTA=0.6`（偏向惩罚漏检 FN，更激进地召回肿瘤）** / `UFL_GAMMA=0.2`
- **实验目录**：`nnUNetTrainer_UFL__nnUNetPlans__3d_fullres/`
- **定义位置**：`pumengyu/trainers/trainer.py` + `pumengyu/mixins.py::UnifiedFocalLossMixin`
- **训练时 mixins.py 状态**：commit `53b863c`（delta=0.6 版本）
- **备注**：用于与 CopyPaste 形成 2×2 消融（有/无 CopyPaste × 有/无 UFL）

---

### nnUNetTrainer\_UFL\_v2
- **核心**：与 UFL v1 思路相同，但 **`UFL_DELTA=0.5`（对称惩罚 FN/FP，不再偏向召回）**
- **关键超参**：`UFL_LAMBDA=0.5` / **`UFL_DELTA=0.5`** / `UFL_GAMMA=0.2`
- **实验目录**：`nnUNetTrainer_UFL_v2__nnUNetPlans__3d_fullres/`
- **定义位置**：`pumengyu/trainers/trainer.py`
- **训练时 mixins.py 状态**：commit `17b5592`（delta 从 0.6 改为 0.5）
- **与 v1 的实质差异**：`UFL_DELTA 0.6→0.5`，动机是"消除对 FN 的系统性偏置，降低无肿瘤 case 误报"（见 mixins.py 注释）；但当前 UnifiedFocalLossMixin 的类变量已是 delta=0.5，如需重跑 delta=0.6 版本，在子类中覆盖 `UFL_DELTA = 0.6` 即可，无需 revert git

### nnUNetTrainer\_UFL\_delta06
- **核心**：与 UFL_v2 代码完全相同，仅将 `UFL_DELTA` 覆盖回 0.6，用于与 UFL_v2（delta=0.5）做显式对照消融
- **关键超参**：`UFL_LAMBDA=0.5` / **`UFL_DELTA=0.6`** / `UFL_GAMMA=0.2`
- **实验目录**：`nnUNetTrainer_UFL_delta06__nnUNetPlans__3d_fullres/`
- **定义位置**：`pumengyu/trainers/trainer.py`
- **备注**：尚未跑实验；等价于 nnUNetTrainer_UFL 的干净重现版本

---

> ⭐ **重大发现（2026-05-24）**：长期以为 UFL v1 和 v2 是"同代码的干净重跑"，实际上两者 **UFL_DELTA 不同（0.6 vs 0.5）**，这才是结果差异的根本原因，而非随机性。该差异通过 `git show 53b863c:pumengyu/mixins.py` vs `git show 17b5592:pumengyu/mixins.py` 比对确认。delta=0.6 让模型更激进召回（fold_4 Dice 0.639），但也拉高了 fold_1/fold_2 的误报率；delta=0.5 更保守（fold_4 Dice 0.597）。**UFL_DELTA 是一个值得系统消融的超参**，建议后续用 `nnUNetTrainer_UFL_delta06` / `nnUNetTrainer_UFL_delta05` 这样的命名显式区分，不要再出现隐式变更。

---

### nnUNetTrainer\_CopyPaste
- **核心**：小肿瘤过采样（identifiers 重复 3x） + 在线 CopyPaste（50% 概率将小肿瘤 ROI 粘贴进其他 case 的肝脏区域）
- **关键超参**：`CP_PROB=0.5` / `CP_MAX_LOCS=5000`（小肿瘤判定上限） / `SMALL_TUMOR_REPEAT=3` / `SMALL_TUMOR_THRESH_LOCS=6000`
- **实验目录**：`nnUNetTrainer_CopyPaste__nnUNetPlans__3d_fullres/`
- **定义位置**：`pumengyu/trainers/trainer.py` + `pumengyu/mixins.py::CopyPasteMixin + SmallTumorOversampleMixin`
- **⚠️ 已知 Bug**：多连通域提取 bug——抽 ROI 时可能把多个非相邻连通域打包粘贴，导致合成样本不自然；v2 已修复

---

### nnUNetTrainer\_CopyPaste\_v2
- **核心**：与 CopyPaste v1 完全相同，修复了多连通域 bug（每次只粘贴单一连通域）
- **关键超参**：同 v1（`CP_PROB=0.5` / `CP_MAX_LOCS=5000` / `SMALL_TUMOR_REPEAT=3` / `CP_MARGIN=3`）
- **实验目录**：`nnUNetTrainer_CopyPaste_v2__nnUNetPlans__3d_fullres/`
- **定义位置**：`pumengyu/trainers/trainer.py`

---

### nnUNetTrainer\_CopyPaste\_Diff
- **核心**：用内在难度加权替代均匀随机抽 ROI，其余与 CopyPaste_v2 完全一致；唯一变量 = 粘谁
- **实验目录**：`nnUNetTrainer_CopyPaste_Diff__nnUNetPlans__3d_fullres/`
- **定义位置**：`pumengyu/trainers/trainer.py` + `pumengyu/mixins.py::DifficultyCopyPasteMixin`
- **前置条件**：需先运行 `pumengyu/tools/data_analysis/compute_difficulty.py` 生成 `difficulty.json`
- **备注**：尚未跑实验

---

### nnUNetTrainer\_CopyPasteUFL
- **核心**：CopyPaste_v1 + UFL 叠加，用于验证两者是否有正交增益
- **实验目录**：`nnUNetTrainer_CopyPasteUFL__nnUNetPlans__3d_fullres/`
- **定义位置**：`pumengyu/trainers/trainer.py`
- **备注**：尚未跑实验；⚠️ 继承 CopyPaste v1，含多连通域 bug

---

## Dataset004\_LiverTumor（肝脏 ROI 裁剪后专做肿瘤，label1=肿瘤）

> 数据来源：Dataset003 的肝脏预测裁剪出的 ROI，只分割肿瘤

### nnUNetTrainer\_TwoStage
- **核心**：训练逻辑与标准 nnUNet 相同（加小肿瘤过采样），**区别在推理**：Stage1 肝脏预测 → 裁剪 ROI → Stage2 在 ROI 内做肿瘤分割；验证结束后自动触发端到端评估
- **实验目录**：`nnUNetTrainer_TwoStage__nnUNetPlans__3d_fullres/`
- **定义位置**：`pumengyu/twostage/trainer.py`

---

### nnUNetTrainer\_BoundaryLoss
- **核心**：额外预测距离场（并行预测头从最高分辨率 decoder 特征分叉），loss = CE+Dice + L_boundary；L_boundary = mean(\|pred_dist−gt_dist\|³)；距离场 GT 在 DataLoader worker 内在线用 scipy EDT 计算
- **实验目录**：`nnUNetTrainer_BoundaryLoss__nnUNetPlans__3d_fullres/`
- **定义位置**：`pumengyu/boundary/trainer.py`
- **参考**：BATseg arXiv:2412.06507
- **特点**：需修改 DataLoader，有 CPU EDT 计算瓶颈

---

### nnUNetTrainer\_ConvBoundaryLoss
- **核心**：用固定各向异性 Laplacian 卷积核提取预测概率图和 GT one-hot 的边界响应，两者差异作为 boundary loss 叠加在 CE+Dice 上；全程 GPU，无 scipy
- **实验目录**：`nnUNetTrainer_ConvBoundaryLoss__nnUNetPlans__3d_fullres/`
- **定义位置**：`pumengyu/boundary/conv_trainer.py`
- **备注**：BoundaryLoss 的工程简化版，无需额外预测头，训练更快

---

### nnUNetTrainer\_BCE
- **核心**：LoCo 论文的 BCE 变体（Boundary Contrast Enhancement）；在 decoder 最高分辨率特征上挂 hook，采样边界体素中与类原型最远的困难样本，用 InfoNCE 对比 loss（λ=0.1，warmup 50 epoch）强化边界区分度
- **实验目录**：`nnUNetTrainer_BCE__nnUNetPlans__3d_fullres/`
- **定义位置**：`pumengyu/loco/trainer.py`
- **参考**：LoCo arXiv:2412.02314
- **备注**：消融实验 Step1，仅 BCE 不含 ICE

---

### nnUNetTrainer\_ICE（未跑）
- **核心**：LoCo 的 ICE 变体（Inter-class Contrast Enhancement）；找每类中与类原型相似度最低的体素，用对比 loss 拉近，无边界约束
- **实验目录**：`nnUNetTrainer_ICE__nnUNetPlans__3d_fullres/`
- **定义位置**：`pumengyu/loco/trainer.py`
- **备注**：消融实验 Step2，尚未跑

---

### nnUNetTrainer\_BCE\_ICE（未跑）
- **核心**：BCE + ICE 合用
- **实验目录**：`nnUNetTrainer_BCE_ICE__nnUNetPlans__3d_fullres/`
- **定义位置**：`pumengyu/loco/trainer.py`
- **备注**：消融实验 Step3，尚未跑

---

## 新增 Trainer 时的检查清单

1. 在本文件对应数据集下追加一条，填写：类名、核心思路（一句话）、关键超参、实验目录后缀、定义位置
2. 确保类 docstring 里有实验目录名（方便 grep）
3. 如有 Bug 或前置条件，写在备注里
