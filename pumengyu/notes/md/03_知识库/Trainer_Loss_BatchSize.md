# Trainer / Loss / Batch Size

## Trainer 登记册


#### Trainer 登记册

每次新增 Trainer 类，在此登记一条。  
**格式**：类名 | 数据集 | 核心思路（一句话）| 关键超参 | 实验目录后缀

---

##### Dataset003\_Liver（全 CT，联合分割肝脏+肿瘤，label1=肝脏 label2=肿瘤）

###### nnUNetTrainer
- **核心**：标准 nnUNet baseline，CE+Dice loss，无任何改动
- **实验目录**：`nnUNetTrainer__nnUNetPlans__3d_fullres/`
- **定义位置**：nnUNet 原生（无需自定义）

---

###### nnUNetTrainer\_UFL
- **核心**：在 CE+Dice 基础上叠加 AsymmetricUnifiedFocalLoss（仅针对肿瘤类），自动平衡极小肿瘤体素与大背景的梯度贡献
- **关键超参**：`UFL_LAMBDA=0.5` / **`UFL_DELTA=0.6`（偏向惩罚漏检 FN，更激进地召回肿瘤）** / `UFL_GAMMA=0.2`
- **实验目录**：`nnUNetTrainer_UFL__nnUNetPlans__3d_fullres/`
- **定义位置**：`pumengyu/trainers/trainer.py` + `pumengyu/mixins.py::UnifiedFocalLossMixin`
- **训练时 mixins.py 状态**：commit `53b863c`（delta=0.6 版本）
- **备注**：用于与 CopyPaste 形成 2×2 消融（有/无 CopyPaste × 有/无 UFL）

---

###### nnUNetTrainer\_UFL\_v2
- **核心**：与 UFL v1 思路相同，但 **`UFL_DELTA=0.5`（对称惩罚 FN/FP，不再偏向召回）**
- **关键超参**：`UFL_LAMBDA=0.5` / **`UFL_DELTA=0.5`** / `UFL_GAMMA=0.2`
- **实验目录**：`nnUNetTrainer_UFL_v2__nnUNetPlans__3d_fullres/`
- **定义位置**：`pumengyu/trainers/trainer.py`
- **训练时 mixins.py 状态**：commit `17b5592`（delta 从 0.6 改为 0.5）
- **与 v1 的实质差异**：`UFL_DELTA 0.6→0.5`，动机是"消除对 FN 的系统性偏置，降低无肿瘤 case 误报"（见 mixins.py 注释）；但当前 UnifiedFocalLossMixin 的类变量已是 delta=0.5，如需重跑 delta=0.6 版本，在子类中覆盖 `UFL_DELTA = 0.6` 即可，无需 revert git

###### nnUNetTrainer\_UFL\_delta06
- **核心**：与 UFL_v2 代码完全相同，仅将 `UFL_DELTA` 覆盖回 0.6，用于与 UFL_v2（delta=0.5）做显式对照消融
- **关键超参**：`UFL_LAMBDA=0.5` / **`UFL_DELTA=0.6`** / `UFL_GAMMA=0.2`
- **实验目录**：`nnUNetTrainer_UFL_delta06__nnUNetPlans__3d_fullres/`
- **定义位置**：`pumengyu/trainers/trainer.py`
- **备注**：尚未跑实验；等价于 nnUNetTrainer_UFL 的干净重现版本

---

> ⭐ **重大发现（2026-05-24）**：长期以为 UFL v1 和 v2 是"同代码的干净重跑"，实际上两者 **UFL_DELTA 不同（0.6 vs 0.5）**，这才是结果差异的根本原因，而非随机性。该差异通过 `git show 53b863c:pumengyu/mixins.py` vs `git show 17b5592:pumengyu/mixins.py` 比对确认。delta=0.6 让模型更激进召回（fold_4 Dice 0.639），但也拉高了 fold_1/fold_2 的误报率；delta=0.5 更保守（fold_4 Dice 0.597）。**UFL_DELTA 是一个值得系统消融的超参**，建议后续用 `nnUNetTrainer_UFL_delta06` / `nnUNetTrainer_UFL_delta05` 这样的命名显式区分，不要再出现隐式变更。

---

###### nnUNetTrainer\_CopyPaste
- **核心**：小肿瘤过采样（identifiers 重复 3x） + 在线 CopyPaste（50% 概率将小肿瘤 ROI 粘贴进其他 case 的肝脏区域）
- **关键超参**：`CP_PROB=0.5` / `CP_MAX_LOCS=5000`（小肿瘤判定上限） / `SMALL_TUMOR_REPEAT=3` / `SMALL_TUMOR_THRESH_LOCS=6000`
- **实验目录**：`nnUNetTrainer_CopyPaste__nnUNetPlans__3d_fullres/`
- **定义位置**：`pumengyu/trainers/trainer.py` + `pumengyu/mixins.py::CopyPasteMixin + SmallTumorOversampleMixin`
- **⚠️ 已知 Bug**：多连通域提取 bug——抽 ROI 时可能把多个非相邻连通域打包粘贴，导致合成样本不自然；v2 已修复

---

###### nnUNetTrainer\_CopyPaste\_v2
- **核心**：与 CopyPaste v1 完全相同，修复了多连通域 bug（每次只粘贴单一连通域）
- **关键超参**：同 v1（`CP_PROB=0.5` / `CP_MAX_LOCS=5000` / `SMALL_TUMOR_REPEAT=3` / `CP_MARGIN=3`）
- **实验目录**：`nnUNetTrainer_CopyPaste_v2__nnUNetPlans__3d_fullres/`
- **定义位置**：`pumengyu/trainers/trainer.py`

---

###### nnUNetTrainer\_CopyPaste\_Diff
- **核心**：用内在难度加权替代均匀随机抽 ROI，其余与 CopyPaste_v2 完全一致；唯一变量 = 粘谁
- **实验目录**：`nnUNetTrainer_CopyPaste_Diff__nnUNetPlans__3d_fullres/`
- **定义位置**：`pumengyu/trainers/trainer.py` + `pumengyu/mixins.py::DifficultyCopyPasteMixin`
- **前置条件**：需先运行 `pumengyu/tools/data_analysis/compute_difficulty.py` 生成 `difficulty.json`
- **备注**：尚未跑实验

---

###### nnUNetTrainer\_CopyPasteUFL
- **核心**：CopyPaste_v1 + UFL 叠加，用于验证两者是否有正交增益
- **实验目录**：`nnUNetTrainer_CopyPasteUFL__nnUNetPlans__3d_fullres/`
- **定义位置**：`pumengyu/trainers/trainer.py`
- **备注**：尚未跑实验；⚠️ 继承 CopyPaste v1，含多连通域 bug

---

##### Dataset004\_LiverTumor（肝脏 ROI 裁剪后专做肿瘤，label1=肿瘤）

> 数据来源：Dataset003 的肝脏预测裁剪出的 ROI，只分割肿瘤

###### nnUNetTrainer\_TwoStage
- **核心**：训练逻辑与标准 nnUNet 相同（加小肿瘤过采样），**区别在推理**：Stage1 肝脏预测 → 裁剪 ROI → Stage2 在 ROI 内做肿瘤分割；验证结束后自动触发端到端评估
- **实验目录**：`nnUNetTrainer_TwoStage__nnUNetPlans__3d_fullres/`
- **定义位置**：`pumengyu/twostage/trainer.py`

---

###### nnUNetTrainer\_BoundaryLoss
- **核心**：额外预测距离场（并行预测头从最高分辨率 decoder 特征分叉），loss = CE+Dice + L_boundary；L_boundary = mean(\|pred_dist−gt_dist\|³)；距离场 GT 在 DataLoader worker 内在线用 scipy EDT 计算
- **实验目录**：`nnUNetTrainer_BoundaryLoss__nnUNetPlans__3d_fullres/`
- **定义位置**：`pumengyu/boundary/trainer.py`
- **参考**：BATseg arXiv:2412.06507
- **特点**：需修改 DataLoader，有 CPU EDT 计算瓶颈

---

###### nnUNetTrainer\_ConvBoundaryLoss
- **核心**：用固定各向异性 Laplacian 卷积核提取预测概率图和 GT one-hot 的边界响应，两者差异作为 boundary loss 叠加在 CE+Dice 上；全程 GPU，无 scipy
- **实验目录**：`nnUNetTrainer_ConvBoundaryLoss__nnUNetPlans__3d_fullres/`
- **定义位置**：`pumengyu/boundary/conv_trainer.py`
- **备注**：BoundaryLoss 的工程简化版，无需额外预测头，训练更快

---

###### nnUNetTrainer\_BCE
- **核心**：LoCo 论文的 BCE 变体（Boundary Contrast Enhancement）；在 decoder 最高分辨率特征上挂 hook，采样边界体素中与类原型最远的困难样本，用 InfoNCE 对比 loss（λ=0.1，warmup 50 epoch）强化边界区分度
- **实验目录**：`nnUNetTrainer_BCE__nnUNetPlans__3d_fullres/`
- **定义位置**：`pumengyu/loco/trainer.py`
- **参考**：LoCo arXiv:2412.02314
- **备注**：消融实验 Step1，仅 BCE 不含 ICE

---

###### nnUNetTrainer\_ICE（未跑）
- **核心**：LoCo 的 ICE 变体（Inter-class Contrast Enhancement）；找每类中与类原型相似度最低的体素，用对比 loss 拉近，无边界约束
- **实验目录**：`nnUNetTrainer_ICE__nnUNetPlans__3d_fullres/`
- **定义位置**：`pumengyu/loco/trainer.py`
- **备注**：消融实验 Step2，尚未跑

---

###### nnUNetTrainer\_BCE\_ICE（未跑）
- **核心**：BCE + ICE 合用
- **实验目录**：`nnUNetTrainer_BCE_ICE__nnUNetPlans__3d_fullres/`
- **定义位置**：`pumengyu/loco/trainer.py`
- **备注**：消融实验 Step3，尚未跑

---

---

##### 对比实验（论文用，与 MoE_SizeOV4 横向比较）

###### nnUNetTrainer_MedNeXt
- **核心**：MedNeXt-L（ConvNeXt 风格 3D UNet），作为论文对比 baseline
- **关键超参**：n_channels=32, kernel_size=3, exp_r=[3,4,8,8,8,8,8,4,3], block_counts=[3,4,8,8,8,8,8,4,3]
- **实验目录**：`nnUNetTrainer_MedNeXt__nnUNetPlans__3d_fullres/`
- **定义位置**：`pumengyu/trainers/trainer.py` + `pumengyu/architectures/mednext.py`
- **训练命令**：`CUDA_VISIBLE_DEVICES=0 python -m pumengyu.trainers.run_training --trainer nnUNetTrainer_MedNeXt --fold 0`
- **备注**：使用 nnunet_mednext 包中的架构代码，通过 build_network_architecture 接入 nnUNetv2 训练流程；DS 输出 5 层（128³→8³）✅ 验证通过

---

###### nnUNetTrainer_SwinUNETR
- **核心**：SwinUNETR-B（MONAI 官方实现），Swin Transformer encoder + CNN decoder
- **关键超参**：feature_size=48, depths=(2,2,2,2), num_heads=(3,6,12,24), window_size=7, patch_size=2
- **实验目录**：`nnUNetTrainer_SwinUNETR__nnUNetPlans__3d_fullres/`
- **定义位置**：`pumengyu/trainers/trainer.py` + `pumengyu/architectures/swinunetr.py`
- **训练命令**：`CUDA_VISIBLE_DEVICES=0 nnUNetv2_train Dataset003_Liver 3d_fullres 0 -tr nnUNetTrainer_SwinUNETR`
- **备注**：MONAI 1.5.x 新接口已去掉 img_size，用 patch_size=2；无 DS，trainer 关闭 DS ✅ 验证通过

###### nnUNetTrainer_nnFormer
- **核心**：nnFormer（mednextv1 包附带的 nnFormer_tumor 实现），3D Swin Transformer UNet
- **关键超参**：embedding_dim=96, depths=[2,2,2,2], num_heads=[3,6,12,24], patch_size=[4,4,4], window_size=[4,4,8,4]
- **实验目录**：`nnUNetTrainer_nnFormer__nnUNetPlans__3d_fullres/`
- **定义位置**：`pumengyu/trainers/trainer.py` + `pumengyu/architectures/nnformer.py`
- **训练命令**：`CUDA_VISIBLE_DEVICES=1 nnUNetv2_train Dataset003_Liver 3d_fullres 0 -tr nnUNetTrainer_nnFormer`
- **备注**：DS 代码已注释掉，始终返回单 tensor，trainer 关闭 DS ✅ 验证通过

---

##### 新增 Trainer 时的检查清单

1. 在本文件对应数据集下追加一条，填写：类名、核心思路（一句话）、关键超参、实验目录后缀、定义位置
2. 确保类 docstring 里有实验目录名（方便 grep）
3. 如有 Bug 或前置条件，写在备注里


---

## Unified Focal Loss 公式体系完整推导


#### Unified Focal Loss 公式体系完整推导

> 基于 Yeung et al., 2021 *"Unified Focal loss: Generalising Dice and cross entropy-based losses to handle class imbalanced medical image segmentation"*

---

##### 符号说明

| 符号 | 含义 |
|------|------|
| $N$ | 像素总数 |
| $C$ | 类别总数 |
| $p_{c,i}$ | 第 $i$ 个像素属于类别 $c$ 的 **ground truth**（one-hot，0或1） |
| $\hat{p}_{c,i}$ | 模型预测第 $i$ 个像素为类别 $c$ 的**概率**（softmax输出） |
| $g_{c,i}$ | ground truth（与 $p_{c,i}$ 同义，Tversky公式中常用 $g$） |
| $r$ | **稀有类**编号（如肿瘤类） |
| $\epsilon$ | 平滑项，防止分母为零 |

---

##### 一、Cross-Entropy Loss（交叉熵）

标准多类交叉熵：

$$\mathcal{L}_{CE} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{C} p_{c,i} \log \hat{p}_{c,i}$$

二分类 BCE（带类别权重 $\beta$）：

$$\mathcal{L}_{BCE} = -\frac{1}{N} \sum_{i=1}^{N} \left[ \beta \, p_i \log \hat{p}_i + (1-\beta)(1-p_i)\log(1-\hat{p}_i) \right]$$

---

##### 二、Focal Loss

在 CE 基础上加入调制因子，抑制易分样本：

$$\mathcal{L}_{F} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{C} (1 - \hat{p}_{c,i})^{\gamma} \, p_{c,i} \log \hat{p}_{c,i}$$

- $\gamma = 0$：退化为标准 CE
- $\gamma > 0$：$(1-\hat{p})^\gamma$ 对已分对的样本（$\hat{p}$ 大）权重接近0，让模型专注于难样本

---

##### 三、Tversky Index 与相关 Loss

###### 3.1 Tversky Index（TI）

$$\mathrm{TI}_c = \frac{\displaystyle\sum_{i=1}^{N} p_{c,i}\, g_{c,i} + \epsilon}{\displaystyle\sum_{i=1}^{N} p_{c,i}\, g_{c,i} + \alpha \sum_{i=1}^{N} p_{c,i}(1-g_{c,i}) + \beta \sum_{i=1}^{N} (1-p_{c,i})\, g_{c,i} + \epsilon}$$

约束：$\alpha + \beta = 1$

| $\alpha, \beta$ 取值 | 等价形式 |
|---------------------|----------|
| $\alpha=\beta=0.5$ | Dice 系数（DSC） |
| $\beta > \alpha$ | 更重视 FN（适合肿瘤分割） |

$$\mathrm{DSC} = \frac{2\,\mathrm{TP}}{2\,\mathrm{TP} + \mathrm{FP} + \mathrm{FN}}$$

###### 3.2 Tversky Loss

$$\mathcal{L}_{T} = \sum_{c=1}^{C} (1 - \mathrm{TI}_c)$$

###### 3.3 Focal Tversky Loss

$$\mathcal{L}_{FT} = \sum_{c=1}^{C} (1 - \mathrm{TI}_c)^{t}, \quad t \in \left[\frac{1}{3}, 1\right]$$

---

##### 四、Combo Loss（对比参考）

早期 CE + Dice 组合方案：

$$\mathcal{L}_{\mathrm{combo}} = \alpha \, \mathcal{L}_{\mathrm{mCE}} + (1-\alpha)(1 - \mathrm{DSC})$$

---

##### 五、Unified Focal Loss 核心体系

###### 5.1 Modified Tversky Index（mTI）

结构与 TI 相同，但在 Unified Focal 框架中专门用于非对称处理：

$$\mathrm{mTI}_c = \frac{\displaystyle\sum_{i=1}^{N} p_{c,i}\, g_{c,i} + \epsilon}{\displaystyle\sum_{i=1}^{N} p_{c,i}\, g_{c,i} + \alpha \sum_{i=1}^{N} p_{c,i}(1-g_{c,i}) + \beta \sum_{i=1}^{N} (1-p_{c,i})\, g_{c,i} + \epsilon}$$

###### 5.2 Modified Asymmetric Focal Loss（$\mathcal{L}_{\mathrm{maF}}$）

**核心思想**：只对稀有类加 focal 调制，背景类用普通 CE

$$\boxed{\mathcal{L}_{\mathrm{maF}} = - \sum_{c \neq r} \sum_{i=1}^{N} p_{c,i} \log \hat{p}_{c,i} - \sum_{c = r} \sum_{i=1}^{N} (1-\hat{p}_{c,i})^{\delta}\, p_{c,i} \log \hat{p}_{c,i}}$$

| 类别 | 处理方式 | 原因 |
|------|----------|------|
| 背景类 $c \neq r$ | 普通 CE | 体素多、易分，不需要focal |
| 稀有类 $c = r$ | Focal CE，权重 $(1-\hat{p})^\delta$ | 体素少、难分，需要增强 |

###### 5.3 Modified Asymmetric Focal Tversky Loss（$\mathcal{L}_{\mathrm{maFT}}$）

**核心思想**：只对稀有类加 focal 调制，背景类用普通 Tversky

$$\boxed{\mathcal{L}_{\mathrm{maFT}} = \sum_{c \neq r} (1 - \mathrm{mTI}_c) + \sum_{c = r} (1 - \mathrm{mTI}_c)^{1-\gamma}}$$

> ⚠️ 注意：指数是 $1-\gamma$（不是 $\gamma$！）
> - $\gamma \to 1$：指数 $\to 0$，调制消失 → 退化为标准 Tversky
> - $\gamma = 0$：指数 $= 1$ → 退化为标准 Focal Tversky

###### 5.4 Asymmetric Unified Focal Loss（$\mathcal{L}_{\mathrm{aUF}}$）

$$\boxed{\mathcal{L}_{\mathrm{aUF}} = \lambda \cdot \mathcal{L}_{\mathrm{maF}} + (1-\lambda) \cdot \mathcal{L}_{\mathrm{maFT}}}$$

---

##### 六、参数汇总

| 参数 | 含义 | 范围 | 推荐值 |
|------|------|------|--------|
| $\lambda$ | CE侧 vs Tversky侧权重 | $[0,1]$ | $0.5$ |
| $\alpha$ | FP 惩罚权重 | $[0,1]$ | $0.3$ |
| $\beta$ | FN 惩罚权重，$\alpha+\beta=1$ | $[0,1]$ | $0.7$ |
| $\gamma$ | Tversky侧 focal 强度 | $[0,1]$ | $0.75$ |
| $\delta$ | CE侧 focal 强度 | $[0,1]$ | $0.6$ |
| $\epsilon$ | 平滑项 | — | $10^{-6}$ |

---

##### 七、结构总览

```
Asymmetric Unified Focal Loss (L_aUF)
│
├── λ × L_maF         ← CE 侧（asymmetric Focal CE）
│   ├── 背景类：普通 CE（无focal调制）
│   └── 稀有类：(1-p̂)^δ × CE
│
└── (1-λ) × L_maFT    ← Dice/Tversky 侧（asymmetric Focal Tversky）
    ├── 背景类：普通 Tversky loss（无focal调制）
    └── 稀有类：(1-mTI)^(1-γ)
```

---

##### 八、你手写笔记中需要补充/纠正的地方

| 位置 | 问题 | 正确写法 |
|------|------|----------|
| 图1 BCE | 漏了 $\frac{1}{N}$ 归一化 | 见上方公式 |
| 图3 mTI分母 | 漏了 $\beta\sum(1-p_i)g_i$ 项 | 完整分母含3项 |
| 图4 Combo | 符号写了减号 | 应为加权和 $\alpha L + (1-\alpha)(1-\mathrm{DSC})$ |
| 图3 maFT | 背景类求和下标不清晰 | 明确写 $c \neq r$ 和 $c = r$ |
| 全局 | $\alpha+\beta=1$ 约束未标注 | 必须标注 |


---

## 对，完全正确。


对，完全正确。

---

##### 确认一下你的理解

```
batch_size = 2
↓
一个step = 抽2个case，各取1个patch，合并成一个batch，算一次梯度更新
```

**batch_size就是一个step里用几个样本（patch）。**

---

##### 完整的一个step流程

```
step1：
  抽 case_37 → 随机裁一个patch [1,128,128,128]
  抽 case_102 → 随机裁一个patch [1,128,128,128]
  合并 → [2, 1, 128, 128, 128]
  ↓
  正向传播
  ↓
  算loss
  ↓
  反向传播
  ↓
  更新一次权重   ← 这才叫完成一个step
```

---

##### 所以三个概念的关系

```
batch_size = 2      → 一个step用几个patch
250步               → 一个epoch有几个step
1000个epoch         → 训练多少轮

总梯度更新次数 = 250 × 1000 = 250,000次
```

**一句话：batch_size控制每步的宽度，250步控制每轮的长度。**


---
