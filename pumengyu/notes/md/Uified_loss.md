# Unified Focal Loss 公式体系完整推导

> 基于 Yeung et al., 2021 *"Unified Focal loss: Generalising Dice and cross entropy-based losses to handle class imbalanced medical image segmentation"*

---

## 符号说明

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

## 一、Cross-Entropy Loss（交叉熵）

标准多类交叉熵：

$$\mathcal{L}_{CE} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{C} p_{c,i} \log \hat{p}_{c,i}$$

二分类 BCE（带类别权重 $\beta$）：

$$\mathcal{L}_{BCE} = -\frac{1}{N} \sum_{i=1}^{N} \left[ \beta \, p_i \log \hat{p}_i + (1-\beta)(1-p_i)\log(1-\hat{p}_i) \right]$$

---

## 二、Focal Loss

在 CE 基础上加入调制因子，抑制易分样本：

$$\mathcal{L}_{F} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{C} (1 - \hat{p}_{c,i})^{\gamma} \, p_{c,i} \log \hat{p}_{c,i}$$

- $\gamma = 0$：退化为标准 CE
- $\gamma > 0$：$(1-\hat{p})^\gamma$ 对已分对的样本（$\hat{p}$ 大）权重接近0，让模型专注于难样本

---

## 三、Tversky Index 与相关 Loss

### 3.1 Tversky Index（TI）

$$\mathrm{TI}_c = \frac{\displaystyle\sum_{i=1}^{N} p_{c,i}\, g_{c,i} + \epsilon}{\displaystyle\sum_{i=1}^{N} p_{c,i}\, g_{c,i} + \alpha \sum_{i=1}^{N} p_{c,i}(1-g_{c,i}) + \beta \sum_{i=1}^{N} (1-p_{c,i})\, g_{c,i} + \epsilon}$$

约束：$\alpha + \beta = 1$

| $\alpha, \beta$ 取值 | 等价形式 |
|---------------------|----------|
| $\alpha=\beta=0.5$ | Dice 系数（DSC） |
| $\beta > \alpha$ | 更重视 FN（适合肿瘤分割） |

$$\mathrm{DSC} = \frac{2\,\mathrm{TP}}{2\,\mathrm{TP} + \mathrm{FP} + \mathrm{FN}}$$

### 3.2 Tversky Loss

$$\mathcal{L}_{T} = \sum_{c=1}^{C} (1 - \mathrm{TI}_c)$$

### 3.3 Focal Tversky Loss

$$\mathcal{L}_{FT} = \sum_{c=1}^{C} (1 - \mathrm{TI}_c)^{t}, \quad t \in \left[\frac{1}{3}, 1\right]$$

---

## 四、Combo Loss（对比参考）

早期 CE + Dice 组合方案：

$$\mathcal{L}_{\mathrm{combo}} = \alpha \, \mathcal{L}_{\mathrm{mCE}} + (1-\alpha)(1 - \mathrm{DSC})$$

---

## 五、Unified Focal Loss 核心体系

### 5.1 Modified Tversky Index（mTI）

结构与 TI 相同，但在 Unified Focal 框架中专门用于非对称处理：

$$\mathrm{mTI}_c = \frac{\displaystyle\sum_{i=1}^{N} p_{c,i}\, g_{c,i} + \epsilon}{\displaystyle\sum_{i=1}^{N} p_{c,i}\, g_{c,i} + \alpha \sum_{i=1}^{N} p_{c,i}(1-g_{c,i}) + \beta \sum_{i=1}^{N} (1-p_{c,i})\, g_{c,i} + \epsilon}$$

### 5.2 Modified Asymmetric Focal Loss（$\mathcal{L}_{\mathrm{maF}}$）

**核心思想**：只对稀有类加 focal 调制，背景类用普通 CE

$$\boxed{\mathcal{L}_{\mathrm{maF}} = - \sum_{c \neq r} \sum_{i=1}^{N} p_{c,i} \log \hat{p}_{c,i} - \sum_{c = r} \sum_{i=1}^{N} (1-\hat{p}_{c,i})^{\delta}\, p_{c,i} \log \hat{p}_{c,i}}$$

| 类别 | 处理方式 | 原因 |
|------|----------|------|
| 背景类 $c \neq r$ | 普通 CE | 体素多、易分，不需要focal |
| 稀有类 $c = r$ | Focal CE，权重 $(1-\hat{p})^\delta$ | 体素少、难分，需要增强 |

### 5.3 Modified Asymmetric Focal Tversky Loss（$\mathcal{L}_{\mathrm{maFT}}$）

**核心思想**：只对稀有类加 focal 调制，背景类用普通 Tversky

$$\boxed{\mathcal{L}_{\mathrm{maFT}} = \sum_{c \neq r} (1 - \mathrm{mTI}_c) + \sum_{c = r} (1 - \mathrm{mTI}_c)^{1-\gamma}}$$

> ⚠️ 注意：指数是 $1-\gamma$（不是 $\gamma$！）
> - $\gamma \to 1$：指数 $\to 0$，调制消失 → 退化为标准 Tversky
> - $\gamma = 0$：指数 $= 1$ → 退化为标准 Focal Tversky

### 5.4 Asymmetric Unified Focal Loss（$\mathcal{L}_{\mathrm{aUF}}$）

$$\boxed{\mathcal{L}_{\mathrm{aUF}} = \lambda \cdot \mathcal{L}_{\mathrm{maF}} + (1-\lambda) \cdot \mathcal{L}_{\mathrm{maFT}}}$$

---

## 六、参数汇总

| 参数 | 含义 | 范围 | 推荐值 |
|------|------|------|--------|
| $\lambda$ | CE侧 vs Tversky侧权重 | $[0,1]$ | $0.5$ |
| $\alpha$ | FP 惩罚权重 | $[0,1]$ | $0.3$ |
| $\beta$ | FN 惩罚权重，$\alpha+\beta=1$ | $[0,1]$ | $0.7$ |
| $\gamma$ | Tversky侧 focal 强度 | $[0,1]$ | $0.75$ |
| $\delta$ | CE侧 focal 强度 | $[0,1]$ | $0.6$ |
| $\epsilon$ | 平滑项 | — | $10^{-6}$ |

---

## 七、结构总览

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

## 八、你手写笔记中需要补充/纠正的地方

| 位置 | 问题 | 正确写法 |
|------|------|----------|
| 图1 BCE | 漏了 $\frac{1}{N}$ 归一化 | 见上方公式 |
| 图3 mTI分母 | 漏了 $\beta\sum(1-p_i)g_i$ 项 | 完整分母含3项 |
| 图4 Combo | 符号写了减号 | 应为加权和 $\alpha L + (1-\alpha)(1-\mathrm{DSC})$ |
| 图3 maFT | 背景类求和下标不清晰 | 明确写 $c \neq r$ 和 $c = r$ |
| 全局 | $\alpha+\beta=1$ 约束未标注 | 必须标注 |