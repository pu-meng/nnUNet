# 普通卷积 vs ConvNeXt vs MedNeXt

> 核心问题：感受野和计算量是一对矛盾，三者都在绕这个问题

---

## 1. 普通 3D 卷积（Standard Conv）

```
输入: [B, C_in, D, H, W]
↓  3×3×3 conv, C_in → C_out
↓  BN + ReLU
输出: [B, C_out, D, H, W]
```

**参数量**：`C_in × C_out × 3³ = 27 × C_in × C_out`

### BN（Batch Normalization）


```
输入: [B, C, D, H, W]
↓  沿 B、D、H、W 求均值/方差，每个通道 C 独立一套 (μ, σ)
↓  x̂[b,c,d,h,w] = (x[b,c,d,h,w] - μ_c) / σ_c × γ_c + β_c
输出: [B, C, D, H, W]
```






### ReLU（Rectified Linear Unit）

```
ReLU(x) = max(0, x)
```

引入非线性，让网络能拟合复杂函数。没有激活函数，多层线性变换叠起来还是一个线性变换，表达能力和一层一样。

**问题（Dead ReLU）**：
- x < 0 的区域梯度恒为 0，对应的神经元永久死亡，参数不再更新
- 深层网络里这个问题会积累

这是 ConvNeXt/MedNeXt 换成 **GELU** 的原因（见第 2 节）。

**问题**：
- 感受野小（3×3×3），抓不到长程依赖
- 想用大核（7×7×7）？参数量变 `343 × C_in × C_out`，直接爆炸
- 3D 比 2D 更严重：体积增长是三次方

---

## 2. ConvNeXt Block（2D，Liu et al. 2022）

ConvNeXt 的核心思路：**把 Transformer block 的设计哲学用纯卷积实现**
Depthwise conv=深度卷积或逐通道卷积
Depthwise separable conv=深度可分离卷积

```
输入: [B, C, H, W]
↓  7×7 Depthwise Conv（每个通道独立卷积）   ← 大核，但参数极少
↓  LayerNorm
↓  1×1 Conv, C → 4C                          ← Pointwise，升维
↓  GELU
↓  1×1 Conv, 4C → C                          ← Pointwise，降回来
↓  残差连接
输出: [B, C, H, W]
```

### LayerNorm vs BN

输入 tensor 形状：`[B, C, D, H, W]`

**BN：沿 B、D、H、W 求均值/方差，每个通道 C 独立一套 (μ, σ)**

```
[B, C, D, H, W]
 ↑        ↑↑↑
 这四个维度一起参与计算 → 得到 C 个 (μ_c, σ_c)

对通道 c：
  μ_c = mean over [B, D, H, W]   # 形状: (1, C, 1, 1, 1)
  σ_c = std  over [B, D, H, W]
  x̂[b,c,d,h,w] = (x[b,c,d,h,w] - μ_c) / σ_c × γ_c + β_c
```

问题：B × D × H × W 个点一起算统计量。3D 医学图像 B=2，D×H×W 很大，但 B 太小，batch 内只有 2 个样本，μ/σ 估计不稳定。

---

**LN：沿 C、D、H、W 求均值/方差，每个样本 B 独立一套 (μ, σ)**

```
[B, C, D, H, W]
     ↑  ↑↑↑
     这四个维度一起参与计算 → 得到 B 个 (μ_b, σ_b)

对样本 b：
  μ_b = mean over [C, D, H, W]   # 形状: (B, 1, 1, 1, 1)
  σ_b = std  over [C, D, H, W]
  x̂[b,c,d,h,w] = (x[b,c,d,h,w] - μ_b) / σ_b × γ + β
```

每个样本自己算自己的统计量，B=1 也完全没问题，和其他样本完全无关。

---

```
BN：固定 C，在 [B,D,H,W] 上求统计 → B 小则统计不准
LN：固定 B，在 [C,D,H,W] 上求统计 → 永远只看自己这一个样本
```

| | BN | LN |
|--|--|--|
| 归一化的维度 | B, D, H, W | C, D, H, W |
| 统计量数量 | C 个（每通道一个） | B 个（每样本一个） |
| batch size 小时 | μ/σ 估计不准 | 不受影响 |
| 可学习参数 γ,β | 每通道独立 | 所有样本共享 |

### GELU vs ReLU

#### ReLU

**公式**：

$$\text{ReLU}(x) = \max(0, x) = \begin{cases} x, & x > 0 \\ 0, & x \leq 0 \end{cases}$$

**导数**：

$$\text{ReLU}'(x) = \begin{cases} 1, & x > 0 \\ 0, & x < 0 \end{cases}$$

$x=0$ 处次梯度可取 $[0,1]$ 任意值，PyTorch 实现中取 $0$。

**问题（Dead ReLU）**：$x < 0$ 时梯度恒为 $0$，该神经元参数永远不更新，"死亡"。

---

#### GELU

**精确公式**：

$$\text{GELU}(x) = x \cdot \Phi(x)$$

其中 $\Phi(x)$ 是标准正态分布的 CDF：

$$\Phi(x) = P(X \leq x),\quad X \sim \mathcal{N}(0,1)$$

展开为误差函数形式：

$$\text{GELU}(x) = \frac{x}{2}\left[1 + \text{erf}\!\left(\frac{x}{\sqrt{2}}\right)\right]$$

$$\text{erf}(z) = \frac{2}{\sqrt{\pi}}\int_0^z e^{-t^2}\,dt$$

**工程近似公式**（避免计算 erf，实际代码用这个）：

$$\text{GELU}(x) \approx 0.5x\left(1 + \tanh\!\left[\sqrt{\frac{2}{\pi}}\left(x + 0.044715\,x^3\right)\right]\right)$$

**导数**（product rule：$(uv)' = u'v + uv'$）：

$$\text{GELU}'(x) = \Phi(x) + x \cdot \phi(x)$$

其中 $\phi(x)$ 是标准正态分布的 PDF：

$$\phi(x) = \frac{1}{\sqrt{2\pi}}\,e^{-x^2/2}$$

展开完整形式：

$$\text{GELU}'(x) = \frac{1}{2}\left[1 + \text{erf}\!\left(\frac{x}{\sqrt{2}}\right)\right] + \frac{x}{\sqrt{2\pi}}\,e^{-x^2/2}$$

**数值对比**：

| $x$ | $\text{ReLU}'$ | $\text{GELU}'$ |
|-----|---------------|---------------|
| $-2$ | $0$（死亡） | $\approx 0.045$ |
| $-1$ | $0$（死亡） | $\approx 0.167$ |
| $0$  | $0$ | $0.5$ |
| $1$  | $1$ | $\approx 0.833$ |
| $3$  | $1$ | $\approx 0.999$ |

**为什么 GELU 不会 Dead**：$x < 0$ 时 $\Phi(x) \in (0,\, 0.5)$，$\text{GELU}'(x)$ 始终 $> 0$，负区间永远有梯度流动。

Transformer 从一开始就用 GELU，ConvNeXt 为了对齐 Transformer 也换过来了。

### Depthwise Conv 为什么省参数？

| 操作 | 参数量 |
|------|--------|
| 普通 7×7 Conv（C→C） | `C × C × 49` |
| Depthwise 7×7 Conv | `C × 49`（每通道只有一个 49 参数的核） |
| 节省比例 | **1/C 倍** |

空间混合（Depthwise，感受野大）和通道混合（Pointwise，1×1）**解耦**，各做各的。

### ConvNeXt 和 Transformer 的关系（正确理解）

ConvNeXt 论文把自己包装成"借鉴 Transformer 思想"，这个说法是夸大的叙事。

**Transformer 的核心是 Attention**：

$$\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d}}\right)V$$

```
X → 线性映射 → Q, K, V
QK^T → 每个位置和所有位置的相关性矩阵（由输入内容动态生成）
softmax → 归一化成权重
× V → 用权重对所有位置加权求和
```

权重由当前输入内容动态决定，每次输入都不一样，这是 Transformer 真正的核心。

**ConvNeXt 的 7×7 Depthwise 和 Attention 的根本区别**：

```
Attention：  权重 = QK^T，由输入内容动态生成，不同样本权重不同
Depthwise：  权重 = 固定卷积核，训练完不变，对所有样本一样
```

7×7 Depthwise 只是感受野变大了，没有任何动态权重机制，和 Attention 的本质完全不同。

**ConvNeXt 的三个"借鉴"其实来源各异**：

| ConvNeXt 组件 | 真实来源 | 和 Transformer 的关系 |
|--------------|---------|-------------------|
| LN | Ba et al. 2016，为 RNN 提出 | Transformer 也用，但不是 Transformer 发明的 |
| GELU | Hendrycks 2016，BERT 用后出名 | 同上 |
| Inverted Bottleneck | MobileNet v2 2018 | Transformer FFN 恰好结构相似，但来源无关 |
| 7×7 大核 | 扩大感受野 | 只是局部近似，不是 Attention |

**准确的说法**：ConvNeXt 把 ResNet 做了工程现代化（LN/GELU/大核/Inverted Bottleneck），用 Transformer 同款训练配方训练，发现能打平 Swin Transformer。结论是：Transformer 在视觉任务上的优势，部分来自训练配方和设计细节，而不全是 Attention 本身。"借鉴 Transformer"是论文的叙事包装。

---

## 3. MedNeXt Block（3D，Roy et al. MICCAI 2023）

MedNeXt = ConvNeXt 的 3D 版本 + 针对医学图像的适配

```
输入: [B, C, D, H, W]
↓  k×k×k Depthwise Conv（k = 3/5/7，可配置）
↓  LayerNorm（3D）
↓  1×1×1 Conv, C → 4C
↓  GELU
↓  1×1×1 Conv, 4C → C
↓  残差连接
输出: [B, C, D, H, W]
```

**关键改动**：
1. 核从 2D 7×7 → 3D k×k×k，k 可选 3/5/7
2. 有 S / B / M / L 四个规模档，L 版本更深更宽
3. Upsampling 也换成了 3D Depthwise 转置卷积（不是双线性插值）

### Depthwise vs 深度可分离卷积

**Depthwise Conv（深度卷积）** 只是第一步，单独做空间混合，通道之间不交互：

```
[B, C, D, H, W] → [B, C, D, H, W]
每个通道独立做 k×k×k 卷积，参数量 = k³ × C
```

**深度可分离卷积（Depthwise Separable Conv）= 两步合在一起**：

```
第一步：Depthwise Conv   → 空间混合，参数 k³ × C
第二步：Pointwise Conv   → 通道混合，1×1×1，参数 C × C_out
```

MedNeXt block 用的是深度可分离的思想，但把 Pointwise 拆成升维+降维两步（inverted bottleneck）：

```
7×7×7 Depthwise Conv     ← 空间混合，参数 343C
1×1×1 Conv (C → 4C)      ← Pointwise 升维，参数 4C²
GELU
1×1×1 Conv (4C → C)      ← Pointwise 降维，参数 4C²
```

总参数 $= 343C + 8C^2$，与标准 $3\times3\times3$ conv 的 $27C^2$ 对比：

$$8C^2 + 343C < 27C^2 \implies C > \frac{343}{19} \approx 18$$

即 **C ≥ 19 起，MedNeXt block 参数比 3×3×3 conv 更少，感受野却从 3 扩到 7**：

| C | $27C^2$（3×3×3） | $8C^2+343C$（MedNeXt block） | 节省 |
|---|---|---|---|
| 32 | 27,648 | 19,168 | 31% |
| 64 | 110,592 | 54,400 | 51% |
| 128 | 442,368 | 174,464 | 61% |

C 越大优势越明显，医学图像分割网络通常 C=32~320，始终在有利区间。

省下来的参数空间 → 堆更多 block，感受野更大、容量更足。

---

## 4. Block 设计对比

> "UNet Bottleneck（最底层）" 和 "Bottleneck Block（block 设计模式）" 是两个不同概念，名字撞车。
> 下面讲的是 block 内部结构。

---

### Plain Block（nnUNet 使用）

每个 stage 两个卷积，**第一个 conv 负责通道变化（+ 下采样），第二个 conv 通道不变**。没有单独的下采样层，stride 和通道变化都在第一个 conv 里完成。

**stage 内第一个 conv（通道变化 + 可选下采样）**：

```
输入 x: [B, C_in, D, H, W]
│
├─ Conv 3×3×3 (C_in → C_out, stride=2)   ← 通道变化 + 下采样同时做
├─ InstanceNorm3d
├─ ReLU
│
├─ Conv 3×3×3 (C_out → C_out, stride=1)  ← 通道不变，只做空间特征变换
├─ InstanceNorm3d
├─ ReLU
│
输出: [B, C_out, D/2, H/2, W/2]
```

**stage 内后续 conv（通道不变）**：

```
输入 x: [B, C, D, H, W]
│
├─ Conv 3×3×3 (C → C, stride=1)
├─ InstanceNorm3d
├─ ReLU
│
├─ Conv 3×3×3 (C → C, stride=1)
├─ InstanceNorm3d
├─ ReLU
│
输出: [B, C, D, H, W]
```

**残差连接**：nnUNet 的 PlainConvUNet **没有**残差连接。  
ResEncUNet 变体才加，通道不变时直接相加：

```
输入 x ──────────────────────────────┐
│                                    │
├─ Conv 3×3×3 (C→C) → IN → ReLU    │
├─ Conv 3×3×3 (C→C) → IN           │
│                                    │
└──────────── + ─────────────────────┘  ← 直接相加，形状一致
              │
             ReLU
             输出
```

通道变化时（第一个 conv stride=2, C_in→C_out），残差路径需要 1×1 Conv 对齐：

```
输入 x: [B, C_in, D, H, W]
│
├─────────────────────────────────── 1×1 Conv (C_in→C_out, stride=2) ──┐
│                                                                        │
├─ Conv 3×3×3 (C_in→C_out, stride=2) → IN → ReLU                      │
├─ Conv 3×3×3 (C_out→C_out, stride=1) → IN                            │
│                                                                        │
└──────────────────────────────────────── + ────────────────────────────┘
                                           │
                                          ReLU
                                          输出: [B, C_out, D/2, H/2, W/2]
```

1×1 Conv 残差路径和主路径必须 stride 一致，否则形状不匹配无法相加。

---

### Bottleneck Block（ResNet-50/101）

三个卷积，让空间混合（3×3）在通道被压缩后的窄处做，省参数。

```
输入 x: [B, C, D, H, W]
│
├─ Conv 1×1×1 (C → C/4)     ← 压缩通道，参数 C²/4
├─ BN → ReLU
│
├─ Conv 3×3×3 (C/4 → C/4)   ← 空间混合，在窄处做，参数 9·(C/4)²
├─ BN → ReLU
│
├─ Conv 1×1×1 (C/4 → C)     ← 恢复通道，参数 C²/4
├─ BN
│
└─ + x (残差连接)             ← 直接相加，通道数一致
   └─ ReLU
   输出: [B, C, D, H, W]
```

**参数量对比**（C=256）：

$$\text{Plain } 3\times3\times3 \times 2：2 \times 27C^2 = 54C^2$$

$$\text{Bottleneck}：\frac{C^2}{4} + 9\cdot\frac{C^2}{16} + \frac{C^2}{4} \approx 6.6C^2$$

Bottleneck 约省 8 倍参数，代价是需要 3 个 conv 操作。

---

### Inverted Bottleneck Block（MobileNet v2 / ConvNeXt）

三个卷积，但通道方向**反过来**：先扩展再压缩。Depthwise 本来就便宜（无 $C^2$ 项），不需要先压缩。

```
输入 x: [B, C, D, H, W]
│
├─ Depthwise Conv k×k×k (C → C)  ← 空间混合，每通道独立，参数 k³C
├─ LayerNorm
│
├─ Conv 1×1×1 (C → 4C)           ← 扩展通道（升维），参数 4C²
├─ GELU
│
├─ Conv 1×1×1 (4C → C)           ← 压回来（降维），参数 4C²
│
└─ + x (残差连接)                  ← 直接相加，输入输出通道均为 C
   输出: [B, C, D, H, W]
```

**参数量**：$k^3C + 4C^2 + 4C^2 = k^3C + 8C^2$

ConvNeXt 取 k=7，MedNeXt 3D 版本取 k=3/5/7 可配置。

**残差连接的条件**：输入输出形状完全一致（C 不变、分辨率不变）才能直接相加。下采样时残差路径需加 1×1 Conv + stride。

---

### 三种 Block 总览

```
Plain Block
  输入 x
  │
  ├─ Conv(3³) → Norm → Act
  ├─ Conv(3³) → Norm → Act
  │
  输出（无残差）
  
  └─ ResNet 变体：输出 + x → Act（残差版）

Bottleneck Block
  输入 x ──────────────────┐
  │                        │
  ├─ 1×1 Conv (C→C/4)     │  ← 压缩
  ├─ Norm → Act            │
  ├─ 3×3 Conv (C/4→C/4)   │  ← 空间混合（窄处）
  ├─ Norm → Act            │
  ├─ 1×1 Conv (C/4→C)     │  ← 恢复
  ├─ Norm                  │
  └─────── + ──────────────┘  ← 残差
            Act

Inverted Bottleneck Block（ConvNeXt/MedNeXt）
  输入 x ──────────────────────────┐
  │                                │
  ├─ k×k×k Depthwise (C→C)        │  ← 空间混合（不压缩）
  ├─ LayerNorm                     │
  ├─ 1×1 Conv (C→4C)              │  ← 扩展
  ├─ GELU                          │
  ├─ 1×1 Conv (4C→C)              │  ← 压缩
  └──────────── + ─────────────────┘  ← 残差
                输出
```

| | Plain Block | Bottleneck | Inverted Bottleneck |
|--|--|--|--|
| 卷积数 | 2 | 3 | 3 |
| 通道变化 | C→C→C | C→C/4→C | C→4C→C |
| 空间混合核 | 3×3×3（两次） | 3×3×3（窄处） | k×k×k Depthwise |
| Norm | IN/BN | BN | LN |
| 激活 | ReLU | ReLU | GELU |
| 残差 | 可选（ResNet变体有） | 有 | 有 |
| 代表网络 | nnUNet | ResNet-50 | ConvNeXt/MedNeXt |

---

### 下采样在哪里做

Block 本身只负责特征变换，下采样是**单独的操作**，三种网络放的位置不一样。

**nnUNet：下采样和通道变化在同一个 conv 里完成**

nnUNet 没有单独的下采样层，每个 stage 的**第一个 conv** 同时做两件事：stride=2 缩分辨率 + C_in→C_out 改通道。stage 内后续的 conv 通道不变。

```
输入: [B, 1, D, H, W]
│
Stage 0:
  Conv 3×3×3 (1→32, stride=1)    ← 通道变化（无下采样）
  Conv 3×3×3 (32→32, stride=1)   ← 通道不变
  输出: [B, 32, D, H, W]
│
Stage 1:
  Conv 3×3×3 (32→64, stride=2)   ← 通道变化 + 下采样（同一个conv）
  Conv 3×3×3 (64→64, stride=1)   ← 通道不变
  输出: [B, 64, D/2, H/2, W/2]
│
Stage 2:
  Conv 3×3×3 (64→128, stride=2)  ← 通道变化 + 下采样
  Conv 3×3×3 (128→128, stride=1)
  输出: [B, 128, D/4, H/4, W/4]
│
Stage 3:
  Conv 3×3×3 (128→256, stride=2)
  Conv 3×3×3 (256→256, stride=1)
  输出: [B, 256, D/8, H/8, W/8]
│
Stage 4（UNet Bottleneck 层）:
  Conv 3×3×3 (256→320, stride=2)
  Conv 3×3×3 (320→320, stride=1)
  输出: [B, 320, D/16, H/16, W/16]
```

**通道变化规律**：每进一个 stage 变一次（第一个 conv），stage 内部不变。不是只变一次。

**ConvNeXt / MedNeXt：stage 之间用 LN + strided Conv**

```
Stage 1
  Inverted Bottleneck Block × N   ← block 内分辨率不变
  ↓
  LayerNorm + Conv 2×2 stride=2 (C→2C)   ← 单独下采样层
  ↓
Stage 2
  Inverted Bottleneck Block × N
  ↓
  LayerNorm + Conv 2×2 stride=2 (2C→4C)
  ↓
Stage 3 ...
```

加 LayerNorm 是因为 ConvNeXt 全程用 LN，下采样前先归一化，训练更稳。

**ResNet：下采样塞在 stage 第一个 block 里**

```
Stage 1
  Bottleneck Block × N（分辨率不变）
  ↓
Stage 2 第一个 block（特殊）：
  ┌─ 1×1 Conv stride=2 (C→C/2)    ← 顺带下采样
  ├─ 3×3 Conv
  ├─ 1×1 Conv (C/2→2C)
  │
  残差路径：1×1 Conv stride=2 (C→2C)   ← 必须跟着下采样才能对齐形状
  └─ +
  
  Bottleneck Block × (N-1)（分辨率不变）
```

ResNet 把下采样揉进 block 里，导致 stage 第一个 block 的残差路径必须加 1×1 Conv 对齐，结构比较混。

---

**三种下采样方式对比**

| | nnUNet | ConvNeXt/MedNeXt | ResNet |
|--|--|--|--|
| 下采样位置 | stage 之间，单独 strided conv | stage 之间，LN + strided conv | block 内部 stride=2（第一个 block）|
| block 内改分辨率 | 否 | 否 | 是（每个 stage 第一个 block）|
| 残差需要 1×1 对齐 | 否（block 不改通道） | 否（block 不改通道） | 是（每个 stage 第一个 block）|
| 设计是否干净 | ✅ 特征变换/下采样分离 | ✅ 特征变换/下采样分离 | ❌ 揉在一起 |

ConvNeXt/nnUNet 把特征变换和下采样分离，结构更清晰；ResNet 揉在一起，是历史遗留设计。

---

## 5. 参数预算重分配思路的演进脉络

同一个思路从 2012 年到现在一路演进，每一代都在问同一个问题：**怎么在不涨参数的前提下扩大感受野或增加容量**。

---

### 2012 — Grouped Conv（AlexNet）

最早的分组思想，把 C 个通道分成 G 组，每组独立做卷积：

$$\text{参数} = k^3 \times \frac{C}{G} \times \frac{C_{out}}{G} \times G = \frac{k^3 C^2}{G}$$

G=C 时每组只有 1 个通道，即每通道独立卷积，就是 Depthwise Conv 的极端特例（参数最少）。AlexNet 当时用分组是因为两块 GPU 权重显存不够，工程妥协，歪打正着发现参数少了。

> 注意：分组卷积只省权重显存（权重量 ÷ G），激活 feature map 形状不变，不省激活显存。

---

### 2014 — Depthwise Separable Conv（Sifre & Mallat）

正式提出空间混合和通道混合解耦：

```
Depthwise：每通道独立做 k×k 空间卷积   → k²C
Pointwise：1×1 做通道混合              → C²
合计：k²C + C²  vs  标准 k²C²
```

核心发现：C 够大时，解耦后参数量 ≈ 标准的 1/k²。

---

### 2017 — MobileNet v1（Howard et al., Google）

第一个把 Depthwise Separable Conv 系统化用于整个网络的工作，目标是移动端部署。所有 conv 层全部替换，参数量降到原来的 $1/8 \sim 1/9$，准确率只损失 1%。

---

### 2017 — Xception（Chollet, Google）

"Extreme Inception"：把 Inception 里的多分支结构推到极端，全网络用 Depthwise Separable Conv，并发现**先做 Pointwise 再做 Depthwise**（与 MobileNet 顺序相反）效果更好，对应不同的假设：先混通道再混空间。

---

### 2018 — MobileNet v2（Sandler et al., Google）

引入 **Inverted Residual Bottleneck**（反向瓶颈）：

```
标准 bottleneck：宽 → 窄 → 宽（先压缩再扩展）
Inverted：       窄 → 宽 → 窄（先扩展再压缩）

具体：
1×1 Conv C → tC   （扩展，t=6）
Depthwise tC → tC
1×1 Conv tC → C   （压缩）
```

Depthwise 在高维空间做，信息损失更少。这个结构后来被 ConvNeXt 和 MedNeXt 直接沿用。

---

### 2019 — EfficientNet（Tan & Le, Google）

不改结构，改**缩放策略**：同时按比例扩深度/宽度/分辨率（compound scaling），底层 block 仍是 MobileNet v2 的 Inverted Residual。发现三个维度同步扩比单独扩一个效率高得多。

---

### 2021 — RepLKNet（Ding et al.）

第一个在 2D 视觉里系统验证**超大核**（31×31）的工作。用结构重参数化（训练时多分支，推理时合并成单卷积）实现大核，正式证明大核卷积能媲美 Transformer 的全局感受野。

---

### 2022 — ConvNeXt（Liu et al., Meta/FAIR）

集大成：把 Transformer block 的所有设计决策（LN、GELU、Inverted Bottleneck、大核 Depthwise）全部移植进纯卷积网络，在 ImageNet 上与 Swin Transformer 持平。

```
7×7 Depthwise  ← 大核，对齐 Attention 感受野
LN             ← 替换 BN
GELU           ← 替换 ReLU
C → 4C → C    ← Inverted Bottleneck（来自 MobileNet v2）
```

---

### 2023 — MedNeXt（Roy et al., MICCAI）

ConvNeXt 的 3D 医学图像版本，核心贡献是验证了大核（k=3/5/7 可选）在 3D 分割里同样有效，并设计了 3D Depthwise 转置卷积用于上采样。

---

### 演进总结

```
AlexNet（2012）    → 分组卷积，参数/G
    ↓
Depthwise Sep（2014）→ G=C，彻底解耦空间/通道
    ↓
MobileNet v1（2017）→ 全网络系统化应用
    ↓
MobileNet v2（2018）→ Inverted Bottleneck，高维 Depthwise
    ↓
EfficientNet（2019）→ 复合缩放，极致效率
    ↓
RepLKNet（2021）   → 超大核（31×31），结构重参数化
    ↓
ConvNeXt（2022）   → 大核+LN+GELU，对齐 Transformer
    ↓
MedNeXt（2023）    → 3D 化，用于医学图像分割
```

每一步都在回答同一个问题：**省下来的参数额度，怎么花最值**。

| 时间 | 工作 | 核心贡献 | 省下的额度怎么花 |
|------|------|---------|----------------|
| 2012 | AlexNet Grouped Conv | 参数 ÷ G，歪打正着 | 塞进两块 GPU |
| 2014 | Depthwise Separable | 正式解耦空间/通道 | 堆更多层 |
| 2017 | MobileNet v1 | 全网络系统化应用 | 移动端部署 |
| 2018 | MobileNet v2 | Inverted Bottleneck | 高维空间做 Depthwise，信息损失更少 |
| 2019 | EfficientNet | 复合缩放 | 深度/宽度/分辨率三维同步扩 |
| 2021 | RepLKNet | 结构重参数化 | 把核做到 31×31 |
| 2022 | ConvNeXt | 大核+LN+GELU | 对齐 Transformer 感受野 |
| 2023 | MedNeXt | 3D 化 + 可配置核大小 | 3D 医学图像大感受野分割 |

---

## 5. 核心设计思路：参数预算重分配

标准卷积的参数全部堆在一个操作里：空间混合 + 通道混合 **同时完成**，导致参数量是 $k^3 \times C^2$，想扩大感受野（增大 k）代价是三次方增长，根本用不起大核。

**解耦的本质是把参数预算拆开花**：

```
标准 conv：把所有预算堆在一次操作
  → 空间 × 通道 耦合，k 一大全炸

解耦后：
  Depthwise（空间混合）：预算 k³C，感受野大，但参数少（无通道交叉）
  Pointwise（通道混合）：预算 C²，  但 k=1，不占感受野的钱

两部分各司其职，互不干扰
```

节省出来的参数额度有两种用法：

**用法一：扩大感受野（MedNeXt 的选择）**
```
原来：3×3×3 标准 conv，27C²
现在：7×7×7 Depthwise + Pointwise，8C² + 343C
→ 感受野 3→7，参数反而少了（C≥19 时）
```

**用法二：堆更多层/更宽（MobileNet 的选择）**
```
同样参数预算下，解耦后能堆 3~5 倍数量的 block
→ 网络更深，特征更丰富
```

**一句话总结**：  
> 解耦不是目的，是手段——把空间混合和通道混合的参数开销分开，才能在不超预算的前提下把核做大、把网络做深。普通卷积做不到大核，根本原因是没有解耦，参数全被空间×通道的耦合项吃掉了。

---

## 5. 三者对比总结

```
普通卷积
  感受野：小（3×3×3）
  参数：O(C² × k³)
  瓶颈：想大核就爆参数

       ↓ 解耦空间混合和通道混合

ConvNeXt（2D）
  感受野：大（7×7 Depthwise）
  参数：O(C × k²) + O(C² × 1)
  创新：把 Transformer 的设计塞进卷积

       ↓ 直接搬到 3D + 医学图像适配

MedNeXt（3D）
  感受野：大（7×7×7 Depthwise）
  参数：O(C × k³) + O(C² × 1)
  适配：多规模(S/B/M/L)，3D 上下采样
```

---

## 5. 和你的 MLAUNet 的关系

你的模型用 MLA + MoE 解决同一个问题：**在有限计算量下覆盖更多尺度和模式**。

| 方法 | 解决感受野问题的手段 |
|------|-------------------|
| MedNeXt | Depthwise 大核（结构约束） |
| SwinUNETR | 窗口 Attention（分块计算） |
| 你的 MLAUNet | MLA 多尺度 + MoE 路由（动态分配） |

**潜在改进方向**：在 MoE 的每个 expert 内部用不同 k 的 Depthwise 大核（k=3/5/7），expert 天然关注不同尺度，兼容两个思路。
