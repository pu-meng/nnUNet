# 架构设计总览

## 目录

- [0. 方法来源与选型原则](#0-方法来源与选型原则)
  - [为什么选 MLA 和 MoE 而不是从论文里找](#为什么选-mla-和-moe-而不是从论文里找)
  - [后续可用的工程验证方法清单](#后续可用的工程验证方法清单)
- [0.1 各模型的设计出发点](#01-各模型的设计出发点)
- [MLAUNetBot3D 架构详解](#mlaunetbot3d-架构详解)
  - [1. Baseline（PlainConvUNet）架构](#1-baselineplainconvunet架构)
  - [2. MLA 的修改：瓶颈替换](#2-mla-的修改瓶颈替换)
  - [3. 完整架构图](#3-完整架构图)
  - [4. Baseline vs MLA 对比总结](#4-baseline-vs-mla-对比总结)
  - [5. MLA-MoE：MoE-FFN](#5-mla-moemoe-ffn)
  - [6. 代码对应关系](#6-代码对应关系)
- [7. SwinUNETR-B](#7-swinunetr-b)
- [8. MedNeXt-L](#8-mednext-l)
- [9. 五模型横向对比](#9-五模型横向对比)
- [10. 代码对应关系（完整）](#10-代码对应关系完整)
- [11. 实验验证：MLA 和 MoE 的效果分析](#11-实验验证mla-和-moe-的效果分析)
- [12. nnFormer](#12-nnformer)
- [13. Conv 和 Transformer 的真实局限，以及架构改动到底有没有用](#13-conv-和-transformer-的真实局限以及架构改动到底有没有用)

---

## 0. 方法来源与选型原则

### 为什么选 MLA 和 MoE 而不是从论文里找

在尝试了大量学术论文的方法（指标均无法提升）之后，转换思路：**优先参考大公司经过市场和生产环境验证的工程方法，而非直接跟随学术论文**。

论文的局限：
- 人人自称 SOTA，但结果往往是特定数据集 + 精细调参 + 选择性报告的产物
- 方法是否真正通用、是否在不同场景下稳健，论文无法保证
- 复现难度大，超参敏感，换个数据集就可能失效

市场验证的方法为什么更可信：
- 大公司（DeepSeek、OpenAI、Google 等）的方法需要在真实大规模生产中持续运行
- 被数百万用户和下游开发者压测过，不可靠的方法无法在市场上存活
- DeepSeek 的 MLA（低秩 KV 压缩）和 MoE（混合专家路由）是为了解决**真实的显存和计算瓶颈**而设计，不是为了刷 benchmark

**实际结果验证了这个判断**：从 DeepSeek-V2/V3 技术报告里找到 MLA 和 MoE，适配到医学图像分割瓶颈层后，MLA 产生了明确的指标提升（Overall +0.021，Precision +0.098）。

**后续选方法的原则**：遇到瓶颈时，优先查看 DeepSeek、Meta（LLaMA）、Google（Gemini）等公司的技术报告和开源代码，将这些经过工程验证的方法适配到医学图像分割任务。

### 后续可用的工程验证方法清单

**DeepSeek 里已用完的：**

| 技术 | 状态 |
|------|------|
| MLA（低秩KV压缩） | ✅ 已用，有效 |
| MoE（专家路由FFN） | ✅ 已用，效果边际 |
| Loss-free 负载均衡 | ✅ 已用（MoE 内置） |
| FP8混合精度训练 | ⚠️ 只加速训练，不提升指标 |
| Multi-Token Prediction | ❌ 自回归专用，不适用 |

**其他开源来源，未来可以试的：**

| 技术 | 来源 | 核心思路 | 适用性 | 优先级 |
|------|------|---------|--------|--------|
| **Flash Attention** | 斯坦福/PyTorch内置 | 把手写attention换成`F.scaled_dot_product_attention`，显存和速度同时提升 | ✅ 直接可用，零成本 | ⭐⭐⭐ 立刻做 |
| **GQA（Grouped Query Attention）** | Meta LLaMA-2/3 | 把Q头分组，组内共享KV，比MLA更温和的压缩 | ✅ 可替换现有MLA | ⭐⭐ |
| **3D-RoPE（旋转位置编码）** | Meta LLaMA系列 | 比学习式相对位置偏置更好，不需额外参数，3D扩展后能编码体素空间关系 | ✅ 可加入MLA/nnFormer | ⭐⭐ |
| **Mamba-2（State Space Duality）** | 普林斯顿/CMU | 比Mamba-1计算更高效、理论更完整 | ✅ 升级现有UMambaBot3D | ⭐⭐ |
| **KAN（Kolmogorov-Arnold Network）** | MIT | 用样条函数替换FFN的线性层，理论上更强的表达能力 | ⚠️ 很新，未经大规模验证 | ⭐ |

**已不再开源的：**

| 公司 | 状态 | 说明 |
|------|------|------|
| OpenAI（GPT-4o/o1/o3） | ❌ 不开源，不写架构论文 | 只有API，架构完全不透明 |
| Anthropic（Claude） | ❌ 不开源，不写架构论文 | 同上 |
| Google（Gemini） | ⚠️ 有论文但细节有限 | Gemma开源但能力弱于Gemini |

**结论**：真正可用的工程来源就是 DeepSeek、Meta（LLaMA）、Mistral、Mamba 系列。DeepSeek 的技术报告是目前公开最详细的，Meta 次之。OpenAI/Anthropic 已经退出开源生态，盯着他们没有意义。

---

## 0.1 各模型的设计出发点

Baseline（nnUNet）的局限是标准 Conv3d 的两个固有问题：

**问题1：Conv3d 的感受野是局部的**
3×3×3 核只能看到 27 个体素的邻域，无法捕捉远距离依赖（如肝脏轮廓和肿瘤位置的全局关系）。层叠多层可以间接扩大感受野，但深层信息传递效率低。

**问题2：Conv3d 的参数效率低**
空间混合（看邻域）和通道混合（跨通道加权）耦合在同一个权重里，参数量 = C_out × C_in × 27，其中大量参数花在"不同通道之间的空间关系"上，实际有效信息密度不高。

各模型从不同角度解决这两个问题：

```
                    改进 Conv 本身        引入全局感受野
                         │                     │
              ┌──────────┴──────┐    ┌──────────┴──────────┐
              │                 │    │                      │
         MedNeXt           MLA/MoE              SwinUNETR
    ConvNeXt风格block     瓶颈加全局Attn      全Transformer Encoder
    分离空间/通道混合     其余仍是局部Conv      窗口Attention
    更好的参数效率        最小改动代价最小      最彻底的架构替换
```

| 模型 | 解决的问题 | 手段 | 改动范围 |
|------|-----------|------|---------|
| **Baseline** | — | 标准 Conv3d×2 | — |
| **MedNeXt** | Conv 参数效率低 | DWConv+1×1分离空间/通道，GELU+残差 | Encoder+Decoder 全部替换 |
| **MLA-MoE** | 瓶颈感受野局部 | 瓶颈插入全局 Attention（MLA）+ 专家路由（MoE） | 只改瓶颈一处 |
| **SwinUNETR** | Encoder 感受野局部 | 整个 Encoder 换成 Swin Transformer | Encoder 全部替换，Decoder 保留 CNN |

**代价与收益的权衡：**
- MedNeXt 和 SwinUNETR 改动范围大，参数量翻倍（~62M vs ~30M），但也引入了更多归纳偏置
- MLA-MoE 只动一处，参数增量最小（~33M），但全局感受野仅限于瓶颈的 4³=64 个 token
- 实验结果显示改动最小的 MLA 就已经有明显提升（Overall +0.021），更大的架构改动不一定带来正比例的收益

---

# MLAUNetBot3D 架构详解

## 一句话总结

**MLAUNetBot3D = nnUNet 的 Baseline（PlainConvUNet）+ 瓶颈处把 Conv 换成 MLA Transformer**

除了瓶颈这一处修改，**Encoder、Decoder、Skip Connection、Loss、Data Augmentation 全部与 Baseline 完全一致**。

---

## 1. Baseline（PlainConvUNet）架构

### 1.1 整体结构

```
Input (1, 128³)
    │
    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Encoder（PlainConvEncoder）                                        │
│  6 个 stage，每个 stage = StackedConvBlocks(Conv×2)                 │
│  stride=2 的 conv 做下采样（不是 pooling）                          │
│  每层输出同时作为 skip connection 存起来                             │
└─────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Bottleneck（最深层 Enc-5 的输出）                                  │
│  Baseline: 就是 Enc-5 的 Conv×2 输出，直接传给 Decoder              │
│  MLA:      在 Enc-5 和 Decoder 之间插入 MLABottleneck3D             │
└─────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Decoder（UNetDecoder）                                             │
│  5 个 stage，每个 stage = 上采样 + cat(skip) + Conv×2               │
│  上采样用转置卷积（stride=2）                                       │
└─────────────────────────────────────────────────────────────────────┘
    │
    ▼
Output (3, 128³)  ← 1×1×1 conv 映射到 3 类（背景/肝脏/肿瘤）
```

### 1.2 Encoder 的 Conv Block 细节

每个 Encoder stage 内部是一个 `StackedConvBlocks`，包含 **2 个 `ConvDropoutNormReLU`**：

```
StackedConvBlocks(num_convs=2):
    ┌─────────────────────────────────────────────┐
    │ ConvDropoutNormReLU #1（stride=2 或 1）      │
    │   Conv3d(k=3×3×3, stride=s, padding=1)      │
    │   → InstanceNorm3d                           │
    │   → LeakyReLU(negative_slope=1e-2)           │
    ├─────────────────────────────────────────────┤
    │ ConvDropoutNormReLU #2（stride=1）            │
    │   Conv3d(k=3×3×3, stride=1, padding=1)      │
    │   → InstanceNorm3d                           │
    │   → LeakyReLU(negative_slope=1e-2)           │
    └─────────────────────────────────────────────┘
```

**关键点**：
- 所有 conv 都是 **3×3×3**，padding=1（保持空间尺寸）
- 下采样靠 **stride=2 的 conv** 实现，不是 pooling
- 激活函数是 **LeakyReLU**（不是 ReLU）
- 归一化是 **InstanceNorm3d**（不是 BatchNorm）
- 没有 Dropout（dropout_op=None）

### 1.3 Decoder 的细节

```
UNetDecoder stage:
    ┌─────────────────────────────────────────────┐
    │ 转置卷积（上采样，stride=2）                  │
    │   ConvTranspose3d(k=2×2×2, stride=2)        │
    │   通道数减半（例如 320→160）                  │
    ├─────────────────────────────────────────────┤
    │ cat(skip)                                    │
    │   上采样结果 (160ch) + skip (320ch) → 480ch  │
    ├─────────────────────────────────────────────┤
    │ StackedConvBlocks(Conv×2)                    │
    │   Conv3d(k=3×3×3) × 2                        │
    │   480ch → 320ch（降回目标通道数）              │
    └─────────────────────────────────────────────┘
```

### 1.4 各层参数表（Dataset003_Liver 的 nnUNetPlans）

| 层 | 输入通道 | 输出通道 | 分辨率 | stride | Conv 数 | 说明 |
|----|---------|---------|--------|--------|---------|------|
| Enc-0 | 1 | 32 | 128³ | 1 | 2 | 第一层 stride=1，不降分辨率 |
| Enc-1 | 32 | 64 | 64³ | 2 | 2 | |
| Enc-2 | 64 | 128 | 32³ | 2 | 2 | |
| Enc-3 | 128 | 256 | 16³ | 2 | 2 | |
| Enc-4 | 256 | 320 | 8³ | 2 | 2 | |
| **Enc-5** | 320 | **320** | **4³** | 2 | 2 | **最深层，瓶颈位置** |
| Dec-4 | 320+320=640 | 320 | 8³ | — | 2 | cat skip-4 后降回 320 |
| Dec-3 | 320+256=576 | 256 | 16³ | — | 2 | |
| Dec-2 | 256+128=384 | 128 | 32³ | — | 2 | |
| Dec-1 | 128+64=192 | 64 | 64³ | — | 2 | |
| Dec-0 | 64+32=96 | 32 | 128³ | — | 2 | |
| Output | 32 | 3 | 128³ | — | 1×1×1 conv | 映射到 3 类 |

---

## 2. MLA 的修改：瓶颈替换

### 2.1 修改位置

代码 `mla_unetr.py:227-229`：

```python
def forward(self, x: torch.Tensor):
    skips = self.encoder(x)          # ← 完全复用 Baseline 的 Encoder
    skips[-1] = self.mla_bot(skips[-1])  # ← 唯一修改：对最深层特征做 MLA
    return self.decoder(skips)        # ← 完全复用 Baseline 的 Decoder
```

### 2.2 Baseline vs MLA 瓶颈对比

```
Baseline 瓶颈（Enc-5 输出）:
    (B, 320, 4, 4, 4)  ← 就是 Conv×2 的输出
         │
         ▼ 直接传给 Decoder（不做任何额外处理）

MLA 瓶颈（Enc-5 输出 → MLABottleneck3D）:
    (B, 320, 4, 4, 4)
         │
         ▼
    ┌─────────────────────────────────────────────┐
    │ 1. flatten: (B, 64, 320)                    │
    │    4×4×4 = 64 个 token，每个 320 维          │
    ├─────────────────────────────────────────────┤
    │ 2. MLATransformerBlock × 2（默认 2 层）      │
    │    ┌───────────────────────────────────┐    │
    │    │ LayerNorm                          │    │
    │    │ MultiHeadLatentAttention(MLA)      │    │
    │    │   W_Q: 320→320 (Query)             │    │
    │    │   W_DKV: 320→80 (KV 压缩)          │    │
    │    │   W_UK: 80→320 (K 上投影)          │    │
    │    │   W_UV: 80→320 (V 上投影)          │    │
    │    │   Attention: 64×64 full attention  │    │
    │    │   W_O: 320→320 (输出投影)          │    │
    │    │ + residual                         │    │
    │    ├───────────────────────────────────┤    │
    │    │ LayerNorm                          │    │
    │    │ FFN: 320→1280→320 (MLP ratio=4)   │    │
    │    │   Linear → GELU → Dropout → Linear │    │
    │    │ + residual                         │    │
    │    └───────────────────────────────────┘    │
    ├─────────────────────────────────────────────┤
    │ 3. reshape: (B, 320, 4, 4, 4)              │
    └─────────────────────────────────────────────┘
         │
         ▼ 传给 Decoder（与 Baseline 完全相同的后续流程）
```

### 2.3 MLA 的核心创新：低秩 KV 压缩

标准 Self-Attention 和 MLA 的对比：

```
标准 MHA（Multi-Head Attention）:
    Q = x·W_Q    (320→320)
    K = x·W_K    (320→320)  ← 每个 token 存 320 维
    V = x·W_V    (320→320)  ← 每个 token 存 320 维
    KV 存储: 2 × 64 × 320 = 40,960 个 float

MLA（Multi-head Latent Attention）:
    Q = x·W_Q    (320→320)
    c = x·W_DKV  (320→80)   ← 先压缩到 80 维潜变量（Down）
    K = c·W_UK   (80→320)   ← 从 c 还原出 Key（Up-K）
    V = c·W_UV   (80→320)   ← 从 c 还原出 Value（Up-V）
    KV 存储: 1 × 64 × 80 = 5,120 个 float（只有 c）
    K 和 V 在计算 attention 时即时展开，不存储
    KV 显存节省: 40,960 → 5,120 = 87.5% ↓
```

**三个投影矩阵各自的作用：**

- **W_DKV（Down KV，压缩）**：把每个 token 的 320 维表示压缩为 80 维潜变量 `c_kv`。`c_kv` 是 K 和 V 的**共享信息瓶颈**——K 和 V 虽然用途不同，但它们描述的是同一个 token 的内容，存在冗余，可以先提炼成低秩表示再分别解码。

- **W_UK（Up K，还原 Key）**：从 80 维 `c_kv` 解码出 320 维 Key。Key 的职责是**被 Query 打分**（dot-product 相似度），决定"哪些 token 值得关注"。

- **W_UV（Up V，还原 Value）**：从 80 维 `c_kv` 解码出 320 维 Value。Value 的职责是**提供实际内容**，attention 权重加权求和的是 V，最终输出是各 token Value 的加权组合。

W_UK 和 W_UV 从同一个 `c_kv` 出发但学习不同的解码方向：一个让 token 易于被比较，一个让 token 易于被聚合。

**关键 insight**：MLA 的 attention 矩阵仍然是 **64×64 的 full attention**（全局感受野），但 KV 的存储从 O(N·d) 降到 O(N·d_c)。在 4³=64 个 token 的场景下这个优势不明显，但这借鉴自 DeepSeek-V2——在 LLM 推理中 KV cache 是显存瓶颈，MLA 把百亿参数模型的 KV cache 节省 87.5%。

### 2.4 参数量对比

| 组件 | Baseline | MLA | 差异 |
|------|----------|-----|------|
| Encoder | 同 | 同 | 完全相同 |
| 瓶颈 Conv×2 | 320→320, 3×3×3, 2 层 | **移除** | MLA 替换了这 2 层 conv |
| MLA 模块 | 无 | W_Q+W_DKV+W_UK+W_UV+W_O+FFN×2 | 新增 |
| Decoder | 同 | 同 | 完全相同 |
| **总参数量** | ~30M | ~31M（略多 ~3%） | 几乎不变 |

---

## 3. 完整架构图

```
Input (1, 128³)
    │
    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ Encoder（PlainConvEncoder，与 Baseline 完全一致）                            │
│                                                                              │
│  Enc-0: Conv3d×2 (1→32,  k=3, s=1)   输出: 32ch, 128³  ──── skip-0 ────┐  │
│  Enc-1: Conv3d×2 (32→64,  k=3, s=2)  输出: 64ch,  64³  ──── skip-1 ────┤  │
│  Enc-2: Conv3d×2 (64→128, k=3, s=2)  输出: 128ch, 32³  ──── skip-2 ────┤  │
│  Enc-3: Conv3d×2 (128→256,k=3, s=2)  输出: 256ch, 16³  ──── skip-3 ────┤  │
│  Enc-4: Conv3d×2 (256→320,k=3, s=2)  输出: 320ch,  8³  ──── skip-4 ────┤  │
│  Enc-5: Conv3d×2 (320→320,k=3, s=2)  输出: 320ch,  4³  ──── skip-5 ────┤  │
└──────────────────────────────────────────────────────────────────────────────┘
    │  skip-5 (320ch, 4³)
    ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║  ★ MLA Bottleneck（唯一与 Baseline 不同的地方）                              ║
║                                                                             ║
║  输入: (B, 320, 4, 4, 4)                                                    ║
║      │                                                                      ║
║      ▼                                                                      ║
║  flatten → (B, 64, 320)    ← 4×4×4=64 个 token                             ║
║      │                                                                      ║
║      ▼                                                                      ║
║  ┌──────────────────────────────────────────────────────────────────────┐   ║
║  │ MLA Block #1                                                         │   ║
║  │   LayerNorm → MLA(8 heads, d_c=80) → residual +                      │   ║
║  │   LayerNorm → FFN(320→1280→320, GELU) → residual +                   │   ║
║  └──────────────────────────────────────────────────────────────────────┘   ║
║      │                                                                      ║
║      ▼                                                                      ║
║  ┌──────────────────────────────────────────────────────────────────────┐   ║
║  │ MLA Block #2                                                         │   ║
║  │   LayerNorm → MLA(8 heads, d_c=80) → residual +                      │   ║
║  │   LayerNorm → FFN(320→1280→320, GELU) → residual +                   │   ║
║  └──────────────────────────────────────────────────────────────────────┘   ║
║      │                                                                      ║
║      ▼                                                                      ║
║  LayerNorm → reshape → (B, 320, 4, 4, 4)                                   ║
╚══════════════════════════════════════════════════════════════════════════════╝
    │  (B, 320, 4, 4, 4)
    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ Decoder（UNetDecoder，与 Baseline 完全一致）                                 │
│                                                                              │
│  Dec-4: 上采样(s=2) → cat(skip-4:320ch) → Conv3d×2 (640→320) → 320ch, 8³  │
│  Dec-3: 上采样(s=2) → cat(skip-3:256ch) → Conv3d×2 (576→256) → 256ch,16³  │
│  Dec-2: 上采样(s=2) → cat(skip-2:128ch) → Conv3d×2 (384→128) → 128ch,32³  │
│  Dec-1: 上采样(s=2) → cat(skip-1:64ch)  → Conv3d×2 (192→64)  → 64ch, 64³  │
│  Dec-0: 上采样(s=2) → cat(skip-0:32ch)  → Conv3d×2 (96→32)   → 32ch,128³  │
│  Output: Conv1×1×1 (32→3)                                                   │
└──────────────────────────────────────────────────────────────────────────────┘
    │
    ▼
Output (3, 128³)  ← 背景(0) / 肝脏(1) / 肿瘤(2)
```

---

## 4. Baseline vs MLA 对比总结

| 方面 | Baseline (PlainConvUNet) | MLAUNetBot3D |
|------|------------------------|--------------|
| **Encoder** | Conv3d×2 per stage | **完全相同** |
| **Decoder** | 上采样 + cat(skip) + Conv3d×2 | **完全相同** |
| **Skip Connection** | Encoder 每层输出 cat 到 Decoder | **完全相同** |
| **瓶颈（最深层）** | Conv3d×2 (局部 3×3×3) | **MLA Transformer ×2（全局 attention）** |
| **瓶颈感受野** | 局部（3×3×3=27 体素） | **全局（4³=64 个 token 互相看）** |
| **瓶颈参数量** | 2 × (320×320×27) ≈ 5.5M | MLA: ~1.2M + FFN: ~1.6M ≈ 2.8M |
| **Loss** | Dice + CE | **完全相同** |
| **Data Augmentation** | 旋转/缩放/镜像/噪声/亮度/对比度/Gamma | **完全相同** |
| **训练超参** | 1000 epoch, lr=1e-2, SGD | **完全相同** |
| **总参数量** | ~30M | ~31M（+3%） |

### 4.1 修改量统计

```
Baseline 代码行数（PlainConvUNet + Encoder + Decoder + Conv blocks）: ~600 行
MLA 新增代码行数（MLAUNetBot3D + MLABottleneck3D + MLATransformerBlock + MultiHeadLatentAttention）: ~200 行

修改比例: ~200 / (600+200) ≈ 25% 的新代码
但修改位置: 只有 1 处（瓶颈）
```

### 4.2 为什么只在瓶颈做 MLA？

1. **计算效率**：4³=64 个 token，attention 矩阵 64×64=4,096，非常小。如果在 128³ 做 attention，token 数=2,097,152，attention 矩阵 4 万亿，完全不可行
2. **语义层次**：Encoder 浅层（Enc-0~Enc-3）提取的是边缘/纹理等局部特征，不需要全局 attention。最深层（Enc-5）特征语义最丰富，全局上下文最有价值
3. **与 UMamba 对称**：UMambaBot3D 也是在瓶颈插入 Mamba 块，MLA 放在同一位置，方便公平对比

---

## 5. MLA-MoE：MoE-FFN

### 5.1 一句话总结

**MLA-MoE = 普通 Conv Encoder + MoE-FFN 瓶颈 + 普通 Conv Decoder（只动瓶颈一处）**

与 MedNeXt（全部替换）、SwinUNETR（替换 Encoder）不同，MLA-MoE 只在瓶颈的 `MLATransformerBlock` 里使用 MoEFFN，Encoder 和 Decoder 全部是普通 Conv block，与 Baseline 完全一致。

当前代码中 `MLATransformerBlock` 始终使用 `MoEFFN`（`mla_unetr.py:212`），不存在"标准 FFN"版本。`nnUNetTrainer_MLAUNet` 和 `nnUNetTrainer_MLAUNet_MoE` 实际上调用同一套网络架构，两者的区别仅停留在 trainer 注释层面（历史遗留）。

### 5.2 MoE-FFN 结构

```
概念参考（标准 FFN，当前代码中未使用）:
    x → Linear(320→1280) → GELU → Linear(1280→320) → out

实际使用（MoE-FFN）:


    x (320)
     │
     ├──────────────────────────────────────────┐
     │                                          │
     ▼                                          ▼
  [Shared Expert]                       [Router]
  Linear(320 → 640) → GELU → Linear(640 → 320)    Linear(320 → 4)  ← 打分 4 个路由专家
  始终激活，无条件执行                      │
                                           ├── + expert_bias（loss-free 均衡偏置）
                                           ▼
                                      topk(scores, k=2)  → 选 top_2 专家
                                           │
                             ┌─────────────┴─────────────┐
                             ▼                           ▼
                      Expert_i (640d)             Expert_j (640d)
                    Linear(320→640)→GELU        Linear(320→640)→GELU
                    →Linear(640→320)            →Linear(640→320)
                             │                           │
                             └────── gate_weight ────────┘
                                  softmax(原始 scores，无 bias)
                                  加权求和 → routed_out (320)
     │
     ▼
    out = shared_out + routed_out  (320)
```

### 5.3 关键设计细节

**专家宽度减半：**
- 标准 FFN：d_ff = 320 × 4 = **1280**
- 每个 MoE 专家：d_ff = 320 × 4 // 2 = **640**（半宽）
- 每次 forward 激活：shared(640) + top_2 �uted(640×2) = **1920** 激活宽度
- 相比标准 FFN（1280），激活容量约 1.5×，但总参数量更多（存储了 4 个路由专家）

**路由打分机制（极简）：**

`router = nn.Linear(320, 4, bias=False)` 本质是一个 `(4, 320)` 的权重矩阵，每行是某个专家的"代表向量"：

```
score_e = dot(x_token, W_router[e])   # token(320维) · 专家e代表向量(320维) = 标量

四个专家同时算 = x @ W_router.T       # (BN, 320) @ (320, 4) → (BN, 4) 一次矩阵乘
```

没有非线性、没有 bias，就是内积：**token 与哪个专家的代表向量最对齐，就选哪个专家**。

**两套 score：路由 vs 门控分离：**
```python
scores         = router(x)                        # 原始打分（参与梯度）
routing_scores = scores + expert_bias.detach()    # 路由选择用带 bias 的（无梯度）
topk_idx       = routing_scores.topk(top_k)      # 决定选哪些专家
gate_weights   = softmax(scores.gather(topk_idx)) # 门控权重用原始 score（保证梯度干净）
```
- `expert_bias` 是 `register_buffer`（**不是 Parameter**），不进优化器
- 路由选择 ≠ 门控权重：选谁用 bias 修正，但加权多少用原始 score，梯度路径不含 bias

**Loss-free 负载均衡（`update_expert_bias`）：**
```
每个 train step：
  1. 统计 topk_idx 中各专家被选中次数 → counts
  2. load = counts / total_tokens
  3. EMA 更新：expert_load_ema = 0.99 × ema + 0.01 × load
  4. bias += (target - ema) × 1e-3
     target = 1/num_experts = 0.25（均匀分配）
     过载专家 → bias 下降 → 下次不那么容易被选中
     欠载专家 → bias 上升 → 下次更容易被选中
```
不像 auxiliary load-balancing loss（会干扰主任务梯度），这里通过 buffer 调整路由偏置，主任务梯度完全干净。

### 5.4 参数量对比（d_model=320，mlp_ratio=4）

| 组件 | 标准 FFN（MLA） | MoE-FFN（MLA-MoE） |
|------|----------------|-------------------|
| Shared Expert | 无 | 2 × 320 × 640 = 409,600 |
| 路由专家 × 4 | 无 | 4 × 2 × 320 × 640 = 1,638,400 |
| Router | 无 | 320 × 4 = 1,280 |
| 单 FFN | 2 × 320 × 1280 = 819,200 | 无 |
| **FFN 总参数（per block）** | **819,200** | **2,049,280（+150%）** |
| **每 token 激活参数** | **819,200** | **shared + top_2 = 3 × 409,600 ≈ 1,228,800** |
| **每 block 总参数（含 MLA）** | ~2.0M | ~3.2M |
| **瓶颈 2 blocks 总参数** | ~4.0M | ~6.4M |
| **模型总参数** | ~31M | ~33M |

### 5.5 MLATransformerBlock 实际结构

```
MLATransformerBlock（mla_unetr.py:193，唯一版本）:

    x
    │
    ├─ LayerNorm → MLA → + ─── x'
    │
    ├─ LayerNorm → MoEFFN（shared + top_2 of 4 routed，各 320→640→320）→ + ─── out
```

`self.ffn = MoEFFN(...)` 写死在 `__init__` 里，无标准 FFN 分支（`mla_unetr.py:212`）。

### 5.6 三模型对比总结

| 方面 | Baseline | MLA-UNet | **MLA-MoE** |
|------|----------|----------|-------------|
| 瓶颈处理 | Conv×2 | MLA+MoEFFN | **MLA+MoEFFN（同上）** |
| 全局感受野 | ✗ | ✓ | ✓ |
| 专家路由 | ✗ | ✓ | ✓ |
| 负载均衡 | ✗ | ✗ | **✓（loss-free bias）** |
| 瓶颈参数量 | ~5.5M | ~4.0M | **~6.4M** |
| 模型总参数 | ~30M | ~31M | **~33M** |
| Trainer | `nnUNetTrainer_Baseline` | `nnUNetTrainer_MLAUNet` | **`nnUNetTrainer_MLAUNet_MoE`** |

---

## 6. 代码对应关系

| 组件 | 文件 | 类/函数 |
|------|------|---------|
| Baseline 整体 | `dynamic_network_architectures/architectures/unet.py` | `PlainConvUNet` |
| Encoder | `.../building_blocks/plain_conv_encoder.py` | `PlainConvEncoder` |
| Decoder | `.../building_blocks/unet_decoder.py` | `UNetDecoder` |
| Conv Block | `.../building_blocks/simple_conv_blocks.py` | `StackedConvBlocks` + `ConvDropoutNormReLU` |
| **MLA Attention** | `pumengyu/architectures/mla_unetr.py` | `MultiHeadLatentAttention` |
| **单个 FFN 专家** | `pumengyu/architectures/mla_unetr.py` | `FFNExpert` |
| **MoE-FFN** | `pumengyu/architectures/mla_unetr.py` | `MoEFFN` |
| **MLA Block（含 MoE）** | `pumengyu/architectures/mla_unetr.py` | `MLATransformerBlock` |
| **MLA 瓶颈封装** | `pumengyu/architectures/mla_unetr.py` | `MLABottleneck3D` |
| **MLAUNet 整体** | `pumengyu/architectures/mla_unetr.py` | `MLAUNetBot3D` |
| MLA Trainer | `pumengyu/trainers/trainer.py` | `nnUNetTrainer_MLAUNet` |
| **MLA-MoE Trainer** | `pumengyu/trainers/trainer.py` | `nnUNetTrainer_MLAUNet_MoE` |

---

## 7. SwinUNETR-B

### 7.1 一句话总结

**SwinUNETR = 全 Transformer Encoder（Swin）+ CNN Decoder**

与 MLA-UNet 的根本区别：MLA 只在瓶颈一处用 Transformer，SwinUNETR 的**整个 Encoder 全是 Swin Transformer**，卷积仅出现在 Decoder 端。

### 7.2 整体结构

```
Input (1, 128³)
    │
    ▼  patch embedding（stride=2，把 2×2×2 体素映射为 1 个 token）
    (B, 64³, 48)  ← 64³ = 262,144 个 token，每个 48 维
    │
    ▼
┌────────────────────────────────────────────────────────┐
│  Swin Encoder（4 stage）                               │
│                                                        │
│  Stage 0: (B, 64³, 48),  heads=3,  depth=2  → skip-0  │
│      ↓ PatchMerging（2× 下采样，2×ch）                  │
│  Stage 1: (B, 32³, 96),  heads=6,  depth=2  → skip-1  │
│      ↓ PatchMerging                                    │
│  Stage 2: (B, 16³, 192), heads=12, depth=2  → skip-2  │
│      ↓ PatchMerging                                    │
│  Stage 3: (B, 8³, 384),  heads=24, depth=2  → skip-3  │
└────────────────────────────────────────────────────────┘
    │
    ▼  额外下采样 + Linear 投影
    (B, 4³, 768)  ← 瓶颈，768 = feature_size × 16
    │
    ▼
┌────────────────────────────────────────────────────────┐
│  CNN Decoder（5 stage，TransposeConv 上采样 + cat）     │
│                                                        │
│  Dec-3: upsample → cat(skip-3: 384) → Conv×2 → 384ch  │
│  Dec-2: upsample → cat(skip-2: 192) → Conv×2 → 192ch  │
│  Dec-1: upsample → cat(skip-1: 96)  → Conv×2 → 96ch   │
│  Dec-0: upsample → cat(skip-0: 48)  → Conv×2 → 48ch   │
│  Dec-e: upsample → cat(原始embedded)→ Conv×2 → 48ch   │
└────────────────────────────────────────────────────────┘
    │
    ▼  1×1×1 conv
Output (num_classes, 128³)
```

### 7.3 Swin Transformer Block

每个 Stage 内部包含 `depth` 个交替的 W-MSA / SW-MSA block：

```
Block #1（W-MSA，规则窗口）:
    LayerNorm → Window Partition（7×7×7）→ MSA → Window Reverse → + residual
    LayerNorm → FFN（Linear → GELU → Linear）→ + residual

Block #2（SW-MSA，移位窗口，shift=(3,3,3)）:
    LayerNorm → Cyclic Shift → Window Partition → Masked MSA → Reverse Shift → + residual
    LayerNorm → FFN → + residual
```

**Window Attention vs Full Attention 对比：**

| | MLA（瓶颈） | SwinUNETR Stage 0 |
|---|---|---|
| token 总数 | 64（4³） | 262,144（64³） |
| attention 范围 | 全局（64×64） | 局部窗口（7³=343 per window） |
| 窗口数量 | 1 | 64³/7³ ≈ 614 个窗口 |
| attention 矩阵大小 | 64×64 | 343×343 per window |

窗口 attention 让 SwinUNETR 可以处理高分辨率特征图而不爆显存，代价是每层只能看到局部邻域，靠"层叠 + shift"逐渐扩大感受野。

### 7.4 固定配置（SwinUNETR-B）

| 参数 | 值 | 含义 |
|------|-----|------|
| `patch_size` | 2 | token 化步长（2×2×2 voxel → 1 token） |
| `feature_size` | 48 | Stage 0 基础通道数 |
| `depths` | (2, 2, 2, 2) | 每个 Stage 的 Transformer block 数 |
| `num_heads` | (3, 6, 12, 24) | 每个 Stage 的注意力头数 |
| `window_size` | 7 | 注意力窗口边长（7³=343 个 token） |
| `norm_name` | 'instance' | Decoder CNN 使用 InstanceNorm |
| `use_checkpoint` | True | 梯度检查点（显存 ↓40%，速度 ↓20%） |

### 7.5 Deep Supervision

SwinUNETR **不含 Deep Supervision**，Decoder 只有最终一个输出头。
Trainer 中 `enable_deep_supervision=False`，`_get_deep_supervision_scales()` 返回 `None`。

### 7.6 参数量

| 组件 | 参数量 |
|------|--------|
| Patch Embedding | ~0.2M |
| Swin Encoder（4 stage × depth=2） | ~28M |
| CNN Decoder（5 stage） | ~35M |
| **总计** | **~63M** |

比 Baseline（~30M）多约 2×，比 MLA-MoE（~33M）多约 1.9×。

### 7.7 与 Baseline 的对比

| 方面 | Baseline | SwinUNETR-B |
|------|----------|-------------|
| Encoder 类型 | Conv3d×2 per stage | **Swin Transformer（全 Attention）** |
| Decoder 类型 | Conv3d×2 + cat | Conv3d×2 + cat（类似，但通道数不同） |
| 感受野 | 局部（3×3×3） | 局部窗口（7³）+ 跨层堆叠扩展 |
| Deep Supervision | ✓（5 尺度） | **✗** |
| `torch.compile` | ✓ | **✗**（与 use_checkpoint 冲突） |
| 参数量 | ~30M | **~63M（+110%）** |
| Trainer | `nnUNetTrainer_Baseline` | `nnUNetTrainer_SwinUNETR` |

---

## 8. MedNeXt-L

### 8.1 一句话总结

**MedNeXt = nnUNet 的 UNet 拓扑 + ConvNeXt 风格的 block（Encoder 和 Decoder 全部替换）**

整体 UNet 结构（Encoder-Bottleneck-Decoder + Skip）与 Baseline 一致，但 Encoder、Bottleneck、Decoder 每个位置的 block **全部**替换为 MedNeXt block（GroupNorm + DWConv + 膨胀 MLP + 残差）。下采样用 `MedNeXtDownBlock`，上采样用 `MedNeXtUpBlock`，两者均是 `MedNeXtBlock` 的子类（只把第一个卷积换成 stride=2 的普通/转置卷积）。

与 SwinUNETR 的关键区别：SwinUNETR 只替换 Encoder（Transformer），Decoder 仍是普通 CNN；MedNeXt 则是 Encoder 和 Decoder **都是** MedNeXt block，没有任何普通 Conv block。

### 8.2 归一化方式说明（BatchNorm / GroupNorm / InstanceNorm）

归一化都是 `(x - mean) / std`，区别在于 **mean/std 从哪些维度统计**，以及 **B 维度怎么处理**：

```
输入 shape: (B, C, D, H, W)

BatchNorm:
    对每个通道 c，跨 B x D x H x W 统计 mean/std
    B 维度参与统计 → 不同样本互相影响
    CT 灰度范围差异极大，跨 batch 统计不可靠，不适用

GroupNorm(num_groups=G):
    对每个样本 b 独立处理（B 维度完全隔离）
    把 C 通道分成 G 组，每组 C/G 个通道
    在 (C/G) x D x H x W 上统计 mean/std

InstanceNorm:
    对每个 (b, c) 独立处理（B 和 C 都隔离）
    只在 D x H x W 上统计 mean/std

MedNeXt 的 GroupNorm(num_groups=C):
    每组只有 1 个通道 (C/G = C/C = 1)
    在 1 x D x H x W 上统计 = 只在 D x H x W 上统计
    与 InstanceNorm 数学上完全等价
```

**B 维度的处理**：GroupNorm 和 InstanceNorm 都对每个样本完全独立，BatchNorm 才跨 B 混合。Baseline 用 `InstanceNorm3d`，MedNeXt 用 `GroupNorm(num_groups=C)`，两者等价。

### 8.3 普通 Conv3d、DWConv3d、可分离卷积的区别

**普通 Conv3d 的"空间混合"和"通道混合"：**

```
普通 Conv3d(C_in=2, C_out=1, k=3)，单个输出位置 (d, h, w)：

output[d,h,w] =
    input[ch=0, d+-1, h+-1, w+-1] * weight[out=0, ch=0, :,:,:]   <- 通道0的27个邻域位置
  + input[ch=1, d+-1, h+-1, w+-1] * weight[out=0, ch=1, :,:,:]   <- 通道1的27个邻域位置
  + bias

权重形状: (C_out, C_in, 3,3,3)，参数量 = C_out x C_in x 27
```

- **空间混合**：3x3x3 核把 27 个邻域位置聚合成 1 个值（看邻居）
- **通道混合**：所有 C_in 个通道的结果求和，合并成 1 个输出通道（看所有输入通道）
- 两件事**在同一次乘加里同时发生**，无法拆分

**DWConv3d（groups=C）：只做空间混合，禁止通道混合：**

```
DWConv3d(C=3, k=3, groups=C)，权重形状: (3, 1, 3,3,3)

ch=0: output[0, d,h,w] = input[0, d+-1,h+-1,w+-1] * weight[0]   <- 只看通道0的邻域
ch=1: output[1, d,h,w] = input[1, d+-1,h+-1,w+-1] * weight[1]   <- 只看通道1的邻域
ch=2: output[2, d,h,w] = input[2, d+-1,h+-1,w+-1] * weight[2]   <- 只看通道2的邻域

输入通道 i -> 输出通道 i（一一对应，跨通道零交互）
参数量 = C x 27（比普通 Conv 少 C_in 倍）
```

**groups 参数的含义：**

```
Conv3d(C_in=6, C_out=6, k=3, groups=G)

groups=1（默认，普通 Conv）：
    全部 6 个输入通道 → 全部 6 个输出通道，全连接

groups=2：
    输入 [0,1,2] → 输出 [0,1,2]   第1组
    输入 [3,4,5] → 输出 [3,4,5]   第2组，组间零交互

groups=6（DWConv，groups=C）：
    输入 [0] → 输出 [0]
    输入 [1] → 输出 [1]
    ...每组只有 1 个通道，极端隔离

权重形状：(C_out, C_in/groups, k,k,k)
groups 越大 → 每组权重越小 → 总参数越少
```

**"可分离"的含义——注意有两种完全不同的东西同名：**

```
【1】空间可分离卷积（Spatially Separable Conv，信号处理经典概念）：
    把高维卷积核分解成多个低维核的乘积，类似矩阵低秩分解 A = BC

    3×3 核如果是秩1矩阵：K = u · vᵀ（列向量 × 行向量）
        原来：一次 3×3，9 次乘法
        分解：先 3×1（竖方向）再 1×3（横方向），3+3=6 次乘法

    3×3×3 的 3D 核类似：K ≈ u · vᵀ · wᵀ
        变成三步串联：(3×1×1) → (1×3×1) → (1×1×3)
        27 次乘法 → 9 次，省 3×

    前提：卷积核近似低秩（rank-1）才能完全分解，否则有精度损失
    深度学习里较少用（医学图像 3D 卷积核不一定低秩）

【2】深度可分离卷积（Depthwise Separable Conv，MobileNet，DL 里说"可分离"默认指这个）：
    普通 Conv3d 同时做空间混合 + 通道混合，把这两件事拆开：
        第1步 DWConv(groups=C)：只做空间混合，通道间完全隔离
        第2步 1×1 Conv：        只做通道混合，k=1 不看邻域

    "分离" = 空间维度 和 通道维度 的混合解耦，不再耦合在同一个权重里
    无需卷积核低秩，任何情况都适用

参数量对比（C_in=C_out=C，k=3）：
    普通 Conv3d:          C x C x 27
    深度可分离卷积:        C x 27  +  C x C  （C=256 时省约 90%）
    空间可分离 3D:         C x C x 9  （三步各 3×1×1 等）
```

MedNeXt 在两步之间加了 Norm + 激活，比 MobileNet 的原版可分离多了非线性，更接近 ConvNeXt：

```
MobileNet（原版可分离，两步打包无间隔）:
    DWConv(k=3) -> 1x1 Conv

MedNeXt（拆开并插入 Norm + 激活）:
    conv1: DWConv3d(k=3, groups=C)     空间混合
    norm:  GroupNorm(num_groups=C)     归一化
    conv2: 1x1 Conv(C -> C*exp_r)      通道展宽
    GELU
    conv3: 1x1 Conv(C*exp_r -> C)      通道压缩
```

| | 普通 Conv3d | DWConv3d |
|---|---|---|
| 权重形状 | (C_out, C_in, 3,3,3) | (C, 1, 3,3,3) |
| 参数量 | C_out x C_in x 27 | C x 27 |
| 空间混合 | ✓ | ✓ |
| 通道混合 | ✓（所有 C_in -> 所有 C_out） | ✗（ch_i -> ch_i，一一对应） |

### 8.4 MedNeXt Block 完整结构（实际代码顺序）

```
输入 x（C 通道）
    │
    ▼  conv1: DWConv3d(k=3, groups=C)       空间混合，参数量 = C×27
    ▼  norm:  GroupNorm(num_groups=C)        ≡ InstanceNorm，每通道独立归一化
    ▼  conv2: Conv3d(C → C×exp_r, k=1)      通道展宽，expand
    ▼  GELU
    ▼  conv3: Conv3d(C×exp_r → C, k=1)      通道压缩，compress
    │
    └─ + x（残差，do_res=True）
    ▼
输出（C 通道）
```

**exp_r（膨胀比）的作用：**
- `exp_r=3`：中间维度 3×C，浅层用（特征图大，省显存）
- `exp_r=8`：中间维度 8×C，深层用（语义特征，需更强非线性）

与 Baseline Conv Block 的区别：

| | Baseline Conv Block | MedNeXt Block |
|---|---|---|
| 空间混合 | Conv3d(k=3, 所有通道耦合) | DWConv3d(k=3, 每通道独立) |
| 通道混合 | 和空间混合同一步完成 | 分离为 conv2 + conv3（1×1） |
| 非线性 | LeakyReLU | GELU |
| 归一化 | InstanceNorm3d | GroupNorm(num_groups=C) ≡ InstanceNorm |
| 残差 | ✗ | ✓ |
| 残差 | ✗ | ✓ |

### 8.3 整体架构（MedNeXt-L）

固定配置：`block_counts=[3,4,8,8,8,8,8,4,3]`，`exp_r=[3,4,8,8,8,8,8,4,3]`

```
Input (1, 128³)
    │
    ▼
┌──────────────────────────────────────────────────────────────────┐
│  Encoder（4 stage）                                              │
│                                                                  │
│  Stage 0: 32ch,  128³，3 个 MedNeXt blocks（exp_r=3）  → skip-0 │
│      ↓  strided MedNeXt block（下采样，do_res_up_down=True）      │
│  Stage 1: 64ch,   64³，4 个 MedNeXt blocks（exp_r=4）  → skip-1 │
│      ↓  strided block                                            │
│  Stage 2: 128ch,  32³，8 个 MedNeXt blocks（exp_r=8）  → skip-2 │
│      ↓  strided block                                            │
│  Stage 3: 256ch,  16³，8 个 MedNeXt blocks（exp_r=8）  → skip-3 │
│      ↓  strided block                                            │
└──────────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────────┐
│  Bottleneck: 512ch, 8³，8 个 MedNeXt blocks（exp_r=8）           │
└──────────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────────┐
│  Decoder（4 stage，do_res_up_down=True 上采样也有残差）           │
│                                                                  │
│  Dec-3: 上采样 → cat(skip-3) → 8 个 blocks（exp_r=8）  → 256ch  │
│  Dec-2: 上采样 → cat(skip-2) → 8 个 blocks（exp_r=4）  → 128ch  │
│  Dec-1: 上采样 → cat(skip-1) → 4 个 blocks（exp_r=4）  → 64ch   │
│  Dec-0: 上采样 → cat(skip-0) → 3 个 blocks（exp_r=3）  → 32ch   │
└──────────────────────────────────────────────────────────────────┘
    │
    ▼  Deep Supervision 输出头（5 个尺度）
    [Output 128³, Output 64³, Output 32³, Output 16³, Output 8³]
```

**`do_res_up_down=True`** 的含义：下采样/上采样卷积本身也包一个残差连接（跳过 stride conv 用 1×1 conv 匹配尺寸后相加），让梯度更容易反传。

### 8.4 固定配置（MedNeXt-L）

| 参数 | 值 | 含义 |
|------|-----|------|
| `n_channels` | 32 | 第一层基础通道数 |
| `kernel_size` | 3 | DWConv 核大小（3×3×3） |
| `exp_r` | [3,4,8,8,8,8,8,4,3] | 各 stage 的通道膨胀比 |
| `block_counts` | [3,4,8,8,8,8,8,4,3] | 各 stage 的 block 数量 |
| `do_res` | True | block 内部残差连接 |
| `do_res_up_down` | True | 下/上采样层也带残差 |
| `norm_type` | 'group' | GroupNorm |
| `checkpoint_style` | 'outside_block' | block 级别梯度检查点 |
| `deep_supervision` | True | 5 尺度 DS 输出 |

### 8.5 Deep Supervision

MedNeXt 原生支持 DS，输出 5 个尺度（从高分辨率到低分辨率）：
`[full_res(128³), 1/2(64³), 1/4(32³), 1/8(16³), 1/16(8³)]`

与 nnUNetv2 期望格式一致，`set_deep_supervision_enabled` 通过 `mod.do_ds = enabled` 开关。

### 8.6 参数量

| 组件 | 参数量估算 |
|------|-----------|
| Encoder（3+4+8+8 blocks） | ~25M |
| Bottleneck（8 blocks） | ~20M |
| Decoder（8+8+4+3 blocks） | ~17M |
| **总计** | **~62M** |

> 大参数量主要来自深层的 exp_r=8（中间维度 = 512×8 = 4096），FFN 参数量远超 Baseline 的 3×3×3 Conv。

### 8.7 与 Baseline 的对比

| 方面 | Baseline | MedNeXt-L |
|------|----------|-----------|
| Block 类型 | Conv3d 3×3×3 | **DWConv + 膨胀 MLP（ConvNeXt 风格）** |
| Block 内残差 | ✗ | **✓（do_res=True）** |
| 上/下采样残差 | ✗ | **✓（do_res_up_down=True）** |
| 归一化 | InstanceNorm3d | **GroupNorm** |
| 激活函数 | LeakyReLU | **GELU** |
| Deep Supervision | ✓（5 尺度） | ✓（5 尺度，原生） |
| block 数（总） | 2×(2+2+2+2+2) = 20 个 conv | **3+4+8+8+8+8+8+4+3 = 54 个 block** |
| 参数量 | ~30M | **~62M（+107%）** |
| Trainer | `nnUNetTrainer_Baseline` | `nnUNetTrainer_MedNeXt` |

---

## 9. 五模型横向对比

| 方面 | Baseline | MLA-UNet | MLA-MoE | MedNeXt-L | SwinUNETR-B |
|------|----------|----------|---------|-----------|-------------|
| Encoder | Conv | Conv | Conv | **ConvNeXt** | **Swin Transformer** |
| 瓶颈 | Conv | **MLA** | **MLA+MoE** | ConvNeXt×8 | Swin+下采样 |
| 感受野 | 局部 3×3×3 | 局部+**瓶颈全局** | 局部+**瓶颈全局** | 局部 3×3×3 | 局部 7³窗口 |
| Deep Supervision | ✓ | ✓ | ✓ | ✓ | **✗** |
| 参数量 | ~30M | ~31M | ~33M | **~62M** | **~63M** |
| 梯度检查点 | ✗ | ✗ | ✗ | ✓（block级） | ✓（use_checkpoint） |
| 代码来源 | nnUNetv2 内置 | 自研 | 自研 | nnunet_mednext | MONAI |

---

## 10. 代码对应关系（完整）

| 组件 | 文件 | 类/函数 |
|------|------|---------|
| Baseline 整体 | `dynamic_network_architectures/architectures/unet.py` | `PlainConvUNet` |
| Encoder | `.../building_blocks/plain_conv_encoder.py` | `PlainConvEncoder` |
| Decoder | `.../building_blocks/unet_decoder.py` | `UNetDecoder` |
| Conv Block | `.../building_blocks/simple_conv_blocks.py` | `StackedConvBlocks` |
| **MLA Attention** | `pumengyu/architectures/mla_unetr.py` | `MultiHeadLatentAttention` |
| **MoE-FFN** | `pumengyu/architectures/mla_unetr.py` | `MoEFFN` |
| **MLA Block** | `pumengyu/architectures/mla_unetr.py` | `MLATransformerBlock` |
| **MLA 瓶颈封装** | `pumengyu/architectures/mla_unetr.py` | `MLABottleneck3D` |
| **MLAUNet 整体** | `pumengyu/architectures/mla_unetr.py` | `MLAUNetBot3D` |
| **MedNeXt-L** | `pumengyu/architectures/mednext.py` | `build_mednext_large` → `MedNeXt` |
| **SwinUNETR-B** | `pumengyu/architectures/swinunetr.py` | `build_swinunetr` → MONAI `SwinUNETR` |
| Baseline Trainer | `pumengyu/trainers/trainer.py` | `nnUNetTrainer_Baseline` |
| MLA Trainer | `pumengyu/trainers/trainer.py` | `nnUNetTrainer_MLAUNet` |
| MLA-MoE Trainer | `pumengyu/trainers/trainer.py` | `nnUNetTrainer_MLAUNet_MoE` |
| MedNeXt Trainer | `pumengyu/trainers/trainer.py` | `nnUNetTrainer_MedNeXt` |
| SwinUNETR Trainer | `pumengyu/trainers/trainer.py` | `nnUNetTrainer_SwinUNETR` |

---

## 11. 实验验证：MLA 和 MoE 的效果分析

> 内部测试集：LiTS 内部 26 cases（有肿瘤 23，无肿瘤 3）  
> 外部验证集：IRCADb 20 cases（有肿瘤 15，无肿瘤 5）  
> 结果路径：`/home/PuMengYu/nnUNet_workspace/results_v2/ExternalVal_IRCADb/`  
> 详细分析：`results_v2/Dataset003_Liver/实验结果对比分析.md` / `notes/md/02_实验结果/外部验证分析_IRCADb.md`

### 11.1 MLA vs Baseline：效果明确

| 指标 | Baseline | MLAUNet | Δ |
|------|----------|---------|---|
| Overall（内部） | 0.7941 | 0.8148 | **+0.021** |
| Tumor Dice（内部） | 0.6542 | 0.6793 | **+0.025** |
| Precision（内部） | 0.6451 | 0.7427 | **+0.098** |
| FDR（内部） | 0.3549 | 0.2573 | **-0.098** |
| 无肿瘤误报率（内部） | 100% | 66.67% | 改善 |

Overall +0.021、Precision +0.098 在 n=26 的测试集上不是噪声，是结构性改变。MLA 引入全局 attention 后，Precision 大幅提升——模型不再盲目预测肿瘤，假阳性率显著下降。

**外部验证（IRCADb）**：MLAUNet 的 checkpoint 与当前代码存在 weight key 不兼容问题（原始训练时 FFN 键名为 `mlp.*`，当前代码升级为 MoE 后键名为 `experts.*`），外部预测目录为空，**无法直接对比**。重训的 MLAUNet_1500（1500 epoch）内部 Overall = 0.8028，低于原始 MLAUNet（0.8148），原因待查（可能过拟合或超参需调整）。

### 11.2 MoE vs MLA：内部收益边际，但小肿瘤有实质提升

| 指标 | MLAUNet | MLAUNet_MoE | Δ |
|------|---------|-------------|---|
| Overall（内部） | 0.8148 | 0.8166 | +0.0018 |
| Tumor Dice（内部） | 0.6793 | 0.6826 | +0.0033 |
| 极小肿瘤 Dice <5k（内部） | 0.5256 | **0.5637** | **+0.038** |
| 小肿瘤 Dice 5k-50k（内部） | 0.8020 | **0.8031** | +0.0011 |
| FPV 误报体积（内部） | 270,603 mm³ | **250,703 mm³** | -7.3% |
| Overall（外部） | — | **0.7922** | vs Baseline +0.020 |

Overall +0.0018 和 Tumor Dice +0.0033 在 n=26 下处于噪声边缘，**单独作为 MoE 有效的证据不充分**。但极小肿瘤（<5k 体素，n=6）Dice +0.038 比较突出，结合 FPV 下降 7%，说明 MoE 的专家分工对小目标有一定帮助。

外部验证上 `MoE`（即 MLAUNet_MoE）Overall = 0.7922，对比 Baseline 0.7727（+0.020）——但这个 +0.020 包含了 MLA 和 MoE 的综合贡献，**无法拆分**。

### 11.3 关键对照：纯数据策略 vs 架构改进

| 方法 | 内部 Overall | 特点 |
|------|------------|------|
| SizeOversampleV2 | **0.8187** | 纯 nnUNet 架构 + 肿瘤过采样，无 MLA/MoE |
| MLAUNet_MoE | 0.8166 | MLA + MoE，无过采样 |
| MLAUNet | 0.8148 | 仅 MLA，无 MoE 无过采样 |

纯数据策略（SizeOversampleV2）内部 Overall 比 MLAUNet_MoE 还高 0.002。这说明在当前测试集规模下，**数据策略和架构改进的效果量级相当**，难以区分谁更本质。

### 11.4 结论

| 问题 | 结论 |
|------|------|
| MLA 比 Baseline 有效吗？ | **是，效果明确**：Overall +0.021，Precision +0.098，误报率下降 |
| MoE 在 MLA 基础上有效吗？ | **不确定**：Overall Δ 在噪声范围，极小肿瘤 +0.038 有实质意义但 n=6 |
| 外部验证能证明 MLA 有效吗？ | **暂时不能**：MLAUNet 外部预测为空（checkpoint 不兼容），无法单独验证 |
| 要确认 MoE 独立贡献需要什么？ | 修复 MLAUNet checkpoint 兼容性后补跑外部验证，或扩大内部测试集 |

---

## 12. nnFormer

### 12.1 一句话总结

**nnFormer = 全 Transformer UNet（专为医学图像设计），patch_size=4×4×4，不同 stage 使用不同窗口大小**

与 SwinUNETR 同为全 Transformer Encoder+Decoder，但专门针对医学图像做了调整：更大的 patch（4³ vs 2³），瓶颈处使用更大的窗口（8³）来增大全局感受野。

### 12.2 整体结构

```
Input (1, 128³)
    │
    ▼  patch embedding（stride=4，4×4×4 体素 → 1 个 token）
    (B, 32³, 96)  ← 32³ = 32,768 个 token（比 SwinUNETR 的 64³ 少 8×）
    │
    ▼
┌──────────────────────────────────────────────────────────┐
│  Transformer Encoder（4 stage）                          │
│                                                          │
│  Stage 0: (B, 32³, 96),  heads=3,  win=4³,  depth=2    │
│      ↓ PatchMerging（2× 下采样）                         │
│  Stage 1: (B, 16³, 192), heads=6,  win=4³,  depth=2    │
│      ↓ PatchMerging                                      │
│  Stage 2: (B,  8³, 384), heads=12, win=8³,  depth=2    │
│      ↓ PatchMerging                                      │
│  Stage 3: (B,  4³, 768), heads=24, win=4³,  depth=2    │
└──────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────┐
│  Transformer Decoder（3 stage，逆序配置）                 │
│  window_size 逆序，num_heads 逆序                         │
│                                                          │
│  Dec-2: 上采样 → cat(skip) → Transformer blocks         │
│  Dec-1: 上采样 → cat(skip) → Transformer blocks         │
│  Dec-0: 上采样 → cat(skip) → Transformer blocks         │
└──────────────────────────────────────────────────────────┘
    │
    ▼  final_patch_expanding（把 token 映射回原始分辨率）
Output (num_classes, 128³)
```

### 12.3 nnFormer 针对医学图像的三个专项调整

**调整1：3D 体积相对位置偏置（核心创新）**

标准 Swin（2D）的相对位置偏置只编码 (Δrow, Δcol) 两个方向。nnFormer 把它扩展到三维：

```
偏置表大小：(2*ws_D-1) × (2*ws_H-1) × (2*ws_W-1) × num_heads

window=4 时：7 × 7 × 7 × num_heads = 343 × num_heads 个可学习参数
覆盖：D方向相对距离 [-3,+3]，H方向 [-3,+3]，W方向 [-3,+3]
```

意义：CT 体积是各向异性的（slice 方向分辨率通常低于 H/W 方向），3D 位置偏置让模型独立学习三个轴的空间关系，而不是把体积错误地当成各向同性的 2D 图像处理。

**调整2：Decoder 用 Cross-Attention 融合 Skip（而非拼接）**

SwinUNETR 的 Decoder 是普通 CNN，Skip 连接靠 cat() 拼接特征后卷积。nnFormer 的 Decoder 是 Transformer，Skip 连接通过**交叉注意力**完成：

```
Decoder Block（SwinTransformerBlock_kv）：

    Q = x_up（上采样的 decoder 特征）   ← 问题："我在找什么"
    K = skip（encoder 同分辨率特征）     ← 索引："哪里有相关信息"
    V = skip（encoder 同分辨率特征）     ← 内容："相关信息的具体值"

    attention = softmax(Q @ Kᵀ / √d) @ V
```

意义：普通 cat+Conv 是无差别地把所有 skip 特征加权求和，Cross-Attention 允许 Decoder 根据当前解码位置动态决定"去 Encoder 的哪些位置取信息"，是更精准的特征融合。

**调整3：各 stage 使用不同窗口大小 [4,4,8,4]**

```
Stage 0（32³ tokens，语义浅）: window=4³=64 tokens/window
Stage 1（16³ tokens）:         window=4³=64 tokens/window
Stage 2（8³ tokens，语义深）:  window=8³=512 tokens/window  ← 最大窗口在最深层
Decoder:                        window=[4,4,4]（逆序）
```

Stage 2（瓶颈）的 window=8 实际上覆盖了 8³=512 个 token，而整个特征图只有 8³=512 个 token，这意味着**瓶颈层做的是全局 attention**（窗口等于整个特征图）。浅层 window=4 保持局部效率，深层 window=8 获得全局感受野。

---

### 12.4 与 SwinUNETR 的关键区别

> **最重要的差异就一个：patch_size**
>
> ```
> SwinUNETR  patch_size = 2  →  2×2×2 = 8  体素/token  →  第一层 64³ = 262,144 个 token  →  OOM
> nnFormer   patch_size = 4  →  4×4×4 = 64 体素/token  →  第一层 32³ =  32,768 个 token  →  1.45 GB
> ```
>
> token 数少 8 倍，attention 激活显存少约 64 倍。这一个数字决定了 SwinUNETR 必须开梯度检查点才能跑、而 nnFormer 轻松跑。

| | SwinUNETR | nnFormer |
|---|---|---|
| **patch_size** | **2×2×2（8 体素/token）** | **4×4×4（64 体素/token）← 关键差异** |
| **第一层 token 数** | **64³ = 262,144** | **32³ = 32,768（少 8×）** |
| **训练峰值 VRAM (bs=2)** | **OOM** | **1.45 GB（开梯度检查点）** |
| window_size | 全程固定 7³ | 各 stage 不同：4³/4³/8³/4³ |
| 瓶颈感受野 | 局部窗口（7³） | 全局（window=8³ = 整张特征图） |
| Decoder 类型 | 普通 CNN + cat | Transformer Cross-Attention |
| 参数量 | 62.2M | 37.6M |
| Deep Supervision | ✗ | ✗（实现里注释掉了） |
| 设计目标 | 通用视觉 Transformer 迁移 | 专为 3D 医学体积设计 |

### 12.4 固定配置

| 参数 | 值 |
|------|-----|
| `embedding_dim` | 96 |
| `patch_size` | [4, 4, 4] |
| `depths` | [2, 2, 2, 2] |
| `num_heads` | [3, 6, 12, 24] |
| `window_size` | [4, 4, 8, 4]（Stage 2 最大） |
| Deep Supervision | ✗ |
| `torch.compile` | ✗ |

### 12.5 实验状态

> 截至 2026-06-14，nnFormer 正在训练中，尚无内部/外部验证结果。

### 12.6 内存分析：nnFormer 比 SwinUNETR 更轻

实测（RTX 4090 D，PyTorch 2.3，128³ patch，bf16 autocast）：

| 模型 | 参数量 | 训练峰值 VRAM (bs=2) | 备注 |
|------|--------|---------------------|------|
| SwinUNETR-B | 62.2M | **OOM**（bs=2/bs=1 均 OOM） | 必须开 use_checkpoint |
| nnFormer（无检查点） | 37.6M | 2.97 GB | 无显存压力 |
| nnFormer（梯度检查点） | 37.6M | **1.45 GB** | 省 51%，安全边际充足 |

**为什么 nnFormer 比 SwinUNETR 省显存？答案只有一个：patch_size**

```
                    patch_size      体素/token    第一层 token 数     激活显存
                  ─────────────   ────────────   ───────────────   ──────────
SwinUNETR             2×2×2            8           64³ = 262,144      OOM
nnFormer              4×4×4           64           32³ =  32,768     1.45 GB
                                                        ↑
                                                      少 8 倍
                                      attention 矩阵 ∝ N²，激活约少 64 倍
```

nnFormer 的 Decoder 是 Transformer（比 SwinUNETR 的 CNN Decoder 多了 Cross-Attention），但因为 token 数少，实际显存反而更小（37.6M 参数 vs 62.2M）。

**结论：nnFormer 内存不是瓶颈，无需替换为 MLA。**

### 12.7 显存优化实现

已在 `pumengyu/architectures/nnformer.py` 和 `nnUNetTrainer_nnFormer` 中实现两项优化：

**1. 梯度检查点（Gradient Checkpointing）**

`build_nnformer(use_gradient_checkpointing=True)`（trainer 默认开启）通过 monkey-patching 对所有 `SwinTransformerBlock`（Encoder）和 `SwinTransformerBlock_kv`（Decoder Cross-Attention）的 `forward` 方法套上 `torch.utils.checkpoint.checkpoint(use_reentrant=False)`：

```python
# Encoder block
def fn(x):
    return orig(x, mask_matrix)   # mask_matrix 捕获在闭包里
return ck.checkpoint(fn, x, use_reentrant=False)

# Decoder block（Cross-Attention）
def fn(x, skip, x_up):
    return orig(x, mask_matrix, skip=skip, x_up=x_up)
return ck.checkpoint(fn, x, skip, x_up, use_reentrant=False)
```

- 节省 ~51% 激活显存（2.97 GB → 1.45 GB）
- 代价：反向传播重跑一次前向，训练时间约 +33%
- 当前 bs=2 下 1.45 GB 远低于 24 GB 上限，此优化为纯安全边际

**2. bf16 autocast（覆盖基类 fp16 默认）**

nnUNet 基类 `train_step` 使用 `torch.autocast('cuda', enabled=True)` → 默认 **fp16**。`nnUNetTrainer_nnFormer` 覆盖 `train_step`，显式指定 `dtype=torch.bfloat16`：

| | fp16 | bf16 |
|---|---|---|
| 指数位 | 5 位 | 8 位（与 fp32 相同） |
| 尾数位 | 10 位 | 7 位 |
| 溢出风险 | 有（需 GradScaler） | 极低（无需 GradScaler） |
| 矩阵乘法精度 | 略高 | 略低但通常无影响 |

bf16 不需要 GradScaler，`initialize()` 中设 `self.grad_scaler = None`，训练代码更简洁。

**3. FP8 矩阵乘法（待实现，需要 torchao）**

真正的 FP8 矩阵乘法（把 Linear/Attention 权重量化到 `float8_e4m3fn`）需要：

```bash
pip install torchao
```
```python
from torchao.float8 import convert_to_float8_training
model = convert_to_float8_training(model)   # 把 Linear 层换成 Float8Linear
```

RTX 4090 D（Ada Lovelace）原生支持 FP8，理论上再省 50% 显存 + 提速。当前环境未安装 `torchao`，暂不可用。bf16 autocast 已覆盖最主要的显存/稳定性权衡，**FP8 是未来进一步压缩的备选**。

---

## 13. Conv 和 Transformer 的真实局限，以及架构改动到底有没有用

### 13.1 Conv 的局限

**局限1：感受野是局部的**

3×3×3 卷积每层只看 27 个体素邻域。堆叠 6 层后理论感受野是 13×13×13，但有效感受野（对输出影响显著的区域）远小于理论值。对于肝脏肿瘤这类任务：
- 肿瘤在肝内的位置（边缘 vs 中央）需要全局上下文
- 无肿瘤 case 的误报（把血管/囊肿误判为肿瘤）很大程度上是因为模型缺乏"这个暗区周围的大范围上下文是什么"的信息

**局限2：空间混合和通道混合耦合，参数效率低**

标准 Conv3d 的权重把"看哪些邻居"和"怎么组合通道"混在一起，C_out × C_in × 27 的参数里大量花在冗余的跨通道空间关系上。

**局限3：平移等变性是归纳偏置也是限制**

Conv 天然对平移等变，在数据量少时是优势（不需要从数据中学）；但对于需要关注全局位置关系的任务（如"肿瘤在肝脏的哪个区域"），这个偏置反而是约束。

### 13.2 Transformer 的局限

**局限1：缺少空间归纳偏置，需要更多数据**

Conv 天然知道"近邻更重要"，Transformer 需要从数据中学到这个关系。LiTS 只有 105 个训练 case，数据量对纯 Transformer 来说偏少。

**局限2：计算复杂度随 token 数二次增长**

Full attention 的复杂度 O(N²)，N=token 数。128³ 体素直接做 attention 完全不可行（N≈200 万）。Swin 的窗口 attention 和 nnFormer 的大 patch 都是为了绕过这个问题，但代价是牺牲部分全局性。

**局限3：参数量大，过拟合风险高**

SwinUNETR 和 MedNeXt 参数量约 63M，是 Baseline（30M）的 2×。小数据集上更容易过拟合，需要更强的正则化。

### 13.3 这些架构改动在我们的任务上到底有没有用？

**坦率的评估：**

| 改动 | 理论动机 | 实测效果 | 评价 |
|------|---------|---------|------|
| MLA（瓶颈全局 attention） | 解决局部感受野 | +0.021 Overall，+0.098 Precision | **有效，且代价小** |
| MoE FFN | 专家分工处理不同大小肿瘤 | +0.0018 Overall（噪声级） | **效果不明确** |
| SizeOversample（无架构改动） | 均衡训练频率 | +0.024 Overall | **和 MLA 效果量级相当** |
| MedNeXt（全部替换为 ConvNeXt） | 更好的 Conv block | 尚无结果 | 待验证 |
| SwinUNETR（全 Transformer） | 全局感受野 | 尚无结果 | 待验证 |
| nnFormer（医学专用 Transformer） | 全局感受野+医学设计 | 尚无结果 | 待验证 |

**最不舒服的事实**：
- 纯数据策略（SizeOversample，不改任何架构）的提升和 MLA（改了瓶颈架构）几乎一样大
- 说明在 LiTS 这个数据集上，**数据层面的问题（极小/极大肿瘤频率不均衡）比感受野问题更突出**
- 全局 attention 有帮助（+0.021），但不是瓶颈

**Conv vs Transformer 在医学分割的实证结论（文献 + 我们的数据）：**
- 小数据集（<200 cases）：Conv（nnUNet）通常不输给 Transformer，有时更好
- 大数据集（>1000 cases）：Transformer 优势开始显现
- 混合方案（瓶颈/局部加 attention）：通常是最好的 trade-off，代价小、收益稳定

### 13.4 如何定位自己的贡献

**现有贡献的真实强度：**
- MLA 瓶颈：改动最小（+3% 参数），提升明确（+0.021），**这是最强的贡献点**
- MoE：理论合理但实测边际，**需要更多证据才能作为独立贡献**
- 组合策略（MLA + SizeOversample）：有意义但结果低于预期（V2 外部误报 80%）

**如果要调整方向，可能的路径：**

1. **强化 MLA 的故事**：补全 MLAUNet 外部验证（修复 checkpoint），形成完整的内部+外部对照，MLA vs Baseline 的 +0.021 是站得住的
2. **MoE 换思路**：当前 MoE 在 4³=64 token 的瓶颈上意义有限（专家分工在 token 极少时优势不明显），考虑把 MoE 放在 Decoder 的某一层（更多 token，更有意义的路由）
3. **对标大模型而不是自己卷**：SwinUNETR/MedNeXt/nnFormer 跑完后，如果 MLA-MoE（33M 参数）能接近甚至超过它们（~63M 参数），"以最小代价接近最强基线"本身就是贡献
4. **数据+架构组合的消融**：把 MLA + SizeOversampleV5（外部最强的 3× 过采样）系统地做完，4 格消融表（有/无 MLA × 有/无 过采样）
