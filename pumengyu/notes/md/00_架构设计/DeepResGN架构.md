# DeepResGN 架构

## DeepResGN 架构详解


#### DeepResGN 架构详解

##### 动机：为什么要做这个？

旧的 MLAUNet 输给 MedNeXt，原因不是 MLA 没用，而是**底座太弱**。

```
旧 MLAUNet 底座 = 默认 nnUNet PlainConvUNet
  n_stages = 6
  n_conv_per_stage = [2, 2, 2, 2, 2, 2]   ← 每个 stage 只有 2 个 conv
  norm = InstanceNorm3d
  残差连接 = 无

MedNeXt 底座
  block_counts = [3, 4, 8, 8, 8, 8, 8, 4, 3]  ← 每个 stage 多得多
  norm = GroupNorm
  残差连接 = 全程有
  卷积类型 = depthwise separable（倒置瓶颈）
```

**DeepResGN 的目标**：把 MedNeXt 的结构优势（深度 + 残差 + GroupNorm）迁移到 plain conv 上，证明"DW sep conv 不是关键"，并为 MLA 提供一个匹配强度的底座。

---

##### DeepResGN 结构

###### 总体形状（9 个处理位置）

```
输入 (B, 1, 128, 128, 128)
  │
  ▼ Encoder Stage 0  [3 blocks, 32ch,  128³, stride=1]
  ▼ Encoder Stage 1  [4 blocks, 64ch,   64³, stride=2]
  ▼ Encoder Stage 2  [4 blocks, 128ch,  32³, stride=2]
  ▼ Encoder Stage 3  [4 blocks, 256ch,  16³, stride=2]
  ▼ Encoder Stage 4  [4 blocks, 384ch,   8³, stride=2]  ← bottleneck
  ▲ Decoder Stage 3  [4 blocks, 256ch,  16³]
  ▲ Decoder Stage 2  [4 blocks, 128ch,  32³]
  ▲ Decoder Stage 1  [4 blocks,  64ch,  64³]
  ▲ Decoder Stage 0  [3 blocks,  32ch, 128³]
  │
  ▼ 输出 (B, 3, 128, 128, 128)   ← 3类: 背景/肝脏/肿瘤
```

9 个位置（5 enc + 4 dec）与 MedNeXt 的 `block_counts=[3,4,8,8,8,8,8,4,3]` 拓扑一致。

---

###### 每个 Block 的内部结构（BasicBlockD）

```
输入 x (B, C, D, H, W)
  │
  ├──────────────────────────────┐  residual skip
  │                              │
  ▼                              │
Conv3d(C→C, 3×3×3, padding=1)   │
GroupNorm(8, C)                  │
LeakyReLU(inplace)               │
  │                              │
Conv3d(C→C, 3×3×3, padding=1)   │
GroupNorm(8, C)                  │
  │                              │
  └──────────── + ───────────────┘
                │
           LeakyReLU
                │
              输出
```

- **plain 3×3 conv**（不是 DW sep，不是倒置瓶颈）
- **GroupNorm(8)**（替换 nnUNet 默认的 InstanceNorm）
- **残差跳连**（每个 block 内部都有）
- 当 in_ch ≠ out_ch 时（stage 间降维），skip 用 1×1 conv 对齐通道

---

###### 结构层级

```
Network
 └── Stage（分辨率级别，如 128³ / 64³ / 32³ ...）
      └── Block × N（每个 stage 有 N 个 block，堆叠）
           └── Conv layer × 2（每个 block 内部 2 个卷积 + 残差）
```

层级是 **Stage → Block → Conv**，block 是最小的带残差的单元，conv 是最小的计算单元。

---

###### 为什么参数量和 MedNeXt 一样——关键计算

MedNeXt 每个 block 用**倒置瓶颈（Inverted Bottleneck）**，参数极少：

```
MedNeXt 一个 block（exp_r=8, C=256ch）
  DW Conv3d(256ch, groups=256, 3×3×3)  →  256 × 27        ≈    7K  ← 极少
  PW Conv1×1(256 → 2048)               →  256 × 2048      ≈  524K
  PW Conv1×1(2048 → 256)               →  2048 × 256      ≈  524K
  合计：≈ 1.05M / block
```

```
DeepResGN 一个 block（plain conv, C=256ch）
  Conv3d(256→256, 3×3×3)               →  256 × 256 × 27  ≈ 1.77M
  Conv3d(256→256, 3×3×3)               →  256 × 256 × 27  ≈ 1.77M
  合计：≈ 3.54M / block
```

**MedNeXt 一个 block 参数量约是 DeepResGN 的 1/3**，所以 MedNeXt 在深层 stage 堆了 8 个 block，DeepResGN 只堆 4 个，总参数量才大致对齐：

```
MedNeXt  Stage 3 (bottleneck, 512ch)：8 blocks × ~3.5M  ≈ 28M
DeepResGN Stage 4 (bottleneck, 384ch)：4 blocks × ~3.5M  ≈ 14M
（通道上限不同：512 vs 384，综合后总参数 61.8M vs 61.1M）
```

---

###### DeepResGN vs MedNeXt：本质区别

| | DeepResGN | MedNeXt |
|---|---|---|
| 卷积类型 | **plain 3×3×3 conv** | DW sep conv（depthwise + pointwise） |
| Block 结构 | 两个 plain conv + 残差 | **倒置瓶颈**（PW扩通道 → DW卷积 → PW压缩）+ 残差 |
| 感受野 | 3×3×3（每层固定） | 3×3×3（但先扩展到 C×8 再压缩，隐式增大特征混合） |
| 每 block 参数 | 多（3.5M） | 少（1M） |
| 每 stage block 数 | 少（4个） | 多（8个） |
| 参数总量 | ~61.1M | ~61.8M（**故意对齐**） |
| GroupNorm | ✓ | ✓ |
| 残差连接 | ✓ | ✓ |

**消融意义**：两者参数量、深度、GN、残差全部对齐，**唯一变量 = 有没有 DW sep conv + 倒置瓶颈**。

- 若 DeepResGN ≈ MedNeXt → DW sep conv 不是关键，深度+残差+GN 才是
- 若 DeepResGN ＜ MedNeXt → DW sep conv / 倒置瓶颈确实有额外收益

---

###### 与对照架构的参数对比

| 架构 | 结构 | 残差 | Norm | 参数量 |
|---|---|---|---|---|
| nnUNet Baseline | 6 stages, [2,2,2,2,2,2] conv | ✗ | InstanceNorm | ~31M |
| MedNeXt-L | 9 位置, [3,4,8,8,8,...] IB块 | ✓ | GroupNorm | 61.8M |
| **DeepResGN** | 9 位置, [3,4,4,4,4] plain块 | ✓ | GroupNorm | **61.1M** |
| DeepResGN+MLA | DeepResGN + MLA bottleneck | ✓ | GroupNorm | **67.9M** |

参数量故意对齐 MedNeXt，排除"参数量更多所以更好"的混淆变量。

---

##### DeepResGN + MLA 结构

在 DeepResGN 的 bottleneck（Stage 4，384ch，8³分辨率）后插入 MLABottleneck3D：

```
... Encoder Stage 4 → bottleneck特征 (B, 384, 8, 8, 8)
                              │
                      MLABottleneck3D        ← 新增
                        2层 MLA Block
                        8头注意力
                        压缩比 4
                              │
                     增强后的bottleneck特征
                              │
                       Decoder Stage 3 ...
```

MLABottleneck3D 做的事：在 8×8×8=512 个位置上建模全局依赖，让 bottleneck 特征在解码前就具有全局上下文。

---

##### 消融实验链

```
Baseline（弱底座，无残差，InstanceNorm）         → 0.7941
    ↓ 换强底座（深层+残差+GroupNorm）
DeepPlainResGN                                  → ?
    ↓ 加 MLA global attention
DeepResGN + MLA                                 → ?（期望 > MedNeXt）
    ↓ 加 SizeOV4 过采样
DeepResGN + MLA + SizeOV4                       → ?（期望最高）

对照：
MedNeXt（DW sep conv，无MLA）                   → 0.8402
MedNeXt + SizeOV4                               → 0.8431
```

---

##### 核心 Claim

> 旧 MLAUNet 失败的原因是底座太弱，不是 MLA 没用。
> 在对齐参数量的强底座（DeepResGN）上，MLA 能有效提供 MedNeXt 所缺乏的全局依赖建模，从而超越 MedNeXt。

这个 claim 有完整的消融链支撑：每一步只改一个变量。


---
