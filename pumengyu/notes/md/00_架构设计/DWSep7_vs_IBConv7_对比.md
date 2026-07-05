# PlainConvUNet / DWSep7 / IBConv7 结构对比

---

## 一、原始 PlainConvUNet（Baseline）

每个 Stage 由若干 `ConvDropoutNormReLU` 块堆叠，编码器每级末尾用 stride=2 下采样。

```
输入特征图 (B, C_in, D, H, W)
│
▼  [ConvDropoutNormReLU × n_conv_per_stage]
│
│  单个块内部：
│  ┌─────────────────────────────────────────┐
│  │  Conv3d(k=3×3×3, stride=1, C_in→C_out) │  ← self.conv
│  │  InstanceNorm3d(C_out)                  │
│  │  LeakyReLU(inplace=True)                │
│  └─────────────────────────────────────────┘
│
▼  [Conv3d(k=3×3×3, stride=2)]  ← 下采样（ConvDropoutNormReLU，stride=2）
│
下一 Stage ...
```

**感受野**（6 层 stride=1 叠加，k=3）：单层 +2，6 层 ≈ **13 体素**

---

## 二、DWSep7（深度可分离，替换所有 Conv3d）

`_replace_encoder_conv3d_with_dw_sep` 递归找到所有 `nn.Conv3d`，原地替换为 `DWSepConv3d(k=7)`。  
**替换粒度是 `Conv3d` 本身**，外层 `ConvDropoutNormReLU` 的 Norm 和激活函数不变。

```
输入特征图 (B, C_in, D, H, W)
│
▼  [ConvDropoutNormReLU × n_conv_per_stage]
│
│  单个块内部（stride=1）：
│  ┌─────────────────────────────────────────────────────────┐
│  │  DW-Conv3d(k=7×7×7, groups=C_in, stride=1)  ← 空间混合 │
│  │  PW-Conv3d(k=1×1×1, C_in→C_out)             ← 通道混合 │
│  │  InstanceNorm3d(C_out)                                  │  ← 外层 Norm 不变
│  │  LeakyReLU(inplace=True)                                │  ← 外层激活不变
│  └─────────────────────────────────────────────────────────┘
│
▼  [ConvDropoutNormReLU，stride=2]（下采样也被替换）
│  ┌──────────────────────────────────────────────────────────┐
│  │  DW-Conv3d(k=7×7×7, groups=C_in, stride=2)             │
│  │  PW-Conv3d(k=1×1×1, C_in→C_out)                        │
│  │  InstanceNorm3d(C_out)                                  │
│  │  LeakyReLU                                              │
│  └──────────────────────────────────────────────────────────┘
│
下一 Stage ...
```

**关键点：**
- `conv.stride` 保持原值（stride=2 下采样语义不变）
- 无残差连接，无额外非线性
- 参数量：`k³·C + C·C_out`，远小于标准 `k³·C·C_out`

**感受野**（6 层 stride=1，k=7）：单层 +6，6 层 ≈ **37 体素**

---

## 三、IBConv7（倒置瓶颈大核，只替换 stride=1 块）

`_replace_cdnr_with_ib` 递归找到所有 `ConvDropoutNormReLU`，**仅当 stride=1 时**整块替换为 `IBConvBlock3D(k=7)`。  
stride=2 下采样块保持原始 `ConvDropoutNormReLU(k=3, stride=2)` 不变。

```
输入特征图 (B, C_in, D, H, W)
│
▼  [IBConvBlock3D × n_conv_per_stage]  ← 整块替换，非内部 conv 替换
│
│  单个块内部（stride=1）：
│  ┌────────────────────────────────────────────────────────────────────┐
│  │  residual = x（若 C_in ≠ C_out 则 1×1×1 投影）                    │
│  │                                                                    │
│  │  DW-Conv3d(k=7×7×7, groups=C_in, stride=1)  ← 大核空间混合        │
│  │  InstanceNorm3d(C_in)                        ← 自带 Norm           │
│  │  PW-Conv3d(1×1×1, C_in → 4·C_in)            ← 扩张（expand_ratio=4）│
│  │  GELU()                                      ← 自带激活            │
│  │  PW-Conv3d(1×1×1, 4·C_in → C_out)           ← 压缩               │
│  │  + residual                                  ← 残差连接            │
│  └────────────────────────────────────────────────────────────────────┘
│
▼  [ConvDropoutNormReLU(k=3×3×3, stride=2)]  ← 下采样块不变，保持标准 3 核
│
下一 Stage ...
```

**关键点：**
- stride=2 下采样不受影响（大核 stride=2 cuDNN 无优化）
- 每个块自带 Norm + GELU（取代外层 InstanceNorm + LeakyReLU）
- expand_ratio=4：中间通道 = 4×C，参数量比 DWSep 多但非线性更强

**感受野**（6 层 stride=1 IBConv，k=7）：≈ **37 体素**（与 DWSep7 相同）

---

## 四、三者差异汇总

| 维度 | Baseline | DWSep7 | IBConv7 |
|------|:--------:|:------:|:-------:|
| 卷积核（stride=1）| 3×3×3 标准 | 7×7×7 DW + 1×1×1 PW | 7×7×7 DW + PW↑ + GELU + PW↓ |
| 下采样卷积（stride=2）| 3×3×3 标准 | **7×7×7 DW + PW** | **保持 3×3×3 不变** |
| 残差连接 | ✗ | ✗ | ✓ |
| 块内激活 | LeakyReLU（外层）| LeakyReLU（外层）| **GELU（自带，外层被替换）** |
| 块内 Norm | InstanceNorm（外层）| InstanceNorm（外层）| **InstanceNorm（自带，外层被替换）** |
| 替换粒度 | — | `nn.Conv3d`（内部 conv）| `ConvDropoutNormReLU`（整块）|
| 感受野 | ~13 体素 | ~37 体素 | ~37 体素 |
| 参数量变化 | 基准 | 减少（DW 解耦）| 增加（expand×4）|

---

## 五、MedNeXt 与 IBConv7 的关系

MedNeXt 的 MedNeXt Block 结构与 IBConv7 的 `IBConvBlock3D` 几乎相同：

```
MedNeXt Block：
  DW-Conv3d(k×k×k) → Norm → PW(C→r·C) → GELU → PW(r·C→C) → +residual

IBConvBlock3D：
  DW-Conv3d(k×k×k) → InstanceNorm → PW(C→4C) → GELU → PW(4C→C) → +residual
```

本质一致，区别仅在：
- MedNeXt 使用 GroupNorm / LayerNorm，IBConv 用 InstanceNorm3d
- MedNeXt 整体是从头重新设计的完整网络；IBConv7 是对 PlainConvUNet 的 in-place 替换
- MedNeXt 有更多工程细节（encoder/decoder 级联的 expand_ratio 变化等）
