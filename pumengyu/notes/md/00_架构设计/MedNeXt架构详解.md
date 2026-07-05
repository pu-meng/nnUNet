# MedNeXt 架构详解与配置说明

> 代码入口：`pumengyu/architectures/mednext.py`
> 源码位置：`nnunet_mednext/network_architecture/mednextv1/`

---

## 一、我们使用的配置

```python
# pumengyu/architectures/mednext.py → build_mednext_large()
MedNeXt(
    in_channels      = num_input_channels,
    n_channels       = 32,                           # 初始特征通道数
    n_classes        = num_output_channels,
    exp_r            = [3, 4, 8, 8, 8, 8, 8, 4, 3], # 各阶段 expand ratio,倒置卷积的扩张比例
    kernel_size      = 3,                            # ← 大核卷积核大小，是 3 不是 7
    deep_supervision = enable_deep_supervision,
    do_res           = True,                         # 主块残差连接
    do_res_up_down   = True,                         # 下采样/上采样块也加残差
    block_counts     = [3, 4, 8, 8, 8, 8, 8, 4, 3], # 各阶段 block 数量
    checkpoint_style = 'outside_block',              # Gradient Checkpointing
    norm_type        = 'group',                      # GroupNorm
    dim              = '3d',
)
```

**对应官方 trainer**：`nnUNetTrainerV2_MedNeXt_L_kernel3`
位置：`nnunet_mednext/training/network_training/MedNeXt/nnUNetTrainerV2_MedNeXt.py` 第 102 行

---

## 二、为什么 kernel_size=3 而不是 7

### 官方 nnUNet 集成版只提供 k=3 和 k=5，没有 k=7

```
nnUNetTrainerV2_MedNeXt_S_kernel3   Small,  k=3
nnUNetTrainerV2_MedNeXt_B_kernel3   Base,   k=3
nnUNetTrainerV2_MedNeXt_M_kernel3   Medium, k=3
nnUNetTrainerV2_MedNeXt_L_kernel3   Large,  k=3   ← 我们使用的
nnUNetTrainerV2_MedNeXt_S_kernel5   Small,  k=5
nnUNetTrainerV2_MedNeXt_B_kernel5   Base,   k=5
nnUNetTrainerV2_MedNeXt_M_kernel5   Medium, k=5
nnUNetTrainerV2_MedNeXt_L_kernel5   Large,  k=5
（无 k=7 的 3D trainer）
```

`MedNeXt` 类构造函数的默认值是 `kernel_size=7`，但 3D 医学图像分割下 k=7
显存压力过大（DW-Conv3d k=7 stride=2 激活量翻倍），官方在 nnUNet 集成版本选择 k=3 作为标准配置。

### "Large" 指网络规模，不是卷积核大小

| 型号 | block_counts | exp_r | kernel | 参数量（估计）|
|------|-------------|-------|--------|------------|
| S (Small)  | [2,2,2,2,2,2,2,2,2] | 2 | 3 | ~5.6M |
| B (Base)   | [2,2,2,2,2,2,2,2,2] | 3 | 3 | ~9.6M |
| M (Medium) | [3,4,4,4,4,4,4,4,3] | [2,3,4,4,4,4,4,3,2] | 3 | ~18M |
| **L (Large)**  | **[3,4,8,8,8,8,8,4,3]** | **[3,4,8,8,8,8,8,4,3]** | **3** | **~62M** |

---

## 三、网络结构（9 阶段编解码器）

### block_counts 与 nnUNet 对比

`block_counts = [3,4,8,8,8,8,8,4,3]` 共 9 个数，每个数对应一个阶段内 MedNeXtBlock 的数量。
nnUNet 默认每级只有 **2 个** ConvDropoutNormReLU，MedNeXt-L 最多达 **8 个**。

| 阶段索引 | 位置 | block 数 | 通道数 C | exp_r | 中间通道 |
|:-------:|------|:-------:|:-------:|:-----:|:-------:|
| 0 | Encoder 入口 | 3 | 32 | 3 | 96 |
| 1 | Encoder（下采样1次后）| 4 | 64 | 4 | 256 |
| 2 | Encoder（下采样2次后）| 8 | 128 | 8 | 1024 |
| 3 | Encoder（下采样3次后）| 8 | 256 | 8 | 2048 |
| 4 | **Bottleneck**（下采样4次后）| 8 | 512 | 8 | **4096** |
| 5 | Decoder（上采样1次后）| 8 | 256 | 8 | 2048 |
| 6 | Decoder（上采样2次后）| 8 | 128 | 8 | 1024 |
| 7 | Decoder（上采样3次后）| 4 | 64 | 4 | 256 |
| 8 | Decoder 出口（上采样4次后）| 3 | 32 | 3 | 96 |

阶段 0~4 是编码器（4 次下采样），阶段 5~8 是解码器（4 次上采样），共 9 个阶段。nnUNet 是 6 级（5 次下采样）。注意结构是对称的：block 数 [3,4,8,8,**8**,8,8,4,3]，exp_r 一致，通道数呈沙漏形。

---

### 网络展开图（含 skip connection）

```
输入 CT (B, 1, D, H, W)
    │
    stem: Conv3d(1→32, k=1×1×1)      ← 投影到 32 通道，无空间混合
    │
    ├── Stage 0: MedNeXtBlock × 3  (C=32,  exp_r=3, 中间=96)   ──skip0──┐
    │   Down 0:  MedNeXtDownBlock  (32→64,  k=3, stride=2)              │
    │                                                                    │
    ├── Stage 1: MedNeXtBlock × 4  (C=64,  exp_r=4, 中间=256)  ──skip1──┤
    │   Down 1:  MedNeXtDownBlock  (64→128, k=3, stride=2)              │
    │                                                                    │
    ├── Stage 2: MedNeXtBlock × 8  (C=128, exp_r=8, 中间=1024) ──skip2──┤
    │   Down 2:  MedNeXtDownBlock  (128→256,k=3, stride=2)              │
    │                                                                    │
    ├── Stage 3: MedNeXtBlock × 8  (C=256, exp_r=8, 中间=2048) ──skip3──┤
    │   Down 3:  MedNeXtDownBlock  (256→512,k=3, stride=2)              │
    │                                                                    │
    └── Stage 4[Bottleneck]:                                            │
        MedNeXtBlock × 8  (C=512, exp_r=8, 中间=4096)                  │
        Up 3: MedNeXtUpBlock (512→256, k=3, stride=2) ← cat ───────────┘skip3
        Stage 5: MedNeXtBlock × 8  (C=256, exp_r=8)
        Up 2: MedNeXtUpBlock (256→128, k=3, stride=2) ← cat ───────────skip2
        Stage 6: MedNeXtBlock × 8  (C=128, exp_r=8)
        Up 1: MedNeXtUpBlock (128→64,  k=3, stride=2) ← cat ───────────skip1
        Stage 7: MedNeXtBlock × 4  (C=64,  exp_r=4)
        Up 0: MedNeXtUpBlock (64→32,   k=3, stride=2) ← cat ───────────skip0
        Stage 8: MedNeXtBlock × 3  (C=32,  exp_r=3)
    │
    out_conv: Conv3d(32→n_classes, k=1)
```

---

## 四、三种 Block 结构及关键配置说明

### MedNeXtBlock（stride=1，主体块）

源码：`blocks.py` 第 6 行

```
输入 x (B, C, D, H, W)
    │
    ├──────────────────────────────────────────── residual ──┐
    │                                                        │
    conv1: DW-Conv3d(k=3, stride=1, groups=C, C→C)          │  空间混合，通道不变
    norm:  GroupNorm(num_groups=C)                           │  ← 块内 Norm（不是外层）
    conv2: PW-Conv3d(1×1×1, C → exp_r·C)  + GELU            │  扩张（expand）
    conv3: PW-Conv3d(1×1×1, exp_r·C → C)                   │  压缩（compress）
    │                                                        │
    └──────────────────────────────────────── +──────────────┘  do_res=True 时才加
    输出 (B, C, D, H, W)
```

**GroupNorm** 说明：`GroupNorm(num_groups=C_in, num_channels=C_in)` 即每通道一组，等价于 InstanceNorm，但在 block **内部**（DW conv 后、expand PW 前），而非外层包装。作用是归一化 DW 输出的激活分布，防止训练不稳定。

```python
# forward（blocks.py 第 84 行）
x1 = self.conv1(x)                        # DW 空间混合
x1 = self.act(self.conv2(self.norm(x1)))  # GroupNorm → expand PW → GELU
x1 = self.conv3(x1)                       # compress PW
if self.do_res:
    x1 = x + x1                           # 残差（do_res=True）
```

---

### MedNeXtDownBlock（stride=2，下采样）

源码：`blocks.py` 第 104 行

**继承自 MedNeXtBlock**，只做一件事：在 `__init__` 里把父类的 `conv1`（stride=1 DW）**覆盖**为 stride=2：

```python
# __init__ 里覆盖 conv1
self.conv1 = Conv3d(C_in, C_in,
    kernel_size=k, stride=2, padding=k//2, groups=C_in)
#                  ↑ 下采样在 DW 大核里完成，不额外用 stride=2 标准卷积

# do_res_up_down=True 时额外加一条残差支路：
self.res_conv = Conv3d(C_in, C_out, kernel_size=1, stride=2)
#               ↑ 1×1×1 stride=2，负责通道数变换（C_in→C_out）+ 空间下采样
```

```
输入 x (B, C_in, D, H, W)
    │
    ├── res_conv: Conv3d(1×1×1, stride=2, C_in→C_out) ──────────── + ──→ 输出
    │                                                               ↑      (B, C_out, D/2, H/2, W/2)
    conv1: DW-Conv3d(k=3, stride=2, groups=C_in)                   │
    GroupNorm → PW_expand → GELU → PW_compress ────────────────────┘
```

**关键**：下采样不用额外的 MaxPool 或 stride=2 标准卷积，而是直接在 DW 大核上做 stride=2，带来通道不变的空间降采样，再通过 PW compress 改变通道数。

---

### MedNeXtUpBlock（stride=2，上采样）

源码：`blocks.py` 第 146 行

与 DownBlock 对称，`conv1` 换成 `ConvTranspose3d`（转置卷积，做空间上采样）：

```python
self.conv1 = ConvTranspose3d(C_in, C_in,
    kernel_size=k, stride=2, padding=k//2, groups=C_in)
# 上采样输出尺寸有 ±1 体素的对齐问题，手动 pad：
x1 = F.pad(x1, (1,0,1,0,1,0))  # 每维度在左侧 pad 1
```

---

### checkpoint_style='outside_block' 说明

```
正常训练（无 checkpointing）：
  forward:  [Block0 激活] [Block1 激活] ... [Block7 激活]  ← 全部保留在显存
  backward: 直接用已保存的激活算梯度

checkpoint='outside_block'：
  forward:  只保留 Block 输入，丢掉内部激活
  backward: 需要梯度时重跑一遍 Block forward 拿激活 → 多一次计算
  显存节省：~50%，计算代价：~+30%
```

MedNeXt 瓶颈层（C=512, exp_r=8）中间激活是 512×8=4096 通道，8 个 block 叠加，显存极大，必须开 checkpointing。

---

## 五、与项目内其他方案关键区别汇总

| 维度 | nnUNet Baseline | MedNeXt-L(k=3) | IBConvBlock3D(k=7) | DWSepConv3d(k=7) |
|------|:--------------:|:--------------:|:------------------:|:----------------:|
| 编解码器级数 | 6 | **9** | 6 | 6 |
| 每级 block 数 | 2 | **3~8** | 2 | 2 |
| 主块卷积核 | k=3 标准 | k=3 DW | k=7 DW | k=7 DW |
| 下采样卷积 | k=3 标准 stride=2 | **DW(k=3, stride=2)+残差** | k=3 标准 stride=2 | DWSep(k=7, stride=2) |
| expand ratio | 无 | **3~8（各层不同）** | 4（固定）| 无 |
| 块内残差（主块）| ✗ | ✓ (do_res=True) | ✓ | ✗ |
| 下采样残差 | ✗ | ✓ (do_res_up_down=True) | ✗ | ✗ |
| 块内 Norm | InstanceNorm（外层）| GroupNorm（块内）| InstanceNorm（块内）| InstanceNorm（外层）|
| 感受野（stride=1）| ~13 体素 | ~13 体素（k=3）| ~37 体素 | ~37 体素 |
| 参数量 | ~31M | **~62M** | ~35M | ~28M |
| Gradient Checkpointing | ✗ | ✓ | 可选 | ✗ |

---

## 五、与项目内其他方案对比

| 维度 | MedNeXt-L(k=3) | IBConvBlock3D(k=7) | DWSepConv3d(k=7) | nnUNet Baseline |
|------|:--------------:|:------------------:|:----------------:|:---------------:|
| 主块卷积核大小 | k=3 | k=7 | k=7 | k=3 |
| 下采样卷积 | DW(k=3, stride=2)+残差 | **标准 Conv(k=3, stride=2)** | DWSep(k=7, stride=2) | 标准 Conv(k=3, stride=2) |
| 块内 Norm | GroupNorm | InstanceNorm3d | 无（外层 IN）| InstanceNorm3d（外层）|
| expand ratio | 3~8（各层不同）| 4（固定）| 无 | 无 |
| 块内残差 | ✓ | ✓ | ✗ | ✗ |
| 下采样残差 | ✓ | ✗ | ✗ | ✗ |
| 编解码器级数 | 9 | 6（nnUNet 计划）| 6 | 6 |
| 每级 block 数 | 3~8 | 2（nnUNet 默认）| 2 | 2 |
| 感受野（stride=1 层）| ~13 体素 | ~37 体素 | ~37 体素 | ~13 体素 |

---

## 六、结论

1. **我们的 MedNeXt 是官方标准配置**（`MedNeXt_L_kernel3`），k=3 是 3D 任务官方推荐。

2. **MedNeXt 性能优势不来自大核感受野**，而来自：
   - IB 倒置瓶颈结构（DW + expand×r + GELU + compress + 残差）
   - 更深的网络（9 级 vs 6 级）
   - 更多 block（最多 8 个/级 vs 2 个/级）
   - 更大 expand ratio（最高 8×）
   - 下采样块的残差连接

3. **IBConv7(k=7) vs MedNeXt-L(k=3) 不是纯粹的大核对比**，两者差异混杂了核大小、网络深度、expand ratio 等多个因素。若要单独对比大核贡献，需要 MedNeXt-L(k=3) vs MedNeXt-L(k=7) 的消融。
