# EfficientMedNeXt：结构、来源与实验边界

> 本文件由原网络原理分析与官方 baseline 实施记录合并。结构说明面向不了解该网络的读者；来源、适配入口和当前实验边界集中放在末尾。

## 1. 一句话理解

EfficientMedNeXt 是一个三维 U-Net。它用轻量的多感受野 depthwise 卷积替代较重的普通卷积，并把 Decoder 的通道数固定得很窄，从而减少参数量。

它仍然遵循 U-Net 的基本流程：

> 输入 CT → Encoder 逐级下采样、提取语义 → bottleneck → Decoder 逐级上采样 → 与 Encoder 的 skip connection 相加 → 输出分割结果。

---

## 2. 网络的整体形状

以输入 patch `[B, 1, 128, 128, 128]` 为例，`B` 是 batch size，后面三个数字是三维空间尺寸。

| 阶段 | 通道数 | 空间尺寸 |
|---|---:|---:|
| 输入 | 1 | `128×128×128` |
| Stem | 32 | `128×128×128` |
| Encoder 0 | 32 | `128×128×128` |
| Down 0 | 64 | `64×64×64` |
| Encoder 1 | 64 | `64×64×64` |
| Down 1 | 128 | `32×32×32` |
| Encoder 2 | 128 | `32×32×32` |
| Down 2 | 256 | `16×16×16` |
| Encoder 3 | 256 | `16×16×16` |
| Down 3 | 512 | `8×8×8` |
| bottleneck 前 | 512 | `8×8×8` |
| bottleneck 后 | 32 | `8×8×8` |
| Decoder 3 | 32 | `16×16×16` |
| Decoder 2 | 32 | `32×32×32` |
| Decoder 1 | 32 | `64×64×64` |
| Decoder 0 | 32 | `128×128×128` |
| 输出 | 3 | `128×128×128` |

Encoder 的通道数是：

```text
32 → 64 → 128 → 256 → 512
```

但 Decoder 不再反向恢复成 `256、128、64、32`，而是统一使用 32 通道。这是它参数量很少的关键原因。

### 2.1 高分辨率、低分辨率和细节分别在哪里

在 U-Net 中，分辨率和语义信息大致这样变化：

```text
Encoder：高分辨率、局部细节 → 低分辨率、整体语义
                         ↓
                     bottleneck
                         ↓
Decoder：低分辨率、整体语义 → 高分辨率、边界细节
```

具体分工是：

| 部分 | 分辨率变化 | 主要作用 |
|---|---|---|
| Encoder 前部 | 高分辨率 | 保留边缘、纹理、小病灶和位置信息 |
| Encoder 后部 | 低分辨率 | 提取更抽象的器官和肿瘤语义关系 |
| bottleneck | 最低分辨率 | 汇总最强的整体语义 |
| Decoder 前部 | 从低分辨率开始 | 将语义特征逐步恢复到更高分辨率 |
| Decoder 后部 | 高分辨率 | 恢复分割边界和精确空间位置 |

因此，“细节部分”不能简单说只属于 Encoder 或只属于 Decoder：

- Encoder 前部在高分辨率特征中保存细节；
- Decoder 负责把低分辨率语义恢复成高分辨率预测；
- skip connection 把 Encoder 保存的高分辨率细节送回 Decoder；
- Decoder 将“Encoder 的细节”与“bottleneck 的语义”结合，得到最终边界。

在本网络中，skip connection 使用相加：

```text
Decoder 上采样后的特征 + Encoder 对应层的高分辨率特征
```

所以可以记成一句话：

> Encoder 负责“看懂是什么以及在哪里”，Decoder 负责“把看懂的结果恢复到准确的像素/体素位置”；高分辨率细节主要由 Encoder 前部保存，并通过 skip connection 帮助 Decoder 恢复。

### 2.2 纹理、小病灶和整体语义的区别

#### 纹理是不是在 Encoder 前部

是的。Encoder 前部仍然保留较大的空间尺寸，因此更容易保留：

- CT 局部明暗变化；
- 边缘和轮廓；
- 肝脏与周围组织的局部对比度；
- 肿瘤内部的粗糙、均匀或不均匀外观。

这里的“纹理”不是只指二维图像纹理，也包括三维邻近切片之间的局部变化。

#### 小病灶是不是只在 Encoder 前部

不是。小病灶的信息会经过不同层，但每一层保存的内容不同：

| 位置 | 小病灶在这一层的表现 |
|---|---|
| Encoder 前部、高分辨率 | 保留小病灶的真实大小、边缘、明暗和精确位置，但还不容易判断它是不是肿瘤 |
| Encoder 后部、低分辨率 | 空间细节变粗，甚至可能因下采样而变弱，但可以结合肝脏位置、邻近组织和三维上下文判断其语义 |
| Decoder | 用后部的语义判断，加上前部通过 skip connection 传回的细节，恢复小病灶的边界和位置 |

因此，小病灶最怕的是过早下采样；skip connection 的作用就是把高分辨率层中没有完全丢失的小病灶证据送回 Decoder。后部语义不是用来恢复精确边界的，而是帮助判断“这个局部异常是否更像肿瘤”。

另外，肿瘤并不都是小东西。病灶可能从很小的结节到占据较大区域的肿块；“小病灶”只是最容易漏检的一类。大病灶主要依赖整体形状和区域语义，小病灶更依赖高分辨率细节与上下文的共同作用。

#### “整体语义”到底是什么意思

整体语义不是一个模糊的“更高级特征”，而是对整块区域关系的判断，例如：

- 这个位置是否在肝脏内部；
- 这个亮度或密度异常是否位于合理的肿瘤区域；
- 当前局部结构是否和相邻切片中的病灶连续；
- 这个小亮点更像肿瘤，还是血管、噪声或正常组织；
- 预测区域是否与肝脏整体形状相容。

低分辨率特征牺牲了一部分精确位置，换来更大的有效观察范围。因此它更擅长回答：

> “这片区域整体上是什么、和周围是什么关系？”

而高分辨率特征更擅长回答：

> “边界具体在哪里、这个小结构占哪些 voxel？”

最后的分割需要两者同时存在：语义负责减少把血管或噪声当成肿瘤，细节负责不把真实肿瘤边界抹掉。

---

## 3. 一个 EfficientMedNeXt block 怎么工作

输入记作：

```text
[B, C, D, H, W]
```

一个 block 主要有四步：

```text
输入
  ↓
三个 depthwise 3D 卷积分支
  ↓
沿通道维拼接，C → 3C
  ↓
GroupNorm + GELU
  ↓
1×1×1 卷积，3C → Cout
  ↓
残差相加
```

### 3.1 三个卷积分支

代码传入的配置是：

```python
kernel_sizes = [1, 3, 5]
```

但实际不是直接使用 `1×1×1、3×3×3、5×5×5` 三个卷积，而是：

| 分支 | 实际卷积 | dilation | 有效感受野 |
|---|---|---:|---:|
| 1 | `1×1×1` depthwise | 1 | 1 |
| 2 | `3×3×3` depthwise | 1 | 3 |
| 3 | `3×3×3` depthwise | 2 | 5 |

第三个分支用 `3×3×3` 加 dilation=2，获得类似 `5×5×5` 的感受野，但参数比真正的 `5×5×5` 少。

### 3.2 为什么叫 depthwise

普通卷积会同时混合空间和通道：

```text
输入通道之间相互连接
```

depthwise 卷积则是每个通道单独做空间卷积：

```text
第 1 个通道单独卷积
第 2 个通道单独卷积
...
第 C 个通道单独卷积
```

因此三个分支的输出仍然各有 `C` 个通道。拼接后：

```text
[B, C, D, H, W] → [B, 3C, D, H, W]
```

### 3.3 1×1×1 卷积负责通道混合

三个 depthwise 分支只负责空间信息。拼接之后的 `1×1×1` 卷积负责不同通道之间的信息混合，并把 `3C` 压缩到 `Cout`：

```text
[B, 3C, D, H, W] → [B, Cout, D, H, W]
```

所以这个 block 可以理解为：

> 多种空间感受野提取局部特征，再用 1×1×1 卷积混合通道，最后通过残差连接保持原始信息。

---

## 4. DownBlock 和 UpBlock

### 4.1 DownBlock

DownBlock 仍然使用上面的三分支 block，但 stride=2：

```text
[B, C, D, H, W]
        ↓
[B, Cout, D/2, H/2, W/2]
```

它同时完成两件事：

1. 提取特征；
2. 将三维空间尺寸缩小一半。

例如：

```text
[B, 32, 128, 128, 128]
→ [B, 64, 64, 64, 64]
```

### 4.2 UpBlock

UpBlock 将三分支中的卷积替换为 `ConvTranspose3d`，stride=2：

```text
[B, C, D, H, W]
        ↓
[B, Cout, 2D, 2H, 2W]
```

上采样后与对应 Encoder 的 skip 特征相加，而不是通道拼接：

```text
Decoder 特征 + Encoder skip 特征
```

在相加前，Encoder 的 skip 特征会先通过一个 block 压缩到 32 通道。

---

## 5. Encoder、bottleneck、Decoder 的完整过程

### 5.1 Encoder

```text
输入 [B,1,128,128,128]
  ↓ Stem
[B,32,128,128,128]
  ↓ Encoder 0
[B,32,128,128,128]
  ↓ Down 0
[B,64,64,64,64]
  ↓ Encoder 1
[B,64,64,64,64]
  ↓ Down 1
[B,128,32,32,32]
  ↓ Encoder 2
[B,128,32,32,32]
  ↓ Down 2
[B,256,16,16,16]
  ↓ Encoder 3
[B,256,16,16,16]
  ↓ Down 3
[B,512,8,8,8]
```

Encoder 越往下，空间分辨率越低，通道数越高，特征越偏向整体语义。

### 5.2 bottleneck

到达最低分辨率后，网络先把：

```text
[B,512,8,8,8] → [B,32,8,8,8]
```

然后使用 4 个 32 通道 EfficientMedNeXt block。

这里的 bottleneck 不负责 Transformer attention，也没有 token 化；它仍然是卷积 block，只是在最小空间尺寸上进行特征变换。

### 5.3 Decoder

Decoder 每一级都保持 32 通道：

```text
[B,32,8,8,8]
  ↓ Up 3 + skip 3
[B,32,16,16,16]
  ↓ Up 2 + skip 2
[B,32,32,32,32]
  ↓ Up 1 + skip 1
[B,32,64,64,64]
  ↓ Up 0 + skip 0
[B,32,128,128,128]
  ↓ 1×1×1 输出层
[B,3,128,128,128]
```

最后的 3 个通道对应：

```text
background、liver、tumor
```

---

## 6. 为什么参数量只有约 2.19M

主要有三个原因：

1. 空间卷积使用 depthwise，避免了 `Cin×Cout` 的完整通道连接；
2. `3×3×3 dilation=2` 代替真正的 `5×5×5`，扩大感受野但不保存 125 个卷积位置；
3. Decoder 全部固定 32 通道，避免在高分辨率阶段使用很宽的特征图。

要注意：参数量少只说明需要学习的权重少，不等于所有中间特征图都小。比如高分辨率 Encoder 仍然要处理 `128³` 的三维特征图，三个分支拼接时会临时形成：

```text
[B, 96, 128, 128, 128]
```

---

## 7. 梯度检查点是什么

Large 配置默认使用：

```python
checkpoint_style = "outside_block"
```

普通训练流程会这样保存中间结果：

```text
前向：保存大量中间激活
反向：直接使用这些激活计算梯度
```

梯度检查点则改成：

```text
前向：少保存一些中间激活
反向：重新执行部分前向，恢复需要的激活，再计算梯度
```

所以它是一个显存和时间的交换：

| 设置 | 显存 | 反向计算 | 通常速度 |
|---|---|---|---|
| 开启 checkpoint | 较少 | 需要重算 | 较慢 |
| 关闭 checkpoint | 较多 | 少重算 | 可能较快 |

### 7.1 关闭后会发生什么

关闭 checkpoint 不会改变网络的层、参数量和输出形状，只会改变执行方式：

```python
checkpoint_style = None
```

关闭后，训练时会保留更多激活，因此峰值显存通常会明显上升；如果显存足够，训练可能变快；如果显存不够，会 OOM。

### 7.2 为什么参数少仍可能不快

EfficientMedNeXt 的参数和理论计算量虽然较低，但实际速度还受以下因素影响：

- 三个分支需要分别执行卷积；
- 分支输出需要拼接；
- depthwise 和 dilation 算子不一定能充分利用 GPU；
- 三维转置卷积和 padding 会产生额外数据操作；
- 开启梯度检查点后，反向传播要重新计算部分 block。

因此正确的理解是：

> EfficientMedNeXt 主要优化的是参数量和部分计算量；它不保证在所有 GPU、batch size 和执行设置下都有更短的训练时间。

---

## 8. 最后用一句话记住

EfficientMedNeXt = **三种有效感受野的 depthwise 3D 卷积 + 1×1×1 通道混合 + 窄 Decoder 的三维 U-Net**。

它通过减少通道连接和压窄 Decoder 来减少参数，通过 dilation 保留较大感受野；梯度检查点则是额外的显存优化机制，不是 EfficientMedNeXt 结构本身。

---

## 9. 两阶段优化的研究逻辑

EfficientMedNeXt 的三个贡献不是三个独立模块，而是一条连续路线：整体高效架构是最终结果，two-phase architecture optimization 是形成方法，DMRFB 是第二阶段用于补回能力的核心 block。

第一阶段先验证结构冗余：

- **HRR（High-Resolution Redundancy Removal）**：移除完整分辨率 `H×W×D` 的高分辨率解码层及对应输出，不只是减少 stage 内的 block；
- **UDC（Unified Decoder Channels）**：解码器各尺度采用统一的低通道数。

论文消融中，HRR 与 UDC 合计使参数下降 **72.7%**、FLOPs 下降 **53.9%**，平均 Dice 只下降 **0.38 个百分点**。第二阶段加入 DMRFB 后，平均 Dice 相对瘦身模型回升 **1.42 个百分点**。完整论证链是：

> 先用消融证明原解码路径存在冗余，再把节省的预算用于多感受野空间建模，使能力恢复并提升。

这里仍需避免两个泛化错误：DMRFB 的真实三支不是 `dilation=1/2/3`；不同尺寸变体也不一定使用完全相同的 DMRFB 配置，讨论结果时必须注明具体型号。

## 10. 与当前 PlainConvDecoder 实验的关系

当前 `nnUNetTrainer_MedNeXt_MLA_MoE_PlainConvDecoder` 同样检验“U-Net 右侧是否被过度设计”，但不是 EfficientMedNeXt 的复现：

| 对比项 | PlainConvDecoder | EfficientMedNeXt 第一阶段 |
|---|---|---|
| Encoder | 保留完整 MedNeXt-L | 整体架构共同轻量化 |
| Bottleneck | 保留 MedNeXt + MLA/MoE | 无本仓库的 MLA/MoE |
| Decoder | 每级换成单个普通卷积 stage | HRR + UDC |
| 当前证据 | 全模型参数下降 15.30%，但三域 Tumor Dice 均下降 | 已报告参数、FLOPs 与多数据集 Dice 消融 |

因此，PlainConvDecoder 不能声称复现论文的 72.7% 压缩。其三域正式结果已经证明：虽然 decoder 参数下降 75.1%、全模型参数下降 15.30%，但 Internal、IRCADb、HCC Tumor Dice 分别下降 0.0142、0.0144、0.1024。它是已完成的负向消融，不能用来证明原始 decoder 冗余。详细证据见 [`../实验与分析/负向与近中性消融汇总.md`](../实验与分析/负向与近中性消融汇总.md)。

## 11. 官方来源与本仓库适配

- 官方仓库：`https://github.com/SLDGroup/EfficientMedNeXt`；
- 固定 commit：`803f7efed9b728ac93ae4e0d8a2602501135241f`；
- 本地独立仓库：`/home/PuMengYu/EfficientMedNeXt`；
- 许可证：UT Austin Research License，仅按学术/研究用途使用；
- architecture adapter：`pumengyu/architectures/efficient_mednext_official.py`；
- trainer：`pumengyu/trainers/efficient_mednext_trainer.py`；
- trainer class：`nnUNetTrainer_EfficientMedNeXt_L_Official`。

本仓库不复制官方网络源码，而是从独立官方仓库加载，并校验三个核心文件：

| 文件 | SHA-256 |
|---|---|
| `EfficientMedNext_Full.py` | `a7b8348534bddfcac949f56e03af2fa67e91fab45f4b5d388341d2dad456d7c8` |
| `efficient_mednext_blocks.py` | `3e12e9de5d85b2582c03a7d791902711085411eea18564c9aaf9a032bac85c4d` |
| `create_efficient_mednext.py` | `79810104a3b69f5bc5cf4c83165226dcad9ced6027b7a2c0d9090f89e29231a0` |

文件缺失或哈希变化时，构建应显式失败。固定 L 配置为 base/uniform decoder channels 32、kernel specification `[1,3,5]`、block counts `[3,4,4,4,4,4,4,4,3]`、残差与深监督开启。Dataset003 单通道输入、三分类输出时参数量为 2,193,808。

早期接入阶段已通过：Python 编译、哈希检查、Trainer 发现、CPU forward/backward、五级深监督和关闭深监督后的单输出。当前三域正式结果已存在；最新产物状态以 [`../实验与分析/FCSU-Net参照下的项目论文级审计_20260831.md`](../实验与分析/FCSU-Net参照下的项目论文级审计_20260831.md) 为准，不再保留“训练未开始”的历史快照。

## 12. 核对依据

- 论文：`pumengyu/notes/paper/外界参考论文/4895_paper.pdf`
- 官方代码：`/home/PuMengYu/EfficientMedNeXt`
- 当前受控消融：`pumengyu/notes/paper/实验与分析/负向与近中性消融汇总.md`
