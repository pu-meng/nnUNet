# MedNeXt、MLA 与 MoE 基础模块：结构、卷积核与计算量

> 定位：面向代码理解的基础说明，供论文方法部分和汇报材料查阅；不是当前正式主稿。



相关实现：

- [MedNeXt-L wrapper](/home/PuMengYu/nnUNet/pumengyu/architectures/mednext.py)
- [MedNeXt 原始 block 实现](/home/PuMengYu/anaconda3/envs/medseg/lib/python3.10/site-packages/nnunet_mednext/network_architecture/mednextv1/blocks.py)
- [MLA、MoE 与 bottleneck 实现](/home/PuMengYu/nnUNet/pumengyu/architectures/mla_unetr.py)

## 1. 先区分四种连接

| 名称 | 所在位置 | 运算 | 是否改变通道/空间尺寸 |
|---|---|---|---|
| Block residual | 每个 `MedNeXtBlock` 内 | 主分支输出 + block 输入 | 普通 block 中不改变 |
| Resampling residual | `MedNeXtDownBlock`、`MedNeXtUpBlock` | 重采样主分支 + `1×1×1` stride-2 shortcut | 改变空间尺寸和通道数 |
| Encoder–decoder skip | decoder 每一级 | `UpBlock` 输出 + encoder 同尺度输出 | 不 concat，逐元素 add |
| MLA/MoE residual | 每个 `MLATransformerBlock` 内 | MLA 分支或 MoE-FFN 分支 + 输入 | token 形状不变 |

当前配置为 `do_res=True` 和 `do_res_up_down=True`。因此普通 block、DownBlock、UpBlock 和 MLA/MoE block 中的加法不能统称为同一种 skip connection。

### 1.1 Stage、Block 和 Bottleneck 的层次

`Stage` 本身主要是一个组织单元，不是一个额外执行总残差相加的模块。在代码中，`enc_block_i`/`dec_block_i` 是固定分辨率的 block 容器；`down_i`/`up_i` 是容器外、连接相邻尺度的独立过渡模块。当前实现中，stage 与残差的位置可以按下面的规则读取：

```text
Encoder Stage i
  └─ MedNeXtBlock × n
       每一个 block 各自执行一次 block residual
  ↓
DownBlock
  └─ 执行一次 resampling residual，并把特征送到下一尺度
```

当前 encoder 的实际层次为：

```text
Stem
→ Encoder Stage 0: MedNeXtBlock ×3
→ DownBlock 0
→ Encoder Stage 1: MedNeXtBlock ×4
→ DownBlock 1
→ Encoder Stage 2: MedNeXtBlock ×8
→ DownBlock 2
→ Encoder Stage 3: MedNeXtBlock ×8
→ DownBlock 3
→ Conv bottleneck stage: MedNeXtBlock ×8
→ MLABottleneck3D
```

这里必须特别区分两个 bottleneck：

- **MedNeXt 卷积 bottleneck**：最低空间分辨率处的 `MedNeXtBlock ×8`，这 8 个 block 每个都有自己的 block residual；
- **`MLABottleneck3D`**：接在上述 8 个卷积 block 之后的新增 token 上下文模块，内部包含 2 个 MLA+MoE Transformer block。

因此，不能把卷积 bottleneck 简化成“一个 MedNeXtBlock”，也不能把 `MLABottleneck3D` 当成原始 MedNeXt bottleneck 本身。

Decoder 每一级的顺序为：

```text
UpBlock
  └─ 一条 resampling residual
→ 与同尺度 encoder feature 做 skip add
→ MedNeXtBlock × n
  └─ 每一个 block 各自执行 block residual
```

当前 decoder 的 block 数为：

```text
Decoder Stage 3: UpBlock → skip add → MedNeXtBlock ×8
Decoder Stage 2: UpBlock → skip add → MedNeXtBlock ×8
Decoder Stage 1: UpBlock → skip add → MedNeXtBlock ×4
Decoder Stage 0: UpBlock → skip add → MedNeXtBlock ×3
```

所以“stage 没有一个总 residual”是正确的；残差发生在 stage 内部的每个 block、尺度转换的 Down/UpBlock，以及另外独立存在的 encoder–decoder skip add。

### 1.2 真实代码位置

这些关系不是绘图约定，而是 `MedNeXt.__init__()` 和 `MedNeXtMLABot.forward()` 中的真实模块定义与调用顺序：

| 结构 | 代码位置 | 真实内容 |
|---|---|---|
| `enc_block_0` | `MedNextV1.py:51–63` | `MedNeXtBlock × block_counts[0]` |
| `down_0` | `MedNextV1.py:65–73` | 独立的 `MedNeXtDownBlock` |
| `enc_block_1` / `down_1` | `MedNextV1.py:75–98` | stage 容器与下一尺度过渡分开定义 |
| `enc_block_2` / `down_2` | `MedNextV1.py:100–123` | stage 容器与下一尺度过渡分开定义 |
| `enc_block_3` / `down_3` | `MedNextV1.py:125–148` | stage 容器与 bottleneck 过渡分开定义 |
| `bottleneck` | `MedNextV1.py:150–162` | 最低分辨率 `MedNeXtBlock × block_counts[4]` |
| `up_3` / `dec_block_3` | `MedNextV1.py:164–187` | 独立 UpBlock 后接 decoder block 容器 |
| `up_2` / `dec_block_2` | `MedNextV1.py:189–212` | 独立 UpBlock 后接 decoder block 容器 |
| `up_1` / `dec_block_1` | `MedNextV1.py:214–237` | 独立 UpBlock 后接 decoder block 容器 |
| `up_0` / `dec_block_0` | `MedNextV1.py:239–262` | 独立 UpBlock 后接 decoder block 容器 |

在本项目 wrapper 中，真实 forward 顺序位于 [`mednext.py:124–169`](/home/PuMengYu/nnUNet/pumengyu/architectures/mednext.py:124)：先运行 `enc_block_i`，再运行 `down_i`；decoder 中先运行 `up_i`，再执行 `dec_x = x_res_i + x_up_i` 的 skip add，最后运行 `dec_block_i`。因此图 1 的 DownBlock/UpBlock 色块表示代码中的独立模块，而不是 stage 容器内部的 block。

## 2. 单个 MedNeXtBlock

### 2.1 实际 forward 顺序

```text
x
 ├─ identity shortcut ───────────────────────────────┐
 └─ Depthwise Conv3d(k=3, stride=1, groups=C)       │
     → GroupNorm(C groups)                           │
     → Conv3d(1×1×1, C → rC)                         │
     → GELU                                           │
     → Conv3d(1×1×1, rC → C)                         │
                                                     ↓
                              element-wise add <─────┘
```

若输入为 `[B,C,D,H,W]`，普通 MedNeXtBlock 的输出仍为 `[B,C,D,H,W]`。当前 MedNeXt-L 的 9 个 group 使用 `exp_r=[3,4,8,8,8,8,8,4,3]`。

### 2.2 普通卷积如何组合输入通道

对于一般的 `Conv3d(Cin,Cout,k)`，权重形状是：

```text
[Cout, Cin, k, k, k]
```

因此总共有 `Cout×Cin` 个三维卷积核。对第 `o` 个输出通道，计算方式是：

```text
输出[o] = 输入[0] * W[o,0]
        + 输入[1] * W[o,1]
        + ...
        + 输入[Cin-1] * W[o,Cin-1]
        + bias[o]
```

也就是说：

- 一个输出通道对应 `Cin` 个卷积核；
- 这 `Cin` 个卷积核分别处理 `Cin` 个输入通道；
- `Cin` 个处理结果相加，形成一个输出通道；
- 一共有 `Cout` 个这样的输出通道。

例如 `Conv3d(2,3,3)` 有 6 个卷积核：

```text
输出通道 0 = 输入通道 0 用 W[0,0] + 输入通道 1 用 W[0,1]
输出通道 1 = 输入通道 0 用 W[1,0] + 输入通道 1 用 W[1,1]
输出通道 2 = 输入通道 0 用 W[2,0] + 输入通道 1 用 W[2,1]
```

因此普通卷积不是“所有输入通道共用一个卷积核”。卷积核的空间权重会在不同空间位置重复使用，但不同输入通道—输出通道组合通常使用不同参数。

### 2.3 Depthwise convolution 如何不同

Depthwise convolution 使用 `groups=Cin`。此时每个输入通道只对应一个独立卷积核：

```text
输入通道 0 → kernel 0 → 输出通道 0
输入通道 1 → kernel 1 → 输出通道 1
输入通道 2 → kernel 2 → 输出通道 2
```

这些通道之间不做跨通道求和，因此 depthwise convolution 只建模每个通道内部的空间邻域，不负责通道混合。MedNeXtBlock 后面的 `1×1×1` 卷积才负责通道混合：

```text
Depthwise Conv：每个通道独立做空间卷积
1×1×1 Conv C→rC：混合并扩展通道
1×1×1 Conv rC→C：再次混合并压回通道
```

### 2.4 卷积核数量与标量权重

| 卷积 | 输出通道 | 每个 output kernel | kernel 数量 | 标量权重数量 |
|---|---:|---|---:|---:|
| Depthwise `3×3×3` | `C` | `1×3×3×3` | `C` | `27C` |
| Expansion `1×1×1` | `rC` | `C×1×1×1` | `rC` | `rC²` |
| Projection `1×1×1` | `C` | `rC×1×1×1` | `C` | `rC²` |
| **合计** | — | — | **`(r+2)C`** | **`27C+2rC²`** |

例如 bottleneck 的 `C=512,r=8`：Depthwise 为 512 个 kernel、Expansion 为 4096 个 output kernels、Projection 为 512 个 output kernels；合计 5120 个 output kernels，4,208,128 个卷积标量权重。这里不计 bias、GroupNorm affine 参数和 residual add。

## 3. 普通 3D 卷积与 MedNeXt block

以保持通道数不变的 dense `3×3×3 Conv3d (C→C)` 为基准：

```text
Dense parameters = 27C²
Dense MACs       = D×H×W×27C²

MedNeXt parameters = 27C + 2rC²
MedNeXt MACs       = D×H×W×(27C + 2rC²)
```

因此：

```text
MedNeXt / dense = (27C + 2rC²)/(27C²)
```

这里的 MedNeXt block 还包含两个 `1×1×1` 通道投影，所以不能简单说成“普通卷积的 `1/C`”。`r` 越大，通道投影成本越大。

| group | `C` | `r` | 重复次数 | dense 权重 | MedNeXt 权重 | 比例 | 单 block MAC（dense → MedNeXt） |
|---|---:|---:|---:|---:|---:|---:|---:|
| Encoder 0 | 32 | 3 | 3 | 27,648 | 7,008 | 25.3% | 58.0G → 14.7G |
| Encoder 1 | 64 | 4 | 4 | 110,592 | 34,496 | 31.2% | 29.0G → 9.0G |
| Encoder 2 | 128 | 8 | 8 | 442,368 | 265,600 | 60.0% | 14.5G → 8.7G |
| Encoder 3 | 256 | 8 | 8 | 1,769,472 | 1,055,488 | 59.6% | 7.25G → 4.32G |
| Bottleneck | 512 | 8 | 8 | 7,077,888 | 4,208,128 | 59.5% | 3.62G → 2.15G |

MAC 按 `128³` patch 和对应 stage 空间尺寸计算；实际运行还包括归一化、GELU、内存访问、checkpointing、上下采样和 kernel 调度。

当前 9 个 group 共 `3+4+8+8+8+8+8+4+3=54` 个普通 MedNeXtBlock：encoder 与卷积 bottleneck 共 31 个，decoder 共 23 个。DownBlock/UpBlock 是额外的重采样模块，不能简单计入这 54 个保持尺寸的 block。

## 4. Depthwise-Separable Convolution

标准 3D depthwise-separable convolution 是：

```text
Depthwise Conv：每个通道独立做空间卷积
→ Pointwise Conv：用 1×1×1 在每个体素位置混合通道
```

对 `[B,C,D,H,W]`：depthwise 有 `C` 个 `1×3×3×3` kernel；pointwise 在每个空间位置对 `C` 维向量做线性组合。当前 MedNeXtBlock 比最简单的 depthwise-separable convolution 多一个 pointwise projection，并将通道变换组织为 `C→rC→C`。

| 结构 | 卷积权重数量 |
|---|---:|
| Dense `3×3×3 C→C` | `27C²` |
| 单纯 depthwise + pointwise `C→C` | `27C+C²` |
| 当前 MedNeXtBlock | `27C+2rC²` |

## 5. MedNeXt Inverted Bottleneck 与 `1×1×1`

`1×1×1` 卷积不读取邻近体素，只在固定 `(d,h,w)` 位置对通道向量做矩阵变换。当前实际顺序是：

```text
Depthwise Conv → GroupNorm → 1×1 expansion C→rC → GELU → 1×1 projection rC→C
```

因此，inverted bottleneck 指中间通道宽度先扩张、再压回输入宽度；不是空间上采样，也不是转置卷积。空间邻域建模由 depthwise `3×3×3` 完成，通道混合由两个 pointwise `1×1×1` 完成。

## 6. MLA block

MLABottleneck3D 将卷积 bottleneck `[B,C,D,H,W]` 转成 `[B,N,C]`，`N=D×H×W`。主模型中 `C=512,N=512,num_heads=8,d_head=64,d_c=128`。

```text
x → LayerNorm → MLA → residual add
  → LayerNorm → MoE-FFN → residual add
```

MLA 的投影参数为：

| 投影 | 形状 | 权重数 |
|---|---|---:|
| `W_Q` | `512→512` | 262,144 |
| `W_DKV` | `512→128` | 65,536 |
| `W_UK` | `128→512` | 65,536 |
| `W_UV` | `128→512` | 65,536 |
| `W_O` | `512→512` | 262,144 |
| **合计** | — | **720,896** |

标准 MHA 的四个投影为 `4×512²=1,048,576`。MLA 约为其 68.75%，但仍计算完整 `N×N` attention 矩阵，并非线性 attention。

## 7. MoE-FFN

当前配置：`d_model=512`、`mlp_ratio=4`、`d_ff=1024`、4 个 routed experts、`top_k=2`。结构是 1 个始终计算的 shared expert、4 个 routed experts 和 `Router:512→4`。

每个 expert 为 `Linear 512→1024 → GELU → Linear 1024→512`，两层权重共 1,048,576。MoE-FFN 总权重约为：

```text
shared 1,048,576 + routed 4×1,048,576 + router 2,048
= 5,244,928（不计 bias）
```

代码会先计算全部 4 个 routed expert outputs，再 gather Top-2。因此逻辑上只有两个 routed outputs 参与融合，但实际前向仍计算 4 个 routed experts；不能宣称获得严格稀疏计算加速。对 `N=512` token，实际 expert MAC 约 2.68G，理想只算 shared+top-2 约 1.61G。

## 8. 结论

```text
MedNeXtBlock：depthwise 3×3×3 做局部空间建模，1×1×1 做 C→rC→C 通道变换，最后 block residual add。
MLA：低秩压缩 K/V 投影参数，但当前仍使用完整 N×N attention。
MoE：shared expert + routed experts 增加条件 FFN 容量，但当前实现实际计算全部 routed experts。
```
