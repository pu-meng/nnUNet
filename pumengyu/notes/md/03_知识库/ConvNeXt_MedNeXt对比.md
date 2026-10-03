# 普通卷积、ConvNeXt 与 MedNeXt

这篇只回答三种结构如何分配空间混合与通道混合的计算。当前 MedNeXt-L 的具体层数、skip、上下采样与 `forward()` 见 [MedNeXt 架构](MedNeXt架构.md)；本仓库旧 DWSep/IBConv 替换见 [结构和效率边界](DWSep_vs_IBConv机制.md)。

## 核心区别

| 结构 | 空间混合 | 通道混合 | 常见归一化/激活 | 残差与维度 |
|---|---|---|---|---|
| 普通 3D 卷积块 | `Conv3d(k×k×k, C_in→C_out)` 一次完成 | 与空间卷积耦合 | 本仓库 nnU-Net 默认块依 plans 使用 IN 与 LeakyReLU | stage 可以改变通道数和空间尺寸 |
| ConvNeXt block（2D） | 逐通道 depthwise 卷积 | pointwise 扩展、GELU、压缩 | LayerNorm | 同尺度主块有残差；网络另设下采样 |
| MedNeXt block（3D） | 逐通道 3D depthwise 卷积 | pointwise 扩展、GELU、压缩 | 取决于配置；本仓库 MedNeXt-L 用 GroupNorm | 主块及上下采样块的残差、形状处理分别定义 |

普通 3D 卷积的权重约为 `k³·C_in·C_out`。若 `C_in=C_out=C`，depthwise 加一个 pointwise 的权重约为 `k³·C + C²`；倒置瓶颈再加入 `C→rC→C` 两个 pointwise 层时约为 `k³·C + 2rC²`。这些是**单块权重**的近似式，不包含归一化、偏置、投影和整网各 stage 的差异。节省参数可能允许更大的空间核或更多块，但不保证实际训练更快；3D depthwise 的实际 kernel 路径见 [GPU 实测](GPU实战案例.md)。

## 本仓库 MedNeXt-L 的真实配置

`pumengyu/architectures/mednext.py` 的 `_MEDNEXT_L_KWARGS` 固定 `n_channels=32`、`kernel_size=3`、`norm_type='group'`、`do_res=True`、`do_res_up_down=True`、`block_counts=[3,4,8,8,8,8,8,4,3]`，各 stage 的 `exp_r=[3,4,8,8,8,8,8,4,3]`。这里的 “Large” 表示网络规模，**不表示使用 7×7×7 核**。卷积块来自 `nnunet_mednext`；本仓库 wrapper 将它接入 nnU-Net v2 并处理深监督输出。

ConvNeXt 的 2D 主块思想被 MedNeXt 扩展到 3D 医学分割，但两者并非简单地把每个 `7×7` 改为 `7×7×7`。核大小、归一化、block 数、上下采样和训练配方都要逐项核对。比较普通 nnU-Net、ConvNeXt 风格和 MedNeXt 的分割结果时，也不能把整网差异单独归因于“大核”或“depthwise”。

## 继续查阅

- 层级、skip、上/下采样、深监督：[MedNeXt 架构](MedNeXt架构.md)。
- nnU-Net 默认 PlainConvUNet：[网络架构笔记](nnUNet架构笔记.md)。
- 朴素 DWSep 与倒置瓶颈的区别及参数例子：[DWSep 与 IBConv](DWSep_vs_IBConv机制.md)。
- GPU 上真实卷积实现与 Tensor Core：[卷积实现](TensorCore与卷积实现.md)。
