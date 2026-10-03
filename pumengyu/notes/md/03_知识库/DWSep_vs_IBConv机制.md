# DWSep 与 IBConv：结构和效率边界

这篇只对照旧 MLAUNet 卷积替换方案。卷积实现和 Tensor Core 的详细原理见 [`TensorCore与卷积实现.md`](TensorCore与卷积实现.md)，2026-06-19 的 IB7 profiling 见 [`GPU实战案例.md`](GPU实战案例.md)。这里的速度记录是历史排查，不代表当前主模型的正式实验结论。

## 两种替换实际改了什么

| 位置 | 朴素 DWSep：`MLAUNetDWBot3D` | IBConv：`MLAUNetIBBot3D` |
|---|---|---|
| 替换对象 | Encoder、Decoder 中符合条件的 `Conv3d`，包括步幅为 2 的下采样 | Encoder、Decoder 中步幅为 1 的 `ConvDropoutNormReLU` |
| 空间混合 | `DW(k×k×k, groups=C, stride=s)` | `DW(k×k×k, groups=C, stride=1)` |
| 通道混合 | `PW(C_in→C_out, 1×1×1)` | `PW_expand(C→rC) → GELU → PW_compress(rC→C_out)` |
| 归一化与残差 | 保留外围原块行为；新 `DWSepConv3d` 本身没有残差 | `InstanceNorm3d`；输出加原输入或 1×1×1 投影 |
| 下采样 | 也可能被替换为大核 DWSep | 原有步幅为 2 的卷积保持不变 |

IBConv 的输入为 `(B, C_in, D, H, W)`。depthwise 卷积保持空间尺寸和通道数；1×1×1 扩展到 `r·min(C_in,C_out)`，再压缩到 `C_out`，最后加形状匹配的残差。这里的 `r` 默认为 4。实际逻辑见 `pumengyu/architectures/mla_unetr.py` 的 `IBConvBlock3D.__init__`、`_forward_impl()`、`forward()` 和 `_replace_cdnr_with_ib()`。

**注意名字和实际配置：** `nnUNetTrainer_MLAUNet_MoE_DW7_SizeOversampleV4` 保留了历史 “DW7” 名称，但当前类中的 `DW_KERNEL_SIZE=5`；`nnUNetTrainer_MLAUNet_MoE_IB7_SizeOversampleV4` 的 `IB_KERNEL_SIZE=7`。不能按 trainer 名把这两项写成相同卷积核大小的受控比较。

## 参数与实测边界

以 `C_in=C_out=128`、`k=7`、`r=4` 且忽略 bias/归一化参数为例：

| 单个卷积或块 | 卷积权重参数量 |
|---|---:|
| 标准 3×3×3 卷积 | `27·C² = 442,368` |
| 朴素 7×7×7 DWSep | `343·C + C² = 60,288` |
| 7×7×7 IBConv | `343·C + 2·r·C² = 174,976` |

这个参数对比只说明单个局部结构的成本，不是整网参数或速度结论。旧记录中的 baseline 约 48 s/epoch、IB7 约 288 s/epoch；[profiling 案例](GPU实战案例.md)显示 IB7 的 depthwise 前向与反向合计占当时 GPU kernel 时间约 84%。具体百分比应以原始 `ib7.nsys-rep` 重新导出。旧笔记还记过 7×7×7 步幅为 2 的 DWSep 约 514 s/epoch，但没有在本次整理中核对原始日志，不把它当作正式横向数据。

这些观察支持“在该环境下大核 depthwise 是主要耗时位置”，不能仅凭 kernel 名称把慢速全部归因于 Tensor Core 利用率或某一个卷积实现。要比较 DWSep 与 IBConv 的方法收益，还需固定核大小、参数预算、采样、训练周期和评估域。

## 代码与后续查阅

- 结构：`pumengyu/architectures/mla_unetr.py` 中 `DWSepConv3d`、`_replace_encoder_conv3d_with_dw_sep()`、`IBConvBlock3D`、`_replace_cdnr_with_ib()`。
- Trainer 配置：`pumengyu/trainers/trainer.py` 中两个 `...DW7_SizeOversampleV4` / `...IB7_SizeOversampleV4` 类。以类属性为准，不靠名称推断实际 kernel。
- GPU 实测与工具：[`GPU实战案例.md`](GPU实战案例.md)、[`NCU_nsys命令.md`](NCU_nsys命令.md)。
