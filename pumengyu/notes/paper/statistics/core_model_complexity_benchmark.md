# 核心模型复杂度与单 Patch 前向 Benchmark

> 测量时间：2026-07-31T15:27:21+08:00
> GPU：NVIDIA GeForce RTX 4090 D
> PyTorch/CUDA：2.3.0+cu121 / 12.1
> 输入：`1×1×128×128×128`；deep supervision=False；AMP=FP16；torch.compile=False。
> 按当前 Trainer 架构构建网络，不加载 checkpoint；权重数值不改变本表的参数量、执行图和当前全专家 MoE 路径。
> 延迟：warm-up 10 次后重复 30 次；数值为均值±标准差。
> FLOPs 由 PyTorch FlopCounterMode 统计其支持的 Conv/Linear/MatMul 运算；不等同于实际运行时间。
> 峰值显存仅为网络单 patch 前向的 CUDA allocated memory，不包含预处理、滑窗拼接、NIfTI I/O 或训练反向传播。

| 方法 | 参数量（排名） | FLOPs（排名） | 单Patch前向时间（排名） | 峰值显存（排名） |
|---|---:|---:|---:|---:|
| MedNeXt | 61.782M（#2） | 989.22G（#2） | 96.51±0.09 ms（#1） | 2.37 GiB（#1） |
| MedNeXt_MHA | 68.084M（#4） | 996.73G（#4） | 96.76±0.08 ms（#2） | 2.42 GiB（#4） |
| MedNeXt_MLA | 67.429M（#3） | 996.06G（#3） | 96.85±0.07 ms（#3） | 2.41 GiB（#3） |
| MedNeXt_MHA_MoE | 74.390M（#6） | 1003.18G（#6） | 97.18±0.08 ms（#4） | 2.45 GiB（#6） |
| MedNeXt_MLA_MoE | 73.735M（#5） | 1002.51G（#5） | 97.19±0.08 ms（#5） | 2.45 GiB（#5） |
| EfficientMedNeXt-L | 2.194M（#1） | 242.20G（#1） | 157.49±0.10 ms（#6） | 2.41 GiB（#2） |

## 解释边界

- 参数量是精确可训练参数总数；排名越小，参数越少。
- FLOPs 和前向时间排名越小，理论计算或当前环境中的单 patch 前向越低。
- 该时间不是完整 CT 病例的端到端推理时间；nnU-Net sliding-window 的窗口数随病例尺寸变化。
- 当前 MoE 实现会先计算全部 routed experts 再选择 Top-2，因此参数容量增加并未转化为官方稀疏 MoE 式的计算节省。
