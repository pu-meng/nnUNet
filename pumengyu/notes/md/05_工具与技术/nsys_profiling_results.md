# nsys Profiling 结果记录

---

## nsys 输出文件格式说明

运行 `nsys profile` 后会生成两个文件：

```
ib7.nsys-rep    ← 主报告文件（压缩二进制，nsys 专有格式）
ib7.sqlite      ← 从 .nsys-rep 展开的 SQLite 数据库（nsys stats 时自动生成）
```

### .nsys-rep

- **格式**：nsys 专有压缩二进制，内部包含所有 kernel 事件、时间戳、NVTX 标注等
- **作用**：主文件，用于 `nsys stats` 分析和 `nsys-ui` 图形界面打开
- **大小**：通常 10-50 MB（录 30 秒约 20 MB）
- **删除策略**：保留，是唯一的原始分析数据

### .sqlite

- **格式**：标准 SQLite 3 数据库
- **作用**：`nsys stats` 读取 .nsys-rep 后自动生成的中间缓存，Python 报告脚本（`gpukernsum.py` 等）直接查这个 DB
- **大小**：比 .nsys-rep 大很多（展开后 ~114 MB），因为 SQLite 不压缩
- **删除策略**：可以删，下次 `nsys stats` 会重新从 .nsys-rep 生成

```bash
# 重新生成 sqlite（如果删了）
nsys stats ~/nnUNet_workspace/profiling/ib7.nsys-rep --report gpukernsum
```

---

## IB7 Profiling 结果（2026-06-19）

### 实验配置

| 项目 | 内容 |
|------|------|
| Trainer | `nnUNetTrainer_MLAUNet_MoE_IB7_SizeOversampleV4` |
| Dataset | Dataset003_Liver，fold_0 |
| GPU | RTX 4090 |
| 录制命令 | `nsys profile --delay 120 --duration 30 --trace cuda,nvtx` |
| delay=120 原因 | 跳过 epoch 0 的 torch.compile 预热（约 344s），取稳定训练阶段数据 |
| 文件 | `~/nnUNet_workspace/profiling/ib7.nsys-rep`（20 MB） |

### Kernel 时间分布（Top 5）

| 占比 | Kernel | 路径 | 说明 |
|------|--------|------|------|
| 36.1% | `conv_depthwise3d_cuda_backward_input_kernel` | CUDA Core | DW 反向：input gradient |
| 17.3% | `conv_depthwise3d_cuda_backward_weight_kernel`（变体1） | CUDA Core | DW 反向：weight gradient |
| 13.8% | `conv_depthwise3d_cuda_backward_weight_kernel`（变体2） | CUDA Core | DW 反向：weight gradient |
| 12.6% | `conv_depthwise3d_cuda_kernel` | CUDA Core | DW 前向 |
| 4.3% | `conv_depthwise3d_cuda_backward_weight_kernel`（变体3） | CUDA Core | DW 反向：weight gradient |
| **~84%** | **所有 depthwise 合计** | **CUDA Core** | — |
| ~1.7% | `ampere_fp16_s16816gemm_*`（4 种变体） | **Tensor Core** | PW 1×1 卷积（expand/compress） |
| 1.6% | `nchwToNhwcKernel` | — | NCHW↔NHWC 格式转换 |
| 1.5% | `triton__0d1d2de` | Triton | 激活函数 / 归一化 |

### 结论

**IB7 比 baseline 慢 6× 的主因：DW k=7 无法使用 Tensor Core，退回 CUDA Core。**

```
标准 Conv k=3：W=[C_out, C_in×k³]，M=C_out=128 → Tensor Core ✓（~80% 时间在 ampere_fp16_gemm）
IBConv DW k=7： 每通道独立 W_c=[1, k³]，M=1      → Tensor Core 要求 M≥16，完全闲置 ✗
```

- PW expand/compress（C→4C→C）虽然是额外计算，但 Tensor Core 效率极高，只占 **1.7%**，不是瓶颈
- depthwise backward 有 **3 种变体** kernel，对应不同 spatial size 下 PyTorch autograd 选择不同 block 配置，但全部是 CUDA Core 路径
- 84% 时间在 CUDA Core，而 CUDA Core 的 FP16 算力只有 Tensor Core 的 1/8（4090：82 TFLOPS vs 660 TFLOPS），这是速度差距的硬件本质

### 后续方向

| 方向 | 预期效果 | 代价 |
|------|---------|------|
| 降低 DW kernel size（k=5） | 访存量 343→125（2.7×减少），速度提升 | 感受野缩小 |
| 去掉 DW，改用标准 Conv（k=3 Winograd） | 完全切回 Tensor Core，速度恢复 baseline | 感受野缩小 |
| Dilated Conv 替代大核 | 相同感受野，实际计算核更小 | 引入空洞伪影风险 |
| 降低 expansion ratio（4→2） | PW 计算减半，但 PW 只占 1.7%，效果微弱 | — |
