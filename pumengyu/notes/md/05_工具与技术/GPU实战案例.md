# 实战：naive vs Flash Attention

###### 测试脚本（`pumengyu/notes/ncu_test_pure.py`）

```python
import torch, torch.nn.functional as F

B, H, N, d = 2, 8, 2048, 64
Q = torch.randn(B, H, N, d, device='cuda', dtype=torch.float16)
K = torch.randn(B, H, N, d, device='cuda', dtype=torch.float16)
V = torch.randn(B, H, N, d, device='cuda', dtype=torch.float16)

def naive_attention(Q, K, V):
    S = torch.matmul(Q, K.transpose(-2, -1)) * (d ** -0.5)
    A = torch.softmax(S, dim=-1)
    return torch.matmul(A, V)

def flash_attention(Q, K, V):
    return F.scaled_dot_product_attention(Q, K, V)

for _ in range(5): naive_attention(Q, K, V); flash_attention(Q, K, V)
torch.cuda.synchronize()

out1 = naive_attention(Q, K, V); torch.cuda.synchronize()
out2 = flash_attention(Q, K, V); torch.cuda.synchronize()
print("done")
```

###### 速度对比

```
naive attention    0.902 ms/iter
flash attention    0.175 ms/iter
speedup: 5.15×     ← FLOPs 完全相同，差距纯粹来自 HBM 读写次数
```

###### NCU 实测数据（RTX 4090，真实输出）

**naive attention 的 4 个 kernel**：

| Kernel | 功能 | SM利用率 | 等HBM stall | 等计算 stall |
|--------|------|---------|------------|------------|
| `vectorized_elementwise_kernel` | 乘 1/√d | **6.5%** | **92%** | 0% |
| `cunn_SoftMaxForward` | softmax | 45% | **57%** | 0.6% |
| `ampere_fp16_s16816gemm` | QK^T 矩阵乘 | 21% | 34% | 34% |
| `ampere_fp16_s1688gemm` | AV 矩阵乘 | 30% | 15% | 17% |

**flash attention 的 1 个 kernel**：

| Kernel | 功能 | SM利用率 | 等HBM stall | 等计算 stall |
|--------|------|---------|------------|------------|
| `flash_fwd_kernel` | 全部合并 | 33% | **3.5%** | **64%** |

**解读**：

`vectorized_elementwise`（scale 乘法）：SM 才用 6.5%，92% 在等 HBM——把 [B,H,N,N] 从 HBM 读进来乘个标量再写回去，一个毫无价值的 kernel，纯粹是"没做 kernel fusion"的代价。

`cunn_SoftMaxForward`：57% 在等 HBM——softmax 要读整张 [B,H,2048,2048] 矩阵（~8MB），上一个 kernel 写到 HBM，这个 kernel 再读回来，来回搬两次。

`flash_fwd_kernel`：等 HBM stall 只有 3.5%，64% 在等 Tensor Core——中间结果全程留在 SRAM，GPU 真正在算。

**结论**：速度差 5× 不是算法不同，是 kernel 数量从 4 变成 1，HBM 来回次数从 O(N²) 降到 O(N)。这就是 **kernel fusion（算子融合）** 的价值。

---

##### Kernel 名字怎么读

CUDA kernel 名是 C++ 模板实例化的全名，很长但有规律：

```
void at::native::vectorized_elementwise_kernel<4, MulFunctor<float>, ...>
  at::native::                    ← PyTorch 命名空间
  vectorized_elementwise_kernel   ← 功能：逐元素操作
  <4, MulFunctor<float>>          ← 模板参数（忽略）

ampere_fp16_s1688gemm_fp16_128x128_ldg8_f2f_stages_32x1_tn
  ampere_      ← GPU 架构（Ampere = A100/3090/4090）
  fp16_        ← 数据类型
  s1688gemm    ← Tensor Core 矩阵乘（GEMM）
  128x128      ← tile 大小
  tn           ← transpose-normal（矩阵转置模式）

pytorch_flash::flash_fwd_kernel<Flash_fwd_kernel_traits<64,128,128,4,...>>
  pytorch_flash::    ← Flash Attention 库
  flash_fwd_kernel   ← 前向 kernel
  <64,128,128,4>     ← block_M, block_N, head_dim, num_warps
```

只需记功能名，模板参数是编译器填的。

---

##### 完整工具栈

```
层次          工具                    看什么
──────────────────────────────────────────────────────
实时监控      nvidia-smi              GPU利用率/显存/温度/功耗
              nvtop（需安装）         多卡实时监控（比nvidia-smi好看）
Python级      torch.profiler          算子级耗时，不需要root，最轻量
              torch.cuda.memory_summary()  显存分配明细
系统级        nsys                    时间线：哪个kernel慢，CPU/GPU配合
Kernel级      ncu                     硬件指标：为什么慢，stall在哪
多卡通信      nsys + --trace nccl     GPU间通信开销（NVLink/PCIe）
```

###### torch 内置工具（不需要 root，最快上手）

```python
#### 显存使用明细
print(torch.cuda.memory_summary(device=0))

#### 简单计时
start = torch.cuda.Event(enable_timing=True)
end   = torch.cuda.Event(enable_timing=True)
start.record()
#### ... 你的代码 ...
end.record()
torch.cuda.synchronize()
print(f"{start.elapsed_time(end):.2f} ms")
```

---

##### 多卡通信分析（双 4090）

你有两张 4090，训练时 GPU 之间通过 PCIe 通信（非 NVLink）。当瓶颈不在单卡计算而在卡间通信时，用 nsys 加 NCCL trace：

```bash
#### 同时 trace CUDA 和 NCCL 通信
CUDA_VISIBLE_DEVICES=0,1 nsys profile \
  --trace cuda,nvtx,nccl \
  --output /tmp/multi_gpu \
  python -m torch.distributed.launch --nproc_per_node=2 train.py

#### 看报告里的 nccl 通信占多少时间
nsys stats /tmp/multi_gpu.nsys-rep --report gpukernsum | head -20
```

输出里能看到 `ncclAllReduce`（梯度同步）占的时间——如果这个时间很长，说明通信是瓶颈，可以考虑梯度压缩或者减少同步频率。

---

##### CPU-GPU 通信结构与机制

###### 硬件拓扑：数据怎么在 CPU 和 GPU 之间流动

```
CPU (DRAM, ~100 GB/s)
  ↕ PCIe 总线
GPU 0 (HBM/GDDR6X, ~1-2 TB/s)
GPU 1 (HBM/GDDR6X, ~1-2 TB/s)
```

CPU 和 GPU 之间只有 PCIe 这一条路，是整个系统最窄的通道。

###### PCIe 带宽（你的 4090）

```
PCIe 版本    单向带宽     双向带宽
Gen 3 x16    16 GB/s      32 GB/s
Gen 4 x16    32 GB/s      64 GB/s   ← RTX 4090 使用 Gen 4
Gen 5 x16    64 GB/s     128 GB/s

对比：
  GPU 内部 HBM 带宽：1 TB/s（4090 GDDR6X）
  PCIe 4.0 x16：    32 GB/s

→ PCIe 带宽只有 GPU 内存带宽的 1/30
→ 任何需要 CPU↔GPU 来回搬数据的操作都极慢
```

###### 两种 Host 内存：Pageable vs Pinned

```
Pageable Memory（普通 malloc/numpy array）：
  CPU DRAM → 临时 Pinned Buffer → GPU
  需要 CPU 额外拷贝一次，实际速度更慢

Pinned Memory（page-locked，torch.Tensor.pin_memory()）：
  CPU DRAM → GPU
  DMA 引擎直接传输，不经过 CPU
  速度可提升 2×，但 Pinned 内存不能被 OS 换出（占用物理内存）

torch 的 DataLoader 默认 pin_memory=False
→ 训练时建议开 pin_memory=True（dataloader 里加）
```

###### H2D / D2H 传输

```
H2D（Host to Device）：CPU 内存 → GPU 显存（数据送进去）
D2H（Device to Host）：GPU 显存 → CPU 内存（结果取出来）

每次传输都要走 PCIe，延迟高（几微秒）+ 带宽限（32 GB/s）

nsys 里看到的 [CUDA memcpy H2D] / [CUDA memcpy D2H] 就是这个
如果这两项占了大量时间 → 说明 CPU↔GPU 之间数据搬运是瓶颈
  解法：pin_memory=True、异步传输、减少 D2H 次数
```

###### CUDA Stream：计算和传输重叠

不同 stream 的操作可以并行——计算的同时把下一批数据传进来：

```python
stream_compute = torch.cuda.Stream()
stream_transfer = torch.cuda.Stream()

#### 朴素版（串行，浪费等待时间）：
for batch in dataloader:
    data = batch.cuda()          # H2D（等待）
    output = model(data)         # 计算（等待上面完成）

#### 双 buffer 重叠版（并行）：
with torch.cuda.stream(stream_transfer):
    next_data = next_batch.cuda(non_blocking=True)  # H2D 异步

with torch.cuda.stream(stream_compute):
    output = model(current_data)  # 同时在算上一批

#### 效果：H2D 时间被计算时间掩盖，吞吐提升
```

`non_blocking=True` 是关键，告诉 CUDA 不要等 H2D 完成就返回。

###### GPU-GPU 通信：NVLink vs PCIe

```
消费级卡（你的 4090）：
  GPU0 ↔ PCIe Switch ↔ GPU1
  带宽：32 GB/s（PCIe 4.0 x16 单向）
  P2P 直连：如果两卡在同一 PCIe root complex 下可以跳过 CPU 内存

数据中心卡（A100/H100）：
  GPU0 ↔ NVLink ↔ GPU1
  A100 NVLink 3.0：600 GB/s 双向（≈ PCIe 的 18×）
  H100 NVLink 4.0：900 GB/s 双向

这就是为什么训练大模型必须用 A100/H100——
  两卡梯度同步（AllReduce）需要大量 GPU-GPU 传输
  4090 PCIe 互联做大模型训练会被通信卡死
```

###### NVLink / NVSwitch 拓扑（数据中心）

```
DGX A100（8 卡）：
  每张 A100 有 12 条 NVLink
  通过 NVSwitch 芯片全互联（any-to-any）
  任意两卡之间带宽 = 600 GB/s

DGX H100（8 卡）：
  NVSwitch 3.0
  任意两卡之间带宽 = 900 GB/s

NVSwitch 作用：
  没有 NVSwitch → GPU 只能和直连的邻居通信（ring 拓扑）
  有 NVSwitch → 任意两卡直接通信（full mesh），AllReduce 更快
```

###### NCCL 集合通信原语

分布式训练里 GPU 之间同步梯度用的操作：

```
AllReduce：所有 GPU 的梯度求和，每张卡得到完整梯度
  最常用，DDP 训练就是这个
  Ring AllReduce（2 卡）：
    GPU0 → GPU1（发一半数据）
    GPU1 → GPU0（发另一半数据）
    各自算完 → 再各发一次完整结果
    每步传输量 = N/2（N=梯度总大小）

AllGather：每张卡有一部分数据，收集后每张卡得到完整数据
  ZeRO-3 / LoRA 分片后用这个恢复

ReduceScatter：AllReduce 的前半步，每卡只得到一部分结果
  ZeRO-2/3 里梯度分片用这个

Broadcast：一张卡把数据发给所有卡
  模型参数初始化时用
```

###### 实际瓶颈判断

```
单卡训练：
  nsys 里 [CUDA memcpy H2D] 时间占比高 → DataLoader 是瓶颈
    → 开 pin_memory=True，增加 num_workers
  [CUDA memcpy H2D] 时间很少 → GPU 计算是瓶颈 → 用 ncu 看 kernel

双卡 DDP 训练：
  nsys + --trace nccl 看 ncclAllReduce 占比
  ncclAllReduce 时间 > 单步计算时间 → 通信是瓶颈
    → 梯度压缩（FP16 梯度）、增大 batch（减少同步次数）、overlap 通信和计算
  4090 双卡用 PCIe，AllReduce 带宽上限 32 GB/s
  GPT-2 175M 参数 FP32 梯度 = 700 MB，一次同步需要 700/32 ≈ 22ms
    → 这是 4090 双卡做大模型的硬瓶颈
```

###### P2P（Peer-to-Peer）直连

```python
#### 检查两卡是否支持 P2P（跳过 CPU 内存，直接 GPU↔GPU）
import torch
can_p2p = torch.cuda.can_device_access_peer(0, 1)
print(f"GPU0 → GPU1 P2P: {can_p2p}")

#### 启用 P2P
torch.cuda.device(0)
#### PyTorch DDP 自动启用 P2P（如果硬件支持）
```

4090 在同一 PCIe root complex 下通常支持 P2P，NCCL 会自动用。
可以用 `nvidia-smi topo -m` 查看两卡的拓扑关系。

---

##### DeepSeek 的工程循环

```
1. nsys 跑全程 → 找最耗时的 kernel 类型
2. ncu 深入 → 看 stall_long_scoreboard（等 HBM）多高
3. 分析原因：coalescing 差 / bank conflict / 没做 kernel fusion
4. 改 kernel → 重新 ncu 验证 → 循环

具体工作的底层对应：
  MLA（KV 压缩）  → 减少 HBM KV Cache 读写 → 降低 stall_long_scoreboard
  FP8 训练        → 同带宽搬更多信息（每次 HBM 事务信息量翻倍）
  算子融合        → 多个 kernel 合并 → 减少 HBM 中间结果落地次数
  Expert 并行     → 不同 expert 分到不同 GPU → 减少通信等待
```

**工具和 DeepSeek 完全一样**（ncu + nsys，NVIDIA 唯一官方工具），多的是经验和 CUDA 编程能力——看出问题后能自己写 kernel 修掉。工具是望远镜，CUDA 是手术刀，两个都要有。

架构论文写"提出新 attention 机制"，背后真实工作是"写了一个 CUDA kernel 让 stall_long_scoreboard 从 70% 降到 20%"。

---

##### 实战：IBConv(k=7) 为什么比 baseline 慢 6×

###### 背景

| Trainer | 架构 | 实测速度 |
|---------|------|---------|
| `MLAUNet_MoE_SizeOversampleV4`（baseline） | 标准 Conv3d(k=3) | ~48 s/epoch |
| `MLAUNet_MoE_IB7_SizeOversampleV4` | IBConv(k=7) | ~288 s/epoch（**6× 慢**）|

IBConv 结构（stride=1 处）：
```
DW Conv3d(C, k=7, groups=C, s=1)   ← 大核 depthwise，无 Tensor Core
PW Conv3d(C → 4C, 1×1×1)           ← pointwise expand，可用 Tensor Core
PW Conv3d(4C → C, 1×1×1)           ← pointwise compress，可用 Tensor Core
```

DW 用不上 Tensor Core 的根本原因（im2col 视角）：
```
标准 Conv：W=[C_out, C_in×k³]，M=C_out=128 → Tensor Core ✓
depthwise：每通道独立，W_c=[1, k³]，M=1     → Tensor Core 要求 M≥16 ✗

各通道无法合并进同一矩阵乘（不同通道用不同输入数据和权重）
→ 只能做 C 个独立的 [1,k³]×[k³,spatial] 矩阵乘
→ 每次 M=1，Tensor Core 完全闲置，退回 CUDA Core
```

慢的可能原因有三个，profiling 前不确定哪个是主因：
1. DW 用不上 Tensor Core，CUDA Core 算力弱
2. k=7 大核每个输出位置读 k³=343 个输入值，访存量是 k=3 的 12.7 倍
3. PW expand/compress（C→4C→C）是 baseline 完全没有的额外计算

###### 第一步：nsys 对比两个 Trainer 的 kernel 时间分布

```bash
#### 第一步：分别录制两个 Trainer（各自运行，不能同时）
#### baseline
CUDA_VISIBLE_DEVICES=0 nsys profile \
  --output ~/nnUNet_workspace/profiling/baseline \
  --trace cuda,nvtx \
  --delay 60 \
  --duration 30 \
  /home/PuMengYu/anaconda3/envs/medseg/bin/python \
  -m nnunetv2.run.run_training Dataset003_Liver 3d_fullres 0 \
  -tr nnUNetTrainer_MLAUNet_MoE_SizeOversampleV4

#### IB7
CUDA_VISIBLE_DEVICES=0 nsys profile \
  --output ~/nnUNet_workspace/profiling/ib7 \
  --trace cuda,nvtx \
  --delay 60 \
  --duration 30 \
  /home/PuMengYu/anaconda3/envs/medseg/bin/python \
  -m nnunetv2.run.run_training Dataset003_Liver 3d_fullres 0 \
  -tr nnUNetTrainer_MLAUNet_MoE_IB7_SizeOversampleV4

#### 第二步：分析（两个文件各自分析，对比结果）
nsys stats ~/nnUNet_workspace/profiling/baseline.nsys-rep --report gpukernsum | head -20
nsys stats ~/nnUNet_workspace/profiling/ib7.nsys-rep --report gpukernsum | head -20
```

**解读 kernel 名字**：

```
Tensor Core 路径（标准 conv，有 Winograd / GEMM 优化）：
  ampere_fp16_s1688gemm_*      ← FP16 Tensor Core GEMM
  ampere_fp16_s16816gemm_*     ← 同上，不同 tile 配置
  cudnn_winograd_nonfused_*    ← Winograd 路径（k=3, s=1 专用）

depthwise / CUDA Core 路径（无 Tensor Core）：
  cudnn_grouped_direct_*       ← depthwise 直接卷积，无优化
  implicit_convolveNd_*        ← implicit GEMM fallback，小矩阵退化
  vectorized_elementwise_*     ← 逐元素操作，纯 CUDA Core

结论判断：
  IB7 里 cudnn_grouped_direct 或 implicit_convolveNd 占大头时间
  → DW 部分是主因（Tensor Core 用不上 + 大核访存多）

  IB7 里 ampere_fp16_gemm（PW 层）反而是大头
  → expand/compress 新增 FLOPs 是主因
```

###### 实测结果：IB7 nsys kernel 时间分布（已完成）

用以下命令转换已有的 `.qdstrm` 文件（nsys importer 二进制在 `/usr/lib/nsight-systems/host-linux-x64/QdstrmImporter`，但 nsys 调用时找不到它，直接手动调用）：

```bash
/usr/lib/nsight-systems/host-linux-x64/QdstrmImporter \
  -i ~/nnUNet_workspace/profiling/ib7.qdstrm \
  -o ~/nnUNet_workspace/profiling/ib7.nsys-rep -f

nsys stats ~/nnUNet_workspace/profiling/ib7.nsys-rep --report gpukernsum | head -30
```

**实测 Top kernel（2026-06-19，Dataset003_Liver，fold_0）**：

| 占比 | kernel 名 | 类型 | 说明 |
|------|----------|------|------|
| **35.6%** | `conv_depthwise3d_cuda_backward_input_kernel` | CUDA Core | DW 反向：input gradient |
| **16.9%** | `conv_depthwise3d_cuda_backward_weight_kernel` | CUDA Core | DW 反向：weight gradient（变体1） |
| **13.7%** | `conv_depthwise3d_cuda_backward_weight_kernel` | CUDA Core | DW 反向：weight gradient（变体2） |
| **13.3%** | `conv_depthwise3d_cuda_kernel` | CUDA Core | DW 前向 |
| **4.1%** | `conv_depthwise3d_cuda_backward_weight_kernel` | CUDA Core | DW 反向：weight gradient（变体3） |
| 1.5% | `nchwToNhwcKernel` | — | 格式转换开销 |
| 1.5% | `triton__0d1d2de` | Triton | 激活/归一化 |
| 0.5%×3 | `ampere_fp16_s16816gemm_*` | **Tensor Core** | PW 1×1 卷积 |
| **合计 83.6%** | **所有 depthwise 相关** | **CUDA Core** | — |

**结论（确认）**：
- **83.6%** 的 GPU 时间花在 depthwise k=7 卷积（前向 + 反向）——全部是 CUDA Core 标量路径
- Tensor Core (`ampere_fp16_s16816gemm`) 总共只占 **~1.5%**，虽然 PW 1×1 能用 Tensor Core，但它不是时间主体
- 三个 depthwise 的原因都同时成立（CUDA Core 算力弱 + 大核访存多），但**主因是 DW 本身无法用 Tensor Core**
- PyTorch 对 depthwise conv 有 3 种 backward weight 变体（对应不同的 block size 配置），说明 autograd 在不同 spatial size 下用了不同 kernel，但全部是 CUDA Core 路径

###### 第二步：ncu 确认 DW kernel 的具体瓶颈

```bash
sudo ncu \
  --metrics smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,\
smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct,\
sm__throughput.avg.pct_of_peak_sustained_elapsed \
  --kernel-name "cudnn_grouped\|implicit_convolve" \
  --launch-skip 200 --launch-count 50 \
  --target-processes all \
  /home/PuMengYu/anaconda3/envs/medseg/bin/python \
  -m nnunetv2.run.run_training Dataset003_Liver 3d_fullres 0 \
  -tr nnUNetTrainer_MLAUNet_MoE_IB7_SizeOversampleV4
```

**读数解释**：

| 指标结果 | 含义 | 对应原因 |
|---------|------|---------|
| `stall_long_scoreboard` > 50% | 在等 HBM 数据 | k=7 大核访存量大，内存带宽瓶颈 |
| SM 利用率 < 30%，stall 不高 | CUDA Core 算力本就不足 | Tensor Core 用不上，纯算力差距 |
| `stall_math_pipe_throttle` 高 | 在等计算单元 | 计算密集，但 CUDA Core 比 Tensor Core 慢 |

###### 预期结论与后续方向

```
如果主因是 DW 访存（stall_long_scoreboard 高）：
  → k=7 的访存量是 k=3 的 12.7 倍，且 depthwise 无通道复用
  → 可尝试 k=5（访存减少 343→125，约 2.7×）
  → 或改用 dilated conv 替代大核（相同感受野，更小实际核）

如果主因是 PW expand/compress（nsys 里 gemm 占大头）：
  → expansion ratio 从 4 降到 2（参数量减半，但感受野不变）
  → 或去掉 expand/compress，改为纯 DW+PW 两层结构

如果两者都有（最可能）：
  → DW 部分的慢是硬件本质（depthwise 无 Tensor Core），难以绕开
  → PW 部分可以调 expansion ratio 来控制额外开销
  → 最终 tradeoff：感受野收益 vs 速度代价
```


---
