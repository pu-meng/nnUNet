# GPU 性能分析：NCU + nsys 完整指南

> MFU 是单一数字，藏掉了所有细节。NCU 精确到每个 kernel 的硬件指标，nsys 看全局时间线。两者配合才是完整的 GPU 性能分析。

---

## 知识缺口与学习计划

**当前状态**：会 Python/PyTorch 训练，会用 NCU/nsys 看指标，硬件基础接近零。
**学习方式**：AI 问答 + 写入 .md 持续更新，不看教材和视频，用碎片时间推进。
**核心原则**：每个概念都要和真实项目或 NCU 实验连起来，不学脱离实践的纯理论。

---

### 阶段一：概念补齐（AI 问答，现在就可以做）

用零散时间问 AI，理解后写进对应 .md，不需要连续学习块。

| 知识点 | 状态 | 连接到实际的什么 |
|--------|------|----------------|
| Thread/Warp/SM 层次 | ✅ 本文件已覆盖 | NCU 里的 occupancy/stall |
| 内存层次（寄存器→L1→L2→HBM） | ✅ 本文件已覆盖 | stall_long_scoreboard 在哪层 |
| FP32/FP16/BF16/INT8/FP8 | ✅ 本文件已覆盖 | nnUNet 混合精度训练 |
| Roofline 模型 | ✅ 本文件已覆盖 | compute/memory bound 判断 |
| PCIe / NVLink / NCCL | ✅ 本文件已覆盖 | 双卡训练通信瓶颈 |
| Cache 局部性（时间/空间） | ⬜ 未覆盖 | coalesced access 为什么快 |
| Warp Divergence | ⬜ 未覆盖 | GPU 里 if/else 为什么代价高 |
| 流水线与 ILP | ⬜ 未覆盖 | stall_no_instruction 的成因 |
| Bank Conflict | ⬜ 未覆盖 | shared memory 操作为什么慢 |
| 量化原理（PTQ/QAT） | ⬜ 未覆盖 | FP8 训练、INT8 推理 |
| ZeRO / 显存优化 | ⬜ 未覆盖 | 大模型训练显存怎么省 |
| Tensor/Pipeline 并行 | ⬜ 未覆盖 | DeepSeek 多机训练策略 |

### 阶段二：C++ 基础（有空才推进，一个月后）

不需要教材，直接问 AI 要代码例子，自己改、自己跑。

```
目标：能看懂 CUDA C++ 代码，能改别人的 kernel
路线（每个都实际写代码）：
  指针与内存管理（malloc/free，栈vs堆）
  引用 vs 指针
  模板基础（理解 CUDA kernel 名字里的 <T, N>）
  类与构造函数（理解 CUDA 里的 struct）
方式：问 AI 要一个概念 + 一段 20 行以内的例子，改通为止
```

### 阶段三：CUDA kernel 实战（C++ 之后）

**这是唯一不能只靠 .md 的阶段，必须写代码。**

```
5 个必写的 kernel（从简到难）：
  1. 向量加法          ← CUDA 语法入门，理解 threadIdx/blockIdx
  2. 矩阵乘（naive）   ← 理解为什么慢（global memory 访问多）
  3. 矩阵乘（tiling）  ← 理解 shared memory，用 NCU 验证提升
  4. Reduction（求和） ← 理解 warp shuffle 和 __syncthreads
  5. Fused Softmax     ← 第一个真正的 kernel fusion

每个 kernel 写完 → NCU profile → 对比 stall 指标 → 理解优化效果
此时 NCU 数字才真正有意义（知道看什么，知道怎么改）
```

### 期末后有空再看的书单

不用按顺序，按当时的需要选。

| 书名 | 难度 | 读哪部分 | 解决什么问题 |
|------|------|---------|------------|
| 《深入理解计算机系统》(CS:APP) | ⭐⭐ | 第1-6章 | Cache 局部性、流水线、内存层次，所有硬件直觉的根基 |
| 《Programming Massively Parallel Processors》(Kirk & Hwu) | ⭐⭐⭐ | 第1-8章 | CUDA 编程模型、shared memory、tiling，写 kernel 的圣经 |
| CUDA C++ Programming Guide | ⭐⭐⭐ | 全部（当手册查） | NVIDIA 官方，最权威，免费，遇到问题直接搜 |
| 《计算机体系结构：量化研究方法》(Hennessy & Patterson) | ⭐⭐⭐⭐ | 第1-5章 | DRAM/内存控制器/并行体系结构，进阶硬件原理 |
| Flash Attention 论文（Dao et al. 2022） | ⭐⭐⭐ | 全文 | 把 IO-aware 优化想法读透，NCU 数据会完全对上 |
| DeepSeek-V2/V3 技术报告 | ⭐⭐ | 全文 | MLA/MoE/FP8 工程细节，直接对应你现在学的 |

---

### 现在可以立刻做的

```
遇到 NCU 输出里不懂的指标 → 直接问 AI → 写进本文件
遇到不懂的硬件概念（cache/pipeline/divergence）→ 直接问 AI → 写进本文件
用 nsys profile 一次 nnUNet 训练 → 看哪个 kernel 最慢 → 记录结果
```

---

## 目录

**1. 硬件基础**
- 1.1 [CPU vs GPU 设计哲学](#硬件基础cpu-vs-gpu-设计哲学)
- 1.2 [GPU 内部结构：Thread / Warp / SM](#gpu-内部结构thread--warp--sm)
- 1.3 [数值格式：FP32 / FP16 / BF16 / INT8](#数值格式fp32--fp16--bf16--int8)
- 1.4 [内存层次结构](#内存层次结构)
- 1.5 [为什么 FLOPs ≠ 运行时间（Roofline 模型）](#为什么-flops--运行时间)

**2. 性能指标体系**
- 2.1 [指标层次结构（MFU → Occupancy → Warp Stall → 访存模式）](#指标层次结构从粗到细)

**3. CPU-GPU / GPU-GPU 通信**
- 3.1 [CPU-GPU 通信结构与机制（PCIe / Pinned / Stream / H2D）](#cpu-gpu-通信结构与机制)
- 3.2 [GPU-GPU 通信：NVLink vs PCIe / NCCL 集合通信](#gpu-gpu-通信nvlink-vs-pcie)
- 3.3 [多卡通信分析（双 4090 实战）](#多卡通信分析双-4090)

**4. 工具安装与使用**
- 4.1 [完整工具栈（nvidia-smi / nvtop / torch.profiler / nsys / ncu）](#完整工具栈)
- 4.2 [安装](#安装)
- 4.3 [权限设置](#权限设置)
- 4.4 [NCU 使用（命令 / 指标速查 / 计算量 / 分配）](#ncu-使用)
- 4.5 [nsys 使用](#nsys-使用)
- 4.6 [Kernel 名字怎么读](#kernel-名字怎么读)

**5. 实战**
- 5.1 [直接 profile nnUNet 训练（不需要改代码）](#不需要改代码直接-profile-nnunet-训练)
- 5.2 [naive vs Flash Attention 真实 NCU 数据对比](#实战naive-vs-flash-attention)
- 5.3 [IBConv(k=7) 为什么比 baseline 慢 6×](#实战ibconvk7-为什么比-baseline-慢-6)

**6. 工程思维**
- 6.1 [DeepSeek 的工程循环](#deepseek-的工程循环)

---

## 硬件基础：CPU vs GPU 设计哲学

CPU 和 GPU 的设计目标完全相反，这是理解所有后续概念的根基。

```
CPU：为单线程延迟优化
  核心数：4-64 个大核
  晶体管用途：分支预测、乱序执行、大 L3 Cache（几十 MB）
  设计目标：一件事尽快做完（延迟低，单核性能强）
  适合：串行逻辑、操作系统、数据库、复杂控制流

GPU：为吞吐量优化
  核心数：4090 有 16,384 个 CUDA 核心
  每个核简单：没有分支预测，没有乱序执行，Cache 很小
  设计目标：同时做极多件事（吞吐高，单件事延迟反而高）
  适合：矩阵乘、卷积、大规模并行计算
```

**关键比喻**：
```
CPU = 少数几个博士，每个人能独立解决复杂问题
GPU = 数万个小学生，每人只会简单加减法，但同时算
      矩阵乘这种"大量独立简单计算"用小学生更快
```

GPU 靠"数量"而不是"质量"赢，所以它的核越多越好，每个核越简单反而越有利（省晶体管）。

---

## GPU 内部结构：Thread / Warp / SM

这四个层次是理解 NCU 所有指标的基础。

```
层次结构（从小到大）：

Thread（线程）
  最小执行单元，一个线程执行一段 CUDA 代码
  例：矩阵乘里，一个线程负责计算输出矩阵的一个元素

Warp（32 个 Thread）
  GPU 真正的调度单位，32 个线程强制绑在一起
  一条指令同时对 32 个线程执行（SIMT：Single Instruction Multiple Threads）
  → 这就是为什么 block size 必须是 32 的倍数，否则浪费

Block（若干 Warp）
  程序员定义的分组，同一 block 内的线程可以用 Shared Memory 通信
  一个 block 的所有 warp 在同一个 SM 上执行

Grid（若干 Block）
  整个 kernel 的所有 block 合称 Grid
  不同 block 分配到不同 SM 执行
```

**SM（Streaming Multiprocessor）**：GPU 里的"小 CPU"

```
4090 有 128 个 SM，每个 SM 包含：
  ├── 128 个 CUDA 核心（FP32 ALU）
  ├── 4 个 Tensor Core（专门做 FP16/BF16 矩阵乘）
  ├── 65536 个寄存器（所有活跃线程共享）
  ├── 128 KB L1 Cache / Shared Memory（可配置比例）
  └── Warp Scheduler（调度器，决定哪个 warp 下一步执行）

全卡合计：
  128 SM × 128 核 = 16,384 CUDA 核心
  128 SM × 4 TC  = 512 Tensor Core
```

**Warp 调度：GPU 怎么隐藏延迟**

```
问题：HBM 访问需要 600-800 cycle，Warp 发出访存后要等很久

CPU 的解法：乱序执行（找其他指令来填等待时间）
GPU 的解法：切换 Warp（等内存的 Warp 挂起，切到另一个 Warp 继续跑）

时间线：
  Warp A：发出访存 → 等待...（挂起）
  Warp B：                    运行...
  Warp C：                           运行...
  Warp A：                                  数据到了 → 继续运行

前提：SM 上要有足够多的活跃 Warp 才能切换
  → 这就是 Occupancy 低的问题：Warp 不够，等内存时无人可切，GPU 空转
```

**NCU 指标和硬件的对应**

```
你在 NCU 看到的               对应的硬件含义
─────────────────────────────────────────────────────
stall_long_scoreboard        Warp 在等 HBM 数据，没有其他 Warp 可切
stall_math_pipe_throttle     Warp 在等 Tensor Core 算完（算力瓶颈）
sm__warps_active             SM 上平均活跃 Warp 数（= Occupancy）
launch__registers_per_thread 每线程占多少寄存器（影响 Occupancy 上限）
sm__throughput               SM 整体忙碌程度
```

---

## 数值格式：FP32 / FP16 / BF16 / INT8

数值格式直接决定内存占用、带宽压力、计算速度。

```
格式      位数   指数位  尾数位   数值范围         精度
────────────────────────────────────────────────────────
FP32      32     8      23      ±3.4×10³⁸       约7位有效数字
FP16      16     5      10      ±65504          约3位有效数字
BF16      16     8      7       ±3.4×10³⁸       约2位有效数字
INT8      8      —      —       -128~127        整数，无小数
FP8       8      4或5   3或2    更小            极低精度
```

**关键区别：FP16 vs BF16**

```
FP16：尾数位多（10位），精度高，但数值范围小（容易溢出）
BF16：指数位多（8位），和 FP32 范围一样（不溢出），精度低但够用

训练用 BF16 比 FP16 更稳定（不需要 loss scaling）
→ DeepSeek、LLaMA 训练全用 BF16
→ nnUNet 默认 FP32，可以改 FP16/BF16 加速
```

**数值格式对性能的影响**

```
同一块 HBM，每次传输 128 字节（一条 cache line）：

FP32：传 128/4 = 32 个数
FP16：传 128/2 = 64 个数   ← 同带宽下传 2× 数据
INT8：传 128/1 = 128 个数  ← 同带宽下传 4× 数据

→ 量化（FP32→FP16/INT8）不只省显存，也直接减少 HBM 带宽压力

Tensor Core 的吞吐：
  FP32：19.5 TFLOPS（4090）
  FP16：82.6 TFLOPS（4090）  ← 4× 快
  INT8：165 TOPS（4090）     ← 8× 快
  → Tensor Core 专为低精度矩阵乘设计，FP16 比 FP32 快 4×
```

**FP8（最新，DeepSeek V3 用）**

```
FP8 有两种：E4M3（4位指数+3位尾数）和 E5M2（5位指数+2位尾数）
H100 原生支持 FP8 Tensor Core，A100 不支持

DeepSeek V3 训练用 FP8：
  显存减半（相比 BF16），带宽需求减半
  精度损失极小（经过细心的量化策略）
  这是 DeepSeek V3 能用更少计算资源训练的关键工程之一
```

---

## 为什么 FLOPs ≠ 运行时间

GPU 有两个瓶颈，实际速度取决于慢的那个：

```
计算瓶颈：乘加操作太多，Tensor Core 跑满
内存瓶颈：数据从 HBM 搬到片上 SRAM 太慢

RTX 4090 峰值：
  计算：330 TFLOPS (FP16 Tensor Core)
  内存带宽：1 TB/s

临界算术强度 = 1TB / 330T ≈ 3 字节/FLOP

  每个 FLOP 搬超过 3 字节 → 内存瓶颈（等数据，HBM 是瓶颈）
  每个 FLOP 搬不到 3 字节 → 计算瓶颈（等 Tensor Core）
```

这叫 **Roofline 模型**：横轴是算术强度（FLOPs/Byte），纵轴是实际吞吐，两条线的交点是临界值。

---

## 内存层次结构

```
寄存器（~0 cycle，每线程私有）
  ↓ 超出时 spill 到
L1 / Shared Memory（~20-30 cycle，192KB per SM，片上）
  ↓ miss
L2 Cache（~200 cycle，72MB on 4090，片上）
  ↓ miss
HBM / GDDR6X（~600-800 cycle，24GB，2TB/s on 4090）
  ↓
CPU 内存 / 磁盘
```

Flash Attention 的核心：把 [B,H,N,N] 中间注意力矩阵从"每步落到 HBM"变成"全程留在 L1/Shared Memory"。FLOPs 不变，HBM 读写次数从 O(N²) 降到 O(N)。

---

## 指标层次结构（从粗到细）

### 第一层：MFU

```
MFU = 实际 FLOPS / 硬件峰值 FLOPS

普通实现：20-30%（等内存）
顶级工程：50-60%（算子融合后）
```

只告诉你"没跑满"，不告诉你为什么。

### 第二层：SM / Warp Occupancy

```
Warp Occupancy = 平均活跃 warp 数 / 理论最大 warp 数

GPU 靠切换 warp 隐藏延迟：
  一个 warp 等 HBM → 切到另一个 warp 跑 → 隐藏了等待时间
  Occupancy 低 → 没有备用 warp → GPU 空转

压低 Occupancy 的因素：
  寄存器太多：每 SM 65536 个寄存器，每线程用 128 个 → 最多 512 线程 = 16 warp
  Shared Memory 太多：挤占其他 warp 的空间
```

### 第三层：Warp Stall 原因（直达硬件，最重要）

```
stall_long_scoreboard    等 HBM 数据      ← 内存瓶颈的直接体现，优化首选
stall_short_scoreboard   等 L1/L2 数据
stall_math_pipe_throttle 等 Tensor Core   ← compute bound，"好的瓶颈"
stall_sync               等 __syncthreads()
stall_no_instruction     指令发射跟不上（ILP 不足）
stall_mio_throttle       原子操作/非对齐访问堵塞
```

### 第四层：内存访问模式

```
Coalescing（合并访问）：
  一个 warp 32 个线程同时访问内存
  连续地址 → 合并成 1 次 128 字节事务（最优）
  随机地址 → 32 次独立事务（带宽浪费 32×）

Shared Memory Bank Conflict：
  Shared memory 分 32 个 bank
  同 warp 两个线程访问同一 bank → 串行化

Register Spilling：
  寄存器超过上限 → spill 到 L1（local memory）
  每次访问从 0 cycle 变成 ~30 cycle
```

---

## 安装

```bash
# NCU（Nsight Compute）：kernel 级分析
sudo apt install nsight-compute

# nsys（Nsight Systems）：系统级时间线
sudo apt install nsight-systems

# 验证
which ncu && ncu --version
which nsys && nsys --version
```

两个工具在不同的包里，分别安装，不会互相冲突，很干净。

---

## 权限设置

NCU 读取 GPU 硬件性能计数器需要 root 权限：

```bash
# 临时降低权限限制（重启后失效）
sudo sh -c 'echo 1 > /proc/sys/kernel/perf_event_paranoid'

# 更可靠：直接用 sudo 跑 ncu
sudo ncu ... /完整路径/python script.py
```

**为什么 sudo 下要写 Python 完整路径**：
sudo 切换到 root，root 的 `$PATH` 不含 conda 环境，找不到 `python`。

```bash
# 你的 conda 环境 Python 路径
/home/PuMengYu/anaconda3/envs/medseg/bin/python
```

nsys 不需要 root，直接用当前用户跑即可。

---

## NCU 使用

### 不需要改代码

NCU 可以直接 wrap 任何 Python 脚本，**不需要对被 profile 的代码做任何修改**。

唯一限制：**不要在被 profile 的脚本里同时用 `torch.profiler`**，两者都抢 CUPTI 接口，会冲突。

### 常用命令

```bash
# 基础：profile 所有 kernel，看 4 个关键指标
sudo ncu \
  --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
lts__t_hit_rate.pct,\
smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,\
smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct \
  --target-processes all \
  /home/PuMengYu/anaconda3/envs/medseg/bin/python your_script.py

# 查看实际计算量（FLOPs）
sudo ncu \
  --metrics sm__sass_thread_inst_executed_op_ffma_pred_on.sum,\
sm__ops_path_tensor_src_precision_fp16.sum \
  --target-processes all \
  /home/PuMengYu/anaconda3/envs/medseg/bin/python your_script.py

# 查看资源分配（寄存器/shared memory/occupancy）
sudo ncu \
  --metrics launch__registers_per_thread,\
launch__shared_mem_per_block_static,\
sm__warps_active.avg.pct_of_peak_sustained_active \
  --target-processes all \
  /home/PuMengYu/anaconda3/envs/medseg/bin/python your_script.py

# 只 profile 特定 kernel（按名字过滤）
sudo ncu --kernel-name "flash_fwd_kernel" \
  --target-processes all \
  /home/PuMengYu/anaconda3/envs/medseg/bin/python your_script.py

# 保存完整报告（含 Roofline 图，用 ncu-ui 打开）
sudo ncu --set full --export ~/nnUNet_workspace/profiling/ncu_result \
  --target-processes all \
  /home/PuMengYu/anaconda3/envs/medseg/bin/python your_script.py
```

### 关键指标速查

| 指标 | 含义 | 好/坏 |
|------|------|-------|
| `sm__throughput` | SM 利用率 | 越高越好 |
| `lts__t_hit_rate.pct` | L2 命中率 | 越高越好（4090 上 n/a） |
| `stall_long_scoreboard` | 等 HBM stall 占比 | **高 = 内存瓶颈**，优化目标 |
| `stall_math_pipe_throttle` | 等 Tensor Core stall 占比 | 高 = 在算，"好的瓶颈" |
| `op_ffma_pred_on.sum` | FP32 FFMA 指令数（×2=FLOPs） | 实际执行量 |
| `op_tensor_src_precision_fp16` | FP16 Tensor Core 操作 | 矩阵乘实际计算量 |
| `registers_per_thread` | 每线程寄存器数 | 影响 Occupancy |
| `warps_active.pct_of_peak` | Occupancy | 越高延迟隐藏越好 |

---

## nsys 使用

nsys 是**系统级时间线**，不需要 root，速度快，适合先看全局。

**`nsys` 的子命令结构**（`profile` 是子命令，不是参数）：

```
nsys  <子命令>   [选项]            <要运行的程序>
nsys  profile    --output ...      python train.py   ← 运行程序并录制，生成 .nsys-rep
nsys  stats      file.nsys-rep     --report ...      ← 读取文件，输出文字分析
nsys  export     file.nsys-rep     --type ...        ← 转换成其他格式
```

类比：`nsys` 是录像机，`profile` 是"开始录像"按钮，`stats` 是"播放并分析录像"。

**完整选项拆解**：

```bash
nsys        profile                        \  ← 子命令：录制模式
  --output  ~/nnUNet_workspace/profiling/x \  ← 输出文件路径（.nsys-rep 自动补全）
  --trace   cuda,nvtx                      \  ← 录哪些事件（CUDA kernel + 自定义标注）
  --delay   60                             \  ← 延迟60秒后才开始录（跳过预热）
  --duration 30                            \  ← 录30秒后停止并终止程序
  python your_script.py                       ← 被录制的程序
```

两步分开执行，不能同时运行：

```bash
# 第一步：录制（生成 .nsys-rep 文件，程序在 delay+duration 秒后被终止）
CUDA_VISIBLE_DEVICES=0 nsys profile \
  --output ~/nnUNet_workspace/profiling/timeline \
  --trace cuda,nvtx \
  python your_script.py

# 第二步：分析（程序终止后执行）
nsys stats ~/nnUNet_workspace/profiling/timeline.nsys-rep

# GUI 版（需要本地有 nsight-systems 图形界面）
nsys-ui ~/nnUNet_workspace/profiling/timeline.nsys-rep
```

nsys 输出里可以看到：
- 每个 kernel 的名字和执行时间
- kernel 之间的间隔（CPU 调度开销）
- H2D/D2H 数据传输时间
- 哪些地方 GPU 在空转等 CPU

**nsys 和 ncu 的分工**：
```
nsys → 先找"哪个 kernel 最慢"（时间线视角）
ncu  → 再深入"这个 kernel 为什么慢"（硬件指标视角）
```

---

## 不需要改代码：直接 profile nnUNet 训练

**完全不需要修改 trainer 代码**，直接 wrap 训练命令。

### 用 nsys 看训练时间线（轻量，先用这个）

两步分开执行：第一步录制生成 `.nsys-rep` 文件（训练程序跑完或被终止后结束），
第二步再用 `nsys stats` 读取文件分析，不需要同时运行。

```bash
# 第一步：录制（运行训练，录稳定运行阶段）
# --delay 60   跳过前60秒的 torch.compile 编译预热（第0轮344s全是编译噪声）
# --duration 30  预热后录30秒，约26个iteration，足够看kernel时间分布
CUDA_VISIBLE_DEVICES=1 nsys profile \
  --output ~/nnUNet_workspace/profiling/ib7 \
  --trace cuda,nvtx \
  --delay 60 \
  --duration 30 \
  nnUNetv2_train Dataset003_Liver 3d_fullres 0 \
  -tr nnUNetTrainer_MLAUNet_MoE_IB7_SizeOversampleV4
# ↑ 程序在 delay+duration=90秒后被 nsys 终止，文件自动生成

# 第二步：分析（程序终止后执行）
nsys stats ~/nnUNet_workspace/profiling/ib7.nsys-rep --report gpukernsum | head -30
```

### 用 ncu 深入分析特定 kernel（重量级，针对性用）

ncu 会把每个 kernel 重放 9 次，直接 profile 全程训练会极慢。用 `--launch-count` 只 profile 前 N 个 kernel：

```bash
# 只 profile 前 100 个 kernel launch（跳过前 500 个预热）
sudo ncu \
  --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,\
smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct \
  --launch-skip 500 \
  --launch-count 100 \
  --target-processes all \
  /home/PuMengYu/anaconda3/envs/medseg/bin/python \
  -m nnunetv2.run.run_training Dataset003_Liver 3d_fullres 0 \
  -tr nnUNetTrainer_Baseline

# 只看卷积相关 kernel（过滤名字）
sudo ncu \
  --kernel-name "ampere_fp16.*gemm\|cudnn.*conv" \
  --metrics smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,\
smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct \
  --target-processes all \
  /home/PuMengYu/anaconda3/envs/medseg/bin/python \
  -m nnunetv2.run.run_training Dataset003_Liver 3d_fullres 0 \
  -tr nnUNetTrainer_Baseline
```

### 推荐流程

```
1. nsys profile nnUNet 训练 30 秒
   → 看哪类 kernel 占时间最多（conv？BN？loss？）

2. ncu --kernel-name 针对最慢的 kernel 类型深入
   → 看 stall_long_scoreboard 高不高（内存瓶颈）
   → 看 stall_math 高不高（计算瓶颈）

3. 根据结果判断：
   stall_long_scoreboard 高 → 考虑算子融合/减少中间结果
   stall_math 高 → 已经是最优了，加速需要换更快的 kernel 实现
   SM 利用率很低（<20%）→ Occupancy 问题，看寄存器/shared memory 分配
```

---

## 实战：naive vs Flash Attention

### 测试脚本（`pumengyu/notes/ncu_test_pure.py`）

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

### 速度对比

```
naive attention    0.902 ms/iter
flash attention    0.175 ms/iter
speedup: 5.15×     ← FLOPs 完全相同，差距纯粹来自 HBM 读写次数
```

### NCU 实测数据（RTX 4090，真实输出）

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

## Kernel 名字怎么读

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

## 完整工具栈

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

### torch 内置工具（不需要 root，最快上手）

```python
# 显存使用明细
print(torch.cuda.memory_summary(device=0))

# 简单计时
start = torch.cuda.Event(enable_timing=True)
end   = torch.cuda.Event(enable_timing=True)
start.record()
# ... 你的代码 ...
end.record()
torch.cuda.synchronize()
print(f"{start.elapsed_time(end):.2f} ms")
```

---

## 多卡通信分析（双 4090）

你有两张 4090，训练时 GPU 之间通过 PCIe 通信（非 NVLink）。当瓶颈不在单卡计算而在卡间通信时，用 nsys 加 NCCL trace：

```bash
# 同时 trace CUDA 和 NCCL 通信
CUDA_VISIBLE_DEVICES=0,1 nsys profile \
  --trace cuda,nvtx,nccl \
  --output /tmp/multi_gpu \
  python -m torch.distributed.launch --nproc_per_node=2 train.py

# 看报告里的 nccl 通信占多少时间
nsys stats /tmp/multi_gpu.nsys-rep --report gpukernsum | head -20
```

输出里能看到 `ncclAllReduce`（梯度同步）占的时间——如果这个时间很长，说明通信是瓶颈，可以考虑梯度压缩或者减少同步频率。

---

## CPU-GPU 通信结构与机制

### 硬件拓扑：数据怎么在 CPU 和 GPU 之间流动

```
CPU (DRAM, ~100 GB/s)
  ↕ PCIe 总线
GPU 0 (HBM/GDDR6X, ~1-2 TB/s)
GPU 1 (HBM/GDDR6X, ~1-2 TB/s)
```

CPU 和 GPU 之间只有 PCIe 这一条路，是整个系统最窄的通道。

### PCIe 带宽（你的 4090）

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

### 两种 Host 内存：Pageable vs Pinned

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

### H2D / D2H 传输

```
H2D（Host to Device）：CPU 内存 → GPU 显存（数据送进去）
D2H（Device to Host）：GPU 显存 → CPU 内存（结果取出来）

每次传输都要走 PCIe，延迟高（几微秒）+ 带宽限（32 GB/s）

nsys 里看到的 [CUDA memcpy H2D] / [CUDA memcpy D2H] 就是这个
如果这两项占了大量时间 → 说明 CPU↔GPU 之间数据搬运是瓶颈
  解法：pin_memory=True、异步传输、减少 D2H 次数
```

### CUDA Stream：计算和传输重叠

不同 stream 的操作可以并行——计算的同时把下一批数据传进来：

```python
stream_compute = torch.cuda.Stream()
stream_transfer = torch.cuda.Stream()

# 朴素版（串行，浪费等待时间）：
for batch in dataloader:
    data = batch.cuda()          # H2D（等待）
    output = model(data)         # 计算（等待上面完成）

# 双 buffer 重叠版（并行）：
with torch.cuda.stream(stream_transfer):
    next_data = next_batch.cuda(non_blocking=True)  # H2D 异步

with torch.cuda.stream(stream_compute):
    output = model(current_data)  # 同时在算上一批

# 效果：H2D 时间被计算时间掩盖，吞吐提升
```

`non_blocking=True` 是关键，告诉 CUDA 不要等 H2D 完成就返回。

### GPU-GPU 通信：NVLink vs PCIe

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

### NVLink / NVSwitch 拓扑（数据中心）

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

### NCCL 集合通信原语

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

### 实际瓶颈判断

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

### P2P（Peer-to-Peer）直连

```python
# 检查两卡是否支持 P2P（跳过 CPU 内存，直接 GPU↔GPU）
import torch
can_p2p = torch.cuda.can_device_access_peer(0, 1)
print(f"GPU0 → GPU1 P2P: {can_p2p}")

# 启用 P2P
torch.cuda.device(0)
# PyTorch DDP 自动启用 P2P（如果硬件支持）
```

4090 在同一 PCIe root complex 下通常支持 P2P，NCCL 会自动用。
可以用 `nvidia-smi topo -m` 查看两卡的拓扑关系。

---

## DeepSeek 的工程循环

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

## 实战：IBConv(k=7) 为什么比 baseline 慢 6×

### 背景

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

### 第一步：nsys 对比两个 Trainer 的 kernel 时间分布

```bash
# 第一步：分别录制两个 Trainer（各自运行，不能同时）
# baseline
CUDA_VISIBLE_DEVICES=0 nsys profile \
  --output ~/nnUNet_workspace/profiling/baseline \
  --trace cuda,nvtx \
  --delay 60 \
  --duration 30 \
  /home/PuMengYu/anaconda3/envs/medseg/bin/python \
  -m nnunetv2.run.run_training Dataset003_Liver 3d_fullres 0 \
  -tr nnUNetTrainer_MLAUNet_MoE_SizeOversampleV4

# IB7
CUDA_VISIBLE_DEVICES=0 nsys profile \
  --output ~/nnUNet_workspace/profiling/ib7 \
  --trace cuda,nvtx \
  --delay 60 \
  --duration 30 \
  /home/PuMengYu/anaconda3/envs/medseg/bin/python \
  -m nnunetv2.run.run_training Dataset003_Liver 3d_fullres 0 \
  -tr nnUNetTrainer_MLAUNet_MoE_IB7_SizeOversampleV4

# 第二步：分析（两个文件各自分析，对比结果）
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

### 实测结果：IB7 nsys kernel 时间分布（已完成）

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

### 第二步：ncu 确认 DW kernel 的具体瓶颈

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

### 预期结论与后续方向

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
