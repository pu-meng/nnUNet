# GPU 性能基础

按硬件概念查阅。实际 profiling 命令见 [NCU / nsys](NCU_nsys命令.md)，IB7 与 Flash Attention 案例见 [GPU 实战案例](GPU实战案例.md)。

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
