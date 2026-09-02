# Tensor Core 与卷积实现

## 卷积的五种实现方式及对 Depthwise 的适用性


#### 卷积的五种实现方式及对 Depthwise 的适用性

##### 名词解释

**GEMM**（General Matrix Multiply，通用矩阵乘法）：标准矩阵运算 C = A×B + C，cuBLAS 和 Tensor Core 做的就是这个。"通用"是相对于专用（如 FFT、Winograd）而言，指不做任何数学变换、直接矩阵乘。

---

##### 当前项目用的是哪种方法

nsys profiling 的 kernel 名字直接说明了答案：

```
conv_depthwise3d_cuda_kernel                  ← 前向
conv_depthwise3d_cuda_backward_input_kernel   ← 反向（input gradient）
conv_depthwise3d_cuda_backward_weight_kernel  ← 反向（weight gradient）
```

这是 PyTorch 在 `aten/src/ATen/native/cuda/DepthwiseConv3d.cu` 里专门写的 depthwise CUDA kernel，属于**方法一：Direct Conv**。

没有 im2col，没有 FFT，没有 Winograd，是优化过访存模式的嵌套循环，全程 CUDA Core。cuDNN 对 depthwise conv 也走同样路线，因为其他方法要么不支持 k=7（Winograd），要么 M=1 和 Direct Conv 一样慢（im2col/Implicit GEMM）。

---

符号约定：
```
输入：x[b, c, d, h, w]，形状 [B, C_in, D, H, W]
权重：W[c_out, c_in, kd, kh, kw]，形状 [C_out, C_in, k, k, k]
输出：y[b, c_out, d', h', w']，形状 [B, C_out, D_out, H_out, W_out]
N   = D_out × H_out × W_out（输出空间位置总数）
K   = C_in × k³（一个输出通道需要的输入元素数）
```

---

##### 方法一：Direct Conv（朴素直接卷积）

###### 做什么

直接按定义算，六层循环：

```
for c_out in [0, C_out):
  for d' in [0, D_out):
    for h' in [0, H_out):
      for w' in [0, W_out):
        y[c_out, d', h', w'] = 0
        for c_in in [0, C_in):
          for kd,kh,kw in [0,k)³:
            y[c_out,d',h',w'] += W[c_out,c_in,kd,kh,kw]
                                 × x[c_in, d'+kd, h'+kh, w'+kw]
```

###### 对 Depthwise 的适用性

**可以用**。depthwise 只需把 c_in 固定等于 c_out，删掉 c_in 那层循环：

```
for c in [0, C):
  for d',h',w' in output_positions:
    y[c,d',h',w'] = sum_{kd,kh,kw} W[c,kd,kh,kw] × x[c, d'+kd, h'+kh, w'+kw]
```

PyTorch 的 `conv_depthwise3d_cuda_kernel`（nsys 里占 83.6% 时间的那个）本质就是这种方式的优化版。

**不用 Tensor Core，用 CUDA Core，速度取决于访存效率。**

---

##### 方法二：im2col + GEMM

###### 做什么

**im2col 为什么存在，有什么好处**：

```
不做 im2col（Direct Conv）：
  每个输出位置各自去输入里跳着读 k³ 个值
  访存不连续，cache miss 多，GPU 效率低

做 im2col（显式 B 矩阵）：
  先把所有需要的值整理成连续 B 矩阵，再做一次大矩阵乘
  访存全部连续，Tensor Core 满载（标准 conv M=128 时快 8×）
  代价：B 矩阵比原始输入大 k³ 倍（k=7 时 343 倍），纯内存冗余

但 Tensor Core 本身不需要 im2col！
  cuDNN 实际用的是 Implicit GEMM（见方法五）：
  不提前展开 B 矩阵，需要 B[k,n] 时现算坐标直接读原始输入
  → Tensor Core 照常用，内存只是原始输入大小，不是 k³ 倍
  im2col 只是帮助理解 GEMM 结构的概念工具，生产实现早就绕开了
```

**第一步 im2col**（概念上）：把输入重排成矩阵 B，形状 [K, N]：

```
起点：单个样本的输入 x，形状 [C_in, D, H, W]
      （batch 维度 B_batch 单独循环，每次处理一个样本）

符号说明：Ks = 卷积核边长（如 Ks=7），kd/kh/kw = 核内偏移下标，各自取值 [0, Ks)

B 是什么：一张 [K, N] 的二维表
  K 行 → 每行对应一个"读哪个输入值"的下标组合 (c_in, kd, kh, kw)
          共 C_in × Ks × Ks × Ks = C_in × Ks³ = K 种组合
          = 计算任意一个输出位置所需读取的输入元素数

  N 列 → 每列对应一个输出空间位置 (d', h', w')
          共 D_out × H_out × W_out = N 个位置

B 的每个元素：用 (c_in, kd, kh, kw) 定位行，用 (d', h', w') 定位列
  B[ (c_in, kd, kh, kw),  (d', h', w') ] = x[c_in, d'+kd, h'+kh, w'+kw]
  其中 kd, kh, kw ∈ [0, Ks)
  （实际存储时两组多维下标各自顺序压成一个整数行/列号）

为什么 B 不含 C_out：
  所有 C_out 个输出通道都从同一个输入 x 读数据，B 只和输入有关
  → B 构造一次，被 C_out 个不同的滤波器共用

直觉：B 的每一列 = 计算该输出位置时需要的全部 K 个输入值，连续摆放
      → GPU 读一列，连续内存，cache 命中率高

代价：相邻输出位置的感受野大量重叠，重叠部分被复制多次
     → B 的元素个数 = K×N = C_in×Ks³ × D_out×H_out×W_out
                    ≈ Ks³ 倍于原始输入大小（Ks=7 时约 343 倍），纯内存冗余
```

**第二步 GEMM**：构造权重矩阵 A，然后算 A × B：

卷积公式（回顾）：

$$
y[c_{\text{out}},\, d\',\, h\',\, w\'] = \sum_{c_{\text{in}},\, k_d,\, k_h,\, k_w} W[c_{\text{out}},\, c_{\text{in}},\, k_d,\, k_h,\, k_w] \times x[c_{\text{in}},\, d\'+k_d,\, h\'+k_h,\, w\'+k_w]
$$

```
右边两项：
  W[c_out, c_in, kd, kh, kw]      → 这是 A 的元素
  x[c_in, d'+kd, h'+kh, w'+kw]    → 这是 B 的元素

A 是什么：权重 W 按行展平，形状 [C_out, K]
  W 原始形状：[C_out, C_in, Ks, Ks, Ks]
  把后四维 (C_in, kd, kh, kw) 压成一个维度 K = C_in × Ks³
  → A[c_out, (c_in, kd, kh, kw)] = W[c_out, c_in, kd, kh, kw]
  每一行 = 一个输出通道对应的完整滤波器（展平）

卷积 → 矩阵乘（这正好是矩阵乘的定义）：

$$
\sum_{c_{\text{in}},\, k_d,\, k_h,\, k_w} A\bigl[c_{\text{out}},\,(c_{\text{in}},k_d,k_h,k_w)\bigr] \times B\bigl[(c_{\text{in}},k_d,k_h,k_w),\,(d\',h\',w\')\bigr] = (A \times B)\bigl[c_{\text{out}},\,(d\',h\',w\')\bigr]
$$

形状一览：
  A  [C_out, K]   ×   B  [K, N]   =   输出  [C_out, N]
  → reshape 成 [C_out, D_out, H_out, W_out]
```

###### 对 Depthwise 的适用性

**Tensor Core 用不上**。根本原因：depthwise 每个通道只读自己那一层输入，B 矩阵无法跨通道合并。

```
标准 conv：
  所有 C_out 个输出通道都读全部 C_in 层输入 → 共用同一个 B
  A = [C_out, K]，M = C_out = 128
  → 一次大 GEMM，M=128，Tensor Core 满载 ✓

Depthwise：
  通道 c 只读输入通道 c，通道 c+1 只读输入通道 c+1，B 各不相同
  → 没法合并，只能每个通道单独做一次小 GEMM：
      A_c = W[c, :]，形状 [1, Ks³]   ← A 只有 1 行，M = 1
      B_c = 通道 c 的输入展开，[Ks³, N]
      输出_c = A_c × B_c，[1, N]
  → M=1，Tensor Core 需要 M≥16（分块不够填满 warp），自动退回 CUDA Core ✗
```

---

##### 方法三：Winograd

###### 做什么

用代数变换减少乘法次数。以 k=3, stride=1 为例，F(2,3) 算法：

```
朴素算法：计算 2 个输出需要 2×3 = 6 次乘法
Winograd：把输入和权重变换到"Winograd域"，只需 4 次乘法，节省 1/3

变换步骤：
  1. 权重变换（训练后预计算一次）：G = 变换矩阵 × w
  2. 输入变换（每次 forward 都做）：D = 变换矩阵 × 输入 tile
  3. 点乘（这才是真正的乘法，只有 4 次）：M = G ⊙ D
  4. 反变换（全是加法）：y = 变换矩阵 × M
```

3D 的 Winograd 把 k³ 个乘法减少到更少（通过三个维度各自 Winograd 化）。

###### 对 Depthwise 的适用性

**理论上可以用**。depthwise 每个通道独立，对每个通道单独做 Winograd：

```
for c in [0, C):
  G_c = winograd_weight_transform(W[c])   ← 预计算
  D_c = winograd_input_transform(x[c])    ← 每次算
  M_c = G_c ⊙ D_c                        ← 只需 4 次乘法（k=3时）
  y[c] = winograd_output_transform(M_c)
```

**限制**：
- 只对 k=3（和少数小核）高效，k=7 没有实用的 Winograd 变换
- cuDNN **不支持**3D depthwise 的 Winograd（只支持标准 conv 的 Winograd）
- 不使用 Tensor Core，用 CUDA Core 做点乘和加法

---

##### 方法四：FFT（频域卷积）

###### 做什么

卷积定理：**时域卷积 = 频域逐元素乘法**。

1. `FFT(x[c_in])` → 频域输入 $X_{c_{\text{in}}}[f_d, f_h, f_w]$
2. `FFT(W[c_out, c_in])` → 频域权重 $\hat{W}[c_{\text{out}}, c_{\text{in}}, f_d, f_h, f_w]$（预计算一次）
3. 频域相乘并对 $c_{\text{in}}$ 求和：$Y[c_{\text{out}}, f] = \sum_{c_{\text{in}}} X_{c_{\text{in}}}[f] \times \hat{W}[c_{\text{out}}, c_{\text{in}}, f]$
4. `IFFT(Y[c_out])` → 时域输出 $y[c_{\text{out}}, d', h', w']$

频域乘法的计算量：O(C_out × C_in × N)，和时域一样，但 FFT 步骤有 O(N log N) 的额外开销，大核时值得。

###### 对 Depthwise 的适用性

**可以用，且对大核（k=7）有优势**：

```
depthwise FFT：
  1. FFT(x[c])       → X_c[f]        每通道各自 FFT
  2. FFT(W[c])       → Ŵ_c[f]        权重 FFT（预计算）
  3. Y_c[f] = X_c[f] × Ŵ_c[f]       逐元素乘（无跨通道求和！）
  4. IFFT(Y_c)       → y[c]

计算量：C × (2×FFT_cost + N) 而非 C × k³ × N
k=7 时 k³=343，若 N 足够大 FFT 比 Direct Conv 快
```

**限制**：
- 不使用 Tensor Core（逐元素乘法，无矩阵结构）
- FFT 有边界效应，需要 padding 处理
- 小 k 时（k=3）FFT 开销比直接算大，不划算

---

##### 方法五：Implicit GEMM

###### 做什么

和 im2col + GEMM **数学完全等价**，但不真正创建 B 矩阵。在做矩阵乘时，需要 B[k, n] 时实时计算：

```
B[k, n] 对应输入 x[c_in, d'+kd, h'+kh, w'+kw]
→ 通过 k, n 反推 (c_in, kd, kh, kw, d', h', w')，直接从输入里读
→ 不需要额外分配 [K × N] 的矩阵（k=7 时这个矩阵 343 倍于输入大小）
```

cuDNN 对标准 conv 实际用的就是 Implicit GEMM，节省了 im2col 的显存开销。

###### 对 Depthwise 的适用性

**仍然 M=1，Tensor Core 用不上**。Implicit GEMM 解决了 im2col 的内存问题，但 M=1 的根本问题没变。cuDNN 对 depthwise 直接用专用 Direct Conv kernel，跳过整个 GEMM 路线。

**Implicit GEMM 是 cuDNN 对标准 conv 的真实实现方式**——用 Tensor Core，但不创建显式 B 矩阵，内存开销等于原始输入大小。im2col 只作为概念工具帮助理解矩阵结构，实际生产代码早已绕开。

---

##### 汇总对比

| 方法 | Depthwise 可用 | Tensor Core | k=7 效率 | 说明 |
|------|--------------|-------------|---------|------|
| Direct Conv | ✓ | ✗ | 中 | PyTorch 默认用这个 |
| im2col + GEMM | ✓（M=1） | ✗（M=1<16） | 差 | M=1 退回 CUDA Core |
| Winograd | 理论✓ | ✗ | ✗（不支持大核） | 仅 k=3 有效，cuDNN 未实现 depthwise 版 |
| FFT | ✓ | ✗ | **好** | 大核最佳选择，逐元素乘无矩阵结构 |
| Implicit GEMM | ✓（M=1） | ✗（M=1<16） | 差 | 省内存但 M=1 不变 |

**结论：depthwise k=7 如果要加速，最值得尝试的是 FFT 路线（大核天然适合频域），或者自定义 kernel 做 DW+PW 算子融合减少 HBM 读写。Tensor Core 在任何方法下都用不上 depthwise 部分。**


---

## Tensor Core 为什么快：硬件机制深度解析


#### Tensor Core 为什么快：硬件机制深度解析

---

##### 前置概念：时钟周期 / 晶体管 / 面积

###### 时钟周期（Clock Cycle）

GPU 内部有一个"节拍器"，每秒打固定次数的节拍，这叫时钟频率。RTX 4090 是 2.52 GHz，意思是每秒打 25.2 亿次节拍。**每打一次节拍 = 一个时钟周期。**

所有计算必须跟着节拍走，乘法器在节拍响的瞬间"开枪"，下一个节拍响时输出结果：

```
节拍：  ↓        ↓        ↓        ↓
计算：  开始乘   等待...  等待...  结果出来   ← 这个乘法花了 3 个周期（延迟=3）
```

"快"的意思就是：**同样的节拍数内完成更多工作，或者同样的工作用更少节拍完成。**

节拍器的真实硬件是**晶振（晶体振荡器，Crystal Oscillator）**，一块石英晶体。石英通电后以极精确的固定频率振动（像音叉），产生规律的高低电压波形：

```
电压
 高 ─┐  ┌─┐  ┌─┐  ┌─
     │  │ │  │ │  │
 低  └──┘ └──┘ └──┘
     ↑
     每次从低→高的边沿 = 一个时钟周期
```

GPU 里所有触发器（存中间结果的最小存储单元）都连在这根线上，**每次上升沿到来，所有单元同时"咔哒"一下**，推进一步计算。超频就是强行提高晶振频率，更多"咔哒"，但热量同步上升。

###### 晶体管（Transistor）

晶体管是芯片里最小的开关，只有两个状态：通（1）和断（0）。

```
控制端加电压 → 开关闭合 → 电流通过 → 输出 1
控制端无电压 → 开关断开 → 电流断开 → 输出 0
```

一个乘法器由几千个晶体管组合而成（通过逻辑门实现加减乘除）。RTX 4090 整块芯片上有 763 亿个晶体管。晶体管越多，能做的操作越多越复杂。

###### 面积为什么重要（以及芯片是 2D 还是 3D）

**晶体管层是 2D 的**，刻在硅片表面，基本一层。但芯片不只有晶体管：

```
截面（从上到下）：

金属连线层 14  ←── 顶层走线（电源/地）
金属连线层 13
  ...             ←── 共 10-15 层金属，每层是导线网络，负责布线
金属连线层 1  ←── 最底层走线（局部信号）
──────────────────────────── 绝缘层
晶体管层      ←── 所有晶体管都在这一层，2D 平铺，这里是计算发生的地方
──────────────────────────── 硅基底
```

金属层是**走线用的**，不是计算单元。"面积"限制的是晶体管数量，即计算能力。

真正的 3D 是**多个芯片垂直叠放**（如 HBM 显存），用硅通孔（TSV）穿孔连接，带宽极高：

```
HBM（RTX 4090 的显存）：
  DRAM die × 4 层  ─┐
                    ├── TSV 穿孔垂直连接 → 带宽 1 TB/s
  Logic die（控制器）┘
```

这是多个完整芯片叠放，不是在单芯片内做 3D 晶体管。

芯片总面积固定，你决定放什么进去：

```
多放 CUDA Core → 通用计算强，矩阵乘效率一般
多放 Tensor Core → 矩阵乘极快，但只能做矩阵乘
```

这是芯片设计的核心 tradeoff：**有限的 2D 面积，如何分配给不同功能的硬件。**

###### 三者连起来

```
做 4096 次乘法：

方案A（CUDA Core）：
  1 个乘法器（占少量面积）
  → 4096 个时钟周期，串行跑完

方案B（Tensor Core）：
  同样面积放 256 个乘法器 + 共享控制逻辑
  （共享一套寄存器和调度电路，比 256 个独立 CUDA Core 省面积）
  → 16 个时钟周期，并行跑完

本质：用面积（更多晶体管）换时间（更少时钟周期）
```

---

##### 先明确：Tensor Core 不减少总操作数

4×4 矩阵乘，不管用什么硬件，总操作量固定：
```
64 次乘法 + 48 次加法
```

Tensor Core 快，不是因为"少算了什么"，而是**用更多晶体管面积，换更少的时钟周期数**。

---

##### CUDA Core：串行的本质

CUDA Core 做矩阵乘，相当于一个 for 循环：

```python
result = 0
for k in range(K):
    result += A[i][k] * B[k][j]   # 每步依赖上一步的 result
```

每步都要等上一步完成（loop-carried dependency），无法并行，需要 K 个 cycle 算出一个输出元素。4×4 矩阵乘 = 16 个元素 × 4 steps = 64 cycles。

---

##### Tensor Core：两个并行来源

###### 并行来源 1：乘法之间无依赖

```
p0 = A[i][0] × B[0][j]
p1 = A[i][1] × B[1][j]
p2 = A[i][2] × B[2][j]
p3 = A[i][3] × B[3][j]
```

p0、p1、p2、p3 互相没有依赖关系，可以用 4 个乘法器同时计算，1 个 cycle 出来 4 个乘积。

###### 并行来源 2：加法树替代串行累加

串行加法（CUDA Core）：
```
step 1: result = p0 + p1          ← 必须等 step 0 完成
step 2: result = result + p2      ← 必须等 step 1 完成
step 3: result = result + p3      ← 必须等 step 2 完成
3 steps，强制串行
```

加法树（Tensor Core）：
```
step 1: s01 = p0 + p1             ← 同时
        s23 = p2 + p3             ← 两个加法器同时跑，互相独立
step 2: result = s01 + s23
2 steps = log₂(4)，不是线性的 4 步
```

K=4096 时，串行需要 4095 步，加法树只需要 log₂(4096) = 12 步。

---

##### 那为什么不直接堆更多 CUDA Core？

问题很好：既然 Tensor Core 只是"更多并行"，为什么不多造几个 CUDA Core 来并行？

原因：**CUDA Core 不是纯乘法器，每个背后有大量额外开销**。

```
一个 CUDA Core 的硬件成本：
  ALU（乘法器）              ← 真正做计算，占面积小
  指令解码器                  ← "这条指令是什么操作"
  寄存器堆（每线程 255 个）   ← 存所有中间变量，面积大
  Warp 调度逻辑               ← 管理 32 个线程的执行顺序
  操作数收集/分发电路         ← 把正确的数据送到正确的 ALU

Tensor Core 里的 16 个乘法器：
  16 个 ALU                  ← 乘法器数量 16×
  共用 1 个指令解码器         ← 省 15 套
  共用输入输出寄存器           ← 省 15 套寄存器堆
  共用 1 套控制逻辑            ← 省 15 套调度电路
```

结论：**16 个乘法器共享 1 套控制开销，单位面积内乘法器密度远高于 16 个独立 CUDA Core**。这是 Tensor Core 比堆 CUDA Core 更高效的原因。

矩阵乘是这种设计的完美场景——所有乘法操作完全相同、完全独立，控制逻辑极简单，适合共享。

---

##### 真实硬件数字（RTX 4090，Ada Lovelace）

```
CUDA Core（FP32）：
  核心数：16,384
  每核每周期：1 FMA = 2 FLOPs
  峰值：16,384 × 2 × 2.52 GHz = 82 TFLOPS

Tensor Core（FP16）：
  核心数：512
  每核每周期：内部 16×16 乘法器阵列 = 大量 FLOPs
  峰值：661 TFLOPS

比例：661 / 82 ≈ 8×
```

**同样的 GPU，同样的功耗，Tensor Core 做矩阵乘是 CUDA Core 的 8 倍快。**
差距来自：更高的乘法器密度 + 加法树的对数级并行。

---

##### Tensor Core 为什么只能做矩阵乘

Tensor Core 内部 256 个乘法器，每一个的**输入连线都是焊死的**：

```
乘法器 (0,0)：永远接收 A[0行第0个] 和 B[第0列0号]
乘法器 (0,1)：永远接收 A[0行第1个] 和 B[第1列0号]
乘法器 (1,0)：永远接收 A[1行第0个] 和 B[第0列1号]
...
```

没有任何控制信号可以改变这些连接。你无法告诉 Tensor Core"这次只做加法"或"只比较大小"，因为硬件根本没有这条路。就像算盘的珠子只能上下拨，结构决定了只能做加减——不是设计不好，是为了极致速度把不必要的灵活性全去掉了。

CUDA Core 有选择开关（能做加、乘、比较……），Tensor Core 把开关全删了，只保留一条路：**D = A × B + C**。

##### DW k=7 为什么不能用 Tensor Core

**关键问题不是"是不是矩阵乘"，而是"B 矩阵能不能共享"。**

**标准 Conv（C_out=128）**：

```
所有 128 个输出通道，都读"全部输入通道"的数据：

  输出通道 0：y₀ = 全部输入 · w₀  ──┐
  输出通道 1：y₁ = 全部输入 · w₁  ──┤← 输入（B矩阵）完全相同！
  输出通道 2：y₂ = 全部输入 · w₂  ──┤
  ...                               ──┘

  → B（输入展开矩阵）固定不变，只有 A（权重）每行不同
  → 拼成一次大矩阵乘：A=[128×K] × B=[K×spatial]，M=128
  → Tensor Core 满载 ✓
```

**Depthwise Conv（C=128 通道）**：

```
每个通道只读"自己那一个输入通道"的数据：

  输出通道 0：y₀ = 【输入通道0】· w₀  ← B₀ = 输入第0层切片
  输出通道 1：y₁ = 【输入通道1】· w₁  ← B₁ = 输入第1层切片（不同！）
  输出通道 2：y₂ = 【输入通道2】· w₂  ← B₂ = 输入第2层切片（不同！）

  B₀ ≠ B₁ ≠ B₂ ≠ ...，每一行对应不同的输入数据

  → 无法拼成一次 A × B（矩阵乘要求 B 固定）
  → 只能做 128 次独立的 [1×343] × [343×spatial]
  → 每次 M = 1
```

把 128 个通道"堆"成 [128×343] 的权重矩阵，但对应的输入不是同一个 B，而是 128 个不同的 B——这不是矩阵乘，Tensor Core 的硬件没有这条路。

##### A 是什么，B 是什么

```
A = 权重矩阵（卷积核展开）
B = 输入矩阵（im2col 展开，每列 = 一个输出位置需要读取的输入数据）
C = A × B = 输出特征图
```

Tensor Core 做的就是 `C = A × B + C_acc`，A 和 B 必须事先准备好放进寄存器。

##### [1×343] × [343×spatial] 为什么不能用 Tensor Core

depthwise 每个通道独立，一个通道的矩阵乘是：

```
A = [1 × 343]   ← 这个通道的卷积核（1个核，343个权重）
B = [343 × N]   ← 这个通道的输入展开（N个空间位置）
C = A × B       ← 输出 [1 × N]，只有1行
```

Tensor Core 的最小操作单位是 **16×16 瓦片**（NVIDIA 工程选择，16×16=256个乘法器刚好塞满一个SM）。输出只有1行，Tensor Core 必须强行补成16行才能开动：

```
实际想算的（1行）：
  ⎡w₀ w₁ ... w₃₄₂⎤  ×  B  =  ⎡结果第0行⎤  ← 有用

Tensor Core 实际做的（强行填满16行）：
  ⎡w₀ w₁ ... w₃₄₂⎤            ⎡结果第0行  ← 有用⎤
  ⎢0  0  ...  0  ⎥            ⎢结果第1行  ← 全0 ⎥
  ⎢0  0  ...  0  ⎥  ×  B  =  ⎢结果第2行  ← 全0 ⎥
  ⎢...            ⎥            ⎢...              ⎥
  ⎣0  0  ...  0  ⎦            ⎣结果第15行 ← 全0 ⎦

做了16行的计算，有用的只有第0行，其余15行算了白算
利用率 = 1/16 = 6.25%
```

cuDNN 算过这个账——补零填满再用 Tensor Core，比直接用 CUDA Core 老实算那 1 行还慢，所以自动退回 CUDA Core。

```
M=128（标准 Conv，C_out=128）：
  分 128/16 = 8 次，每次 16 行全满
  ████████████████
  ████████████████  × 8次
  利用率 100% ✓

M=1（depthwise，每通道独立）：
  ████████████████  ← 第 1 行有用
  ░░░░░░░░░░░░░░░░  ← 第 2-16 行全是 0
  ...
  利用率 1/16 = 6.25% ✗  → 退回 CUDA Core
```

**一句话：DW 每个通道输出只有1行（M=1），Tensor Core 最小单位是16行，15行在空转，还不如 CUDA Core。**

---

##### 一句话总结

```
Tensor Core 快，不是因为少算，是因为：
  1. 多个乘法器同时开火（并行度高）
  2. 加法树比串行累加少 log 级步骤
  3. 多个乘法器共享控制开销，单位面积乘法器密度高

代价：
  更多晶体管，更多功耗，只对矩阵乘有效
  M=1 的 depthwise conv 利用率 6.25%，完全浪费这套专用硬件
```

---

##### 补充：从晶体管到乘法器，复杂度如何叠加

###### 所有计算都是 0/1 的组合——答案是肯定的

就像所有正整数都能由 1+1 得到，数字电路里**所有计算最终都能由 NAND 门（与非门）组合实现**。NAND 门只有一个功能：两个输入都是 1 时输出 0，否则输出 1。它是数字世界的"原子"。

```
层次（从简单到复杂，每层由下层组合而来）：

晶体管（通/断两态）
  ↓ 两个晶体管组合
NAND 门（通用逻辑原子，任何逻辑都能由它搭出来）
  ↓ 几个 NAND 门组合
XOR 门、AND 门、OR 门
  ↓ 组合
半加器（Half Adder）：算 1 位加法，输出"和"和"进位"
  ↓ 两个半加器 + OR 门
全加器（Full Adder）：算 1 位加法，含上一位的进位输入
  ↓ 32 个全加器串联（或并联加速）
32 位加法器（Adder）：~100-200 个晶体管
  ↓ 多个加法器组合
32 位乘法器（Multiplier）：~10,000 个晶体管
  ↓ 更复杂的迭代逻辑
32 位除法器（Divider）：~50,000+ 个晶体管
```

所以是的，**就像 1+1 能搭出所有数，NAND 门能搭出所有计算**，复杂度来自组合的层数和数量。

###### 加法、乘法、除法，哪个更复杂

```
操作      硬件门数量（32位）    为什么
────────────────────────────────────────────────────────
加法      ~200 个门            直接进位传播，一趟搞定
乘法      ~10,000 个门         本质是 32 次加法叠加（移位相加）
除法      ~50,000+ 个门        需要迭代试商，类似手算长除法
                               或用 Newton-Raphson 迭代逼近
```

**加法最简单，乘法是加法的 50 倍复杂，除法是加法的 250 倍以上。**

这就是为什么：
- GPU 里乘法比加法贵（延迟更长，面积更大）
- 但现代 FMA（融合乘加）把乘法和加法合在一条流水线里，外部看"一样快"
- 除法在 GPU 里极慢（几十个 cycle），写 kernel 时要尽量避免 `a / b`，改成 `a * (1.0f/b)`（预计算倒数，变成乘法）

###### CPU 和 GPU 之间的连线（PCIe）

CPU 和 GPU 是两块独立的芯片，通过 **PCIe（Peripheral Component Interconnect Express）** 总线连接，物理上是主板上的金手指插槽：

```
CPU (DRAM ~100 GB/s)
  │
  PCIe 4.0 x16 总线  ← 单向 32 GB/s，是 GPU 内部 HBM 带宽的 1/30
  │
GPU (HBM ~1 TB/s)
```

PCIe 是串行差分信号：用两根线传一个 bit，一根传正电压、一根传负电压，接收端看两者之差判断 0/1。这样做抗干扰能力强（共模噪声在两根线上一样，相减后消掉）。

PCIe x16 = 16 对差分信号线并行，每对每秒传 16 Gbps（Gen 4），合计单向 32 GB/s。

**这条线是训练时最窄的通道**：GPU 算完的梯度如果要回 CPU、DataLoader 送数据进 GPU，都要过这里。nsys 里看到的 `[CUDA memcpy H2D]` 就是数据经过这条线的过程。


---
