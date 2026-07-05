# 卷积的五种实现方式及对 Depthwise 的适用性

## 名词解释

**GEMM**（General Matrix Multiply，通用矩阵乘法）：标准矩阵运算 C = A×B + C，cuBLAS 和 Tensor Core 做的就是这个。"通用"是相对于专用（如 FFT、Winograd）而言，指不做任何数学变换、直接矩阵乘。

---

## 当前项目用的是哪种方法

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

## 方法一：Direct Conv（朴素直接卷积）

### 做什么

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

### 对 Depthwise 的适用性

**可以用**。depthwise 只需把 c_in 固定等于 c_out，删掉 c_in 那层循环：

```
for c in [0, C):
  for d',h',w' in output_positions:
    y[c,d',h',w'] = sum_{kd,kh,kw} W[c,kd,kh,kw] × x[c, d'+kd, h'+kh, w'+kw]
```

PyTorch 的 `conv_depthwise3d_cuda_kernel`（nsys 里占 83.6% 时间的那个）本质就是这种方式的优化版。

**不用 Tensor Core，用 CUDA Core，速度取决于访存效率。**

---

## 方法二：im2col + GEMM

### 做什么

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

### 对 Depthwise 的适用性

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

## 方法三：Winograd

### 做什么

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

### 对 Depthwise 的适用性

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

## 方法四：FFT（频域卷积）

### 做什么

卷积定理：**时域卷积 = 频域逐元素乘法**。

1. `FFT(x[c_in])` → 频域输入 $X_{c_{\text{in}}}[f_d, f_h, f_w]$
2. `FFT(W[c_out, c_in])` → 频域权重 $\hat{W}[c_{\text{out}}, c_{\text{in}}, f_d, f_h, f_w]$（预计算一次）
3. 频域相乘并对 $c_{\text{in}}$ 求和：$Y[c_{\text{out}}, f] = \sum_{c_{\text{in}}} X_{c_{\text{in}}}[f] \times \hat{W}[c_{\text{out}}, c_{\text{in}}, f]$
4. `IFFT(Y[c_out])` → 时域输出 $y[c_{\text{out}}, d', h', w']$

频域乘法的计算量：O(C_out × C_in × N)，和时域一样，但 FFT 步骤有 O(N log N) 的额外开销，大核时值得。

### 对 Depthwise 的适用性

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

## 方法五：Implicit GEMM

### 做什么

和 im2col + GEMM **数学完全等价**，但不真正创建 B 矩阵。在做矩阵乘时，需要 B[k, n] 时实时计算：

```
B[k, n] 对应输入 x[c_in, d'+kd, h'+kh, w'+kw]
→ 通过 k, n 反推 (c_in, kd, kh, kw, d', h', w')，直接从输入里读
→ 不需要额外分配 [K × N] 的矩阵（k=7 时这个矩阵 343 倍于输入大小）
```

cuDNN 对标准 conv 实际用的就是 Implicit GEMM，节省了 im2col 的显存开销。

### 对 Depthwise 的适用性

**仍然 M=1，Tensor Core 用不上**。Implicit GEMM 解决了 im2col 的内存问题，但 M=1 的根本问题没变。cuDNN 对 depthwise 直接用专用 Direct Conv kernel，跳过整个 GEMM 路线。

**Implicit GEMM 是 cuDNN 对标准 conv 的真实实现方式**——用 Tensor Core，但不创建显式 B 矩阵，内存开销等于原始输入大小。im2col 只作为概念工具帮助理解矩阵结构，实际生产代码早已绕开。

---

## 汇总对比

| 方法 | Depthwise 可用 | Tensor Core | k=7 效率 | 说明 |
|------|--------------|-------------|---------|------|
| Direct Conv | ✓ | ✗ | 中 | PyTorch 默认用这个 |
| im2col + GEMM | ✓（M=1） | ✗（M=1<16） | 差 | M=1 退回 CUDA Core |
| Winograd | 理论✓ | ✗ | ✗（不支持大核） | 仅 k=3 有效，cuDNN 未实现 depthwise 版 |
| FFT | ✓ | ✗ | **好** | 大核最佳选择，逐元素乘无矩阵结构 |
| Implicit GEMM | ✓（M=1） | ✗（M=1<16） | 差 | 省内存但 M=1 不变 |

**结论：depthwise k=7 如果要加速，最值得尝试的是 FFT 路线（大核天然适合频域），或者自定义 kernel 做 DW+PW 算子融合减少 HBM 读写。Tensor Core 在任何方法下都用不上 depthwise 部分。**
