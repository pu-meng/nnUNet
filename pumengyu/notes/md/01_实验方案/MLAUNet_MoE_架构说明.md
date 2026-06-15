# MLAUNet-MoE 模型架构说明

> 对应代码：`pumengyu/architectures/mla_unetr.py`
> 对应 Trainer：`nnUNetTrainer_MLAUNet_MoE`

---

## 一、总体结构

```
输入 CT (B, 1, D, H, W)
        │
   ┌────▼────────────────────────────────────┐
   │  Encoder（nnUNet PlainConvUNet，纯卷积）  │
   │  5 个 stage，逐层下采样，特征图 32→320 ch │
   └────┬────────────────────────────────────┘
        │  skips[-1]: (B, 320, d, h, w)  ← 最深层特征
        │
   ┌────▼────────────────────────────────────┐
   │         MLA Bottleneck（新增）            │
   │  展平 → 2× MLATransformerBlock → 还原    │
   └────┬────────────────────────────────────┘
        │
   ┌────▼────────────────────────────────────┐
   │  Decoder（nnUNet，卷积 + skip concat）   │
   └────┬────────────────────────────────────┘
        │
   输出分割图 (B, 3, D, H, W)   [背景/肝脏/肿瘤]
```

**核心思路**：Encoder 和 Decoder 与原始 nnUNet 完全一致，只在**瓶颈层**插入 MLA Transformer，用最小改动引入全局感受野。

---

## 二、MLA Bottleneck（瓶颈层）

### 2.1 3D 特征图展平

瓶颈层特征图尺寸极小（约 `4×8×8`），展平为序列：

```
(B, 320, 4, 8, 8)  →  flatten  →  (B, 256, 320)  →  transpose  →  (B, 256, 320)
         C  D  H  W              N=D×H×W=256       序列长度N, 特征维C
```

然后经过 **2 个 MLATransformerBlock**，最后 LayerNorm 后还原回 3D。

### 2.2 MLATransformerBlock（Pre-LN）

每个 Block 结构：

```
x  →  LayerNorm  →  MLA（多头潜变量注意力）  →  残差加  →  x'
x' →  LayerNorm  →  MoE FFN               →  残差加  →  输出
```

超参数（当前配置）：

| 参数 | 值 |
|------|---|
| `d_model` | 320（nnUNet 瓶颈通道数） |
| `num_heads` | 8 |
| `num_blocks` | 2 |
| `compression_ratio` | 4 |
| `mlp_ratio` | 4 |

---

## 三、Multi-head Latent Attention（MLA）

### 3.1 动机

标准 MHA 的 K、V 投影各需 `d_model × d_model` 参数和 `N × d_model` 激活内存。MLA 借鉴 DeepSeek-V2，先把 x 压缩到低秩潜变量，再投影出 K、V，大幅降低参数量和显存。

### 3.2 计算流程

```
输入 x: (B, N, 320)

Q = W_Q(x)                          # (B, N, 320)  全维直接投影

c_kv = LayerNorm( W_DKV(x) )        # (B, N, 80)   低秩压缩  d_c = 320/4 = 80
K = W_UK(c_kv)                      # (B, N, 320)  上投影
V = W_UV(c_kv)                      # (B, N, 320)  上投影

attn = softmax( Q·Kᵀ / √d_head )   # (B, 8, N, N)  全局 full attention
out  = attn · V
输出 = W_O(out)                      # (B, N, 320)
```

### 3.3 参数量对比（d_model=320, h=8）

| 模块 | 参数量 |
|------|-------|
| 标准 MHA（W_Q, W_K, W_V, W_O） | 4 × 320² ≈ **410k** |
| MLA（W_Q, W_DKV, W_UK, W_UV, W_O） | 320²×2 + 320×80×3 ≈ **282k**（少 31%） |

---

## 四、MoE FFN（Mixture of Experts）

### 4.1 结构

借鉴 DeepSeek-V3 的 MoE 设计：**1 个 Shared Expert（始终激活）+ 4 个路由专家（每次激活 top-2）**。

```
输入 x_flat: (B×N, 320)

┌──────────────────────────────────────────────────────┐
│                    MoE FFN                           │
│                                                      │
│  Shared Expert ──────────────────────────────┐       │
│  (始终激活, d_ff=640)                          │       │
│                                               ▼      │
│  Router → top-2 scores → Expert 0 ─┐        加法    │
│                          Expert 1 ─┤                 │
│                          Expert 2 ─┤  gate加权求和   │
│                          Expert 3 ─┘                 │
└──────────────────────────────────────────────────────┘

每个 Expert：Linear(320→640) → GELU → Linear(640→320)
```

### 4.2 专家宽度设计

```
d_ff = d_model × mlp_ratio // 2 = 320 × 4 // 2 = 640  （半宽）

激活容量 = shared(1个全宽等价) + top_k×半宽
         = 1×640 + 2×640 = 3×640
         ≈ 原标准 FFN（1×1280）的 1.5 倍
```

### 4.3 Loss-free Bias 负载均衡

不用 auxiliary loss，而是用**自适应 bias** 调节路由选择：

```python
# 每个 train step 自动执行（不进优化器）
load_ema = 0.99 × load_ema + 0.01 × 本batch各专家负载
bias += 1e-3 × (目标均匀负载 - load_ema)

# 路由时：
routing_scores = router_scores + bias   # bias 只影响选谁
gate_weights   = softmax(router_scores) # gate 用原始 scores，梯度路径干净
```

过载专家 → bias 下降 → 被选概率下降；欠载专家 → bias 上升 → 被选概率上升。

---

## 五、整体参数规模（估算）

| 模块 | 参数量（估算） |
|------|-------------|
| Encoder（nnUNet 5 stage） | ~9M |
| MLA Bottleneck（2 block） | ~3M |
| └─ MLA × 2 | ~1.1M |
| └─ MoE FFN × 2（1 shared + 4 routed） | ~1.9M |
| Decoder（nnUNet） | ~5M |
| **总计** | **~17M** |

---

## 六、与其他变体的关系

```
nnUNetTrainer_Baseline
    └── 纯 nnUNet PlainConvUNet，无 Transformer

nnUNetTrainer_MLAUNet
    └── PlainConvUNet + MLA Bottleneck（FFN 为标准 MLP）

nnUNetTrainer_MLAUNet_MoE              ← 当前架构
    └── PlainConvUNet + MLA Bottleneck（FFN 替换为 MoE-FFN）

nnUNetTrainer_MLAUNet_MoE_SizeOversampleV2/V4
    └── 同上架构 + 不同的训练数据过采样策略（架构不变）
```

---

## 七、关键设计选择说明

1. **只改瓶颈层**：瓶颈特征图最小（~256 tokens），full attention 计算量可接受（256² = 65k），同时全局感受野在此最有价值

2. **低秩 KV 压缩**：`d_c = 80 << d_model = 320`，KV cache 从 320 维压到 80 维，在瓶颈多层 block 之间显存占用可控

3. **MoE 替代 MLP**：增加模型容量同时控制推理时激活参数量（4 个专家只激活 2 个），避免单纯加宽 FFN 带来的显存/速度代价

4. **Loss-free 负载均衡**：瓶颈序列长度短（N≈256），传统 auxiliary balance loss 效果不稳定；bias 机制更平滑，无需调权重
