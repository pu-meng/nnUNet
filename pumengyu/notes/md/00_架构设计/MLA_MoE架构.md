# MLAUNetBot3D 架构详解

##### 一句话总结

**MLAUNetBot3D = nnUNet 的 Baseline（PlainConvUNet）+ 瓶颈处把 Conv 换成 MLA Transformer**

除了瓶颈这一处修改，**Encoder、Decoder、Skip Connection、Loss、Data Augmentation 全部与 Baseline 完全一致**。

---

##### 1. Baseline（PlainConvUNet）架构

###### 1.1 整体结构

```
Input (1, 128³)
    │
    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Encoder（PlainConvEncoder）                                        │
│  6 个 stage，每个 stage = StackedConvBlocks(Conv×2)                 │
│  stride=2 的 conv 做下采样（不是 pooling）                          │
│  每层输出同时作为 skip connection 存起来                             │
└─────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Bottleneck（最深层 Enc-5 的输出）                                  │
│  Baseline: 就是 Enc-5 的 Conv×2 输出，直接传给 Decoder              │
│  MLA:      在 Enc-5 和 Decoder 之间插入 MLABottleneck3D             │
└─────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Decoder（UNetDecoder）                                             │
│  5 个 stage，每个 stage = 上采样 + cat(skip) + Conv×2               │
│  上采样用转置卷积（stride=2）                                       │
└─────────────────────────────────────────────────────────────────────┘
    │
    ▼
Output (3, 128³)  ← 1×1×1 conv 映射到 3 类（背景/肝脏/肿瘤）
```

###### 1.2 Encoder 的 Conv Block 细节

每个 Encoder stage 内部是一个 `StackedConvBlocks`，包含 **2 个 `ConvDropoutNormReLU`**：

```
StackedConvBlocks(num_convs=2):
    ┌─────────────────────────────────────────────┐
    │ ConvDropoutNormReLU #1（stride=2 或 1）      │
    │   Conv3d(k=3×3×3, stride=s, padding=1)      │
    │   → InstanceNorm3d                           │
    │   → LeakyReLU(negative_slope=1e-2)           │
    ├─────────────────────────────────────────────┤
    │ ConvDropoutNormReLU #2（stride=1）            │
    │   Conv3d(k=3×3×3, stride=1, padding=1)      │
    │   → InstanceNorm3d                           │
    │   → LeakyReLU(negative_slope=1e-2)           │
    └─────────────────────────────────────────────┘
```

**关键点**：
- 所有 conv 都是 **3×3×3**，padding=1（保持空间尺寸）
- 下采样靠 **stride=2 的 conv** 实现，不是 pooling
- 激活函数是 **LeakyReLU**（不是 ReLU）
- 归一化是 **InstanceNorm3d**（不是 BatchNorm）
- 没有 Dropout（dropout_op=None）

###### 1.3 Decoder 的细节

```
UNetDecoder stage:
    ┌─────────────────────────────────────────────┐
    │ 转置卷积（上采样，stride=2）                  │
    │   ConvTranspose3d(k=2×2×2, stride=2)        │
    │   通道数减半（例如 320→160）                  │
    ├─────────────────────────────────────────────┤
    │ cat(skip)                                    │
    │   上采样结果 (160ch) + skip (320ch) → 480ch  │
    ├─────────────────────────────────────────────┤
    │ StackedConvBlocks(Conv×2)                    │
    │   Conv3d(k=3×3×3) × 2                        │
    │   480ch → 320ch（降回目标通道数）              │
    └─────────────────────────────────────────────┘
```

###### 1.4 各层参数表（Dataset003_Liver 的 nnUNetPlans）

| 层 | 输入通道 | 输出通道 | 分辨率 | stride | Conv 数 | 说明 |
|----|---------|---------|--------|--------|---------|------|
| Enc-0 | 1 | 32 | 128³ | 1 | 2 | 第一层 stride=1，不降分辨率 |
| Enc-1 | 32 | 64 | 64³ | 2 | 2 | |
| Enc-2 | 64 | 128 | 32³ | 2 | 2 | |
| Enc-3 | 128 | 256 | 16³ | 2 | 2 | |
| Enc-4 | 256 | 320 | 8³ | 2 | 2 | |
| **Enc-5** | 320 | **320** | **4³** | 2 | 2 | **最深层，瓶颈位置** |
| Dec-4 | 320+320=640 | 320 | 8³ | — | 2 | cat skip-4 后降回 320 |
| Dec-3 | 320+256=576 | 256 | 16³ | — | 2 | |
| Dec-2 | 256+128=384 | 128 | 32³ | — | 2 | |
| Dec-1 | 128+64=192 | 64 | 64³ | — | 2 | |
| Dec-0 | 64+32=96 | 32 | 128³ | — | 2 | |
| Output | 32 | 3 | 128³ | — | 1×1×1 conv | 映射到 3 类 |

---

##### 2. MLA 的修改：瓶颈替换

###### 2.1 修改位置

代码 `mla_unetr.py:227-229`：

```python
def forward(self, x: torch.Tensor):
    skips = self.encoder(x)          # ← 完全复用 Baseline 的 Encoder
    skips[-1] = self.mla_bot(skips[-1])  # ← 唯一修改：对最深层特征做 MLA
    return self.decoder(skips)        # ← 完全复用 Baseline 的 Decoder
```

###### 2.2 Baseline vs MLA 瓶颈对比

```
Baseline 瓶颈（Enc-5 输出）:
    (B, 320, 4, 4, 4)  ← 就是 Conv×2 的输出
         │
         ▼ 直接传给 Decoder（不做任何额外处理）

MLA 瓶颈（Enc-5 输出 → MLABottleneck3D）:
    (B, 320, 4, 4, 4)
         │
         ▼
    ┌─────────────────────────────────────────────┐
    │ 1. flatten: (B, 64, 320)                    │
    │    4×4×4 = 64 个 token，每个 320 维          │
    ├─────────────────────────────────────────────┤
    │ 2. MLATransformerBlock × 2（默认 2 层）      │
    │    ┌───────────────────────────────────┐    │
    │    │ LayerNorm                          │    │
    │    │ MultiHeadLatentAttention(MLA)      │    │
    │    │   W_Q: 320→320 (Query)             │    │
    │    │   W_DKV: 320→80 (KV 压缩)          │    │
    │    │   W_UK: 80→320 (K 上投影)          │    │
    │    │   W_UV: 80→320 (V 上投影)          │    │
    │    │   Attention: 64×64 full attention  │    │
    │    │   W_O: 320→320 (输出投影)          │    │
    │    │ + residual                         │    │
    │    ├───────────────────────────────────┤    │
    │    │ LayerNorm                          │    │
    │    │ FFN: 320→1280→320 (MLP ratio=4)   │    │
    │    │   Linear → GELU → Dropout → Linear │    │
    │    │ + residual                         │    │
    │    └───────────────────────────────────┘    │
    ├─────────────────────────────────────────────┤
    │ 3. reshape: (B, 320, 4, 4, 4)              │
    └─────────────────────────────────────────────┘
         │
         ▼ 传给 Decoder（与 Baseline 完全相同的后续流程）
```

###### 2.3 MLA 的核心创新：低秩 KV 压缩

标准 Self-Attention 和 MLA 的对比：

```
标准 MHA（Multi-Head Attention）:
    Q = x·W_Q    (320→320)
    K = x·W_K    (320→320)  ← 每个 token 存 320 维
    V = x·W_V    (320→320)  ← 每个 token 存 320 维
    KV 存储: 2 × 64 × 320 = 40,960 个 float

MLA（Multi-head Latent Attention）:
    Q = x·W_Q    (320→320)
    c = x·W_DKV  (320→80)   ← 先压缩到 80 维潜变量（Down）
    K = c·W_UK   (80→320)   ← 从 c 还原出 Key（Up-K）
    V = c·W_UV   (80→320)   ← 从 c 还原出 Value（Up-V）
    KV 存储: 1 × 64 × 80 = 5,120 个 float（只有 c）
    K 和 V 在计算 attention 时即时展开，不存储
    KV 显存节省: 40,960 → 5,120 = 87.5% ↓
```

**三个投影矩阵各自的作用：**

- **W_DKV（Down KV，压缩）**：把每个 token 的 320 维表示压缩为 80 维潜变量 `c_kv`。`c_kv` 是 K 和 V 的**共享信息瓶颈**——K 和 V 虽然用途不同，但它们描述的是同一个 token 的内容，存在冗余，可以先提炼成低秩表示再分别解码。

- **W_UK（Up K，还原 Key）**：从 80 维 `c_kv` 解码出 320 维 Key。Key 的职责是**被 Query 打分**（dot-product 相似度），决定"哪些 token 值得关注"。

- **W_UV（Up V，还原 Value）**：从 80 维 `c_kv` 解码出 320 维 Value。Value 的职责是**提供实际内容**，attention 权重加权求和的是 V，最终输出是各 token Value 的加权组合。

W_UK 和 W_UV 从同一个 `c_kv` 出发但学习不同的解码方向：一个让 token 易于被比较，一个让 token 易于被聚合。

**关键 insight**：MLA 的 attention 矩阵仍然是 **64×64 的 full attention**（全局感受野），但 KV 的存储从 O(N·d) 降到 O(N·d_c)。在 4³=64 个 token 的场景下这个优势不明显，但这借鉴自 DeepSeek-V2——在 LLM 推理中 KV cache 是显存瓶颈，MLA 把百亿参数模型的 KV cache 节省 87.5%。

###### 2.4 参数量对比

| 组件 | Baseline | MLA | 差异 |
|------|----------|-----|------|
| Encoder | 同 | 同 | 完全相同 |
| 瓶颈 Conv×2 | 320→320, 3×3×3, 2 层 | **移除** | MLA 替换了这 2 层 conv |
| MLA 模块 | 无 | W_Q+W_DKV+W_UK+W_UV+W_O+FFN×2 | 新增 |
| Decoder | 同 | 同 | 完全相同 |
| **总参数量** | ~30M | ~31M（略多 ~3%） | 几乎不变 |

---

##### 3. 完整架构图

```
Input (1, 128³)
    │
    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ Encoder（PlainConvEncoder，与 Baseline 完全一致）                            │
│                                                                              │
│  Enc-0: Conv3d×2 (1→32,  k=3, s=1)   输出: 32ch, 128³  ──── skip-0 ────┐  │
│  Enc-1: Conv3d×2 (32→64,  k=3, s=2)  输出: 64ch,  64³  ──── skip-1 ────┤  │
│  Enc-2: Conv3d×2 (64→128, k=3, s=2)  输出: 128ch, 32³  ──── skip-2 ────┤  │
│  Enc-3: Conv3d×2 (128→256,k=3, s=2)  输出: 256ch, 16³  ──── skip-3 ────┤  │
│  Enc-4: Conv3d×2 (256→320,k=3, s=2)  输出: 320ch,  8³  ──── skip-4 ────┤  │
│  Enc-5: Conv3d×2 (320→320,k=3, s=2)  输出: 320ch,  4³  ──── skip-5 ────┤  │
└──────────────────────────────────────────────────────────────────────────────┘
    │  skip-5 (320ch, 4³)
    ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║  ★ MLA Bottleneck（唯一与 Baseline 不同的地方）                              ║
║                                                                             ║
║  输入: (B, 320, 4, 4, 4)                                                    ║
║      │                                                                      ║
║      ▼                                                                      ║
║  flatten → (B, 64, 320)    ← 4×4×4=64 个 token                             ║
║      │                                                                      ║
║      ▼                                                                      ║
║  ┌──────────────────────────────────────────────────────────────────────┐   ║
║  │ MLA Block #1                                                         │   ║
║  │   LayerNorm → MLA(8 heads, d_c=80) → residual +                      │   ║
║  │   LayerNorm → FFN(320→1280→320, GELU) → residual +                   │   ║
║  └──────────────────────────────────────────────────────────────────────┘   ║
║      │                                                                      ║
║      ▼                                                                      ║
║  ┌──────────────────────────────────────────────────────────────────────┐   ║
║  │ MLA Block #2                                                         │   ║
║  │   LayerNorm → MLA(8 heads, d_c=80) → residual +                      │   ║
║  │   LayerNorm → FFN(320→1280→320, GELU) → residual +                   │   ║
║  └──────────────────────────────────────────────────────────────────────┘   ║
║      │                                                                      ║
║      ▼                                                                      ║
║  LayerNorm → reshape → (B, 320, 4, 4, 4)                                   ║
╚══════════════════════════════════════════════════════════════════════════════╝
    │  (B, 320, 4, 4, 4)
    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ Decoder（UNetDecoder，与 Baseline 完全一致）                                 │
│                                                                              │
│  Dec-4: 上采样(s=2) → cat(skip-4:320ch) → Conv3d×2 (640→320) → 320ch, 8³  │
│  Dec-3: 上采样(s=2) → cat(skip-3:256ch) → Conv3d×2 (576→256) → 256ch,16³  │
│  Dec-2: 上采样(s=2) → cat(skip-2:128ch) → Conv3d×2 (384→128) → 128ch,32³  │
│  Dec-1: 上采样(s=2) → cat(skip-1:64ch)  → Conv3d×2 (192→64)  → 64ch, 64³  │
│  Dec-0: 上采样(s=2) → cat(skip-0:32ch)  → Conv3d×2 (96→32)   → 32ch,128³  │
│  Output: Conv1×1×1 (32→3)                                                   │
└──────────────────────────────────────────────────────────────────────────────┘
    │
    ▼
Output (3, 128³)  ← 背景(0) / 肝脏(1) / 肿瘤(2)
```

---

##### 4. Baseline vs MLA 对比总结

| 方面 | Baseline (PlainConvUNet) | MLAUNetBot3D |
|------|------------------------|--------------|
| **Encoder** | Conv3d×2 per stage | **完全相同** |
| **Decoder** | 上采样 + cat(skip) + Conv3d×2 | **完全相同** |
| **Skip Connection** | Encoder 每层输出 cat 到 Decoder | **完全相同** |
| **瓶颈（最深层）** | Conv3d×2 (局部 3×3×3) | **MLA Transformer ×2（全局 attention）** |
| **瓶颈感受野** | 局部（3×3×3=27 体素） | **全局（4³=64 个 token 互相看）** |
| **瓶颈参数量** | 2 × (320×320×27) ≈ 5.5M | MLA: ~1.2M + FFN: ~1.6M ≈ 2.8M |
| **Loss** | Dice + CE | **完全相同** |
| **Data Augmentation** | 旋转/缩放/镜像/噪声/亮度/对比度/Gamma | **完全相同** |
| **训练超参** | 1000 epoch, lr=1e-2, SGD | **完全相同** |
| **总参数量** | ~30M | ~31M（+3%） |

###### 4.1 修改量统计

```
Baseline 代码行数（PlainConvUNet + Encoder + Decoder + Conv blocks）: ~600 行
MLA 新增代码行数（MLAUNetBot3D + MLABottleneck3D + MLATransformerBlock + MultiHeadLatentAttention）: ~200 行

修改比例: ~200 / (600+200) ≈ 25% 的新代码
但修改位置: 只有 1 处（瓶颈）
```

###### 4.2 为什么只在瓶颈做 MLA？

1. **计算效率**：4³=64 个 token，attention 矩阵 64×64=4,096，非常小。如果在 128³ 做 attention，token 数=2,097,152，attention 矩阵 4 万亿，完全不可行
2. **语义层次**：Encoder 浅层（Enc-0~Enc-3）提取的是边缘/纹理等局部特征，不需要全局 attention。最深层（Enc-5）特征语义最丰富，全局上下文最有价值
3. **与 UMamba 对称**：UMambaBot3D 也是在瓶颈插入 Mamba 块，MLA 放在同一位置，方便公平对比

---

##### 5. MLA-MoE：MoE-FFN

###### 5.1 一句话总结

**MLA-MoE = 普通 Conv Encoder + MoE-FFN 瓶颈 + 普通 Conv Decoder（只动瓶颈一处）**

与 MedNeXt（全部替换）、SwinUNETR（替换 Encoder）不同，MLA-MoE 只在瓶颈的 `MLATransformerBlock` 里使用 MoEFFN，Encoder 和 Decoder 全部是普通 Conv block，与 Baseline 完全一致。

当前代码中 `MLATransformerBlock` 始终使用 `MoEFFN`（`mla_unetr.py:212`），不存在"标准 FFN"版本。`nnUNetTrainer_MLAUNet` 和 `nnUNetTrainer_MLAUNet_MoE` 实际上调用同一套网络架构，两者的区别仅停留在 trainer 注释层面（历史遗留）。

###### 5.2 MoE-FFN 结构

```
概念参考（标准 FFN，当前代码中未使用）:
    x → Linear(320→1280) → GELU → Linear(1280→320) → out

实际使用（MoE-FFN）:


    x (320)
     │
     ├──────────────────────────────────────────┐
     │                                          │
     ▼                                          ▼
  [Shared Expert]                       [Router]
  Linear(320 → 640) → GELU → Linear(640 → 320)    Linear(320 → 4)  ← 打分 4 个路由专家
  始终激活，无条件执行                      │
                                           ├── + expert_bias（loss-free 均衡偏置）
                                           ▼
                                      topk(scores, k=2)  → 选 top_2 专家
                                           │
                             ┌─────────────┴─────────────┐
                             ▼                           ▼
                      Expert_i (640d)             Expert_j (640d)
                    Linear(320→640)→GELU        Linear(320→640)→GELU
                    →Linear(640→320)            →Linear(640→320)
                             │                           │
                             └────── gate_weight ────────┘
                                  softmax(原始 scores，无 bias)
                                  加权求和 → routed_out (320)
     │
     ▼
    out = shared_out + routed_out  (320)
```

###### 5.3 关键设计细节

**专家宽度减半：**
- 标准 FFN：d_ff = 320 × 4 = **1280**
- 每个 MoE 专家：d_ff = 320 × 4 // 2 = **640**（半宽）
- 每次 forward 激活：shared(640) + top_2 �uted(640×2) = **1920** 激活宽度
- 相比标准 FFN（1280），激活容量约 1.5×，但总参数量更多（存储了 4 个路由专家）

**路由打分机制（极简）：**

`router = nn.Linear(320, 4, bias=False)` 本质是一个 `(4, 320)` 的权重矩阵，每行是某个专家的"代表向量"：

```
score_e = dot(x_token, W_router[e])   # token(320维) · 专家e代表向量(320维) = 标量

四个专家同时算 = x @ W_router.T       # (BN, 320) @ (320, 4) → (BN, 4) 一次矩阵乘
```

没有非线性、没有 bias，就是内积：**token 与哪个专家的代表向量最对齐，就选哪个专家**。

**两套 score：路由 vs 门控分离：**
```python
scores         = router(x)                        # 原始打分（参与梯度）
routing_scores = scores + expert_bias.detach()    # 路由选择用带 bias 的（无梯度）
topk_idx       = routing_scores.topk(top_k)      # 决定选哪些专家
gate_weights   = softmax(scores.gather(topk_idx)) # 门控权重用原始 score（保证梯度干净）
```
- `expert_bias` 是 `register_buffer`（**不是 Parameter**），不进优化器
- 路由选择 ≠ 门控权重：选谁用 bias 修正，但加权多少用原始 score，梯度路径不含 bias

**Loss-free 负载均衡（`update_expert_bias`）：**
```
每个 train step：
  1. 统计 topk_idx 中各专家被选中次数 → counts
  2. load = counts / total_tokens
  3. EMA 更新：expert_load_ema = 0.99 × ema + 0.01 × load
  4. bias += (target - ema) × 1e-3
     target = 1/num_experts = 0.25（均匀分配）
     过载专家 → bias 下降 → 下次不那么容易被选中
     欠载专家 → bias 上升 → 下次更容易被选中
```
不像 auxiliary load-balancing loss（会干扰主任务梯度），这里通过 buffer 调整路由偏置，主任务梯度完全干净。

###### 5.4 参数量对比（d_model=320，mlp_ratio=4）

| 组件 | 标准 FFN（MLA） | MoE-FFN（MLA-MoE） |
|------|----------------|-------------------|
| Shared Expert | 无 | 2 × 320 × 640 = 409,600 |
| 路由专家 × 4 | 无 | 4 × 2 × 320 × 640 = 1,638,400 |
| Router | 无 | 320 × 4 = 1,280 |
| 单 FFN | 2 × 320 × 1280 = 819,200 | 无 |
| **FFN 总参数（per block）** | **819,200** | **2,049,280（+150%）** |
| **每 token 激活参数** | **819,200** | **shared + top_2 = 3 × 409,600 ≈ 1,228,800** |
| **每 block 总参数（含 MLA）** | ~2.0M | ~3.2M |
| **瓶颈 2 blocks 总参数** | ~4.0M | ~6.4M |
| **模型总参数** | ~31M | ~33M |

###### 5.5 MLATransformerBlock 实际结构

```
MLATransformerBlock（mla_unetr.py:193，唯一版本）:

    x
    │
    ├─ LayerNorm → MLA → + ─── x'
    │
    ├─ LayerNorm → MoEFFN（shared + top_2 of 4 routed，各 320→640→320）→ + ─── out
```

`self.ffn = MoEFFN(...)` 写死在 `__init__` 里，无标准 FFN 分支（`mla_unetr.py:212`）。

###### 5.6 三模型对比总结

| 方面 | Baseline | MLA-UNet | **MLA-MoE** |
|------|----------|----------|-------------|
| 瓶颈处理 | Conv×2 | MLA+MoEFFN | **MLA+MoEFFN（同上）** |
| 全局感受野 | ✗ | ✓ | ✓ |
| 专家路由 | ✗ | ✓ | ✓ |
| 负载均衡 | ✗ | ✗ | **✓（loss-free bias）** |
| 瓶颈参数量 | ~5.5M | ~4.0M | **~6.4M** |
| 模型总参数 | ~30M | ~31M | **~33M** |
| Trainer | `nnUNetTrainer_Baseline` | `nnUNetTrainer_MLAUNet` | **`nnUNetTrainer_MLAUNet_MoE`** |

---

##### 6. 代码对应关系

| 组件 | 文件 | 类/函数 |
|------|------|---------|
| Baseline 整体 | `dynamic_network_architectures/architectures/unet.py` | `PlainConvUNet` |
| Encoder | `.../building_blocks/plain_conv_encoder.py` | `PlainConvEncoder` |
| Decoder | `.../building_blocks/unet_decoder.py` | `UNetDecoder` |
| Conv Block | `.../building_blocks/simple_conv_blocks.py` | `StackedConvBlocks` + `ConvDropoutNormReLU` |
| **MLA Attention** | `pumengyu/architectures/mla_unetr.py` | `MultiHeadLatentAttention` |
| **单个 FFN 专家** | `pumengyu/architectures/mla_unetr.py` | `FFNExpert` |
| **MoE-FFN** | `pumengyu/architectures/mla_unetr.py` | `MoEFFN` |
| **MLA Block（含 MoE）** | `pumengyu/architectures/mla_unetr.py` | `MLATransformerBlock` |
| **MLA 瓶颈封装** | `pumengyu/architectures/mla_unetr.py` | `MLABottleneck3D` |
| **MLAUNet 整体** | `pumengyu/architectures/mla_unetr.py` | `MLAUNetBot3D` |
| MLA Trainer | `pumengyu/trainers/trainer.py` | `nnUNetTrainer_MLAUNet` |
| **MLA-MoE Trainer** | `pumengyu/trainers/trainer.py` | `nnUNetTrainer_MLAUNet_MoE` |

---
