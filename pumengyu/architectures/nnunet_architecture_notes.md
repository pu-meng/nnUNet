# nnUNet 网络架构笔记（PlainConvUNet + UMambaBot3D）

> 源码位置：
> - `dynamic_network_architectures/architectures/unet.py` → `PlainConvUNet`
> - `dynamic_network_architectures/building_blocks/plain_conv_encoder.py` → `PlainConvEncoder`
> - `dynamic_network_architectures/building_blocks/unet_decoder.py` → `UNetDecoder`
> - `dynamic_network_architectures/building_blocks/simple_conv_blocks.py` → `StackedConvBlocks` / `ConvDropoutNormReLU`
> - `pumengyu/architectures/umamba.py` → `UMambaBot3D`

---

## 一、最小单元：ConvDropoutNormReLU

每一个卷积操作的最小单元，顺序固定：

```
Conv3d(in_ch → out_ch, kernel, stride, same padding)
  → [Dropout]          可选
  → InstanceNorm3d     nnUNet 默认用 InstanceNorm
  → LeakyReLU          nnUNet 默认激活
```

`nonlin_first=True` 时顺序变为 `Conv → ReLU → Norm`，nnUNet 默认是 `Conv → Norm → ReLU`。

---

## 二、卷积堆：StackedConvBlocks

每个 encoder/decoder stage 内部由若干个 `ConvDropoutNormReLU` 串联而成：

```
输入 (B, in_ch, D, H, W)
  → ConvDNR(in_ch → out_ch, stride=stride)   ← 第一个卷积负责下采样（或保持）
  → ConvDNR(out_ch → out_ch, stride=1)        ← 后续卷积 stride 都是 1
  → ConvDNR(out_ch → out_ch, stride=1)        ← 重复 n_conv_per_stage 次
输出 (B, out_ch, D', H', W')
```

nnUNet 3d_fullres 典型配置：每个 stage `n_conv_per_stage = 2`，即两个卷积。

---

## 三、编码器：PlainConvEncoder

N 个 stage 串行，每个 stage 就是一个 `StackedConvBlocks`。

**关键**：`return_skips=True` 时，每个 stage 的输出都被保存到列表并返回。

```
输入图像 (B, 1, D, H, W)
  │
  ├─ stage 0: StackedConvBlocks(1→32,   stride=1)    → skip[0]: (B, 32,  D,    H,    W   )
  ├─ stage 1: StackedConvBlocks(32→64,  stride=2)    → skip[1]: (B, 64,  D/2,  H/2,  W/2 )
  ├─ stage 2: StackedConvBlocks(64→128, stride=2)    → skip[2]: (B, 128, D/4,  H/4,  W/4 )
  ├─ stage 3: StackedConvBlocks(128→256,stride=2)    → skip[3]: (B, 256, D/8,  H/8,  W/8 )
  ├─ stage 4: StackedConvBlocks(256→320,stride=2)    → skip[4]: (B, 320, D/16, H/16, W/16)
  └─ stage 5: StackedConvBlocks(320→320,stride=2)    → skip[5]: (B, 320, D/32, H/32, W/32)  ← 瓶颈

返回：skips = [skip[0], skip[1], skip[2], skip[3], skip[4], skip[5]]
```

实际通道数和 stage 数由 nnUNet plans 自动计算，随任务不同而变（典型 3d_fullres 是 6 个 stage，通道数上限 320）。

---

## 四、解码器：UNetDecoder

从瓶颈开始，逐层上采样，每步 3 个操作：

```
① ConvTranspose3d（反卷积）：把低分辨率特征上采样到上一层分辨率
② concat：与同分辨率的 skip 在通道维拼接（通道数变为 2×）
③ StackedConvBlocks：卷积融合，通道压回 skip 通道数
[④ seg_layer：1×1 Conv 出分割图，deep_supervision 时每层都输出]
```

具体过程（以 6 stage 为例，从最深往上逐层恢复）：

```
lres = skip[5]  (B, 320, d, h, w)
  │
  ├─ step 0:
  │    TransConv(320→320, stride=2)          → (B, 320, 2d, 2h, 2w)
  │    cat(skip[4])                          → (B, 640, 2d, 2h, 2w)
  │    StackedConv(640→320)                  → (B, 320, 2d, 2h, 2w)
  │
  ├─ step 1:
  │    TransConv(320→256, stride=2)          → (B, 256, 4d, 4h, 4w)
  │    cat(skip[3])                          → (B, 512, 4d, 4h, 4w)
  │    StackedConv(512→256)                  → (B, 256, 4d, 4h, 4w)
  │
  ├─ step 2:
  │    TransConv(256→128, stride=2)          → (B, 128, 8d, 8h, 8w)
  │    cat(skip[2])                          → (B, 256, 8d, 8h, 8w)
  │    StackedConv(256→128)                  → (B, 128, 8d, 8h, 8w)
  │
  ├─ step 3:
  │    TransConv(128→64, stride=2)           → (B, 64, 16d, 16h, 16w)
  │    cat(skip[1])                          → (B, 128, 16d, 16h, 16w)
  │    StackedConv(128→64)                   → (B, 64, 16d, 16h, 16w)
  │
  └─ step 4（最后一步，必然输出分割图）:
       TransConv(64→32, stride=1)            → (B, 32, D, H, W)
       cat(skip[0])                          → (B, 64, D, H, W)
       StackedConv(64→32)                    → (B, 32, D, H, W)
       Conv1x1(32→num_classes)               → (B, num_classes, D, H, W)  ← 最终输出
```

`deep_supervision=True` 时，每个 step 都接一个 `Conv1x1` 输出分割图，返回列表（分辨率从高到低），用于多尺度监督。推理时只用第 0 个（最高分辨率）。

---

## 五、PlainConvUNet 完整 forward

```python
def forward(self, x):
    skips = self.encoder(x)    # 跑编码器，拿到 skip 列表
    return self.decoder(skips) # 跑解码器，消费 skip 列表
```

就这两行。架构的全部复杂度都在 encoder/decoder 内部。

---

## 六、UMambaBot3D：只改了一处

`UMambaBot3D` 继承 `PlainConvUNet`，只在 `__init__` 里额外注册了一个 `MambaBottleneck3D`，并覆盖 `forward`：

```python
def forward(self, x):
    skips     = self.encoder(x)
    skips[-1] = self.mamba_bot(skips[-1])  # ← 唯一改动：瓶颈过 Mamba
    return self.decoder(skips)
```

### MambaBottleneck3D 的作用

```
输入:  skips[-1] = (B, C_bot, d, h, w)     ← 瓶颈特征图，空间分辨率最小
  │
  ① permute + reshape → (B, d×h×w, C_bot) ← 把空间全展平成序列
  ② LayerNorm
  ③ Mamba SSM                              ← 序列建模，每个位置都能看到全局
  ④ + 残差（加回展平前的特征）
  ⑤ reshape + permute → (B, C_bot, d, h, w) ← 还原回三维

输出:  (B, C_bot, d, h, w)                ← 形状不变，但已注入全局上下文
```

### 为什么选瓶颈层

| 位置 | 空间尺寸 | 序列长度 L | Mamba 代价 |
|------|---------|-----------|-----------|
| stage 0（最浅） | D × H × W（原图） | 极大（百万级） | 不可接受 |
| stage 5（瓶颈） | ≈ 3×5×5（典型值） | 约 75 | 极小，可接受 |

瓶颈处序列最短，Mamba 开销最低；同时瓶颈是编码器语义最抽象的地方，全局上下文建模效益最高。

---

## 七、整体数据流总图

```
输入 CT patch (B, 1, D, H, W)
        │
   PlainConvEncoder
   ┌────┴──────────────────────────────────────────────────┐
   │  stage 0 ──────────────────────────── skip[0] (浅层)  │
   │  stage 1 ──────────────────────────── skip[1]         │
   │  stage 2 ──────────────────────────── skip[2]         │
   │  stage 3 ──────────────────────────── skip[3]         │
   │  stage 4 ──────────────────────────── skip[4]         │
   │  stage 5 ──────────────────────────── skip[5] (瓶颈)  │
   └───────────────────────────────────────────────────────┘
        │
   [UMamba only] MambaBottleneck3D 修改 skip[5]
        │
   UNetDecoder
   ┌────┴───────────────────────────────────────────────────┐
   │  step 0: TransConv + cat(skip[4]) + StackedConv        │
   │  step 1: TransConv + cat(skip[3]) + StackedConv        │
   │  step 2: TransConv + cat(skip[2]) + StackedConv        │
   │  step 3: TransConv + cat(skip[1]) + StackedConv        │
   │  step 4: TransConv + cat(skip[0]) + StackedConv + seg  │
   └────────────────────────────────────────────────────────┘
        │
   输出分割图 (B, num_classes, D, H, W)
   （deep_supervision=True 时返回多尺度列表，推理取第 0 个）
```