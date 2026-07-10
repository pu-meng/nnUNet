# pipeline_nnUNet




#### 目录
[toc]

#### 医学图像分割项目搭建全流程设计速查表

> 参考来源：nnUNet v2 设计 + 本项目 (MSD_LiverTumorSeg) 实践  
> 适用：新建一个 3D CT/MRI 分割项目时，逐节核对

---

##### 第一节 数据准备

###### 1.1 数据集划分

| 子集 | 比例 |
|------|------|
| train | 60~70% |
| val | 10~15% |
| test | 15~20% |

- 用固定随机种子（`seed=42`）做一次，保存为 json，所有实验共用
- `val` 只用于选 best checkpoint，`test` 只在最终报告时用一次
- nnUNet 方案：5-fold 交叉验证，test 集独立。小数据集时必须用 k-fold

###### 1.2 数据检查（必做，避免后期踩坑）

- [ ] 检查 spacing 是否一致，记录 min/max/median spacing
- [ ] 检查 image 和 label 的 shape 是否对应
- [ ] 检查 label 值域：`{0,1,2,...,N}`，有没有意外的大值
- [ ] 可视化几个 case：确认 label 对齐 image，没有偏移
- [ ] 统计前景体素比例：liver/tumor 各占多少（前景稀疏时需要 oversampling）
- [ ] 记录无目标类别的 case 数量（如无肿瘤 case），决定是否需要 hard mining

###### 1.3 数据指纹（nnUNet 核心思路）

计算并记录以下统计量（**只统计前景区域内的体素**）：

| 统计量 | 说明 |
|--------|------|
| spacing | median, min, max（每个轴）|
| shape after crop to nonzero | median |
| CT 强度 | mean, std, percentile_0.5, percentile_99.5 |
| relative_size | crop 后体积 / 原始体积 |

这些数值驱动后续所有决策（见各节）。

---

##### 第二节 预处理流程

###### 2.1 裁剪（Crop）

- **目的**：去掉大量空气背景，减小体积，加速训练
- **方法**：找所有通道 nonzero 的 union bounding box，crop 到此范围
- **技巧**：保存 crop 的 bbox 坐标，推理后用于还原到原始坐标系

###### 2.2 重采样（Resample）

- 目标 spacing = 数据集 spacing 的第 50 百分位（中位数）
- 异向性判断：`max_spacing / min_spacing > 3.0` → 各向异性
  - 若异向：取第 10 百分位（更保守，避免插值模糊）

| 数据类型 | 插值方法 |
|----------|----------|
| 图像 | 三次样条（order=3）或线性（order=1）|
| 标签 | **最近邻（order=0）**，绝对不能用高阶（会产生中间值）|

实现：SimpleITK 或 `scipy.ndimage`

###### 2.3 归一化（Normalize）

**CT 图像：**
1. 先 clip 到 `[percentile_0.5, percentile_99.5]`（去除极值噪声）
2. 再 Z-score：`(x - mean_fg) / (std_fg + 1e-8)`
3. mean/std 来自整个训练集的前景体素

**MRI 图像：**
- Z-score per case（每个 case 单独算 mean/std）
- 若 `relative_size < 0.75`：只用前景 mask 内的像素算统计量

###### 2.4 离线预处理 vs 在线预处理

- **推荐**：预处理结果保存为 `.pt` 文件（`torch.save`）
- **格式**：`{"image": Tensor[C,D,H,W], "label": Tensor[1,D,H,W]}`
- **优点**：训练时直接加载，省去重复计算；支持 mmap 加速（`map_location="cpu", mmap=True`）

在线做的部分（每次 `getitem` 时随机做）：bbox jitter、随机 margin、patch crop、数据增强

---

##### 第三节 网络架构

###### 3.1 基础选择

- 3D 分割首选：**3D U-Net**（encoder-decoder 对称结构）
- nnUNet 使用：DynUNet（MONAI 实现，支持 deep supervision）
- 编码器：3~5 个下采样 stage，特征图翻倍（32→64→128→256→320）
- 解码器：对称上采样，skip connection（concatenate）

###### 3.2 关键组件选择

| 组件 | 选择 | 说明 |
|------|------|------|
| 归一化（3D）| Instance Normalization | 不依赖 batch size，适合 batch=2 |
| 归一化（2D）| BN 或 IN | — |
| 激活函数 | LeakyReLU（slope=0.01）| 不用 ReLU |
| 上采样 | ConvTranspose3d 或 trilinear | — |
| 下采样 | stride=2 的卷积 | 不用 MaxPool（nnUNet 设计）|

###### 3.3 Deep Supervision（必加，nnUNet 核心）

- 在 decoder 每个 stage 都输出一个预测头
- 训练时所有输出参与 loss 计算，权重递减：
  - `weights[i] = 1/2^i`（i=0 最浅层，权重最大=1）
  - 最深层权重设为 0（分辨率太低，信息无效）
  - 归一化：`weights / sum(weights)`
- 推理时只用最浅层（最终分辨率）的输出
- **效果**：缓解梯度消失，加速收敛

###### 3.4 输入通道数

| 场景 | in_channels |
|------|-------------|
| 单模态 CT | 1 |
| CT + 粗分割概率图（本项目）| 2 |
| 多模态 | 模态数 |

---

##### 第四节 Patch 采样策略

###### 4.1 Patch Size 选择

原则：尽可能大（更多上下文），但受限于显存。

nnUNet 自动计算规则：
1. 按 spacing 的倒数设置宽高比（等效物理分辨率一致）
2. 缩放至参考体积（3D：256³ 个体素）
3. 不超过数据集的 median shape
4. 对齐到 2ⁿ（满足 pooling 整除约束）

实践经验（3D，8~24GB 显存）：`~192×192×64` 或 `~128×128×128`（batch=2）

###### 4.2 前景过采样（Foreground Oversampling）— 必做

- **问题**：随机裁 patch → 大量背景 patch → 肿瘤出现频率极低
- **nnUNet 方案**：每个 batch 中 1/3（33%）的 patch 强制包含前景
- **实现**：先找前景体素坐标，随机选一个作为 patch 中心，裁剪时保证它在范围内
- **本项目**：用 `RandCropByPosNegLabel`（MONAI），`pos_neg_ratio=1` 控制比例

###### 4.3 Hard Case Mining（本项目新增，nnUNet 没有）

- **问题**：小肿瘤/无肿瘤 case 训练信号弱，容易被忽略
- **方案**：构建 `_indices` 列表，让困难 case 在列表中出现更多次：

| case 类型 | 重复倍数 |
|-----------|----------|
| 无肿瘤 case | × `no_tumor_repeat_scale`（如 ×2）|
| 小肿瘤 case | × `small_tumor_repeat_scale`（如 ×3）|
| 大肿瘤 case | 不变或轻微增加 |

> 注意：用 `itertools.cycle` + 固定步数会破坏这个比例，不要混用

###### 4.4 Repeats 参数的作用

- `repeats=N` → 每个 case 在 `_indices` 里出现 N 次
- 等价于：每个 epoch 把整个数据集过 N 遍（但随机 patch 不同）
- 选择原则：让 `steps_per_epoch ≈ 200~500`（太少波动大，太多 epoch 太慢）
- 跨实验对比：用 `--total_steps` 对齐总训练步数，而不是依赖 `repeats × epochs`

---

##### 第五节 数据增强（Training Only）

> 所有增强只在训练时做，验证/推理不做（除 TTA）

###### 5.1 空间增强（最重要）

**随机旋转 (Rotation)**
- 3D 各向同性：每个轴 ±30°
- 3D 各向异性（max/min spacing > 3）：只在 spacing 大的轴 ±30°，其他轴 ±0°
- 2D：±15°（patch 宽高比 < 1.5 时）或 ±180°（近似正方形时）
- 概率：0.2~0.3，或直接 always
- 实现：MONAI `RandRotate3D` 或 batchgenerators `SpatialTransform`

**随机缩放 (Scale)**
- 范围：[0.7, 1.4]（70%~140%）
- 概率：0.2~0.3
- 与旋转共同做（同一个 SpatialTransform 里）

**弹性形变 (Elastic Deformation)**
- nnUNet 默认关闭（p=0）
- 实际上对 CT 分割提升有限，计算慢，按需加

**随机翻转 (Flip/Mirror)**
- 概率：0.5（每个轴独立）
- 3D：(z, y, x) 三轴都可翻
- 效果极好，几乎零成本，**必加**
- 实现：MONAI `RandFlip`

###### 5.2 强度增强（次重要）

| 增强方法 | 概率 | 参数 |
|----------|------|------|
| 高斯噪声 | 0.1 | variance ∈ [0, 0.1]，加在归一化之后 |
| 高斯模糊 | 0.2 | sigma ∈ [0.5, 1.0]，per-channel p=0.5 |
| 亮度（乘法）| 0.15 | multiplier ∈ [0.75, 1.25] |
| 对比度 | 0.15 | range [0.75, 1.25]，clip 回原范围 |
| 模拟低分辨率 | 0.25 | downsample scale ∈ [0.5, 1.0] 再 upsample |
| Gamma 校正 | 0.1 (+0.3 inverted)| gamma ∈ [0.7, 1.5]，保留正负性 |

###### 5.3 增强顺序（重要）

1. 空间增强（旋转、缩放、翻转）— 在 patch crop 之后或之中
2. 高斯模糊
3. 亮度 / 对比度 / gamma
4. 高斯噪声（最后加，否则噪声会被模糊掉）

###### 5.4 增强的 Deep Supervision 标签处理

- 做了空间增强后，标签要做完全相同的空间变换
- 多尺度标签：对变换后的标签按照 decoder 各 stage 的 stride 做降采样
- 降采样方式：最近邻插值（order=0）

---

##### 第六节 Loss 函数

###### 6.1 nnUNet 标准 Loss：Dice + CrossEntropy（等权）

- **Dice Loss**：`MemoryEfficientSoftDiceLoss`，smooth=1e-5
  - `batch_dice=True`：在整个 batch 维度合并再算 Dice（对小目标更稳定）
  - `include_background=False`：不算背景类的 Dice
- **Cross Entropy**：标准多类别 CE，`include_background=True`
- **组合**：`total_loss = 0.5 * Dice + 0.5 * CE`

###### 6.2 二分类（本项目）

```python
DiceCELoss(include_background=False, to_onehot_y=True, softmax=True, batch=True)
```

- 标签格式：`[B, 1, D, H, W]`，值为 0/1 的整数
- 网络输出：`[B, 2, D, H, W]`（双通道 softmax）而不是 `[B,1,...]` sigmoid
- **原因**：softmax 更稳定，不会出现 NaN（本项目踩过 sigmoid NaN 的坑）

###### 6.3 小目标强化（本项目新增）

**Focal Tversky Loss：**

```
TI = TP / (TP + α*FP + β*FN)
```

- α=0.3, β=0.7（加大 FN 惩罚，减少漏检）
- gamma=0.75（focal 加权，hard sample 贡献更大）
- 组合：`DiceCE + FocalTversky`（各 0.5 权重）
- **使用场景**：肿瘤 Recall 低，漏检多时切换

###### 6.4 Deep Supervision 的 Loss 计算

```python
total = sum(w[i] * loss_fn(logits[:,i], y_downsampled[i]) for i if w > 0)
```

- 对 decoder 每个输出 scale 单独算 loss
- weights = [1, 0.5, 0.25, ...]，最深层=0，归一化后相加

---

##### 第七节 优化器与学习率

###### 7.1 nnUNet 标准配置

```python
optimizer = SGD(model.parameters(), lr=1e-2, momentum=0.99,
                weight_decay=3e-5, nesterov=True)
```

> 为什么用 SGD 不用 Adam：nnUNet 实验证明 SGD+大 momentum 在长训练（1000 epoch）下最终性能更好，Adam 收敛快但容易陷入局部最优

###### 7.2 学习率调度：PolyLR（对齐 nnUNet）

$$lr_t = lr_{init} \times \left(1 - \frac{t}{T_{max}}\right)^{0.9}$$

- t = 当前 epoch，T_max = 总 epoch 数
- 特点：前期衰减慢，后期加速衰减（幂次<1）

```python
scheduler = LambdaLR(optimizer, lr_lambda=lambda ep: (1 - ep/T_max)**0.9)
#### 每 epoch 结束调用 scheduler.step()
```

###### 7.3 T_max 对齐（本项目踩坑）

**问题**：改 repeats → steps_per_epoch 变化 → epochs 变化 → LR 曲线形状变

**修正**：T_max 锚定到 total_steps（步数），而不是 epoch 数：

```python
poly_T_steps = total_steps  # 固定
lr_lambda = lambda ep: max(0, 1 - ep * steps_per_epoch / poly_T_steps) ** 0.9
```

效果：不同 repeats 实验，同一 step 处 LR 完全一致，曲线可比

###### 7.4 Gradient Clipping

```python
clip_grad_norm_(model.parameters(), 1.0)  # 防 NaN
```

- nnUNet：不做（默认无 clip）
- 建议：loss 偶发 NaN 时开启，稳定后可去掉

---

##### 第八节 训练循环设计

###### 8.1 nnUNet 训练规模

- epochs = 1000
- iter/epoch = 250（固定，不随数据集大小变化）
- → 总计 **250,000** 次梯度更新

###### 8.2 本项目方案（数据集小时更合适）

- 由数据集大小决定 `steps/epoch = len(dataset) // batch_size`
- 用 `--total_steps` 控制总训练量（跨实验对齐）
- 本质等价，但保留了 hard mining 的完整性（nnUNet 固定步数会破坏）

###### 8.3 AMP（混合精度训练）— 强烈推荐

```python
scaler = torch.cuda.amp.GradScaler()
```

- 前向用 fp16，loss/梯度用 fp32
- 速度提升 ~1.5~2x，显存减少 ~40%
- 注意：loss 计算前 `logits.float()`（防止 fp16 overflow 导致 NaN）

###### 8.4 训练循环必备元素

- [ ] `optimizer.zero_grad(set_to_none=True)`（比 `zero_grad()` 快）
- [ ] NaN/Inf 检测：`torch.isnan(loss)` → skip + warn，不要 crash
- [ ] 进度打印：每 N 步打印 loss（N=10~50）
- [ ] Checkpoint：每 `val_every` 个 epoch 保存一次
- [ ] Best model：val Dice 最高时保存 `best.pth`
- [ ] 保存完整状态：model + optimizer + scheduler + epoch（支持 resume）
- [ ] 保存 `config.json`：记录所有超参，方便复现

###### 8.5 DataLoader 配置

**train：**
```python
DataLoader(train_ds, shuffle=True, num_workers=2~4,
           pin_memory=True, prefetch_factor=2,
           collate_fn=list_data_collate)
```

**val：**
```python
DataLoader(val_ds, shuffle=False, batch_size=1,
           num_workers=2)
```

---

##### 第九节 验证策略

###### 9.1 验证时用滑窗推理（Sliding Window Inference）

原因：验证时要评测整个 volume 的 Dice，patch 大小无法覆盖全图

```python
sliding_window_inference(inputs, roi_size=patch_size,
                         overlap=0.5, mode="gaussian",
                         sw_batch_size=1~4)
```

- `overlap=0.5`：nnUNet 默认
- `mode="gaussian"`：高斯加权，比 constant 更平滑，消除拼接伪影
- 实现：MONAI 的 `sliding_window_inference`

###### 9.2 验证指标

| 指标 | 说明 |
|------|------|
| **Dice Score** | 主指标，按类别分别计算；只统计 GT 有该类别的 case |
| NSD | Normalized Surface Dice，对边界质量敏感 |
| Recall / Precision | 分析漏检 vs 误检 |
| HD95 | Hausdorff 距离 95th 百分位，对 outlier 不敏感 |

> 注意：GT 为空的 case 不计入 Dice，避免预测全 0 时 Dice=1 虚高

###### 9.3 验证频率

- nnUNet：每 50 个 epoch 验证一次
- 本项目：`val_every=5~10`（数据集小，验证快，可以更频繁）
- 原则：验证太频繁浪费时间，太少错过 best checkpoint

---

##### 第十节 推理（Inference）流程

###### 10.1 推理 pipeline（必须与训练预处理完全一致）

1. Crop（保存 bbox）
2. Resample 到训练时的 target_spacing
3. Normalize（用**训练集**的 mean/std，不是测试集自身的）
4. 滑窗推理（同验证）
5. Argmax → 二值预测
6. 后处理（见第十一节）
7. Resample 回原始 spacing
8. Pad 回 crop 前的 shape

###### 10.2 Test Time Augmentation (TTA)

- nnUNet 方案：只做 mirror（翻转），不做旋转
- 实现：对每个 flip 组合推理一次，softmax 结果取平均，再 argmax
- flip 组合（3D）：8 种（三轴各是否翻转）
- 提升：Dice 通常 +0.5~1%；代价：推理时间 ×8
- **建议**：最终提交时开，平时调试关

###### 10.3 两阶段推理（本项目特有）

1. **Stage1**：全图滑窗 → 肝脏分割 → 计算 liver bounding box
2. **Stage2**：裁到 liver ROI → 肿瘤分割 → 映射回全图坐标
3. **关键**：Stage2 推理时的 bbox 要和训练时保持一致的 margin

---

##### 第十一节 后处理

###### 11.1 最大连通域过滤（Largest Connected Component）

- 操作：删除预测中除最大连通域以外的所有碎片
- **适用**：肝脏（通常只有一个连通体）
- **不适用**：肿瘤（可能有多个独立转移灶）
- nnUNet 方案：在验证集上测试是否有效再决定用不用（自动决策）

###### 11.2 阈值调整（Threshold Tuning）

| 目标 | 阈值方向 |
|------|----------|
| 提高 Recall（减少漏检）| 降低阈值（如 0.3）|
| 提高 Precision（减少误检）| 提高阈值（如 0.7）|

- nnUNet 不做此项（固定 0.5），本项目可在 val 上 sweep

###### 11.3 体积过滤（Volume Filtering）

- 删除面积/体积低于阈值的预测区域
- 本项目：删除体素数 `< min_size` 的肿瘤预测
- `min_size` 需在验证集上调

---

##### 第十二节 实验管理

###### 12.1 每次实验必须保存的信息

- [ ] `config.json`：所有超参（lr, batch_size, patch, loss, augmentation 参数...）
- [ ] 运行命令：保存为 `cmd.txt`（方便复现）
- [ ] 数据集统计：train/val/test case 数，有无肿瘤 case 数
- [ ] 训练 log：每 epoch 的 loss, lr, val Dice
- [ ] Best checkpoint：val Dice 最高时的 `model.pth`
- [ ] Last checkpoint：最后一 epoch 的完整状态（供 resume）

###### 12.2 实验目录结构

```
experiments/
└── exp_name/
    └── train/
        └── MM-DD-HH-MM-SS/       # 时间戳防覆盖
            ├── config.json
            ├── cmd.txt
            ├── log.txt
            ├── best.pth
            ├── last.pth
            └── diag.txt          # 数据集检查日志
```

###### 12.3 跨实验对齐（消除混淆变量）

- [ ] 固定随机种子：`set_seed(42)`
- [ ] 固定数据集划分：所有实验用同一个 train/val/test split
- [ ] 固定总训练步数：`--total_steps` 而不是 `--epochs`
- [ ] 单次只改一个变量（不要同时改 loss 和 augmentation）

###### 12.4 调试时的快速验证技巧

- 先用小数据（`--train_n 5`）跑通完整流程
- 先跑少量 epoch（`--epochs 5`）确认 loss 下降、无 NaN
- 看前 3 个 batch 的 label 统计：确认前景体素数不为 0
- 验证前先看：预测图有没有合理的前景区域

---

##### 第十三节 常见坑和经验教训

###### 坑1：标签插值用错阶数

- **症状**：label 出现 0.5 这样的中间值，Dice 全为 0 或很低
- **原因**：resample 标签用了 order=3（三次样条）
- **修复**：标签 resample 永远用 order=0（最近邻）

###### 坑2：验证用了训练时的数据增强

- **症状**：val Dice 虚高，test Dice 很低，泛化差
- **修复**：`build_val_transforms()` 里不加随机增强，只做 normalize

###### 坑3：Sigmoid + DiceLoss 出现 NaN

- **症状**：训练几十步后 loss 突然变 NaN
- **原因**：sigmoid 输出极端值（0/1）导致 log(0)=-inf
- **修复**：改用 Softmax（双通道输出）+ DiceCE，或给 sigmoid 加 epsilon

###### 坑4：val Dice 统计包含了 GT 为空的 case

- **症状**：val Dice 很高但实际漏检严重
- **原因**：GT 无肿瘤的 case，预测全 0 时 Dice=1.0
- **修复**：只统计 GT > 0 的 case

###### 坑5：Hard mining + 固定步数导致采样比例失控

- **症状**：小肿瘤 case 出现频率与预期不符
- **原因**：用 `itertools.cycle` 在 `_indices` 列表中间截断 epoch
- **修复**：硬挖掘和固定步数不要混用，选其一

###### 坑6：Coarse tumor channel 训练验证不一致

- **症状**：用 GT mask 训练，用模型预测测试，性能落差大（domain gap）
- **原因**：验证/推理时 Ch2 换成了模型预测（带噪声），与训练时 GT 不符
- **修复**：验证时也用 Stage1 模型预测输出，而不是 GT

###### 坑7：batch Dice vs sample Dice

- **症状**：小目标 case Loss 接近 0，实际 Dice 很低
- **原因**：sample-level Dice 对含大量前景的 case 权重过高
- **修复**：`batch=True`（把 batch 内所有体素合并再算 Dice），对小目标更公平

---

*END*


---
