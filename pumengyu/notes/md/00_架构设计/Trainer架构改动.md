# nnUNet Trainer 架构改动说明


#### nnUNet Trainer 架构改动说明

**核心原则：所有 Trainer 继承 `nnUNetTrainer`，通过 Mixin 重写特定方法，
不修改基类。每个 Mixin 只改动流水线的一个环节。**

---

##### nnUNet 训练流水线 & 改动插入点

```
┌─────────────────────────────────────────────────────────────────┐
│  ① get_tr_and_val_datasets()                                    │
│     决定"谁能进训练集"                                           │
│                                                                  │
│     [全部 case identifiers]                                      │
│           │                                                      │
│           ├── SizeStratifiedOversampleMixin  → 小/极大/无肿瘤   │
│           │    case 在列表中重复 N 次（极小×6, 小×5 ...）        │
│           ├── ExternalNoTumorMixin           → 追加外部 25 case  │
│           └── TumorOnlyTrainMixin            → 剔除无肿瘤 case   │
└───────────────────────────────┬─────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│  ② DataLoader + 数据增强（nnUNet 默认 batchgenerators）         │
│     随机裁剪 patch、旋转、缩放、镜像...                          │
│                                                                  │
│           └── NoMirrorMixin   → 关闭所有镜像增强                 │
└───────────────────────────────┬─────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│  ③ train_step(batch)                                            │
│     拿到一个 batch 后的处理                                      │
│                                                                  │
│     batch = { "data": image, "target": label }                  │
│           │                                                      │
│           └── CopyPasteMixin → 50% 概率把小肿瘤 ROI              │
│                粘贴进 batch 里其他 case 的肝脏区域               │
│                （在送入网络之前改了 image 和 label）              │
└───────────────────────────────┬─────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│  ④ build_network_architecture()  ← 网络结构本身                 │
│                                                                  │
│  输入: image (B,C,Z,Y,X)                                        │
│                                                                  │
│  Encoder（3层下采样）                                            │
│  ┌──────────────────┐                                           │
│  │ ConvBlock×2      │ 64ch   ← 高分辨率，局部纹理               │
│  └────────┬─────────┘                                           │
│           ↓ stride=2                                             │
│  ┌──────────────────┐                                           │
│  │ ConvBlock×2      │ 128ch                                     │
│  └────────┬─────────┘                                           │
│           ↓ stride=2                                             │
│  ┌──────────────────┐                                           │
│  │ ConvBlock×2      │ 256ch                                     │
│  └────────┬─────────┘                                           │
│           ↓ stride=2                                             │
│  ┌──────────────────┐                                           │
│  │ Bottleneck 320ch │  ← nnUNetTrainer_UMamba 在这里            │
│  │ (默认 ConvBlock)  │     替换成 UMambaBot3D 的 Mamba 块        │
│  └────────┬─────────┘     （长程全局上下文建模）                  │
│           ↓                                                      │
│  Decoder（逐层上采样，接 encoder skip connection）               │
│  ┌──────────────────┐                                           │
│  │ Upsample + Conv  │ 256ch  → deep supervision output 2        │
│  └────────┬─────────┘                                           │
│           ↓                                                      │
│  ┌──────────────────┐                                           │
│  │ Upsample + Conv  │ 128ch  → deep supervision output 1        │
│  └────────┬─────────┘                                           │
│           ↓                                                      │
│  ┌──────────────────┐                                           │
│  │ Upsample + Conv  │  64ch  → deep supervision output 0 (主输出)│
│  └────────┬─────────┘                                           │
│           ↓                                                      │
│   logits (B, num_classes, Z, Y, X)                              │
│                                                                  │
│  ── Stage2 改动 ────────────────────────────────────────────    │
│  Tr_Stage2_FPSup: 输入改为 3 通道                                │
│    [CT,  Stage1概率图,  Stage1二值图]                            │
│    ↑ build_network_architecture 里把 num_input_channels 改成 3  │
└───────────────────────────────┬─────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│  ⑤ _build_loss()  ← Loss 函数                                   │
│                                                                  │
│  默认：CE + Dice（深监督，各层加权求和）                          │
│                                                                  │
│  logits[0] (全分辨率) ──┐                                       │
│  logits[1] (1/2)       ├── 加权求和 → base_loss                 │
│  logits[2] (1/4)       ─┘                                       │
│                                                                  │
│  各 Mixin 在 base_loss 上叠加：                                  │
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ UnifiedFocalLossMixin                                     │  │
│  │   只取 logits[0]（全分辨率），提取肿瘤通道概率             │  │
│  │   base_loss + λ×AsymmetricUFL(p_tumor, y_tumor)          │  │
│  │   δ=0.6 → 更重惩罚 FN（漏检）                            │  │
│  │   δ=0.5 → 对称惩罚 FN/FP                                 │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ TverskyLossMixin                                          │  │
│  │   base_loss + AsymmetricFocalTverskyLoss                  │  │
│  │   δ=0.7 (FN权重) / γ=0.75 (focal)                        │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ NoTumorFPPenaltyMixin                                     │  │
│  │   GT 无肿瘤的 patch：额外 loss += λ×mean(p_tumor)         │  │
│  │   有肿瘤的 patch：loss 路径不变                            │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ Stage2FPSupMixin                                          │  │
│  │   完全替换成：MSE（概率图回归）+ BCE（肿瘤通道）            │  │
│  └───────────────────────────────────────────────────────────┘  │
└───────────────────────────────┬─────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│  ⑥ 训练结束钩子 on_train_end()                                  │
│                                                                  │
│     AutoReportMixin        → 自动 evaluate，生成 report.txt      │
│     AutoInternalTestMixin  → 自动推理内部 26-case 测试集          │
└─────────────────────────────────────────────────────────────────┘
```

---

##### 各 Trainer 改动位置速查

| Trainer | ① 数据集 | ② 增强 | ③ train_step | ④ 网络结构 | ⑤ Loss |
|---------|---------|--------|--------------|-----------|--------|
| `nnUNetTrainer_Baseline` | 默认 | 默认 | 默认 | 默认 | CE+Dice |
| `nnUNetTrainer_UFL` | 默认 | 默认 | 默认 | 默认 | +UFL δ=0.6 |
| `nnUNetTrainer_UFL_v2` | 默认 | 默认 | 默认 | 默认 | +UFL δ=0.5 |
| `nnUNetTrainer_UFL_delta06` | 默认 | 默认 | 默认 | 默认 | +UFL δ=0.6 |
| `nnUNetTrainer_CopyPaste` | 小肿瘤重复×3 | 默认 | **粘贴** | 默认 | CE+Dice |
| `nnUNetTrainer_CopyPaste_v2` | 小肿瘤重复×3 | 默认 | **粘贴(修复)** | 默认 | CE+Dice |
| `nnUNetTrainer_SizeOversample` | **V1重复** | 默认 | 默认 | 默认 | CE+Dice |
| `nnUNetTrainer_SizeOversample_UFL` | **V1重复** | 默认 | 默认 | 默认 | +UFL |
| `nnUNetTrainer_SizeOversampleV2` | **V2重复** | 默认 | 默认 | 默认 | CE+Dice |
| `nnUNetTrainer_SizeOversampleV2_NTFP` | **V2重复** | 默认 | 默认 | 默认 | +NTFP |
| `nnUNetTrainer_SizeOversampleV2_Ext25` | **V2重复+外部25** | 默认 | 默认 | 默认 | CE+Dice |
| `nnUNetTrainer_SizeOversampleV2_NTFP_Ext25` | **V2重复+外部25** | 默认 | 默认 | 默认 | +NTFP |
| `nnUNetTrainer_SizeOversampleV2_Tversky` | **V2重复** | 默认 | 默认 | 默认 | +Tversky |
| `nnUNetTrainer_SizeOversampleV3` | **V3全尺寸** | 默认 | 默认 | 默认 | CE+Dice |
| `nnUNetTrainer_SizeOversampleV3_NoMirror` | **V3全尺寸** | **无镜像** | 默认 | 默认 | CE+Dice |
| `nnUNetTrainer_NoMirror` | 默认 | **无镜像** | 默认 | 默认 | CE+Dice |
| `nnUNetTrainer_Ext25` | **+外部25** | 默认 | 默认 | 默认 | CE+Dice |
| `Tr_Stage1_TumorOnly` | **只有肿瘤case** | 默认 | 默认 | 默认 | CE+Dice |
| `Tr_Stage2_FPSup` | 无肿瘤重复×3 | 默认 | 默认 | **3通道输入** | MSE+BCE |
| `nnUNetTrainer_UMamba` | 默认 | 默认 | 默认 | **Mamba瓶颈** | CE+Dice |
| `nnUNetTrainer_UMamba_SizeOversample` | **V1重复** | 默认 | 默认 | **Mamba瓶颈** | CE+Dice |

---

##### 改动点说明

###### ① `get_tr_and_val_datasets()` — 训练集构成

`SizeStratifiedOversampleMixin` 在 identifiers 列表里重复困难 case 的 key：
```
dataset_tr.identifiers = [
    "liver_0", "liver_0", "liver_0",   # 极小肿瘤 → 重复 6 次（V2）
    "liver_1",                          # 普通 case → 1 次
    "liver_2", "liver_2",              # 小肿瘤 → 5 次（V2）
    ...
]
```
DataLoader 按这个列表采样，频率直接控制曝光次数，不改任何网络代码。

`ExternalNoTumorMixin` 在同一方法里追加外部 case key，
splits_final.json 本身不改，保持干净。

---

###### ③ `train_step()` — CopyPaste 粘贴位置

```
batch["data"]   shape: (B, 1, Z, Y, X)   CT 图像
batch["target"] shape: (B, 1, Z, Y, X)   标签（0=背景 1=肝脏 2=肿瘤）
      │
      └─ CopyPasteMixin.train_step():
             50% 概率：从 small_tumor_pool 抽一个 ROI
             找 batch 里某 case 的肝脏区域
             把 ROI 的体素值贴进去（image + label 同步修改）
             │
             ↓
         修改后的 batch → 送入网络
```
修改只在这一个 batch 里，不写磁盘，不改 dataset。

---

###### ④ `build_network_architecture()` — 网络结构

`UMambaBot3D`：瓶颈层用 Mamba 块替换最后一个 ConvBlock。
其他层（encoder/decoder/skip）完全不变。

```
Encoder 保持不变
        ↓
  ┌────────────┐        ┌─────────────────┐
  │ ConvBlock  │  →→→   │  Mamba Block    │  ← 唯一改动
  │ (默认)     │        │  (SSM 长程建模) │
  └────────────┘        └─────────────────┘
        ↓
Decoder 保持不变
```

`Tr_Stage2_FPSup`：`num_input_channels` 从 1 改为 3，
网络第一个卷积层接收 `[CT, Stage1概率图, Stage1二值图]`，
其他结构不变。

---

###### ⑤ `_build_loss()` — Loss 函数叠加

```
默认 _build_loss() 返回：
  loss = DC_and_CE_loss(...)    CE + Dice 加权

UnifiedFocalLossMixin._build_loss() 返回：
  loss = _UFLWrapper(
      base_loss = DC_and_CE_loss(...),
      ufl_fn    = AsymmetricUnifiedFocalLoss(δ, γ),
      tumor_idx = 2,
      λ         = 0.5,
  )
  → forward() 时：total = base + λ×UFL

NoTumorFPPenaltyMixin 在 train_step 里额外判断：
  if GT 无肿瘤:
      loss += λ × mean(softmax(logits)[:, tumor_idx])
```

---

##### 代码位置

| 组件 | 文件 | 关键方法 |
|------|------|---------|
| 所有自定义 Trainer | `pumengyu/trainers/trainer.py` | 类定义 |
| 所有 Mixin 实现 | `pumengyu/mixins.py` | 见下表 |
| UMambaBot3D | `pumengyu/architectures/umamba.py` | `forward()` |

| Mixin | 重写的方法 |
|-------|----------|
| `SizeStratifiedOversampleMixin` | `get_tr_and_val_datasets()` |
| `ExternalNoTumorMixin` | `get_tr_and_val_datasets()` |
| `TumorOnlyTrainMixin` | `get_tr_and_val_datasets()` |
| `NoMirrorMixin` | `initialize()` 里关闭镜像参数 |
| `CopyPasteMixin` | `train_step()` |
| `UnifiedFocalLossMixin` | `_build_loss()` |
| `TverskyLossMixin` | `_build_loss()` |
| `NoTumorFPPenaltyMixin` | `_build_loss()` + `train_step()` |
| `Stage2FPSupMixin` | `build_network_architecture()` + `_build_loss()` |
| `AutoReportMixin` | `on_train_end()` |
| `AutoInternalTestMixin` | `on_train_end()` |


---
