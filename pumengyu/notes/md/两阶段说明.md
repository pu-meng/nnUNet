# 两阶段肝脏肿瘤分割 Pipeline

## 数据集

| 数据集 | 内容 | 来源 |
|---|---|---|
| Dataset003_Liver | 131 cases，标签：background=0, liver=1, tumor=2 | MSD Task03_Liver |
| Dataset004_LiverTumor | 131 cases，肝脏 ROI 裁剪，标签：background=0, tumor=1 | 由 Dataset003 生成 |

5-fold 划分固定（`splits_final.json`），两个数据集完全一致，同一 fold 的 val cases 相同。

---

## 训练策略

### Stage-1：肝脏分割

- 数据集：Dataset003_Liver（完整 CT）
- 训练器：`nnUNetTrainer`
- 目标：分割肝脏（class 1）+ 肿瘤（class 2）
- 状态：**5 fold 全部训练完成** ✓

```bash
nnUNetv2_train 3 3d_fullres 0  # fold 0~4
```

Stage-1 肝脏 Dice：

| fold | 肝脏 Dice | 肿瘤 Dice |
|---|---|---|
| 0 | 0.9652 | 0.6951 |
| 1 | 0.9602 | 0.6257 |
| 2 | 0.9509 | 0.6791 |
| 3 | 0.9572 | 0.6934 |
| 4 | 0.9505 | 0.5771 |
| **均值** | **0.957** | 0.657 |

### Dataset004 生成

用 GT 肝脏+肿瘤 mask（class 1 ∪ class 2）定义 ROI，加 30 mm margin 裁剪。

```bash
bash pumengyu/twostage/run_create_dataset.sh
```

- 训练时用 GT 裁剪：边界完美，无漏掉的肿瘤
- 推理时用 Stage-1 预测裁剪：30 mm margin 补偿预测误差
- 这是两阶段分割的标准做法（H-DenseUNet, LiTS 冠军等均如此）

### Stage-2：肿瘤分割

- 数据集：Dataset004_LiverTumor（肝脏 ROI 裁剪后的 CT）
- 目标：分割肿瘤（class 1）

**Baseline**
```bash
nnUNetv2_train 4 3d_fullres 0  # fold 0~4，标准 nnUNetTrainer
```

**加 Boundary Loss**
```bash
nnUNetv2_train 4 3d_fullres 0 -tr nnUNetTrainer_BoundaryLoss
```

---

## 最终评估（原始空间，无 GT 参与推理）

```
Dataset003/imagesTr/liver_xxx.nii.gz   ← 完整原始 CT
        ↓ Stage-1 fold k（从未训练过此 case）
    肝脏预测 mask
        ↓ 裁剪 ROI（30 mm margin，不用 GT）
    裁剪后 CT patch
        ↓ Stage-2 fold k（从未训练过此 case）
    肿瘤预测（裁剪空间）
        ↓ 映射回原始空间
    肿瘤预测（原始空间）
        ↓ 对比 Dataset003/labelsTr/liver_xxx.nii.gz（class 2）
    Dice / HD95（真实 pipeline 性能）
```

5-fold CV：131 个 case 每个恰好被预测一次，无数据泄露。

```bash
python pumengyu/twostage/eval.py --workspace /home/PuMengYu/nnUNet_workspace
```

---

## 对比实验

| 方法 | 训练器 | 数据集 |
|---|---|---|
| 一阶段 baseline | nnUNetTrainer | Dataset003 |
| 两阶段 baseline | nnUNetTrainer | Dataset003 → Dataset004 |
| 两阶段 + BoundaryLoss | nnUNetTrainer_BoundaryLoss | Dataset004 |

最终指标均在 **Dataset003 原始空间**计算，使用完整 pipeline 推理。

---

## 代码结构

```
pumengyu/
├── twostage/
│   ├── create_dataset.py      # 生成 Dataset004（GT 裁剪）
│   ├── run_create_dataset.sh  # 一键生成 + 预处理
│   └── eval.py                # 最终端到端评估
└── boundary/
    ├── dist_field.py          # 距离场计算
    └── trainer.py             # nnUNetTrainer_BoundaryLoss
```

---

## 关键路径

```
nnUNet_workspace/
├── raw/
│   ├── Dataset003_Liver/          # 原始数据，评估用
│   └── Dataset004_LiverTumor/     # GT 裁剪，训练 Stage-2 用
├── preprocessed/
│   ├── Dataset003_Liver/
│   └── Dataset004_LiverTumor/
└── results/
    ├── Dataset003_Liver/
    │   └── nnUNetTrainer__nnUNetPlans__3d_fullres/   # Stage-1 模型
    └── Dataset004_LiverTumor/
        ├── nnUNetTrainer__nnUNetPlans__3d_fullres/   # Stage-2 baseline
        └── nnUNetTrainer_BoundaryLoss__...           # Stage-2 + BL
```
