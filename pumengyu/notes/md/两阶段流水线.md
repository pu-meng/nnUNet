# 两阶段肝脏肿瘤分割流程

## 动机

Dataset003_Liver 原始 CT 中位体积约 113M voxels，肝脏 ROI 仅约 7M voxels。
当前直接在全图上训练存在两个问题：

- patch（128³）只覆盖全图 ~1.8%，肿瘤体素极少，训练效率低
- 每折训练约 2 天，实验迭代慢

将训练域收窄到肝脏 ROI 后，体积压缩约 16x，每折训练时间预期降至数小时。

---

## 整体架构

```
Stage 1（已完成）
  Dataset003_Liver → 5折 nnUNetTrainer → 肝脏分割模型
                                          ↓
                              fold_k validation 预测（已存在）

Stage 2（本流程）
  bbox_crops.json ← pred_liver only + 20mm padding   ← 与测试时完全一致
        ↓
  Dataset004_LiverTumor（物理裁剪，label: 0=bg, 1=tumor）
        ↓
  5折 nnUNetTrainer → 肿瘤分割模型
        ↓
  端到端评估（输入原始 .nii.gz，只用模型权重）
```

---

## 目录结构

```
pumengyu/scripts/
├── build_crop_json.py       # Step 1: 生成 bbox JSON
├── create_dataset004.py     # Step 2: 裁剪生成新数据集
├── eval_two_stage.py        # Step 4: 端到端评估
└── bbox_crops.json          # 生成物（每个 case 的裁剪坐标）

nnUNet_workspace/
├── raw/
│   ├── Dataset003_Liver/    # 原始数据（不动）
│   └── Dataset004_LiverTumor/   # 裁剪后数据（生成物）
├── preprocessed/
│   └── Dataset004_LiverTumor/   # nnUNet 预处理后（splits 复用 D003）
└── results/
    ├── Dataset003_Liver/    # Stage 1 模型（5折已完成）
    └── Dataset004_LiverTumor/   # Stage 2 模型（待训练）
```

---

## 执行步骤

### Step 1 — 生成裁剪坐标 JSON

```bash
python pumengyu/scripts/build_crop_json.py
```

**逻辑：**
- 读取 Stage 1 已有的 5折 cross-val 预测（`fold_k/validation/`）
- 对每个 case：只用 `pred_liver`，外扩 20mm（不用 GT）
- 输出 `bbox_crops.json`，记录每个 case 的 xyz 裁剪范围

> fold_k 的验证 case 由 fold_k 的模型预测（训练时未见该 case），无 data leakage。
> crop 策略与测试时完全相同，训练/测试分布一致。

---

### Step 2 — 生成 Dataset004

```bash
python pumengyu/scripts/create_dataset004.py
```

**逻辑：**
- 按 `bbox_crops.json` 裁剪原图和 label
- label 重映射：liver(1)→0，tumor(2)→1
- 更新 affine origin，保证坐标系正确
- 复制 `splits_final.json`（与 Dataset003 完全一致，fold 定义不变）

---

### Step 3 — 预处理 + 训练

```bash
# 预处理
nnUNetv2_plan_and_preprocess -d 4 --verify_dataset_integrity

# 训练（5折，可并行）
nnUNetv2_train 4 3d_fullres 0
nnUNetv2_train 4 3d_fullres 1
nnUNetv2_train 4 3d_fullres 2
nnUNetv2_train 4 3d_fullres 3
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 4 3d_fullres 4

```

---

### Step 4 — 端到端评估

```bash
# Dataset004 所有折训完后，直接运行
python pumengyu/scripts/eval_two_stage.py

# 也可以只评估某一折（训完一折即可先看结果）
python pumengyu/scripts/eval_two_stage.py --fold 0
```

**评估 pipeline（输入原始 .nii.gz，完全不用 GT）：**
```
原始 CT（Dataset003 imagesTr）
  → fold_k 肝脏模型实时推理  → liver mask
  → pred bbox + 20mm padding  → crop CT（内存中，不存文件）
  → fold_k 肿瘤模型实时推理  → tumor mask（裁剪空间）
  → 映射回原始坐标系
  → Dice(liver) + Dice(tumor)  vs  GT（Dataset003 labelsTr）
```

输出 `eval_two_stage_results.json`，包含 per-case 和汇总指标。

---

## 指标为什么可信

两个独立保证：

**1. 训练/测试分布完全一致**
- Dataset004 的 crop = pred only + padding
- 评估时的 crop = pred only + padding
- 没有任何 GT 参与 crop 过程，训练看到的和测试看到的是同一分布

**2. 最终在原始数据上验证**
- 评估输入：原始 .nii.gz
- 评估输出：原始空间的 Dice
- fold_k 验证 case 只用 fold_k 的模型（训练时未见）
- 与其他方法口径完全一致，结果无可质疑

---

## 预期收益

| 指标 | Stage 1（全图） | Stage 2（ROI） |
|---|---|---|
| 训练域体积 | ~113M voxels | ~7M voxels |
| 每折训练时间 | ~2 天 | 预计 2~4 小时 |
| patch 覆盖率 | ~1.8% | ~30% |
| 肿瘤 Dice | 受肝脏平均拉高 | 专注肿瘤，预期提升 |
