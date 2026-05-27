# 外部无肿瘤 Case 导入框架

> 目标：向 Dataset003_Liver 注入外部数据集的无肿瘤肝脏 case，缓解类别不平衡（当前每 fold 仅 3 个无肿瘤 case）。  
> **验证集永远只用 Dataset003 原始 case，历史实验结果完全可比。**

---

## 数据来源

| 数据集 | 无肿瘤 case | 状态 | staging 目录 |
|--------|------------|------|-------------|
| **3D-IRCADb** (case 5/7/11/14/20) | 5 个 | ✅ 本地已有 | `external_staging/ircad/` |
| **CHAOS CT** (20 个正常肝脏) | 20 个 | ⏳ 需下载 | `external_staging/chaos/` |

CHAOS 下载地址：`grand-challenge.org` 搜索 CHAOS 2019，注册后免费下载。  
下载后解压到 `/home/PuMengYu/8T/Datasets/CHAOS/Train_Sets/`，再跑一次导入脚本即可。

---

## 文件说明

| 文件 | 作用 |
|------|------|
| `_preprocess.py` | nnUNet preprocessing 核心（resampling + CTNormalization + .b2nd 写入） |
| `convert_ircad.py` | IRCADb DICOM → staging nii.gz |
| `convert_chaos.py` | CHAOS DICOM + PNG Ground → staging nii.gz |
| `inject.py` | staging nii.gz → preprocessed + 修改 splits + 写 log |
| `eject.py` | 根据 log 完整回退 |
| `../../notes/sh/run_external_import.sh` | 一键导入脚本 |

---

## 使用流程

### 第一步：dry_run 确认计划

```bash
cd /home/PuMengYu/nnUNet
bash pumengyu/notes/sh/run_external_import.sh --dry_run
```

### 第二步：正式导入

```bash
bash pumengyu/notes/sh/run_external_import.sh
```

脚本会自动：
1. 将 IRCADb 5 个无肿瘤 case 从 DICOM 转为 nii.gz
2. 对每个 case 做 nnUNet preprocessing（resampling 到 `[1.0, 0.768, 0.768]`，CTNormalization clip [-15, 197]）
3. 写入 `.b2nd` + `.pkl` 到 `preprocessed/Dataset003_Liver/nnUNetPlans_3d_fullres/`
4. 备份 `splits_final.json`，将新 case 加入所有 fold 的训练集
5. 写入 `external_cases_log.json`

CHAOS 下载好后，同一命令再跑一次，脚本自动检测目录并处理。

### 第三步：训练加载外部 case 的 trainer

使用方式与原有 trainer 完全相同，splits 已修改，无需改代码：

```bash
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 3 3d_fullres 4 -tr nnUNetTrainer_SizeOversampleV2_NTFP
```

---

## 回退

任何时候都可以完整回退到原始状态：

```bash
# 先确认会删什么
python pumengyu/tools/external_data/eject.py --dry_run

# 实际回退
python pumengyu/tools/external_data/eject.py
```

回退操作：
- 删除 preprocessed 目录中外部 case 的 `.b2nd` / `.pkl`
- 恢复最近一次 `splits_final.json.bak_*` 备份
- 清空 `external_cases_log.json`

备份文件（`splits_final.json.bak_<timestamp>`）永久保留，不会被删除。

---

## 隔离保证

- **验证集**：`splits_final.json` 的 `val` 字段只含 Dataset003 原始 case，inject.py 有断言检查，val 变动时直接报错终止
- **log 记录**：每次注入记录 case_id、来源路径、时间戳、splits 备份文件名
- **可重入**：重复运行 inject.py 会跳过已注入的 case，不重复处理

---

## Preprocessing 参数（来自 Dataset003 nnUNetPlans）

| 参数 | 值 | 来源 |
|------|----|------|
| target spacing (z,y,x) | `[1.0, 0.7676, 0.7676]` | nnUNetPlans.json |
| normalization | CTNormalization | nnUNetPlans.json |
| clip min | -15.0 | dataset_fingerprint.json (p0.5) |
| clip max | 197.0 | dataset_fingerprint.json (p99.5) |
| norm mean | 99.48 | dataset_fingerprint.json |
| norm std | 37.14 | dataset_fingerprint.json |
| data resampling | tricubic (order=3) | nnUNet default |
| seg resampling | nearest neighbor (order=0) | nnUNet default |

---

## 关键文件路径

```
nnUNet_workspace/
├── preprocessed/Dataset003_Liver/
│   ├── splits_final.json               ← 修改此文件（val 不变）
│   ├── splits_final.json.bak_*         ← 自动备份
│   ├── external_cases_log.json         ← 注入记录
│   └── nnUNetPlans_3d_fullres/
│       ├── ircad_005.b2nd              ← 注入的 case
│       ├── ircad_005.pkl
│       └── ...
└── external_staging/
    ├── ircad/                          ← IRCADb 转换结果
    └── chaos/                          ← CHAOS 转换结果（下载后）
```
