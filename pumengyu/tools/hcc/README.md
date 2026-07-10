# HCC-TACE-Seg 多期 CT 工具

目标：把 `/home/PuMengYu/HCC/HCC-TACE-Seg_v1_202201` 转成独立 nnU-Net 数据集，例如：

```text
/home/PuMengYu/nnUNet_workspace/raw/Dataset013_HCCMultiPhase
```

硬约束：

- 不写入 `Dataset003_Liver`
- 不修改 MSD/LiTS 的 `splits_final.json`
- HCC 与 MSD 物理隔离；后续混合训练只能通过 Trainer/Mixin 运行时组合

## 1. 扫描 DICOM series

```bash
python -m pumengyu.tools.hcc.build_inventory \
  --hcc-root /home/PuMengYu/HCC/HCC-TACE-Seg_v1_202201 \
  --out-csv pumengyu/notes/data/hcc_series_inventory.csv \
  --out-json pumengyu/notes/data/hcc_series_inventory_summary.json
```

当前扫描结果：

```text
patients: 105
series:   676
CT:       572
SEG:      104
missing SEG: HCC_103
```

## 2. 生成 2 通道 case plan

```bash
python -m pumengyu.tools.hcc.plan_multiphase_cases \
  --inventory pumengyu/notes/data/hcc_series_inventory.csv \
  --out-csv pumengyu/notes/data/hcc_multiphase_case_plan.csv \
  --out-json pumengyu/notes/data/hcc_multiphase_case_plan_summary.json
```

当前 plan：

```text
ready_2ch:  89
review_2ch: 7
exclude:    9
```

2 通道定义：

```text
0000 = PRE / non-contrast CT
0001 = SEG-referenced contrast CT
label = DICOM SEG converted to liver=1, tumor=2
```

## 3. 转换为独立 nnU-Net raw dataset

先在 `/tmp` 做 smoke test：

```bash
python -m pumengyu.tools.hcc.convert_multiphase \
  --plan-csv pumengyu/notes/data/hcc_multiphase_case_plan.csv \
  --out-dir /tmp/Dataset013_HCCMultiPhase_smoke \
  --max-cases 5
```

正式转换需要写入 `nnUNet_workspace/raw`，不要写进 `Dataset003_Liver`：

```bash
python -m pumengyu.tools.hcc.convert_multiphase \
  --plan-csv pumengyu/notes/data/hcc_multiphase_case_plan.csv \
  --out-dir /home/PuMengYu/nnUNet_workspace/raw/Dataset013_HCCMultiPhase
```

如需包含 `review_2ch` case：

```bash
python -m pumengyu.tools.hcc.convert_multiphase \
  --plan-csv pumengyu/notes/data/hcc_multiphase_case_plan.csv \
  --out-dir /home/PuMengYu/nnUNet_workspace/raw/Dataset013_HCCMultiPhase \
  --include-review
```

当前正式转换结果：

```text
out_dir: /home/PuMengYu/nnUNet_workspace/raw/Dataset013_HCCMultiPhase
converted ready_2ch: 89
QC excluded: HCC_089, HCC_099
remaining labelsTr: 87
remaining imagesTr: 174
QC ok: 85
QC review: HCC_065, HCC_075
```

QC 命令：

```bash
python -m pumengyu.tools.hcc.qc_converted_dataset \
  --dataset-dir /home/PuMengYu/nnUNet_workspace/raw/Dataset013_HCCMultiPhase \
  --out-csv pumengyu/notes/data/hcc_converted_qc.csv \
  --out-json pumengyu/notes/data/hcc_converted_qc_summary.json
```

## 当前实现说明

- SEG 中 `Liver` 转 label 1。
- SEG 中 `Mass` 和 `Necrosis` 转 label 2。
- `Portal vein`、`Abdominal aorta` 暂不写入 label。
- contrast CT 使用 SEG frame 的 `SourceImageSequence` 精确过滤 SOPInstanceUID。
- 对同一 series 内重复 z-position 的 CT slice，保留较大 `InstanceNumber` 的 slice。
- PRE CT 重采样到 contrast CT 空间。

## 4. 预处理与训练入口

预处理已完成：

```bash
nnUNetv2_plan_and_preprocess -d 13 --verify_dataset_integrity
```

当前 Dataset013 plans：

```text
dataset: /home/PuMengYu/nnUNet_workspace/preprocessed/Dataset013_HCCMultiPhase
cases: 87
3d_fullres patch: [40, 224, 224]
3d_fullres batch: 2
3d_fullres spacing: [2.5, 0.78125, 0.78125]
normalization: ZScoreNormalization, ZScoreNormalization
```

纯 HCC 多期 CT 训练入口：

```bash
nnUNetv2_train 13 3d_fullres 0 -tr nnUNetTrainer_MedNeXt_MLA_HCCOnly
nnUNetv2_train 13 3d_fullres 0 -tr nnUNetTrainer_DeepDWIBMedConfig_HCCOnly
```

纯 MSD/LiTS 对照入口：

```bash
nnUNetv2_train 3 3d_fullres 0 -tr nnUNetTrainer_MedNeXt_MLA_MSDOnly
nnUNetv2_train 3 3d_fullres 0 -tr nnUNetTrainer_DeepDWIBMedConfig_MSDOnly
```

MSD + HCC 运行时混合入口：

```bash
nnUNetv2_train 13 3d_fullres 0 -tr nnUNetTrainer_MedNeXt_MLA_MSDHCCMix
nnUNetv2_train 13 3d_fullres 0 -tr nnUNetTrainer_DeepDWIBMedConfig_MSDHCCMix
```

这些 Trainer 不修改 `Dataset003_Liver` 或 `Dataset013_HCCMultiPhase`。纯 MSD / 纯 HCC 数据集由 `-d` 决定；混合 Trainer 第一版要求 `-d 13`，以 HCC 为主训练集和验证集，运行时追加 Dataset003 训练样本。

混合开关在 `HCCMixTrainingMixin` 里：

```text
MIX_HCC_ENABLE = True
MIX_MSD_DATASET = "Dataset003_Liver"
MIX_MSD_RATIO = 1.0
MIX_HCC_CHANNEL_MAP = "msd_repeat"
```
