# HCC-TACE-Seg Referenced CT 重建记录

## 背景

旧版 `Dataset013_HCCMultiPhase` 已删除。旧版使用 2 通道输入：

```text
0000 = PRE / non-contrast CT
0001 = DICOM-SEG 明确引用的 contrast CT
label = DICOM-SEG 转换标签
```

问题是 `0000` 并不是 DICOM-SEG 直接标注的 CT series，只是同一次 baseline study 内的另一个 CT series，存在相位、呼吸和配准偏差风险。为了得到和 MSD/LiTS 更一致的监督分割数据，应改为只使用 DICOM-SEG 明确引用的 CT。

## 新数据集标准

数据集名：

```text
Dataset013_HCCReferencedCT
```

图像：

```text
imagesTr/<case>_0000.nii.gz
```

来源仅限 DICOM-SEG 内部 `ReferencedSeriesSequence` 指向的 CT series。不要使用：

```text
同 study 内未被 SEG 引用的 CT
follow-up CT
PRE CT
其他无标签 CT series
```

标签：

```text
labelsTr/<case>.nii.gz
```

标签映射：

```text
0 = background
1 = liver
2 = tumor lesion
```

原始 DICOM-SEG segment 处理：

```text
Liver -> 1
Mass -> 2
Necrosis -> 2
Portal vein -> 0
Abdominal aorta -> 0
```

`Mass` 在 DICOM-SEG 中的说明为 `Tumor`，按肿瘤病灶处理。`Necrosis` 表示坏死区域，仍属于肿瘤病灶范围，因此并入 label 2。`Portal vein` 和 `Abdominal aorta` 不属于 MSD/LiTS 的肝脏肿瘤任务标签，忽略为背景。

## 病例筛选

本地 HCC 数据：

```text
/home/PuMengYu/HCC/HCC-TACE-Seg_v1_202201
```

扫描结论：

```text
总病例数: 105
缺少 SEG: HCC_103
SEG 引用 CT 未找到: HCC_048
可转换 referenced-CT 病例数: 103
```

初始可转换结果：

```text
imagesTr: 103 个 _0000.nii.gz
labelsTr: 103 个 .nii.gz
```

转换后 QC 排除：

```text
HCC_089: label 全 0，empty liver / empty tumor
HCC_099: liver 体素异常少，tumor/liver ratio 极高
```

最终训练集：

```text
imagesTr: 101
labelsTr: 101
numTraining: 101
_excluded_qc: HCC_089, HCC_099
```

保留 review：

```text
HCC_065: tumor/liver ratio = 3.067923
HCC_075: tumor/liver ratio = 1.516528
```

这两个病例没有空标签，暂不删除，只在 QC summary 中标记 review。

## 元数据保留

训练只使用 image/label，但必须额外保留追溯元数据：

```text
case_metadata.csv
case_metadata.json
```

字段包括：

```text
case_id
image_source_path
label_seg_path
study_folder
image_series_folder
label_series_folder
study_date
referenced_ct_description
referenced_ct_series_uid
seg_series_uid
segment_labels
image_shape_zyx
label_values
liver_voxels
tumor_voxels
```

## 执行命令

单病例 smoke test：

```bash
python -m pumengyu.tools.hcc.convert_referenced_ct \
  --inventory pumengyu/notes/data/hcc_series_inventory.csv \
  --out-dir /tmp/Dataset013_HCCReferencedCT_smoke \
  --cases HCC_001
```

正式转换：

```bash
python -m pumengyu.tools.hcc.convert_referenced_ct \
  --inventory pumengyu/notes/data/hcc_series_inventory.csv \
  --out-dir /home/PuMengYu/nnUNet_workspace/raw/Dataset013_HCCReferencedCT
```

转换后再运行 nnU-Net 数据完整性检查和预处理。

## 本次执行结果

已生成 raw dataset：

```text
/home/PuMengYu/nnUNet_workspace/raw/Dataset013_HCCReferencedCT
```

关键文件：

```text
dataset.json
case_metadata.csv
case_metadata.json
imagesTr/
labelsTr/
_excluded_qc/
```

QC 文件：

```text
pumengyu/notes/data/hcc_referenced_ct_qc.csv
pumengyu/notes/data/hcc_referenced_ct_qc_summary.json
```

`dataset.json`：

```text
channel_names: 0 = CT
labels: background=0, liver=1, tumor=2
numTraining: 101
```

完整性检查和预处理命令：

```bash
nnUNetv2_plan_and_preprocess -d 13 --verify_dataset_integrity
```

执行结果：

```text
verify_dataset_integrity: passed
preprocessed dataset: /home/PuMengYu/nnUNet_workspace/preprocessed/Dataset013_HCCReferencedCT
gt_segmentations: 101
```

生成配置：

```text
2d:
  spacing = [0.78125, 0.78125]
  patch = [512, 512]
  batch = 12
  normalization = CTNormalization

3d_fullres:
  spacing = [2.5, 0.78125, 0.78125]
  patch = [40, 224, 224]
  batch = 2
  normalization = CTNormalization

3d_lowres:
  spacing = [2.5, 1.253676905545928, 1.253676905545928]
  patch = [56, 224, 192]
  batch = 2
  normalization = CTNormalization
```

## 训练策略

第一阶段先跑纯 HCC sanity trainer，不混入 MSD/LiTS，不加过采样，不使用旧多期 CT 逻辑：

```bash
nnUNetv2_train 13 3d_fullres 0 -tr nnUNetTrainer_MedNeXt_MLA_HCCRefOnly
```

对应 trainer：

```text
nnUNetTrainer_MedNeXt_MLA_HCCRefOnly
```

设计目的：

```text
1. 验证 referenced-CT 单通道数据能否稳定训练。
2. 验证 HCC 标签和 MSD/LiTS 风格的 0/1/2 任务是否兼容。
3. 得到纯 HCC baseline，作为后续 MSD+HCC 混合训练的对照。
```

暂不直接使用旧的 mixed trainer：

```text
nnUNetTrainer_MedNeXt_MLA_MSDHCCMix
```

原因是旧 `HCCMixTrainingMixin` 面向 `Dataset013_HCCMultiPhase` 2 通道数据，会要求主数据集至少 2 通道；新的 `Dataset013_HCCReferencedCT` 是单通道，不应走旧的 PRE/contrast 通道适配逻辑。

建议后续顺序：

```text
1. HCCRefOnly fold 0：检查训练是否稳定、验证 Dice、典型失败 case。
2. 如纯 HCC 正常，再新建单通道 MSD+HCC mix trainer。
3. mix trainer 中 HCC 和 MSD 都是单通道，不需要 channel repeat/zero-fill。
4. 最后再考虑 SizeOV、FP-safe、loss 改动等策略。
```
