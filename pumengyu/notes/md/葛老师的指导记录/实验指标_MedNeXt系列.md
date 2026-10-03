# MedNeXt 系列三域实验指标

范围：Dataset003_Liver/fold 0 的 MedNeXt source-only 对照与三项修复后 MoE 重训。修复前 MoE、HCC 适配、联合训练和 Task02 不进入本表。

LiTS 为自定义划分中的固定 26 例；IRCADb 为 20 例；HCC-TACE-Seg 为固定 21 例。Liver Dice 对全部病例取均值，Tumor Dice 只对真实有肿瘤的病例取均值，Overall=(Liver+Tumor)/2。排名按本表 8 个方法在对应数据域的 Overall 从高到低计算。

## 指标表

### LiTS

| 方法 | Liver Dice | Tumor Dice | Overall | 排名 |
|---|---:|---:|---:|---:|
| EfficientMedNeXt_L_Official | 0.9520 | 0.7515 | 0.8518 | 7/8 |
| MedNeXt | 0.9521 | 0.7600 | 0.8561 | 3/8 |
| MedNeXt_MHA | 0.9533 | 0.7544 | 0.8538 | 5/8 |
| MedNeXt_MLA | 0.9513 | 0.7535 | 0.8524 | 6/8 |
| MedNeXt_SizeOV4 | 0.9545 | 0.7635 | 0.8590 | 1/8 |
| MedNeXt_MHA_MoE | 0.9506 | 0.7576 | 0.8541 | 4/8 |
| MedNeXt_MLA_MoE | 0.9497 | 0.7509 | 0.8503 | 8/8 |
| MedNeXt_MLA_MoE_SizeOV4 | 0.9533 | 0.7589 | 0.8561 | 2/8 |

### IRCADb

| 方法 | Liver Dice | Tumor Dice | Overall | 排名 |
|---|---:|---:|---:|---:|
| EfficientMedNeXt_L_Official | 0.9656 | 0.6973 | 0.8315 | 5/8 |
| MedNeXt | 0.9660 | 0.6900 | 0.8280 | 6/8 |
| MedNeXt_MHA | 0.9672 | 0.7303 | 0.8487 | 1/8 |
| MedNeXt_MLA | 0.9660 | 0.6778 | 0.8219 | 7/8 |
| MedNeXt_SizeOV4 | 0.9651 | 0.7131 | 0.8391 | 3/8 |
| MedNeXt_MHA_MoE | 0.9650 | 0.7070 | 0.8360 | 4/8 |
| MedNeXt_MLA_MoE | 0.9618 | 0.6729 | 0.8174 | 8/8 |
| MedNeXt_MLA_MoE_SizeOV4 | 0.9645 | 0.7203 | 0.8424 | 2/8 |

### HCC

| 方法 | Liver Dice | Tumor Dice | Overall | 排名 |
|---|---:|---:|---:|---:|
| EfficientMedNeXt_L_Official | 0.8375 | 0.3468 | 0.5921 | 7/8 |
| MedNeXt | 0.8383 | 0.4175 | 0.6279 | 5/8 |
| MedNeXt_MHA | 0.8435 | 0.4035 | 0.6235 | 6/8 |
| MedNeXt_MLA | 0.8418 | 0.4645 | 0.6532 | 1/8 |
| MedNeXt_SizeOV4 | 0.8146 | 0.3405 | 0.5775 | 8/8 |
| MedNeXt_MHA_MoE | 0.8394 | 0.4342 | 0.6368 | 4/8 |
| MedNeXt_MLA_MoE | 0.8405 | 0.4511 | 0.6458 | 2/8 |
| MedNeXt_MLA_MoE_SizeOV4 | 0.8465 | 0.4431 | 0.6448 | 3/8 |

## 三域综合排名

综合排名按 LiTS、IRCADb、HCC 三个 Overall 排名的平均值从小到大排列；平均排名相同时，三域 Overall 均值更高者优先。

| 综合排名 | 方法 | LiTS 排名 | IRCADb 排名 | HCC 排名 | 平均排名 | 三域 Overall 均值 |
|---:|---|---:|---:|---:|---:|---:|
| 1/8 | MedNeXt_MLA_MoE_SizeOV4 | 2/8 | 2/8 | 3/8 | 2.33 | 0.7811 |
| 2/8 | MedNeXt_MHA_MoE | 4/8 | 4/8 | 4/8 | 4.00 | 0.7756 |
| 3/8 | MedNeXt_MHA | 5/8 | 1/8 | 6/8 | 4.00 | 0.7753 |
| 4/8 | MedNeXt_SizeOV4 | 1/8 | 3/8 | 8/8 | 4.00 | 0.7585 |
| 5/8 | MedNeXt_MLA | 6/8 | 7/8 | 1/8 | 4.67 | 0.7758 |
| 6/8 | MedNeXt | 3/8 | 6/8 | 5/8 | 4.67 | 0.7706 |
| 7/8 | MedNeXt_MLA_MoE | 8/8 | 8/8 | 2/8 | 6.00 | 0.7712 |
| 8/8 | EfficientMedNeXt_L_Official | 7/8 | 5/8 | 7/8 | 6.33 | 0.7585 |
