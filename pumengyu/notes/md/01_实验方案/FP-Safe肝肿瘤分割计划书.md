# FP-Safe 肝肿瘤分割计划书

日期：2026-06-28

## 一、问题定义

当前实验显示，单纯追求内部测试集 Dice 不足以支撑论文主线：

- Baseline 在内部测试中 Recall 高，但无肿瘤 case 误报严重。
- MedNeXt / MedNeXt_SizeOV4 内部测试最强，但 IRCADb 外部验证 drop 明显。
- 极小肿瘤仍是共同难点，简单连通域阈值后处理不可靠，因为真实 TP 连通域可能小到 1 体素。

因此新主线从“刷最高 Dice”调整为：

> 临床安全的肝肿瘤分割：降低无肿瘤误报、保持小肿瘤召回、提升跨域泛化。

## 二、第一阶段方法：Top-K No-Tumor FP Penalty

已有 `NoTumorFPPenaltyMixin` 使用无肿瘤 patch 的 mean tumor probability 作为惩罚，但 mean 会被大量背景体素稀释。

本阶段新增 `TopKNoTumorFPPenaltyMixin`：

1. 对每个 batch，判断每个 sample 的 GT 是否含肿瘤 label。
2. 仅对 GT 无肿瘤 sample 生效。
3. 取预测 tumor probability 最高的 top-k% voxel。
4. 惩罚这些高置信假阳性区域的均值。

Loss：

```text
L = L_CE+Dice + lambda_fp * mean(topk(P_tumor | no-tumor sample))
```

默认参数：

```text
TKN_TOPK_PERCENT = 0.01
TKN_TUMOR_FP_LAMBDA = 1.0
```

## 三、新增 Trainer

### 1. nnUNetTrainer_FPSafe

原版 nnUNet + Top-K 无肿瘤误报惩罚。

目的：验证 loss 层 FP-Safe 约束是否有效。

### 2. nnUNetTrainer_SizeOV4_FPSafe

SizeOV4 均匀全量 2x 过采样 + Top-K 无肿瘤误报惩罚。

目的：验证“采样 + FP loss”是否协同降低误报，同时保留小肿瘤召回。

### 3. nnUNetTrainer_MedNeXt_FPSafe

MedNeXt-L + Top-K 无肿瘤误报惩罚。

目的：验证强 DW+IB backbone 的内部高分能否在降低误报和提升外部泛化的同时保留。

## 四、训练命令

通用环境变量：

```bash
cd /home/PuMengYu/nnUNet
export nnUNet_raw=/home/PuMengYu/nnUNet_workspace/raw
export nnUNet_preprocessed=/home/PuMengYu/nnUNet_workspace/preprocessed
export nnUNet_results=/home/PuMengYu/nnUNet_workspace/results_v2
export nnUNet_extTrainer=/home/PuMengYu/nnUNet/pumengyu
```

### 1. FP-Safe baseline

```bash
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 003 3d_fullres 0 -tr nnUNetTrainer_FPSafe
```

### 2. SizeOV4 + FP-Safe

```bash
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 003 3d_fullres 0 -tr nnUNetTrainer_SizeOV4_FPSafe
```

### 3. MedNeXt + FP-Safe

```bash
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 003 3d_fullres 0 -tr nnUNetTrainer_MedNeXt_FPSafe
```

## 五、内部测试补跑命令

如果训练完成后 `test_report_custom.txt` 未自动生成：

```bash
python pumengyu/tools/run_internal_test.py --trainer nnUNetTrainer_FPSafe --dataset Dataset003_Liver --fold 0 --gpu 0

python pumengyu/tools/run_internal_test.py --trainer nnUNetTrainer_SizeOV4_FPSafe --dataset Dataset003_Liver --fold 0 --gpu 0

python pumengyu/tools/run_internal_test.py --trainer nnUNetTrainer_MedNeXt_FPSafe --dataset Dataset003_Liver --fold 0 --gpu 0
```

## 六、外部验证命令

```bash
python pumengyu/ext_val/03_gen_method_report.py --method FPSafe --predict --trainer nnUNetTrainer_FPSafe --fold 0 --gpu 0

python pumengyu/ext_val/03_gen_method_report.py --method SizeOV4_FPSafe --predict --trainer nnUNetTrainer_SizeOV4_FPSafe --fold 0 --gpu 0

python pumengyu/ext_val/03_gen_method_report.py --method MedNeXt_FPSafe --predict --trainer nnUNetTrainer_MedNeXt_FPSafe --fold 0 --gpu 0
```

## 七、判定标准

第一阶段成功不要求 Overall 全局第一，而要求至少满足以下方向之一：

1. 无肿瘤误报率明显下降。
2. Precision / FDR 明显改善。
3. IRCADb 外部验证 drop 小于对应 backbone。
4. Tumor Dice 和 Recall 不发生不可接受下降。

关键对照：

| 新方法 | 对照 |
|---|---|
| nnUNetTrainer_FPSafe | Baseline |
| nnUNetTrainer_SizeOV4_FPSafe | SizeOversampleV2 / SizeOV4 系列 |
| nnUNetTrainer_MedNeXt_FPSafe | MedNeXt / MedNeXt_SizeOV4 |

## 八、下一阶段候选改进

如果 Top-K FP loss 有效但 Recall 下降：

- 降低 `TKN_TUMOR_FP_LAMBDA` 到 0.3 或 0.5。
- 降低 `TKN_TOPK_PERCENT` 到 0.005。
- 只在训练后半程启用 FP loss。

如果 Top-K FP loss 对误报改善有限：

- 加 Tumor Presence Auxiliary Head。
- 加 candidate-level FP classifier。
- 加 hard negative mining，把无肿瘤误报 case 动态提高采样概率。

## 九、论文叙事

论文主线不再是“新架构刷最高 Dice”，而是：

> Existing high-Dice liver tumor segmentation models are not necessarily clinically safe. FP-Safe explicitly targets tumor-free false positives and cross-domain robustness while preserving tumor segmentation performance.
