# results_v2 内部与外部验证结果列表

> 更新：2026-07-02  
> 口径：Overall = (Liver Dice + Tumor Dice) / 2。内部来自 `Dataset003_Liver/*/fold_0/test_report_custom.txt`，外部来自 `ExternalVal_IRCADb/*/report_custom.txt`。

---

## 外部验证 Overall 排名（IRCADb）

| Rank | Method | Overall | Liver | Tumor | Recall | Precision | FP率 | 严重/改进 | 内部Overall | 外-内 | Report |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | MedNeXt_MLA | 0.8079 | 0.9673 | 0.6484 | 0.6665 | 0.7437 | 40% (2/5) | 2/0 | 0.8259 | -0.0180 | [report](/home/PuMengYu/nnUNet_workspace/results_v2/ExternalVal_IRCADb/MedNeXt_MLA/report_custom.txt) |
| 2 | MoE_SizeOV5 | 0.8025 | 0.9679 | 0.6371 | 0.6437 | 0.7464 | 40% (2/5) | 2/1 | 0.8167 | -0.0142 | [report](/home/PuMengYu/nnUNet_workspace/results_v2/ExternalVal_IRCADb/MoE_SizeOV5/report_custom.txt) |
| 3 | MLAUNet | 0.8008 | 0.9675 | 0.6341 | 0.6320 | 0.7580 | 40% (2/5) | 1/2 | 0.8148 | -0.0140 | [report](/home/PuMengYu/nnUNet_workspace/results_v2/ExternalVal_IRCADb/MLAUNet/report_custom.txt) |
| 4 | SizeOV2 | 0.7992 | 0.9676 | 0.6307 | 0.6352 | 0.7547 | 40% (2/5) | 1/3 | 0.8187 | -0.0195 | [report](/home/PuMengYu/nnUNet_workspace/results_v2/ExternalVal_IRCADb/SizeOV2/report_custom.txt) |
| 5 | MLA_GK5_V4 | 0.7957 | 0.9656 | 0.6258 | 0.6472 | 0.7291 | 40% (2/5) | 1/2 | 0.8173 | -0.0216 | [report](/home/PuMengYu/nnUNet_workspace/results_v2/ExternalVal_IRCADb/MLA_GK5_V4/report_custom.txt) |
| 6 | MoE | 0.7922 | 0.9669 | 0.6175 | 0.6413 | 0.7140 | 40% (2/5) | 2/2 | 0.8166 | -0.0244 | [report](/home/PuMengYu/nnUNet_workspace/results_v2/ExternalVal_IRCADb/MoE/report_custom.txt) |
| 7 | DeepDWIBResGN | 0.7886 | 0.9630 | 0.6142 | 0.6255 | 0.7576 | 40% (2/5) | 2/1 | 0.8198 | -0.0312 | [report](/home/PuMengYu/nnUNet_workspace/results_v2/ExternalVal_IRCADb/DeepDWIBResGN/report_custom.txt) |
| 8 | MLAUNet_1500 | 0.7886 | 0.9673 | 0.6099 | 0.6453 | 0.7070 | 40% (2/5) | 2/2 | 0.8028 | -0.0142 | [report](/home/PuMengYu/nnUNet_workspace/results_v2/ExternalVal_IRCADb/MLAUNet_1500/report_custom.txt) |
| 9 | MoE_SizeOV4 | 0.7871 | 0.9674 | 0.6068 | 0.6371 | 0.7081 | 40% (2/5) | 1/3 | 0.8330 | -0.0459 | [report](/home/PuMengYu/nnUNet_workspace/results_v2/ExternalVal_IRCADb/MoE_SizeOV4/report_custom.txt) |
| 10 | MedNeXt_MLA_SizeOV4 | 0.7870 | 0.9650 | 0.6091 | 0.6580 | 0.7019 | 60% (3/5) | 2/0 | 0.8285 | -0.0415 | [report](/home/PuMengYu/nnUNet_workspace/results_v2/ExternalVal_IRCADb/MedNeXt_MLA_SizeOV4/report_custom.txt) |
| 11 | SizeOV3 | 0.7855 | 0.9675 | 0.6034 | 0.6565 | 0.6970 | 60% (3/5) | 1/2 | 0.8143 | -0.0288 | [report](/home/PuMengYu/nnUNet_workspace/results_v2/ExternalVal_IRCADb/SizeOV3/report_custom.txt) |
| 12 | nnFormer | 0.7826 | 0.9600 | 0.6052 | 0.6199 | 0.7461 | 40% (2/5) | 2/2 | 0.7732 | +0.0094 | [report](/home/PuMengYu/nnUNet_workspace/results_v2/ExternalVal_IRCADb/nnFormer/report_custom.txt) |
| 13 | MedNeXt_SizeOV4 | 0.7797 | 0.9651 | 0.5943 | 0.6500 | 0.6795 | 60% (3/5) | 2/1 | 0.8431 | -0.0634 | [report](/home/PuMengYu/nnUNet_workspace/results_v2/ExternalVal_IRCADb/MedNeXt_SizeOV4/report_custom.txt) |
| 14 | MedNeXt_MLA_FPSafe | 0.7744 | 0.9637 | 0.5852 | 0.6442 | 0.6711 | 60% (3/5) | 2/1 | 0.8326 | -0.0582 | [report](/home/PuMengYu/nnUNet_workspace/results_v2/ExternalVal_IRCADb/MedNeXt_MLA_FPSafe/report_custom.txt) |
| 15 | Baseline | 0.7727 | 0.9673 | 0.5781 | 0.6253 | 0.6814 | 60% (3/5) | 1/3 | 0.7941 | -0.0214 | [report](/home/PuMengYu/nnUNet_workspace/results_v2/ExternalVal_IRCADb/Baseline/report_custom.txt) |
| 16 | MedNeXt | 0.7705 | 0.9660 | 0.5750 | 0.6554 | 0.6564 | 60% (3/5) | 3/0 | 0.8402 | -0.0697 | [report](/home/PuMengYu/nnUNet_workspace/results_v2/ExternalVal_IRCADb/MedNeXt/report_custom.txt) |
| 17 | MLAUNet_MoE_IB7_SizeOV4 | 0.7670 | 0.9675 | 0.5664 | 0.6251 | 0.6796 | 60% (3/5) | 3/0 | 0.8192 | -0.0522 | [report](/home/PuMengYu/nnUNet_workspace/results_v2/ExternalVal_IRCADb/MLAUNet_MoE_IB7_SizeOV4/report_custom.txt) |
| 18 | DeepPlainResGN_SizeOV4 | 0.7623 | 0.9536 | 0.5710 | 0.6359 | 0.6933 | 60% (3/5) | 2/1 | 0.7908 | -0.0285 | [report](/home/PuMengYu/nnUNet_workspace/results_v2/ExternalVal_IRCADb/DeepPlainResGN_SizeOV4/report_custom.txt) |
| 19 | MoE_SizeOV2 | 0.7596 | 0.9668 | 0.5523 | 0.6308 | 0.6470 | 80% (4/5) | 2/2 | 0.8152 | -0.0556 | [report](/home/PuMengYu/nnUNet_workspace/results_v2/ExternalVal_IRCADb/MoE_SizeOV2/report_custom.txt) |
| 20 | DeepResGN_MLA | 0.7551 | 0.9577 | 0.5526 | 0.5933 | 0.7088 | 60% (3/5) | 2/3 | 0.7969 | -0.0418 | [report](/home/PuMengYu/nnUNet_workspace/results_v2/ExternalVal_IRCADb/DeepResGN_MLA/report_custom.txt) |
| 21 | DeepPlainResGN | 0.7442 | 0.9584 | 0.5300 | 0.5775 | 0.6646 | 60% (3/5) | 3/2 | 0.7966 | -0.0524 | [report](/home/PuMengYu/nnUNet_workspace/results_v2/ExternalVal_IRCADb/DeepPlainResGN/report_custom.txt) |
| 22 | SwinUNETR | 0.7385 | 0.9543 | 0.5226 | 0.6034 | 0.6052 | 80% (4/5) | 3/0 | 0.7846 | -0.0461 | [report](/home/PuMengYu/nnUNet_workspace/results_v2/ExternalVal_IRCADb/SwinUNETR/report_custom.txt) |
| 23 | NoMirror | 0.5512 | 0.7766 | 0.3257 | 0.3541 | 0.4616 | 80% (4/5) | 7/2 | 0.8133 | -0.2621 | [report](/home/PuMengYu/nnUNet_workspace/results_v2/ExternalVal_IRCADb/NoMirror/report_custom.txt) |
| 24 | SizeOV3_NoMirror | 0.5444 | 0.7668 | 0.3221 | 0.3267 | 0.4574 | 60% (3/5) | 6/4 | 0.8120 | -0.2676 | [report](/home/PuMengYu/nnUNet_workspace/results_v2/ExternalVal_IRCADb/SizeOV3_NoMirror/report_custom.txt) |
| 25 | DWSepRes4_MoE_SizeOV4 | 0.3645 | 0.7290 | 0.0000 | 0.0000 | 0.0000 | 0% (0/5) | 15/0 | N/A | N/A | [report](/home/PuMengYu/nnUNet_workspace/results_v2/ExternalVal_IRCADb/DWSepRes4_MoE_SizeOV4/report_custom.txt) |

---

## 内部 Test Overall 排名（Top 15）

| Rank | Trainer | Overall | Liver | Tumor | Recall | Precision | FP率 | 严重/改进 | Report |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | MedNeXt_SizeOV4 | 0.8431 | 0.9545 | 0.7317 | 0.7361 | 0.8187 | 33% (1/3) | 1/5 | [report](/home/PuMengYu/nnUNet_workspace/results_v2/Dataset003_Liver/nnUNetTrainer_MedNeXt_SizeOV4__nnUNetPlans__3d_fullres/fold_0/test_report_custom.txt) |
| 2 | MedNeXt | 0.8402 | 0.9521 | 0.7283 | 0.7334 | 0.8128 | 33% (1/3) | 1/5 | [report](/home/PuMengYu/nnUNet_workspace/results_v2/Dataset003_Liver/nnUNetTrainer_MedNeXt__nnUNetPlans__3d_fullres/fold_0/test_report_custom.txt) |
| 3 | MLAUNet_MoE_SizeOV4 | 0.8330 | 0.9514 | 0.7146 | 0.7140 | 0.7776 | 33% (1/3) | 1/6 | [report](/home/PuMengYu/nnUNet_workspace/results_v2/Dataset003_Liver/nnUNetTrainer_MLAUNet_MoE_SizeOversampleV4__nnUNetPlans__3d_fullres/fold_0/test_report_custom.txt) |
| 4 | MedNeXt_MLA_FPSafe | 0.8326 | 0.9510 | 0.7143 | 0.7171 | 0.7741 | 33% (1/3) | 1/4 | [report](/home/PuMengYu/nnUNet_workspace/results_v2/Dataset003_Liver/nnUNetTrainer_MedNeXt_MLA_FPSafe__nnUNetPlans__3d_fullres/fold_0/test_report_custom.txt) |
| 5 | MedNeXt_MLA_SizeOV4 | 0.8285 | 0.9529 | 0.7040 | 0.7459 | 0.7775 | 67% (2/3) | 1/4 | [report](/home/PuMengYu/nnUNet_workspace/results_v2/Dataset003_Liver/nnUNetTrainer_MedNeXt_MLA_SizeOV4__nnUNetPlans__3d_fullres/fold_0/test_report_custom.txt) |
| 6 | MedNeXt_MLA | 0.8259 | 0.9535 | 0.6982 | 0.7323 | 0.7921 | 67% (2/3) | 1/5 | [report](/home/PuMengYu/nnUNet_workspace/results_v2/Dataset003_Liver/nnUNetTrainer_MedNeXt_MLA__nnUNetPlans__3d_fullres/fold_0/test_report_custom.txt) |
| 7 | DeepDWIBResGN | 0.8198 | 0.9494 | 0.6902 | 0.7354 | 0.7334 | 67% (2/3) | 1/5 | [report](/home/PuMengYu/nnUNet_workspace/results_v2/Dataset003_Liver/nnUNetTrainer_DeepDWIBResGN__nnUNetPlans__3d_fullres/fold_0/test_report_custom.txt) |
| 8 | MLAUNet_MoE_IB7_SizeOV4 | 0.8192 | 0.9528 | 0.6857 | 0.7148 | 0.7509 | 67% (2/3) | 1/6 | [report](/home/PuMengYu/nnUNet_workspace/results_v2/Dataset003_Liver/nnUNetTrainer_MLAUNet_MoE_IB7_SizeOversampleV4__nnUNetPlans__3d_fullres/fold_0/test_report_custom.txt) |
| 9 | SizeOV2 | 0.8187 | 0.9516 | 0.6858 | 0.7105 | 0.7490 | 67% (2/3) | 1/6 | [report](/home/PuMengYu/nnUNet_workspace/results_v2/Dataset003_Liver/nnUNetTrainer_SizeOversampleV2__nnUNetPlans__3d_fullres/fold_0/test_report_custom.txt) |
| 10 | MLA_GK5_V4 | 0.8173 | 0.9535 | 0.6811 | 0.7084 | 0.7622 | 67% (2/3) | 1/5 | [report](/home/PuMengYu/nnUNet_workspace/results_v2/Dataset003_Liver/nnUNetTrainer_MLA_GK5_V4__nnUNetPlans__3d_fullres/fold_0/test_report_custom.txt) |
| 11 | MoE_SizeOV5 | 0.8167 | 0.9499 | 0.6835 | 0.7141 | 0.7816 | 67% (2/3) | 1/6 | [report](/home/PuMengYu/nnUNet_workspace/results_v2/Dataset003_Liver/nnUNetTrainer_MLAUNet_MoE_SizeOversampleV5__nnUNetPlans__3d_fullres/fold_0/test_report_custom.txt) |
| 12 | MoE | 0.8166 | 0.9506 | 0.6826 | 0.7043 | 0.7554 | 67% (2/3) | 1/5 | [report](/home/PuMengYu/nnUNet_workspace/results_v2/Dataset003_Liver/nnUNetTrainer_MLAUNet_MoE__nnUNetPlans__3d_fullres/fold_0/test_report_custom.txt) |
| 13 | MoE_SizeOV2 | 0.8152 | 0.9516 | 0.6788 | 0.7113 | 0.7723 | 67% (2/3) | 1/6 | [report](/home/PuMengYu/nnUNet_workspace/results_v2/Dataset003_Liver/nnUNetTrainer_MLAUNet_MoE_SizeOversampleV2__nnUNetPlans__3d_fullres/fold_0/test_report_custom.txt) |
| 14 | MLAUNet | 0.8148 | 0.9503 | 0.6793 | 0.7104 | 0.7427 | 67% (2/3) | 1/5 | [report](/home/PuMengYu/nnUNet_workspace/results_v2/Dataset003_Liver/nnUNetTrainer_MLAUNet__nnUNetPlans__3d_fullres/fold_0/test_report_custom.txt) |
| 15 | SizeOV3 | 0.8143 | 0.9513 | 0.6774 | 0.7228 | 0.7641 | 67% (2/3) | 1/6 | [report](/home/PuMengYu/nnUNet_workspace/results_v2/Dataset003_Liver/nnUNetTrainer_SizeOversampleV3__nnUNetPlans__3d_fullres/fold_0/test_report_custom.txt) |

---

## MedNeXt 消融对比

| Method | 内部Overall | 外部Overall | 外-内 | 内部Tumor | 外部Tumor | 外部Recall | 外部Precision | 外部FP率 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MedNeXt | 0.8402 | 0.7705 | -0.0697 | 0.7283 | 0.5750 | 0.6554 | 0.6564 | 60% |
| MedNeXt_SizeOV4 | 0.8431 | 0.7797 | -0.0634 | 0.7317 | 0.5943 | 0.6500 | 0.6795 | 60% |
| MedNeXt_MLA | 0.8259 | **0.8079** | **-0.0180** | 0.6982 | **0.6484** | **0.6665** | **0.7437** | **40%** |
| MedNeXt_MLA_SizeOV4 | 0.8285 | 0.7870 | -0.0415 | 0.7040 | 0.6091 | 0.6580 | 0.7019 | 60% |
| MedNeXt_MLA_FPSafe | 0.8326 | 0.7744 | -0.0582 | 0.7143 | 0.5852 | 0.6442 | 0.6711 | 60% |

简要结论：

- 外部 Overall 当前第一是 `MedNeXt_MLA`：0.8079。
- `MedNeXt_MLA_FPSafe` 内部 FP 控制变好，但外部退化，不能作为主线。
- `DWSepRes4_MoE_SizeOV4` 外部 Tumor Dice 为 0，应标记为无效实验，不纳入有效结论。
