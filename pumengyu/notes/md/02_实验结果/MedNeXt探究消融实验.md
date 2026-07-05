# MedNeXt 探究消融实验

> 更新：2026-07-02  
> 数据源：`results_v2/Dataset003_Liver/*/test_report_custom.txt` 与 `results_v2/ExternalVal_IRCADb/*/report_custom.txt`

---

## 1. 核心结论

`MedNeXt_MLA` 是当前 MedNeXt 族的外部最优；`MedNeXt_MLA_FPSafe` 内部更好但外部失败。

| Variant | 内部 Overall | 外部 Overall | Δ外-内 | 内部 Tumor | 外部 Tumor | 内部 FP | 外部 FP |
|---|---:|---:|---:|---:|---:|---:|---:|
| MedNeXt | 0.8402 | 0.7705 | -0.0697 | **0.7283** | 0.5750 | 33% | 60% |
| MedNeXt_SizeOV4 | **0.8431** | 0.7797 | -0.0634 | **0.7317** | 0.5943 | 33% | 60% |
| MedNeXt_MLA | 0.8259 | **0.8079** | **-0.0180** | 0.6982 | **0.6484** | 67% | **40%** |
| MedNeXt_MLA_SizeOV4 | 0.8285 | 0.7870 | -0.0415 | 0.7040 | 0.6091 | 67% | 60% |
| MedNeXt_MLA_FPSafe | 0.8326 | 0.7744 | -0.0582 | 0.7143 | 0.5852 | 33% | 60% |

判断：

1. `MedNeXt_SizeOV4` 是内部最强，Overall 0.8431。
2. `MedNeXt_MLA` 是外部最强，Overall 0.8079，Tumor Dice 0.6484。
3. `MedNeXt_MLA_FPSafe` 把内部 FP 率从 67% 降到 33%，但外部降到 0.7744，说明 FP-safe 约束没有跨域泛化。
4. `MedNeXt_MLA_SizeOV4` 比 `MedNeXt_MLA` 差，说明 MLA 后继续叠 SizeOV4 不稳定。

---

## 2. FPSafe 结果解读

内部收益：

- Overall 0.8326，高于 `MedNeXt_MLA` 0.8259。
- Tumor Dice 0.7143，高于 `MedNeXt_MLA` 0.6982。
- 无肿瘤 FP 率 33%，优于 `MedNeXt_MLA` 的 67%。

外部问题：

- Overall 0.7744，低于 `MedNeXt_MLA` 0.8079。
- Tumor Dice 0.5852，低于 `MedNeXt_MLA` 0.6484。
- Precision 0.6711，低于 `MedNeXt_MLA` 0.7437。
- 无肿瘤 FP 率 60%，没有优于 MedNeXt 原生路线。

结论：FPSafe 可以作为“同域 FP 控制有效、跨域泛化失败”的 ablation。它不适合作为最终主模型。

---

## 3. 论文叙事

推荐写法：

- MedNeXt 是强同域 baseline：`MedNeXt_SizeOV4` 内部 Overall 最高。
- MLA 是跨域泛化模块：`MedNeXt_MLA` 外部第一，且 drop 明显变小。
- FPSafe 是负/中性消融：内部 FP 改善，但外部泛化退化，说明单纯 FP 抑制不足以解决跨域误报。

主线不要写成“FPSafe 进一步提升 MedNeXt”，当前数据不支持。
