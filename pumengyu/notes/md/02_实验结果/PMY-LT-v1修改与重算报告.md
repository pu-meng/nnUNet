# PMY-LT-v1 修改与重算报告

> 执行日期：2026-07-18  
> 指标标准：`PMY-LT-v1`  
> 目的：固定肝脏/肿瘤分割的主指标口径，避免 GT 无肿瘤病例的 0/1/NaN 约定改变 Tumor Dice 分母、方法排名和消融结论。

## 1. 以后必须遵守的原则

### 1.1 固定评估总体

- `Liver Dice`：在固定测试集的全部病例上求算术平均。
- `Tumor Dice/Jaccard/Recall/FNR/Precision/FDR`：只在 GT 有肿瘤病例上求算术平均。
- GT 无肿瘤病例的 Tumor 分割指标恒为 `N/A`，不允许根据预测是否为空动态记为 1、0 或 NaN。

### 1.2 固定 Overall 定义

```text
Overall = (mean Liver Dice on all fixed-test cases
           + mean Tumor Dice on GT-positive cases) / 2
```

- Overall 是两个类别级均值的平均，不是逐病例 Overall 的平均。
- Dataset003、IRCADb 和 HCC 必须分域计算，禁止将两个外部数据集再求平均。

### 1.3 阴性肿瘤误报独立报告

GT 无肿瘤病例不进入 Tumor Dice，但必须独立报告：

- 无肿瘤病例 FP 率；
- 误报病例清单和预测肿瘤体素数；
- Tumor FPV 总量与每例均值；
- 假阳性连通域数量和体积。

这样可以避免同一个阴性误报先作为 Tumor Dice=0 惩罚，又可能通过 Liver Dice 影响 Overall。独立 FP 指标只用于补充描述安全性，不再混入 Tumor Dice 或 Overall 的分母。

### 1.4 区分论文主指标与 nnUNet 追溯指标

- nnUNet 生成的原始 `summary.json` 保留不变，作为工具兼容和原始评估证据。
- nnUNet `foreground_mean` 只能写在“追溯参考”区域。
- 论文表格、方法排名、消融差值和文字结论必须使用 `PMY-LT-v1`。

### 1.5 单一代码源

所有新的内部和外部报告必须调用：

```python
aggregate_liver_tumor_metrics(...)
```

禁止在其他报告脚本中再手写一套 `TN -> NaN` / `FP -> 0` / `TN -> 1` 的聚合逻辑。修改标准时必须先修改这一共享函数及测试，再重生成报告和汇总文档。

## 2. 这次修了哪些代码

| 代码 | 修改内容 |
|---|---|
| `pumengyu/tools/analyasis/metric_standard.py` | 新增唯一 PMY-LT-v1 聚合函数，同时返回论文主指标和 nnUNet 追溯参考 |
| `pumengyu/tools/analyasis/test_metric_standard.py` | 新增 3 个合成测试，固定阴性 FP 不改变主 Tumor Dice、Overall 公式和全阳性数据集兼容性 |
| `pumengyu/tools/analyasis/eval_fold_report.py` | 内部报告改用共享聚合函数；阴性 Tumor Dice 改为 N/A；单列 nnUNet 参考 |
| `pumengyu/ext_val/03_gen_method_report.py` | IRCADb/HCC 外部报告改用共享聚合函数；终端摘要也改为主口径 |
| `pumengyu/ext_val/02_eval_ircadb.py` | 修正旧 IRCADb 脚本中阴性 TN=1、FP=0 的旧约定，主 Tumor Dice 改为 positive-only |
| `pumengyu/analysis/metrics.py` | 分层统计模块的 Tumor Dice 改为 positive-only，增加 nnUNet 参考字段 |
| `pumengyu/analysis/report.py` | 报告标签改为 PMY-LT-v1 Tumor Dice，不再将它误写成 Overall |
| `pumengyu/tools/analyasis/batch_regen_reports.py` | 恢复“默认生成可视化”；只有显式 `--report-only` 才不生成新 PNG |
| `pumengyu/tools/analyasis/refresh_pmy_lt_v1_reports.py` | 新增仅复用现有 summary/报告的批量重算工具，不运行推理、不删预测、不删可视化 |

`05_gen_hcc_test_report.py` 动态复用 `03_gen_method_report.py`，因此 HCC 也已统一到同一聚合函数，不再维护第三套指标逻辑。

## 3. “实验数据修了吗”的准确回答

### 3.1 没有修改的原始证据

以下内容没有改动：

- 预测 NIfTI；
- GT 标签和 CT 数据；
- checkpoint；
- nnUNet 原始 `summary.json`；
- 已有 PNG 可视化；
- FPV/FNV、连通域和 per-case 原始统计段。

因此这次不是修改模型预测或篡改实验结果，而是用统一分母重算派生主指标。

### 3.2 已经重算和写回的派生数据

| 数据域 | 写回报告 | 重算来源 | 说明 |
|---|---:|---|---|
| Dataset003 internal | 29 份 | 27 份从 `summary.json`；NoMirror/SizeOV3 从旧报告恢复 | 包含 27 个公平方法和 2 个 stage 报告 |
| IRCADb | 31 份 | 现存 `summary.json` | 28 个 source-only 公平方法 + 3 个 HCC 训练/适配方法 |
| HCCReferencedCT v2 | 28 份 | 现存 `summary.json` | 21 例全部为 GT 肿瘤阳性，数值不变，仅统一口径声明 |
| **合计** | **88 份** | 不重新推理 | 报告主指标段均已标记 `PMY-LT-v1` |

Internal 的 `NoMirror` 和 `SizeOV3` 历史目录已缺失预测 NIfTI 和 `test_prediction/summary.json`，只能由旧报告中 23 个阳性病例的四位小数 per-case 指标恢复，可能存在约 `0.0001` 的四舍五入误差。

## 4. 关键数字修正示例

### 4.1 Baseline vs NoMirror

| 指标 | Baseline | NoMirror | 正确结论 |
|---|---:|---:|---|
| Liver Dice（all 26） | 0.9340 | 0.9581 | NoMirror 上升 0.0241 |
| Tumor Dice（positive 23） | 0.7395 | 0.7267 | NoMirror 下降 0.0128 |
| Overall（PMY-LT-v1） | 0.8368 | 0.8424 | NoMirror 小幅上升 0.0056，由 Liver 提升驱动 |
| 无肿瘤 FP 率 | 3/3 | 2/3 | NoMirror 减少 1 个阴性误报，但 liver_89 误报体素明显增加 |

因此，现在“四个肿瘤大小组都下降”与“总 Tumor Dice 下降”完全一致，旧报告中总 Tumor Dice 反而上升的动态分母矛盾已消除。

### 4.2 新口径下的三域第 1

| 数据域 | 方法 | Tumor Dice | Overall |
|---|---|---:|---:|
| Dataset003 internal | MedNeXt_MLA_MoE_SizeOV4 | 0.7653 | 0.8591 |
| 3D-IRCADb | MedNeXt_MLA_MoE | 0.7349 | 0.8511 |
| HCCReferencedCT v2 | MedNeXt_MLA | 0.4645 | 0.6532 |

IRCADb 和 internal 的排名已重排；HCC 因 21 例全部为阳性，主指标数值不变。

## 5. 修改了哪些报告与汇总

| 文件 | 定位 |
|---|---|
| `指标统计口径.md` | PMY-LT-v1 正式定义、分母、边界情况和历史例外 |
| `三个数据集全Trainer实验结果与消融汇总.md` | 28 种公平方法三域总表、新排名、消融差值和产物状态 |
| `MedNeXt系列消融实验结果汇总.md` | MedNeXt/MHA/MLA/MoE/SizeOV4/FP-Safe 主线结论 |
| `Mirror消融分析.md` | Baseline vs NoMirror 主指标、大小分组与结论统一 |
| `README.md` | 只保留索引和当前口径，不重复维护第三份排名 |

## 6. 验证结果

- 3 个 PMY-LT-v1 合成测试全部通过。
- 28 行三域总表与当前磁盘报告逐项比较，不一致数为 0。
- 内部 29 份、IRCADb 31 份、HCC 28 份报告均包含 `PMY-LT-v1` 主指标段。
- 相关 Python 文件语法检查通过，相关文档 `git diff --check` 通过。
- 重算过程没有运行 GPU 推理，没有删除任何预测或 PNG。

## 7. 仍然存在的产物缺口

按当前 `AGENTS.md` 严格完成标准：

- Dataset003 internal：27 个历史公平方法的测试预测 NIfTI 已被删除；只有 5/27 有实际 PNG；NoMirror/SizeOV3 缺 summary，因此不能宣称完整交付。
- IRCADb：28 个公平方法中 27 个具备预测、summary、report 和 PNG；FP-Safe 的 PNG=0。
- HCCReferencedCT v2：28/28 个公平方法均有 21/21 预测、summary、report 和实际 PNG。
- 多数历史外部报告尚未将实际 checkpoint 文件名和来源完整写入报告，因此除已核验的 MHA/MLA 外，历史方法的 provenance 仍应标记为未完全验证。

## 8. 以后新实验的固定流程

```text
fixed case list
  -> predictions
  -> immutable nnUNet summary.json
  -> aggregate_liver_tumor_metrics (PMY-LT-v1)
  -> report_custom.txt / test_report_custom.txt
  -> report + summary + PNG + checkpoint provenance audit
  -> update the two canonical result ledgers
```

只有上述链条全部完成，才能将方法标记为“完成”并写入论文主排名。
