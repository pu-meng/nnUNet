# 目录
[toc]
# pumengyu/tools — 脚本说明

本目录存放用于实验分析和报告生成的工具脚本，均针对 `Dataset003_Liver` 的 nnUNet 训练结果。

---

## 文件列表

### `eval_fold_report.py`
从 `fold_X/validation/summary.json` 读取指标，生成两类输出：
- `report_custom.txt`：对齐格式的文本报告，有/无肿瘤分开统计，按 tumor_dice 分级（严重失败 / 需要改进 / 没问题）
- `vis_png_custom/`：每个 case 的轴向切片可视化（GT / Pred / Diff 对比图）

这个是因为当时没有report.txt不想要重新实验,我们做的`eval_fold_json.py`脚本




---

### `gen_report_json.py`
从 `fold_X/validation/summary.json` 生成结构化的 `report_custom.json`，便于程序化读取各 case 的详细指标。

**用法：**
```bash
python pumengyu/tools/gen_report_json.py \
  --fold_dir <results/.../fold_X>
```

**输出：** `fold_dir/report_custom.json`

---

### `analyze_hu_failure.py`
综合分析各 fold 验证集中肿瘤的 HU 分布、大小、预测误差与 Dice 的关联，定位分割失败的根本原因。输出到 `pumengyu/notes/hu_analysis.txt`。

**用法：**
```bash
python pumengyu/tools/analyze_hu_failure.py
```

---

### `analyze_hard_cases.py`
对**全部 5 个 fold** 的验证集 case 进行逐 case HU 强度深度分析，重点关注指标差的困难样本（如 liver_121 漏检、liver_43 误报）。分析内容包括：

- 肝脏 / 肿瘤 / 背景 HU 均值、标准差、p5-p95 分位
- 肿瘤 vs 肝脏 HU 重叠率（量化"等密度"程度）
- 漏检(FN) vs 命中(TP) 体素的 HU 对比
- 全数据集各 case 肿瘤 HU 排序概览

**输出：** `pumengyu/notes/hard_cases_analysis.txt`

**用法：**
```bash
python pumengyu/tools/analyze_hard_cases.py
```
