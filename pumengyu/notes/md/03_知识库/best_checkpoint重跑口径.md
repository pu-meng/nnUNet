# best checkpoint 重跑口径

## 目的

为避免 `checkpoint_best.pth` 与 `checkpoint_final.pth` 混用造成论文口径不清，投稿前可用 `checkpoint_best.pth` 统一重跑内部测试和两个外部测试。

本次重跑只做推理和评估，不重新训练模型。

## 不覆盖旧结果

旧结果目录保持不动：

```text
/home/PuMengYu/nnUNet_workspace/results_v2/
```

best-only 重跑结果写入新目录：

```text
/home/PuMengYu/nnUNet_workspace/results_v2_best/
```

输出结构：

```text
results_v2_best/
  Dataset003_Liver/<Method>/test_report_custom.txt
  IRCADb/source_only/<Method>/report_custom.txt
  Dataset013_HCCReferencedCT/source_only/<Method>/report_custom.txt
```

## checkpoint 规则

所有方法均读取 Dataset003_Liver 训练得到的 best checkpoint：

```text
/home/PuMengYu/nnUNet_workspace/results_v2/Dataset003_Liver/<Trainer>__nnUNetPlans__3d_fullres/fold_0/checkpoint_best.pth
```

不使用 `checkpoint_final.pth`。

## 数据集规则

### 1. Dataset003_Liver internal test

内部测试只使用固定 26 例 test set：

```text
/home/PuMengYu/nnUNet_workspace/preprocessed/Dataset003_Liver/split_info_712.json
```

规则：

- train 92 例用于训练。
- val 13 例只用于训练监控和 checkpoint 选择。
- test 26 例用于论文 internal 指标。
- 不使用 validation 指标做横向对比。

### 2. 3D-IRCADb external validation

IRCADb 使用完整外部验证集：

```text
/home/PuMengYu/nnUNet_workspace/external_val/ircadb_full/
```

该数据集包含有肿瘤和无肿瘤病例，因此可报告 no-tumor FP rate。

### 3. HCCReferencedCT v2 held-out test

HCC 只使用我们固定划分中的 test 21 例：

```text
/home/PuMengYu/nnUNet_workspace/preprocessed/Dataset013_HCCReferencedCT/split_info_701020_stratified_v2.json
```

代码读取规则：

```python
cases = split_info["test"]["cases"]
```

当前划分：

```text
train 70
val   10
test  21
```

HCC train/val 不参与外部测试结果的模型选择或评估。当前 HCC test 全部为有肿瘤病例，因此不报告 no-tumor FP rate。

## 重跑方法范围

脚本当前重跑正文主表涉及的方法：

```text
Baseline
SizeOV2
SizeOV3
MLAUNet
MoE_SizeOV5
SwinUNETR
nnFormer
DeepPlainResGN
DeepResGN_MLA
DeepDWIBResGN
DeepDWIBMedConfig
MedNeXt
MedNeXt_SizeOV4
MedNeXt_MLA
MedNeXt_MLA_SizeOV4
```

不包含未进入正文主表的探索 trainer。

## 执行命令

建议在 `tmux` 中运行，避免终端断开。

```bash
cd /home/PuMengYu/nnUNet
GPU=1 bash pumengyu/notes/sh/rerun_best_reports_gpu1.sh 2>&1 | tee /home/PuMengYu/nnUNet_workspace/results_v2_best/rerun_best_gpu1.log
```

如果不用 `tee`，直接运行：

```bash
cd /home/PuMengYu/nnUNet
GPU=1 bash pumengyu/notes/sh/rerun_best_reports_gpu1.sh
```

## 完成后检查

```bash
find /home/PuMengYu/nnUNet_workspace/results_v2_best -name "report_custom.txt" -o -name "test_report_custom.txt"
```

检查 HCC 是否只包含 21 个 test cases：

```bash
python - <<'PY'
import json
from pathlib import Path
p = Path('/home/PuMengYu/nnUNet_workspace/preprocessed/Dataset013_HCCReferencedCT/split_info_701020_stratified_v2.json')
d = json.loads(p.read_text())
for k in ['train', 'val', 'test']:
    print(k, len(d[k]['cases']))
PY
```

预期：

```text
train 70
val 10
test 21
```
