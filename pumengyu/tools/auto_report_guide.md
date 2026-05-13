# auto_report 接入指南

## 改了哪些文件

```
pumengyu/
├── tools/
│   ├── gen_report_json.py      ← 改动：抽出函数
│   ├── eval_fold_report.py     ← 改动：抽出函数
│   └── auto_report.py          ← 新建：统一出口
└── trainers/
    ├── nnUNetTrainer_Exp.py         ← 改动：加了 perform_actual_validation
    └── nnUNetTrainer_FocalTversky.py ← 改动：加了 perform_actual_validation
```

---

## 每个文件改了什么

### gen_report_json.py
**看第 18 行**

之前 `main()` 把所有逻辑写在里面。  
现在把逻辑抽成了 `generate_report_json(fold_dir)` 函数，`main()` 只剩一行调用它。

```python
# 新增的函数（第 18 行开始）
def generate_report_json(fold_dir: Path) -> Path | None:
    ...

# main() 现在只有两行
def main():
    args = p.parse_args()
    generate_report_json(Path(args.fold_dir))  # 调用上面的函数
```

---

### eval_fold_report.py
**看第 155 行**（`run_eval_report` 函数定义处）

同样把 `main()` 的逻辑抽成了 `run_eval_report(val_dir, gt_dir, img_dir, ...)` 函数。  
`main()` 只负责解析命令行参数，然后调用它。

```python
# 新增的函数（第 155 行开始）
def run_eval_report(val_dir, gt_dir, img_dir, vis_slices=5, no_vis=False, min_tumor_size=0):
    ...

# main() 现在只负责 argparse，最后调用 run_eval_report()
```

---

### auto_report.py（新建）
**整个文件就一个函数 `run_auto_report()`**

它是唯一出口，内部按顺序调用上面两个函数：

```python
def run_auto_report(fold_dir, gt_dir, img_dir, ...):
    generate_report_json(fold_dir)       # 第一步
    run_eval_report(val_dir, gt_dir, ...) # 第二步
```

两步都套了 `try/except`，任何一步报错只打印错误，不会中断训练。

---

### nnUNetTrainer_Exp.py
**看第 57 行**

在 `__init__` 下面新增了一个方法：

```python
def perform_actual_validation(self, save_probabilities=False):
    super().perform_actual_validation(save_probabilities)  # 原有逻辑（生成 summary.json）
    if self.local_rank == 0:                               # DDP 多卡时只跑一次
        from pumengyu.tools.auto_report import run_auto_report
        run_auto_report(
            fold_dir=self.output_folder,
            gt_dir=...,   # preprocessed/gt_segmentations
            img_dir=...,  # raw/imagesTr
        )
```

---

### nnUNetTrainer_FocalTversky.py
**看第 157 行**

和 Exp trainer 完全一样的 `perform_actual_validation`，加在 `_build_loss()` 下面。

---

## 触发时机

```
nnUNetv2_train ... （训练完成）
        │
        └─ perform_actual_validation()    ← nnUNet 训练结束后自动调用
                │
                ├─ super()                → 生成 validation/summary.json
                │
                └─ run_auto_report()      → 生成以下三个文件：
                        │
                        ├─ fold_X/report_custom.json
                        ├─ fold_X/report_custom.txt
                        └─ fold_X/vis_png_custom/*.png
```

---

## shell 脚本还能用吗

能，`gen_report_json.py` 和 `eval_fold_report.py` 的 `main()` 都保留了，  
`gen_fold_report.sh` 不需要任何修改。
