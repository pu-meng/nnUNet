# 目录
[toc]

# nnUNet 改进计划 — Dataset003_Liver

基于 MSD_LiverTumorSeg 两阶段实验经验，按优先级排序。

---

## 1. FocalTversky Loss ★★★★★

**依据：** 实验记录中从 DiceFocal(0.58) → FocalTversky(0.71)，最大单次跳跃。  
**原理：** `L = (1 - TP/(TP + α·FP + β·FN))^γ`，β=0.7 > α=0.3 加重 FN 惩罚，提升小肿瘤 recall。  
**参数：** alpha=0.3, beta=0.7, gamma=0.75  
**实现方式：** 继承 `nnUNetTrainer`，重写 `_build_loss()`，不动源码。  
**文件位置：** `pumengyu/nnUNetTrainer_FocalTversky.py`（待创建）

- [ ] 实现 `FocalTverskyLoss`（参考 `MSD_LiverTumorSeg/medseg_project/medseg/engine/train_eval.py`）
- [ ] 实现 `DC_and_FocalTversky_loss`（DiceCE weight=0.7 + FocalTversky weight=0.3）
- [ ] 继承 Trainer，注册新 loss
- [ ] 跑 fold_0 对比验证

---

## 2. 后处理 min_tumor_size ★★★

**依据：** 两阶段实验中 min_tumor_size=100 使 FDR 从 0.22 降到 0.10。  
**实现方式：** 在 `eval_fold_report.py` 或单独推理脚本里，对预测结果做连通域过滤，去掉体素数 < 100 的 cancer 预测。  
**注意：** 只过滤 cancer（class 2），不动 liver（class 1）。

- [ ] 在 `eval_fold_report.py` 加 `--min_tumor_size` 参数（默认 100）
- [ ] 统计过滤前后 FDR / Recall 变化

---

## 3. TTA（Test-Time Augmentation） ★★

**依据：** 8 方向 flip TTA 带来约 +1.5% recall，几乎免费。  
**实现方式：** nnUNet 推理命令加 `--tta`（已内置）。  
**当前状态：** 不确定现在推理是否开启，需确认。

- [ ] 确认 `nnUNetv2_predict` 默认是否开启 TTA
- [ ] 对比开启/关闭 TTA 的验证 Dice

---

## 4. Foreground Oversampling 比例调整 ★★

**依据：** 两阶段实验 tumor_ratios=0.95 → FP 爆炸；改 0.60 后 precision 大幅提升。  
nnUNet 默认 `oversample_foreground_percent=0.33`（比 0.60 还低），可能是 liver_21 等误报严重的原因。  
**实现方式：** 继承 Trainer，重写 `oversample_foreground_percent = 0.5`。

- [ ] 实验 oversample=0.33 vs 0.5 vs 0.6 的 Dice/FDR 对比
- [ ] 可以和 FocalTversky Trainer 合并在一个子类里

---

## 5. Small Tumor Zoom ★

**依据：** <5k voxel 极小肿瘤 zoom 3× 后裁 patch，让极小目标在 patch 里可见。  
**实现难度：** 需要修改 nnUNet DataLoader，改动较大。  
**暂缓**，先看 FocalTversky + oversampling 效果。

- [ ] 暂缓，待前几项实验出结果后评估是否必要

---

## 实验对比基线

| 配置 | Tumor Dice | Recall | Precision | 备注 |
|------|-----------|--------|-----------|------|
| nnUNet 默认 DiceCE fold_1 | 0.6257 | 0.6199 | 0.7008 | 当前基线 |
| 两阶段 DiceCE (test 19 cases) | 0.7549 | 0.7523 | 0.7924 | 参考上限 |
| FocalTversky Trainer | TBD | | | 待跑 |
| FocalTversky + oversample=0.5 | TBD | | | 待跑 |

> 注：nnUNet val=26 cases，两阶段 test=19 cases，划分不同，不能直接比较，仅参考趋势。

---

## 进度

- [x] 5-fold 训练脚本 `run_train_and_eval.sh`
- [x] 自动报告 + 可视化 `eval_fold_report.py`
- [ ] FocalTversky Trainer
- [ ] 后处理 min_tumor_size
- [ ] TTA 确认
- [ ] Oversampling 调整
