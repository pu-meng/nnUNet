# 会话记录 & Claude 上下文引导文件

> **固定路径**：`pumengyu/notes/session_log.md`  
> 每次关机前更新，下次开机把这个文件发给 Claude，说"继续"即可恢复上下文。

---

## ▶ Claude 读取指南（每次会话开始时先读这里）

你是我的科研助手，帮我做医学图像分割方向的论文实验。  
读完本文件后，**不要复述文件内容**，直接说"已恢复上下文，上次停在 [停在哪里]，继续吗？"然后等我指令。

### 项目基本信息
- **任务**：肝脏肿瘤分割，数据集 LiTS（MSD Task03_Liver，131 case，13 个无肿瘤）
- **框架**：nnUNet v2，3d_fullres，fold_0 为主力实验折
- **核心问题**：无肿瘤 case 大量假阳性（FP），体积阈值后处理在本数据集上理论无解
- **当前方案**：两阶段 FP 抑制（Stage1 只训有肿瘤 case → Stage2 用 Stage1 概率图作为额外输入通道）
- **参考论文**：`/home/PuMengYu/FP_Reduction_Network.pdf`（Peng et al. MICCAI 2022）
- **论文草稿**：`pumengyu/notes/paper/draft_v1.md`
- **方案计划书**：`pumengyu/notes/md/两阶段FP抑制方案.md`

### 工作目录结构
```
/home/PuMengYu/nnUNet/               ← 主工作目录（cd 到这里）
  pumengyu/
    trainers/trainer.py               ← 所有 Trainer 类定义
    mixins.py                         ← 所有 Mixin 实现
    tools/analyasis/                  ← 分析脚本
    notes/
      paper/draft_v1.md               ← 论文草稿
      md/两阶段FP抑制方案.md           ← 实验计划书
      sh/                             ← 运行脚本
      session_log.md                  ← 本文件

/home/PuMengYu/nnUNet_workspace/
  preprocessed/Dataset003_Liver/gt_segmentations/   ← GT 标注
  raw/Dataset003_Liver/imagesTr/                    ← 原始图像
  results_v2/Dataset003_Liver/                      ← 实验结果
```

### 我的偏好（Claude 行为规范）
- 回复用**中文**，简洁，不要复述我说的话
- **不要擅自运行 Python 脚本**，写好后告诉我命令，由我自己执行
- 读文件、grep、ls 等只读操作直接做，不用问我
- 编辑文件前说明改什么，不要大段改动后才告诉我
- 写代码前如果涉及不确定的 API，先提醒我验证
- 论文相关内容要严谨，数字来源要注明是哪个报告文件

### 常用路径变量（脚本里常用）
```bash
GT=/home/PuMengYu/nnUNet_workspace/preprocessed/Dataset003_Liver/gt_segmentations
IMG=/home/PuMengYu/nnUNet_workspace/raw/Dataset003_Liver/imagesTr
V2=/home/PuMengYu/nnUNet_workspace/results_v2/Dataset003_Liver
```

---

## 已归档的重要发现（从旧分析文件提取）

### HU 分析关键结论（来自旧 hu_analysis.txt，5折全集）

**不可解决的硬骨头 case：**
| case | 原因 | 关键数据 |
|------|------|---------|
| liver_116 | 等密度肿瘤，HU 重叠率 90.7% | contrast=-8.9 HU，任何模型理论失败 |
| liver_39  | 等密度大肿瘤，Dice=0 | contrast=0.0 HU，overlap=88% |
| liver_43  | 肝脏 HU=28.7（疑似脂肪肝/非增强CT） | 正常肝脏应 50-80 HU，异常采集 case |
| liver_127 | 298 体素，极微小 | 占肝脏体积 0.01%，标注可靠性存疑 |

**最重要结论（可写入论文）：**
> Dice<0.3 组的 HU 重叠率均值 57.5% vs Dice>0.7 组 57.9%——几乎相同。
> 整体 HU 重叠度不是失败的决定性因素，真正不可解决的是极端等密度（liver_116/39）和极微小（liver_127）。

### 旧两阶段方案（twostage/，已放弃）
- **架构**：先裁肝脏 ROI → 再分割肿瘤（和现在的 FP 抑制两阶段完全不同）
- **失败原因**：Stage1 裁剪框偏差导致 Stage2 推理时 distribution gap，BboxJitterMixin 未能解决
- **与现方案的区别**：现方案是 Stage1 产生 FP 概率图 → Stage2 抑制无肿瘤 case 误报

---

## 会话记录

---

### 2026-06-04

**本次做了什么**

1. `Tr_Stage2_FPSup` 500 epoch 训练完成（26.4h），验证集结果（n=13）：

   | 指标 | Stage1 (验证集) | Stage2 (验证集) | 说明 |
   |------|:--------------:|:--------------:|------|
   | 肝脏 Dice | 0.9464 | **0.1148** | ⚠️ 异常低 |
   | 肿瘤 Dice | 0.6454 | 0.6401 | 几乎无提升 |
   | Overall | 0.7959 | 0.3775 | Stage2 被肝脏拖垮 |

2. **问题根因分析**：Stage2 的 `_Stage2LossWrapper` 只优化肿瘤通道（MSE+BCE），肝脏通道从未被 Loss 约束，导致肝脏 Dice=0.11（接近随机）。而肝脏分割对肿瘤有强解剖约束（肿瘤必须在肝脏内），网络连肝脏都不认识，无法通过上下文区分真假肿瘤，Stage2 效果与 Stage1 持平。

3. **已修复**（`pumengyu/mixins.py`）：
   - `Stage2FPSupMixin._build_loss()` 改为调用 `super()._build_loss()`（标准 Dice+CE，肝脏+肿瘤同时优化）
   - 删去 `_Stage2LossWrapper` 的使用（类定义保留备查）

4. **加速优化**（`pumengyu/mixins.py` + `pumengyu/trainers/trainer.py`）：
   - 新增 `Stage2FPSupMixin._load_stage1_weights()`：从 Stage1 `checkpoint_final.pth` 热启动，第一层 conv 从 1 通道扩展到 3 通道（新 2 通道初始化为 0），网络跳过重新学肝脏/肿瘤的阶段
   - `Tr_Stage2_FPSup.num_epochs` 从 500 → **250**（热启动后 250 epoch 预计足够，节省约 13h）

5. **DDP 双卡**：第二块 4090D 空闲，可用双卡训练进一步 ~2x 加速（见下次继续命令）

**停在哪里**

Stage2 v1（Loss 有误）已训完，结果不理想。代码已修复，等待重新训练。

**下次继续**

1. 从头重训 Stage2（修复版，热启动 Stage1 权重，250 epoch）：
   ```bash
   # 单卡
   CUDA_VISIBLE_DEVICES=0 nnUNetv2_train Dataset003_Liver 3d_fullres 0 -tr Tr_Stage2_FPSup

   # 双卡 DDP（~2x 加速，推荐）
   CUDA_VISIBLE_DEVICES=0,1 nnUNetv2_train Dataset003_Liver 3d_fullres 0 -tr Tr_Stage2_FPSup
   ```
2. 观察训练日志开头，确认两行日志正确输出：
   - `[Stage2Init] 扩展 ...: [32, 1, 3, 3, 3] → [32, 3, 3, 3, 3]`
   - `[Stage2FPSup] Loss = Dice+CE (liver+tumor，与 nnUNetTrainer 一致)`
3. 训练完成后对比验证集：肝脏 Dice 应回升到 0.90+，肿瘤 Dice 应超过 Stage1 (0.6454)，无肿瘤误报率目标 0%

---

### 2026-06-03

**本次做了什么**

1. `SizeOversampleV3_NoMirror` 训练完成，收集并对比了所有对照实验测试集结果（n=26）：

   | Trainer | 肝脏Dice | 肿瘤Dice | Overall | 无肿瘤误报率 |
   |---------|:--------:|:--------:|:-------:|:-----------:|
   | Baseline | 0.9340 | 0.6542 | 0.7941 | 100% (3/3) |
   | NoMirror | 0.9581 | 0.6685 | 0.8133 | 66.7% |
   | SizeOversampleV2 | 0.9516 | **0.6858** | **0.8187** | 66.7% |
   | SizeOversampleV3 | 0.9513 | 0.6774 | 0.8143 | 66.7% |
   | SizeOversampleV3_NoMirror | **0.9591** | 0.6649 | 0.8120 | 66.7% |
   | Stage1_TumorOnly | 0.9502 | 0.6592 | 0.8047 | 100%（符合预期）|

   **关键结论**：NoMirror 去掉镜像后肝脏 Dice 大幅回升（0.9340→0.9591）；**SizeOversampleV2 综合最优**（Overall 0.8187，Tumor Dice 0.6858），V3 + NoMirror 组合并未带来额外提升。论文对比 baseline 仍选 SizeOversampleV2。

2. `Tr_Stage2_FPSup` 已在运行，epoch 45 时中断（log 文件：`training_log_2026_6_2_23_53_59.txt`，368 行），有 `checkpoint_best.pth`，需 `--c` 续训。

3. 确认 Stage2 续训命令正确（`nnUNet_results` 已全局指向 `results_v2`，无需额外设 `RESULTS_FOLDER`）：
   ```bash
   CUDA_VISIBLE_DEVICES=1 \
   nnUNet_n_proc_DA=6 \
   nnUNetv2_train 3 3d_fullres 0 -tr Tr_Stage2_FPSup --c
   ```

4. 规划了 log 文件清理方案（待执行）：删除 0 行 + 短暂中断日志（SizeOversampleV2/V3/Stage2 各1-2个），保留主训练 log。

**停在哪里**

Stage2 训练中断在 epoch 45，待 `--c` 续训完成。

**下次继续**

1. 确认 Stage2 训练完成（`checkpoint_final.pth` 存在）：
   ```bash
   ls /home/PuMengYu/nnUNet_workspace/results_v2/Dataset003_Liver/Tr_Stage2_FPSup__nnUNetPlans__3d_fullres/fold_0/
   ```
2. Stage2 推理 + 评测（`AutoInternalTestMixin` 应自动触发，若没有则手动跑 report）
3. 与 SizeOversampleV2（Overall 0.8187, FP 误报率 66.7%）对比 Stage2 效果

---

### 2026-05-31

**本次做了什么**

1. 确认 results_v2 中两个中止实验的状态：
   - `nnUNetTrainer_NoMirror`：epoch 0 前中止，无 checkpoint，需从头训
   - `Tr_Stage1_TumorOnly`：epoch 680 中止，有 `checkpoint_latest.pth`，续训完成（✅）

2. 系统分析 4 个已完成实验的测试集结果（n=26）：

   | Trainer | 肿瘤Dice | Overall | 无肿瘤误报率 |
   |---------|:--------:|:-------:|:-----------:|
   | Baseline | 0.6542 | 0.7941 | 100% |
   | SizeOversampleV2 | 0.6858 | **0.8187** | 66.7% |
   | SizeOversampleV3 | 0.6774 | 0.8143 | 66.7% |
   | Stage1_TumorOnly | 0.6592 | 0.8047 | 100%（符合预期） |

3. 全面梳理并更新了两个文档：
   - `pumengyu/notes/md/实验配置记录.md`：补全 Mixin 速查表、所有未跑实验列表、测试集+验证集结果双表
   - `pumengyu/notes/md/两阶段FP抑制方案.md`：Stage1 标记已完成，步骤进度表更新，Step 2 命令写好

4. 写了 Stage1 推理脚本：`pumengyu/notes/sh/stage1_predict.sh`

5. 开始执行 Stage1 全量推理（`--save_probabilities`），运行到一半中止后重启

**停在哪里**

Stage1 推理（`stage1_predict.sh`）**正在运行**，对 131 个 case 保存 softmax 概率图到 `stage1_softmax/`。

**下次继续**

1. 确认推理完成（131 个 `.npz` + `.nii.gz`）：
   ```bash
   ls /home/PuMengYu/nnUNet_workspace/results_v2/Dataset003_Liver/Tr_Stage1_TumorOnly__nnUNetPlans__3d_fullres/fold_0/stage1_softmax/ | wc -l
   # 期望：262（每 case 1 个 .nii.gz + 1 个 .npz）
   ```
2. 启动 NoMirror 训练（可与 Stage2 实现并行）：
   ```bash
   RESULTS_FOLDER=/home/PuMengYu/nnUNet_workspace/results_v2 nnUNetv2_train 3 3d_fullres 0 -tr nnUNetTrainer_NoMirror
   ```
3. 实现 `Tr_Stage2_FPSup`（`pumengyu/trainers/trainer.py` + `pumengyu/mixins.py`）：
   - 3 通道输入（CT + Stage1 prob + Stage1 binary）
   - Loss：MSE + CE
   - 无肿瘤 case 过采样 3×

---

### 2026-05-30

**本次做了什么**

1. 跑完 `regen_reports_v2.sh`，拿到 Baseline / SizeOversampleV2 / SizeOversampleV3 三个模型的 test 报告（26 case）

2. 读了参考论文 `FP_Reduction_Network.pdf`，发现两处改进点并更新计划书：
   - Stage2 输入：2通道 → **3通道**（CT + Stage1概率图 + Stage1二值图）
   - Stage2 Loss：Dice+CE → **MSE+CE**（对 FP 区域惩罚更直接）

3. CC 分析发现并写入论文：min TP CC = 1 体素，max 无肿瘤假CC ≈ 30k 体素，**体积阈值在 LiTS 上理论无解**（已加入论文 §4.4，推翻了 §5.4 原有建议）

4. 写了 CC 分析脚本：`pumengyu/tools/analyasis/cc_dataset_analysis.py`（separability_gap 指标，可推广到其他数据集）

5. 实现 Stage1 Trainer：`TumorOnlyTrainMixin`（mixins.py 末尾）+ `Tr_Stage1_TumorOnly`（trainer.py）

**停在哪里**

Stage1 代码写完，**尚未训练**。CC 分析脚本待运行（或运行中）。

**下次继续**

1. 确认 CC 分析结果，把数字补入论文 §4.4
2. 训练 Stage1：
   ```bash
   CUDA_VISIBLE_DEVICES=0 nnUNetv2_train Dataset003_Liver 3d_fullres 0 -tr Tr_Stage1_TumorOnly
   ```
3. Stage1 训完后跑推理（`--save_probabilities`）
4. 实现 Stage2 dataloader（等概率图落盘后再写）

---

<!-- 新会话记录追加在上方，格式如下：

### YYYY-MM-DD

**本次做了什么**
（简要列出，3-8 条）

**停在哪里**
（一句话，精确到哪个步骤）

**下次继续**
（列出具体命令或任务，可直接执行）

-->
