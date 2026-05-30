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
