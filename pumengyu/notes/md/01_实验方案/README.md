# 实验方案

## FP-Safe 肝肿瘤分割计划书


#### FP-Safe 肝肿瘤分割计划书

日期：2026-06-28

##### 一、问题定义

当前实验显示，单纯追求内部测试集 Dice 不足以支撑论文主线：

- Baseline 在内部测试中 Recall 高，但无肿瘤 case 误报严重。
- MedNeXt / MedNeXt_SizeOV4 内部测试最强，但 IRCADb 外部验证 drop 明显。
- 极小肿瘤仍是共同难点，简单连通域阈值后处理不可靠，因为真实 TP 连通域可能小到 1 体素。

因此新主线从“刷最高 Dice”调整为：

> 临床安全的肝肿瘤分割：降低无肿瘤误报、保持小肿瘤召回、提升跨域泛化。

##### 二、第一阶段方法：Top-K No-Tumor FP Penalty

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

##### 三、新增 Trainer

###### 1. nnUNetTrainer_FPSafe

原版 nnUNet + Top-K 无肿瘤误报惩罚。

目的：验证 loss 层 FP-Safe 约束是否有效。

###### 2. nnUNetTrainer_SizeOV4_FPSafe

SizeOV4 均匀全量 2x 过采样 + Top-K 无肿瘤误报惩罚。

目的：验证“采样 + FP loss”是否协同降低误报，同时保留小肿瘤召回。

###### 3. nnUNetTrainer_MedNeXt_FPSafe

MedNeXt-L + Top-K 无肿瘤误报惩罚。

目的：验证强 DW+IB backbone 的内部高分能否在降低误报和提升外部泛化的同时保留。

##### 四、训练命令

通用环境变量：

```bash
cd /home/PuMengYu/nnUNet
export nnUNet_raw=/home/PuMengYu/nnUNet_workspace/raw
export nnUNet_preprocessed=/home/PuMengYu/nnUNet_workspace/preprocessed
export nnUNet_results=/home/PuMengYu/nnUNet_workspace/results_v2
export nnUNet_extTrainer=/home/PuMengYu/nnUNet/pumengyu
```

###### 1. FP-Safe baseline

```bash
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 003 3d_fullres 0 -tr nnUNetTrainer_FPSafe
```

###### 2. SizeOV4 + FP-Safe

```bash
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 003 3d_fullres 0 -tr nnUNetTrainer_SizeOV4_FPSafe
```

###### 3. MedNeXt + FP-Safe

```bash
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 003 3d_fullres 0 -tr nnUNetTrainer_MedNeXt_FPSafe
```

##### 五、内部测试补跑命令

如果训练完成后 `test_report_custom.txt` 未自动生成：

```bash
python pumengyu/tools/run_internal_test.py --trainer nnUNetTrainer_FPSafe --dataset Dataset003_Liver --fold 0 --gpu 0

python pumengyu/tools/run_internal_test.py --trainer nnUNetTrainer_SizeOV4_FPSafe --dataset Dataset003_Liver --fold 0 --gpu 0

python pumengyu/tools/run_internal_test.py --trainer nnUNetTrainer_MedNeXt_FPSafe --dataset Dataset003_Liver --fold 0 --gpu 0
```

##### 六、外部验证命令

```bash
python pumengyu/ext_val/03_gen_method_report.py --method FPSafe --predict --trainer nnUNetTrainer_FPSafe --fold 0 --gpu 0

python pumengyu/ext_val/03_gen_method_report.py --method SizeOV4_FPSafe --predict --trainer nnUNetTrainer_SizeOV4_FPSafe --fold 0 --gpu 0

python pumengyu/ext_val/03_gen_method_report.py --method MedNeXt_FPSafe --predict --trainer nnUNetTrainer_MedNeXt_FPSafe --fold 0 --gpu 0
```

##### 七、判定标准

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

##### 八、下一阶段候选改进

如果 Top-K FP loss 有效但 Recall 下降：

- 降低 `TKN_TUMOR_FP_LAMBDA` 到 0.3 或 0.5。
- 降低 `TKN_TOPK_PERCENT` 到 0.005。
- 只在训练后半程启用 FP loss。

如果 Top-K FP loss 对误报改善有限：

- 加 Tumor Presence Auxiliary Head。
- 加 candidate-level FP classifier。
- 加 hard negative mining，把无肿瘤误报 case 动态提高采样概率。

##### 九、论文叙事

论文主线不再是“新架构刷最高 Dice”，而是：

> Existing high-Dice liver tumor segmentation models are not necessarily clinically safe. FP-Safe explicitly targets tumor-free false positives and cross-domain robustness while preserving tumor segmentation performance.


---

## 二道攻略指南：从当前 results_v2 走向二区论文


#### 二道攻略指南：从当前 results_v2 走向二区论文

日期：2026-06-30

##### 0. 当前判断

当前材料不是没有故事，而是故事还没有完全升级到二区强度。

已有结果足够支撑一篇完整的应用型论文雏形，核心故事是：

> 内部 Dice 最优不等于外部泛化最优。MedNeXt 在内部测试集很强，但外部验证 drop 大；在 MedNeXt bottleneck 加入低秩全局注意力 MLA 后，内部 Dice 略降，但外部 Overall 第一，internal-external drop 明显变小，同时无肿瘤误报率下降。

目前更像三区/四区可写材料。若要冲二区，需要把故事从“模块拼接提升外部结果”升级成：

> 面向跨域泛化和临床安全的肝肿瘤分割框架，显式处理外部域偏移、小肿瘤漏检和无肿瘤误报。

不要一直找新方向，也不要先按三区随便写完再说。最佳策略是：

1. 先把当前故事写成完整论文骨架。
2. 用二区标准倒逼补实验和补创新。
3. 如果补强成功，投二区；如果补强不成功，降级投三区/四区。

##### 1. 当前最强证据

###### 1.1 MedNeXt 消融结果

| Method | Internal Overall | External Overall | Drop | External Tumor | External FP |
|---|---:|---:|---:|---:|---:|
| MedNeXt | 0.8402 | 0.7705 | -0.0697 | 0.5750 | 60% (3/5) |
| MedNeXt_SizeOV4 | 0.8431 | 0.7797 | -0.0634 | 0.5943 | 60% (3/5) |
| MedNeXt_MLA | 0.8259 | 0.8079 | -0.0180 | 0.6484 | 40% (2/5) |
| MedNeXt_MLA_SizeOV4 | 0.8285 | 0.7870 | -0.0415 | 0.6091 | 60% (3/5) |

关键结论：

- SizeOV4 对 MedNeXt 只有小幅收益：外部 Overall +0.0092。
- MLA 对 MedNeXt 外部泛化收益明显：外部 Overall +0.0374，外部 Tumor +0.0734。
- MedNeXt_MLA 的 drop 只有 -0.0180，明显小于 MedNeXt 的 -0.0697。
- MedNeXt_MLA_SizeOV4 不如单独 MedNeXt_MLA，说明采样策略和 MLA 不一定叠加。

###### 1.2 外部排名信号

当前外部 IRCADb Overall 前几名：

| Rank | Method | External Overall | Liver | Tumor |
|---:|---|---:|---:|---:|
| 1 | MedNeXt_MLA | 0.8079 | 0.9673 | 0.6484 |
| 2 | MoE_SizeOV5 | 0.8025 | 0.9679 | 0.6371 |
| 3 | MLAUNet | 0.8008 | 0.9675 | 0.6341 |
| 4 | SizeOV2 | 0.7992 | 0.9676 | 0.6307 |
| 5 | MLA_GK5_V4 | 0.7957 | 0.9656 | 0.6258 |

这说明 MedNeXt_MLA 不只是 MedNeXt 系列内部最好，而是当前外部主榜第一。

###### 1.3 Case-level 证据

`ircadb_016` 是最强图例候选：

| Method | Tumor Dice on ircadb_016 |
|---|---:|
| MedNeXt | 0.2546 |
| MedNeXt_SizeOV4 | 0.5515 |
| MedNeXt_MLA | 0.8121 |
| MedNeXt_MLA_SizeOV4 | 0.7932 |

这个 case 可以用于可视化展示 MLA 如何从严重失败变成高质量预测。

##### 2. 现在不能乱讲的点

###### 2.1 不能说“找到 MedNeXt 的全部秘诀”

当前消融还不能严格拆出：

- GroupNorm 是否关键。
- residual 是否关键。
- inverted bottleneck 是否关键。
- depthwise conv 是否关键。
- block_counts 深度是否关键。

所以不能写成“我们证明 MedNeXt 的秘诀是某某模块”。

更稳的表述是：

> MedNeXt-style 强局部骨干在内部测试集表现突出，但跨域泛化存在明显 drop。瓶颈低秩全局注意力可以补充其局部卷积归纳偏置，显著改善外部泛化。

###### 2.2 DWSepRes4_MoE_SizeOV4 不能纳入主表

该实验外部 Tumor Dice 为 0，所有有肿瘤 case 都预测空；且没有内部 test report。它应标记为未完成/无效实验，不纳入主线结论。

###### 2.3 FP-Safe 当前只是计划，不是主结论

`FP-Safe肝肿瘤分割计划书.md` 是下一阶段思路。如果 FPSafe / MedNeXt_FPSafe / MedNeXt_MLA_FPSafe 没有完整结果，不能把 FP-Safe 写成主贡献。

##### 3. 二区需要升级的核心

当前 MedNeXt_MLA 仍容易被审稿人质疑为：

> 只是把一个 attention 模块插到已有 backbone 上。

要冲二区，必须回答：为什么这不是简单模块拼接？

建议升级成以下问题定义：

> Internal Dice-oriented liver tumor segmentation can overfit local appearance and fail under external domain shift. We propose a cross-domain robust bottleneck context module for MedNeXt, using low-rank latent attention to model global liver-tumor context while preserving MedNeXt's local inductive bias.

中文主线：

> 面向跨域泛化的肝肿瘤分割：MedNeXt 局部骨干与瓶颈低秩全局注意力融合。

如果 FP-Safe 实验成功，可进一步升级为：

> 面向临床安全跨域泛化的肝肿瘤分割：同时降低无肿瘤误报并保持小肿瘤召回。

##### 4. 推荐论文主线

###### 4.1 Problem

肝肿瘤分割中，内部测试集 Dice 高不等于外部临床数据可靠。主要风险包括：

- 外部域偏移导致 tumor Dice 明显 drop。
- 小病灶漏检。
- 无肿瘤 case 误报，临床安全性不足。

###### 4.2 Observation

MedNeXt / MedNeXt_SizeOV4 内部 Overall 最高，但外部 drop 大。单纯采样策略不能解决跨域问题。

###### 4.3 Method

在 MedNeXt bottleneck 处加入 MLABottleneck3D：

- MedNeXt encoder/decoder 保留强局部表征。
- bottleneck 低分辨率 512 通道特征上做低秩全局注意力。
- 目标是在计算可控的情况下建立全局 liver-tumor context。

###### 4.4 Result

MedNeXt_MLA：

- 外部 Overall 第一：0.8079。
- 外部 Tumor Dice：0.6484。
- Internal-external drop：-0.0180，明显低于 MedNeXt 的 -0.0697。
- 无肿瘤 FP：从 60% 降到 40%。

###### 4.5 Analysis

必须包含：

- MedNeXt 消融表。
- 外部主表。
- size-group analysis。
- no-tumor FP analysis。
- case visualization，尤其 `ircadb_016`。

##### 5. 下一步实验优先级

###### 5.1 必做：写论文骨架

不要等所有实验都完美再写。先写完整骨架：

1. Title
2. Abstract
3. Introduction 三段逻辑
4. Method 图和模块描述
5. Experiment 表格框架
6. MedNeXt ablation 表
7. External validation 表
8. Case visualization 占位
9. Discussion 和 limitation

写作会暴露故事缺口，然后再定向补实验。

###### 5.2 强烈建议补：MedNeXt_MLA_FPSafe

当前 FP-Safe 计划的价值在于临床安全性。如果能在 MedNeXt_MLA 上降低外部无肿瘤误报，同时不明显损害 tumor Dice，论文档次会提升。

建议新增或确认 trainer：

```text
nnUNetTrainer_MedNeXt_MLA_FPSafe
```

目标对照：

| Method | 目的 |
|---|---|
| MedNeXt | 强局部骨干 baseline |
| MedNeXt_MLA | 加全局上下文 |
| MedNeXt_FPSafe | 加 FP-safe loss |
| MedNeXt_MLA_FPSafe | 全局上下文 + FP-safe |

成功标准：

- 外部 FP 率下降。
- Precision / FDR 改善。
- Tumor Dice 不明显下降。
- Drop 不变大，最好进一步变小。

###### 5.3 必做：case 图

优先做这些 case：

- `ircadb_016`：MedNeXt_MLA 明显修复失败。
- `ircadb_008`：普遍困难小病灶。
- `ircadb_018`：极小病灶，几乎所有方法失败，用于 limitation。
- `ircadb_014`：无肿瘤误报典型 case。

###### 5.4 可选：更细 MedNeXt 机制消融

如果时间足够，可以补 1-2 个干净消融，但不要发散太多：

- MedNeXt_MLA_Block1 vs Block2：MLA 层数是否必要。
- MedNeXt_MLA_NoFFN 或低秩比例变化：证明 MLA 设计不是随便插。
- MedNeXt_MLA_FPSafe：比结构消融更有临床价值。

不建议继续大量尝试无关 trainer。

##### 6. 论文表格建议

###### 6.1 主表：内部 + 外部

主表不要放全部 24 个 trainer。保留有代表性的：

- Baseline
- SizeOV2 / SizeOV3
- MLAUNet
- MoE_SizeOV5
- MedNeXt
- MedNeXt_SizeOV4
- MedNeXt_MLA
- MedNeXt_MLA_SizeOV4
- nnFormer
- SwinUNETR

###### 6.2 消融表：MedNeXt 系列

必须单独成表：

- MedNeXt
- MedNeXt_SizeOV4
- MedNeXt_MLA
- MedNeXt_MLA_SizeOV4
- 如果补了：MedNeXt_FPSafe / MedNeXt_MLA_FPSafe

###### 6.3 安全性表

重点列：

- no-tumor FP rate
- Precision
- FDR
- Recall
- Tiny/small tumor Dice

##### 7. 投稿策略

###### 7.1 不推荐的策略

不推荐一直找思路，直到感觉能二区再写。这样容易无限拖延，最后没有完整论文。

也不推荐先随便写一个三区，再事后硬改二区。因为从问题定义开始就会偏弱。

###### 7.2 推荐策略

推荐：

1. 现在立刻按二区目标写论文骨架。
2. 同时补 MedNeXt_MLA_FPSafe 和关键可视化。
3. 用结果决定投稿档次。

如果 FP-Safe 或其他临床安全模块成功：

```text
冲二区应用型医学影像 / 生物医学工程 / computer methods 方向期刊。
```

如果只有 MedNeXt_MLA，但没有更强机制或 FP-safe 结果：

```text
三区/四区应用型期刊更稳。
```

##### 8. 给下一个 AI 的重点

如果下一个 AI 接手，请优先做以下事情：

1. 不要重新发散设计大量 trainer。
2. 先读取：
   - `pumengyu/notes/md/02_实验结果/results_v2_内部外部结果列表.md`
   - `pumengyu/notes/md/01_实验方案/FP-Safe肝肿瘤分割计划书.md`
   - `pumengyu/trainers/trainer.py` 中 MedNeXt 系列 trainer
   - `pumengyu/architectures/mednext.py`
3. 帮用户把论文骨架写出来。
4. 若要补代码，优先实现/确认 `nnUNetTrainer_MedNeXt_MLA_FPSafe`。
5. 所有结果分析必须按 Overall = (Liver + Tumor) / 2，并同时报告 external drop。

##### 9. 当前一句话结论

当前最强可写故事是：

> MedNeXt gives strong in-domain performance, but its external robustness is limited. Adding a low-rank latent attention bottleneck trades a small amount of internal Dice for substantially better external tumor segmentation, lower internal-external drop, and reduced tumor-free false positives. To reach a stronger paper, this should be developed into a clinically safer cross-domain liver tumor segmentation framework, ideally by combining MedNeXt_MLA with FP-Safe false-positive suppression.


---

## 下一步实验计划


#### 下一步实验计划

> 更新：2026-06-27
> 核心目标：**证明 DW+IB conv 是有效成分，而非 MedNeXt 精心调参的结构本身**

---

##### 2026-07-05 决策更新：加入 HCC-TACE-Seg 多期 CT 训练路线

本地数据：

```text
/home/PuMengYu/HCC/HCC-TACE-Seg_v1_202201
```

已确认样例 `HCC_001`：

| Series | Modality | SeriesDescription | 切片数 | 说明 |
|---|---|---|---:|---|
| `00377/76970` | CT | `PRE LIVER` | 43 | 平扫/术前肝脏 CT |
| `00377/42120` | CT | `C-A-P` | 66 | 增强期之一 |
| `00377/99942` | SEG | `Segmentation` | 1 | DICOM SEG 标注 |
| `49771/07012` | CT | `Recon 2: PRE LIVER` | 87 | 另一次 study 的平扫 |
| `49771/35194` | CT | 空 | 92 | 派生 CT series |
| `49771/46705` | CT | 空 | 92 | 派生 CT series |

判断：

- HCC 数据不是 nnUNet-ready，需要先做 DICOM 多 series 转换。
- 它适合作为“多期 CT / 多通道输入”路线，目标不是继续单期 LiTS 刷分，而是解决不同期相 CT 下肿瘤表征不稳定的问题。
- 这条路线与当前 MedNeXt/DWIB 机制消融互补：DWIB/MedNeXt 解决局部 block 设计，HCC 多期 CT 解决输入信息不足。
- **硬约束**：HCC 不允许注入或拷贝到 `Dataset003_Liver` 文件夹；MSD 与 HCC 必须物理隔离，混合训练只能通过 Trainer/Mixin 的运行时开关完成。

###### 新目标

构建一个新的 HCC 多期 CT 训练数据集。它必须独立存在，不和 MSD/LiTS 混放：

```text
Dataset013_HCCMultiPhase
```

建议先采用多通道输入：

```text
case_0000.nii.gz = pre / non-contrast phase
case_0001.nii.gz = arterial / C-A-P / enhancement phase
case_0002.nii.gz = portal / delayed / matched enhancement phase（若可稳定识别）
case.nii.gz      = segmentation label
```

如果某些 case 不能稳定识别 3 个期相，则先退化为 2 通道：

```text
0000 = PRE LIVER
0001 = best contrast phase
```

###### 数据隔离与混合训练设计

不再使用旧的 `inject.py` 思路。旧方案会把外部 case 写进 `Dataset003_Liver/preprocessed` 并改 splits，适合少量无肿瘤外部样本，但不适合 HCC 多期 CT。

新的组织方式：

```text
nnUNet_workspace/
├── raw/
│   ├── Dataset003_Liver/              # 原始 MSD/LiTS，保持不动
│   └── Dataset013_HCCMultiPhase/      # HCC 多期 CT，独立数据集
├── preprocessed/
│   ├── Dataset003_Liver/              # 纯 MSD preprocessing
│   └── Dataset013_HCCMultiPhase/      # 纯 HCC preprocessing
└── results_v2/
    ├── Dataset003_Liver/
    └── Dataset013_HCCMultiPhase/
```

训练入口分三类：

| 类型 | Trainer | 数据来源 | 用途 |
|---|---|---|---|
| 纯 MSD | `nnUNetTrainer_MedNeXt_MLA_MSDOnly` | 只读 `Dataset003_Liver` | 保持历史结果可比 |
| 纯 HCC | `nnUNetTrainer_MedNeXt_MLA_HCCOnly` | 只读 `Dataset013_HCCMultiPhase` | 验证多期 CT 本身效果 |
| 混合训练 | `nnUNetTrainer_MedNeXt_MLA_MSDHCCMix` | 运行时读 `Dataset003_Liver` + `Dataset013_HCCMultiPhase` | 联合训练，不改任何 dataset 文件 |

混合训练必须通过隔离开关控制：

```text
MIX_HCC_ENABLE      = True / False
MIX_HCC_DATASET     = "Dataset013_HCCMultiPhase"
MIX_HCC_RATIO       = 0.25 / 0.5 / 1.0
MIX_HCC_CHANNEL_MAP = "msd_repeat" 或 "msd_zero_fill"
```

设计原则：

1. `Dataset003_Liver` 永远不写入 HCC case。
2. `Dataset013_HCCMultiPhase` 永远不写入 MSD case。
3. 混合训练只在 `get_tr_and_val_datasets()` 或 dataloader 层做运行时组合。
4. 验证集默认仍用主 dataset 的 val split，不能被 HCC 混入污染。
5. 要能一键关闭混合：同一个架构只要换 Trainer 或关闭 `MIX_HCC_ENABLE`，就回到纯 MSD。

###### 单期 MSD 与多期 HCC 的 channel 对齐

HCC 可能是 2-3 通道，多期输入；MSD 是单期 CT。混合训练时网络输入通道必须一致，因此不能直接把两个 dataset 拼起来。

建议先使用简单 channel adapter：

| 模式 | MSD 输入适配 | HCC 输入 | 说明 |
|---|---|---|---|
| `msd_repeat` | `CT -> [CT, CT]` 或 `[CT, CT, CT]` | 原始多期通道 | 最稳，避免空通道分布突变 |
| `msd_zero_fill` | `CT -> [CT, 0]` 或 `[CT, 0, 0]` | 原始多期通道 | 明确告诉模型缺失期相，但分布差异更大 |

第一版推荐 `msd_repeat`，因为实现简单，且不会让 MSD 样本带大量全 0 phase 通道。

后续如果多期有效，再考虑显式 phase dropout / missing phase mask。

###### 必做工程任务

| 优先级 | 任务 | 输出 | 状态 |
|---|---|---|---|
| 1 | 扫描 HCC 全数据 DICOM header | `hcc_series_inventory.csv` | 已完成 |
| 2 | 识别 CT series 与 SEG series | `hcc_multiphase_case_plan.csv` | 已完成：89 ready / 7 review / 9 exclude |
| 3 | 将 DICOM SEG 转为 NIfTI label | `labelsTr/HCC_xxx.nii.gz` | 已完成：正式 raw dataset 保留 87 case |
| 4 | 多期 CT 配准/重采样到同一空间 | `imagesTr/HCC_xxx_0000/0001/...nii.gz` | 已完成：PRE 重采样到 contrast CT 空间 |
| 5 | 生成 nnUNet `dataset.json` | 多通道 channel_names | 已完成：2 通道 `pre_ct/contrast_ct` |
| 6 | 跑 `plan_and_preprocess` | `Dataset013_HCCMultiPhase` preprocessed | 已完成 |
| 7 | 纯 HCC Trainer 训练 | `nnUNetTrainer_MedNeXt_MLA_HCCOnly` | 待做 |
| 8 | 混合训练 Mixin | `HCCMixTrainingMixin` | 已完成：第一版，以 Dataset013 为主，运行时追加 MSD |
| 9 | MSD+HCC 混合 Trainer | `nnUNetTrainer_MedNeXt_MLA_MSDHCCMix` | 已完成 |

###### Trainer 设计

第一阶段不要改网络结构，先验证“多期输入本身是否有效”：

| Trainer | 目的 |
|---|---|
| `nnUNetTrainer_Baseline_HCCOnly` | HCC 多期输入 baseline |
| `nnUNetTrainer_MedNeXt_HCCOnly` | 检查 MedNeXt 在 HCC 多期上是否仍同域强 |
| `nnUNetTrainer_MedNeXt_MLA_HCCOnly` | 纯 HCC 主候选 |
| `nnUNetTrainer_MedNeXt_MLA_MSDOnly` | 只用 MSD，保持历史基线 |
| `nnUNetTrainer_MedNeXt_MLA_MSDHCCMix` | MSD + HCC 运行时混合训练 |
| `nnUNetTrainer_DeepDWIBMedConfig_MSDHCCMix` | 自写 DWIB 混合训练可迁移性验证 |

实现上不需要为多期 CT 单独写网络，只要 `dataset.json` 中 channel 数正确，现有 `build_network_architecture` 会通过 `num_input_channels` 自动适配第一层输入通道。

但混合训练时必须额外处理 channel 数：

- 如果主 dataset 是 `Dataset003_Liver`，Trainer 需要把 `num_input_channels` 提升到 HCC 通道数，并对 MSD 样本做 channel adapter。
- 如果主 dataset 是 `Dataset013_HCCMultiPhase`，HCC 原样输入，MSD 样本通过 adapter 对齐。
- 不能通过修改 `Dataset003_Liver/dataset.json` 来伪装多通道；所有适配应在混合 Trainer 内完成。

###### 关键风险

1. **期相命名不稳定**：SeriesDescription 可能为空或不统一，不能只靠名字判断。
2. **同一病人多 Study**：例如 `HCC_001` 有 `00377` 和 `49771` 两个 study，需要选择与 SEG 同 study 或同空间最匹配的 CT。
3. **DICOM SEG 对齐问题**：SEG 往往引用特定 source series，必须按 ReferencedSeriesSequence 找到对应 CT。
4. **多期未配准**：不同期相可能呼吸位移明显，需要先 rigid/affine 或至少重采样到 label 参考空间。
5. **标签定义可能不是 liver+tumor 双标签**：必须检查 DICOM SEG 的 SegmentLabel，确认是否只有 tumor，还是包含 liver/lesion/necrosis 等。

###### 建议决策

HCC 多期 CT 应该作为下一阶段高优先级，但顺序必须是：

```text
先做 HCC 数据转换与质控
    ↓
跑纯 HCC Baseline / MedNeXt_MLA
    ↓
实现 HCCMixTrainingMixin，跑 MSD+HCC 混合训练
    ↓
再决定是否设计跨期融合模块
```

不建议一开始就写复杂跨期 attention Trainer。先让 nnUNet 的多通道输入跑起来，拿到一个 baseline，才能判断多期 CT 是否真正解决当前外部泛化和小肿瘤问题。

---

##### 2026-07-02 决策更新：FP-Safe 不作为主创新

`MedNeXt_MLA_FPSafe` 已完成内部测试和 IRCADb 外部验证。

| 方法 | 内部 Overall | 外部 Overall | 外部 Tumor | 外部 Precision | 外部 FP率 |
|---|---:|---:|---:|---:|---:|
| MedNeXt_MLA | 0.8259 | **0.8079** | **0.6484** | **0.7437** | **40%** |
| MedNeXt_MLA_FPSafe | **0.8326** | 0.7744 | 0.5852 | 0.6711 | 60% |

结论：

1. FP-Safe 的 Top-K no-tumor FP loss 对内部 FP 控制有效：内部无肿瘤误报从 `MedNeXt_MLA` 的 67% 降到 33%。
2. 但 FP-Safe 没有跨域泛化：IRCADb 外部 Overall 从 0.8079 降到 0.7744，外部 FP 率从 40% 变为 60%。
3. 因此 FP-Safe 不能作为论文主创新继续押注；最多作为“内部 FP 改善但外部失败”的负/中性消融。
4. 主线仍然回到本文件原始科学问题：证明 DW+IB 是可迁移有效成分，并验证 MLA 在强 block 设计上是否带来外部泛化收益。

后续资源分配：

| 优先级 | 方向 | 决策 |
|---|---|---|
| 高 | DeepDWIBResGN / SizeOV4 / MLA | 继续推进，作为主线 |
| 中 | MedNeXt_MLA 可视化 | 解释为什么它外部第一 |
| 低 | FP-Safe 调参 | 不再系统扫参；如有空闲，只做 `lambda=0.3/0.5` 或后半程启用的低成本验证 |

---

##### 核心科学问题

MedNeXt（0.8402）显著优于 Baseline（0.7941），原因是什么？

- ~~假设 A~~：深度 + 残差 + GN 是关键 → **已被 DeepPlainResGN 推翻**（仅 0.7966，+0.003）
- **假设 B**：DW sep conv + Inverted Bottleneck 是关键成分，不依赖 MedNeXt 的精心架构

若假设 B 成立：只要在**任意合理的 UNet 骨干**上换用 DW+IB block，参数量匹配，指标就能上去。
这比"MedNeXt 效果好"更有科学价值——**方法可迁移，不是 MedNeXt 作者调参的功劳**。

更准确的问题表述：

> 如果只根据论文结构图/模块思想，独立写一个 MedNeXt-like 的 DW+IB 网络，不复用官方 MedNeXt 代码，也不精确照抄每个 stage 的实现细节，是否仍能接近官方 MedNeXt？

如果能接近，说明方法鲁棒，核心贡献是 DW+IB 设计范式；如果明显低于官方 MedNeXt，说明性能依赖作者在通道数、block 数、上下采样、decoder 组织、残差路径等细节上的调节。

---

##### 消融实验链（重新设计）

```
Baseline（plain conv, IN, 无残差, 6 stages）               → 0.7941
    ↓ 加深度 + 残差 + GN，保持 plain 3×3 conv
DeepPlainResGN（plain 3×3, deep, GN, ~61M）               → 0.7966  ← 仅 +0.003，block 设计是瓶颈
    ↓ 只换 block：BasicBlockD → DW+IB block，其余不变
DeepDWIBResGN（DW+IB, ~55M，参数对齐）                    → 0.8198  ← 明显提升，但未复现 MedNeXt 0.8402
    ↓ + SizeOV4 过采样
DeepDWIBResGN_SizeOV4                                      → ?       ← 期望 ≈ MedNeXt_SizeOV4 0.8431
    ↓ + MLA Bottleneck（我的贡献）
DeepDWIBResGN_MLA_SizeOV4                                  → ?       ← 期望超越当前最优

参照线：
MedNeXt（精心调参专有结构，DW+IB）                        → 0.8402
MedNeXt_SizeOV4                                            → 0.8431
```

**关键对比**：DeepPlainResGN vs DeepDWIBResGN
- 骨干结构完全相同（n_stages, channels, depth, GN, 残差）
- **唯一区别**：block 类型（plain 3×3 vs DW+IB）
- DeepDWIBResGN 从 0.7966 提升到 0.8198，证明 DW+IB 是有效成分；但仍低于 MedNeXt 0.8402，说明 MedNeXt 的完整结构细节仍有额外贡献。

---

##### DeepDWIBResGN 架构设计

###### Block 设计（ConvNeXt-v1 style）

```
Input (C channels)
  ↓ DW Conv3d(C, C, k=3, groups=C)   ← 空间混合（轻量）
  ↓ GroupNorm(8, C)
  ↓ PW Conv3d(C → C×8)               ← 通道展开（expansion ratio=8）
  ↓ GELU
  ↓ PW Conv3d(C×8 → C)               ← 通道压缩
  ↓ + residual
Output (C channels)
```

###### 架构参数

| 参数 | 值 | 说明 |
|---|---|---|
| n_stages | 5 | 同 DeepPlainResGN |
| features_per_stage | [32, 64, 128, 256, **512**] | bottleneck 从 384 扩到 512（对齐 MedNeXt） |
| enc block counts | [3, 4, 8, 8, 8] | 借用 MedNeXt block_counts 分布 |
| dec block counts | [8, 8, 4, 3] | 同上 |
| expansion ratio | 8 | 同 MedNeXt |
| norm | GroupNorm(8) | 同 DeepPlainResGN |
| activation | GELU | 同 MedNeXt |
| **总参数量** | **58.1M** | MedNeXt-L 61.8M，少 5.9%，属于近似参数量控制 |

> 注：block counts 的分布借用 MedNeXt，但整体骨干结构（ResidualEncoderUNet 框架 / encoder-decoder 分离方式）不依赖 MedNeXt 代码。

###### 与 MedNeXt-L 的差异

| 维度 | MedNeXt-L | DeepDWIBResGN | 是否完全一致 |
|---|---|---|---|
| 参数量 | 61.78M | 58.12M | 否，接近 |
| stem / down / up | MedNeXt 官方结构，`do_res_up_down=True` | 自写 DWIB encoder + nnUNet `UNetDecoder` | 否 |
| block counts | [3,4,8,8,8,8,8,4,3] | encoder [3,4,8,8,8] + decoder [4,4,4,3] | 不完全一致 |
| expansion ratio | [3,4,8,8,8,8,8,4,3] | 全部 r=8 | 否 |
| channels | base 32，bottleneck 512 | [32,64,128,256,512] | 大体一致 |
| block 思想 | DW conv + inverted bottleneck + residual + GN | DW conv + inverted bottleneck + residual + GN | 是 |
| 代码来源 | `nnunet_mednext` 官方/移植实现 | 根据结构思想自写 | 否 |

因此这个实验不是“严格复刻 MedNeXt”，而是“结构图级独立复现”。当前 0.8198 的结果说明 DW+IB 思想有效，但独立复现没有达到官方 MedNeXt 的 0.8402，说明官方结构细节仍然重要。

---

##### 实验队列

###### 当前状态

| 状态 | Trainer | Overall | 备注 |
|---|---|---|---|
| ✅ 完成 | MedNeXt_SizeOV4 | **0.8431** | 当前最优 |
| ✅ 完成 | MedNeXt | 0.8402 | |
| ✅ 完成 | MedNeXt_MLA_SizeOV4 | 0.8285 | MLA 插 MedNeXt 变差 |
| ✅ 完成 | MedNeXt_MLA | 0.8259 / 外部0.8079 | 外部当前第一，MLA 提升泛化 |
| ✅ 完成 | MedNeXt_MLA_FPSafe | 0.8326 / 外部0.7744 | 内部 FP 改善，外部失败 |
| ✅ 完成 | MLAUNet_MoE_SizeOV4 | 0.8330 | 旧底座 + MLA |
| ✅ 完成 | DeepPlainResGN | 0.7966 | plain conv 不够 → 推翻假设 A |
| ✅ 完成 | DeepPlainResGN_SizeOV4 | 0.7908 | 比 Baseline 还差 |
| ✅ 完成 | DeepDWIBResGN | 0.8198 / 外部0.7886 | DW+IB 明显有效，但未达到 MedNeXt |
| ✅ 完成 | DeepResGN_MLA | 0.7969 / 外部0.7551 | plain conv + MLA 救不回来 |

###### 待跑队列

| 优先级 | Trainer | 依赖 | GPU |
|---|---|---|---|
| 1️⃣ 下一个 | **DeepDWIBMedConfig** | 新增 trainer 已实现 | GPU 空闲时 |
| 2️⃣ | **DeepDWIBResGN_SizeOV4** | DeepDWIBResGN 已完成 | GPU 空闲时 |
| 3️⃣ | **DeepDWIBResGN_MLA_SizeOV4** | DeepDWIBResGN_SizeOV4 完成 | GPU 空闲时 |
| 4️⃣ 低优先级 | FP-Safe 小参数验证 | 仅验证 lambda=0.3/0.5 或后半程启用 | 有空再跑 |

###### 当前决策树

```
已知：
DeepResGN_MLA = 0.7969，plain conv + MLA 救不回来。
DeepDWIBResGN = 0.8198，DW+IB 明显有效，但未达到 MedNeXt。

下一步：
DeepDWIBResGN_SizeOV4 >= 0.833
    → DW+IB + 采样策略接近旧自研最优，可继续跑 DeepDWIBResGN_MLA_SizeOV4

DeepDWIBResGN_SizeOV4 < 0.833
    → DW+IB 链条不适合作主线冲分，只保留为机制消融
    → 论文主模型回到 MedNeXt_MLA / MoE_SizeOV5 / MLAUNet
```

---

##### 核心 Claim（等结果支撑）

###### Claim 1：DW+IB 是可迁移的有效成分

> 在参数量对齐（~55M vs 61M）、骨干结构基本相同的情况下，
> 仅将 block 设计从 plain 3×3 换为 DW+IB，
> DeepDWIBResGN 从 0.7966 提升到 0.8198。
>
> 这说明 DW+IB 是明确有效成分，但尚不能声称“MedNeXt 的提升完全来自 DW+IB”。
> MedNeXt 相比 DeepDWIBResGN 仍高 0.0204，提示完整 stage/block 组织、kernel 设计或训练细节仍有贡献。

###### Claim 2：MLA 在强 block 设计上提供额外泛化收益

> DeepDWIBResGN + MLA 的组合：
> - 内部测试集（LiTS）：期望超过 MedNeXt_SizeOV4（0.8431）
> - 外部验证（IRCADb）：MLA 全局建模提升域外泛化（对比 MLAUNet 的 0.8008）

###### Claim 3：MedNeXt 泛化差的根本原因

> MedNeXt DW conv 学习到的 per-channel 空间模式与训练域强绑定，
> 在跨域数据（IRCADb，不同扫描仪协议）上失效（drop -0.063）。
> 加入 MLA 的全局建模可部分对冲这一问题（MedNeXt_MLA_SizeOV4 在 IRCADb 达 0.7870，优于 MedNeXt_SizeOV4 的 0.7797）。

---

##### 参数量核对

| 架构 | 估算参数量 | 备注 |
|---|---|---|
| MedNeXt-L | 61.8M | 参照基准 |
| DeepPlainResGN | ~61M | 对齐成功，但效果差 |
| **DeepDWIBResGN** | **58.1M** | ch=[32,64,128,256,512], enc blocks=[3,4,8,8,8], decoder=[4,4,4,3], r=8 |
| DeepResGN_MLA | ~61M + MLA | 正在训练 |

---

##### 已取消的实验

| Trainer | 原因 |
|---|---|
| MedNeXt_MLA（无SizeOV4） | MedNeXt_MLA_SizeOV4 已说明 MLA+MedNeXt 无效 |
| DWSepRes4_MoE_SizeOV4 | 方向已收敛，放弃 |
| IBConv7 / DWSep7 | 大核消融不是论文主线 |
| Tr_Stage2_FPSup | Liver Dice 崩了（0.1044） |
| DeepResGN_MLA_SizeOV4 | 视 DeepResGN_MLA 结果决定，大概率跳过 |


---
