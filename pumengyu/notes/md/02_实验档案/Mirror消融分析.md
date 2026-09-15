# Mirror 消融分析

> 文档定位：只保留 Baseline vs NoMirror 的详细证据；MedNeXt、MHA、MLA、MoE、SizeOV 和 FP-Safe 的总体结论统一见 [MedNeXt 系列消融实验结果汇总](MedNeXt系列消融实验结果汇总.md)。

## Baseline vs NoMirror 实验结果对比分析

##### 1. 实验设计

| 项目 | Baseline | NoMirror |
|------|----------|----------|
| **Trainer 类** | `nnUNetTrainer_Baseline` | `nnUNetTrainer_NoMirror` |
| **Mixin 继承** | `AutoInternalTestMixin, AutoReportMixin` | `AutoInternalTestMixin, NoMirrorMixin, AutoReportMixin` |
| **唯一变量** | nnUNet 默认配置 | **关闭所有轴的镜像增强**（`mirror_axes = None`） |
| **动机** | — | 肝脏是右侧不对称器官，左右镜像产生"肝脏在左侧"的假图像，可能是噪音 |

###### NoMirrorMixin 实现（`mixins.py:1331-1347`）

```python
class NoMirrorMixin:
    """
    关闭所有轴的镜像增强。
    动机：肝脏是右侧不对称器官，左右镜像会生成"肝脏在左侧"的假图像，
    对模型来说是噪音而非有效增强。关掉后观察是否改善分割精度。
    实现：覆盖 configure_rotation_dummyDA_mirroring_and_inital_patch_size
    的返回值，将 mirror_axes 置为空 tuple，同时清空推理时的镜像轴。
    """
    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        rotation_for_DA, do_dummy_2d, initial_patch_size, _ = \
            super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        self.inference_allowed_mirroring_axes = None
        return rotation_for_DA, do_dummy_2d, initial_patch_size, None
```

###### nnUNet 默认镜像配置（`nnUNetTrainer.py:472-513`）

3D 模式下默认 `mirror_axes = (0, 1, 2)`，即三个空间轴全部镜像：
- 轴 0（前后）：解剖上合理
- 轴 1（左右）：**肝脏在右侧，左右镜像产生"肝脏在左侧"的假图像**
- 轴 2（上下）：解剖上合理

推理时 `inference_allowed_mirroring_axes = (0, 1, 2)`，TTA 对 8 种镜像组合取平均。

---

##### 2. 核心指标对比

| 指标 | Baseline | NoMirror | 变化方向 |
|------|----------|----------|----------|
| **Liver Dice** | 0.9516 | **0.9581** | ↑ +0.0065 (+0.7%) |
| **Tumor Dice（仅 GT 阳性 23 例）** | **0.7311** | 0.7267 | ↓ -0.0044 (-0.6%) |
| **Overall（PMY-LT-v1）** | 0.8414 | **0.8424** | ↑ +0.0010 (+0.1%) |
| **Recall** | **0.7226** | 0.6769 | ↓ -0.0457 (-6.3%) |
| **Precision（仅 GT 阳性 23 例）** | 0.7836 | **0.8353** | ↑ +0.0517 (+6.6%) |
| **FDR（仅 GT 阳性 23 例）** | 0.1729 | **0.1212** | ↓ -0.0517 (-29.9%) |

> Baseline 已替换为 2026-09-03 完成的可信重训结果。这里不将 3 个 GT 无肿瘤病例人为记为 Tumor Dice 0/1。NoMirror 的 Overall 仅高 0.0010：其 Liver Dice 更高，但 Tumor Dice 和 Recall 都略低。两者是各自完整训练的一次运行，差值可描述现象，不能当作多随机种子下的稳定因果效应。

---

##### 3. 无肿瘤误报率对比

| 指标 | Baseline | NoMirror | 变化 |
|------|----------|----------|------|
| **误报率** | **66.67% (2/3)** | **66.67% (2/3)** | 不变 |
| 误报 case | liver_41(28,693), liver_91(41) | liver_41(32,318), liver_89(20,837) | 病例身份发生交换 |
| FP 均值（3 个阴性病例） | **9,578.0** | 17,718.3 | NoMirror 更高 |

**关键发现**：两者都误报 2/3 个阴性病例。NoMirror 使 **liver_91** 从 FP 变为 TN，但同时使 Baseline 中为 TN 的 **liver_89** 出现 20,837 体素误报，因此不能再写成“关闭镜像降低了病例级误报率”。

###### 无肿瘤 case 详细对比

| case | Baseline liver_dice | NoMirror liver_dice | Baseline pred_tumor | NoMirror pred_tumor | 结论 |
|------|-------------------|-------------------|-------------------|-------------------|------|
| liver_41 | 0.9680 | 0.9700 | 28,693 | 32,318 | 两者均有大体积误报 |
| liver_89 | **0.9761** | 0.8973 | **0** | 20,837 | NoMirror 恶化 |
| liver_91 | 0.9853 | **0.9854** | 41 | **0** | NoMirror 改善 |

---

##### 4. 肿瘤 Dice 按大小分组对比

| 大小分类 | n | Baseline Dice | NoMirror Dice | 变化 |
|----------|---|---------------|---------------|------|
| **极小(<5k)** | 6 | 0.5107 | **0.5151** | ↑ +0.0044 |
| **小(5k-50k)** | 8 | **0.7996** | 0.7854 | ↓ -0.0142 |
| **中等(50k-300k)** | 1 | 0.7090 | **0.7389** | ↑ +0.0299 |
| **大(>=300k)** | 8 | **0.8307** | 0.8251 | ↓ -0.0056 |

###### Recall 按大小分组

| 大小分类 | Baseline Recall | NoMirror Recall | 变化 |
|----------|----------------|-----------------|------|
| 极小(<5k) | **0.5394** | 0.4442 | ↓ -0.0952 |
| 小(5k-50k) | **0.7893** | 0.7528 | ↓ -0.0365 |
| 中等(50k-300k) | 0.5705 | **0.6221** | ↑ +0.0516 |
| 大(>=300k) | **0.8124** | 0.7823 | ↓ -0.0301 |

###### Precision 按大小分组

| 大小分类 | Baseline Precision | NoMirror Precision | 变化 |
|----------|-------------------|-------------------|------|
| 极小(<5k) | 0.5649 | **0.7309** | ↑ +0.1660 |
| 小(5k-50k) | 0.8288 | **0.8393** | ↑ +0.0105 |
| 中等(50k-300k) | **0.9360** | 0.9099 | ↓ -0.0261 |
| 大(>=300k) | 0.8835 | **0.9003** | ↑ +0.0168 |

---

##### 5. 严重失败 case 对比

| 等级 | Baseline | NoMirror |
|------|----------|----------|
| **严重失败 (<0.3)** | 1 case: liver_127 (0.0000) | **2 cases**: liver_127 (0.0000), liver_63 (0.2742) |
| **需要改进 (0.3-0.7)** | 7 cases | 5 cases |
| **没问题 (>=0.7)** | 15 cases | 16 cases |

###### liver_127 详细对比

| 指标 | Baseline | NoMirror |
|------|----------|----------|
| tumor_dice | 0.0000 | 0.0000 |
| recall | 0.0000 | 0.0000 |
| precision | 0.0000 | 0.0000 |
| pred_tumor | 0 | 0 |
| gt_tumor | 298 | 298 |
| size_cat | 极小(<5k) | 极小(<5k) |

**liver_127**：新 Baseline 与 NoMirror 都完全漏检，说明该病例是两次训练共有的困难样本，不能再归因于关闭镜像。

---

##### 6. FPV/FNV 体积误差对比

| 指标 | Baseline | NoMirror | 变化 |
|------|----------|----------|------|
| **Tumor FPV 总量** | 303,324 mm³ | **273,117 mm³** | ↓ -30,207 (-10.0%) |
| **Tumor FNV 总量** | **652,227 mm³** | 687,156 mm³ | ↑ +34,929 (+5.4%) |
| **Liver FPV 总量** | 3,501,973 mm³ | **2,342,931 mm³** | ↓ -1,159,042 (-33.1%) |
| **Liver FNV 总量** | **1,281,403 mm³** | 1,318,027 mm³ | ↑ +36,624 (+2.9%) |

###### Per-case Tumor FPV 前 5 对比

| Baseline | NoMirror |
|----------|----------|
| liver_117: 51,358 | liver_117: 48,875 |
| liver_97: 43,822 | liver_97: 45,907 |
| liver_128: 43,300 | liver_41: 32,318 |
| liver_41: 28,693 | liver_129: 25,934 |
| liver_100: 26,956 | liver_100: 22,061 |

---

##### 7. 连通域分析对比

| 指标 | Baseline | NoMirror |
|------|----------|----------|
| 无肿瘤 case 假CC | 2/3 cases | 2/3 cases |
| 假CC 最大体素 | **26,756** | 30,784 |
| 假CC 均值 | **5,747** | 5,906 |
| TP CC 最小体素 | **1** | **4** |

###### 假连通域详细对比

**Baseline**：
- liver_41: 4 个假CC，体素数=[310, 452, 1175, 26756]
- **liver_89: 无假CC**
- liver_91: 1 个假CC，体素数=[41]

**NoMirror**：
- liver_41: 6 个假CC，体素数=[9, 60, 133, 249, 1083, 30784]
- liver_89: 3 个假CC，体素数=[1, 25, 20811]
- **liver_91: 无假CC** ✅

---

##### 8. 综合分析

###### NoMirror 的优势

1. Liver Dice 小幅提高（0.9516 → 0.9581）
2. 阳性病例 Precision 提高（0.7836 → 0.8353），FDR 从 17.29% 降至 12.12%
3. Tumor FPV 减少 10.0%，Liver FPV 减少 33.1%
4. 极小、小和大肿瘤分组的 Precision 更高，预测整体更保守

###### NoMirror 的代价

1. Recall 从 0.7226 降至 0.6769，Tumor FNV 增加 5.4%
2. 阳性病例 Tumor Dice 从 0.7311 降至 0.7267
3. 极小肿瘤 Recall 从 0.5394 降至 0.4442
4. 无肿瘤病例误报率没有改善，仍为 2/3；且 liver_89 从 TN 变为 20,837 体素误报
5. liver_127 在两种配置中均完全漏检

###### 结论

1. NoMirror 的 Overall 仅提高 0.0010，主要来自 Liver Dice；Tumor Dice 反而低 0.0044
2. NoMirror 呈现“更高 Precision、更低 Recall”的保守倾向，但病例级无肿瘤误报率与 Baseline 相同
3. Baseline 的肿瘤 Recall 更高；NoMirror 的 FPV 更低、FNV 更高，体现精确率—召回率权衡
4. 两个配置均只有一次完整训练，现有差值不足以证明镜像增强本身是收益或噪音；正式论文应写成观察性消融结果
5. liver_127 的共同完全漏检和阴性病例误报身份交换，说明结论必须结合病例层面证据，不能只看 0.0010 的 Overall 差距

###### 后续改进方向

1. **部分镜像**：尝试 `nnUNetTrainer_onlyMirror01`（只镜像 0/1 轴，不镜像上下轴），在 Precision 和 Recall 之间取得平衡
2. **NoMirror + 过采样**：NoMirror 基础上叠加 SizeOversample（如 `nnUNetTrainer_SizeOversampleV3_NoMirror`），用数据层过采样补偿小肿瘤召回损失
3. **NoMirror + UFL**：NoMirror 基础上叠加 UnifiedFocalLoss（delta>0.5 惩罚 FN），用 loss 层补偿召回损失


---
