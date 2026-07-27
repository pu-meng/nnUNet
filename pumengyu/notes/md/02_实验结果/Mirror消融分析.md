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
| **Liver Dice** | **0.9340** | **0.9581** | **↑ +0.0241 (+2.6%)** ✅ |
| **Tumor Dice（仅 GT 阳性 23 例）** | **0.7395** | **0.7267** | **↓ -0.0128 (-1.7%)** ❌ |
| **Overall（PMY-LT-v1）** | **0.8368** | **0.8424** | **↑ +0.0056 (+0.7%)** ✅ |
| **Recall** | 0.7853 | 0.6769 | ↓ -0.1084 (-13.8%) ❌ |
| **Precision（仅 GT 阳性 23 例）** | 0.7292 | **0.8353** | **↑ +0.1061 (+14.6%)** ✅ |
| **FDR（仅 GT 阳性 23 例）** | 0.2708 | **0.1212** | **↓ -0.1496 (-55.2%)** ✅ |

> 这里不再将 3 个 GT 无肿瘤病例人为记为 Tumor Dice 0/1。因此总 Tumor Dice 与下方四个阳性肿瘤大小分组的方向一致：NoMirror 均下降。Overall 仍小幅上升，是因为 Liver Dice 的 +0.0241 超过了 Tumor Dice 的 -0.0128。

---

##### 3. 无肿瘤误报率对比

| 指标 | Baseline | NoMirror | 变化 |
|------|----------|----------|------|
| **误报率** | **100% (3/3)** | **66.67% (2/3)** | **↓ 改善** ✅ |
| 误报 case | liver_41(32,055), liver_89(321), liver_91(248) | liver_41(32,318), liver_89(20,837) | **liver_91 正确** ✅ |
| FP 均值 | 10,874.7 | 17,718.3 | ↑ 但 liver_41 基本持平 |

**关键发现**：NoMirror 成功让 **liver_91 从误报变为正确**（TN），但 liver_89 的误报体素从 321 激增到 20,837。

###### 无肿瘤 case 详细对比

| case | Baseline liver_dice | NoMirror liver_dice | Baseline pred_tumor | NoMirror pred_tumor | 结论 |
|------|-------------------|-------------------|-------------------|-------------------|------|
| liver_41 | 0.9547 | 0.9700 | 32,055 | 32,318 | 基本持平 |
| liver_89 | 0.9703 | 0.8973 | 321 | **20,837** | **NoMirror 恶化** ❌ |
| liver_91 | 0.9788 | **0.9854** | 248 | **0** | **NoMirror 改善** ✅ |

---

##### 4. 肿瘤 Dice 按大小分组对比

| 大小分类 | n | Baseline Dice | NoMirror Dice | 变化 |
|----------|---|---------------|---------------|------|
| **极小(<5k)** | 6 | **0.5386** | **0.5151** | ↓ -0.0235 ❌ |
| **小(5k-50k)** | 8 | **0.7928** | **0.7854** | ↓ -0.0074 |
| **中等(50k-300k)** | 1 | 0.7486 | 0.7389 | ↓ -0.0097 |
| **大(>=300k)** | 8 | **0.8359** | **0.8251** | ↓ -0.0108 |

###### Recall 按大小分组

| 大小分类 | Baseline Recall | NoMirror Recall | 变化 |
|----------|----------------|-----------------|------|
| 极小(<5k) | **0.6687** | **0.4442** | **↓ -22.5%** ❌ |
| 小(5k-50k) | **0.8261** | **0.7528** | ↓ -7.3% |
| 中等(50k-300k) | 0.6646 | 0.6221 | ↓ -4.3% |
| 大(>=300k) | **0.8470** | **0.7823** | ↓ -6.5% |

###### Precision 按大小分组

| 大小分类 | Baseline Precision | NoMirror Precision | 变化 |
|----------|-------------------|-------------------|------|
| 极小(<5k) | 0.4766 | **0.7309** | **↑ +25.4%** ✅ |
| 小(5k-50k) | 0.7795 | **0.8393** | ↑ +6.0% |
| 中等(50k-300k) | 0.8568 | **0.9099** | ↑ +5.3% |
| 大(>=300k) | 0.8524 | **0.9003** | ↑ +4.8% |

---

##### 5. 严重失败 case 对比

| 等级 | Baseline | NoMirror |
|------|----------|----------|
| **严重失败 (<0.3)** | 1 case: liver_127 (0.1580) | **2 cases**: liver_127 (0.0000), liver_63 (0.2742) |
| **需要改进 (0.3-0.7)** | 6 cases | 5 cases |
| **没问题 (>=0.7)** | 16 cases | 16 cases |

###### liver_127 详细对比

| 指标 | Baseline | NoMirror |
|------|----------|----------|
| tumor_dice | 0.1580 | **0.0000** |
| recall | 0.1779 | **0.0000** |
| precision | 0.1421 | 0.0000 |
| pred_tumor | 373 | **0** |
| gt_tumor | 298 | 298 |
| size_cat | 极小(<5k) | 极小(<5k) |

**liver_127**：Baseline 还能召回 17.8%（Dice=0.158），NoMirror 直接 **完全漏检**（Dice=0.000，recall=0.000）。

---

##### 6. FPV/FNV 体积误差对比

| 指标 | Baseline | NoMirror | 变化 |
|------|----------|----------|------|
| **Tumor FPV 总量** | 419,617 mm³ | **273,117 mm³** | **↓ -34.9%** ✅ |
| **Tumor FNV 总量** | 540,349 mm³ | **687,156 mm³** | ↑ +27.2% ❌ |
| **Liver FPV 总量** | 5,656,937 mm³ | **2,342,931 mm³** | **↓ -58.6%** ✅ |
| **Liver FNV 总量** | 946,282 mm³ | 1,318,027 mm³ | ↑ +39.3% |

###### Per-case Tumor FPV 前 5 对比

| Baseline | NoMirror |
|----------|----------|
| liver_128: 92,422 | liver_117: 48,875 |
| liver_117: 65,264 | liver_97: 45,907 |
| liver_97: 51,087 | liver_41: 32,318 |
| liver_84: 34,692 | liver_129: 25,934 |
| liver_41: 32,055 | liver_100: 22,061 |

---

##### 7. 连通域分析对比

| 指标 | Baseline | NoMirror |
|------|----------|----------|
| 无肿瘤 case 假CC | 3/3 cases | **2/3 cases** ✅ |
| 假CC 最大体素 | 29,531 | **30,784** |
| 假CC 均值 | 1,919 | **5,906** |
| TP CC 最小体素 | **1** | **4** |

###### 假连通域详细对比

**Baseline**：
- liver_41: 5 个假CC，体素数=[1, 445, 560, 1518, 29531]
- liver_89: 5 个假CC，体素数=[8, 10, 27, 104, 172]
- liver_91: 7 个假CC，体素数=[1, 3, 8, 12, 12, 24, 188]

**NoMirror**：
- liver_41: 6 个假CC，体素数=[9, 60, 133, 249, 1083, 30784]
- liver_89: 3 个假CC，体素数=[1, 25, 20811]
- **liver_91: 无假CC** ✅

---

##### 8. 综合分析

###### NoMirror 的优势

1. **Liver Dice 大幅提升**（0.9340 → 0.9581），肝脏分割更准确
2. **阳性病例 Precision 提升**（0.7292 → 0.8353），预测更保守
3. **无肿瘤误报率下降**（100% → 66.67%），liver_91 从 FP 变为 TN
4. **Tumor FPV 减少 34.9%**，Liver FPV 减少 58.6%
5. **阳性病例 FDR 从 27.1% 降至 12.1%**
6. **所有大小类别的 Precision 均提升**，极小肿瘤 Precision 从 47.7% 提升到 73.1%

###### NoMirror 的代价

1. **Recall 大幅下降**（0.7853 → 0.6769），漏检增多
2. **阳性病例 Tumor Dice 从 0.7395 降至 0.7267**，与所有大小分组下降一致
3. **极小肿瘤 Recall 从 66.9% 降至 44.4%**，下降 22.5%
4. **liver_127 完全漏检**（Dice=0.000，Baseline 尚能召回 17.8%）
5. **Tumor FNV 增加 27.2%**，且 liver_89 误报体素从 321 激增到 20,837

###### 结论

1. **关闭镜像增强确实减少了无肿瘤误报**，但**付出了小肿瘤召回率下降的代价**
2. 肝脏 Dice 提升 2.6%，但这是当前单次消融的相关性结果，不足以单独证明镜像本身是噪音
3. 肿瘤 Recall 下降 13.8% 说明镜像对肿瘤检测有正面作用（肿瘤位置多变，镜像提供了有用的视角多样性）
4. NoMirror 呈现“更高 Precision、更低 Recall”的保守倾向，但不建议据此直接宣称其更适合临床场景
5. Baseline 的肿瘤召回和 Dice 更高；NoMirror 的 Overall 小幅上升主要由 Liver Dice 驱动

###### 后续改进方向

1. **部分镜像**：尝试 `nnUNetTrainer_onlyMirror01`（只镜像 0/1 轴，不镜像上下轴），在 Precision 和 Recall 之间取得平衡
2. **NoMirror + 过采样**：NoMirror 基础上叠加 SizeOversample（如 `nnUNetTrainer_SizeOversampleV3_NoMirror`），用数据层过采样补偿小肿瘤召回损失
3. **NoMirror + UFL**：NoMirror 基础上叠加 UnifiedFocalLoss（delta>0.5 惩罚 FN），用 loss 层补偿召回损失


---
