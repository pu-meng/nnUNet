# 3D-IRCADb-01 标注问题调查报告

**发现日期**：2026-05-28  
**数据集**：3D-IRCADb-01，20 patients，法国 IRCAD 机构标注  
**问题性质**：mask 命名不规范导致多数论文的评估包含错误 GT

---

## 背景

3D-IRCADb-01 是肝脏肿瘤分割领域最常用的外部验证集之一。该数据集每个 patient 的标注以独立 DICOM mask 目录存储，**目录名由标注者自由命名，无统一规范**。我们在数据处理时发现两处严重标注问题，可能影响大量使用该数据集的论文的评估结果。

---

## 发现一：Case 7 的 `tumor` mask 是肾上腺肿瘤，不是肝脏肿瘤

### 事实

Case 7 的 mask 目录中有一个名为 `tumor` 的 mask，体素数 **683,556**（属于"大型肿瘤"级别）。

通过空间重叠分析：

```
tumor mask 总体素：    683,556
与 liver mask 重叠：        230  (0.03%)
不在肝脏内：          683,326  (99.97%)
```

**该 mask 几乎 100% 在肝脏外部。** 结合官方文档（IRCAD 官网明确列出 case 7 为"0 tumour in Liver，1 adrenal tumor"）和 mask 目录中存在 `leftsurrenalgland`（左肾上腺正常组织），可以确认：**`tumor` 是肾上腺肿瘤，不是肝脏肿瘤。**

### 为什么容易出错

命名惯例上，IRCADb 的肝脏肿瘤 mask 通常叫 `livertumor`、`livertumor01` 等。Case 7 的标注者省略了器官前缀，直接命名为 `tumor`。处理脚本若用 `"tumor" in mask_name` 或 `mask_name == "tumor"` 来匹配肿瘤 mask，就会错误纳入。

### 对评估的影响

若将 case 7 错误地标注为"有肝脏肿瘤"（GT label=2，683,556 体素），而模型正确地预测该 case 无肝脏肿瘤（pred≈0），则：

- **Dice = 0**（模型被错误地判定为完全失败）
- 有肿瘤 case 变为 16 个（实际应为 15 个）
- 无肿瘤 case 变为 4 个（实际应为 5 个）

这会系统性地低估模型在大型肿瘤上的性能，且制造出一个"巨大肿瘤 Dice=0"的假失败 case。

---

## 发现二：Case 14 的 `metastasectomie` mask 是术后切除灶，不是活体肿瘤

### 事实

Case 14 的 mask 目录中只有一个与肿瘤相关的 mask：`metastasectomie`，体素数 **32,727**。

- `metastasectomie` 是法语，意为"转移瘤切除术"（surgical resection of metastasis）
- 该 mask 标注的是**手术切除肿瘤后留下的空腔/瘢痕区域**，无活体肿瘤组织
- 官方文档明确列出 case 14 为"0 tumour in Liver"

### 为什么容易出错

该 mask 名包含隐含的医学知识（法语 + 外科术语），处理时容易被误判为"肝脏转移瘤"。实际上 IRCADb 标注团队专门使用不同于 `livertumor` 的命名，正是为了区分"有活体肿瘤"与"术后痕迹"。

### 对评估的影响

若将 case 14 错误标注为"有肿瘤"（GT label=2，32,727 体素），且模型在该区域预测出 16,630 体素：

- 会计算出 Dice ≈ 0.65（把手术瘢痕当成真实肿瘤来打分）
- 无肿瘤 case 变为 3 个（实际应为 4 个，含 case 14）

若将 case 14 正确标注为"无肿瘤"，模型在该区域的预测变为 **FP（误报）**，这是合理的——手术瘢痕在 CT 上外观异常，模型将其误认为肿瘤是真实存在的泛化误差，应当被计入评估。

---

## 两个发现的共同根因

IRCADb 是法国 IRCAD 机构于 2010 年前后标注的数据集，全部 mask 采用**人工自由命名**，无统一规范。主要风险点：

| mask 类型 | 实际出现的命名 | 问题 |
|-----------|--------------|------|
| 肝脏肿瘤 | `livertumor`, `livertumor01~07`, `livertumor1/2`, `livertumors` | 命名分散但尚可识别 |
| 肝脏囊肿 | `liverkyst`, `liverkyste`, `livercyst` | 法语 kyst=cyst，需知道法语 |
| 肾上腺肿瘤 | `leftsurretumor`, `rightsurretumor`, **`tumor`** | `tumor` 无器官前缀，极易误判 |
| 术后切除灶 | **`metastasectomie`** | 法语外科术语，需医学背景才能识别 |

---

## 正确处理方式

```python
def is_liver_tumor(mask_name: str) -> bool:
    name = mask_name.lower()
    if name == "liver":
        return False
    if "metastasectomi" in name:   # 术后切除灶，无活体肿瘤
        return False
    if "liver" not in name:        # 只纳入明确含 liver 的 mask
        return False
    if "cyst" in name or "kyst" in name:   # 囊肿（含法语 kyst）
        return False
    return True
```

关键原则：**只纳入名称中明确含有 `liver` 的 mask**，排除仅叫 `tumor` 的歧义命名。

---

## 修正前后的评估对比（基线 nnUNet，5折ensemble）

| 指标 | 修正前（含错误GT） | 修正后（正确GT） |
|------|:-----------------:|:--------------:|
| 有肿瘤 case 数 | 16 | **15** |
| 无肿瘤 case 数 | 4 | **5**（与官方一致）|
| 有肿瘤 Dice | 0.708 | **0.755**（+4.7pp）|
| 综合 Dice | 0.716 | 0.716（无变化）|
| 无肿瘤 FP | 1/4 | 2/5 |
| 严重失败（Dice=0）| 2 cases | **1 case** |

有肿瘤 Dice 提升 4.7pp，原因：case 7 的错误大 GT（683k 体素）被移除，不再产生虚假的"Dice=0 大肿瘤失败"。

---

## 对其他论文的影响推测

使用 IRCADb 进行肿瘤分割评估的论文，若未对 mask 名称做精细处理，很可能存在以下问题之一：

1. **将 case 7 纳入有肿瘤 case**，导致出现一个"模型完全漏检 68 万体素大肿瘤"的虚假失败 case
2. **将 case 14 纳入有肿瘤 case**，在手术瘢痕上计算出约 0.65 的 Dice，混入有肿瘤 case 的均值

这两处错误的方向相反：case 7 会拉低有肿瘤 Dice，case 14 会略微抬高有肿瘤 Dice，效果相互抵消，导致错误在最终均值上不易察觉，但逐 case 分析会暴露。

---

## 建议（写论文时）

在 Methods 或 Dataset 部分加一段：

> We conducted a careful audit of the 3D-IRCADb-01 mask annotations. We found that case 7 contains a mask named `tumor` which, upon spatial overlap analysis (0.03% overlap with the liver), corresponds to an adrenal tumor rather than a hepatic tumor, consistent with the official IRCAD documentation ("0 tumour in Liver"). Case 14 contains a mask named `metastasectomie` (French: "metastasis resection"), indicating a post-surgical resection bed with no active tumor tissue, also documented as "0 tumour in Liver." Both cases were correctly classified as tumor-negative. After this correction, the dataset contains 15 tumor-positive and 5 tumor-negative cases, matching the official documentation.
