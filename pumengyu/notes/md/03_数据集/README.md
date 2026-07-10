# 数据集

## 公开肝脏肿瘤分割数据集汇总


#### 公开肝脏肿瘤分割数据集汇总

> 全球可公开获取、同时具备**肝脏+肿瘤标注**的医学影像数据集整理
> 更新时间：2026年5月

---

##### 一、核心 CT 数据集

###### 1. LiTS17（Liver Tumor Segmentation Benchmark）
| 字段 | 内容 |
|------|------|
| 样本数 | 131例训练集（全部有标签）+ 70例测试集（无公开标签，仅用于挑战赛评分） |
| 模态 | CT |
| 标注内容 | 肝脏（label=1）、肿瘤（label=2） |
| 格式 | `.nii.gz` |
| 来源 | MICCAI 2017/2018 挑战赛，7家医院合作 |
| 特点 | 最权威、最常用；包含原发和转移性肿瘤；大小和外观多样 |
| 获取 | 免费，需注册申请 |
| 链接 | https://competitions.codalab.org/competitions/17094 |

---

###### 2. MSD Task03_Liver（Medical Segmentation Decathlon）
| 字段 | 内容 |
|------|------|
| 样本数 | 训练集131例（部分无肿瘤需排除） |
| 模态 | CT |
| 标注内容 | 肝脏（label=1）、肿瘤（label=2） |
| 格式 | `.nii.gz` |
| 来源 | MICCAI 2018 Decathlon |
| 特点 | 与LiTS17有部分重叠，合并使用需去重 |
| 获取 | 免费 |
| 链接 | http://medicaldecathlon.com/ |

---

###### 3. 3D-IRCADb-01
| 字段 | 内容 |
|------|------|
| 样本数 | 20例（75%有肝脏肿瘤，约15例有效） |
| 模态 | CT |
| 标注内容 | 肝脏、肿瘤、血管等多结构精细标注 |
| 格式 | DICOM + VTK |
| 来源 | 法国 IRCAD 研究所 |
| 特点 | 标注质量极高，专家级；样本量小但精度高，适合作外部测试集 |
| 获取 | 免费 |
| 链接 | https://www.ircad.fr/research/data-sets/liver-segmentation-3d-ircadb-01/ |

---

###### 4. 3D-IRCADb-02
| 字段 | 内容 |
|------|------|
| 样本数 | 2例 |
| 模态 | 多期相CT（动脉期+门静脉期） |
| 标注内容 | 多结构精细标注 |
| 来源 | 法国 IRCAD 研究所 |
| 特点 | 样本极少，标注极精细 |
| 获取 | 免费 |
| 链接 | https://www.ircad.fr/research/data-sets/liver-segmentation-3d-ircadb-02/ |

---

###### 5. MCT-LTDiag
| 字段 | 内容 |
|------|------|
| 样本数 | 517例 |
| 模态 | 多期相CT（平扫、动脉期、门静脉期、延迟期） |
| 标注内容 | 肝脏、肿瘤（5种类型：HCC、ICC、CRLM、BCLM、HH） |
| 来源 | 北京协和医院；发表于 Nature Scientific Data（2025） |
| 特点 | 目前**样本量最大**的公开多期相肝脏肿瘤CT数据集；肿瘤类型最全 |
| 获取 | 申请获取 |
| 链接 | https://www.nature.com/articles/s41597-025-06343-4 |

---

###### 6. CRLM-CT-Seg
| 字段 | 内容 |
|------|------|
| 样本数 | 197例 |
| 模态 | CT |
| 标注内容 | 肝脏、结直肠癌肝转移灶（CRLM）、残余肝体积（FLR） |
| 来源 | 公开数据集，经放射科医生手动精修（arXiv 2025） |
| 特点 | 结直肠癌肝转移专项；适合转移瘤分割研究 |
| 获取 | 免费（Zenodo） |
| 链接 | https://arxiv.org/abs/2604.07999 |

---

###### 7. HCC-TACE
| 字段 | 内容 |
|------|------|
| 样本数 | 105例 |
| 模态 | CT（治疗前后多时间点） |
| 标注内容 | 肝实质、存活肿瘤、坏死肿瘤、肝内血管、主动脉 |
| 来源 | MD Anderson癌症中心（2002-2012） |
| 特点 | 含TACE介入治疗前后影像；标注最详细；含临床随访数据（OS、TTP） |
| 获取 | TCIA 免费 |
| 链接 | https://www.cancerimagingarchive.net/ |

---

##### 二、MRI 数据集

###### 8. LiverHccSeg
| 字段 | 内容 |
|------|------|
| 样本数 | 小规模（来自TCGA-LIHC子集） |
| 模态 | 多期相MRI（平扫+动脉期+门静脉期+延迟期） |
| 标注内容 | 肝脏、HCC肿瘤 |
| 特点 | 提供多标注者一致性分析（inter-rater agreement）；适合评估标注可靠性 |
| 获取 | 免费（PMC） |
| 链接 | https://pmc.ncbi.nlm.nih.gov/articles/PMC10587725/ |

---

###### 9. ATLAS（2023）
| 字段 | 内容 |
|------|------|
| 样本数 | 90例 |
| 模态 | 增强MRI（CE-MRI，T1加权） |
| 标注内容 | 肝脏、HCC肿瘤 |
| 来源 | MICCAI 2023 Workshop；法国勃艮第大学 |
| 特点 | 首个专为放射栓塞（TARE）治疗计划设计的公开MRI数据集；不可切除HCC专项 |
| 获取 | 免费（Grand Challenge） |
| 链接 | https://atlas-challenge.u-bourgogne.fr/ |

---

##### 三、多任务数据集

###### 10. LiMT
| 字段 | 内容 |
|------|------|
| 样本数 | 150例 |
| 模态 | CT（动脉期增强） |
| 标注内容 | 肝脏分割、4种病变类型分类标注、病灶检测标注 |
| 来源 | 江苏大学附属医院；2025年发表 |
| 特点 | 支持分割+分类+检测三任务联合训练；含正常肝脏对照样本 |
| 获取 | 申请获取（Google Drive） |
| 链接 | https://arxiv.org/abs/2511.19889 |

---

##### 四、汇总对比表

| 数据集 | 可用例数 | 模态 | 肿瘤标注 | 多期相 | 获取方式 |
|--------|---------|------|---------|--------|---------|
| LiTS17 | 131（训练集全有标签） | CT | ✅ | ❌ | 免费注册 |
| MSD Task03 | ~100有效（去重后新增~10-20） | CT | ✅ | ❌ | 免费 |
| 3D-IRCADb-01 | ~15有效 | CT | ✅ | ❌ | 免费（建议留测试集） |
| 3D-IRCADb-02 | 2 | 多期CT | ✅ | ✅ | 免费 |
| MCT-LTDiag | 517（只取门静脉期） | 多期CT | ✅ | ✅ | 申请 |
| CRLM-CT-Seg | 197 | CT | ✅（转移瘤） | ❌ | 免费 |
| HCC-TACE | ~80（只取治疗前） | CT | ✅ | ❌ | TCIA免费 |
| LiverHccSeg | 小规模 | 多期MRI | ✅（HCC） | ✅ | 免费 |
| ATLAS | 90 | CE-MRI | ✅（HCC） | ❌ | 免费 |
| LiMT | 150 | CT | ✅ | ❌ | 申请 |

---

##### 五、多期相（Multi-phase）说明

CT扫描注射造影剂后可在**不同时间点**拍摄多张，每张叫一个"期相"：

| 期相 | 注射后时间 | 特点 |
|------|-----------|------|
| **平扫期** | 注射前 | 无增强，基础结构 |
| **动脉期** | 25-30秒 | 肿瘤强化明显，HCC典型表现 |
| **门静脉期** | 60-70秒 | 肝实质最亮，肿瘤对比最清晰（**LiTS/Task03均为此期**） |
| **延迟期** | 3-5分钟 | 肿瘤廓清，ICC典型表现 |

> LiTS 和 Task03 **只有门静脉期单期**；MCT-LTDiag 有四期，一个病人=4张CT。
> 合并时只取门静脉期即可与LiTS保持一致。

---

##### 六、数据集合并策略

###### CT类：可合并，坑各不同

####### ✅ 几乎无障碍
- **LiTS131 + Task03去重** → 格式、标签完全一致，直接合并

####### ⚠️ 需处理但完全可用

**CRLM-CT-Seg（197例）**
- 问题：标注的是**结直肠癌转移瘤**，label含义略不同
- 解决：把肿瘤label统一重映射为label=2
- 难度：⭐（简单）

**HCC-TACE（105例）**
- 问题1：含TACE治疗后影像，肿瘤有坏死/碘油沉积，外观异常
- 问题2：DICOM格式，需转nii
- 解决：**只取治疗前影像**，用dcm2niix转格式
- 难度：⭐⭐

**MCT-LTDiag（517例）**
- 问题：四期相，每人4张CT
- 解决：**只取门静脉期**，与LiTS保持一致
- 难度：⭐⭐（需申请后处理）

###### MRI类：不建议直接混入CT训练

- MRI 与 CT 的 HU 值体系完全不同，混合训练会严重干扰模型
- ✅ 正确做法：单独训练MRI专用模型，或使用**域适应（Domain Adaptation）**方法

---

##### 七、推荐训练方案

###### 方案A：快速扩充
```
训练+验证：LiTS131 + Task03去重后新增
外部测试：3D-IRCADb-01
合计：约150例
```

###### 方案B：中等规模（推荐）
```
训练+验证：LiTS + Task03 + CRLM-CT-Seg + HCC-TACE治疗前
外部测试：3D-IRCADb-01
合计：约550-600例
```

###### 方案C：大规模（~1000例）
```
训练+验证：LiTS + Task03 + CRLM-CT-Seg + HCC-TACE + MCT-LTDiag门静脉期
外部测试：3D-IRCADb-01
合计：约1000例（MCT-LTDiag需申请）
```

###### 统一预处理步骤
```
1. 去重：Task03 与 LiTS17 有重叠，通过 patient ID 或图像 hash 排查
2. 标签统一：label=1（肝脏），label=2（肿瘤）
3. 格式统一：全部转为 .nii.gz
4. Spacing重采样：建议 1×1×1 mm（或1×1×2 mm）
5. HU值裁剪：-200 ~ 250 HU
6. 归一化：min-max 或 z-score
```

---

##### 八、关键结论

> **全世界公开CT/MRI影像类、肝脏+肿瘤均有标注的数据集，去除重叠后有效病例总数约1000~1500例**，这是当前领域的客观上限。

> CT类数据集经过预处理**完全可以合并**，理论上能达到接近1000例。

**合并前需想清楚的核心问题：**
- 目标是**通用肝脏肿瘤分割** → 数据多样性是优势，建议方案C
- 目标是**特定类型**（如HCC专项） → 混入太多其他类型可能降低性能，建议方案A/B

> 这也是为什么合成肿瘤数据（Label-Free / SyntheticTumor）成为2023年后重要研究方向——真实标注数据即使充分利用也只有这个量级。


---

## 外部无肿瘤 Case 导入方案


#### 外部无肿瘤 Case 导入方案

##### 背景

Dataset003_Liver 每个 fold 仅 3 个无肿瘤 case，导致类别严重不平衡（肿瘤/无肿瘤 ≈ 9:1）。
引入外部无肿瘤 case 以缓解该问题，同时保证验证集不变、历史实验结果完全可比。

---

##### 数据来源

| 数据集 | 模态 | 无肿瘤 case 数 | 本地路径 |
|--------|------|--------------|---------|
| 3D-IRCADb（case 5/7/11/14/20） | CT | 5 | `/home/PuMengYu/8T/Datasets/3Dircadb1` |
| CHAOS CT（20 个正常肝脏） | CT | 20 | `/home/PuMengYu/8T/Datasets/CHAOS/Train_Sets` |

两个数据集均为腹部 CT，包含肝脏分割标注，无肿瘤标注（label 2 全为 0）。

###### CHAOS 数据集说明

CHAOS（Combined Healthy Abdominal Organ Segmentation）2019 挑战赛数据集，官方文档 `NOTES_PLEASE_READ.txt` 明确说明：

- **CT 模态**：仅标注肝脏，`Any value greater than zero represents the liver`
- **MR 模态**（本方案不使用）：标注肝脏/左肾/右肾/脾脏，用不同像素值区分

因此 CT Ground PNG 中 `> 0` 即肝脏，无需硬编码像素值。

---

##### 完整流程

###### Step 1：格式转换（DICOM → nii.gz）

**IRCADb**

```bash
python pumengyu/tools/external_data/convert_ircad.py \
    --ircad_dir /home/PuMengYu/8T/Datasets/3Dircadb1 \
    --out_dir   /home/PuMengYu/nnUNet_workspace/external_staging/ircad \
    --cases 5 7 11 14 20
```

**CHAOS CT**

```bash
python pumengyu/tools/external_data/convert_chaos.py \
    --chaos_dir /home/PuMengYu/8T/Datasets/CHAOS/Train_Sets \
    --out_dir   /home/PuMengYu/nnUNet_workspace/external_staging/chaos
```

输出格式：`{case_id}_0000.nii.gz`（CT）+ `{case_id}.nii.gz`（seg，label 1=肝脏，label 2=肿瘤全0）

###### Step 2：nnUNet 预处理（nii.gz → .b2nd + .pkl）

由 `inject.py` 内部自动调用 `_preprocess.py`，流程：

1. **Resampling**：各向异性插值到目标 spacing
   - CT 图像：tricubic（order=3）
   - 分割 mask：最近邻（order=0）
2. **CTNormalization**：`clip → z-score`
3. **写入 .b2nd + .pkl**：nnUNet 训练直接读取的格式，同时记录 `class_locations` 用于前景过采样

预处理参数均来自 Dataset003 的 `nnUNetPlans.json` 和 `dataset_fingerprint.json`：

| 参数 | 值 | 来源 |
|------|----|------|
| target spacing (z,y,x) | `[1.0, 0.7676, 0.7676]` mm | nnUNetPlans.json |
| clip min | -15.0 HU | dataset_fingerprint.json (p0.5) |
| clip max | 197.0 HU | dataset_fingerprint.json (p99.5) |
| norm mean | 99.48 | dataset_fingerprint.json |
| norm std | 37.14 | dataset_fingerprint.json |

###### Step 3：注入 splits_final.json

- 备份原始 splits → `splits_final.json.bak_<timestamp>`
- 25 个外部 case 加入所有 fold 的 `train`
- `val` 字段断言检查，任何变动直接报错终止
- 写入 `external_cases_log.json` 记录来源和时间戳

---

##### 一键运行

```bash
cd /home/PuMengYu/nnUNet
bash pumengyu/notes/sh/run_external_import.sh
```

脚本自动检测 CHAOS 目录是否存在，不存在则跳过，**幂等**（重复运行不重复处理）。

---

##### 回退

```bash
#### 确认会删什么
python pumengyu/tools/external_data/eject.py --dry_run

#### 实际回退
python pumengyu/tools/external_data/eject.py
```

回退操作：删除注入的 `.b2nd`/`.pkl`，恢复 splits 备份，清空 log。

---

##### 隔离保证

- 验证集永远只用 Dataset003 原始 case，inject.py 有断言，val 变动直接报错
- 训练使用方式与原有完全相同，无需改 trainer 代码
- 外部 case 命名（`ircad_xxx`、`chaos_xxx`）与原始 case（`liver_xxx`）不冲突

---

##### 注意事项

###### CHAOS CT 肝脏像素值

**踩坑**：`convert_chaos.py` 最初硬编码 `LIVER_PIXEL_VALUE = 55`（照抄 MR 模态的肝脏值），导致所有 case 转换结果 `liver=0 voxels`，分割 mask 全空。

**原因**：CHAOS MR 模态用 63/126/189/252 区分四个器官（肝脏约 63，范围 55-70），CT 模态只有肝脏，像素值为 255（纯白）。

**正确做法**：读官方文档 `NOTES_PLEASE_READ.txt`，CT 明确写明 `Any value greater than zero represents the liver`，代码改为 `ground_arr > 0`，不依赖任何硬编码值。

**验证方法**：转换完成后检查每个 case 的 `liver=X voxels`，正常应在 100万～300万之间（占 CT 体积 4-7%），为 0 则说明 mask 有问题。

###### MR 模态不可用

CHAOS 数据集同时包含 MR（T1DUAL、T2SPIR），不能混入，原因：
- Dataset003 是纯 CT 数据，CTNormalization 参数（HU clip/mean/std）对 MR 无意义
- MR 图像强度与 CT 完全不同，resampling 后分布不匹配

脚本只读 `Train_Sets/CT/` 目录，自动排除 MR。

###### staging 文件必须先删再重跑

如果 staging 目录已有旧的（错误的）nii.gz，直接重跑 `convert_chaos.py` 会覆盖，但 inject 有跳过已存在 `.b2nd` 的逻辑。因此如果之前 inject 过错误数据，必须先 eject 再重新 inject：

```bash
python pumengyu/tools/external_data/eject.py
#### 然后重新跑转换和注入
```

---

##### 文件路径

```
nnUNet_workspace/
├── preprocessed/Dataset003_Liver/
│   ├── splits_final.json               ← 修改此文件（val 不变）
│   ├── splits_final.json.bak_*         ← 自动备份（永久保留）
│   ├── external_cases_log.json         ← 注入记录
│   └── nnUNetPlans_3d_fullres/
│       ├── ircad_005.b2nd / .pkl
│       ├── chaos_001.b2nd / .pkl
│       └── ...
└── external_staging/
    ├── ircad/                          ← IRCADb 转换结果
    └── chaos/                          ← CHAOS 转换结果

pumengyu/tools/external_data/
├── _preprocess.py                      ← preprocessing 核心
├── convert_ircad.py                    ← IRCADb DICOM → nii.gz
├── convert_chaos.py                    ← CHAOS DICOM + PNG → nii.gz
├── inject.py                           ← 注入主脚本
└── eject.py                            ← 回退脚本
```


---
