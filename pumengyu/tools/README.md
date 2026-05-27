# pumengyu/tools — 工具集说明

这个目录是一整条**「训练 → 报告 → 分析 → 难度 → 再训练」闭环**的脚本集合：训练验证后把 nnUNet 的原始 `summary.json` 加工成可读报告，再汇总成可分析的表，最后把内在特征变成喂回训练的难度权重。

---

## 数据流总览

```
训练验证 → fold_X/validation/summary.json              (nnUNet 原生产物)
   │  AutoReportMixin 训练后自动调 analyasis/auto_report.py
   ├─ gen_report_json.py    → report_custom.json         (per-case Dice)
   └─ eval_fold_report.py   → report_custom.txt + vis_png_custom/

原始 CT + GT
   └─ feature_extraction/extract_intrinsic_features.py
              → intrinsic_features.json                  (分布级内在特征，零泄露)
                    │
          ┌─────────┴──────────────────────────┐
   compute_difficulty.py          feature_representativeness.py
   → difficulty.json              → feature_representativeness.txt
      ↓ 喂难度加权 CopyPaste         找最强难度预测指标

   (独立) data_analysis/size_density_analysis.py
   → size_density_comparison.txt   (多方法 Dice × 大小/密度对比)
```

---

## 目录结构

```
tools/
├── run_analysis.py                      一键启动分析流水线
├── analyasis/                           报告生成（训练后自动触发）← 被 mixins.py 依赖，勿改名
│   ├── auto_report.py                   统一出口
│   ├── gen_report_json.py               summary.json → report_custom.json
│   └── eval_fold_report.py              summary.json → report_custom.txt + 可视化
├── feature_extraction/                  从原始体素提取分布级特征（Step 1）
│   └── extract_intrinsic_features.py   → intrinsic_features.csv / .json
├── data_analysis/                       手动运行的分析脚本
│   ├── compute_difficulty.py            per-case 难度权重 → difficulty.json（Step 2）
│   ├── feature_representativeness.py    特征代表性排名（Step 3）
│   ├── size_density_analysis.py         体积×密度分组，多方法 Dice 对比
│   └── update_hu_analysis_fold2.py      补全 hu_analysis.txt（历史脚本）
├── readme_style.css                     PDF 导出样式
└── README.md
```

---

## 一键启动（推荐入口）

```bash
# 首次完整运行（按顺序跑三步）
python pumengyu/tools/run_analysis.py

# Step1 先用5个 case 试跑，确认无报错
python pumengyu/tools/run_analysis.py --limit 5 --only 1

# 已有 intrinsic_features.json，跳过 Step1
python pumengyu/tools/run_analysis.py --skip 1

# 跳过 Step1，且 Step2 额外做难度 vs Dice 体检
python pumengyu/tools/run_analysis.py --skip 1 --check

# 只重新跑特征排名
python pumengyu/tools/run_analysis.py --only 3
```

---

## 各脚本详解

### A. 报告生成 — `analyasis/`（训练后自动触发）

#### `auto_report.py`
- **作用**：训练结束后生成报告的**统一出口**；内部依次调 `gen_report_json` + `eval_fold_report`。自动检测数据集模式（liver_tumor: 1=肝/2=肿瘤；tumor_only: 1=肿瘤）。
- **谁调它**：`pumengyu/mixins.py` 的 `AutoReportMixin.perform_actual_validation` 在验证后**自动调用**，一般无需手动跑。
- **函数接口**：`run_auto_report(fold_dir, gt_dir, img_dir)`
- **产物**：`report_custom.json` + `report_custom.txt` + `vis_png_custom/`

#### `gen_report_json.py`
- **作用**：从 `fold_X/validation/summary.json` 抽取每个 case 的 Dice。
- **产物**：`fold_X/report_custom.json`（`case` / `dice_liver` / `dice_cancer`；无肿瘤 case 的 `dice_cancer=None`）
- **单独运行**：`python pumengyu/tools/analyasis/gen_report_json.py --fold_dir <fold_X>`

#### `eval_fold_report.py`
- **作用**：从 `summary.json` 生成可读报告（有/无肿瘤分开，按 cancer_dice 分级）+ 每 case 的 GT/Pred/Diff 切片图。
- **输入**：`--val_dir <fold_X/validation>` `--gt_dir <preprocessed/.../gt_segmentations>` `--img_dir <raw/.../imagesTr>` `[--vis_slices 5] [--no_vis] [--min_tumor_size 0]`
- **产物**：`report_custom.txt` + `vis_png_custom/`

---

### B. 分析流水线 — Step 1→2→3

#### Step 1 · `feature_extraction/extract_intrinsic_features.py`
- **作用**：直接读**原始 CT（原始 HU）+ GT**，对每个 case 用**完整 HU 分布**算约 20 个分布级特征（CNR、Mann-Whitney AUC、Bhattacharyya、直方图 overlap、偏度/峰度、Sarle 双峰系数、体积等）。纯内在、零泄露，是 Step2/3 的数据基础。
- **输入**：`raw/.../imagesTr/{case}_0000.nii.gz` + `preprocessed/.../gt_segmentations/{case}.nii.gz`
- **产物**：`notes/实验结果分析/intrinsic_features.csv` + `.json`
- **运行**：`python pumengyu/tools/feature_extraction/extract_intrinsic_features.py [--limit N]`

#### Step 2 · `data_analysis/compute_difficulty.py`
- **作用**：给每个有肿瘤 case 算**纯内在、零泄露**的连续难度权重，供 `DifficultyCopyPasteMixin` 使用。
- **公式**：`difficulty = rank_norm(DIFFICULTY_FEAT)`，`weight = 0.05 + 0.95 × difficulty`
- **换指标**：修改脚本顶部 `DIFFICULTY_FEAT = 'hist_overlap'` 一行即可。
- **输入**：`intrinsic_features.json`（Step1 产物）
- **产物**：`notes/实验结果分析/difficulty.json`（`{case: weight}`）
- **运行**：
  ```bash
  python pumengyu/tools/data_analysis/compute_difficulty.py
  python pumengyu/tools/data_analysis/compute_difficulty.py --check  # 体检：难度 vs OOF Dice 相关性
  ```

#### Step 3 · `data_analysis/feature_representativeness.py`
- **作用**：对 intrinsic_features.json 里的全部 ~20 个特征，与 OOF Dice 做 Spearman ρ 排名（带 95% CI），找出**最能预测分割失败**的指标，指导下一步改进方向。
- **输入**：`intrinsic_features.json` + 各折 `report_custom.json`
- **产物**：`notes/实验结果分析/feature_representativeness.txt`
- **运行**：`python pumengyu/tools/data_analysis/feature_representativeness.py`

---

### C. 独立分析脚本

#### `data_analysis/size_density_analysis.py`
- **作用**：按肿瘤大小（ml/体素）和 HU 对比度分组，对比各方法（Baseline / TwoStage / UFL / CopyPaste）的 Dice 均值、Dice<0.5 比例、体积×密度二维矩阵。
- **输入**：`hu_analysis.txt`（内在元数据）+ 各方法各折 `report_custom.json`
- **产物**：`notes/实验结果分析/size_density_comparison.txt`
- **运行**：`python pumengyu/tools/data_analysis/size_density_analysis.py`

#### `data_analysis/update_hu_analysis_fold2.py`
- **作用**：补全 fold_2 的 Dice 并重建 `hu_analysis.txt`（历史脚本，维护旧汇总表用）。
- **运行**：`python pumengyu/tools/data_analysis/update_hu_analysis_fold2.py`

---

## 关键数据文件速查

| 文件 | 路径 | 内容 |
|---|---|---|
| `summary.json` | `results/<Dataset>/<Trainer>/fold_X/validation/` | nnUNet 原生 per-case 指标 |
| `report_custom.json` | `results/<Dataset>/<Trainer>/fold_X/` | per-case `dice_liver`/`dice_cancer` |
| `intrinsic_features.json` | `notes/实验结果分析/` | ~20 个分布级内在特征（Step1 产物） |
| `difficulty.json` | `notes/实验结果分析/` | per-case 难度权重（喂 CopyPaste） |
| `feature_representativeness.txt` | `notes/实验结果分析/` | 各特征与 Dice 相关性排名（带 CI） |
| `size_density_comparison.txt` | `notes/实验结果分析/` | 体积×密度多方法对比表 |
| `hu_analysis.txt` | `notes/实验结果分析/` | 历史汇总表（LiTS 专用，新流程不依赖） |

---

## 备注 / 待办

- `analyasis/` 是历史拼写（应为 analysis），且**被 `mixins.py:41` 依赖**；改名需同步更新引用，建议无实验运行时再做。
- `DIFFICULTY_FEAT` 当前用 `hist_overlap`，后续查文献选定更优指标后改这一行即可，其余逻辑不变。
- `size_density_analysis.py` 仍读旧的 `hu_analysis.txt`，后续可迁移到读 `intrinsic_features.json`。
- Python 脚本一律由用户手动运行（项目约定）。
