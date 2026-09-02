# nnUNet v2 核心代码地图


#### nnUNet v2 核心代码地图

> 版本：2.6.4 | 路径：`/home/PuMengYu/nnUNet/nnUNet/nnunetv2/`  
> 目标：快速定位"我要改 X，应该看哪个文件"

---

##### 目录
[toc]

---

##### 1. 项目根目录结构

```
nnUNet/
├── pyproject.toml          # CLI 入口、依赖声明
└── nnunetv2/               # 主包
    ├── paths.py            # 环境变量：nnUNet_raw / preprocessed / results
    ├── configuration.py    # 全局常量：default_num_processes, ANISO_THRESHOLD=3.0
    ├── training/           # 训练
    ├── inference/          # 推理
    ├── experiment_planning/# 数据指纹 + 实验规划
    ├── preprocessing/      # 离线预处理
    ├── utilities/          # plans解析、label管理、网络初始化
    ├── evaluation/         # 指标计算
    ├── postprocessing/     # 后处理
    ├── ensembling/         # 多模型集成
    ├── imageio/            # 多格式图像读写
    ├── run/                # 训练入口
    └── dataset_conversion/ # 数据集格式转换
```

---

##### 2. 训练核心 (`training/`)

###### 2.1 主 Trainer

| 文件 | 行数 | 职责 |
|------|------|------|
| `training/nnUNetTrainer/nnUNetTrainer.py` | **1414行** | 训练循环的一切：DDP初始化、损失构建、数据加载、前后向、LR调度、checkpoint |

**关键方法（行号）：**
- `:151` `num_iterations_per_epoch = 250` — 固定步数/epoch
- `:149` `oversample_foreground_percent = 0.33` — 前景过采样比例
- `:416` `_build_loss()` — 构建 Dice+CE 损失 + deep supervision 包装
- `:472` `get_training_transforms()` — 增广参数（旋转角、mirror轴等）
- `:531` `configure_optimizers()` — SGD(lr=1e-2, momentum=0.99, nesterov=True) + PolyLR
- `:738` `SpatialTransform(p_elastic_deform=0)` — 关闭弹性形变
- `:806` `MirrorTransform(allowed_axes=(0,1,2))` — 三轴镜像

###### 2.2 损失函数 (`training/loss/`)

| 文件 | 关键类 |
|------|--------|
| `dice.py` | `SoftDiceLoss`、`MemoryEfficientSoftDiceLoss`（支持 batch_dice、DDP 聚合）|
| `compound_losses.py` | `DC_and_CE_loss`（Dice+CE，默认各权重1）、`DC_and_BCE_loss` |
| `robust_ce_loss.py` | `RobustCrossEntropyLoss`、`TopKLoss` |
| `deep_supervision.py` | `DeepSupervisionWrapper`（按 `1/2^i` 权重加权各尺度loss）|

**nnUNet 默认 loss 公式：**
```
total = Dice(smooth=1e-5, batch_dice=True, do_bg=False) + CE
```

###### 2.3 数据加载 (`training/dataloading/`)

| 文件 | 关键类 | 说明 |
|------|--------|------|
| `nnunet_dataset.py` | `nnUNetDatasetBlosc2` | 读 `.b2nd` 格式，按 chunk 随机读 patch，节省内存 |
| `data_loader.py` | `nnUNetDataLoader` | 前景过采样、patch 裁剪、padding，基于 batchgenerators |
| `utils.py` | — | dataset 解包、dataloader 初始化工具 |

###### 2.4 学习率调度 (`training/lr_scheduler/`)

| 文件 | 公式 |
|------|------|
| `polylr.py` | `lr_t = lr_init × (1 - t/T_max)^0.9` |
| `warmup.py` | 支持 constant/linear/exponential warmup |

###### 2.5 数据增强 (`training/data_augmentation/`)

增广参数全部在 `nnUNetTrainer.py` 里，这里只有辅助工具：

| 文件 | 说明 |
|------|------|
| `compute_initial_patch_size.py` | 从 fingerprint 计算初始 patch size |
| `custom_transforms/` | Cascade变换、mask处理、deep supervision标签降采样、pseudo-2D |

---

##### 3. 预处理核心 (`preprocessing/`)

###### 整体流程

```
nii.gz
  ↓ 1. DatasetFingerprintExtractor  → dataset_fingerprint.json
  ↓ 2. DefaultPreprocessor.run_case_npy（每个case）
       ├── 2.1 transpose（轴重排，统一朝向）
       ├── 2.2 crop_to_nonzero（裁掉空气背景，保存bbox）
       ├── 2.3 normalize（归一化，必须在resample之前！）
       ├── 2.4 resample to target_spacing
       └── 2.5 _sample_foreground_locations（预存前景坐标，训练时直接用）
  ↓ 输出：case.b2nd + case_seg.b2nd + case.pkl（properties）
```

**注意：normalize 必须在 resample 之前** — 见 `default_preprocessor.py:109` 的注释

###### 关键文件

| 文件 | 行数 | 说明 |
|------|------|------|
| `preprocessors/default_preprocessor.py` | **626行** | 完整预处理 pipeline 编排 |
| `cropping/cropping.py` | 40行 | `crop_to_nonzero()`，保存 bbox 供推理还原 |
| `normalization/default_normalization_schemes.py` | 80行 | `CTNormalization`（clip→zscore，用全局统计）、`ZScoreNormalization`（per-case）|
| `resampling/default_resampling.py` | — | 各向异性判断（ANISO_THRESHOLD=3.0），separate_z 两步重采样 |
| `resampling/resample_torch.py` | — | PyTorch GPU 加速重采样 |

###### CT vs MRI 归一化区别

| | CT | MRI |
|--|----|----|
| 统计来源 | 全训练集前景体素 | per-case |
| 步骤 | clip[p0.5, p99.5] → zscore | zscore（可选只在前景mask内）|
| 触发 mask_for_norm | — | `median_relative_size < 0.75` |

---

##### 4. 实验规划核心 (`experiment_planning/`)

```
dataset_fingerprint.json
  ↓ ExperimentPlanner
  ↓ 输出 plans.json（网络拓扑、patch_size、spacing、配置名）
```

| 文件 | 行数 | 说明 |
|------|------|------|
| `dataset_fingerprint/fingerprint_extractor.py` | 211行 | 统计所有case的spacing/shape/前景强度，输出 fingerprint.json |
| `experiment_planners/default_experiment_planner.py` | **593行** | 自动决定 patch_size、pool层数、特征图数、重采样策略，生成 plans.json |
| `plan_and_preprocess_api.py` | 159行 | 高层 API：`plan_and_preprocess_dataset()` 一键完成 |

**fingerprint.json 关键字段：**

| 字段 | 用途 |
|------|------|
| `spacings` | 计算 target_spacing（取中位数）|
| `shapes_after_crop` | 决定 patch_size 上限 |
| `mean/std/percentile_00_5/99_5` | CT 归一化参数 |
| `median_relative_size_after_cropping` | 判断是否需要 mask_for_norm |

---

##### 5. 推理核心 (`inference/`)

```
trained_model/ + raw_image
  ↓ nnUNetPredictor.initialize_from_trained_model_folder()
  ↓ predict_from_files()
       ├── 预处理（同训练：transpose→crop→normalize→resample）
       ├── 滑窗推理 + 高斯加权（overlap=0.5）
       ├── 多fold结果平均
       └── 导出（resample回原spacing，pad回原shape）
```

| 文件 | 行数 | 关键类/函数 |
|------|------|------------|
| `predict_from_raw_data.py` | **1065行** | `nnUNetPredictor`，完整推理接口 |
| `sliding_window_prediction.py` | 65行 | `compute_gaussian()`，`compute_steps_for_sliding_window()` |
| `export_prediction.py` | 149行 | logits→seg，resample回原坐标，多格式导出 |
| `data_iterators.py` | 315行 | 数据预处理适配器，支持多线程 |

---

##### 6. 辅助工具 (`utilities/`)

| 文件 | 关键类 | 说明 |
|------|--------|------|
| `plans_handling/plans_handler.py` | `PlansManager`、`ConfigurationManager` | 解析 plans.json，获取 patch_size、spacing、网络参数 |
| `label_handling/label_handling.py` | `LabelManager` | 管理类别映射，控制 softmax vs sigmoid，region-based training |
| `get_network_from_plans.py` | — | 从 plans.json 动态实例化网络架构 |

---

##### 7. 网络架构

nnUNet v2 **不在本仓库内**定义网络结构，而是依赖外部包：

```
pip install dynamic-network-architectures
```

支持的网络类：
- `PlainConvUNet`：标准 3D UNet
- `ResidualEncoderUNet`：残差编码器 UNet

网络参数全部存在 `plans.json` 的 `architecture` 字段中，由 `get_network_from_plans.py` 动态加载。

---

##### 8. CLI 入口 (pyproject.toml)

| 命令 | 对应函数 |
|------|---------|
| `nnUNetv2_plan_and_preprocess` | 一键指纹提取+规划+预处理 |
| `nnUNetv2_train` | `run/run_training.py` → `run_training_entry()` |
| `nnUNetv2_predict` | `inference/predict_from_raw_data.py` |
| `nnUNetv2_find_best_configuration` | 选最佳配置（3d_fullres/3d_lowres/2d）|
| `nnUNetv2_ensemble` | 多模型/多fold集成 |
| `nnUNetv2_evaluate_folder` | 计算 Dice/HD95 等指标 |

---

##### 9. 环境变量（必须设置）

```bash
export nnUNet_raw="/data/nnUNet_raw"
export nnUNet_preprocessed="/data/nnUNet_preprocessed"
export nnUNet_results="/data/nnUNet_results"
```

定义位置：`nnunetv2/paths.py`

---

##### 10. 本仓库自定义扩展

这份 nnUNet 源码里有针对肝脏肿瘤任务的自定义代码：

| 文件 | 内容 |
|------|------|
| `training/loss/my_loss/tumor_loss.py` | `SimpleTumorLoss`（骨架，待实现）|
| `training/nnUNetTrainer/my_traniners/nnUNetTrainer_MyTumor.py` | 继承 nnUNetTrainer，覆写 `_build_loss()` |
| `postprocessing/my_postprocess/tumor_postprocess.py` | `keep_tumor_inside_liver()`、`remove_small_components()` |

---

##### 11. 快速定位索引

| 想改什么 | 看哪个文件 |
|---------|-----------|
| 数据增强参数 | `training/nnUNetTrainer/nnUNetTrainer.py:472~810` |
| 损失函数 | `training/loss/compound_losses.py` + `dice.py` |
| 学习率 / 优化器 | `nnUNetTrainer.py:531` + `training/lr_scheduler/polylr.py` |
| 前景过采样比例 | `nnUNetTrainer.py:149` |
| 固定步数/epoch | `nnUNetTrainer.py:151` |
| CT归一化参数 | `preprocessing/normalization/default_normalization_schemes.py:58` |
| 各向异性判断阈值 | `nnunetv2/configuration.py` (`ANISO_THRESHOLD`) |
| patch size 计算 | `experiment_planning/experiment_planners/default_experiment_planner.py` |
| 滑窗推理 overlap | `inference/predict_from_raw_data.py` |
| Deep supervision 权重 | `nnUNetTrainer.py:597` + `training/loss/deep_supervision.py` |
| blosc2 格式读写 | `training/dataloading/nnunet_dataset.py` |
| 前景坐标预存 | `preprocessing/preprocessors/default_preprocessor.py:239` |


---
