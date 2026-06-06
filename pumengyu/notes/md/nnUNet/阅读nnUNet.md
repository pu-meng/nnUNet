
# nnUNet 代码阅读指南

## 读代码之前你需要知道的一件事

nnUNet 的难点不是 PyTorch，而是**“代码和算法决策的对应关系”太隐蔽**。

比如你读到这一行：
```python
self.oversample_foreground_percent = 0.33
```
如果没人告诉你，你不知道这是“前景过采样”，不知道为什么是 0.33，不知道它在哪里被用到。

**所以正确的姿势：带着问题读，不要从头到尾读。**

---

## 建议的起点：先读论文的一部分

nnUNet 有一篇 Nature Methods 2021 的论文，不需要全读，只看这些：

- **Abstract（摘要）**：2 分钟，知道它在解决什么问题
- **Figure 1**：一张图解释整个流程
- **“Rule-based pipeline” 那一节**：列了 nnUNet 的 5 个自动决策规则

有了这个背景，再读代码，每一行都有意义。没有这个背景，代码就是一堆配置数字。

---

## 按顺序读代码

### 第 1 步：最简单的两个文件（30 分钟）

先读最短的，建立信心：

- `nnunetv2/configuration.py` ~ 11 行
- `nnunetv2/paths.py` ~ 40 行

读完之后你知道：`ANISO_THRESHOLD = 3.0` 是什么意思，三个环境变量放什么数据。

---

### 第 2 步：预处理（最好读的算法代码，1-2 天）

这 3 个文件逻辑直线，没有复杂的类继承：

1. `preprocessing/cropping/cropping.py` ~ 40 行 ← **先读这个**
2. `preprocessing/normalization/default_normalization_schemes.py` ~ 80 行
3. `preprocessing/resampling/default_resampling.py` 前 150 行（读到 `separate_z` 为止）

**读的时候问自己：**
- crop 的目的是什么？存了什么信息供后续用？
- CT 和 MRI 的归一化为什么不一样？
- 各向异性的 CT（z 轴分辨率很低）重采样时为什么要特殊处理？

---

### 第 3 步：训练循环——从最小的部分开始（2-3 天）

不要从头读 `nnUNetTrainer.py`，它有 1414 行，开头全是初始化样板，读了会放弃。

按这个顺序，**每次只读一个方法**：

#### 第 1 天
- `nnUNetTrainer.py:413~448`  `_build_loss()` → 用什么损失
- `nnUNetTrainer.py:529~534`  `configure_optimizers()` → SGD + PolyLR，只有 6 行

#### 第 2 天
- `nnUNetTrainer.py:995~1026` `train_step()` → 一步训练干了什么
- `nnUNetTrainer.py:1042~1108` `validation_step()` → 验证步

#### 第 3 天
- `nnUNetTrainer.py:368~412`  `_set_batch_size_and_oversample()` → 前景过采样
- `nnUNetTrainer.py:716~867`  `get_training_transforms()` → 8 种增广

---

### 第 4 步：自动规划（最难，放最后，1 周）

- `experiment_planning/dataset_fingerprint/fingerprint_extractor.py` ~ 211 行，**全读**
- `experiment_planning/experiment_planners/default_experiment_planner.py`
  - 重点读：
    - `:155~197`  `determine_fullres_target_spacing()`
    - `:228~404`  `get_plans_for_configuration()`
    - `:405~542`  `plan_experiment()`

这部分是 nnUNet 和普通 UNet 差距最大的地方，也是最难读的。放最后，有了前面的基础再来。

---

## 一个实用技巧

读每个方法之前，先在终端里跑一遍对应的功能，看输出：

```bash
# 看 fingerprint 长什么样
cat $nnUNet_preprocessed/Dataset003_Liver/dataset_fingerprint.json

# 看 plans 长什么样
cat $nnUNet_preprocessed/Dataset003_Liver/nnUNetPlans.json
```

**先看输出，再读生成这个输出的代码**，比直接啃代码快 3 倍。

---

## 总结

| 阶段 | 内容 | 时间估计 |
|------|------|----------|
| 0    | 读论文 Figure 1 + 规则部分 | 1 小时 |
| 1    | configuration + paths | 30 分钟 |
| 2    | 预处理 3 个文件 | 1-2 天 |
| 3    | 训练循环 6 个方法 | 2-3 天 |
| 4    | 自动规划 | 1 周 |

**不要一口气全读**，每天读一个方法，搞清楚为什么这样写，比快速过一遍强很多。

