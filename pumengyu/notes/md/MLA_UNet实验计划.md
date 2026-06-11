# MLA-UNet 实验计划

> Dataset003_Liver，nnUNet v2 3d_fullres  
> 优先用 fold_4 快速筛选，fold_0 交叉验证，两折一致才推全量  
> 最后更新：2026-06-09

---

## 一、核心想法与实现

### 动机

标准卷积 UNet 的瓶颈层缺少全局感受野——卷积核只能捕捉局部模式，对跨区域的长程依赖建模能力有限。
把 Transformer attention 插入瓶颈是自然的想法，但 standard self-attention 在瓶颈层的 token 数通常为 4³=64（轻量），显存不是主要矛盾；而一旦想用更小 patch（如 8³）或更浅池化，token 数会剧增，此时 KV 显存就成为瓶颈。

**MLA（Multi-head Latent Attention，DeepSeek-V2, 2024）的核心**：

```
标准 MHA：K = W_K · x  (B, N, d)
          V = W_V · x  (B, N, d)   →  KV 存储 O(N · d)

MLA：     c_kv = W_DKV · x        (B, N, d_c)  ← 低秩压缩，d_c = d // 4
          K = W_UK · c_kv         (B, N, d)
          V = W_UV · c_kv         (B, N, d)   →  实际只存 c_kv，O(N · d_c)
```

attention 矩阵仍是完整 N×N（full attention，全局感受野不变），KV 参数量和激活显存降低 4×。

### 与 UMamba 的本质区别

| | UMamba（已放弃）| MLA-UNet |
|--|----------------|----------|
| 全局感受野 | 否（Mamba 是因果序列模型） | **是**（full attention） |
| 在 64 token 瓶颈层的收益 | 几乎没有 | **有**（64 token 全局 attention 意义明确） |
| 失败风险 | 已确认（fold_4 Dice −4pp）| 未知，需实验验证 |

> **教训**：UMamba 放弃的原因是 bottleneck-only Mamba 在 64 token 上无额外收益，且 Mamba 对空间数据的表达不如 attention。MLA 的全局 attention 在原理上更适合此处，但实验结论仍以数字为准。

### 已实现的代码

```
pumengyu/architectures/mla_unetr.py  → MultiHeadLatentAttention, MLABottleneck3D, MLAUNetBot3D
pumengyu/trainers/trainer.py         → nnUNetTrainer_MLAUNet
                                       nnUNetTrainer_MLAUNet_deeper
                                       nnUNetTrainer_MLAUNet_SizeOversample
```

默认超参：`num_heads=8, num_blocks=2, compression_ratio=4, mlp_ratio=4`

---

## 二、实验设计

### 实验总览

| # | Trainer | 变量 | 对照 | 目的 |
|---|---------|------|------|------|
| E1 | `nnUNetTrainer_MLAUNet` | MLA 瓶颈 | Baseline | MLA 是否有基础收益 |
| E2 | `nnUNetTrainer_MLAUNet_deeper` | MLA × 4 block | E1（× 2 block） | 更深 MLA 是否过拟合或更优 |
| E3 | `nnUNetTrainer_MLAUNet_SizeOversample` | MLA + 过采样 | SizeOversampleV2 | 架构改进能否叠加数据策略 |

**不与 UMamba 做对照**：UMamba 已确认无效，MLA 的参照系是 Baseline 和 SizeOversampleV2。

### 阶段规划

```
Phase 1：快速筛选（fold_4，~900 epoch，约 1~2 天/实验）
    E1 → fold_4 Dice 与 Baseline(0.610) / UFL(0.639) 比较
    决策门控 → Dice > 0.60 且不高于 UFL 太多则继续

Phase 2：交叉验证（fold_0，前提是 Phase 1 fold_4 > 0.61）
    E1 fold_0 → 与 Baseline fold_0(0.706) 比较
    决策门控 → fold_0 Dice > 0.706 才继续 E2/E3

Phase 3：组合实验（前提是 Phase 2 通过）
    E2（deeper）与 E3（+SizeOversample）并行跑 fold_4
    E3 若改善 FP 且维持 Dice，推全量 5-fold CV
```

---

## 三、训练命令

### Phase 1：fold_4 快速筛选

```bash
# E1：基础 MLA（必跑，第一优先级）
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train Dataset003_Liver 3d_fullres 4 \
    -tr nnUNetTrainer_MLAUNet

# 训练过程监控（另一个终端）
tail -f ~/nnUNet_workspace/results_v2/Dataset003_Liver/\
nnUNetTrainer_MLAUNet__nnUNetPlans__3d_fullres/fold_4/training_log*.txt
```

### Phase 2：fold_0 交叉验证（E1 fold_4 结果达标后执行）

```bash
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train Dataset003_Liver 3d_fullres 0 \
    -tr nnUNetTrainer_MLAUNet
```

### Phase 3：消融与组合（Phase 2 通过后执行）

```bash
# E2：MLA × 4 block
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train Dataset003_Liver 3d_fullres 4 \
    -tr nnUNetTrainer_MLAUNet_deeper

# E3：MLA + 大小分层过采样（V2 倍数：极小×6，小×5，无肿瘤×6）
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train Dataset003_Liver 3d_fullres 4 \
    -tr nnUNetTrainer_MLAUNet_SizeOversample
```

### 推理（单折评估用，训练完成后执行）

```bash
RESULTS_DIR=~/nnUNet_workspace/results/Dataset003_Liver

# E1 fold_4 推理
CUDA_VISIBLE_DEVICES=0 nnUNetv2_predict \
    -i ~/nnUNet_workspace/raw/Dataset003_Liver/imagesTr \
    -o ${RESULTS_DIR}/nnUNetTrainer_MLAUNet__nnUNetPlans__3d_fullres/fold_4/internal_test \
    -d Dataset003_Liver -c 3d_fullres -tr nnUNetTrainer_MLAUNet -f 4
```

---

## 四、决策逻辑

```
E1 fold_4 Dice ≥ 0.61 且 FP率 ≤ 33%？
  ├── YES → 推进 Phase 2（fold_0）
  └── NO  → MLA bottleneck 无效，与 UMamba 同命运，放弃整条线

E1 fold_0 Dice > 0.706（Baseline fold_0）？
  ├── YES → 启动 E2（deeper）+ E3（+SizeOversample），并行 fold_4
  └── NO  → 仅 fold_4 正向 → 不稳定，暂停；分析 fold_0 失败原因

E3 fold_4 FP率 ≤ 33% 且 Dice ≥ SizeOversampleV2？
  ├── YES → 推全量 5-fold，作为主线候选
  └── NO  → MLA 对 FP 无帮助，记录结论，不推全量
```

---

## 五、结果记录表（训练完成后填写）

| Trainer | fold | Dice | vs Baseline | FP率 | 小肿瘤Dice | 极小Dice | 结论 |
|---------|------|------|-------------|------|-----------|---------|------|
| **Baseline** | 4 | 0.610 | — | 33% | — | — | 参考 |
| **Baseline** | 0 | 0.706 | — | — | — | — | 参考 |
| **UFL（当前最优）** | 4 | 0.639 | +2.9pp | 33% | — | 0.406 | 参考 |
| nnUNetTrainer_MLAUNet | 4 | — | — | — | — | — | Phase 1 |
| nnUNetTrainer_MLAUNet | 0 | — | — | — | — | — | Phase 2 |
| nnUNetTrainer_MLAUNet_deeper | 4 | — | — | — | — | — | Phase 3 |
| nnUNetTrainer_MLAUNet_SizeOversample | 4 | — | — | — | — | — | Phase 3 |

---

## 六、风险与预期

**最可能的失败模式**：和 UMamba 类似，瓶颈层 token 数（4³=64）太少，full attention 在 64×64 矩阵上能学到的全局模式有限，改善量不显著。

**最可能的成功模式**：attention 显式建模了 64 个 patch 之间的所有成对关系，对于肝脏这类形状高度可变的器官，全局上下文有助于判断边界位置，小肿瘤检测间接受益。

**如果 bottleneck-only 失败的后续方向**：参考原版 UNETR（每层 encoder 都插 Transformer），在 encoder 最后 2 层插 MLA block，而不仅仅是最深层。代价是需要重写 encoder，复杂度上升，但理论依据更充分。
