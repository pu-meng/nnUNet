# EfficientMedNeXt-L 官方 baseline 实施记录

更新时间：2026-07-16

## 1. 目的

在与 MedNeXt-L、MedNeXt-L+MLA 相同的 nnU-Net v2 数据、训练和评估流水线中，
引入 EfficientMedNeXt-L 作为效率—效果权衡 baseline。

本 baseline 只替换网络结构。数据划分、预处理、数据增强、loss、优化器、学习率
计划、滑窗推理、指标、报告和可视化均复用当前仓库的统一实现。

## 2. 官方来源与许可证

- 官方仓库：`https://github.com/SLDGroup/EfficientMedNeXt`
- 固定 commit：`803f7efed9b728ac93ae4e0d8a2602501135241f`
- 本地官方仓库：`/home/PuMengYu/EfficientMedNeXt`
- 许可证：UT Austin Research License，仅按学术/研究用途使用。

当前适配器不复制官方网络源码，而是从独立官方仓库加载，并校验以下三个
EfficientMedNeXt-L 核心文件的 SHA-256：

| 文件 | SHA-256 |
| --- | --- |
| `EfficientMedNext_Full.py` | `a7b8348534bddfcac949f56e03af2fa67e91fab45f4b5d388341d2dad456d7c8` |
| `efficient_mednext_blocks.py` | `3e12e9de5d85b2582c03a7d791902711085411eea18564c9aaf9a032bac85c4d` |
| `create_efficient_mednext.py` | `79810104a3b69f5bc5cf4c83165226dcad9ced6027b7a2c0d9090f89e29231a0` |

文件不存在或哈希变化时，构建会显式失败，不允许静默使用来源不明的结构。

## 3. 固定网络配置

- 模型：官方 `EfficientMedNeXt-L`
- 输入通道：由 nnU-Net dataset plans 决定；Dataset003 为 1
- 输出通道：由 label manager 决定；Dataset003 为 3
- base channels：32
- uniform decoder channels：32
- multi-receptive kernel specification：`[1, 3, 5]`
- 实际 DMRFB 分支：1×1×1、3×3×3 dilation 1、3×3×3 dilation 2
- block counts：`[3, 4, 4, 4, 4, 4, 4, 4, 3]`
- residual blocks：开启
- residual down/up：开启
- deep supervision：开启
- deep-supervision 输出：full、1/2、1/4、1/8、1/16

Dataset003（1 输入通道、3 输出类别）实测参数量：`2,193,808`。

## 4. 代码入口

- architecture adapter：
  `pumengyu/architectures/efficient_mednext_official.py`
- trainer：
  `pumengyu/trainers/efficient_mednext_trainer.py`
- trainer class：
  `nnUNetTrainer_EfficientMedNeXt_L_Official`

## 5. 已完成验证

- Python 语法检查：通过；
- 官方 L 版核心文件哈希检查：通过；
- nnU-Net external trainer 自动发现：通过；
- CPU 前向传播：通过；
- deep-supervision 五级输出尺寸：通过；
- 关闭 deep supervision 后单输出：通过；
- CPU 反向传播及参数梯度：通过。

GPU 显存、FLOPs、训练吞吐和推理时间：尚未验证。

## 6. 正式训练命令

```bash
cd /home/PuMengYu/nnUNet
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 3 3d_fullres 0 \
  -tr nnUNetTrainer_EfficientMedNeXt_L_Official
```

训练前若官方仓库不在默认兄弟目录，可显式设置：

```bash
export EFFICIENT_MEDNEXT_ROOT=/home/PuMengYu/EfficientMedNeXt
```

## 7. 当前实验状态

- 预期方法总数：1；
- 方法：EfficientMedNeXt-L Official；
- 训练：未开始；
- 当前运行：无；
- 内部测试预测病例数：0，未开始；
- IRCADb 外部验证预测病例数：0，未开始；
- `checkpoint_best.pth`：不存在，来源检查尚不适用；
- `summary.json`：不存在；
- `.txt` 报告：不存在；
- `test_viz` PNG：0；
- 完整实验产物：未完成。

正式训练完成后，只有预测、`summary.json`、`.txt` 报告、`test_viz/` PNG 和
checkpoint 来源信息全部通过检查，才能将该 baseline 标记为“实验完成”。

