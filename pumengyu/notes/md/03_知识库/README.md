# 知识库

按问题查阅，不必从头读。当前实验状态和论文结论分别看 [当前项目](../01_当前项目/README.md) 与 [实验档案](../02_实验档案/README.md)；这里保留机制、代码和方法论。旧实验命令不作为当前执行入口。

## nnU-Net 与 Trainer

| 要查什么 | 入口 |
|---|---|
| 改代码先定位文件 | [nnUNet 代码地图](nnUNet代码地图.md) |
| 数据、预处理、训练、推理的全流程 | [nnUNet pipeline](nnUNet_pipeline.md) |
| PlainConvUNet 与 UMamba 的层级 | [nnUNet 架构笔记](nnUNet架构笔记.md) |
| Trainer 扩展点与旧方法登记 | [Trainer 架构改动](Trainer架构改动.md)、[Trainer / Loss / Batch Size](Trainer_Loss_BatchSize.md) |
| 输入与特征归一化 | [nnUNet pipeline](nnUNet_pipeline.md)、[nnUNet 架构笔记](nnUNet架构笔记.md) |
| 镜像训练与推理 | [镜像翻转](镜像翻转.md) |

## MedNeXt 与卷积

- 主结构：[MedNeXt 架构](MedNeXt架构.md)；设计脉络：[ConvNeXt / MedNeXt 对比](ConvNeXt_MedNeXt对比.md)。
- 旧卷积替换的结构和效率边界：[DWSep 与 IBConv](DWSep_vs_IBConv机制.md)；独立架构：[DeepResGN](DeepResGN架构.md)、[EfficientMedNeXt](EfficientMedNeXt架构与实验边界.md)。
- 按机制检索：[架构机制方法池](架构机制.md)、[通用方法](通用方法.md)。两篇覆盖的机制类别不同，均为备查材料。

## GPU 与工程

- 硬件概念：[GPU 性能基础](GPU性能基础.md)；命令：[NCU / nsys](NCU_nsys命令.md)；实测案例：[GPU 实战案例](GPU实战案例.md)。
- 原理细节：[Tensor Core 与卷积实现](TensorCore与卷积实现.md)；训练行为：[torch.compile 与 MedNeXt 深监督](torch_compile与MedNeXt深监督问题.md)。

## 研究问题与论文阅读

- 如何定义问题：[方法论文的研究问题定义与创新来源](方法论文的研究问题定义与创新来源.md)；逐篇核对说服链：[论文问题、解法与说服链](论文问题_解法_说服链总表.md)。
- 尚未充分验证的方向：[候选方法假设与实验门槛](候选方法假设与实验门槛.md)；异质标注问题：[多数据集标注协调与联合训练](多数据集标注协调与联合训练问题定义.md)；预训练探索：[MAE 与 Attention 可视化](MAE与Attention可视化研究思路.md)。

## 其他

- [新生 VS Code 远程连接与 Miniconda 环境配置](新生_VSCode远程连接与Miniconda环境配置.md)
- [论文英语单词本](论文英语单词本.md)
