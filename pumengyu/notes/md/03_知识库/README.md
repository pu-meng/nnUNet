# 知识库

这里是按问题查阅的参考层，不需要按顺序全部阅读。每类主题只保留一个主要入口；实验状态和当前论文结论不在这里维护，分别看 [`../01_当前项目/`](../01_当前项目/README.md) 和 [`../02_实验档案/`](../02_实验档案/README.md)。

## nnU-Net 与 Trainer

- 先看 [`nnUNet代码地图.md`](nnUNet代码地图.md)
- 流水线细节：[`nnUNet_pipeline.md`](nnUNet_pipeline.md)、[`nnUNet_details.md`](nnUNet_details.md)
- 网络实现：[`nnUNet架构笔记.md`](nnUNet架构笔记.md)
- Trainer 与训练行为：[`Trainer架构改动.md`](Trainer架构改动.md)、[`Trainer_Loss_BatchSize.md`](Trainer_Loss_BatchSize.md)

## MedNeXt 与卷积

- 先看 [`MedNeXt架构.md`](MedNeXt架构.md)
- 设计来源与对比：[`ConvNeXt_MedNeXt对比.md`](ConvNeXt_MedNeXt对比.md)、[`DWSep_vs_IBConv机制.md`](DWSep_vs_IBConv机制.md)
- 独立架构：[`DeepResGN架构.md`](DeepResGN架构.md)、[`EfficientMedNeXt架构与实验边界.md`](EfficientMedNeXt架构与实验边界.md)
- 机制索引：[`架构机制.md`](架构机制.md)、[`通用方法.md`](通用方法.md)

## 研究问题与候选机制

- 按论文逐篇查 [`论文问题_解法_说服链总表.md`](论文问题_解法_说服链总表.md)
- 先看 [`方法论文的研究问题定义与创新来源.md`](方法论文的研究问题定义与创新来源.md)
- 候选实验：[`候选方法假设与实验门槛.md`](候选方法假设与实验门槛.md)
- 多数据集问题的具体映射：[`多数据集标注协调与联合训练问题定义.md`](多数据集标注协调与联合训练问题定义.md)

## GPU 与工程工具

- 先看 [`GPU性能基础.md`](GPU性能基础.md)
- profiling：[`NCU_nsys命令.md`](NCU_nsys命令.md)、[`nsys结果记录.md`](nsys结果记录.md)
- 训练问题：[`torch_compile与MedNeXt深监督问题.md`](torch_compile与MedNeXt深监督问题.md)、[`best_checkpoint重跑口径.md`](best_checkpoint重跑口径.md)
- 训练器/归一化/镜像等专题：[`归一化方案.md`](归一化方案.md)、[`镜像翻转.md`](镜像翻转.md)、[`MixedValidation模型关系与MSD报告补算.md`](MixedValidation模型关系与MSD报告补算.md)

## 其他专题

- 新人环境：[`新生_VSCode远程连接与Miniconda环境配置.md`](新生_VSCode远程连接与Miniconda环境配置.md)
- 论文阅读：[`论文英语单词本.md`](论文英语单词本.md)、[`论文问题_解法_说服链总表.md`](论文问题_解法_说服链总表.md)
- MAE/Attention 探索：[`MAE与Attention可视化研究思路.md`](MAE与Attention可视化研究思路.md)

其余文件作为专题细节保留，不进入默认阅读路径。已删除的旧总稿、重复架构入口和过时研究汇报不再作为历史活文档维护；可追溯实验证据以 `02_实验档案/` 和 Git 历史为准。
