# 方法与架构

本目录保存理解当前论文网络所需的机制说明和后续结构假设。正式论文表述以 [`../论文v3.md`](../论文v3.md) 为准。

- [`MedNeXt_MLA_MoE基础模块与计算量.md`](MedNeXt_MLA_MoE基础模块与计算量.md)：从真实代码解释 stage、block、残差、卷积、MLA 和 MoE，并给出基础计算量关系。
- [`候选方法假设与实验门槛.md`](候选方法假设与实验门槛.md)：只保留尚有研究价值的候选机制、最低对照和停止条件；基础机制与已完成负向实验不在此重复。
- [`EfficientMedNeXt结构与来源.md`](EfficientMedNeXt结构与来源.md)：按官方实现解释网络、DMRFB、窄解码器和两阶段优化逻辑，并合并官方 commit、哈希、许可证、适配入口与当前实验边界。

绘图层级、箭头和残差关系的专项规范位于 [`../figure_factory/MedNeXt结构图绘制经验与检查模板.md`](../figure_factory/MedNeXt结构图绘制经验与检查模板.md)。
