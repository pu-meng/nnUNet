# MedNeXt_MLA_MoE 论文方向调整会议记录

时间：2026-07-16

## 一、导师会议核心结论

论文后续重点不应仅强调：

> 提出 MLA 模块，提高肝肿瘤 CT 分割 Dice。

而应转向：

> 在有限医学数据和有限计算资源条件下，探索局部卷积表征、全局上下文建模以及模型效率之间的平衡。

核心关键词：

-   Effective（有效性）
-   Efficient（效率）
-   Balance（平衡）

即：

**Efficiency-Effectiveness Balance**

------------------------------------------------------------------------

# 二、论文核心贡献方向调整

## 1. 有限数据条件下的有效分割

医学图像特点：

-   标注成本高
-   数据规模有限
-   多中心数据存在分布差异

因此不能简单依赖：

-   更大的模型
-   更复杂 Transformer
-   更多训练数据

论文角度：

> 在有限医学数据条件下，通过合理结构设计获得稳定分割性能。

当前设计：

MedNeXt 提供：

-   强局部卷积表征
-   depthwise convolution
-   inverted bottleneck
-   residual learning

加入 MLA：

-   少量引入全局上下文
-   建模肝脏-肿瘤长程关系
-   避免整体 Transformer 化带来的计算和数据需求

核心：

    强卷积局部特征
            +
    低成本全局上下文
            =
    有限数据下性能-效率平衡

------------------------------------------------------------------------

# 三、Efficiency（效率）方向

## 1. 轻量化模型

注意：

当前 MedNeXt_MLA_MoE 不一定比所有模型更轻量。

已有：

  模型              参数量
  ----------------- ----------
  MedNeXt           约61.78M
  MedNeXt_MLA_MoE   约67.9M

因此不要直接声称：

> lightweight model

更准确：

> 在有限额外参数增加下获得全局上下文能力。

重点比较：

-   参数量
-   FLOPs
-   显存
-   推理时间

------------------------------------------------------------------------

## 2. 推理速度与部署

需要补充：

  指标             说明
  ---------------- ----------
  Parameters       模型规模
  FLOPs            计算量
  GPU Memory       部署资源
  Inference Time   实际速度
  Training Cost    训练成本

目标：

证明不是单纯追求最高 Dice，而是：

> performance-efficiency trade-off 更优。

------------------------------------------------------------------------

# 四、重点实验1：MLA位置研究

当前：

    Encoder

    Enc1
     |
    Enc2
     |
    Enc3
     |
    Enc4
     |
    Bottleneck
     |
    MLA
     |
    Decoder

优势：

-   token 数量最低
-   attention 计算成本最低
-   高语义特征适合全局建模

但需要验证：

> bottleneck 是否是最佳位置？

------------------------------------------------------------------------

## Ablation

### Variant 1：Bottleneck MLA

当前方案。

### Variant 2：Encoder MLA

例如：

Enc3 / Enc4 插入 MLA。

研究：

更高分辨率特征是否受益。

### Variant 3：Skip Connection MLA

研究：

MLA 是否能够改善 encoder-decoder 特征融合。

目标：

> Attention placement affects the efficiency-effectiveness trade-off.

------------------------------------------------------------------------

# 五、重点实验2：MoE动态 Skip Connection

## 当前问题

U-Net skip connection：

固定融合：

\[ Y=X\_{enc}+X\_{dec} \]

所有病例使用相同信息。

------------------------------------------------------------------------

## 新思路

动态门控：

\[ Y=g(X)X\_{enc}+X\_{dec} \]

其中：

\[ g(X)`\in[0,1]`{=tex}\]

由：

-   MoE router
-   gating network

决定。

------------------------------------------------------------------------

## 目标

简单病例：

-   减少无效信息

困难病例：

-   保留更多细节

从：

固定 skip

变为：

adaptive skip connection。

------------------------------------------------------------------------

# 六、新论文叙事

旧：

    提出 MLA 模块
            ↓
    提升 Dice

新：

    医学数据有限
            ↓
    需要性能与效率平衡
            ↓
    MedNeXt提供局部建模能力
            ↓
    MLA提供低成本全局上下文
            ↓
    研究最佳插入位置和动态融合机制
            ↓
    获得可靠、高效、易部署的医学分割模型

------------------------------------------------------------------------

# 七、后续实验优先级

## Priority 1

MLA位置消融：

-   bottleneck
-   Enc4
-   Enc3
-   skip connection

## Priority 2

Efficiency实验：

-   Params
-   FLOPs
-   GPU Memory
-   Inference Time

## Priority 3

Dynamic Skip MoE：

-   adaptive routing
-   feature selection

## Priority 4

数据效率实验：

训练比例：

-   100%
-   50%
-   25%

验证有限数据优势。

------------------------------------------------------------------------

# 八、最终论文定位

英文：

> An efficient and reliable medical segmentation framework that balances
> local convolutional representation, global context modeling, and
> computational cost under limited medical data.

中文：

> 面向有限医学数据和实际部署需求，在局部卷积表征、全局上下文建模与计算效率之间取得平衡的可靠肝肿瘤CT分割框架。
