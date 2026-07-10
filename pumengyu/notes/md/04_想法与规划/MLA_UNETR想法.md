# MLA-UNETR 想法记录


#### MLA-UNETR 想法记录
日期：2026-06-08

##### 核心想法
把 DeepSeek MLA 的低秩 KV 压缩引入 UNETR/Swin UNETR，
解决 3D medical image segmentation 中 self-attention O(N²) 复杂度问题，
实现在保持 full attention（全局感受野）的同时降低显存到 O(N·d_c)。

##### 主要竞争者
- UNETR++（TMI 2024）：EPA block，线性 attention，不是低秩 KV 压缩
- SegFormer3D（2024）：轻量 Transformer，没用 MLA
- TCSAFormer（2025）：token 压缩 + 稀疏 attention，不同方向

##### 差异化
MLA 是唯一保持 full attention 同时压缩 KV 的方法，
可以用更小 patch（8³ 而不是 16³）做细粒度分割。

##### 下一步
BATseg 论文投出后开始实现，先在 Synapse 复现 UNETR 基线。


---
