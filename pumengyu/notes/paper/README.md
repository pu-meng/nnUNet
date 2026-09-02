# 论文交付物入口

`paper/` 只保留正式稿、汇报稿、投稿清单、参考论文和可复现的图表产物。研究笔记、方法解释、实验计划与结果审计统一放在 [`../md/`](../md/README.md)，不再在两个目录维护平行副本。

## 直接交付物

| 文件 | 用途 |
|---|---|
| [`论文v3.md`](论文v3.md) / [`论文v3.pdf`](论文v3.pdf) | 当前唯一正式主稿及导出版 |
| [`论文口头汇报稿.md`](论文口头汇报稿.md) / [`论文口头汇报稿.pdf`](论文口头汇报稿.pdf) | 导师汇报稿及 PDF |
| [`待补图表统计与对比清单.md`](待补图表统计与对比清单.md) | 投稿前图表、统计和产物修复清单 |
| [`paper_style.css`](paper_style.css) | Markdown/PDF 导出样式 |

## 产物目录

| 目录 | 用途 |
|---|---|
| [`figure_factory/`](figure_factory/README.md) | 绘图规范与可复现脚本 |
| [`figures/`](figures/README.md) | 正文及补充材料 SVG/PNG 成品 |
| [`statistics/`](statistics/README.md) | 统计表、逐病例数据和 provenance |
| [`assets/`](assets/README.md) | 病例图的原始 PNG 材料 |
| [`外界参考论文/`](外界参考论文/README.md) | 本地参考论文 PDF |

## 研究笔记入口

- 当前主线：[`../md/论文主线.md`](../md/论文主线.md)
- 当前实验审计与重训：[`../md/01_当前项目/当前实验与重训.md`](../md/01_当前项目/当前实验与重训.md)
- 实验结果与失败病例：[`../md/02_实验档案/`](../md/02_实验档案/README.md)
- 当前 MLA+MoE 结构：[`../md/01_当前项目/当前方法与架构.md`](../md/01_当前项目/当前方法与架构.md)

正文图片引用 `figures/` 中的 SVG；修图必须回到 `figure_factory/` 或对应统计脚本。`assets/`、`statistics/` 和 `figures/` 分别是来源、统计和成品，不是重复副本。
