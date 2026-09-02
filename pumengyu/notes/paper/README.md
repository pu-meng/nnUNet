# 论文材料索引

本目录按“当前交付物 → 方法依据 → 实验证据 → 图表产物 → 规划与参考资料”组织。README 只负责导航，不重复正文内容。

## 当前使用入口

| 入口 | 用途 | 状态 |
|---|---|---|
| [`论文v3.md`](论文v3.md) | 当前唯一正式主稿 | 使用中；精简版 |
| [`论文v3.pdf`](论文v3.pdf) | 当前主稿的 10 页 PDF 导出 | 使用中 |
| [`待补图表统计与对比清单.md`](待补图表统计与对比清单.md) | 投稿前统计、产物修复与终审任务 | 使用中，不能按文件名判断为过期 |
| [`paper_style.css`](paper_style.css) | Markdown/PDF 导出样式 | 工具文件 |

## 内容分区

| 目录 | 放什么 | 从哪里开始 |
|---|---|---|
| [`方法与架构/`](方法与架构/) | MedNeXt、MLA、MoE 机制说明与改进假设 | [`方法与架构/README.md`](方法与架构/README.md) |
| [`实验与分析/`](实验与分析/) | 三域实验、消融、异常病例与产物审计 | [`实验与分析/README.md`](实验与分析/README.md) |
| [`figures/`](figures/) | 正文及补充材料的 SVG/PNG 成品 | [`figures/README.md`](figures/README.md) |
| [`figure_factory/`](figure_factory/) | 绘图规范和可复现 Python 脚本 | [`figure_factory/README.md`](figure_factory/README.md) |
| [`statistics/`](statistics/) | 表格、逐病例统计和 provenance 元数据 | [`statistics/README.md`](statistics/README.md) |
| [`assets/`](assets/) | 病例图生成所需的原始 PNG 材料 | [`assets/README.md`](assets/README.md) |
| [`规划与汇报/`](规划与汇报/) | 贡献边界、方向调整和口头汇报材料 | [`规划与汇报/README.md`](规划与汇报/README.md) |
| [`外界参考论文/`](外界参考论文/) | 本地参考论文 PDF | [`外界参考论文/README.md`](外界参考论文/README.md) |

## 使用规则

1. 论文内容只改 [`论文v3.md`](论文v3.md)，不再新建平行主稿。
2. 正文图片引用 `figures/` 中的 SVG；图片修改必须回到 `figure_factory/` 或对应统计脚本。
3. 实验结论先在 `实验与分析/` 核验，再回填主稿；预测、`summary.json`、报告和可视化不齐全时标为 `partial`。
4. `assets/`、`statistics/` 和 `figures/` 是来源、统计、成品三层，不相互替代。
5. 旧版本名称若仍出现在历史叙述中可以保留；作为可点击入口时必须指向当前文件。
6. 汇报和 PPT 直接使用 `figures/` 中的 300 DPI PNG，不再维护第二套图片副本。
