# Paper Figure Factory

这个目录集中管理正式论文结构图、流程图和机制示意图的可复现绘图代码。论文正文只引用相邻 `figures/` 目录中的正式成品。

## 从这里开始

- 长期绘图规范与工具方法：[`论文绘图规范与工具方法.md`](论文绘图规范与工具方法.md)
- 正式图片目录：[`../figures/`](../figures/)
- 正式论文主稿：[`../论文v3.md`](../论文v3.md)
- MedNeXt 专项绘图检查模板：[`MedNeXt结构图绘制经验与检查模板.md`](MedNeXt结构图绘制经验与检查模板.md)

当前主稿图 1–4 是结构与机制图的视觉基线；图 5–6 分别是结果总览和病例级可视化。

## 当前脚本与图号

```bash
python pumengyu/notes/paper/figure_factory/draw_mednext_mla_architecture.py
python pumengyu/notes/paper/figure_factory/draw_mednext_stage_block.py
python pumengyu/notes/paper/figure_factory/draw_mla_bottleneck3d.py
python pumengyu/notes/paper/figure_factory/draw_failure_case_figures.py
```

输出：

```text
pumengyu/notes/paper/figures/mednext_mla_architecture.svg
pumengyu/notes/paper/figures/mednext_mla_architecture.png
pumengyu/notes/paper/figures/mla_bottleneck3d.svg
pumengyu/notes/paper/figures/mla_bottleneck3d.png
```

## 原则

- 论文正文只引用 `pumengyu/notes/paper/figures/` 下的图片。
- 图的布局、颜色、文字和箭头全部由脚本控制。
- 后续新增结构图时，在本目录新增独立脚本，不直接手动改输出图片。

## 目录维护

- 当前正文图 1：`draw_mednext_mla_architecture.py`。
- 当前正文图 2–4：`draw_mla_bottleneck3d.py`。
- 当前正文图 5：`pumengyu/tools/analyasis/generate_paper_three_domain_main_figure.py`。
- 当前正文图 6：`pumengyu/tools/analyasis/generate_paper_case_composite.py`。
- `draw_mednext_stage_block.py`、`draw_mednext_conv_details.py` 和 `draw_failure_case_figures.py` 生成保留的补充/历史图片，不再按旧版正文图号理解。
- 每张正式图同时导出 SVG 和 300 DPI PNG。
- 修改图片必须回到 Python 源码，不直接修改导出文件。
- 临时补丁、`.orig`、截图和一次性试验文件不作为正式资产保留。
- 病例图的模型、checkpoint 和输入产物来源记录在 `../statistics/` 的对应 metadata/provenance 文件中。
