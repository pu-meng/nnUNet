# Paper Figure Factory

这个目录用于集中管理论文结构图、流程图和示意图的代码生成脚本，避免用 WPS/PowerPoint 手动画后难以复现。

## 当前脚本

```bash
python pumengyu/notes/paper/figure_factory/draw_mednext_mla_architecture.py
python pumengyu/notes/paper/figure_factory/draw_mla_bottleneck3d.py
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
