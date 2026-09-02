from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/pumengyu_matplotlib_cache")
#MPLCONFIGDIR 是告诉Matplotlib在哪里存储配置文件和缓存数据的地方。
#/tmp的写入不需要权限,可以避免权限问题,setdefault是只在变量尚未设置时候填写默认值
#pyplot是matplotlib中负责创建画布,坐标轴,文字和保存图片的模块
#patches是matplotlib中负责绘制二维图形的子模块,包括,矩形,圆形,多边形等,箭头,圆角方框,路径图像
#FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14, linewidth=1.8,)
#fancyarrowpatch控制箭头起点,终点,箭头头部样式,线条粗细,颜色,实线或虚线,直线或曲线
#FancyBboxPatch,用来绘制圆角彩色色块的类
#patch=FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.11", linewidth=1.8, edgecolor=edge, facecolor=fill,)
#(x,y)是方框左下角坐标,boxstyle="round"是使用圆角,rounding_size是圆角大小,edgecolor是边框颜色,facecolor是填充颜色

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "figures"

COLORS = {
    "feature": ("#eaf4ff", "#2563eb"),#这里的两个颜色第一个是填充颜色,第二个是边框颜色
    "token": ("#eef2ff", "#4f46e5"),
    "latent": ("#fff3d6", "#d97706"),
    "block": ("#fef9c3", "#ca8a04"),
    "moe": ("#f3e8ff", "#9333ea"),
    "expert": ("#fae8ff", "#c026d3"),
    "output": ("#ecfdf5", "#047857"),
    "arrow": "#263238",
    "soft": "#6b7280",
}


def box(
    ax,
    x,
    y,
    w,
    h,
    title,
    subtitle="",
    note="",
    kind="feature",
    fs=12,
    title_ratio=0.68,
    subtitle_ratio=0.42,
):
    fill, edge = COLORS[kind]
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.11",
        linewidth=1.8, edgecolor=edge, facecolor=fill,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h * title_ratio, title, ha="center", va="center", fontsize=fs, weight="bold")
    if subtitle:
        ax.text(x + w / 2, y + h * subtitle_ratio, subtitle, ha="center", va="center", fontsize=fs - 2)
    if note:
        ax.text(x + w / 2, y + h * 0.18, note, ha="center", va="center", fontsize=fs - 3, color="#4b5563")


def arrow(ax, start, end, *, color=None, curved=0.0, dashed=False):
    patch = FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=14, linewidth=1.8,
        color=color or COLORS["arrow"],
        linestyle=(0, (4, 4)) if dashed else "solid",
        connectionstyle=f"arc3,rad={curved}",
    )
    ax.add_patch(patch)


def branched_arrows(ax, source, branch_y, targets, *, color=None):
    """Draw a clean orthogonal one-to-many split without curved connectors."""
    line_color = color or COLORS["arrow"]
    source_x, source_y = source
    target_xs = [target[0] for target in targets]
    ax.plot([source_x, source_x], [source_y, branch_y], color=line_color, linewidth=2.0)
    ax.plot([min(target_xs), max(target_xs)], [branch_y, branch_y], color=line_color, linewidth=2.0)
    ax.scatter([source_x], [branch_y], s=18, color=line_color, zorder=2)
    for target in targets:
        arrow(ax, (target[0], branch_y), target, color=line_color)


def add_sum(ax, x, y):
    circle = Circle((x, y), 0.18, facecolor="white", edgecolor=COLORS["arrow"], linewidth=1.6)
    ax.add_patch(circle)
    ax.text(x, y - 0.01, "+", ha="center", va="center", fontsize=13, weight="bold")


def save(fig, stem: str) -> None:
    fig.savefig(FIGURE_DIR / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(FIGURE_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def draw_wrapper() -> None:
    # 图 3(a) 只回答 MLABottleneck3D 外层张量怎样变化。
    fig, ax = plt.subplots(figsize=(6.4, 7.2))
    ax.set_xlim(0, 6.4)
    ax.set_ylim(0, 7.2)
    ax.axis("off")
    center = 3.20
    ax.text(center, 6.92, "(a) MLABottleneck3D Feature Transformation", ha="center", va="center", fontsize=15, weight="bold")

    box(ax, 1.05, 5.75, 4.30, 0.72, "MedNeXt bottleneck output", r"$[B,512,8,8,8]$", kind="feature", fs=10)
    box(ax, 1.05, 4.55, 4.30, 0.72, "Flatten + transpose", r"$[B,512,512]$", kind="token", fs=10)
    box(ax, 1.05, 3.35, 4.30, 0.72, r"MLA + MoE Block $\times2$", r"$[B,512,512]$", kind="block", fs=10)
    box(ax, 1.55, 2.15, 3.30, 0.72, "Final LayerNorm", r"$[B,512,512]$", kind="token", fs=10)
    box(ax, 1.05, 0.75, 4.30, 0.72, "Transpose + reshape", r"$[B,512,8,8,8]$", kind="output", fs=10)

    for start_y, end_y in [(5.75, 5.27), (4.55, 4.07), (3.35, 2.87), (2.15, 1.47)]:
        arrow(ax, (center, start_y), (center, end_y))
    save(fig, "mla_bottleneck3d")


def draw_block() -> None:
    # 图 3(b) 单独展开一个 block，给两条 shortcut、箭头和加法节点充足留白。
    fig, ax = plt.subplots(figsize=(7.4, 7.6))
    ax.set_xlim(0, 7.4)
    ax.set_ylim(0, 7.6)
    ax.axis("off")
    center = 3.35
    text_layout = {"title_ratio": 0.72, "subtitle_ratio": 0.27}
    ax.text(3.70, 7.30, "(b) One MLA + MoE Block", ha="center", va="center", fontsize=15, weight="bold")

    box(ax, 1.35, 6.20, 4.00, 0.72, "Input tokens", r"$[B,512,512]$", kind="token", fs=10, **text_layout)
    box(ax, 1.35, 4.75, 4.00, 0.82, "LayerNorm  →  MLA", r"$[B,512,512]$", kind="latent", fs=10, **text_layout)
    add_sum(ax, center, 3.95)
    box(ax, 1.35, 2.35, 4.00, 0.82, "LayerNorm  →  MoE-FFN", r"$[B,512,512]$", kind="moe", fs=10, **text_layout)
    add_sum(ax, center, 1.45)
    box(ax, 1.35, 0.25, 4.00, 0.72, "Block output", r"$[B,512,512]$", kind="output", fs=10, **text_layout)

    arrow(ax, (center, 6.20), (center, 5.57))
    arrow(ax, (center, 4.75), (center, 4.13))
    ax.plot([center, center], [3.77, 3.55], color=COLORS["arrow"], linewidth=1.8)
    arrow(ax, (center, 3.55), (center, 3.17))
    arrow(ax, (center, 2.35), (center, 1.63))
    arrow(ax, (center, 1.27), (center, 0.97))

    # 第一条 shortcut 跨过 MLA；第二条从第一次相加后的特征跨过 MoE-FFN。
    ax.plot([5.35, 6.15, 6.15], [6.56, 6.56, 3.95], color=COLORS["soft"], linewidth=1.8)
    arrow(ax, (6.15, 3.95), (center + 0.18, 3.95), color=COLORS["soft"])
    ax.text(6.28, 5.12, "identity", ha="left", va="center", fontsize=9.5, color=COLORS["soft"])

    ax.plot([center, 6.45, 6.45], [3.55, 3.55, 1.45], color=COLORS["soft"], linewidth=1.8)
    arrow(ax, (6.45, 1.45), (center + 0.18, 1.45), color=COLORS["soft"])
    ax.text(6.58, 2.50, "identity", ha="left", va="center", fontsize=9.5, color=COLORS["soft"])
    save(fig, "mla_moe_block")


def draw_attention() -> None:
    # 与图 2、图 3 保持同一版式，同时增加框内和框间的纵向留白。
    fig, ax = plt.subplots(figsize=(6.0, 8.4))
    ax.set_xlim(0, 6.0)
    ax.set_ylim(0, 8.4)
    ax.axis("off")
    ax.text(3.0, 8.18, "Low-rank Multi-head Latent Attention", ha="center", va="center", fontsize=14, weight="bold")

    text_layout = {"title_ratio": 0.72, "subtitle_ratio": 0.27}
    box(ax, 1.35, 7.05, 3.30, 0.82, "Input tokens  x", "B × N × 512", kind="token", fs=12, **text_layout)
    box(ax, 0.30, 5.20, 2.50, 1.00, "Query projection", "512 → 512", kind="token", fs=12, **text_layout)
    box(ax, 3.20, 5.20, 2.50, 1.00, "Shared KV latent", "512 → 128", kind="latent", fs=12, **text_layout)
    box(ax, 3.20, 3.80, 2.50, 0.95, "K/V projections", "128 → 512 each", kind="latent", fs=12, **text_layout)
    box(ax, 0.75, 2.45, 4.50, 1.00, "8-head attention", "full N × N interaction", kind="block", fs=12, **text_layout)
    box(ax, 1.35, 1.35, 3.30, 0.75, "Output projection  W_O", "512 → 512", kind="output", fs=12, **text_layout)
    box(ax, 1.55, 0.25, 2.90, 0.75, "Output", "B × N × 512", kind="output", fs=11, **text_layout)

    branched_arrows(
        ax, (3.00, 7.05), 6.58,
        [(1.55, 6.20), (4.45, 6.20)],
    )
    arrow(ax, (1.55, 5.20), (1.55, 3.45))
    arrow(ax, (4.45, 5.20), (4.45, 4.75))
    arrow(ax, (4.45, 3.80), (4.45, 3.45))
    arrow(ax, (3.00, 2.45), (3.00, 2.10))
    arrow(ax, (3.00, 1.35), (3.00, 1.00))
    save(fig, "mla_low_rank_attention")


def draw_moe() -> None:
    # 以一个空间位置的 512 维向量为主线，避免把 reshape 误画成网络层。
    fig, ax = plt.subplots(figsize=(8.0, 11.8))
    ax.set_xlim(0, 8.0)
    ax.set_ylim(0, 11.8)
    ax.axis("off")
    ax.text(4.0, 11.50, "MoE Feed-forward Network", ha="center", va="center", fontsize=16, weight="bold")

    text_layout = {"title_ratio": 0.72, "subtitle_ratio": 0.27}
    box(
        ax, 1.55, 10.15, 4.90, 0.95,
        "Feature at one spatial position", "512 values",
        note=r"applied independently at all $B \times N$ positions",
        kind="token", fs=12, title_ratio=0.76, subtitle_ratio=0.43,
    )
    box(
        ax, 0.15, 8.15, 2.40, 1.25,
        "Shared expert", "Linear: 512 → 1024",
        note="GELU → Linear: 1024 → 512",
        kind="moe", fs=11, title_ratio=0.78, subtitle_ratio=0.48,
    )
    box(
        ax, 2.80, 8.15, 2.40, 1.25,
        "Expert selector", "one Linear layer",
        note="512 → 4 scores",
        kind="moe", fs=11, title_ratio=0.78, subtitle_ratio=0.48,
    )
    box(
        ax, 5.45, 8.15, 2.40, 1.25,
        "4 routed experts", "each: 512 → 1024 → 512",
        note="Linear → GELU → Linear",
        kind="expert", fs=10, title_ratio=0.78, subtitle_ratio=0.48,
    )
    box(
        ax, 2.80, 6.45, 2.40, 0.95,
        "Select top-2", "scores + balance bias",
        kind="moe", fs=10, **text_layout,
    )
    box(
        ax, 5.45, 6.45, 2.40, 0.95,
        "Selected outputs", "2 × 512 values",
        kind="expert", fs=10, **text_layout,
    )
    box(
        ax, 2.80, 4.65, 2.40, 0.95,
        "Two weights", "softmax(original scores)",
        kind="moe", fs=10, **text_layout,
    )
    box(
        ax, 3.00, 2.85, 4.00, 1.00,
        "Weighted routed output", "512 values",
        kind="moe", fs=11, **text_layout,
    )
    box(
        ax, 0.45, 0.45, 7.10, 1.00,
        "Add shared + routed", "512 per position → B × N × 512",
        kind="output", fs=11, **text_layout,
    )

    branched_arrows(
        ax, (4.00, 10.15), 9.75,
        [(1.35, 9.40), (4.00, 9.40), (6.65, 9.40)],
    )
    arrow(ax, (4.00, 8.15), (4.00, 7.40))
    arrow(ax, (4.00, 6.45), (4.00, 5.60))
    arrow(ax, (6.65, 8.15), (6.65, 7.40))
    arrow(ax, (5.20, 6.93), (5.45, 6.93))

    # 权重和被选中的输出分别垂直进入加权聚合，避免在聚合框上方挤成折线。
    arrow(ax, (4.00, 4.65), (4.00, 3.85))
    arrow(ax, (6.65, 6.45), (6.65, 3.85))

    # Shared 与 routed 两条输出均直接向下进入最终加法，不再绕行多次转角。
    arrow(ax, (1.35, 8.15), (1.35, 1.45))
    arrow(ax, (5.00, 2.85), (5.00, 1.45))
    save(fig, "moe_ffn")


def draw(selected: str = "all") -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    if selected in {"all", "wrapper"}:
        draw_wrapper()
    if selected in {"all", "block"}:
        draw_block()
    if selected in {"all", "attention"}:
        draw_attention()
    if selected in {"all", "moe"}:
        draw_moe()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate MedNeXt_MLA paper figures.")
    parser.add_argument(
        "--figure",
        choices=("all", "wrapper", "block", "attention", "moe"),
        default="all",
        help="Generate all figures or only one selected figure.",
    )
    draw(parser.parse_args().figure)
