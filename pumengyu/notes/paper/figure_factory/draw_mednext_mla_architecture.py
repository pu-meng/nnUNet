from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/pumengyu_matplotlib_cache")

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "figures"
OUT_STEM = "mednext_mla_architecture"


COLORS = {
    "input": ("#eef2ff", "#4f46e5"),
    "conv": ("#eaf4ff", "#2563eb"),
    "bottleneck": ("#f1f5f9", "#475569"),
    "mla": ("#fff3d6", "#d97706"),
    "decoder": ("#e8f7ef", "#16a34a"),
    "transition": ("#fff3d6", "#d97706"),
    "output": ("#ecfdf5", "#047857"),
    "panel": ("#f8fafc", "#cbd5e1"),
    "arrow": "#263238",
    "skip": "#6b7280",
}


def add_round_box(
    ax,
    xy: tuple[float, float],
    width: float,
    height: float,
    title: str,
    subtitle: str = "",
    note: str = "",
    kind: str = "conv",
    fontsize: int = 12,
    title_ratio: float = 0.67,
    subtitle_ratio: float = 0.42,
) -> None:
    fill, edge = COLORS[kind]
    x, y = xy
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        linewidth=1.8,
        edgecolor=edge,
        facecolor=fill,
    )
    ax.add_patch(box)
    ax.text(x + width / 2, y + height * title_ratio, title, ha="center", va="center", fontsize=fontsize, weight="bold")
    if subtitle:
        ax.text(x + width / 2, y + height * subtitle_ratio, subtitle, ha="center", va="center", fontsize=fontsize - 2)
    if note:
        ax.text(x + width / 2, y + height * 0.20, note, ha="center", va="center", fontsize=fontsize - 3, color="#4b5563")


def add_panel(ax, xy: tuple[float, float], width: float, height: float, label: str) -> None:
    fill, edge = COLORS["panel"]
    x, y = xy
    panel = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.03,rounding_size=0.18",
        linewidth=1.2,
        edgecolor=edge,
        facecolor=fill,
        zorder=-2,
    )
    ax.add_patch(panel)
    ax.text(x + 0.22, y + height - 0.26, label, ha="left", va="center", fontsize=11, weight="bold", color="#334155")


def add_arrow(ax, start: tuple[float, float], end: tuple[float, float], *, curved: float = 0.0) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=1.8,
        color=COLORS["arrow"],
        connectionstyle=f"arc3,rad={curved}",
    )
    ax.add_patch(arrow)


def add_orthogonal_arrow(ax, points: list[tuple[float, float]]) -> None:
    """Draw an orthogonal path with one arrowhead on its final segment."""
    for start, end in zip(points[:-2], points[1:-1]):
        ax.plot([start[0], end[0]], [start[1], end[1]], color=COLORS["arrow"], linewidth=1.8)
    add_arrow(ax, points[-2], points[-1])


def add_skip(ax, start: tuple[float, float], end: tuple[float, float]) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.5,
        linestyle=(0, (4, 4)),
        color=COLORS["skip"],
    )
    ax.add_patch(arrow)


def add_skip_path(ax, points: list[tuple[float, float]]) -> None:
    """Draw an orthogonal dashed skip path ending at an explicit add node."""
    for start, end in zip(points[:-2], points[1:-1]):
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color=COLORS["skip"],
            linewidth=1.5,
            linestyle=(0, (4, 4)),
        )
    add_skip(ax, points[-2], points[-1])


def add_sum(ax, center: tuple[float, float]) -> None:
    circle = Circle(
        center,
        0.09,
        facecolor="white",
        edgecolor=COLORS["arrow"],
        linewidth=1.6,
        zorder=3,
    )
    ax.add_patch(circle)
    ax.text(*center, "+", ha="center", va="center", fontsize=8.5, weight="bold", zorder=4)


def draw() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(13.5, 14.0))
    ax.set_xlim(0, 13.5)
    ax.set_ylim(0, 14.0)
    ax.axis("off")

    ax.text(6.75, 13.65, "MedNeXt_MLA_MoE Overall Architecture", ha="center", va="center", fontsize=17, weight="bold")
    ax.text(4.55, 13.18, "Encoder", ha="center", va="center", fontsize=12, weight="bold", color="#1d4ed8")
    ax.text(10.15, 13.18, "Decoder", ha="center", va="center", fontsize=12, weight="bold", color="#15803d")

    w = 2.05
    h = 0.92
    enc_x = 3.55
    dec_x = 9.15
    stage_y = [9.70, 7.40, 5.10, 2.80]

    # The input projection is only a channel projection. Encoder stages correspond to
    # enc_block_0 ... enc_block_3 in the implementation.
    stage_center = enc_x + w / 2
    add_round_box(ax, (stage_center - 0.72, 12.10), 1.45, 0.78, "Input CT", "3D volume", kind="input", fontsize=10)
    add_round_box(ax, (stage_center - 0.72, 10.85), 1.45, 0.78, "Input Projection", "Conv3d 1x1x1", "1→32 channels", kind="conv", fontsize=9)

    encoder_specs = [
        ("Stage 1", "×3", "[B,32,D,H,W]"),
        ("Stage 2", "×4", "[B,64,D/2,H/2,W/2]"),
        ("Stage 3", "×8", "[B,128,D/4,H/4,W/4]"),
        ("Stage 4", "×8", "[B,256,D/8,H/8,W/8]"),
    ]
    decoder_specs = [
        ("Stage 1", "×3", "[B,32,D,H,W]"),
        ("Stage 2", "×4", "[B,64,D/2,H/2,W/2]"),
        ("Stage 3", "×8", "[B,128,D/4,H/4,W/4]"),
        ("Stage 4", "×8", "[B,256,D/8,H/8,W/8]"),
    ]
    for y, spec in zip(stage_y, encoder_specs):
        add_round_box(ax, (enc_x, y), w, h, *spec, kind="conv", fontsize=11)
    for y, spec in zip(stage_y, decoder_specs):
        add_round_box(ax, (dec_x, y), w, h, *spec, kind="decoder", fontsize=11)

    bot_x = stage_center - 1.30
    bot_y = 0.50
    bot_w = 2.60
    mla_w = 2.25
    # Align MLA's vertical center with the decoder center and the first UpBlock.
    # This keeps the bottleneck -> MLA bridge horizontal and the MLA -> UpBlock
    # path continuous instead of leaving a diagonal or broken-looking arrow.
    mla_x = dec_x + w / 2 - mla_w / 2
    add_round_box(
        ax,
        (bot_x, bot_y),
        bot_w,
        1.05,
        "MedNeXt Bottleneck",
        "×8",
        "[B,512,D/16,H/16,W/16]",
        kind="bottleneck",
        fontsize=11,
    )
    add_round_box(
        ax,
        (mla_x, bot_y),
        mla_w,
        1.05,
        "MLA + MoE",
        "×2",
        "global token interaction",
        kind="mla",
        fontsize=11,
    )
    add_round_box(
        ax,
        (dec_x + 0.30, 11.15),
        1.75,
        0.88,
        "OutBlock (out_0)",
        "ConvTranspose3d 1x1x1",
        "[B,3,D,H,W]",
        kind="output",
        fontsize=8,
        title_ratio=0.78,
        subtitle_ratio=0.50,
    )

    # The input and stem are now a vertical chain above Encoder Stage 1.
    add_arrow(ax, (stage_center, 12.10), (stage_center, 11.63))
    add_arrow(ax, (stage_center, 10.85), (stage_center, 10.32))
    add_arrow(ax, (dec_x + w / 2, stage_y[0] + h), (dec_x + w / 2, 11.15))

    # Encoder descends along the left arm.
    for upper_y, lower_y in zip(stage_y[:-1], stage_y[1:]):
        add_arrow(ax, (enc_x + w / 2, upper_y), (enc_x + w / 2, lower_y + h))
    # Independent DownBlocks are shown as transition boxes on the main path.
    down_specs = [
        (8.75, "DownBlock", "32→64, s=2"),
        (6.45, "DownBlock", "64→128, s=2"),
        (4.15, "DownBlock", "128→256, s=2"),
    ]
    for y, title, subtitle in down_specs:
        add_round_box(ax, (stage_center - 0.75, y), 1.50, 0.52, title, subtitle, kind="transition", fontsize=8, title_ratio=0.75, subtitle_ratio=0.27)
    add_round_box(ax, (stage_center - 0.75, 1.95), 1.50, 0.52, "DownBlock", "256→512, s=2", kind="transition", fontsize=8, title_ratio=0.75, subtitle_ratio=0.27)
    # Stage 4 -> bottleneck stays on one vertical axis.
    add_arrow(ax, (stage_center, stage_y[-1]), (stage_center, 2.47))
    add_arrow(ax, (stage_center, 1.95), (stage_center, bot_y + 1.05))

    # Bottom bridge: convolutional bottleneck -> MLA/MoE context.
    add_arrow(ax, (bot_x + bot_w, bot_y + 0.525), (mla_x, bot_y + 0.525))

    # Independent UpBlocks are shown as transition boxes on the main path.
    up_specs = [
        (1.95, "UpBlock", "512→256, s=2"),
        (4.15, "UpBlock", "256→128, s=2"),
        (6.45, "UpBlock", "128→64, s=2"),
        (8.75, "UpBlock", "64→32, s=2"),
    ]
    for y, title, subtitle in up_specs[:1]:
        add_round_box(ax, (dec_x + 0.28, y), 1.50, 0.52, title, subtitle, kind="transition", fontsize=8, title_ratio=0.75, subtitle_ratio=0.27)
    for y, title, subtitle in up_specs[1:]:
        add_round_box(ax, (dec_x + 0.28, y), 1.50, 0.52, title, subtitle, kind="transition", fontsize=8, title_ratio=0.75, subtitle_ratio=0.27)

    # Decoder order follows forward() exactly:
    # UpBlock -> shape alignment -> elementwise add x_res_i -> decoder stage.
    decoder_center = dec_x + w / 2
    sum_y = [y - 0.165 for y in stage_y]
    for y in sum_y:
        add_sum(ax, (decoder_center, y))

    # MLA -> UpBlock 3 -> add -> decoder stage 4.
    add_arrow(ax, (decoder_center, bot_y + 1.05), (decoder_center, up_specs[0][0]))
    add_arrow(ax, (decoder_center, up_specs[0][0] + 0.52), (decoder_center, sum_y[-1] - 0.09))
    add_arrow(ax, (decoder_center, sum_y[-1] + 0.09), (decoder_center, stage_y[-1]))

    # Each decoded stage then feeds the next UpBlock and explicit skip-add node.
    for lower_index, upper_index, up_spec in [(3, 2, up_specs[1]), (2, 1, up_specs[2]), (1, 0, up_specs[3])]:
        up_y = up_spec[0]
        add_arrow(ax, (decoder_center, stage_y[lower_index] + h), (decoder_center, up_y))
        add_arrow(ax, (decoder_center, up_y + 0.52), (decoder_center, sum_y[upper_index] - 0.09))
        add_arrow(ax, (decoder_center, sum_y[upper_index] + 0.09), (decoder_center, stage_y[upper_index]))

    # Same-resolution x_res_i skips enter the add nodes, not the stage boxes.
    for index, y in enumerate(stage_y):
        elbow_x = 7.85
        add_skip_path(
            ax,
            [
                (enc_x + w, y + h / 2),
                (elbow_x, y + h / 2),
                (elbow_x, sum_y[index]),
                (decoder_center - 0.09, sum_y[index]),
            ],
        )
        ax.text(
            (enc_x + w + elbow_x) / 2,
            y + h / 2 + 0.16,
            r"skip $x_{res}$",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color=COLORS["skip"],
        )

    fig.savefig(FIGURE_DIR / f"{OUT_STEM}.svg", bbox_inches="tight")
    fig.savefig(FIGURE_DIR / f"{OUT_STEM}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    draw()
