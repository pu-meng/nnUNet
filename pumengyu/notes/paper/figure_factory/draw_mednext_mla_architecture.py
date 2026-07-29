from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/pumengyu_matplotlib_cache")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "figures"
OUT_STEM = "mednext_mla_architecture"


COLORS = {
    "input": ("#eef2ff", "#4f46e5"),
    "conv": ("#eaf4ff", "#2563eb"),
    "bottleneck": ("#f1f5f9", "#475569"),
    "mla": ("#fff3d6", "#d97706"),
    "decoder": ("#e8f7ef", "#16a34a"),
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
    ax.text(x + width / 2, y + height * 0.67, title, ha="center", va="center", fontsize=fontsize, weight="bold")
    if subtitle:
        ax.text(x + width / 2, y + height * 0.42, subtitle, ha="center", va="center", fontsize=fontsize - 2)
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


def draw() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(13.5, 9.2))
    ax.set_xlim(0, 13.5)
    ax.set_ylim(0, 9.2)
    ax.axis("off")

    ax.text(6.75, 8.92, "MedNeXt_MLA_MoE Overall Architecture", ha="center", va="center", fontsize=17, weight="bold")
    ax.text(4.55, 8.48, "Encoder", ha="center", va="center", fontsize=12, weight="bold", color="#1d4ed8")
    ax.text(10.15, 8.48, "Decoder", ha="center", va="center", fontsize=12, weight="bold", color="#15803d")

    w = 2.05
    h = 0.92
    enc_x = 3.55
    dec_x = 9.15
    stage_y = [7.15, 5.60, 4.05, 2.50]

    # Stem is only an input projection. Encoder stages correspond to
    # enc_block_0 ... enc_block_3 in the implementation.
    add_round_box(ax, (0.20, 7.20), 1.35, 0.82, "Input CT", "3D volume", kind="input", fontsize=11)
    add_round_box(ax, (1.82, 7.20), 1.42, 0.82, "Stem", "1x1x1 Conv", "feature projection", fontsize=11)

    encoder_specs = [
        ("Stage 1", "×3", "32 ch | D × H × W"),
        ("Stage 2", "×4", "64 ch | D/2 × H/2 × W/2"),
        ("Stage 3", "×8", "128 ch | D/4 × H/4 × W/4"),
        ("Stage 4", "×8", "256 ch | D/8 × H/8 × W/8"),
    ]
    decoder_specs = [
        ("Stage 1", "×3", "32 ch | D × H × W"),
        ("Stage 2", "×4", "64 ch | D/2 × H/2 × W/2"),
        ("Stage 3", "×8", "128 ch | D/4 × H/4 × W/4"),
        ("Stage 4", "×8", "256 ch | D/8 × H/8 × W/8"),
    ]
    for y, spec in zip(stage_y, encoder_specs):
        add_round_box(ax, (enc_x, y), w, h, *spec, kind="conv", fontsize=11)
    for y, spec in zip(stage_y, decoder_specs):
        add_round_box(ax, (dec_x, y), w, h, *spec, kind="decoder", fontsize=11)

    bot_x = 4.65
    bot_y = 0.82
    bot_w = 2.60
    mla_x = 7.50
    mla_w = 2.25
    add_round_box(
        ax,
        (bot_x, bot_y),
        bot_w,
        1.05,
        "MedNeXt Bottleneck",
        "×8",
        "512 ch | D/16 × H/16 × W/16",
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
    add_round_box(ax, (11.47, 7.20), 1.75, 0.82, "Segmentation", "Head / Logits", "bg | liver | tumor", kind="output", fontsize=11)

    # Horizontal entry and exit at the two open ends of the U.
    top_mid = 7.20 + 0.41
    add_arrow(ax, (1.55, top_mid), (1.82, top_mid))
    add_arrow(ax, (3.24, top_mid), (enc_x, top_mid))
    add_arrow(ax, (dec_x + w, top_mid), (11.47, top_mid))

    # Encoder descends along the left arm.
    for upper_y, lower_y in zip(stage_y[:-1], stage_y[1:]):
        add_arrow(ax, (enc_x + w / 2, upper_y), (enc_x + w / 2, lower_y + h))
    add_arrow(ax, (enc_x + w / 2, stage_y[-1]), (bot_x + bot_w / 2, bot_y + 1.05))

    # Bottom bridge: convolutional bottleneck -> MLA/MoE context.
    add_arrow(ax, (bot_x + bot_w, bot_y + 0.525), (mla_x, bot_y + 0.525))

    # Decoder rises along the right arm.
    add_arrow(ax, (mla_x + mla_w, bot_y + 1.05), (dec_x + w / 2, stage_y[-1]))
    for lower_y, upper_y in zip(reversed(stage_y[1:]), reversed(stage_y[:-1])):
        add_arrow(ax, (dec_x + w / 2, lower_y + h), (dec_x + w / 2, upper_y))

    # Same-resolution skip connections span the interior of the U.
    for y in stage_y:
        add_skip(ax, (enc_x + w, y + h / 2), (dec_x, y + h / 2))

    fig.savefig(FIGURE_DIR / f"{OUT_STEM}.svg", bbox_inches="tight")
    fig.savefig(FIGURE_DIR / f"{OUT_STEM}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    draw()
