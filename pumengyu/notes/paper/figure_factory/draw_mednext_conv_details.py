from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/pumengyu_matplotlib_cache")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "figures"
COLORS = {
    "input": ("#f8fafc", "#475569"),
    "depthwise": ("#eaf4ff", "#2563eb"),
    "pointwise": ("#fff3d6", "#d97706"),
    "activation": ("#f3e8ff", "#9333ea"),
    "output": ("#ecfdf5", "#047857"),
    "arrow": "#263238",
    "soft": "#6b7280",
}


def box(ax, x, y, w, h, title, subtitle, kind, fs=11):
    fill, edge = COLORS[kind]
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.10",
            linewidth=1.7,
            edgecolor=edge,
            facecolor=fill,
        )
    )
    ax.text(x + w / 2, y + h * 0.64, title, ha="center", va="center", fontsize=fs, weight="bold")
    ax.text(x + w / 2, y + h * 0.28, subtitle, ha="center", va="center", fontsize=fs - 2, color="#4b5563")


def arrow(ax, start, end, color=None, dashed=False):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.7,
            color=color or COLORS["arrow"],
            linestyle=(0, (4, 4)) if dashed else "solid",
        )
    )


def save(fig, stem):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(FIGURE_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def draw_depthwise_separable():
    fig, ax = plt.subplots(figsize=(6.8, 8.0))
    ax.set_xlim(0, 6.8)
    ax.set_ylim(0, 8.0)
    ax.axis("off")
    center = 3.4
    ax.text(center, 7.68, "(c) Depthwise-Separable Convolution", ha="center", va="center", fontsize=13, weight="bold")
    box(ax, center - 1.65, 6.45, 3.30, 0.70, "Input feature", r"$[B,C,D,H,W]$", "input")
    box(ax, center - 1.85, 4.75, 3.70, 0.86, "Depthwise Conv $3\\times3\\times3$", r"groups=$C$; one kernel per channel", "depthwise", fs=10)
    box(ax, center - 1.85, 3.05, 3.70, 0.86, "Pointwise Conv $1\\times1\\times1$", r"channel mixing at each voxel", "pointwise", fs=10)
    box(ax, center - 1.65, 1.35, 3.30, 0.70, "Output feature", r"spatial size preserved", "output")
    arrow(ax, (center, 6.45), (center, 5.61))
    arrow(ax, (center, 4.75), (center, 3.91))
    arrow(ax, (center, 3.05), (center, 2.05))
    ax.text(center, 0.62, "Depthwise step models local spatial neighborhoods.\nPointwise step mixes channels without enlarging the spatial kernel.", ha="center", va="center", fontsize=9, color=COLORS["soft"])
    save(fig, "mednext_depthwise_separable")


def draw_inverted_bottleneck():
    fig, ax = plt.subplots(figsize=(7.0, 9.0))
    ax.set_xlim(0, 7.0)
    ax.set_ylim(0, 9.0)
    ax.axis("off")
    center = 3.5
    ax.text(center, 8.68, "(d) MedNeXt Inverted Bottleneck", ha="center", va="center", fontsize=13, weight="bold")
    box(ax, center - 1.65, 7.35, 3.30, 0.68, "Input", r"$C$ channels", "input")
    box(ax, center - 1.85, 5.82, 3.70, 0.78, "Depthwise Conv + GroupNorm", r"$C$ channels; spatial modeling", "depthwise", fs=10)
    box(ax, center - 1.85, 4.30, 3.70, 0.78, "Pointwise expansion", r"$1\times1\times1$: $C\rightarrow rC$", "pointwise", fs=10)
    box(ax, center - 1.30, 2.92, 2.60, 0.70, "GELU", r"nonlinear activation", "activation")
    box(ax, center - 1.85, 1.52, 3.70, 0.78, "Pointwise projection", r"$1\times1\times1$: $rC\rightarrow C$", "pointwise", fs=10)
    arrow(ax, (center, 7.35), (center, 6.60))
    arrow(ax, (center, 5.82), (center, 5.08))
    arrow(ax, (center, 4.30), (center, 3.62))
    arrow(ax, (center, 2.92), (center, 2.30))
    ax.text(center, 0.65, "Depthwise models space; 1×1×1 convolutions expand to rC\nand then project back to C.", ha="center", va="center", fontsize=9, color=COLORS["soft"])
    save(fig, "mednext_inverted_bottleneck")


if __name__ == "__main__":
    draw_depthwise_separable()
    draw_inverted_bottleneck()
