from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/pumengyu_matplotlib_cache")

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "figures"
OUT_STEM = "mednext_stage_block"


COLORS = {
    "encoder": ("#eaf4ff", "#2563eb"),
    "block": ("#f8fafc", "#475569"),
    "operator": ("#eef2ff", "#4f46e5"),
    "decoder": ("#e8f7ef", "#16a34a"),
    "transition": ("#fff3d6", "#d97706"),
    "arrow": "#263238",
    "skip": "#6b7280",
}


def box(ax, x, y, w, h, title, subtitle="", kind="block", fs=11):
    fill, edge = COLORS[kind]
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        linewidth=1.7,
        edgecolor=edge,
        facecolor=fill,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h * (0.62 if subtitle else 0.50), title, ha="center", va="center", fontsize=fs, weight="bold")
    if subtitle:
        ax.text(x + w / 2, y + h * 0.27, subtitle, ha="center", va="center", fontsize=fs - 2, color="#4b5563")


def arrow(ax, start, end, *, color=None, curved=0.0, dashed=False):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=13,
        linewidth=1.7,
        color=color or COLORS["arrow"],
        linestyle=(0, (4, 4)) if dashed else "solid",
        connectionstyle=f"arc3,rad={curved}",
    )
    ax.add_patch(patch)


def add_sum(ax, x, y):
    circle = Circle((x, y), 0.16, facecolor="white", edgecolor=COLORS["arrow"], linewidth=1.5)
    ax.add_patch(circle)
    ax.text(x, y - 0.01, "+", ha="center", va="center", fontsize=12, weight="bold")


def draw() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(15.2, 6.6))
    ax.set_xlim(0, 15.2)
    ax.set_ylim(0, 6.6)
    ax.axis("off")

    # (a) Encoder stage.
    ax.text(2.25, 6.25, "(a) Encoder Stage", ha="center", va="center", fontsize=14, weight="bold")
    box(ax, 1.25, 5.20, 2.00, 0.70, "Stage input", "C, D x H x W", kind="encoder")
    box(ax, 1.05, 3.65, 2.40, 0.88, "MedNeXt Block x n", "same resolution and channels", kind="encoder")
    box(ax, 1.05, 1.80, 2.40, 0.88, "MedNeXt Down Block", "stride-2; 2C", kind="transition")
    box(ax, 1.25, 0.55, 2.00, 0.70, "Next-stage input", "2C, D/2 x H/2 x W/2", kind="encoder", fs=10)
    arrow(ax, (2.25, 5.20), (2.25, 4.53))
    arrow(ax, (2.25, 3.65), (2.25, 2.68))
    arrow(ax, (2.25, 1.80), (2.25, 1.25))
    arrow(ax, (3.45, 4.09), (4.30, 4.09), color=COLORS["skip"], dashed=True)
    ax.text(4.32, 4.09, "skip feature", ha="left", va="center", fontsize=10, color="#4b5563")

    # (b) One MedNeXt residual block, following the concise ResNet block style.
    ax.text(7.55, 6.25, "(b) MedNeXt Block", ha="center", va="center", fontsize=14, weight="bold")
    ax.text(7.55, 5.83, "x", ha="center", va="center", fontsize=11, weight="bold")
    ops = [
        (4.96, "Depthwise Conv 3x3x3", "groups = C"),
        (4.05, "GroupNorm", "num_groups = C"),
        (3.14, "Pointwise Conv 1x1x1", "C -> rC"),
        (2.23, "GELU", ""),
        (1.32, "Pointwise Conv 1x1x1", "rC -> C"),
    ]
    for y, title, subtitle in ops:
        box(ax, 6.30, y, 2.50, 0.62, title, subtitle, kind="operator", fs=10)
    arrow(ax, (7.55, 5.72), (7.55, 5.58))
    for (upper_y, _, _), (lower_y, _, _) in zip(ops[:-1], ops[1:]):
        arrow(ax, (7.55, upper_y), (7.55, lower_y + 0.62))
    add_sum(ax, 7.55, 0.72)
    arrow(ax, (7.55, 1.32), (7.55, 0.88))
    arrow(ax, (7.55, 0.56), (7.55, 0.25))
    ax.text(7.55, 0.10, "y = x + F(x)", ha="center", va="center", fontsize=10, weight="bold")
    arrow(ax, (7.55, 5.72), (7.72, 0.72), curved=-0.38)
    ax.text(9.40, 2.68, "identity\nshortcut", ha="center", va="center", fontsize=10, color="#4b5563")

    # (c) Decoder stage.
    ax.text(12.75, 6.25, "(c) Decoder Stage", ha="center", va="center", fontsize=14, weight="bold")
    box(ax, 11.65, 0.55, 2.20, 0.70, "Lower-scale input", "2C, D/2 x H/2 x W/2", kind="decoder", fs=10)
    box(ax, 11.55, 1.80, 2.40, 0.88, "MedNeXt Up Block", "stride-2 transpose; C/2", kind="transition")
    add_sum(ax, 12.75, 3.23)
    box(ax, 11.55, 3.72, 2.40, 0.88, "MedNeXt Block x n", "local refinement", kind="decoder")
    box(ax, 11.75, 5.20, 2.00, 0.70, "Stage output", "C, D x H x W", kind="decoder")
    arrow(ax, (12.75, 1.25), (12.75, 1.80))
    arrow(ax, (12.75, 2.68), (12.75, 3.07))
    arrow(ax, (12.75, 3.39), (12.75, 3.72))
    arrow(ax, (12.75, 4.60), (12.75, 5.20))
    arrow(ax, (10.55, 3.23), (12.59, 3.23), color=COLORS["skip"], dashed=True)
    ax.text(10.52, 3.23, "encoder skip", ha="right", va="center", fontsize=10, color="#4b5563")

    fig.savefig(FIGURE_DIR / f"{OUT_STEM}.svg", bbox_inches="tight")
    fig.savefig(FIGURE_DIR / f"{OUT_STEM}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    draw()
