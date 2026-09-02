from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/pumengyu_matplotlib_cache")

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "figures"
COLORS = {
    "encoder": ("#eaf4ff", "#2563eb"),
    "block": ("#f8fafc", "#475569"),
    "operator": ("#eef2ff", "#4f46e5"),
    "decoder": ("#e8f7ef", "#16a34a"),
    "output": ("#ecfdf5", "#047857"),
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


def add_sum(ax, x, y, radius=0.16):
    circle = Circle((x, y), radius, facecolor="white", edgecolor=COLORS["arrow"], linewidth=1.5)
    ax.add_patch(circle)
    ax.text(x, y - 0.01, "+", ha="center", va="center", fontsize=12, weight="bold")


def save(fig, stem: str) -> None:
    fig.savefig(FIGURE_DIR / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(FIGURE_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def draw_encoder_stage() -> None:
    """Draw the repeated-block structure shared by encoder, bottleneck, and decoder stages."""
    fig, ax = plt.subplots(figsize=(7.0, 9.0))
    ax.set_xlim(0, 7.0)
    ax.set_ylim(0, 9.0)
    ax.axis("off")
    center = 2.8
    ax.text(center, 8.84, "(a) Actual MedNeXt Stage Structure", ha="center", va="center", fontsize=13, weight="bold")
    box(ax, center - 1.55, 7.76, 3.10, 0.62, "Stage input", "[B,C,D,H,W]", kind="block", fs=10)
    block_ys = [6.25, 4.25, 2.25]
    sum_ys = [5.72, 3.72, 1.72]
    for idx, y in enumerate(block_ys, start=1):
        box(ax, center - 1.65, y, 3.30, 0.70, f"MedNeXtBlock {idx}", "shape unchanged", kind="encoder", fs=9)
        add_sum(ax, center, sum_ys[idx - 1], radius=0.14)
        arrow(ax, (center, y), (center, y - 0.38))
        # Each block has its own identity shortcut; shortcuts do not span blocks.
        ax.plot([center + 1.65, 5.55, 5.55], [y + 0.35, y + 0.35, sum_ys[idx - 1]], color=COLORS["skip"], linewidth=1.4)
        arrow(ax, (5.55, sum_ys[idx - 1]), (center + 0.14, sum_ys[idx - 1]), color=COLORS["skip"])
    arrow(ax, (center, 7.76), (center, 6.95))
    arrow(ax, (center, sum_ys[0] - 0.14), (center, block_ys[1] + 0.70))
    arrow(ax, (center, sum_ys[1] - 0.14), (center, block_ys[2] + 0.70))
    box(ax, center - 1.55, 0.72, 3.10, 0.62, "Stage output", "[B,C,D,H,W]", kind="output", fs=10)
    arrow(ax, (center, sum_ys[2] - 0.14), (center, 1.34))
    ax.text(5.70, 5.72, "identity", ha="left", va="center", fontsize=8.5, color="#4b5563")
    ax.text(5.70, 3.72, "identity", ha="left", va="center", fontsize=8.5, color="#4b5563")
    ax.text(5.70, 1.72, "identity", ha="left", va="center", fontsize=8.5, color="#4b5563")
    save(fig, "mednext_stage_structure")
    return

    # Legacy whole-encoder drawing retained below for easy recovery.
    fig, ax = plt.subplots(figsize=(6.2, 10.2))
    ax.set_xlim(0, 6.2)
    ax.set_ylim(0, 10.2)
    ax.axis("off")

    center = 3.10
    ax.text(center, 9.90, "(a) MedNeXt Encoder", ha="center", va="center", fontsize=14, weight="bold")
    layers = [
        (8.95, "Input CT", r"$[B,1,128,128,128]$", "block", 2.85),
        (7.60, "Stem Conv $1\\times1\\times1$", r"$1\rightarrow32$", "transition", 2.85),
        (6.25, "Encoder Stage 0", r"Block $\times3$  |  $[B,32,128,128,128]$", "encoder", 3.75),
        (4.90, "Encoder Stage 1", r"Block $\times4$  |  $[B,64,64,64,64]$", "encoder", 3.75),
        (3.55, "Encoder Stage 2", r"Block $\times8$  |  $[B,128,32,32,32]$", "encoder", 3.75),
        (2.20, "Encoder Stage 3", r"Block $\times8$  |  $[B,256,16,16,16]$", "encoder", 3.75),
        (0.70, "MedNeXt Bottleneck", r"Block $\times8$  |  $[B,512,8,8,8]$", "encoder", 3.75),
    ]
    for y, title, subtitle, kind, width in layers:
        box(ax, center - width / 2, y, width, 0.76, title, subtitle, kind=kind, fs=10)

    for upper, lower in zip(layers[:-1], layers[1:]):
        start_y = upper[0]
        end_y = lower[0] + 0.76
        arrow(ax, (center, start_y), (center, end_y))

    down_labels = [
        (5.93, r"DownBlock  $32\rightarrow64$, stride 2"),
        (4.58, r"DownBlock  $64\rightarrow128$, stride 2"),
        (3.23, r"DownBlock  $128\rightarrow256$, stride 2"),
        (1.85, r"DownBlock  $256\rightarrow512$, stride 2"),
    ]
    for y, label in down_labels:
        ax.text(center + 0.16, y, label, ha="left", va="center", fontsize=8.5, color="#4b5563")

    for y in (6.63, 5.28, 3.93, 2.58):
        arrow(ax, (center + 1.88, y), (5.65, y), color=COLORS["skip"], dashed=True)
        ax.text(5.68, y, "skip", ha="left", va="center", fontsize=8.5, color="#4b5563")
    save(fig, "mednext_encoder_stage")


def draw_mednext_block() -> None:
    fig, ax = plt.subplots(figsize=(6.2, 8.1))
    ax.set_xlim(0, 6.2)
    ax.set_ylim(0, 8.1)
    ax.axis("off")

    center = 2.70
    ax.text(center, 7.82, "(b) MedNeXtBlock: Actual Implementation", ha="center", va="center", fontsize=13, weight="bold")
    box(ax, center - 1.45, 6.82, 2.90, 0.66, "Input", r"generic $[B,C,D,H,W]$; example $C=512$", kind="block", fs=9)
    ops = [
        (5.62, r"Depthwise Conv $3\times3\times3$", r"groups=$C$  |  shape unchanged"),
        (4.50, "GroupNorm", r"$[B,C,D,H,W]$"),
        (3.38, r"Conv $1\times1\times1$", r"$C\rightarrow rC$  |  example $512\rightarrow4096$"),
        (2.26, "GELU", r"$[B,rC,D,H,W]$"),
        (1.14, r"Conv $1\times1\times1$", r"$rC\rightarrow C$  |  example $4096\rightarrow512$"),
    ]
    for y, title, subtitle in ops:
        box(ax, center - 1.55, y, 3.10, 0.72, title, subtitle, kind="operator", fs=10)
    arrow(ax, (center, 6.82), (center, 6.34))
    for (upper_y, _, _), (lower_y, _, _) in zip(ops[:-1], ops[1:]):
        arrow(ax, (center, upper_y), (center, lower_y + 0.72))
    add_sum(ax, center, 0.58)
    arrow(ax, (center, 1.14), (center, 0.74))
    ax.text(center, 0.20, r"Output  $[B,C,D,H,W]$  (example $[B,512,8,8,8]$)", ha="center", va="center", fontsize=9, weight="bold")
    # Branch the identity shortcut from the middle of the input-to-depthwise
    # connection, so the bypass origin is visually separate from the input box.
    identity_y = 6.58
    ax.plot(
        [center, 5.15, 5.15],
        [identity_y, identity_y, 0.58],
        color=COLORS["arrow"],
        linewidth=1.7,
        solid_capstyle="round",
        solid_joinstyle="round",
    )
    arrow(ax, (5.15, 0.58), (center + 0.16, 0.58))
    ax.text(5.28, 3.70, "identity\nshortcut", ha="left", va="center", fontsize=9.5, color="#4b5563")
    save(fig, "mednext_block")


def draw_decoder_stage() -> None:
    """Draw one representative decoder stage in implementation order."""
    fig, ax = plt.subplots(figsize=(7.0, 9.0))
    ax.set_xlim(0, 7.0)
    ax.set_ylim(0, 9.0)
    ax.axis("off")
    center = 4.00
    sum_radius = 0.20
    ax.text(center, 8.70, "(e) MedNeXt Decoder Stage", ha="center", va="center", fontsize=13, weight="bold")
    box(ax, center - 1.75, 7.45, 3.50, 0.72, "Previous decoder feature", r"$[B,256,16,16,16]$", kind="block", fs=10)
    box(ax, center - 1.55, 5.85, 3.10, 0.66, "MedNeXt UpBlock", r"$256\rightarrow128$, stride $2$", kind="transition", fs=9)
    add_sum(ax, center, 4.75, radius=sum_radius)
    box(ax, center - 1.95, 2.85, 3.90, 0.86, "MedNeXtBlocks", r"$\times8$  |  $[B,128,32,32,32]$", kind="decoder", fs=10)
    box(ax, center - 1.65, 1.20, 3.30, 0.72, "Stage output", r"$[B,128,32,32,32]$", kind="output", fs=10)
    arrow(ax, (center, 7.45), (center, 6.51))
    arrow(ax, (center, 5.85), (center, 4.95))
    arrow(ax, (center, 4.55), (center, 3.71))
    arrow(ax, (center, 2.85), (center, 1.92))
    arrow(ax, (0.45, 4.75), (center - sum_radius, 4.75), color=COLORS["skip"], dashed=True)
    ax.text(0.48, 5.05, r"encoder skip $[B,128,32,32,32]$", ha="left", va="bottom", fontsize=8.5, color="#4b5563")
    save(fig, "mednext_decoder_stage")
    return

    # Legacy whole-decoder drawing retained below for easy recovery.
    fig, ax = plt.subplots(figsize=(6.6, 14.5))
    ax.set_xlim(0, 6.6)
    ax.set_ylim(0, 14.5)
    ax.axis("off")

    center = 3.65
    sum_radius = 0.20
    ax.text(center, 14.20, "(c) MedNeXt Decoder", ha="center", va="center", fontsize=14, weight="bold")
    box(ax, center - 1.90, 13.20, 3.80, 0.75, "MLABottleneck3D output", r"$[B,512,8,8,8]$", kind="block", fs=10)

    stages = [
        (12.00, 11.35, 10.15, r"$512\rightarrow256$", "Decoder Stage 3", r"Block $\times8$  |  $[B,256,16,16,16]$", r"skip (add) $[B,256,16,16,16]$"),
        (9.15, 8.50, 7.30, r"$256\rightarrow128$", "Decoder Stage 2", r"Block $\times8$  |  $[B,128,32,32,32]$", r"skip (add) $[B,128,32,32,32]$"),
        (6.30, 5.65, 4.45, r"$128\rightarrow64$", "Decoder Stage 1", r"Block $\times4$  |  $[B,64,64,64,64]$", r"skip (add) $[B,64,64,64,64]$"),
        (3.45, 2.80, 1.60, r"$64\rightarrow32$", "Decoder Stage 0", r"Block $\times3$  |  $[B,32,128,128,128]$", r"skip (add) $[B,32,128,128,128]$"),
    ]

    previous_bottom = 13.20
    for up_y, sum_y, block_y, channels, title, subtitle, skip_label in stages:
        box(ax, center - 1.45, up_y, 2.90, 0.58, "MedNeXt UpBlock", channels, kind="transition", fs=9)
        arrow(ax, (center, previous_bottom), (center, up_y + 0.58))
        add_sum(ax, center, sum_y, radius=sum_radius)
        arrow(ax, (center, up_y), (center, sum_y + sum_radius))
        box(ax, center - 1.95, block_y, 3.90, 0.72, title, subtitle, kind="decoder", fs=9)
        arrow(ax, (center, sum_y - sum_radius), (center, block_y + 0.72))
        arrow(ax, (0.40, sum_y), (center - sum_radius, sum_y), color=COLORS["skip"], dashed=True)
        ax.text(0.42, sum_y + 0.24, skip_label, ha="left", va="bottom", fontsize=8.2, color="#4b5563")
        previous_bottom = block_y

    box(ax, center - 1.75, 0.25, 3.50, 0.75, r"Segmentation Head  $1\times1\times1$", r"$32\rightarrow3$  |  $[B,3,128,128,128]$", kind="output", fs=9)
    arrow(ax, (center, 1.60), (center, 1.00))
    save(fig, "mednext_decoder_stage")


def draw_downblock_detail() -> None:
    """Show the actual MedNeXtDownBlock path and optional resampling shortcut."""
    fig, ax = plt.subplots(figsize=(7.8, 8.8))
    ax.set_xlim(0, 7.8)
    ax.set_ylim(0, 8.8)
    ax.axis("off")
    center = 3.5
    ax.text(3.5, 8.48, "(f) MedNeXtDownBlock: Detail", ha="center", va="center", fontsize=13, weight="bold")
    box(ax, center - 1.65, 7.55, 3.30, 0.62, "Input", "[B,64,64,64,64]", kind="block", fs=10)
    ops = [
        (6.40, "Depthwise Conv3d", "k=3, stride=2; C=64"),
        (5.38, "GroupNorm", "[B,64,32,32,32]"),
        (4.36, "1x1x1 expansion", "64 -> 256 (r=4)"),
        (3.34, "GELU", "[B,256,32,32,32]"),
        (2.32, "1x1x1 projection", "256 -> 128"),
    ]
    for y, title, subtitle in ops:
        box(ax, center - 1.65, y, 3.30, 0.62, title, subtitle, kind="operator", fs=9)
    box(ax, center - 1.65, 1.10, 3.30, 0.62, "Main path output", "[B,128,32,32,32]", kind="output", fs=9)
    arrow(ax, (center, 7.55), (center, 7.02))
    for (upper_y, _, _), (lower_y, _, _) in zip(ops[:-1], ops[1:]):
        arrow(ax, (center, upper_y), (center, lower_y + 0.62))
    arrow(ax, (center, 2.32), (center, 1.72))
    # do_res_up_down=True: a separate shortcut branch is summed at the end.
    shortcut_box_x, shortcut_box_y, shortcut_box_w, shortcut_box_h = 5.35, 4.55, 1.70, 1.05
    box(ax, shortcut_box_x, shortcut_box_y, shortcut_box_w, shortcut_box_h, "Resampling shortcut", "1x1x1 Conv3d, stride=2", kind="transition", fs=7)
    shortcut_x = shortcut_box_x + shortcut_box_w / 2
    ax.plot([center + 1.65, shortcut_x, shortcut_x], [7.86, 7.86, shortcut_box_y + shortcut_box_h], color=COLORS["skip"], linewidth=1.7)
    arrow(ax, (shortcut_x, shortcut_box_y + shortcut_box_h), (shortcut_x, shortcut_box_y + shortcut_box_h - 0.12), color=COLORS["skip"])
    ax.plot([shortcut_x, shortcut_x], [shortcut_box_y, 0.70], color=COLORS["skip"], linewidth=1.7)
    arrow(ax, (shortcut_x, 0.70), (center + 0.16, 0.70), color=COLORS["skip"])
    add_sum(ax, center, 0.70)
    arrow(ax, (center, 1.10), (center, 0.86))
    ax.text(center, 0.42, "DownBlock output [B,128,32,32,32]", ha="center", va="center", fontsize=8.5, weight="bold")
    save(fig, "mednext_downblock_detail")


def draw_upblock_detail() -> None:
    """Show the actual MedNeXtUpBlock path and optional transposed shortcut."""
    fig, ax = plt.subplots(figsize=(7.8, 8.8))
    ax.set_xlim(0, 7.8)
    ax.set_ylim(0, 8.8)
    ax.axis("off")
    center = 2.8
    ax.text(3.5, 8.48, "(g) MedNeXtUpBlock: Detail", ha="center", va="center", fontsize=13, weight="bold")
    box(ax, center - 1.65, 7.55, 3.30, 0.62, "Input", "[B,256,16,16,16]", kind="block", fs=10)
    ops = [
        (6.40, "Depthwise ConvTranspose3d", "k=3, stride=2; C=256"),
        (5.38, "GroupNorm", "[B,256,32,32,32]"),
        (4.36, "1x1x1 expansion", "256 -> 2048 (r=8)"),
        (3.34, "GELU", "[B,2048,32,32,32]"),
        (2.32, "1x1x1 projection + pad", "2048 -> 128; shape aligned"),
    ]
    for y, title, subtitle in ops:
        box(ax, center - 1.65, y, 3.30, 0.62, title, subtitle, kind="operator", fs=8.5)
    box(ax, center - 1.65, 1.10, 3.30, 0.62, "Main path output", "[B,128,32,32,32]", kind="output", fs=9)
    arrow(ax, (center, 7.55), (center, 7.02))
    for (upper_y, _, _), (lower_y, _, _) in zip(ops[:-1], ops[1:]):
        arrow(ax, (center, upper_y), (center, lower_y + 0.62))
    arrow(ax, (center, 2.32), (center, 1.72))
    shortcut_box_x, shortcut_box_y, shortcut_box_w, shortcut_box_h = 5.35, 4.55, 1.85, 1.05
    box(ax, shortcut_box_x, shortcut_box_y, shortcut_box_w, shortcut_box_h, "Transposed shortcut", "1x1x1 ConvTranspose3d, stride=2", kind="transition", fs=6.8)
    shortcut_x = shortcut_box_x + shortcut_box_w / 2
    ax.plot([center + 1.65, shortcut_x, shortcut_x], [7.86, 7.86, shortcut_box_y + shortcut_box_h], color=COLORS["skip"], linewidth=1.7)
    arrow(ax, (shortcut_x, shortcut_box_y + shortcut_box_h), (shortcut_x, shortcut_box_y + shortcut_box_h - 0.12), color=COLORS["skip"])
    ax.plot([shortcut_x, shortcut_x], [shortcut_box_y, 0.70], color=COLORS["skip"], linewidth=1.7)
    arrow(ax, (shortcut_x, 0.70), (center + 0.16, 0.70), color=COLORS["skip"])
    add_sum(ax, center, 0.70)
    arrow(ax, (center, 1.10), (center, 0.86))
    ax.text(center, 0.42, "UpBlock output [B,128,32,32,32]", ha="center", va="center", fontsize=8.5, weight="bold")
    save(fig, "mednext_upblock_detail")


def draw() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    draw_encoder_stage()
    draw_mednext_block()
    draw_decoder_stage()
    draw_downblock_detail()
    draw_upblock_detail()


if __name__ == "__main__":
    draw()
