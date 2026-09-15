#!/usr/bin/env python3
"""Draw a code-faithful MedNeXt-L MHA+MoE architecture map."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch


COLORS = {
    "feature": ("#eaf4ff", "#2563eb"),
    "block": ("#fef9c3", "#ca8a04"),
    "transition": ("#fff3d6", "#d97706"),
    "token": ("#eef2ff", "#4f46e5"),
    "moe": ("#f3e8ff", "#9333ea"),
    "expert": ("#fae8ff", "#c026d3"),
    "output": ("#ecfdf5", "#047857"),
    "panel": ("#f8fafc", "#cbd5e1"),
    "arrow": "#263238",
    "skip": "#6b7280",
}


def box(ax, x, y, w, h, title, subtitle="", kind="feature", fs=9):
    fill, edge = COLORS[kind]
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.10",
        linewidth=1.5, edgecolor=edge, facecolor=fill,
    ))
    ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center", fontsize=fs, weight="bold")
    if subtitle:
        ax.text(x + w / 2, y + h * 0.28, subtitle, ha="center", va="center", fontsize=fs - 1.5)


def arrow(ax, start, end, color=None, dashed=False):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=11, linewidth=1.5,
        color=color or COLORS["arrow"],
        linestyle=(0, (4, 3)) if dashed else "solid",
    ))


def orth_arrow(ax, points, color=None, dashed=False):
    """Draw a horizontal/vertical path with one arrowhead at the end."""
    line_color = color or COLORS["arrow"]
    for start, end in zip(points[:-2], points[1:-1]):
        ax.plot([start[0], end[0]], [start[1], end[1]], color=line_color, linewidth=1.5,
                linestyle=(0, (4, 3)) if dashed else "solid")
    arrow(ax, points[-2], points[-1], color=line_color, dashed=dashed)


def plus(ax, x, y):
    ax.add_patch(Circle((x, y), 0.12, facecolor="white", edgecolor=COLORS["arrow"], linewidth=1.4, zorder=3))
    ax.text(x, y, "+", ha="center", va="center", fontsize=10, weight="bold", zorder=4)


def save(fig, output_dir: Path, stem: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def draw_overview(output_dir: Path):
    fig, ax = plt.subplots(figsize=(11.5, 15))
    ax.set_xlim(0, 11.5)
    ax.set_ylim(0, 16)
    ax.axis("off")
    ax.text(5.75, 15.65, "MedNeXt-L + MHA + MoE: code-faithful network map", ha="center", fontsize=15, weight="bold")
    ax.text(2.35, 15.20, "Encoder: local 3D features", ha="center", fontsize=11, weight="bold", color="#1d4ed8")
    ax.text(9.15, 15.20, "Decoder: recover spatial resolution", ha="center", fontsize=11, weight="bold", color="#15803d")

    enc_x, dec_x, w, h = 1.05, 8.0, 2.65, 0.70
    enc = [
        (14.25, "Input CT", "[B,1,128,128,128]", "feature"),
        (13.20, "Input Projection", "1 → 32 channels", "transition"),
        (12.15, "Encoder Stage 0 ×3", "[B,32,128³]", "block"),
        (10.75, "DownBlock 0", "32 → 64; /2", "transition"),
        (9.70, "Encoder Stage 1 ×4", "[B,64,64³]", "block"),
        (8.30, "DownBlock 1", "64 → 128; /2", "transition"),
        (7.25, "Encoder Stage 2 ×8", "[B,128,32³]", "block"),
        (5.85, "DownBlock 2", "128 → 256; /2", "transition"),
        (4.80, "Encoder Stage 3 ×8", "[B,256,16³]", "block"),
        (3.40, "DownBlock 3", "256 → 512; /2", "transition"),
    ]
    for y, title, subtitle, kind in enc:
        box(ax, enc_x, y, w, h, title, subtitle, kind)
    for first, second in zip(enc[:-1], enc[1:]):
        arrow(ax, (enc_x + w / 2, first[0]), (enc_x + w / 2, second[0] + h))

    bx, by, bw, bh = 4.25, 2.55, 3.0, 1.35
    box(ax, bx, by, bw, bh, "MedNeXt Bottleneck ×8", "[B,512,8³]", "block", fs=10)
    orth_arrow(ax, [(enc_x + w / 2, 3.40), (3.55, 3.40), (3.55, by + bh / 2), (bx, by + bh / 2)])
    box(ax, 4.25, 0.75, 3.0, 1.05, "MHA Bottleneck ×2", "8 heads; [B,512,512] tokens", "token", fs=10)
    arrow(ax, (bx + bw / 2, by), (5.75, 1.80))

    dec = [
        (4.25, "UpBlock 3", "512 → 256; ×2", "transition"),
        (5.35, "Skip add + Decoder Stage 3 ×8", "[B,256,16³]", "block"),
        (6.75, "UpBlock 2", "256 → 128; ×2", "transition"),
        (7.85, "Skip add + Decoder Stage 2 ×8", "[B,128,32³]", "block"),
        (9.25, "UpBlock 1", "128 → 64; ×2", "transition"),
        (10.35, "Skip add + Decoder Stage 1 ×4", "[B,64,64³]", "block"),
        (11.75, "UpBlock 0", "64 → 32; ×2", "transition"),
        (12.85, "Skip add + Decoder Stage 0 ×3", "[B,32,128³]", "block"),
        (14.25, "Segmentation Head", "32 → 3 classes", "output"),
    ]
    for y, title, subtitle, kind in dec:
        box(ax, dec_x, y, w, h, title, subtitle, kind)
    for first, second in zip(dec[:-1], dec[1:]):
        arrow(ax, (dec_x + w / 2, first[0] + h), (dec_x + w / 2, second[0]))
    orth_arrow(ax, [(7.25, 1.27), (7.45, 1.27), (7.45, 4.60), (dec_x, 4.60)])

    # Dashed lines are the four encoder-decoder additive skips.
    skip_pairs = [(12.15, 5.70), (9.70, 7.20), (7.25, 9.80), (4.80, 12.30)]
    for sy, dy in skip_pairs:
        ax.plot([enc_x + w, 6.9, 6.9, dec_x], [sy + h / 2, sy + h / 2, dy + h / 2, dy + h / 2],
                color=COLORS["skip"], linewidth=1.3, linestyle=(0, (4, 3)))
        arrow(ax, (6.9, dy + h / 2), (dec_x, dy + h / 2), color=COLORS["skip"], dashed=True)
        plus(ax, dec_x, dy + h / 2)
    ax.text(0.25, 0.15, "Solid: forward path   Dashed: encoder skip/add   Shapes use the 128³ source plan.", fontsize=8.5, color="#475569")
    save(fig, output_dir, "mednext_mha_moe_overview")


def draw_bottleneck(output_dir: Path):
    fig, ax = plt.subplots(figsize=(12, 8.0))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis("off")
    ax.text(6, 7.65, "MHA + MoE bottleneck: what is actually computed", ha="center", fontsize=15, weight="bold")
    box(ax, 0.35, 5.55, 2.0, 0.9, "Feature map", "[B,512,8,8,8]", "feature", fs=10)
    box(ax, 2.85, 5.55, 2.0, 0.9, "Flatten + transpose", "[B,512,512]", "token", fs=10)
    box(ax, 5.35, 5.55, 2.0, 0.9, "MHA block ×2", "8 heads; LN + residual", "block", fs=10)
    arrow(ax, (2.35, 6.0), (2.85, 6.0)); arrow(ax, (4.85, 6.0), (5.35, 6.0))

    panel = FancyBboxPatch((0.35, 0.55), 11.3, 4.15, boxstyle="round,pad=0.04,rounding_size=0.15", linewidth=1.2, edgecolor=COLORS["panel"][1], facecolor=COLORS["panel"][0])
    ax.add_patch(panel)
    ax.text(0.60, 4.38, "One Transformer block: x = x + MHA(LN(x)); then x = x + MoE(LN(x))", fontsize=10, weight="bold")
    box(ax, 0.75, 2.55, 1.55, 0.75, "LayerNorm", "512 → 512", "token", fs=8)
    box(ax, 2.65, 2.55, 1.55, 0.75, "8-head MHA", "Q,K,V: 512", "block", fs=8)
    plus(ax, 4.70, 2.93)
    box(ax, 5.15, 2.55, 1.55, 0.75, "LayerNorm", "512 → 512", "token", fs=8)
    box(ax, 7.05, 2.55, 1.55, 0.75, "Router", "512 → 4 scores", "moe", fs=8)
    box(ax, 9.00, 3.25, 1.9, 0.55, "Shared expert", "512→1024→512", "expert", fs=8)
    box(ax, 9.00, 2.45, 1.9, 0.55, "Routed expert 0", "512→1024→512", "expert", fs=8)
    box(ax, 9.00, 1.65, 1.9, 0.55, "Routed expert 1", "512→1024→512", "expert", fs=8)
    box(ax, 9.00, 0.85, 1.9, 0.55, "Routed experts 2, 3", "also computed", "expert", fs=8)
    box(ax, 5.15, 0.95, 1.55, 0.75, "Top-2 + gate", "select / weight", "moe", fs=8)
    plus(ax, 7.25, 1.32)
    box(ax, 7.75, 0.95, 1.0, 0.75, "Output", "[B,512,512]", "output", fs=8)
    arrow(ax, (2.30, 2.93), (2.65, 2.93)); arrow(ax, (4.20, 2.93), (4.58, 2.93)); arrow(ax, (4.82, 2.93), (5.15, 2.93)); arrow(ax, (6.70, 2.93), (7.05, 2.93))
    for y in (3.52, 2.72, 1.92, 1.12):
        arrow(ax, (8.60, 2.93), (9.00, y))
    for y in (3.52, 2.72, 1.92, 1.12):
        arrow(ax, (9.95, y), (6.0, 1.70))
    arrow(ax, (6.70, 1.32), (7.13, 1.32)); arrow(ax, (7.37, 1.32), (7.75, 1.32))
    ax.text(0.60, 0.18, "Code-faithful warning: Top-2 chooses two outputs, but all 4 routed experts are computed before gather.", fontsize=9, color="#7c2d12")
    save(fig, output_dir, "mednext_mha_moe_bottleneck")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("pumengyu/tools/model_audit/output"))
    args = parser.parse_args()
    draw_overview(args.output_dir)
    draw_bottleneck(args.output_dir)
    print("graph generated")


if __name__ == "__main__":
    main()
