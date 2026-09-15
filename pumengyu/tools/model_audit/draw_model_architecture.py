#!/usr/bin/env python3
"""Draw one zoomable, code-faithful MedNeXt-L MHA+MoE architecture sheet."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


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
    "skip": "#64748b",
    "note": "#7c2d12",
}


def box(ax, x, y, w, h, title, subtitle="", kind="feature", fs=9):
    fill, edge = COLORS[kind]
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.10",
        linewidth=1.4, edgecolor=edge, facecolor=fill,
    ))
    ax.text(x + w / 2, y + h * 0.65, title, ha="center", va="center", fontsize=fs, weight="bold")
    if subtitle:
        ax.text(x + w / 2, y + h * 0.27, subtitle, ha="center", va="center", fontsize=max(fs - 1.6, 6))


def arrow(ax, start, end, color=None, dashed=False, lw=1.3):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=10, linewidth=lw,
        color=color or COLORS["arrow"],
        linestyle=(0, (4, 3)) if dashed else "solid",
    ))


def orth_arrow(ax, points, color=None, dashed=False, lw=1.3):
    line_color = color or COLORS["arrow"]
    for start, end in zip(points[:-2], points[1:-1]):
        ax.plot([start[0], end[0]], [start[1], end[1]], color=line_color, linewidth=lw,
                linestyle=(0, (4, 3)) if dashed else "solid")
    arrow(ax, points[-2], points[-1], color=line_color, dashed=dashed, lw=lw)


def plus(ax, x, y, r=0.11):
    ax.add_patch(Circle((x, y), r, facecolor="white", edgecolor=COLORS["arrow"], linewidth=1.3, zorder=4))
    ax.text(x, y, "+", ha="center", va="center", fontsize=9, weight="bold", zorder=5)


def panel(ax, x, y, w, h, title):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.14",
        linewidth=1.1, edgecolor=COLORS["panel"][1], facecolor=COLORS["panel"][0], zorder=-2,
    ))
    ax.text(x + 0.22, y + h - 0.28, title, ha="left", va="center", fontsize=10, weight="bold", color="#334155")


def save(fig, output_dir: Path, stem: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def draw_sheet(output_dir: Path):
    fig, ax = plt.subplots(figsize=(18, 26))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 26)
    ax.axis("off")
    ax.text(9, 25.65, "MedNeXt-L + MHA + MoE — code-faithful architecture worksheet", ha="center", fontsize=16, weight="bold")
    ax.text(9, 25.28, "上半部分看信息流；下半部分看每个模块内部如何计算以及可以改哪里", ha="center", fontsize=9.5, color="#475569")

    # ------------------------------------------------------------------
    # A. Network overview: one readable forward path, explicit skip names.
    # ------------------------------------------------------------------
    panel(ax, 0.35, 14.45, 17.3, 10.35, "A. 网络总览：每个张量从哪里来、到哪里去")
    enc_x, dec_x, w, h = 1.0, 13.25, 2.65, 0.58
    enc = [
        (23.65, "Input CT", "[B,1,128³]", "feature"),
        (22.72, "Input Projection", "1 → 32", "transition"),
        (21.79, "Encoder Stage 0 ×3", "[B,32,128³]", "block"),
        (20.86, "DownBlock 0", "32 → 64; /2", "transition"),
        (19.93, "Encoder Stage 1 ×4", "[B,64,64³]", "block"),
        (19.00, "DownBlock 1", "64 → 128; /2", "transition"),
        (18.07, "Encoder Stage 2 ×8", "[B,128,32³]", "block"),
        (17.14, "DownBlock 2", "128 → 256; /2", "transition"),
        (16.21, "Encoder Stage 3 ×8", "[B,256,16³]", "block"),
        (15.28, "DownBlock 3", "256 → 512; /2", "transition"),
    ]
    for y, title, subtitle, kind in enc:
        box(ax, enc_x, y, w, h, title, subtitle, kind, fs=8.1)
    for first, second in zip(enc[:-1], enc[1:]):
        arrow(ax, (enc_x + w / 2, first[0]), (enc_x + w / 2, second[0] + h))

    bx, by, bw, bh = 4.25, 15.12, 3.10, 0.90
    mx, my, mw, mh = 8.00, 15.12, 3.10, 0.90
    box(ax, bx, by, bw, bh, "MedNeXt Bottleneck ×8", "[B,512,8³]", "block", fs=9.2)
    box(ax, mx, my, mw, mh, "MHA Bottleneck ×2", "[B,512,512] tokens; 8 heads", "token", fs=8.8)
    # The bottom of the overview is a straight, code-order path:
    # DownBlock3 -> MedNeXt bottleneck -> MHA bottleneck -> UpBlock3.
    mid_y = by + bh / 2
    arrow(ax, (enc_x + w, mid_y), (bx, mid_y))
    arrow(ax, (bx + bw, mid_y), (mx, mid_y))

    dec = [
        (15.28, "UpBlock 3", "512 → 256; ×2", "transition"),
        (16.20, "Add3 + Decoder Stage 3 ×8", "[B,256,16³]", "block"),
        (17.12, "UpBlock 2", "256 → 128; ×2", "transition"),
        (18.04, "Add2 + Decoder Stage 2 ×8", "[B,128,32³]", "block"),
        (18.96, "UpBlock 1", "128 → 64; ×2", "transition"),
        (19.88, "Add1 + Decoder Stage 1 ×4", "[B,64,64³]", "block"),
        (20.80, "UpBlock 0", "64 → 32; ×2", "transition"),
        (21.72, "Add0 + Decoder Stage 0 ×3", "[B,32,128³]", "block"),
        (22.64, "Segmentation Head", "32 → 3 classes", "output"),
    ]
    for y, title, subtitle, kind in dec:
        box(ax, dec_x, y, w, h, title, subtitle, kind, fs=7.8)
    for first, second in zip(dec[:-1], dec[1:]):
        arrow(ax, (dec_x + w / 2, first[0] + h), (dec_x + w / 2, second[0]))
    arrow(ax, (mx + mw, mid_y), (dec_x, mid_y))

    # Explicitly name every skip source and destination. The line need not be
    # dashed; the label and arrowhead carry the semantics.
    skips = [
        (21.79, 21.72, "E0 → Add0"),
        (19.93, 19.88, "E1 → Add1"),
        (18.07, 18.04, "E2 → Add2"),
        (16.21, 16.20, "E3 → Add3"),
    ]
    for sy, dy, label in skips:
        y1 = sy + h / 2
        y2 = dy + h / 2
        orth_arrow(ax, [(enc_x + w, y1), (5.25, y1), (5.25, y2), (dec_x, y2)], color=COLORS["skip"], lw=1.15)
        ax.text(7.25, (y1 + y2) / 2 + 0.08, label, fontsize=7.5, color=COLORS["skip"], ha="center", va="center", bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5})

    # ------------------------------------------------------------------
    # B. Local MedNeXt block and stage.
    # ------------------------------------------------------------------
    panel(ax, 0.35, 7.55, 8.45, 6.75, "B. Stage / Block：局部三维特征如何被反复加工")
    ax.text(0.72, 13.72, "Stage = 同一个 MedNeXtBlock 按顺序重复；不是 Stage 外再套一个大残差。", fontsize=8.5, color="#475569")
    box(ax, 0.75, 12.15, 1.35, 0.58, "Stage input", "[B,C,D,H,W]", "feature", fs=8)
    box(ax, 2.45, 12.15, 1.45, 0.58, "Block 1", "主分支 + identity", "block", fs=8)
    box(ax, 4.25, 12.15, 1.45, 0.58, "Block 2", "主分支 + identity", "block", fs=8)
    box(ax, 6.05, 12.15, 1.45, 0.58, "Block … ×n", "顺序连接", "block", fs=8)
    for x1, x2 in [(2.10, 2.45), (3.90, 4.25), (5.70, 6.05)]:
        arrow(ax, (x1, 12.44), (x2, 12.44))
    small = [(11.0, "Depthwise Conv3d", "k=3; C→C"), (10.25, "GroupNorm", "batch-independent"), (9.50, "1×1×1 expansion", "C→4C"), (8.75, "GELU", "非线性"), (8.00, "1×1×1 projection", "4C→C")]
    for y, title, sub in small:
        box(ax, 2.55, y, 2.6, 0.46, title, sub, "feature" if "Norm" not in title else "token", fs=7.7)
    for (y1, _, _), (y2, _, _) in zip(small[:-1], small[1:]):
        arrow(ax, (3.85, y1 + 2), (3.85, y2 + 0.46 + 2))
    plus(ax, 6.25, 8.23)
    orth_arrow(ax, [(1.42, 12.15), (1.42, 10.95), (2.55, 10.95)])
    orth_arrow(ax, [(5.15, 8.23), (6.25, 8.23)])
    orth_arrow(ax, [(1.42, 12.15), (1.42, 7.92), (6.25, 7.92), (6.25, 8.11)], color=COLORS["skip"], dashed=True, lw=1.1)
    ax.text(5.20, 7.78, "identity", fontsize=7.8, color=COLORS["skip"], ha="center")
    box(ax, 6.75, 7.92, 1.55, 0.58, "Block output", "main + identity", "output", fs=7.8)
    arrow(ax, (6.37, 8.23), (6.75, 8.23))

    # ------------------------------------------------------------------
    # C. Down/Up block semantics.
    # ------------------------------------------------------------------
    panel(ax, 9.05, 7.55, 8.60, 6.75, "C. DownBlock / UpBlock：改变分辨率时仍保留一条 shortcut")
    ax.text(9.40, 13.72, "DownBlock 与 UpBlock 内部各有一次 resampling shortcut add。", fontsize=8.5, color="#475569")
    box(ax, 9.45, 12.15, 2.0, 0.58, "Input", "[B,C,D,H,W]", "feature", fs=8)
    box(ax, 12.05, 12.55, 2.05, 0.58, "Main path", "DW Conv / ConvT; /2 or ×2", "transition", fs=7.6)
    box(ax, 12.05, 10.95, 2.05, 0.58, "Shortcut", "1×1×1 projection; same resample", "transition", fs=7.4)
    plus(ax, 15.00, 11.77)
    box(ax, 15.35, 11.48, 1.65, 0.58, "Output", "main + shortcut", "output", fs=7.5)
    arrow(ax, (11.45, 12.44), (12.05, 12.84)); arrow(ax, (11.45, 12.44), (12.05, 11.24))
    arrow(ax, (14.10, 12.84), (15.00, 11.90)); arrow(ax, (14.10, 11.24), (15.00, 11.64))
    arrow(ax, (15.12, 11.77), (15.35, 11.77))
    ax.text(9.48, 10.05, "DownBlock：主分支 Conv3d(stride=2)，shortcut 为 1×1×1 Conv3d(stride=2)。", fontsize=7.8, color="#475569")
    ax.text(9.48, 9.62, "UpBlock：主分支 ConvTranspose3d(stride=2)，shortcut 为 1×1×1 ConvTranspose3d(stride=2)。", fontsize=7.8, color="#475569")
    ax.text(9.48, 9.19, "必要时先做 shape 对齐，再 add；不是 concat。", fontsize=7.8, color="#475569")

    # ------------------------------------------------------------------
    # D. Transformer container and one block.  MHA/MoE internals are moved to
    # a separate detail sheet so this panel can keep two genuinely vertical,
    # readable diagrams instead of compressing three levels of detail together.
    # ------------------------------------------------------------------
    panel(ax, 0.35, 0.15, 17.3, 6.95, "D. Bottleneck：竖向看两个 MHA Transformer Block 如何串联")
    ax.text(0.75, 6.62, "左侧是 MHABottleneck3D.forward 的真实顺序；右侧展开一个 block，Block 1 和 Block 2 都各自使用这套结构。", fontsize=8.1, color="#475569")

    # Left: the sequential container is deliberately vertical so X1 is
    # visibly the output of Block 1 and the input to Block 2.
    chain_x, chain_w, chain_h = 0.78, 3.15, 0.43
    chain = [
        (5.72, "Feature map", "[B,512,8³]", "feature"),
        (5.13, "Flatten + transpose", "[B,512,512]", "token"),
        (4.54, "X₀", "token sequence", "token"),
        (3.95, "Transformer Block 1", "pre-LN + MHA + MoE", "block"),
        (3.36, "X₁", "Block 1 output", "token"),
        (2.77, "Transformer Block 2", "pre-LN + MHA + MoE", "block"),
        (2.18, "X₂", "Block 2 output", "token"),
        (1.59, "Final LayerNorm", "512→512", "token"),
        (1.00, "transpose + reshape", "[B,512,8³]", "output"),
    ]
    for y, title, sub, kind in chain:
        box(ax, chain_x, y, chain_w, chain_h, title, sub, kind, fs=7.4)
    for (y1, *_), (y2, *_rest) in zip(chain[:-1], chain[1:]):
        arrow(ax, (chain_x + chain_w / 2, y1), (chain_x + chain_w / 2, y2 + chain_h), lw=1.0)
    ax.text(2.36, 0.62, "Block 1 → X₁ → Block 2", ha="center", fontsize=7.8, color=COLORS["note"])

    # Right: one exact pre-LN transformer block, read from top to bottom.
    ax.text(5.05, 6.18, "单个 Transformer Block：主链和两条 residual 都画清楚", fontsize=8.2, weight="bold", color="#334155")
    block_x, block_w, block_h = 5.25, 2.75, 0.43
    block = [
        (5.60, "Xᵢ", "输入", "token"),
        (5.02, "LayerNorm₁", "512→512", "token"),
        (4.44, "MHA", "Q/K/V + 8 heads", "block"),
        (3.86, "+", "Add_attn", "output"),
        (3.28, "Yᵢ", "MHA 后", "token"),
        (2.70, "LayerNorm₂", "512→512", "token"),
        (2.12, "MoE-FFN", "shared + routed", "moe"),
        (1.54, "+", "Add_moe", "output"),
        (0.96, "Xᵢ₊₁", "block 输出", "output"),
    ]
    for y, title, sub, kind in block:
        if title == "+":
            plus(ax, block_x + block_w / 2, y + block_h / 2, r=0.15)
            ax.text(block_x + block_w / 2, y - 0.13, sub, ha="center", fontsize=6.5, color=COLORS["skip"])
        else:
            box(ax, block_x, y, block_w, block_h, title, sub, kind, fs=7.3)
    for (y1, *_), (y2, *_rest) in zip(block[:-1], block[1:]):
        arrow(ax, (block_x + block_w / 2, y1), (block_x + block_w / 2, y2 + block_h), lw=1.0)
    orth_arrow(ax, [(block_x, 5.82), (4.65, 5.82), (4.65, 4.08), (block_x + block_w / 2 - 0.15, 4.08)], color=COLORS["skip"], lw=1.0)
    ax.text(4.45, 4.88, "Xᵢ → Add_attn", rotation=90, fontsize=6.8, color=COLORS["skip"], ha="center", va="center")
    orth_arrow(ax, [(block_x, 3.50), (4.35, 3.50), (4.35, 1.76), (block_x + block_w / 2 - 0.15, 1.76)], color=COLORS["skip"], lw=1.0)
    ax.text(4.15, 2.62, "Yᵢ → Add_moe", rotation=90, fontsize=6.8, color=COLORS["skip"], ha="center", va="center")

    ax.text(8.75, 0.78, "MHA 与 MoE 的内部投影、路由和专家计算见独立细节图：model_attention_moe_details。", fontsize=7.4, color=COLORS["note"])
    ax.text(8.75, 0.43, "层级核对：Encoder 3/4/8/8；Decoder 8/8/4/3；MedNeXt bottleneck ×8；MHA Transformer Block ×2。", fontsize=7.4, color=COLORS["note"])

    save(fig, output_dir, "model_architecture")


def draw_attention_moe_details(output_dir: Path):
    """Draw MHA and MoE internals as two spacious vertical diagrams."""
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.axis("off")
    ax.text(8, 11.68, "MHA and MoE internal computation", ha="center", va="center", fontsize=16, weight="bold")
    ax.text(8, 11.30, "从 Transformer Block 中拆出两个模块，分别沿纵向展示真实张量流", ha="center", fontsize=9.5, color="#475569")

    panel(ax, 0.35, 0.35, 7.35, 10.55, "A. Multi-head attention：完整 N×N 交互")
    panel(ax, 8.30, 0.35, 7.35, 10.55, "B. MoE-FFN：shared + routed experts")

    # MHA column
    cx = 4.02
    box(ax, 1.55, 9.75, 4.95, 0.62, "Input tokens X", "[B,N,512]", "token", fs=9)
    box(ax, 1.55, 8.55, 2.20, 0.72, "W_Q", "512 → 512", "token", fs=8.5)
    box(ax, 4.30, 8.55, 2.20, 0.72, "W_K / W_V", "512 → 512 each", "token", fs=8.2)
    box(ax, 1.15, 7.05, 5.75, 0.78, "Split into 8 heads", "Q,K,V: [B,8,N,64]", "block", fs=8.7)
    box(ax, 1.15, 5.70, 5.75, 0.78, "Scaled dot-product attention", "QKᵀ / √64 → [B,8,N,N]", "block", fs=8.2)
    box(ax, 1.55, 4.35, 4.95, 0.72, "Softmax weights × V", "[B,8,N,64]", "block", fs=8.5)
    box(ax, 1.55, 3.05, 4.95, 0.72, "Merge 8 heads", "[B,N,512]", "token", fs=8.5)
    box(ax, 1.55, 1.75, 4.95, 0.72, "Output projection WO", "512 → 512", "output", fs=8.5)
    box(ax, 1.55, 0.65, 4.95, 0.62, "MHA output", "[B,N,512]", "output", fs=8.5)
    for y1, y2 in [(9.75, 9.27), (8.55, 8.05), (7.05, 6.55), (5.70, 5.07), (4.35, 3.77), (3.05, 2.47), (1.75, 1.37)]:
        arrow(ax, (cx, y1), (cx, y2), lw=1.1)
    arrow(ax, (2.65, 8.55), (2.65, 7.83), lw=1.0)
    arrow(ax, (5.40, 8.55), (5.40, 7.83), lw=1.0)
    orth_arrow(ax, [(cx, 9.75), (0.95, 9.75), (0.95, 8.91), (1.55, 8.91)], color=COLORS["skip"], lw=1.0)
    orth_arrow(ax, [(cx, 9.75), (7.10, 9.75), (7.10, 8.91), (6.50, 8.91)], color=COLORS["skip"], lw=1.0)

    # MoE column
    mx = 11.98
    box(ax, 9.50, 9.75, 4.95, 0.62, "Feature at one spatial position", "[512] × B×N positions", "token", fs=8.5)
    box(ax, 9.50, 8.45, 2.25, 0.78, "Shared expert", "512 → 1024 → 512", "moe", fs=8.0)
    box(ax, 12.15, 8.45, 2.25, 0.78, "Router", "512 → 4 scores", "moe", fs=8.0)
    box(ax, 9.50, 7.00, 4.95, 0.78, "4 routed experts", "each: 512 → 1024 → 512", "expert", fs=8.2)
    box(ax, 12.15, 5.65, 2.25, 0.78, "Top-2 selection", "scores + balance bias", "moe", fs=8.0)
    box(ax, 9.50, 5.65, 2.25, 0.78, "Selected outputs", "2 × 512", "expert", fs=8.0)
    box(ax, 10.30, 4.20, 3.35, 0.78, "Two gate weights", "softmax(original scores)", "moe", fs=8.0)
    box(ax, 9.50, 2.65, 4.95, 0.78, "Weighted routed output", "512 values", "moe", fs=8.5)
    box(ax, 9.50, 1.20, 4.95, 0.78, "Add shared + routed", "[512] per position", "output", fs=8.5)
    box(ax, 9.50, 0.35, 4.95, 0.56, "MoE output", "[B,N,512]", "output", fs=8.2)
    arrow(ax, (mx, 9.75), (mx, 9.23), lw=1.1)
    orth_arrow(ax, [(mx, 9.75), (9.05, 9.75), (9.05, 8.84), (9.50, 8.84)], lw=1.0)
    orth_arrow(ax, [(mx, 9.75), (14.90, 9.75), (14.90, 8.84), (14.40, 8.84)], lw=1.0)
    arrow(ax, (10.62, 8.45), (10.62, 7.78), lw=1.1)
    arrow(ax, (13.27, 8.45), (13.27, 7.78), lw=1.1)
    arrow(ax, (13.27, 7.00), (13.27, 6.43), lw=1.1)
    arrow(ax, (10.62, 7.00), (10.62, 6.43), lw=1.1)
    arrow(ax, (13.27, 5.65), (13.27, 4.98), lw=1.1)
    arrow(ax, (11.42, 5.65), (11.42, 4.98), lw=1.1)
    orth_arrow(ax, [(13.27, 5.65), (14.75, 5.65), (14.75, 3.04), (14.45, 3.04)], lw=1.0)
    arrow(ax, (11.98, 4.20), (11.98, 3.43), lw=1.1)
    arrow(ax, (11.98, 2.65), (11.98, 1.98), lw=1.1)
    arrow(ax, (11.98, 1.20), (11.98, 0.91), lw=1.1)
    save(fig, output_dir, "model_attention_moe_details")


def write_description(output_dir: Path):
    previous = {}
    json_path = output_dir / "model_architecture.json"
    if json_path.exists():
        try:
            previous = json.loads(json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            previous = {}
    payload = {
        "model": "nnUNetTrainer_MedNeXt_MHA_MoE",
        "architecture": "MedNeXt-L + standard MHA + MoE-FFN",
        "input_shape": "[B,1,128,128,128]",
        "bottleneck_shape": "[B,512,8,8,8]",
        "token_shape": "[B,512,512]",
        "outputs": {
            "png": "model_architecture.png",
            "svg": "model_architecture.svg",
            "detail_png": "model_attention_moe_details.png",
            "detail_svg": "model_attention_moe_details.svg",
            "pipeline_png": "model_pipeline.png",
            "pipeline_svg": "model_pipeline.svg",
            "markdown": "model_architecture.md",
            "json": "model_architecture.json",
        },
        "layer_coverage": {
            "encoder_stages": ["Stage0 ×3", "Stage1 ×4", "Stage2 ×8", "Stage3 ×8"],
            "decoder_stages": ["Stage3 ×8", "Stage2 ×8", "Stage1 ×4", "Stage0 ×3"],
            "bottleneck": ["MedNeXtBlock ×8", "MHATransformerBlock ×2", "final LayerNorm"],
            "transformer_block": ["LayerNorm1", "MHA", "residual add", "LayerNorm2", "MoE-FFN", "residual add"],
            "mha": ["W_Q/W_K/W_V", "8 heads", "attention [B,8,N,N]", "W_O"],
            "moe": ["Router 512→4", "Top-2 + gate", "shared expert", "routed experts 0..3", "gather + sum"],
            "decoder_connections": ["E3 → Add3", "E2 → Add2", "E1 → Add1", "E0 → Add0"],
        },
        "pipeline_figure": {
            "command": "~/model.sh pipeline",
            "png": "model_pipeline.png",
            "svg": "model_pipeline.svg",
            "training": "case list → DefaultPreprocessor → patch/augmentation → forward → deep-supervision Dice+CE → backward/optimizer",
            "inference": "raw case → same preprocessor → sliding window + Gaussian fusion + optional mirror → export/resample/label",
        },
        "skip_connections": ["E0 -> Add0", "E1 -> Add1", "E2 -> Add2", "E3 -> Add3"],
        "modification_map": {
            "local_features": "MedNeXtBlock",
            "resolution": "DownBlock / UpBlock",
            "global_context": "MHA bottleneck",
            "capacity_and_routing": "MoE-FFN",
            "feature_statistics": "GroupNorm / LayerNorm",
        },
        "code_faithful_warning": "Top-2 selects two routed outputs, but all four routed experts are computed before gather.",
    }
    audit = previous.get("static_audit")
    if audit is None and "parameters_total" in previous:
        audit = previous
    if audit is not None:
        payload["static_audit"] = audit
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    audit_note = ""
    if audit is not None:
        audit_note = (
            "\n## 静态审计摘要\n\n"
            f"- 总参数量：`{audit['parameters_total']:,}`\n"
            f"- 归一化层：`{', '.join(sorted({row['class'] for row in audit['normalization_modules']}))}`\n"
            f"- Forward shape trace：`{'已执行' if audit['forward_executed'] else '未执行'}`\n"
        )
    description = (
        "# MedNeXt-MHA-MoE 模型结构图\n\n"
        "这是一张用于代码分析和提出修改思路的结构工作图，不是训练结果。\n\n"
        "## 怎么看\n\n"
        "- 上半部分：从输入到输出的真实信息流，以及 `E0 → Add0` 等明确 skip 来源和目标。\n"
        "- 左下：Stage、MedNeXtBlock 主分支和 identity 残差。\n"
        "- 右下：DownBlock/UpBlock 的重采样 shortcut。\n"
        "- 最下方：3D feature map 如何变成 token，经过 MHA 和 MoE，再还原。\n\n"
        "- MHA/MoE 的投影、注意力矩阵、Router、Top-2 和 expert 汇合见 `model_attention_moe_details.png/svg`。\n\n"
        "- 需要看 case、预处理、损失函数和推理后处理：执行 `~/model.sh pipeline`，查看 `model_pipeline.png/svg`。\n\n"
        "## 修改位置\n\n"
        "- 想改变局部特征：看 MedNeXtBlock。\n"
        "- 想改变空间尺度：看 DownBlock / UpBlock。\n"
        "- 想增强全局关系：看 MHA bottleneck。\n"
        "- 想改变容量或专家选择：看 Router、shared expert 和 routed experts。\n"
        "- 想研究医院域强度统计：看 GroupNorm / LayerNorm，并结合 feature hook 做验证。\n\n"
        "## 当前代码的重要事实\n\n"
        "Router 选择 Top-2，但当前实现会先计算全部 4 个 routed experts，再 gather 选中的输出。\n"
        "因此它不是严格意义上的稀疏计算；这是一个可单独 benchmark 的优化方向。\n"
    )
    (output_dir / "model_architecture.md").write_text(description + audit_note, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("pumengyu/tools/model_audit/output"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    draw_sheet(args.output_dir)
    draw_attention_moe_details(args.output_dir)
    write_description(args.output_dir)
    print("model architecture sheet generated")


if __name__ == "__main__":
    main()
