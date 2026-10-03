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
        shrinkA=0, shrinkB=0, zorder=3,
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


def save(fig, output_dir: Path, stem: str, export_png: bool = False):
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.svg", bbox_inches="tight")
    if export_png:
        png_dir = output_dir / "PNG格式"
        png_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(png_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def expanded_block(ax, y0, number, input_name, mid_name, output_name, cx=6):
    panel(ax, cx - 5.35, y0, 10.70, 8.05, f"Transformer Block {number}")
    node_x, node_w, node_h = cx - 1.75, 3.50, 0.48
    nodes = [
        (y0 + 6.90, input_name, "block 输入", "token"),
        (y0 + 6.03, "LayerNorm₁", "512 → 512", "token"),
        (y0 + 5.16, "MHA", "Q/K/V；8 heads", "block"),
        (y0 + 3.65, mid_name, "Attention residual 输出", "token"),
        (y0 + 2.78, "LayerNorm₂", "512 → 512", "token"),
        (y0 + 1.91, "MoE-FFN", "shared expert + Top-2 routed", "moe"),
        (y0 + 0.35, output_name, "MoE residual 输出", "output"),
    ]
    for y, title, sub, kind in nodes:
        box(ax, node_x, y, node_w, node_h, title, sub, kind, fs=8.0)

    attn_plus_y = y0 + 4.57
    moe_plus_y = y0 + 1.32
    plus(ax, cx, attn_plus_y, r=0.15)
    plus(ax, cx, moe_plus_y, r=0.15)
    ax.text(cx + 0.35, attn_plus_y, "Add_attn", va="center", fontsize=7.2, color=COLORS["skip"])
    ax.text(cx + 0.35, moe_plus_y, "Add_moe", va="center", fontsize=7.2, color=COLORS["skip"])

    main_items = [
        (y0 + 6.90, False), (y0 + 6.03, False), (y0 + 5.16, False),
        (attn_plus_y, True), (y0 + 3.65, False), (y0 + 2.78, False),
        (y0 + 1.91, False), (moe_plus_y, True), (y0 + 0.35, False),
    ]
    for (upper_y, upper_plus), (lower_y, lower_plus) in zip(main_items[:-1], main_items[1:]):
        upper_bottom = upper_y - 0.15 if upper_plus else upper_y
        lower_top = lower_y + 0.15 if lower_plus else lower_y + node_h
        arrow(ax, (cx, upper_bottom), (cx, lower_top), lw=1.0)

    input_center_y = y0 + 6.90 + node_h / 2
    mid_center_y = y0 + 3.65 + node_h / 2
    orth_arrow(ax, [(node_x, input_center_y), (cx - 3.28, input_center_y),
                   (cx - 3.28, attn_plus_y), (cx - 0.15, attn_plus_y)], color=COLORS["skip"], lw=1.05)
    orth_arrow(ax, [(node_x, mid_center_y), (cx - 3.65, mid_center_y),
                   (cx - 3.65, moe_plus_y), (cx - 0.15, moe_plus_y)], color=COLORS["skip"], lw=1.05)
    ax.text(cx - 3.50, (input_center_y + attn_plus_y) / 2, f"{input_name} residual", rotation=90,
            ha="center", va="center", fontsize=7.0, color=COLORS["skip"])
    ax.text(cx - 3.87, (mid_center_y + moe_plus_y) / 2, f"{mid_name} residual", rotation=90,
            ha="center", va="center", fontsize=7.0, color=COLORS["skip"])


def draw_moe_panel(ax, y0):
    """MoEFFN.forward: data branches and score/index dependencies are explicit."""
    panel(ax, 0.35, y0, 17.3, 18.0, "E. MoE-FFN：共享专家 + 按空间位置选择的 Top-2 路由专家")
    ax.text(0.75, y0 + 17.30, "每个 token 是三维特征图中一个位置的 512 维向量；每个位置独立选择专家。N = 8×8×8 = 512。",
            fontsize=8.5, color="#475569")

    def b(x, y, w, title, sub, kind="token", h=0.72):
        box(ax, x, y0 + y, w, h, title, sub, kind, fs=8.4)

    def a(start, end, **kwargs):
        arrow(ax, (start[0], y0 + start[1]), (end[0], y0 + end[1]), **kwargs)

    def route(points, **kwargs):
        orth_arrow(ax, [(x, y0 + y) for x, y in points], **kwargs)

    b(6.50, 16.05, 5.0, "Input: LayerNorm₂ output", "[B,N,512]")
    b(6.50, 14.85, 5.0, "reshape", "[B×N,512]")
    a((9, 16.05), (9, 15.57))
    # All three branches consume the same input, independently.
    for cx in (2.95, 7.80, 13.80):
        route([(9, 14.85), (9, 14.35), (cx, 14.35), (cx, 13.32)])
    b(1.20, 12.60, 3.50, "Shared expert · 共享专家", "始终计算：512 → 1024 → 512", "expert")
    b(5.60, 12.60, 4.40, "4 个独立路由专家 E₀–E₃", "每个：512 → 1024 → 512", "expert")
    b(11.70, 12.60, 4.20, "Router · 专家打分器", "Linear 512 → 4；无 bias", "moe")
    ax.text(8.05, y0 + 12.05, "全部计算并 stack：[B×N,4,512]", ha="left", fontsize=7.6, color=COLORS["note"])

    b(10.40, 10.75, 3.0, "Top-2 选择", "scores + expert_bias", "moe")
    b(14.10, 10.75, 2.80, "原始 scores", "不加均衡偏置", "moe")
    route([(13.8, 12.60), (13.8, 11.95), (11.90, 11.95), (11.90, 11.47)])
    route([(13.8, 12.60), (13.8, 11.95), (15.50, 11.95), (15.50, 11.47)])

    b(5.60, 8.30, 4.40, "gather：选中专家的输出", "[B×N,2,512]", "expert")
    b(11.70, 8.30, 4.20, "gather 原始分数 → softmax", "两个 gate 权重：[B×N,2]", "moe")
    a((7.80, 12.60), (7.80, 9.02))
    route([(10.40, 11.11), (10.20, 11.11), (10.20, 9.55), (9.20, 9.55), (9.20, 9.02)], color=COLORS["skip"])
    a((11.90, 10.75), (11.90, 9.02), color=COLORS["skip"])
    a((15.50, 10.75), (15.50, 9.02))
    ax.text(12.10, y0 + 9.88, "选中索引", fontsize=7.2, color=COLORS["skip"])
    ax.text(14.45, y0 + 9.88, "原始分数", fontsize=7.2, color=COLORS["skip"])

    b(6.0, 6.45, 8.0, "加权求和：w₁ × selected₁ + w₂ × selected₂", "沿 2 个选中专家求和 → [B×N,512]", "moe")
    a((7.80, 8.30), (7.80, 7.17))
    route([(13.80, 8.30), (13.80, 7.75), (12.20, 7.75), (12.20, 7.17)])
    plus(ax, 10, y0 + 5.35, r=0.16)
    a((10, 6.45), (10, 5.51))
    route([(2.95, 12.60), (2.95, 5.35), (9.84, 5.35)])
    ax.text(3.15, y0 + 5.58, "shared output：[B×N,512]", fontsize=7.8, color=COLORS["skip"])
    ax.text(10.35, y0 + 5.35, "共享输出 + 路由输出", va="center", fontsize=8.0)
    b(7.50, 3.95, 5.0, "reshape → MoE-FFN output", "[B,N,512] → 图 D 的 Add_moe", "output")
    a((10, 5.19), (10, 4.67))

    ax.text(0.85, y0 + 3.15, "每个专家的内部结构（共享专家和路由专家结构相同，参数各自独立）", fontsize=9, weight="bold", color="#334155")
    expert_layers = [
        (1.0, "Linear", "512 → 1024"),
        (4.25, "GELU", "非线性激活"),
        (7.50, "Dropout", "当前 p=0"),
        (10.75, "Linear", "1024 → 512"),
        (14.0, "Dropout", "当前 p=0"),
    ]
    for x, title, sub in expert_layers:
        b(x, 1.85, 2.5, title, sub, "expert")
    for (x1, _, _), (x2, _, _) in zip(expert_layers[:-1], expert_layers[1:]):
        a((x1 + 2.5, 2.21), (x2, 2.21))
    ax.text(0.85, y0 + 1.10, "Top-2 选择输出，不代表只计算两个专家；均衡偏置只用于选择，不进入 gate 权重。", fontsize=8.3, color=COLORS["note"])
    ax.text(0.85, y0 + 0.55, "训练时记录被选次数；反向传播后更新负载 EMA 与均衡偏置。图内加号不包含图 D 的外层 residual。", fontsize=8.1, color="#475569")


def draw_sheet(output_dir: Path, export_png: bool = False):
    fig, ax = plt.subplots(figsize=(18, 60.5))
    ax.set_xlim(0, 18)
    ax.set_ylim(-34.5, 26)
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
    box(ax, mx, my, mw, mh, "MHA + MoE Blocks ×2", "each: attention → MoE-FFN", "token", fs=8.8)
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
    panel(ax, 0.35, 7.25, 8.45, 7.05, "B. Stage / Block：局部三维特征如何被反复加工")
    ax.text(0.72, 13.72, "Stage = 同一个 MedNeXtBlock 按顺序重复；不是 Stage 外再套一个大残差。", fontsize=8.5, color="#475569")
    box(ax, 0.72, 12.15, 1.35, 0.58, "Stage input", "[B,C,D,H,W]", "feature", fs=8)
    box(ax, 2.59, 12.15, 1.45, 0.58, "Block 1", "主分支 + identity", "block", fs=8)
    box(ax, 4.56, 12.15, 1.45, 0.58, "Block 2", "主分支 + identity", "block", fs=8)
    box(ax, 6.53, 12.15, 1.45, 0.58, "Block … ×n", "顺序连接", "block", fs=8)
    for x1, x2 in [(2.07, 2.59), (4.04, 4.56), (6.01, 6.53)]:
        arrow(ax, (x1, 12.44), (x2, 12.44))
    small = [(11.0, "Depthwise Conv3d", "k=3; C→C"), (10.25, "GroupNorm", "batch-independent"), (9.50, "1×1×1 expansion", "C→4C"), (8.75, "GELU", "非线性"), (8.00, "1×1×1 projection", "4C→C")]
    for y, title, sub in small:
        box(ax, 2.55, y, 2.6, 0.46, title, sub, "feature" if "Norm" not in title else "token", fs=7.7)
    for (y1, _, _), (y2, _, _) in zip(small[:-1], small[1:]):
        arrow(ax, (3.85, y1), (3.85, y2 + 0.46))
    plus(ax, 6.25, 8.23)
    orth_arrow(ax, [(1.395, 12.15), (1.395, 11.23), (2.55, 11.23)])
    arrow(ax, (5.15, 8.23), (6.14, 8.23))
    orth_arrow(ax, [(1.395, 12.15), (1.395, 7.70), (6.25, 7.70), (6.25, 8.12)], color=COLORS["skip"], dashed=True, lw=1.1)
    ax.text(5.20, 7.40, "identity", fontsize=7.8, color=COLORS["skip"], ha="center")
    box(ax, 6.75, 7.94, 1.55, 0.58, "Block output", "main + identity", "output", fs=7.8)
    arrow(ax, (6.36, 8.23), (6.75, 8.23))

    # ------------------------------------------------------------------
    # C. Down/Up block semantics.
    # ------------------------------------------------------------------
    panel(ax, 9.05, 7.25, 8.60, 7.05, "C. DownBlock / UpBlock：改变分辨率时仍保留一条 shortcut")
    ax.text(9.40, 13.72, "DownBlock 与 UpBlock 内部各有一次 resampling shortcut add。", fontsize=8.5, color="#475569")
    box(ax, 9.45, 12.15, 2.0, 0.58, "Input", "[B,C,D,H,W]", "feature", fs=8)
    box(ax, 12.05, 12.55, 2.05, 0.58, "Main path", "DW Conv / ConvT; /2 or ×2", "transition", fs=7.6)
    box(ax, 12.05, 10.95, 2.05, 0.58, "Shortcut", "1×1×1 projection; same resample", "transition", fs=7.4)
    plus(ax, 15.00, 11.77)
    box(ax, 15.55, 11.48, 1.45, 0.58, "Output", "main + shortcut", "output", fs=7.5)
    orth_arrow(ax, [(11.45, 12.44), (11.75, 12.44), (11.75, 12.84), (12.05, 12.84)])
    orth_arrow(ax, [(11.45, 12.44), (11.75, 12.44), (11.75, 11.24), (12.05, 11.24)])
    orth_arrow(ax, [(14.10, 12.84), (15.00, 12.84), (15.00, 11.88)])
    orth_arrow(ax, [(14.10, 11.24), (15.00, 11.24), (15.00, 11.66)])
    arrow(ax, (15.11, 11.77), (15.55, 11.77))
    ax.text(9.48, 10.05, "DownBlock：主分支 Conv3d(stride=2)，shortcut 为 1×1×1 Conv3d(stride=2)。", fontsize=7.8, color="#475569")
    ax.text(9.48, 9.62, "UpBlock：主分支 ConvTranspose3d(stride=2)，shortcut 为 1×1×1 ConvTranspose3d(stride=2)。", fontsize=7.8, color="#475569")
    ax.text(9.48, 9.19, "必要时先做 shape 对齐，再 add；不是 concat。", fontsize=7.8, color="#475569")

    # ------------------------------------------------------------------
    # D. One vertical main path, with local residuals entering from the side.
    # Increase sheet height rather than folding sequential blocks into rows.
    # ------------------------------------------------------------------
    panel(ax, 0.35, -15.80, 17.3, 22.90,
          "D. Bottleneck：两个完整的 MHA + MoE Transformer Block 自上而下串联")
    ax.text(0.75, 6.48, "沿中轴从上往下读；每个 Block 先完成 Attention 残差，再完成 MoE 残差。",
            fontsize=8.5, color="#475569")

    cx = 9
    box(ax, cx - 2.45, 5.30, 4.90, 0.62, "3D feature map", "[B,512,8,8,8]", "feature", fs=8.8)
    box(ax, cx - 2.45, 4.20, 4.90, 0.62, "Flatten + transpose", "X₀: [B,512,512]", "token", fs=8.8)
    arrow(ax, (cx, 5.30), (cx, 4.82), lw=1.1)

    expanded_block(ax, -4.25, 1, "X₀", "Y₀", "X₁", cx=cx)
    arrow(ax, (cx, 4.20), (cx, 3.13), lw=1.1)
    expanded_block(ax, -12.75, 2, "X₁", "Y₁", "X₂", cx=cx)
    arrow(ax, (cx, -3.90), (cx, -5.37), lw=1.1)

    box(ax, cx - 2.45, -13.85, 4.90, 0.62, "Final LayerNorm", "X₂: [B,512,512]", "token", fs=8.8)
    box(ax, cx - 2.45, -14.95, 4.90, 0.62, "transpose + reshape", "[B,512,8,8,8]", "output", fs=8.8)
    arrow(ax, (cx, -12.40), (cx, -13.23), lw=1.1)
    arrow(ax, (cx, -13.85), (cx, -14.33), lw=1.1)
    ax.text(cx, -15.48, "输出送入 UpBlock 3；MHA / MoE 内部计算见 model_attention_moe_details。",
            fontsize=8, color=COLORS["note"], ha="center")

    draw_moe_panel(ax, -34.15)
    save(fig, output_dir, "model_architecture", export_png=export_png)


def draw_bottleneck_blocks(output_dir: Path, export_png: bool = False):
    """Expand both sequential MHA+MoE blocks, including both residual paths."""
    fig, ax = plt.subplots(figsize=(12, 22))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 22)
    ax.axis("off")
    ax.text(6, 21.65, "MHA + MoE Bottleneck：两个完整 Transformer Block", ha="center", fontsize=16, weight="bold")
    ax.text(6, 21.25, "每个 block 都依次执行 Attention 子层和 MoE-FFN 子层；两处都带 residual add", ha="center", fontsize=9.4, color="#475569")

    box(ax, 3.55, 20.20, 4.90, 0.62, "3D feature map", "[B,512,8,8,8]", "feature", fs=8.8)
    box(ax, 3.55, 19.15, 4.90, 0.62, "Flatten + transpose", "X₀: [B,512,512]", "token", fs=8.8)
    arrow(ax, (6, 20.20), (6, 19.77), lw=1.1)

    expanded_block(ax, 10.75, 1, "X₀", "Y₀", "X₁")
    arrow(ax, (6, 19.15), (6, 18.13), lw=1.1)
    arrow(ax, (6, 10.75 + 0.35), (6, 9.63), lw=1.2)
    ax.text(6.25, 10.53, "X₁ 继续进入 Block 2", fontsize=7.6, color=COLORS["note"], va="center")
    expanded_block(ax, 2.25, 2, "X₁", "Y₁", "X₂")

    box(ax, 3.55, 1.22, 4.90, 0.56, "Final LayerNorm", "X₂: [B,512,512]", "token", fs=8.4)
    box(ax, 3.55, 0.22, 4.90, 0.56, "transpose + reshape", "[B,512,8,8,8]", "output", fs=8.4)
    arrow(ax, (6, 2.25 + 0.35), (6, 1.78), lw=1.1)
    arrow(ax, (6, 1.22), (6, 0.78), lw=1.1)
    save(fig, output_dir, "model_bottleneck_blocks", export_png=export_png)


def draw_attention_moe_details(output_dir: Path, export_png: bool = False):
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
    save(fig, output_dir, "model_attention_moe_details", export_png=export_png)


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
            "svg": "model_architecture.svg",
            "bottleneck_svg": "model_bottleneck_blocks.svg",
            "detail_svg": "model_attention_moe_details.svg",
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
        "- 最下方：3D feature map 如何变成 token，并依次通过两个 MHA+MoE Transformer Block。\n\n"
        "- 两个 block 各自完整的 `MHA → residual → MoE → residual` 路径见 `model_bottleneck_blocks.svg`。\n"
        "- MHA/MoE 的投影、注意力矩阵、Router、Top-2 和 expert 汇合见 `model_attention_moe_details.svg`。\n\n"
        "- 需要看 case、预处理、损失函数和推理后处理：执行 `~/model.sh pipeline`，查看 `model_pipeline.svg`。\n\n"
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
    parser.add_argument("--output-dir", type=Path, default=Path("pumengyu/notes/模型结构图"))
    parser.add_argument("--png", action="store_true", help="Also export 300 DPI PNG for analysis or papers")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    draw_sheet(args.output_dir, export_png=args.png)
    draw_bottleneck_blocks(args.output_dir, export_png=args.png)
    draw_attention_moe_details(args.output_dir, export_png=args.png)
    write_description(args.output_dir)
    print("model architecture sheet generated")


if __name__ == "__main__":
    main()
