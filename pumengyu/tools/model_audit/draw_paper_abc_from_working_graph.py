#!/usr/bin/env python3
"""Create paper-ready A/B/C figures from the original detailed working graph.

The geometry and colours intentionally reuse draw_model_architecture.py, while
panel A follows the locked paper method: MedNeXt-L + MLA + MoE-FFN. The source
working sheet is never changed by this script.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.patches import FancyBboxPatch

from draw_model_architecture import COLORS, arrow, orth_arrow, plus


DEFAULT_OUT = Path("/home/PuMengYu/葛老师的指导/模型结构图/论文拆分图")
CT_REFERENCE = Path(
    "/home/PuMengYu/8T/nnUNet_result/results_v2_moe_sizeov4_retrain_after_fix_20260910/"
    "IRCADb/source_only/MedNeXt_MLA_MoE_SizeOV4/test_viz/ircadb_008/ircadb_008_z110_full.png"
)
PREDICTION_REFERENCE = Path(
    "/home/PuMengYu/8T/nnUNet_result/results_v2_moe_sizeov4_retrain_after_fix_20260910/"
    "IRCADb/source_only/MedNeXt_MLA_MoE_SizeOV4/predictions/ircadb_008.nii.gz"
)
REFERENCE_SLICE = 110

# Stronger paper-figure colour blocks. Updating the imported dictionary in
# place also keeps arrow/plus helpers on the same palette.
COLORS.update({
    "feature": ("#f7f8fa", "#5f6b7a"),
    "backbone": ("#fde9a9", "#b86b00"),
    "block": ("#fde9a9", "#b86b00"),
    "transition": ("#eceff3", "#5f6b7a"),
    "resample": ("#d8f3ee", "#007c6a"),
    "mla": ("#d9edfa", "#0072b2"),
    "token": ("#e4e0f7", "#5e55a8"),
    "moe": ("#f0ddf0", "#9c4f96"),
    "expert": ("#f8d9e8", "#b64e82"),
    "output": ("#d2f1e6", "#007c6a"),
    "panel": ("#ffffff", "#cbd5e1"),
    "skip": "#b86b00",
})


def box(ax, x, y, w, h, title, subtitle="", kind="feature", fs=10):
    """Compact colour block with text sized to use the available box."""
    fill, edge = COLORS[kind]
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.10",
        linewidth=1.5, edgecolor=edge, facecolor=fill,
    ))
    title_y = y + h * (0.63 if subtitle else 0.50)
    ax.text(x + w / 2, title_y, title, ha="center", va="center", fontsize=fs, weight="bold", color="#111827")
    if subtitle:
        ax.text(x + w / 2, y + h * 0.27, subtitle, ha="center", va="center",
                fontsize=max(fs - 1.0, 7.2), color="#1f2937")


def panel(ax, x, y, w, h, title):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.14",
        linewidth=1.1, edgecolor=COLORS["panel"][1], facecolor=COLORS["panel"][0], zorder=-2,
    ))
    ax.text(x + w / 2, y + h - 0.28, title, ha="center", va="center",
            fontsize=11.5, weight="bold", color="#1e293b")


def image_tile(ax, image, extent, label: str, edge: str) -> None:
    """Small paper-style image tile with two offset slices suggesting a volume."""
    x0, x1, y0, y1 = extent
    dx, dy = 0.08, 0.08
    for scale in (2, 1):
        ax.add_patch(FancyBboxPatch(
            (x0 - scale * dx, y0 + scale * dy), x1 - x0, y1 - y0,
            boxstyle="round,pad=0.01,rounding_size=0.03",
            linewidth=0.8, edgecolor="#64748b", facecolor="#e2e8f0", zorder=0,
        ))
    ax.imshow(image, extent=extent, interpolation="bilinear", zorder=1, aspect="auto")
    ax.add_patch(FancyBboxPatch(
        (x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0.01,rounding_size=0.03",
        linewidth=1.5, edgecolor=edge, facecolor="none", zorder=2,
    ))
    ax.text((x0 + x1) / 2, y1 + 0.14, label, ha="center", va="bottom",
            fontsize=9.5, weight="bold", color="#111827")


def prediction_mask_tile() -> np.ndarray:
    """Render the real three-class prediction for the same case/slice as the CT tile."""
    prediction = nib.load(PREDICTION_REFERENCE).get_fdata()
    if prediction.ndim != 3 or REFERENCE_SLICE >= prediction.shape[2]:
        raise ValueError(
            f"Invalid prediction shape {prediction.shape} for z={REFERENCE_SLICE}: "
            f"{PREDICTION_REFERENCE}"
        )

    # The evaluation visualizer uses prediction[:, :, z].T with origin='lower'.
    # image_tile uses Matplotlib's default upper origin, hence the vertical flip.
    labels = np.flipud(prediction[:, :, REFERENCE_SLICE].T)
    rgb = np.zeros((*labels.shape, 3), dtype=np.float32)
    rgb[labels == 1] = (0.20, 0.72, 0.48)  # liver
    rgb[labels == 2] = (0.96, 0.43, 0.18)  # tumour
    return rgb


def legend_item(ax, x, y, label: str, kind: str) -> None:
    fill, edge = COLORS[kind]
    ax.add_patch(FancyBboxPatch(
        (x, y), 0.34, 0.24, boxstyle="round,pad=0.01,rounding_size=0.03",
        linewidth=1.0, edgecolor=edge, facecolor=fill,
    ))
    ax.text(x + 0.44, y + 0.12, label, ha="left", va="center",
            fontsize=8.4, color="#111827")


def vertical_legend(ax, x, y, w, h) -> None:
    """Compact right-side legend, following common medical-network figures."""
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.12",
        linewidth=1.0, linestyle=(0, (5, 3)), edgecolor="#94a3b8",
        facecolor="#ffffff",
    ))
    ax.text(x + w / 2, y + h - 0.28, "Module colors", ha="center", va="center",
            fontsize=9.2, weight="bold", color="#1f2937")
    entries = [
        ("MedNeXt stage", "", "backbone"),
        ("Basic operation", "Projection / Head\nDown / Up", "transition"),
        ("MLA", "", "mla"),
        ("MoE-FFN", "", "moe"),
        ("Prediction", "", "output"),
    ]
    row_y = y + h - 0.88
    for title, subtitle, kind in entries:
        fill, edge = COLORS[kind]
        ax.add_patch(FancyBboxPatch(
            (x + 0.16, row_y - 0.13), 0.38, 0.27,
            boxstyle="round,pad=0.01,rounding_size=0.03",
            linewidth=1.0, edgecolor=edge, facecolor=fill,
        ))
        ax.text(x + 0.66, row_y + (0.07 if subtitle else 0.0), title,
                ha="left", va="center", fontsize=8.1, color="#111827", weight="bold")
        if subtitle:
            ax.text(x + 0.66, row_y - 0.14, subtitle, ha="left", va="center",
                    fontsize=6.4, color="#475569", linespacing=0.95)
        row_y -= 0.64


def mla_moe_pair(ax, x, y, w, h) -> None:
    """Compact, truthful MLA then MoE sequence repeated twice."""
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.025,rounding_size=0.10",
        linewidth=1.2, edgecolor="#475569", facecolor="#ffffff",
    ))
    ax.text(x + w / 2, y + h - 0.16, "Transformer Block ×2",
            ha="center", va="center", fontsize=9.5, weight="bold", color="#111827")
    inner_y, inner_h = y + 0.10, h * 0.52
    inner_w = w * 0.34
    box(ax, x + 0.18, inner_y, inner_w, inner_h, "MLA", "", "mla", fs=9.7)
    box(ax, x + w - inner_w - 0.18, inner_y, inner_w, inner_h,
        "MoE-FFN", "", "moe", fs=9.1)
    arrow(ax, (x + 0.18 + inner_w, inner_y + inner_h / 2),
          (x + w - inner_w - 0.18, inner_y + inner_h / 2), lw=1.0)


def save(fig, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    png_dir = output_dir / "PNG格式"
    png_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(png_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def setup(width: float, height: float, title: str = "", subtitle: str = ""):
    # Embed all glyphs as vector paths. VS Code/browser SVG renderers otherwise
    # substitute fonts independently and produce visibly broken Latin spacing.
    plt.rcParams["font.family"] = "sans-serif"
    # Matplotlib registers the installed Noto CJK collection under its JP face;
    # the glyph coverage still includes Simplified Chinese and Latin.
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP"]
    plt.rcParams["svg.fonttype"] = "path"
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis("off")
    if title:
        ax.text(width / 2, height - 0.28, title, ha="center", va="top", fontsize=15, weight="bold")
        ax.text(width / 2, height - 0.68, subtitle, ha="center", va="top", fontsize=9.2, color="#475569")
    return fig, ax


def draw_a(output_dir: Path) -> None:
    # Figure numbers and captions belong in the manuscript. Keeping only the
    # source panel restores the original working graph's typography and spacing.
    fig, ax = setup(15.0, 11.1)
    # Original panel A coordinates shifted vertically into this standalone canvas.
    # Lower the network body slightly so both image-to-network arrows have a
    # visible shaft. The two CT tiles remain locked to the same y coordinates.
    sy = lambda y: y - 14.40
    # The manuscript caption names the method, so the figure itself has no
    # redundant centred title. A compact legend is placed at the right.
    panel(ax, 0.25, 0.35, 14.50, 10.35, "")
    # Width is based on rendered text measurement: the longest A label is
    # ~1.84 data units; 2.20 leaves ~0.18 units of padding on each side.
    enc_x, dec_x, w, h = 1.10, 8.90, 2.20, 0.58

    # Paper-style visual entry/exit. The output is the actual prediction mask for
    # exactly the same case and slice as the CT thumbnail (not a CT overlay or GT).
    reference = mpimg.imread(CT_REFERENCE)
    img_h, img_w = reference.shape[:2]
    input_crop = reference[int(img_h * 0.09):int(img_h * 0.86), int(img_w * 0.07):int(img_w * 0.32)]
    output_mask = prediction_mask_tile()
    thumb_y0, thumb_y1 = 9.30, 10.17
    input_extent = (1.705, 2.695, thumb_y0, thumb_y1)
    output_extent = (9.505, 10.495, thumb_y0, thumb_y1)
    image_tile(ax, input_crop, input_extent, "Input CT", COLORS["feature"][1])
    image_tile(ax, output_mask, output_extent, "Prediction mask", COLORS["output"][1])

    vertical_legend(ax, 12.35, 0.72, 2.10, 3.90)

    enc = [
        (22.72, "Input Projection", "1 → 32", "transition"),
        (21.79, "Encoder Stage 0", "Block ×3; [B,32,128³]", "backbone"),
        (20.86, "DownBlock 0", "32 → 64", "transition"),
        (19.93, "Encoder Stage 1", "Block ×4; [B,64,64³]", "backbone"),
        (19.00, "DownBlock 1", "64 → 128", "transition"),
        (18.07, "Encoder Stage 2", "Block ×8; [B,128,32³]", "backbone"),
        (17.14, "DownBlock 2", "128 → 256", "transition"),
        (16.21, "Encoder Stage 3", "Block ×8; [B,256,16³]", "backbone"),
        (15.28, "DownBlock 3", "256 → 512", "transition"),
    ]
    for y, title, subtitle, kind in enc:
        box(ax, enc_x, sy(y), w, h, title, subtitle, kind, fs=10.1)
    arrow(ax, ((input_extent[0] + input_extent[1]) / 2, input_extent[2]),
          (enc_x + w / 2, sy(enc[0][0]) + h), lw=1.2)
    for first, second in zip(enc[:-1], enc[1:]):
        arrow(ax, (enc_x + w / 2, sy(first[0])), (enc_x + w / 2, sy(second[0]) + h))

    bx, by, bw, bh = 3.55, sy(15.12), 2.20, 0.90
    mx, my, mw, mh = 6.05, sy(15.12), 2.55, 0.90
    box(ax, bx, by, bw, bh, "MedNeXt Bottleneck", "Block ×8; [B,512,8³]", "backbone", fs=10.2)
    mla_moe_pair(ax, mx, my, mw, mh)
    mid_y = by + bh / 2
    arrow(ax, (enc_x + w, mid_y), (bx, mid_y))
    arrow(ax, (bx + bw, mid_y), (mx, mid_y))

    dec = [
        (15.28, "UpBlock 3", "512 → 256", "transition"),
        (16.20, "Decoder Stage 3", "Block ×8; [B,256,16³]", "backbone"),
        (17.12, "UpBlock 2", "256 → 128", "transition"),
        (18.04, "Decoder Stage 2", "Block ×8; [B,128,32³]", "backbone"),
        (18.96, "UpBlock 1", "128 → 64", "transition"),
        (19.88, "Decoder Stage 1", "Block ×4; [B,64,64³]", "backbone"),
        (20.80, "UpBlock 0", "64 → 32", "transition"),
        (21.72, "Decoder Stage 0", "Block ×3; [B,32,128³]", "backbone"),
        (22.64, "Segmentation Head", "32 → 3 classes", "transition"),
    ]
    for y, title, subtitle, kind in dec:
        box(ax, dec_x, sy(y), w, h, title, subtitle, kind, fs=10.0)
    dec_cx = dec_x + w / 2
    for first, second in zip(dec[:-1], dec[1:]):
        arrow(ax, (dec_cx, sy(first[0]) + h), (dec_cx, sy(second[0])))
    arrow(ax, (mx + mw, mid_y), (dec_x, mid_y))
    arrow(ax, (dec_x + w / 2, sy(dec[-1][0]) + h),
          ((output_extent[0] + output_extent[1]) / 2, output_extent[2]), lw=1.2)

    skip_pairs = [
        (21.79, 21.72), (19.93, 19.88),
        (18.07, 18.04), (16.21, 16.20),
    ]
    for src_y, dst_y in skip_pairs:
        y1, y2 = sy(src_y) + h / 2, sy(dst_y) + h / 2
        arrow(ax, (enc_x + w, y1), (dec_x, y2), color=COLORS["skip"], lw=1.35)
    save(fig, output_dir, "A_网络总览")


def draw_b(output_dir: Path) -> None:
    # Panel A already gives every stage repeat count. This panel therefore
    # explains one MedNeXt Block only, avoiding a second competing overview.
    fig, ax = setup(4.20, 7.3)
    panel(ax, 0.20, 0.15, 3.80, 6.95, "")
    # The longest rendered B label is ~1.62 data units. Width 2.10 preserves
    # ~0.24 units of horizontal padding per side without wasting space.
    x, w, h, cx = 1.025, 2.10, 0.50, 2.075
    nodes = [
        (6.28, "Block input", "[B,C,D,H,W]", "transition"),
        (5.40, "Depthwise Conv3d", "k=3; C→C", "backbone"),
        (4.52, "GroupNorm", "batch-independent", "transition"),
        (3.64, "1×1×1 expansion", "C→rC; r=3/4/8", "backbone"),
        (2.76, "GELU", "非线性", "transition"),
        (1.88, "1×1×1 projection", "rC→C", "backbone"),
    ]
    for y, title, subtitle, kind in nodes:
        box(ax, x, y, w, h, title, subtitle, kind, fs=9.8)
    for first, second in zip(nodes[:-1], nodes[1:]):
        arrow(ax, (cx, first[0]), (cx, second[0] + h), lw=1.15)

    add_y = 1.15
    plus(ax, cx, add_y, r=0.15)
    arrow(ax, (cx, nodes[-1][0]), (cx, add_y + 0.15), lw=1.15)
    box(ax, x, 0.22, w, 0.50, "Block output", "", "output", fs=9.8)
    arrow(ax, (cx, add_y - 0.15), (cx, 0.72), lw=1.15)

    # The single bypass is the defining residual connection of this block.
    orth_arrow(ax, [(x, 6.53), (0.45, 6.53), (0.45, add_y), (cx - 0.15, add_y)],
               color=COLORS["backbone"][1], dashed=True, lw=1.2)
    save(fig, output_dir, "B_Stage与Block")


def draw_c(output_dir: Path) -> None:
    fig, ax = setup(7.5, 10.4)
    panel(ax, 0.30, 0.25, 6.90, 9.75, "C. DownBlock / UpBlock")

    def resample_block(y_top, label, main_text, shortcut_text, output_text):
        box(ax, 2.10, y_top, 3.30, 0.62, f"{label} input", "[B,C,D,H,W]", "feature", fs=9.7)
        box(ax, 0.58, y_top - 1.43, 2.95, 0.86, "Main path", main_text, "transition", fs=9.0)
        box(ax, 3.97, y_top - 1.43, 2.95, 0.86, "Resampling shortcut", shortcut_text,
            "transition", fs=8.7)
        plus(ax, 3.75, y_top - 2.13, r=0.16)
        box(ax, 2.10, y_top - 3.02, 3.30, 0.62, f"{label} output", output_text, "output", fs=9.7)
        orth_arrow(ax, [(3.75, y_top), (3.75, y_top - 0.36), (2.05, y_top - 0.36),
                        (2.05, y_top - 0.57)], lw=1.15)
        orth_arrow(ax, [(3.75, y_top), (3.75, y_top - 0.36), (5.45, y_top - 0.36),
                        (5.45, y_top - 0.57)], lw=1.15)
        orth_arrow(ax, [(2.05, y_top - 1.43), (2.05, y_top - 1.86),
                        (3.59, y_top - 1.86), (3.59, y_top - 2.13)], lw=1.15)
        orth_arrow(ax, [(5.45, y_top - 1.43), (5.45, y_top - 1.86),
                        (3.91, y_top - 1.86), (3.91, y_top - 2.13)], lw=1.15)
        arrow(ax, (3.75, y_top - 2.29), (3.75, y_top - 2.40), lw=1.15)

    ax.text(3.75, 9.12, "DownBlock：降采样 + 通道扩展", ha="center", va="center",
            fontsize=10.3, weight="bold", color="#334155")
    resample_block(8.25, "Down", "DW Conv3d, s=2\nNorm → Expand → GELU → Project",
                   "1×1×1 Conv3d\nstride=2", "[B,2C,D/2,H/2,W/2]")

    ax.text(3.75, 4.72, "UpBlock：上采样 + 通道压缩", ha="center", va="center",
            fontsize=10.3, weight="bold", color="#334155")
    resample_block(3.86, "Up", "DW ConvTranspose3d, s=2\nNorm → Expand → GELU → Project",
                   "1×1×1 ConvTranspose3d\nstride=2", "[B,C/2,2D,2H,2W]")

    ax.text(0.58, 0.59,
            "两条路径同步尺度与通道后逐元素相加（不是 concat）；UpBlock 内部用 pad 对齐尺寸。",
            fontsize=8.4, weight="bold", color="#111827", ha="left")
    save(fig, output_dir, "C_Down与Up")


def draw_d_mla(output_dir: Path) -> None:
    """Teacher-facing MLA and MoE internals; standard outer shells are omitted."""
    # Keep x/y scaling equal and make the source lettering deliberately large:
    # the figure is usually inspected in a half-width editor preview before it
    # is placed at 183 mm in the paper.
    fig, ax = setup(12.0, 9.6)
    ax.set_xlim(0, 15.0)
    ax.set_ylim(0, 12.0)

    # MLA: Q remains full-width; K/V share one low-rank latent representation.
    panel(ax, 0.25, 5.35, 14.50, 6.45, "")
    box(ax, 0.55, 11.15, 0.58, 0.42, "A", "", "mla", fs=10.8)
    box(ax, 6.15, 10.70, 2.70, 0.65, r"Input tokens $X$", "[B,N,C]", "transition", fs=11.0)
    box(ax, 2.00, 9.25, 1.85, 0.70, r"$W_{\mathrm{Q}}$", "C → C", "mla", fs=11.0)
    box(ax, 2.00, 8.05, 1.85, 0.65, r"$Q$", "[B,h,N,C/h]", "transition", fs=10.5)
    box(ax, 9.20, 9.25, 1.95, 0.70, r"$W_{\mathrm{DKV}}$", "C → C/4", "mla", fs=11.0)
    box(ax, 9.05, 8.05, 2.25, 0.65, r"$c_{\mathrm{KV}}$", "[B,N,C/4]", "transition", fs=10.3)
    box(ax, 4.40, 6.75, 1.90, 0.70, r"$W_{\mathrm{UK}}$", "C/4 → C", "mla", fs=10.5)
    box(ax, 7.00, 6.75, 1.90, 0.70, r"$W_{\mathrm{UV}}$", "C/4 → C", "mla", fs=10.5)
    box(ax, 2.45, 5.70, 5.90, 0.65, "Attention", "", "backbone", fs=11.0)
    box(ax, 8.95, 5.65, 1.85, 0.75, r"$W_{\mathrm{O}}$", "C → C", "output", fs=10.8)

    # Only the two mathematically necessary splits use bends; all downstream
    # Q/K/V arrows enter attention vertically without decorative staircases.
    orth_arrow(ax, [(7.50, 10.70), (7.50, 10.35), (2.93, 10.35), (2.93, 9.95)], lw=1.10)
    orth_arrow(ax, [(7.50, 10.70), (7.50, 10.35), (10.18, 10.35), (10.18, 9.95)], lw=1.10)
    arrow(ax, (2.93, 9.25), (2.93, 8.70), lw=1.10)
    arrow(ax, (10.18, 9.25), (10.18, 8.70), lw=1.10)
    orth_arrow(ax, [(10.18, 8.05), (10.18, 7.70), (5.35, 7.70), (5.35, 7.45)], lw=1.10)
    orth_arrow(ax, [(10.18, 8.05), (10.18, 7.70), (7.95, 7.70), (7.95, 7.45)], lw=1.10)
    arrow(ax, (2.93, 8.05), (2.93, 6.35), color=COLORS["mla"][1], lw=1.10)
    arrow(ax, (5.35, 6.75), (5.35, 6.35), color=COLORS["mla"][1], lw=1.10)
    arrow(ax, (7.95, 6.75), (7.95, 6.35), color=COLORS["mla"][1], lw=1.10)
    arrow(ax, (8.35, 6.03), (8.95, 6.03), lw=1.10)

    # MoE-FFN: shared expert, router and all four routed experts receive the
    # same input. Top-2 selects/gathers already-computed routed outputs.
    panel(ax, 0.25, 0.15, 14.50, 4.90, "")
    box(ax, 0.55, 4.45, 0.58, 0.42, "B", "", "moe", fs=10.3)
    box(ax, 0.55, 2.35, 1.75, 0.70, "Input tokens", "[BN,C]", "transition", fs=10.5)
    box(ax, 3.00, 3.65, 1.95, 0.70, "Shared expert", "always active", "backbone", fs=10.3)
    box(ax, 3.00, 0.65, 1.55, 0.70, "Router", "", "moe", fs=10.0)

    group_x, group_y, group_w, group_h = 3.00, 2.05, 4.40, 1.00
    ax.add_patch(FancyBboxPatch(
        (group_x, group_y), group_w, group_h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.25, edgecolor=COLORS["backbone"][1], facecolor="#fff9e6",
    ))
    ax.text(group_x + group_w / 2, group_y + group_h - 0.16,
            "4 routed experts", ha="center", va="center",
            fontsize=9.3, weight="bold", color="#111827")
    for i, x in enumerate([3.20, 4.20, 5.20, 6.20], 1):
        box(ax, x, 2.18, 0.82, 0.50, f"E{i}", "", "backbone", fs=9.5)

    box(ax, 4.95, 0.65, 1.60, 0.70, "Routing scores", "", "transition", fs=9.2)
    box(ax, 6.90, 0.65, 1.70, 0.70, "+ balance bias", "", "transition", fs=9.0)
    box(ax, 8.95, 0.65, 1.70, 0.70, "Top-2 selection", "", "moe", fs=9.0)
    box(ax, 8.95, 2.20, 1.80, 0.70, "Gather experts", "", "moe", fs=9.3)
    box(ax, 11.00, 0.65, 2.00, 0.70, "Gate weights", "", "moe", fs=9.5)
    box(ax, 11.05, 2.20, 1.90, 0.70, "Weighted sum", "", "backbone", fs=9.7)
    plus(ax, 12.00, 4.00, r=0.17)
    box(ax, 13.18, 3.65, 1.30, 0.70, "MoE output", "", "output", fs=9.8)

    # Three true input branches; bends are only used for fan-out.
    orth_arrow(ax, [(2.30, 2.70), (2.60, 2.70), (2.60, 4.00), (3.00, 4.00)], lw=1.05)
    arrow(ax, (2.30, 2.70), (3.00, 2.70), lw=1.05)
    orth_arrow(ax, [(2.30, 2.70), (2.60, 2.70), (2.60, 1.00), (3.00, 1.00)], lw=1.05)

    arrow(ax, (4.55, 1.00), (4.95, 1.00), lw=1.0)
    arrow(ax, (6.55, 1.00), (6.90, 1.00), lw=1.0)
    arrow(ax, (8.60, 1.00), (8.95, 1.00), lw=1.0)
    arrow(ax, (7.40, 2.55), (8.95, 2.55), lw=1.0)
    arrow(ax, (9.80, 1.35), (9.80, 2.20), lw=1.0)

    # Gate weights use raw (unbiased) Top-2 scores. The lower bypass is a true
    # dependency, not a decorative route.
    orth_arrow(ax, [(5.75, 0.65), (5.75, 0.35), (12.00, 0.35), (12.00, 0.65)],
               color=COLORS["moe"][1], lw=1.0)
    arrow(ax, (10.65, 1.00), (11.00, 1.00), color=COLORS["moe"][1], lw=1.0)
    arrow(ax, (10.75, 2.55), (11.05, 2.55), lw=1.0)
    arrow(ax, (12.00, 1.35), (12.00, 2.20), color=COLORS["moe"][1], lw=1.0)

    # Shared output and routed weighted sum meet only at the final add.
    arrow(ax, (4.95, 4.00), (11.83, 4.00), lw=1.0)
    arrow(ax, (12.00, 2.90), (12.00, 3.83), lw=1.0)
    arrow(ax, (12.17, 4.00), (13.18, 4.00), lw=1.0)

    save(fig, output_dir, "D_MLA瓶颈与注意力")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--figure", choices=("all", "a", "b", "c", "d"), default="all")
    args = parser.parse_args()
    # C is retained as an optional legacy/teaching diagram, but excluded from
    # the default paper set because A already communicates the standard Down/Up flow.
    default_paper_set = {"a", "b", "d"}
    for selector, draw in (("a", draw_a), ("b", draw_b), ("c", draw_c), ("d", draw_d_mla)):
        if args.figure == selector or (args.figure == "all" and selector in default_paper_set):
            draw(args.output_dir)


if __name__ == "__main__":
    main()
