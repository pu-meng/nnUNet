#!/usr/bin/env python3
"""Draw the code-faithful case -> preprocessing -> loss -> inference worksheet."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


COLORS = {
    "data": ("#eaf4ff", "#2563eb"),
    "process": ("#fff3d6", "#d97706"),
    "model": ("#fef9c3", "#ca8a04"),
    "loss": ("#f3e8ff", "#9333ea"),
    "output": ("#ecfdf5", "#047857"),
    "panel": ("#f8fafc", "#cbd5e1"),
    "arrow": "#263238",
    "note": "#7c2d12",
}


def box(ax, x, y, w, h, title, subtitle="", kind="data", fs=8.2):
    fill, edge = COLORS[kind]
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.10",
        linewidth=1.35, edgecolor=edge, facecolor=fill,
    ))
    ax.text(x + w / 2, y + h * 0.67, title, ha="center", va="center", fontsize=fs, weight="bold")
    if subtitle:
        ax.text(x + w / 2, y + h * 0.27, subtitle, ha="center", va="center", fontsize=max(fs - 1.55, 6.1))


def arrow(ax, start, end, color=None, lw=1.25):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=10, linewidth=lw,
        color=color or COLORS["arrow"],
    ))


def panel(ax, x, y, w, h, title):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.14",
        linewidth=1.1, edgecolor=COLORS["panel"][1], facecolor=COLORS["panel"][0], zorder=-2,
    ))
    ax.text(x + 0.22, y + h - 0.28, title, ha="left", va="center", fontsize=10, weight="bold", color="#334155")


def save(fig, output_dir: Path, export_png: bool = False):
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "model_pipeline.svg", bbox_inches="tight")
    if export_png:
        png_dir = output_dir / "PNG格式"
        png_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(png_dir / "model_pipeline.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def draw_pipeline(output_dir: Path, export_png: bool = False):
    fig, ax = plt.subplots(figsize=(18, 18))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 18)
    ax.axis("off")
    ax.text(9, 17.66, "MedNeXt-MHA-MoE — case、训练损失与推理产物 worksheet", ha="center", fontsize=16, weight="bold")
    ax.text(9, 17.28, "这张图回答：一个 case 如何进入网络、如何产生 loss、预测如何还原为病例级结果", ha="center", fontsize=9.5, color="#475569")

    # A. Case extraction and training input.
    panel(ax, 0.35, 11.45, 17.3, 5.25, "A. Case 被提取后如何进入训练")
    box(ax, 0.75, 14.55, 2.35, 0.78, "原始 case", "NIfTI CT + 可选 seg", "data", fs=9)
    box(ax, 3.55, 14.55, 2.20, 0.78, "固定 case list", "HCC k=3 / LiTS 92", "data", fs=8.5)
    arrow(ax, (3.10, 14.94), (3.55, 14.94))
    box(ax, 6.25, 15.18, 2.45, 0.78, "HCC loader", "Dataset013；k=3", "data", fs=8.6)
    box(ax, 6.25, 13.92, 2.45, 0.78, "LiTS loader", "replay 分支；92 train", "data", fs=8.6)
    arrow(ax, (5.75, 14.94), (6.25, 15.57))
    arrow(ax, (5.75, 14.94), (6.25, 14.31))
    box(ax, 9.25, 14.55, 3.00, 0.78, "DefaultPreprocessor", "transpose → crop nonzero", "process", fs=8.4)
    arrow(ax, (8.70, 15.57), (9.25, 14.94))
    arrow(ax, (8.70, 14.31), (9.25, 14.94))
    ax.text(10.75, 13.98, "随后：normalize → resample → foreground locations", ha="center", fontsize=7.5, color="#475569")
    box(ax, 12.85, 14.55, 2.10, 0.78, "Patch sampler", "[B,1,128³] + target", "process", fs=8.2)
    box(ax, 15.35, 14.55, 1.85, 0.78, "Train aug", "Spatial / noise / gamma", "process", fs=7.5)
    arrow(ax, (12.25, 14.94), (12.85, 14.94)); arrow(ax, (14.95, 14.94), (15.35, 14.94))
    ax.text(0.78, 12.48, "实际训练增强（按 nnUNet get_training_transforms）：SpatialTransform、GaussianNoise、GaussianBlur、亮度/对比度、低分辨率模拟、Gamma、Mirror。", fontsize=7.7, color="#475569")
    ax.text(0.78, 12.08, "HCC-only：每次更新使用 HCC batch；replay：HCC 与 LiTS 各自 forward/loss 后合成一次 optimizer step。", fontsize=7.7, color=COLORS["note"])

    # B. Forward and loss.
    panel(ax, 0.35, 6.05, 17.3, 4.85, "B. Forward、深监督与关键损失函数")
    box(ax, 0.75, 8.18, 2.20, 0.78, "训练 patch", "image + target", "data", fs=8.8)
    box(ax, 3.45, 8.18, 3.00, 0.78, "MedNeXt-L + MHA + MoE", "Encoder → bottleneck → Decoder", "model", fs=8.5)
    box(ax, 6.95, 8.18, 2.45, 0.78, "Deep supervision", "out_0 … out_4", "model", fs=8.5)
    box(ax, 9.85, 8.18, 3.05, 0.78, "DC_and_CE_loss", "L = L_softDice + L_CE", "loss", fs=8.7)
    box(ax, 13.35, 8.18, 2.15, 0.78, "AMP backward", "grad + MoE state", "process", fs=8.2)
    box(ax, 15.95, 8.18, 1.05, 0.78, "step", "update", "output", fs=7.8)
    for x1, x2 in [(2.95, 3.45), (6.45, 6.95), (9.40, 9.85), (12.90, 13.35), (15.50, 15.95)]:
        arrow(ax, (x1, 8.57), (x2, 8.57))
    ax.text(1.0, 7.30, "Dice 项：softmax logits 后计算 MemoryEfficientSoftDiceLoss", fontsize=7.9, color="#475569")
    ax.text(1.0, 6.88, "CE 项：RobustCrossEntropyLoss；两项当前权重均为 1", fontsize=7.9, color="#475569")
    ax.text(8.95, 7.30, "深监督：各输出按 [1, 1/2, 1/4, …] 归一化加权；最低分辨率输出在 DDP 下可为 1e-6。", fontsize=7.9, color="#475569")
    ax.text(8.95, 6.88, "HCC-only：L = L_HCC；replay：L_step = L_HCC + L_LiTS。", fontsize=7.9, color=COLORS["note"])
    ax.text(1.0, 6.43, "随后执行：deferred MoE expert-bias update → gradient clipping → optimizer.step。", fontsize=7.7, color="#475569")

    # C. Inference and export.
    panel(ax, 0.35, 0.25, 17.3, 5.25, "C. 测试时如何从 case 还原为病例级预测与完整产物")
    infer = [
        (0.75, 1.85, "Test case", "NIfTI CT", "data"),
        (2.95, 2.20, "Same preprocessor", "transpose/crop/normalize/resample", "process"),
        (5.50, 2.05, "Pad + sliding window", "patch_size / step", "process"),
        (7.90, 2.15, "Predictor", "Gaussian fusion + optional mirror", "model"),
        (10.40, 1.85, "Logits", "aggregated volume", "loss"),
        (12.60, 2.30, "Export prediction", "undo pad → resample → label", "process"),
        (15.25, 1.75, "NIfTI seg", "case prediction", "output"),
    ]
    for x, w, title, sub, kind in infer:
        box(ax, x, 3.55, w, 0.78, title, sub, kind, fs=7.8 if w < 2 else 8.0)
    for (x1, w1, *_), (x2, *_rest) in zip(infer[:-1], infer[1:]):
        arrow(ax, (x1 + w1, 3.94), (x2, 3.94))
    ax.text(1.0, 2.48, "Export 的实际逆序：resample → restore crop → transpose；LabelManager 再把 logits 转成分割标签。", fontsize=7.9, color="#475569")
    ax.text(1.0, 2.05, "正式评估还要保留：预测病例清单、summary.json、可读 txt 报告、test_viz/*.png，以及 checkpoint / dataset / fold provenance。", fontsize=7.9, color=COLORS["note"])
    ax.text(1.0, 1.45, "这张流程图与结构图的关系：结构图解释网络内部；本图解释数据和监督信号如何穿过网络并成为可审计结果。", fontsize=8.0, color="#334155")

    save(fig, output_dir, export_png=export_png)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("pumengyu/notes/模型结构图"))
    parser.add_argument("--png", action="store_true", help="Also export 300 DPI PNG for analysis or papers")
    args = parser.parse_args()
    draw_pipeline(args.output_dir, export_png=args.png)
    print("model pipeline sheet generated")


if __name__ == "__main__":
    main()
