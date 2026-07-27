from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/pumengyu_matplotlib_cache")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "figures"

COLORS = {
    "feature": ("#eaf4ff", "#2563eb"),
    "token": ("#eef2ff", "#4f46e5"),
    "latent": ("#fff3d6", "#d97706"),
    "block": ("#fef9c3", "#ca8a04"),
    "moe": ("#f3e8ff", "#9333ea"),
    "expert": ("#fae8ff", "#c026d3"),
    "output": ("#ecfdf5", "#047857"),
    "arrow": "#263238",
    "soft": "#6b7280",
}


def box(ax, x, y, w, h, title, subtitle="", note="", kind="feature", fs=12):
    fill, edge = COLORS[kind]
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.11",
        linewidth=1.8, edgecolor=edge, facecolor=fill,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h * 0.68, title, ha="center", va="center", fontsize=fs, weight="bold")
    if subtitle:
        ax.text(x + w / 2, y + h * 0.42, subtitle, ha="center", va="center", fontsize=fs - 2)
    if note:
        ax.text(x + w / 2, y + h * 0.18, note, ha="center", va="center", fontsize=fs - 3, color="#4b5563")


def arrow(ax, start, end, *, color=None, curved=0.0, dashed=False):
    patch = FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=14, linewidth=1.8,
        color=color or COLORS["arrow"],
        linestyle=(0, (4, 4)) if dashed else "solid",
        connectionstyle=f"arc3,rad={curved}",
    )
    ax.add_patch(patch)


def save(fig, stem: str) -> None:
    fig.savefig(FIGURE_DIR / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(FIGURE_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def draw_wrapper() -> None:
    # 竖版单栏布局：在双栏论文中无需跨栏，缩放后仍能保持文字可读。
    fig, ax = plt.subplots(figsize=(4.8, 7.8))
    ax.set_xlim(0, 4.8)
    ax.set_ylim(0, 7.8)
    ax.axis("off")
    ax.text(2.4, 7.52, "MLABottleneck3D", ha="center", va="center", fontsize=16, weight="bold")

    box(ax, 0.65, 6.20, 3.50, 0.82, "MedNeXt bottleneck feature", r"$(B, 512, D_b, H_b, W_b)$", kind="feature", fs=11)
    box(ax, 0.50, 4.82, 3.80, 0.96, "Flatten spatial dims + transpose", r"$(B, 512, D_b, H_b, W_b) \rightarrow (B, N, 512)$", r"$N=D_bH_bW_b$; one spatial position per token", kind="token", fs=10)
    box(ax, 0.38, 3.00, 4.04, 1.28, "MLA + MoE Block × 2", "Pre-LN attention + Pre-LN MoE", "two residual sublayers in each block", kind="block", fs=11)
    box(ax, 1.05, 1.95, 2.70, 0.62, "Final LayerNorm", "C = 512", kind="token", fs=10)
    box(ax, 0.50, 0.52, 3.80, 0.98, "Transpose + reshape", r"$(B, N, 512) \rightarrow (B, 512, D_b, H_b, W_b)$", "enhanced 3D feature", kind="output", fs=10)

    for start, end in [
        ((2.40, 6.20), (2.40, 5.78)),
        ((2.40, 4.82), (2.40, 4.28)),
        ((2.40, 3.00), (2.40, 2.57)),
        ((2.40, 1.95), (2.40, 1.50)),
    ]:
        arrow(ax, start, end)
    save(fig, "mla_bottleneck3d")


def draw_attention() -> None:
    fig, ax = plt.subplots(figsize=(10.8, 5.2))
    ax.set_xlim(0, 10.8)
    ax.set_ylim(0, 5.2)
    ax.axis("off")
    ax.text(5.4, 4.88, "Low-rank Multi-head Latent Attention", ha="center", va="center", fontsize=17, weight="bold")

    box(ax, 0.35, 2.00, 1.45, 1.05, "Tokens x", "(B, N, 512)", kind="token", fs=11)
    box(ax, 2.45, 3.35, 1.55, 0.95, "Query", "Q = x W_Q", "512 -> 512", kind="token", fs=11)
    box(ax, 2.30, 0.75, 1.85, 1.12, "KV latent", "c_KV = LN(x W_DKV)", "512 -> 128", kind="latent", fs=11)
    box(ax, 4.75, 0.75, 1.75, 1.12, "Recover K, V", "c_KV W_UK / W_UV", "128 -> 512 each", kind="latent", fs=11)
    box(ax, 4.70, 3.10, 2.25, 1.20, "8-head attention", "softmax(QK^T / sqrt(d))V", "full N x N interaction", kind="block", fs=11)
    box(ax, 7.50, 3.10, 1.30, 1.20, "W_O", "512 -> 512", kind="output", fs=11)
    box(ax, 9.25, 3.10, 1.20, 1.20, "Output", "(B, N, 512)", kind="output", fs=11)

    arrow(ax, (1.80, 2.53), (2.45, 3.82), curved=0.10)
    arrow(ax, (1.80, 2.53), (2.30, 1.31), curved=-0.10)
    arrow(ax, (4.15, 1.31), (4.75, 1.31))
    arrow(ax, (4.00, 3.82), (4.70, 3.78))
    arrow(ax, (5.63, 1.87), (5.80, 3.10), curved=-0.12)
    arrow(ax, (6.95, 3.70), (7.50, 3.70))
    arrow(ax, (8.80, 3.70), (9.25, 3.70))
    ax.text(5.40, 0.28, "Q remains full-dimensional; K and V share a 128-dimensional latent representation.", ha="center", va="center", fontsize=10, color="#4b5563")
    save(fig, "mla_low_rank_attention")


def draw_moe() -> None:
    fig, ax = plt.subplots(figsize=(11.6, 5.4))
    ax.set_xlim(0, 11.6)
    ax.set_ylim(0, 5.4)
    ax.axis("off")
    ax.text(5.8, 4.92, "MoE Feed-forward Network", ha="center", va="center", fontsize=17, weight="bold")

    box(ax, 0.25, 2.15, 1.40, 1.05, "Tokens x", "(B, N, 512)", kind="token", fs=11)
    box(ax, 2.05, 3.70, 1.80, 1.00, "Shared expert", "always active", "512 -> 1024 -> 512", kind="moe", fs=11)
    box(ax, 2.05, 0.55, 1.80, 1.12, "Router", "scores + balance bias", "512 -> 4; select top-2", kind="moe", fs=11)

    expert_ys = [0.45, 1.45, 2.45, 3.45]
    for i, y in enumerate(expert_ys, start=1):
        box(ax, 4.65, y, 1.05, 0.78, f"Expert {i}", "FFN", kind="expert", fs=9)

    box(ax, 6.45, 1.72, 1.85, 1.12, "Routed mixture", "top-2 weighted sum", kind="moe", fs=11)
    box(ax, 8.90, 2.78, 1.45, 1.10, "Add", "shared + routed", kind="output", fs=11)
    box(ax, 10.75, 2.78, 0.60, 1.10, "Out", "512", kind="output", fs=10)

    arrow(ax, (1.65, 2.68), (2.05, 4.20), curved=0.10)
    arrow(ax, (1.65, 2.68), (2.05, 1.11), curved=-0.10)
    for i, y in enumerate(expert_ys):
        arrow(ax, (3.85, 1.11), (4.65, y + 0.39), color=COLORS["soft"], dashed=True, curved=(i - 1.5) * 0.07)
        arrow(ax, (5.70, y + 0.39), (6.45, 2.28), color=COLORS["soft"], curved=(1.5 - i) * 0.07)
    arrow(ax, (3.85, 4.20), (8.90, 3.48), curved=-0.18)
    arrow(ax, (8.30, 2.28), (8.90, 3.18), curved=-0.08)
    arrow(ax, (10.35, 3.33), (10.75, 3.33))
    ax.text(6.30, 0.14, "Each expert: 512 -> 1024 -> 512; the router selects top-2 outputs per token.", ha="center", va="center", fontsize=9, color="#4b5563")
    save(fig, "moe_ffn")


def draw(selected: str = "all") -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    if selected in {"all", "wrapper"}:
        draw_wrapper()
    if selected in {"all", "attention"}:
        draw_attention()
    if selected in {"all", "moe"}:
        draw_moe()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate MedNeXt_MLA paper figures.")
    parser.add_argument(
        "--figure",
        choices=("all", "wrapper", "attention", "moe"),
        default="all",
        help="Generate all figures or only one selected figure.",
    )
    draw(parser.parse_args().figure)
