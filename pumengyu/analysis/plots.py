"""
所有图表生成函数。

每个函数接收分析结果数据，保存 PNG 到指定目录。
"""
from __future__ import annotations
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["font.family"] = ["Noto Sans CJK JP", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.patches as mpatches


import nibabel as nib
from pathlib import Path as _Path

COLORS = {
    "极小(<5k)":      "#e74c3c",
    "小(5k-50k)":    "#e67e22",
    "中等(50k-300k)": "#2ecc71",
    "大(>=300k)":    "#3498db",
    "无肿瘤":         "#95a5a6",
    "TP":  "#2ecc71",
    "FP_tumor":   "#e67e22",
    "FP_notumor": "#e74c3c",
}


# ──────────────────────────────────────────────────────────────────────
# 1. CC 体积分布直方图（对数坐标）
# ──────────────────────────────────────────────────────────────────────

def plot_cc_distribution(
    gt_cc: list[int],
    tp_cc: list[int],
    fp_tumor_cc: list[int],
    fp_notumor_cc: list[int],
    out_dir: Path,
    title: str = "",
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"CC 体积分布  {title}", fontsize=13)

    bins = np.logspace(0, 7, 60)

    # 左：GT CC 分布
    ax = axes[0]
    ax.hist(gt_cc, bins=bins, color="#3498db", alpha=0.8, edgecolor="white", linewidth=0.3)
    ax.set_xscale("log")
    ax.set_xlabel("CC 体素数（对数坐标）")
    ax.set_ylabel("CC 数量")
    ax.set_title(f"GT 肿瘤 CC 分布（共 {len(gt_cc)} 个）")
    ax.axvline(np.percentile(gt_cc, 50) if gt_cc else 1, color="red",
               linestyle="--", linewidth=1.2, label=f"P50={int(np.percentile(gt_cc,50)) if gt_cc else 0}")
    ax.axvline(5000, color="orange", linestyle=":", linewidth=1.2, label="5k 阈值")
    ax.legend(fontsize=9)

    # 右：预测 CC 分类叠加
    ax = axes[1]
    if tp_cc or fp_tumor_cc or fp_notumor_cc:
        data = [
            (tp_cc,         COLORS["TP"],          f"TP CC ({len(tp_cc)})"),
            (fp_tumor_cc,   COLORS["FP_tumor"],    f"FP-有肿瘤case ({len(fp_tumor_cc)})"),
            (fp_notumor_cc, COLORS["FP_notumor"],  f"FP-无肿瘤case ({len(fp_notumor_cc)})"),
        ]
        for vals, color, label in data:
            if vals:
                ax.hist(vals, bins=bins, alpha=0.65, color=color,
                        edgecolor="white", linewidth=0.3, label=label)
        ax.set_xscale("log")
        ax.set_xlabel("CC 体素数（对数坐标）")
        ax.set_ylabel("CC 数量")
        ax.set_title("预测 CC 分类分布")
        ax.legend(fontsize=9)
        if fp_notumor_cc:
            ax.axvline(max(fp_notumor_cc), color=COLORS["FP_notumor"],
                       linestyle="--", linewidth=1.5,
                       label=f"max FP_notumor={max(fp_notumor_cc)}")
    else:
        ax.text(0.5, 0.5, "无预测数据", ha="center", va="center",
                transform=ax.transAxes, fontsize=14, color="gray")
        ax.set_title("预测 CC 分类分布")

    plt.tight_layout()
    out = out_dir / "01_cc_distribution.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  保存: {out.name}")


# ──────────────────────────────────────────────────────────────────────
# 2. Per-case Tumor Dice 柱状图（按大小着色）
# ──────────────────────────────────────────────────────────────────────

def plot_per_case_dice(records: list[dict], out_dir: Path, title: str = "") -> None:
    # 排序：无肿瘤在右，有肿瘤按 dice 升序
    tumor = sorted([r for r in records if r["size_cat"] != "无肿瘤"],
                   key=lambda r: r["dice_tumor"] or 0)
    notumor = [r for r in records if r["size_cat"] == "无肿瘤"]
    ordered = tumor + notumor

    cases = [r["case"] for r in ordered]
    dices = [(r["dice_tumor"] if r["dice_tumor"] is not None else 0.0)
             if r["size_cat"] != "无肿瘤"
             else (0.0 if (r["n_pred_tumor"] or 0) > 0 else float("nan"))
             for r in ordered]
    colors = [COLORS.get(r["size_cat"], "#888888") for r in ordered]

    fig, ax = plt.subplots(figsize=(max(14, len(cases) * 0.38), 5))
    x = np.arange(len(cases))
    bars = ax.bar(x, [d if not (isinstance(d, float) and math.isnan(d)) else 0
                       for d in dices], color=colors, width=0.7, edgecolor="white", linewidth=0.3)

    # TN（无肿瘤正确）标记
    for i, (r, d) in enumerate(zip(ordered, dices)):
        if r["size_cat"] == "无肿瘤" and isinstance(d, float) and math.isnan(d):
            ax.text(i, 0.02, "TN", ha="center", va="bottom", fontsize=6, color="gray")

    ax.set_xticks(x)
    ax.set_xticklabels(cases, rotation=90, fontsize=6)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Tumor Dice")
    ax.set_title(f"Per-case Tumor Dice  {title}")
    ax.axhline(0.7, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

    legend_patches = [mpatches.Patch(color=COLORS[k], label=k)
                      for k in ["极小(<5k)", "小(5k-50k)", "中等(50k-300k)", "大(>=300k)", "无肿瘤"]]
    ax.legend(handles=legend_patches, loc="upper left", fontsize=8, ncol=2)

    plt.tight_layout()
    out = out_dir / "02_per_case_dice.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  保存: {out.name}")


# ──────────────────────────────────────────────────────────────────────
# 3. 大小分层箱线图
# ──────────────────────────────────────────────────────────────────────

def plot_size_stratified(records: list[dict], out_dir: Path, title: str = "") -> None:
    from pumengyu.analysis.metrics import SIZE_ORDER
    cats = SIZE_ORDER[:-1]  # 排除"无肿瘤"
    data_by_cat = {c: [] for c in cats}
    for r in records:
        if r["size_cat"] in data_by_cat and r["dice_tumor"] is not None:
            d = r["dice_tumor"]
            if not math.isnan(d):
                data_by_cat[r["size_cat"]].append(d)

    fig, ax = plt.subplots(figsize=(9, 5))
    positions = range(len(cats))
    bp_data = [data_by_cat[c] for c in cats]
    box_colors = [COLORS[c] for c in cats]

    bps = ax.boxplot(bp_data, positions=list(positions), patch_artist=True,
                     widths=0.5, showfliers=True,
                     flierprops=dict(marker="o", markersize=4, alpha=0.5))
    for patch, color in zip(bps["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xticks(list(positions))
    labels = [f"{c}\n(n={len(data_by_cat[c])})" for c in cats]
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Tumor Dice")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"大小分层 Dice 分布  {title}")
    ax.axhline(0.7, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

    plt.tight_layout()
    out = out_dir / "03_size_stratified_boxplot.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  保存: {out.name}")


# ──────────────────────────────────────────────────────────────────────
# 4. 阈值敏感性曲线
# ──────────────────────────────────────────────────────────────────────

def plot_threshold_curve(thresh_result: dict, out_dir: Path, title: str = "") -> None:
    T    = thresh_result["thresholds"]
    tp_r = thresh_result["tp_retain"]
    fp_r = thresh_result["fp_remove"]

    if not T:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(T, tp_r, color=COLORS["TP"],          linewidth=2, label="TP 保留率（越高越好）")
    ax.plot(T, fp_r, color=COLORS["FP_notumor"],  linewidth=2, label="无肿瘤FP 清除率（越高越好）")

    # 标注两线交叉点（如果存在）
    tp_arr = np.array(tp_r)
    fp_arr = np.array(fp_r)
    cross = np.where(np.diff(np.sign(tp_arr - fp_arr)))[0]
    for idx in cross:
        t_cross = T[idx]
        ax.axvline(t_cross, color="purple", linestyle=":", linewidth=1.2,
                   label=f"交叉点 T≈{int(t_cross)}")

    ax.set_xscale("log")
    ax.set_xlabel("体积阈值 T（体素，对数坐标）")
    ax.set_ylabel("比率")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"阈值敏感性曲线  {title}")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 标注 gap 区间（若 TP 保留=1 且 FP 清除>0 同时成立的 T 区间）
    safe = [T[i] for i in range(len(T)) if tp_r[i] >= 1.0 and fp_r[i] > 0]
    if safe:
        ax.axvspan(min(safe), max(safe), alpha=0.12, color="green", label="安全阈值区间")

    plt.tight_layout()
    out = out_dir / "04_threshold_curve.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  保存: {out.name}")


# ──────────────────────────────────────────────────────────────────────
# 5. 无肿瘤 case FP 体素数柱状图
# ──────────────────────────────────────────────────────────────────────

def _find_file(directory: Path, stem: str) -> Path | None:
    for suf in (".nii.gz", ".nii"):
        p = directory / (stem + suf)
        if p.exists():
            return p
    return None


def _load_vol(path: Path) -> np.ndarray:
    return np.asarray(nib.load(str(path)).dataobj)


def plot_hard_cases(
    records: list[dict],
    pred_dir: Path,
    gt_dir: Path,
    img_dir: Path,
    out_dir: Path,
    dice_thresh: float = 0.5,
    n_slices: int = 5,
    title: str = "",
) -> None:
    """
    对 Dice < dice_thresh 的困难 case 生成轴向切片对比图。
    每个 case 输出一张 PNG：n_slices 列 × 3 行（CT / GT / Pred）。
    无肿瘤误报 case 也会包含（pred_tumor > 0 且 GT 无肿瘤）。
    """
    hard = []
    for r in records:
        is_fp_notumor = (r["size_cat"] == "无肿瘤" and (r["n_pred_tumor"] or 0) > 0)
        d = r["dice_tumor"]
        is_hard_tumor = (r["size_cat"] != "无肿瘤" and d is not None
                         and not math.isnan(d) and d < dice_thresh)
        if is_fp_notumor or is_hard_tumor:
            hard.append(r)

    if not hard:
        print(f"  无困难 case（阈值 Dice<{dice_thresh}）")
        return

    print(f"  困难 case {len(hard)} 个，生成切片图...")

    for r in hard:
        case = r["case"]
        gt_path   = _find_file(gt_dir,   case)
        pred_path = _find_file(pred_dir,  case)
        img_path  = _find_file(img_dir,   case)

        if gt_path is None or pred_path is None:
            print(f"    [{case}] 文件缺失，跳过")
            continue

        gt_vol   = _load_vol(gt_path).astype(np.int16)
        pred_vol = _load_vol(pred_path).astype(np.int16)
        if img_path is not None:
            img_vol = _load_vol(img_path).astype(np.float32)
            # 窗宽窗位（腹部软组织窗：WL=60, WW=400）
            lo, hi = -140.0, 260.0
            img_vol = np.clip(img_vol, lo, hi)
            img_vol = (img_vol - lo) / (hi - lo)
        else:
            img_vol = None

        # 找有标注或有预测的切片
        gt_mask   = gt_vol   == 2
        pred_mask = pred_vol == 2
        roi_mask  = gt_mask | pred_mask
        slice_ids = np.where(roi_mask.any(axis=(0, 1)))[0]  # axial slices

        if len(slice_ids) == 0:
            slice_ids = np.linspace(0, gt_vol.shape[2] - 1, n_slices, dtype=int)
        else:
            # 均匀选 n_slices 个切片
            idx = np.linspace(0, len(slice_ids) - 1, min(n_slices, len(slice_ids)), dtype=int)
            slice_ids = slice_ids[idx]

        n_cols = len(slice_ids)
        n_rows = 3 if img_vol is not None else 2
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
        if n_cols == 1:
            axes = axes[:, np.newaxis]

        d = r["dice_tumor"]
        label = (f"Dice={d:.3f}" if d is not None and not math.isnan(d)
                 else "无肿瘤误报" if r["size_cat"] == "无肿瘤" else "N/A")
        fig.suptitle(f"{case}  [{r['size_cat']}]  {label}\n{title}", fontsize=10)

        for col, sl in enumerate(slice_ids):
            # CT
            row = 0
            if img_vol is not None:
                axes[row, col].imshow(img_vol[:, :, sl].T, cmap="gray",
                                      origin="lower", vmin=0, vmax=1)
                axes[row, col].set_title(f"CT  z={sl}", fontsize=7)
                axes[row, col].axis("off")
                row += 1

            # GT（绿=肝脏，红=肿瘤）
            gt_rgb = np.zeros((*gt_vol.shape[:2], 3))
            gt_rgb[gt_vol[:, :, sl] == 1] = [0.2, 0.8, 0.2]
            gt_rgb[gt_vol[:, :, sl] == 2] = [1.0, 0.1, 0.1]
            axes[row, col].imshow(gt_rgb.transpose(1, 0, 2), origin="lower")
            axes[row, col].set_title("GT", fontsize=7)
            axes[row, col].axis("off")
            row += 1

            # Pred（绿=肝脏，橙=肿瘤）
            pred_rgb = np.zeros((*pred_vol.shape[:2], 3))
            pred_rgb[pred_vol[:, :, sl] == 1] = [0.2, 0.8, 0.2]
            pred_rgb[pred_vol[:, :, sl] == 2] = [1.0, 0.5, 0.0]
            axes[row, col].imshow(pred_rgb.transpose(1, 0, 2), origin="lower")
            axes[row, col].set_title("Pred", fontsize=7)
            axes[row, col].axis("off")

        plt.tight_layout()
        out = out_dir / f"{case}.png"
        plt.savefig(out, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"    保存: hard_cases/{case}.png")


def plot_notumor_fp(notumor_records: list[dict], out_dir: Path, title: str = "") -> None:
    cases = [r["case"] for r in notumor_records]
    fp_vox = [r["n_pred_tumor"] or 0 for r in notumor_records]

    fig, ax = plt.subplots(figsize=(max(6, len(cases) * 1.2), 4))
    bars = ax.bar(cases, fp_vox,
                  color=[COLORS["FP_notumor"] if v > 0 else COLORS["TP"] for v in fp_vox],
                  edgecolor="white")
    ax.set_ylabel("预测肿瘤体素数（GT=0）")
    ax.set_title(f"无肿瘤 case FP 误报体素数  {title}")
    for bar, v in zip(bars, fp_vox):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(fp_vox) * 0.01,
                str(v), ha="center", va="bottom", fontsize=9)
    ax.axhline(0, color="black", linewidth=0.5)

    plt.tight_layout()
    out = out_dir / "05_notumor_fp.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  保存: {out.name}")
