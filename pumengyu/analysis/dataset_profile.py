"""
数据集画像 —— 对全集 GT + CT 进行分析，输出统计报告和图表。

不依赖任何模型预测，结果直接用于论文数据集描述章节。

用法：
  cd /home/PuMengYu/nnUNet
  python -m pumengyu.analysis.dataset_profile \\
    --dataset Dataset003_Liver \\
    --output_tag lits_full

输出到：pumengyu/notes/实验结果分析/dataset_profile_{tag}/
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["font.family"] = ["Noto Sans CJK JP", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.ticker as mticker

from pumengyu.analysis.image_features import extract_dataset

WORKSPACE = Path("/home/PuMengYu/nnUNet_workspace")
NOTES_DIR = Path(__file__).parent.parent / "notes" / "实验结果分析"

PCTS = [5, 10, 25, 50, 75, 90, 95]   # 统一用这几个百分位


# ──────────────────────────── 统计工具 ───────────────────────────────

def pct_table(arr: list[float], name: str) -> str:
    """生成带百分位的统计行，适合打印。"""
    if not arr:
        return f"  {name}: 无数据"
    a = np.array([x for x in arr if not np.isnan(x)])
    pct_vals = {p: float(np.percentile(a, p)) for p in PCTS}
    line = (f"  {name}:\n"
            f"    n={len(a)}  mean={a.mean():.1f}  std={a.std():.1f}\n"
            f"    min={a.min():.1f}  "
            + "  ".join(f"P{p}={pct_vals[p]:.1f}" for p in PCTS)
            + f"  max={a.max():.1f}")
    return line


# ──────────────────────────── 图表 ───────────────────────────────────

def _pct_vlines(ax, arr: list[float], pcts=(25, 50, 75),
                colors=("#e67e22", "#e74c3c", "#e67e22"), alpha=0.8):
    a = np.array([x for x in arr if not np.isnan(x)])
    styles = ["--", "-", "--"]
    for p, c, ls in zip(pcts, colors, styles):
        v = float(np.percentile(a, p))
        ax.axvline(v, color=c, linestyle=ls, linewidth=1.3, alpha=alpha,
                   label=f"P{p}={v:.0f}")


def plot_size_distribution(cc_sizes: list[int], out_dir: Path) -> None:
    """CC 体积分布：直方图（对数坐标）+ 百分位标注 + 累积分布。"""
    if not cc_sizes:
        print("  [跳过] cc_sizes 为空，请检查 GT/图像路径是否正确")
        return
    arr = np.array(cc_sizes)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("肿瘤 CC 体积分布（全集 GT）", fontsize=13, fontweight="bold")

    # 左：直方图（对数 x 轴）
    ax = axes[0]
    bins = np.logspace(0, np.ceil(np.log10(arr.max())), 60)
    ax.hist(arr, bins=bins, color="#3498db", alpha=0.8, edgecolor="white", linewidth=0.3)
    ax.set_xscale("log")
    _pct_vlines(ax, arr.tolist())
    ax.legend(fontsize=8)
    ax.set_xlabel("CC 体素数（对数）")
    ax.set_ylabel("CC 数量")
    ax.set_title(f"直方图（共 {len(arr)} 个 CC）")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{int(x):,}" if x >= 1 else ""))

    # 右：累积分布
    ax = axes[1]
    sorted_arr = np.sort(arr)
    cdf = np.arange(1, len(arr) + 1) / len(arr) * 100
    ax.plot(sorted_arr, cdf, color="#2c3e50", linewidth=2)
    ax.set_xscale("log")
    # 百分位水平线 + 对应的 x 位置
    for p, color in zip(PCTS, ["#95a5a6"]*len(PCTS)):
        val = float(np.percentile(arr, p))
        ax.axhline(p, color=color, linestyle=":", linewidth=0.8, alpha=0.7)
        ax.axvline(val, color=color, linestyle=":", linewidth=0.8, alpha=0.7)
        ax.text(val * 1.05, p + 1, f"P{p}\n{int(val):,}", fontsize=6.5,
                color="#555555", va="bottom")
    ax.set_xlabel("CC 体素数（对数）")
    ax.set_ylabel("累积百分比 (%)")
    ax.set_title("累积分布（横线=百分位，竖线=对应体素数）")
    ax.set_ylim(0, 103)
    ax.grid(True, alpha=0.25)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{int(x):,}" if x >= 1 else ""))

    plt.tight_layout()
    out = out_dir / "01_cc_size_distribution.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  保存: {out.name}")


def plot_cc_count_per_case(cc_counts: list[int], out_dir: Path) -> None:
    """每个有肿瘤 case 的 CC 数量分布（柱状图 + 百分位）。"""
    arr = np.array(cc_counts)
    bins_edge = np.arange(0.5, arr.max() + 1.5, 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle("每个 Case 的肿瘤 CC 数量（病灶数）", fontsize=13, fontweight="bold")

    # 左：频次柱状图
    ax = axes[0]
    counts, edges = np.histogram(arr, bins=bins_edge)
    centers = (edges[:-1] + edges[1:]) / 2
    ax.bar(centers, counts, width=0.7, color="#8e44ad", alpha=0.8, edgecolor="white")
    ax.set_xlabel("CC 数量（个 case 内病灶数）")
    ax.set_ylabel("case 数")
    ax.set_title(f"n={len(arr)}  mean={arr.mean():.1f}  max={arr.max()}")
    _pct_vlines(ax, arr.tolist(), pcts=(25, 50, 75))
    ax.legend(fontsize=8)

    # 右：分组饼图（1 / 2-5 / 6-15 / >15）
    ax = axes[1]
    groups = {
        "单病灶 (=1)":  (arr == 1).sum(),
        "少量 (2-5)":   ((arr >= 2) & (arr <= 5)).sum(),
        "多发 (6-15)":  ((arr >= 6) & (arr <= 15)).sum(),
        "弥漫 (>15)":   (arr > 15).sum(),
    }
    labels = [f"{k}\nn={v}" for k, v in groups.items() if v > 0]
    values = [v for v in groups.values() if v > 0]
    colors = ["#3498db", "#2ecc71", "#e67e22", "#e74c3c"][:len(values)]
    wedges, texts, autotexts = ax.pie(
        values, labels=labels, colors=colors,
        autopct="%1.0f%%", startangle=90,
        textprops={"fontsize": 9})
    ax.set_title("病灶数量分组")

    plt.tight_layout()
    out = out_dir / "02_cc_count_per_case.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  保存: {out.name}")


def plot_hu_contrast(tumor_hu: list[float], liver_hu: list[float],
                     contrast: list[float], out_dir: Path) -> None:
    """肿瘤 HU vs 肝脏 HU 分布 + 对比度分布（各带百分位）。"""
    t_arr = np.array([x for x in tumor_hu if not np.isnan(x)])
    l_arr = np.array([x for x in liver_hu if not np.isnan(x)])
    c_arr = np.array([x for x in contrast if not np.isnan(x)])

    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    fig.suptitle("HU 分布与对比度分析（全集 GT CC）", fontsize=13, fontweight="bold")

    # 左：肿瘤 vs 肝脏 HU 叠加直方图
    ax = axes[0]
    all_vals = np.concatenate([t_arr, l_arr])
    bins = np.linspace(all_vals.min() - 10, all_vals.max() + 10, 80)
    ax.hist(t_arr, bins=bins, alpha=0.6, color="#e74c3c", label=f"肿瘤 HU (n={len(t_arr)})")
    ax.hist(l_arr, bins=bins, alpha=0.6, color="#3498db", label=f"周围肝脏 HU (n={len(l_arr)})")
    ax.axvline(np.median(t_arr), color="#c0392b", linestyle="--", linewidth=1.5,
               label=f"肿瘤中位={np.median(t_arr):.0f}")
    ax.axvline(np.median(l_arr), color="#1a5276", linestyle="--", linewidth=1.5,
               label=f"肝脏中位={np.median(l_arr):.0f}")
    ax.set_xlabel("HU 值")
    ax.set_ylabel("CC 数量")
    ax.set_title("肿瘤 vs 周围肝脏 HU 分布")
    ax.legend(fontsize=8)

    # 中：对比度分布（直方图）
    ax = axes[1]
    bins_c = np.linspace(c_arr.min() - 5, c_arr.max() + 5, 60)
    ax.hist(c_arr, bins=bins_c, color="#8e44ad", alpha=0.8, edgecolor="white")
    ax.axvline(0, color="black", linewidth=1.2, linestyle="-", label="对比度=0")
    _pct_vlines(ax, c_arr.tolist())
    ax.legend(fontsize=8)
    ax.set_xlabel("对比度（肿瘤HU - 肝脏HU）")
    ax.set_ylabel("CC 数量")
    ax.set_title(f"对比度分布  负值=低密度肿瘤  正值=高密度肿瘤")

    # 右：对比度百分位汇总条形图
    ax = axes[2]
    pct_vals = [float(np.percentile(c_arr, p)) for p in PCTS]
    bar_colors = ["#e74c3c" if v < 0 else "#27ae60" for v in pct_vals]
    bars = ax.barh([f"P{p}" for p in PCTS], pct_vals,
                   color=bar_colors, alpha=0.8, edgecolor="white")
    ax.axvline(0, color="black", linewidth=1.0)
    for bar, val in zip(bars, pct_vals):
        ax.text(val + (2 if val >= 0 else -2), bar.get_y() + bar.get_height() / 2,
                f"{val:.0f}", va="center", ha="left" if val >= 0 else "right", fontsize=8)
    ax.set_xlabel("对比度（HU）")
    ax.set_title("对比度各百分位值")
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    out = out_dir / "03_hu_contrast.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  保存: {out.name}")


def plot_size_contrast_scatter(sizes: list[int], contrasts: list[float],
                               out_dir: Path) -> None:
    """体积 vs 对比度散点图：找出最难分割的区域（小体积 + 低对比度）。"""
    s = np.array(sizes)
    c = np.array([x for x in contrasts])
    valid = ~np.isnan(c)
    s, c = s[valid], c[valid]

    fig, ax = plt.subplots(figsize=(9, 6))
    sc = ax.scatter(s, c, alpha=0.4, s=15, c=c, cmap="RdYlGn",
                    vmin=np.percentile(c, 5), vmax=np.percentile(c, 95))
    plt.colorbar(sc, ax=ax, label="对比度（HU）")
    ax.set_xscale("log")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.axvline(5000, color="gray", linewidth=0.8, linestyle=":", alpha=0.7,
               label="5k 体素（极小阈值）")

    # 标注"最难区域"
    hard_mask = (s < 5000) & (c < np.percentile(c, 25))
    ax.scatter(s[hard_mask], c[hard_mask], alpha=0.7, s=25,
               color="#e74c3c", label=f"最难区域：小体积+低对比度 (n={hard_mask.sum()})")

    ax.set_xlabel("CC 体素数（对数）")
    ax.set_ylabel("对比度（肿瘤HU - 肝脏HU）")
    ax.set_title("体积 vs 对比度散点图\n红点=小体积+低对比度（最难分割区域）")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{int(x):,}" if x >= 1 else ""))

    plt.tight_layout()
    out = out_dir / "04_size_contrast_scatter.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  保存: {out.name}")


# ──────────────────────────── Per-case CSV ───────────────────────────

def _size_cat(vox: int) -> str:
    if vox < 5000:    return "极小(<5k)"
    if vox < 50000:   return "小(5k-50k)"
    if vox < 300000:  return "中等(50k-300k)"
    return "大(>=300k)"


def _size_cat(vox: int) -> str:
    if vox == 0:      return "无肿瘤"
    if vox < 5000:    return "极小(<5k)"
    if vox < 50000:   return "小(5k-50k)"
    if vox < 300000:  return "中等(50k-300k)"
    return "大(>=300k)"


def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    return f"{v:.2f}" if isinstance(v, float) else str(v)


def save_gt_csvs(results: dict, out_dir: Path) -> None:
    """
    per_cc_gt.csv  — 每行一个 CC，完整 HU 百分位（GT 来源）
    per_case_gt.csv — 每行一个 case，汇总信息（GT 来源）
    """
    # ── per_cc_gt.csv ──
    from pumengyu.analysis.image_features import PCTS
    cc_fields = (
        ["case", "cc_idx", "size", "size_cat"]
        + [f"tumor_hu_{s}" for s in ["mean", "std"] + [f"p{p}" for p in PCTS]]
        + [f"liver_hu_{s}" for s in ["mean", "std"] + [f"p{p}" for p in PCTS]]
        + ["contrast"]
    )
    cc_rows = []
    for case, feat in sorted(results.items()):
        if feat is None:
            continue
        for idx, cc in enumerate(feat["cc_list"], 1):
            row = {"case": case, "cc_idx": idx,
                   "size": cc["size"], "size_cat": _size_cat(cc["size"])}
            for f in cc_fields[4:]:
                row[f] = _fmt(cc.get(f, np.nan))
            cc_rows.append(row)

    cc_path = out_dir / "per_cc_gt.csv"
    with open(cc_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cc_fields)
        w.writeheader()
        w.writerows(cc_rows)
    print(f"  per_cc_gt.csv    ({len(cc_rows)} 行)")

    # ── per_case_gt.csv ──
    case_fields = ["case", "has_tumor", "n_cc", "total_voxels", "size_cat",
                   "tumor_hu_mean", "liver_hu_mean", "contrast_mean",
                   "min_cc_size", "max_cc_size", "median_cc_size"]
    case_rows = []
    for case, feat in sorted(results.items()):
        if feat is None:
            case_rows.append({"case": case, "has_tumor": 0, "n_cc": 0,
                               "total_voxels": 0, "size_cat": "无肿瘤",
                               **{f: "" for f in case_fields[5:]}})
            continue
        cc_sizes = [cc["size"] for cc in feat["cc_list"]]
        t_hu = [cc["tumor_hu_mean"] for cc in feat["cc_list"]
                if not np.isnan(cc.get("tumor_hu_mean", np.nan))]
        l_hu = [cc["liver_hu_mean"] for cc in feat["cc_list"]
                if not np.isnan(cc.get("liver_hu_mean", np.nan))]
        c    = [cc["contrast"]      for cc in feat["cc_list"]
                if not np.isnan(cc.get("contrast", np.nan))]
        case_rows.append({
            "case":           case,
            "has_tumor":      1,
            "n_cc":           feat["n_cc"],
            "total_voxels":   feat["total_voxels"],
            "size_cat":       _size_cat(feat["total_voxels"]),
            "tumor_hu_mean":  _fmt(np.mean(t_hu) if t_hu else np.nan),
            "liver_hu_mean":  _fmt(np.mean(l_hu) if l_hu else np.nan),
            "contrast_mean":  _fmt(np.mean(c)    if c    else np.nan),
            "min_cc_size":    min(cc_sizes),
            "max_cc_size":    max(cc_sizes),
            "median_cc_size": int(np.median(cc_sizes)),
        })

    case_path = out_dir / "per_case_gt.csv"
    with open(case_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=case_fields)
        w.writeheader()
        w.writerows(case_rows)
    print(f"  per_case_gt.csv  ({len(case_rows)} 行)")


# ──────────────────────────── 文本报告 ───────────────────────────────

def build_text_report(results: dict, tag: str) -> str:
    lines = []
    W = "=" * 72

    # 收集各维度数据
    cc_sizes, cc_counts = [], []
    tumor_hu, liver_hu, contrast = [], [], []
    notumor_cases = []

    for case, feat in results.items():
        if feat is None:
            notumor_cases.append(case)
            continue
        cc_counts.append(feat["n_cc"])
        for cc in feat["cc_list"]:
            cc_sizes.append(cc["size"])
            tumor_hu.append(cc["tumor_hu_mean"])
            liver_hu.append(cc["liver_hu_mean"])
            contrast.append(cc["contrast"])

    arr_s = np.array(cc_sizes)
    arr_c = np.array([x for x in contrast if not np.isnan(x)])

    lines += [
        W,
        f"数据集画像报告  [{tag}]",
        W, "",
        "【数据集概况】",
        f"  总 case 数    : {len(results)}",
        f"  有肿瘤 case   : {len(results) - len(notumor_cases)}",
        f"  无肿瘤 case   : {len(notumor_cases)}",
        f"  无肿瘤列表    : {notumor_cases}",
        "",
        "【CC（病灶）体积 —— 体素数】",
        pct_table(cc_sizes, "全部 CC 体积"),
        "",
        f"  分组统计：",
        f"    极小 (<5k 体素)     : {(arr_s < 5000).sum():4d} 个  ({(arr_s < 5000).mean()*100:.1f}%)",
        f"    小   (5k-50k 体素)  : {((arr_s >= 5000) & (arr_s < 50000)).sum():4d} 个  ({((arr_s >= 5000) & (arr_s < 50000)).mean()*100:.1f}%)",
        f"    中等 (50k-300k 体素): {((arr_s >= 50000) & (arr_s < 300000)).sum():4d} 个  ({((arr_s >= 50000) & (arr_s < 300000)).mean()*100:.1f}%)",
        f"    大   (>=300k 体素)  : {(arr_s >= 300000).sum():4d} 个  ({(arr_s >= 300000).mean()*100:.1f}%)",
        "",
        "【CC 数量 —— 每个有肿瘤 case 的病灶数】",
        pct_table(cc_counts, "CC 数量"),
        "",
        "【HU 分析】",
        pct_table(tumor_hu, "肿瘤 HU（均值/CC）"),
        pct_table(liver_hu, "周围肝脏 HU（均值/CC）"),
        "",
        "【对比度（肿瘤HU - 肝脏HU）】",
        pct_table(contrast, "对比度"),
        f"  负值（低密度肿瘤）占比 : {(arr_c < 0).sum()} / {len(arr_c)} ({(arr_c < 0).mean()*100:.1f}%)",
        f"  绝对对比度 < 20 HU 占比: {(np.abs(arr_c) < 20).sum()} / {len(arr_c)} ({(np.abs(arr_c) < 20).mean()*100:.1f}%)",
        "",
        W,
    ]
    return "\n".join(lines)


# ──────────────────────────── 主入口 ─────────────────────────────────

def run(dataset: str, tag: str, tumor_label: int = 2) -> None:
    gt_dir  = WORKSPACE / "preprocessed" / dataset / "gt_segmentations"
    img_dir = WORKSPACE / "raw" / dataset / "imagesTr"

    out_dir = NOTES_DIR / f"dataset_profile_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n数据集画像分析：{dataset}")
    print(f"输出目录：{out_dir}\n")

    # 提取特征（最耗时，约数分钟）
    results = extract_dataset(gt_dir, img_dir, tumor_label=tumor_label)

    # 整理数据
    cc_sizes, cc_counts = [], []
    tumor_hu_list, liver_hu_list, contrast_list = [], [], []
    for feat in results.values():
        if feat is None:
            continue
        cc_counts.append(feat["n_cc"])
        for cc in feat["cc_list"]:
            cc_sizes.append(cc["size"])
            tumor_hu_list.append(cc["tumor_hu_mean"])
            liver_hu_list.append(cc["liver_hu_mean"])
            contrast_list.append(cc["contrast"])

    # 保存 JSON（供后续 correlate.py 使用）
    cache = out_dir / "features_cache.json"
    json.dump({k: v for k, v in results.items()}, open(cache, "w"), default=float)
    print(f"特征缓存：{cache}")

    # GT CSV（per-CC + per-case，GT 来源）
    print("\n保存 GT CSV...")
    save_gt_csvs(results, out_dir)

    # 文本报告
    report = build_text_report(results, tag)
    report_path = out_dir / "report.txt"
    report_path.write_text(report, encoding="utf-8")
    print(report)

    # 图表
    print("\n生成图表...")
    plot_size_distribution(cc_sizes, out_dir)
    plot_cc_count_per_case(cc_counts, out_dir)
    plot_hu_contrast(tumor_hu_list, liver_hu_list, contrast_list, out_dir)
    plot_size_contrast_scatter(cc_sizes, contrast_list, out_dir)

    print(f"\n✅ 完成  输出目录: {out_dir}")


def main():
    p = argparse.ArgumentParser(description="数据集画像分析")
    p.add_argument("--dataset",     default="Dataset003_Liver")
    p.add_argument("--output_tag",  default="lits", help="输出目录后缀")
    p.add_argument("--tumor_label", default=2, type=int)
    args = p.parse_args()
    run(args.dataset, args.output_tag, args.tumor_label)


if __name__ == "__main__":
    main()
