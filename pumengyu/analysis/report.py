"""
文本报告生成。
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from pumengyu.analysis.cc import pct_str, size_groups


def build_report(
    trainer: str,
    fold: int,
    split: str,
    gt_result: dict,
    pred_result: dict | None,
    metrics_records: list[dict] | None,
    agg: dict | None,
    sep: dict | None,
    thresh_result: dict | None,
) -> str:
    lines = []
    W = "=" * 72

    lines += [
        W,
        f"实验分析报告",
        f"  Trainer  : {trainer}",
        f"  Fold     : {fold}",
        f"  Split    : {split}",
        W, "",
    ]

    # ── 数据集概况 ──
    lines += [
        "【数据集概况】",
        f"  总 case 数    : {gt_result['cases_total']}",
        f"  有肿瘤 case   : {gt_result['cases_tumor']}",
        f"  无肿瘤 case   : {gt_result['cases_notumor']}",
        f"  无肿瘤列表    : {gt_result['notumor_list']}",
        "",
    ]

    # ── GT CC 分布 ──
    gt_cc = gt_result["gt_cc_sizes"]
    grp = size_groups(gt_cc)
    total = len(gt_cc)
    lines += [
        "【GT 肿瘤 CC 分布（全集）】",
        f"  CC 总数     : {total}",
        f"  分位数（体素）: {pct_str(gt_cc)}",
        f"  每 case CC 数: min={min(gt_result['per_case_cc_count'].values()) if gt_result['per_case_cc_count'] else 0}"
        f"  mean={np.mean(list(gt_result['per_case_cc_count'].values())):.1f}"
        f"  max={max(gt_result['per_case_cc_count'].values()) if gt_result['per_case_cc_count'] else 0}",
        "",
        "  大小分组：",
    ]
    for k, v in grp.items():
        pct = v / total * 100 if total else 0
        lines.append(f"    {k:16s}: {v:4d} 个  ({pct:.1f}%)")
    lines.append("")

    # ── 预测 CC 分析 ──
    if pred_result:
        tp   = pred_result["tp_cc"]
        fp_t = pred_result["fp_in_tumor_cc"]
        fp_n = pred_result["fp_in_notumor_cc"]
        lines += [
            "【预测 CC 分类分析】",
            f"  覆盖 case 数: {gt_result['cases_total'] - len(pred_result['missing'])} / {gt_result['cases_total']}",
            f"  （未覆盖 {len(pred_result['missing'])} 个 case，仅有预测文件的 case 计入）",
            "",
            f"  TP CC（预测正确）    : {len(tp):4d} 个    {pct_str(tp)}",
            f"  FP CC（有肿瘤case）  : {len(fp_t):4d} 个    {pct_str(fp_t)}",
            f"  FP CC（无肿瘤case）  : {len(fp_n):4d} 个    {pct_str(fp_n)}",
            "",
            "  无肿瘤 case 逐 case 明细：",
        ]
        for case, sizes in sorted(pred_result["notumor_fp_detail"].items()):
            if sizes:
                lines.append(f"    {case}: {len(sizes)} 个假CC，体素数={sorted(sizes)}")
            else:
                lines.append(f"    {case}: 无误报 ✅")
        lines.append("")

    # ── Separability Gap ──
    if sep:
        lines += [
            "【可分离性指标（Separability Gap）】",
            f"  min GT CC 体素数     : {sep['min_tp_cc']}",
            f"  max 无肿瘤 FP CC     : {sep['max_fp_notumor_cc']}",
            f"  gap                  : {sep['gap']}",
        ]
        if sep["feasible"] is True:
            lines.append("  结论: ✅ 体积阈值理论可行")
        elif sep["feasible"] is False:
            lines.append("  结论: ❌ 体积阈值理论失效（TP/FP 体积范围重叠，无解）")
        else:
            lines.append("  结论: — 数据不足，无法判断")
        lines.append("")

    # ── Metrics 摘要 ──
    if agg:
        lines += [
            "【模型性能摘要（来自 summary.json）】",
            f"  Overall Dice      : {agg['overall']:.4f}",
            f"  有肿瘤 case 数    : {agg['n_tumor_cases']}",
            f"  无肿瘤 case 数    : {agg['n_notumor_cases']}",
            f"  无肿瘤误报        : {agg['n_fp_notumor']} / {agg['n_notumor_cases']}"
            f"  ({agg['notumor_fp_rate']*100:.1f}%)",
            "",
            "  大小分层 Dice：",
        ]
        for cat, s in agg["size_stats"].items():
            if s["n"] > 0:
                lines.append(f"    {cat:16s}: n={s['n']:2d}  mean={s['mean']:.4f}  std={s['std']:.4f}")
        lines.append("")

    lines.append(W)
    return "\n".join(lines)
