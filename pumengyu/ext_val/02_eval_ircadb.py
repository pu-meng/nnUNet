"""
3D-IRCADb 外部验证评估脚本
用法：python 02_eval_ircadb.py

读取各方法的 ensemble 预测，与 GT label 对比：
  - 有肿瘤 case：计算 liver Dice + tumor Dice + Recall + Precision
  - 无肿瘤 case：计算 FP（是否误报肿瘤）
  - 综合 Dice（与 LiTS 评估口径一致）
  - 按肿瘤大小分组统计
  - Per-case 分级（严重失败/需要改进/没问题）
"""

import json
import numpy as np
import nibabel as nib
from pathlib import Path

PRED_ROOT  = Path("/home/PuMengYu/nnUNet_workspace/external_val/ircadb_full/predictions")
LABEL_DIR  = Path("/home/PuMengYu/nnUNet_workspace/external_val/ircadb_full/labels")
INFO_FILE  = Path("/home/PuMengYu/nnUNet_workspace/external_val/ircadb_full/case_info.json")
OUT_FILE   = PRED_ROOT / "ircadb_eval_report.txt"


def dice_score(pred, gt, label):
    p = (pred == label).astype(np.float32)
    g = (gt   == label).astype(np.float32)
    inter = (p * g).sum()
    denom = p.sum() + g.sum()
    return float(2 * inter / denom) if denom > 0 else 1.0


def size_cat(voxels):
    if voxels < 5000:
        return "极小(<5k)"
    elif voxels < 50000:
        return "小(5k-50k)"
    elif voxels < 300000:
        return "中等(50k-300k)"
    else:
        return "大(>=300k)"


def load_nii(path):
    return np.asarray(nib.load(str(path)).dataobj)


def eval_one_method(pred_dir: Path, cases: list, case_info: dict) -> dict:
    results = {}
    for case_id in cases:
        pred_file = pred_dir / f"{case_id}.nii.gz"
        gt_file   = LABEL_DIR / f"{case_id}.nii.gz"
        if not pred_file.exists() or not gt_file.exists():
            results[case_id] = None
            continue

        pred = load_nii(pred_file)
        gt   = load_nii(gt_file)

        has_tumor        = case_info[case_id]["has_tumor"]
        gt_tumor_vox     = int(np.sum(gt == 2))
        gt_liver_vox     = int(np.sum(gt >= 1))   # 全肝（含肿瘤区域）
        pred_tumor_vox   = int(np.sum(pred == 2))

        liver_dice = dice_score(pred, gt, 1)

        if has_tumor:
            tumor_dice = dice_score(pred, gt, 2)
            tp = int(np.sum((pred == 2) & (gt == 2)))
            recall    = tp / gt_tumor_vox  if gt_tumor_vox  > 0 else 0.0
            precision = tp / pred_tumor_vox if pred_tumor_vox > 0 else (1.0 if gt_tumor_vox == 0 else 0.0)
            fnr = 1.0 - recall
            fdr = 1.0 - precision
            is_fp = None
        else:
            tumor_dice = None
            recall = precision = fnr = fdr = None
            is_fp = (pred_tumor_vox > 0)

        results[case_id] = {
            "has_tumor":        has_tumor,
            "liver_dice":       liver_dice,
            "tumor_dice":       tumor_dice,
            "recall":           recall,
            "precision":        precision,
            "fnr":              fnr,
            "fdr":              fdr,
            "gt_tumor_voxels":  gt_tumor_vox,
            "gt_liver_voxels":  gt_liver_vox,
            "pred_tumor_voxels": pred_tumor_vox,
            "is_fp":            is_fp,
            "size_cat":         size_cat(gt_tumor_vox) if has_tumor else None,
        }
    return results


def comprehensive_dice(results: dict) -> float:
    scores = []
    for r in results.values():
        if r is None:
            continue
        if r["has_tumor"]:
            scores.append(r["tumor_dice"])
        else:
            scores.append(0.0 if r["is_fp"] else 1.0)
    return float(np.mean(scores)) if scores else float("nan")


def print_method_detail(method_name, pred_label, results, no_tumor_cases, out):
    """输出单个方法的详细分析，格式对齐 report_custom.txt"""
    valid = {k: v for k, v in results.items() if v is not None}
    tumor_cases_r  = [k for k, v in valid.items() if v["has_tumor"]]
    no_tumor_cases_r = [k for k, v in valid.items() if not v["has_tumor"]]

    out(f"\n{'=' * 80}")
    out(f"方法: {method_name}  {pred_label}")
    out(f"{'=' * 80}")

    # ── 肝脏 Dice ──────────────────────────────────────────────────
    liver_dices = [v["liver_dice"] for v in valid.values()]
    out(f"\nLiver")
    out(f"  Dice: mean={np.mean(liver_dices):.4f}  std={np.std(liver_dices):.4f}")

    # ── 无肿瘤 FP 分析 ──────────────────────────────────────────────
    fp_cases = [k for k in no_tumor_cases_r if valid[k]["is_fp"]]
    n_nt = len(no_tumor_cases_r)
    out(f"\nTumor (无肿瘤 case, n={n_nt})")
    if n_nt > 0:
        fp_voxels = [valid[k]["pred_tumor_voxels"] for k in fp_cases]
        fp_rate = len(fp_cases) / n_nt * 100
        out(f"  误报率(预测出肿瘤但GT无肿瘤): {fp_rate:.2f}%  ({len(fp_cases)}/{n_nt} cases)")
        if fp_voxels:
            all_fp_vox = [valid[k]["pred_tumor_voxels"] for k in no_tumor_cases_r]
            out(f"  FP pred_tumor : mean={np.mean(all_fp_vox):.1f}  std={np.std(all_fp_vox):.1f}")
            out(f"  误报 cases:")
            for k in fp_cases:
                out(f"    {k:<22} pred_tumor={valid[k]['pred_tumor_voxels']:,}")
        out(f"\n  无肿瘤 case 列表:")
        for k in sorted(no_tumor_cases_r):
            tag = "  [误报]" if valid[k]["is_fp"] else ""
            out(f"    {k:<22} liver_dice={valid[k]['liver_dice']:.4f}  pred_tumor={valid[k]['pred_tumor_voxels']:,}{tag}")

    # ── 综合 tumor 指标 ────────────────────────────────────────────
    n_all = len(valid)
    comp = comprehensive_dice(valid)

    t_vals = valid  # shorthand

    # Dice for all cases (综合口径)
    comp_scores = []
    for k, v in valid.items():
        if v["has_tumor"]:
            comp_scores.append(v["tumor_dice"])
        else:
            comp_scores.append(0.0 if v["is_fp"] else 1.0)
    jaccard_scores = [d / (2 - d) if (2 - d) > 0 else 0.0 for d in comp_scores]

    recall_vals    = [v["recall"]    for v in valid.values() if v["has_tumor"]]
    fnr_vals       = [v["fnr"]       for v in valid.values() if v["has_tumor"]]
    # precision: has_tumor cases + FP no_tumor cases (FP -> precision=0)
    prec_cases = [(v["precision"] if v["has_tumor"] else (0.0 if v["is_fp"] else None))
                  for v in valid.values()]
    prec_vals  = [p for p in prec_cases if p is not None]
    fdr_cases  = [(v["fdr"] if v["has_tumor"] else (1.0 if v["is_fp"] else None))
                  for v in valid.values()]
    fdr_vals   = [f for f in fdr_cases if f is not None]

    n_tumor    = len(tumor_cases_r)
    n_nt_ok    = len([k for k in no_tumor_cases_r if not valid[k]["is_fp"]])
    n_nt_fp    = len(fp_cases)
    mean_tumor_dice = np.mean([valid[k]["tumor_dice"] for k in tumor_cases_r]) if tumor_cases_r else float("nan")

    out(f"\nTumor 综合指标（共 {n_all} cases）")
    out(f"  无肿瘤正确(pred=0)→Dice=1(约定)，无肿瘤误报(pred>0)→Dice=0(数学精确)")
    out(f"  Dice        : mean={np.mean(comp_scores):.4f}  std={np.std(comp_scores):.4f}  (全 {n_all} cases)")
    out(f"  Jaccard     : mean={np.mean(jaccard_scores):.4f}  std={np.std(jaccard_scores):.4f}  (全 {n_all} cases，同 Dice 约定)")
    if recall_vals:
        out(f"  Recall      : mean={np.mean(recall_vals):.4f}  std={np.std(recall_vals):.4f}  (有肿瘤 n={n_tumor}，无肿瘤无GT故不计入)")
        out(f"  FNR         : mean={np.mean(fnr_vals):.4f}  std={np.std(fnr_vals):.4f}  (有肿瘤 n={n_tumor})")
    if prec_vals:
        out(f"  Precision   : mean={np.mean(prec_vals):.4f}  std={np.std(prec_vals):.4f}  (有肿瘤 n={n_tumor} + 无肿瘤误报 n={n_nt_fp}，误报计0)")
        out(f"  FDR         : mean={np.mean(fdr_vals):.4f}  std={np.std(fdr_vals):.4f}  (有肿瘤 n={n_tumor} + 无肿瘤误报 n={n_nt_fp}，误报计1)")
    out(f"  构成        : 有肿瘤 n={n_tumor} mean={mean_tumor_dice:.4f}  |  无肿瘤正确(Dice=1) n={n_nt_ok}  |  无肿瘤误报(Dice=0) n={n_nt_fp}")

    # ── 按大小分组 ─────────────────────────────────────────────────
    cats = ["极小(<5k)", "小(5k-50k)", "中等(50k-300k)", "大(>=300k)"]
    out(f"\nTumor Dice 按大小分组 (有肿瘤 case)")
    out(f"  {'大小分类':<20} {'n':>4}   {'Dice mean':>9}   {'Dice std':>8}    {'Recall':>7}   {'Precision':>9}")
    out(f"  {'-'*68}")
    for cat in cats:
        grp = [v for v in valid.values() if v["has_tumor"] and v["size_cat"] == cat]
        if not grp:
            continue
        d  = [v["tumor_dice"] for v in grp]
        rc = [v["recall"]     for v in grp]
        pr = [v["precision"]  for v in grp]
        out(f"  {cat:<20} {len(grp):>4}      {np.mean(d):>7.4f}     {np.std(d):>7.4f}    {np.mean(rc):>7.4f}   {np.mean(pr):>9.4f}")

    # ── Per-case 分级 ──────────────────────────────────────────────
    out(f"\n{'=' * 84}")
    out(f"Per-Case 分级（按 tumor_dice 从低到高）")
    out(f"{'=' * 84}")

    hdr = f"  {'case':<20} {'tumor_dice':>10}   {'recall':>7}  {'precision':>9}     {'FDR':>5}   {'pred_tumor':>10}   {'gt_tumor':>9}   {'gt_liver':>9}  size_cat"
    sep = "-" * 100

    tiers = [
        ("[严重失败] tumor_dice < 0.3", lambda d: d < 0.3),
        ("[需要改进] 0.3 <= tumor_dice < 0.7", lambda d: 0.3 <= d < 0.7),
        ("[没问题]   tumor_dice >= 0.7", lambda d: d >= 0.7),
    ]

    sorted_tumor = sorted(
        [(k, valid[k]) for k in tumor_cases_r],
        key=lambda x: x[1]["tumor_dice"]
    )
    for tier_label, tier_fn in tiers:
        tier_cases = [(k, v) for k, v in sorted_tumor if tier_fn(v["tumor_dice"])]
        out(f"\n{tier_label}  (n={len(tier_cases)})")
        out(sep)
        out(hdr)
        out(sep)
        for k, v in tier_cases:
            out(f"  {k:<20} {v['tumor_dice']:>10.4f}   {v['recall']:>7.4f}  {v['precision']:>9.4f}     {v['fdr']:>5.4f}   "
                f"{v['pred_tumor_voxels']:>10,}   {v['gt_tumor_voxels']:>9,}   {v['gt_liver_voxels']:>9,}  {v['size_cat']}")


def main():
    if not INFO_FILE.exists():
        print(f"[ERROR] 未找到 {INFO_FILE}，请先运行 python prepare_ircadb.py")
        return

    case_info = json.load(open(INFO_FILE))
    all_cases = sorted(case_info.keys())
    tumor_cases    = [c for c in all_cases if case_info[c]["has_tumor"]]
    no_tumor_cases = [c for c in all_cases if not case_info[c]["has_tumor"]]

    method_dirs = sorted([
        d for d in PRED_ROOT.iterdir()
        if d.is_dir() and (
            (d / "ensemble").is_dir() or
            any((d / f"fold_{i}").is_dir() for i in range(5))
        )
    ])

    if not method_dirs:
        print(f"[ERROR] 未找到预测结果：{PRED_ROOT}")
        print("请先运行 bash 01_run_inference.sh")
        return

    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    # ══════════════════════════════════════════════════════════════
    # 概览汇总表
    # ══════════════════════════════════════════════════════════════
    out(f"3D-IRCADb 外部验证集：{len(all_cases)} cases")
    out(f"  有肿瘤：{len(tumor_cases)}  无肿瘤：{len(no_tumor_cases)}")
    out(f"  无肿瘤 case：{no_tumor_cases}")
    out()

    all_results = {}
    method_labels = {}

    out("=" * 90)
    out(f"{'方法':<45} {'综合Dice':>8} {'肿瘤Dice':>9} {'肝脏Dice':>9} {'无肿瘤FP':>9}")
    out("=" * 90)

    for method_dir in method_dirs:
        ens_dir = method_dir / "ensemble"
        if ens_dir.is_dir() and any(ens_dir.glob("*.nii.gz")):
            pred_dir = ens_dir
            label = "(ens)"
        else:
            fold_dirs = sorted([d for d in method_dir.iterdir()
                                 if d.name.startswith("fold_") and any(d.glob("*.nii.gz"))])
            if not fold_dirs:
                continue
            pred_dir = fold_dirs[-1]
            label = f"({fold_dirs[-1].name})"

        results = eval_one_method(pred_dir, all_cases, case_info)
        valid = {k: v for k, v in results.items() if v is not None}
        if not valid:
            continue

        comp_dice   = comprehensive_dice(valid)
        tumor_dices = [v["tumor_dice"] for v in valid.values() if v["has_tumor"]]
        liver_dices = [v["liver_dice"] for v in valid.values()]
        fp_cases    = [k for k, v in valid.items() if not v["has_tumor"] and v["is_fp"]]

        tumor_mean = float(np.mean(tumor_dices)) if tumor_dices else float("nan")
        liver_mean = float(np.mean(liver_dices)) if liver_dices else float("nan")
        fp_str     = f"{len(fp_cases)}/{len(no_tumor_cases)}"

        method_label = f"{method_dir.name} {label}"
        out(f"{method_label:<45} {comp_dice:>8.3f} {tumor_mean:>9.3f} {liver_mean:>9.3f} {fp_str:>9}")
        all_results[method_dir.name] = results
        method_labels[method_dir.name] = label

    out("=" * 90)

    # ══════════════════════════════════════════════════════════════
    # 跨方法横向对比：无肿瘤 FP 和有肿瘤逐 case
    # ══════════════════════════════════════════════════════════════
    out("\n" + "=" * 70)
    out("无肿瘤 case 详细 FP 报告（跨方法）")
    out("=" * 70)
    col_w = 16
    out(f"{'case':<15}" + "".join(f"{m[:col_w-2]:>{col_w}}" for m in all_results.keys()))
    out("-" * (15 + col_w * len(all_results)))
    for case_id in no_tumor_cases:
        row = f"{case_id:<15}"
        for method, results in all_results.items():
            r = results.get(case_id)
            if r is None:
                row += f"{'—':>{col_w}}"
            elif r["is_fp"]:
                row += f"{'FP '+str(r['pred_tumor_voxels']):>{col_w}}"
            else:
                row += f"{'OK':>{col_w}}"
        out(row)

    out("\n" + "=" * 70)
    out("有肿瘤 case 逐 case 肿瘤 Dice（跨方法）")
    out("=" * 70)
    out(f"{'case':<15}" + "".join(f"{m[:col_w-2]:>{col_w}}" for m in all_results.keys()))
    out("-" * (15 + col_w * len(all_results)))
    for case_id in tumor_cases:
        row = f"{case_id:<15}"
        for method, results in all_results.items():
            r = results.get(case_id)
            if r is None:
                row += f"{'—':>{col_w}}"
            else:
                row += f"{r['tumor_dice']:>{col_w}.3f}"
        out(row)

    # ══════════════════════════════════════════════════════════════
    # 每个方法的详细分析（类似 report_custom.txt）
    # ══════════════════════════════════════════════════════════════
    for method_name, results in all_results.items():
        valid = {k: v for k, v in results.items() if v is not None}
        nt_cases = [k for k, v in valid.items() if not v["has_tumor"]]
        print_method_detail(method_name, method_labels[method_name], results, nt_cases, out)

    # ── 写入文件 ────────────────────────────────────────────────
    OUT_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n[保存] 报告已写入 {OUT_FILE}")


if __name__ == "__main__":
    main()
