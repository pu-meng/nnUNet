"""
为外部验证集（ircadb）的单个方法生成 report_custom.txt + test_viz/。

目录结构（统一在 results_v2/ 下，与内部实验并列）：
    results_v2/IRCADb/source_only/MoE_SizeOV4/
        predictions/  *.nii.gz          ← nnUNetv2_predict 原始输出
        report_custom.txt               ← 指标报告
        test_viz/{case}/                ← 可视化 PNG

用法：
    # 只生成报告和 viz（预测已存在）
    python pumengyu/ext_val/03_gen_method_report.py --method MoE_SizeOV4

    # 先推理再生成报告+viz（一键 pipeline）
    python pumengyu/ext_val/03_gen_method_report.py \\
        --method MoE_SizeOV4 \\
        --predict \\
        --trainer nnUNetTrainer_MLAUNet_MoE_SizeOversampleV4 \\
        --gpu 1

    # 跳过 viz
    python pumengyu/ext_val/03_gen_method_report.py --method MoE_SizeOV4 --no_vis
"""

from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage
from tqdm import tqdm

# ── 路径常量 ──────────────────────────────────────────────────────────────
EXT_VAL_ROOT  = Path("/home/PuMengYu/nnUNet_workspace/external_val/ircadb_full")
LABEL_DIR     = EXT_VAL_ROOT / "labels"
IMAGE_DIR     = EXT_VAL_ROOT / "images"
INFO_FILE     = EXT_VAL_ROOT / "case_info.json"

# 外部验证结果统一写入 results_v2/IRCADb/source_only/，与内部实验并列
EXT_RESULT_ROOT = Path("/home/PuMengYu/nnUNet_workspace/results_v2/IRCADb/source_only")
NNUNET_RESULTS = Path("/home/PuMengYu/nnUNet_workspace/results_v2")

TUMOR_CLS = 2   # label 1=liver, 2=tumor（同 Dataset003）
LIVER_CLS = 1

NNUNET_ROOT = "/home/PuMengYu/nnUNet"
if NNUNET_ROOT not in sys.path:
    sys.path.insert(0, NNUNET_ROOT)

from pumengyu.tools.analyasis.metric_standard import aggregate_liver_tumor_metrics

# ── 工具函数 ──────────────────────────────────────────────────────────────

def _load(path: Path) -> np.ndarray:
    return np.asarray(nib.load(str(path)).dataobj)


def _vox_vol(path: Path) -> float:
    hdr = nib.load(str(path)).header
    z = hdr.get_zooms()[:3]
    return float(z[0] * z[1] * z[2])


def _dice(pred, gt, cls):
    p = (pred == cls).astype(np.float32)
    g = (gt   == cls).astype(np.float32)
    inter = (p * g).sum()
    denom = p.sum() + g.sum()
    return float(2 * inter / denom) if denom > 0 else 1.0


def size_cat(n: int) -> str:
    if n < 5_000:    return "极小(<5k)"
    if n < 50_000:   return "小(5k-50k)"
    if n < 300_000:  return "中等(50k-300k)"
    return "大(>=300k)"


def fmt_n(n) -> str:
    return f"{int(n):,}" if n is not None else "N/A"


# ── 推理 pipeline ─────────────────────────────────────────────────────────

def _resolve_dataset_name(dataset: str) -> str:
    """Accept either nnU-Net ids like 003/13 or full DatasetXXX names."""
    if dataset.startswith("Dataset"):
        return dataset
    dataset_id = int(dataset)
    matches = sorted(NNUNET_RESULTS.glob(f"Dataset{dataset_id:03d}_*"))
    if not matches:
        raise FileNotFoundError(f"找不到 Dataset{dataset_id:03d}_* 结果目录: {NNUNET_RESULTS}")
    if len(matches) > 1:
        raise RuntimeError(f"Dataset id {dataset_id:03d} 匹配到多个目录: {matches}")
    return matches[0].name


def _dataset_id_arg(dataset: str) -> str:
    if dataset.startswith("Dataset"):
        return str(int(dataset.removeprefix("Dataset")[:3]))
    return str(int(dataset))


def _expected_channels(dataset: str, trainer: str) -> int:
    dataset_name = _resolve_dataset_name(dataset)
    dataset_json = NNUNET_RESULTS / dataset_name / f"{trainer}__nnUNetPlans__3d_fullres" / "dataset.json"
    if not dataset_json.exists():
        raise FileNotFoundError(f"模型目录缺 dataset.json: {dataset_json}")
    data = json.loads(dataset_json.read_text(encoding="utf-8"))
    return len(data.get("channel_names", {"0": "CT"}))


def _prepare_predict_input(method: str, dataset: str, trainer: str) -> Path:
    channels = _expected_channels(dataset, trainer)
    if channels <= 1:
        return IMAGE_DIR

    out_dir = EXT_RESULT_ROOT / method / f"images_{channels}ch_repeat"
    out_dir.mkdir(parents=True, exist_ok=True)
    for src in sorted(IMAGE_DIR.glob("*_0000.nii.gz")):
        stem = src.name.removesuffix("_0000.nii.gz")
        for c in range(channels):
            dst = out_dir / f"{stem}_{c:04d}.nii.gz"
            if dst.exists():
                continue
            try:
                dst.symlink_to(src)
            except OSError:
                import shutil
                shutil.copy2(src, dst)
    print(f"[推理] Dataset {dataset} 需要 {channels} 通道，使用重复通道输入: {out_dir}")
    return out_dir


def run_predict(method: str, trainer: str, fold: int, gpu: int, checkpoint: str = "", dataset: str = "003"):
    """调用 nnUNetv2_predict 生成预测文件。"""
    out_dir = EXT_RESULT_ROOT / method / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    input_dir = _prepare_predict_input(method, dataset, trainer)
    cmd = [
        "nnUNetv2_predict",
        "-i", str(input_dir),
        "-o", str(out_dir),
        "-d", _dataset_id_arg(dataset),
        "-c", "3d_fullres",
        "-tr", trainer,
        "-f", str(fold),
    ]
    if checkpoint:
        cmd += ["-chk", checkpoint]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    print(f"[推理] 执行: CUDA_VISIBLE_DEVICES={gpu} {' '.join(cmd)}")
    subprocess.run(cmd, env=env, check=True)
    print("[推理] 完成")


# ── 评估（单次循环，同时做 metrics + CC 分析）────────────────────────────

def evaluate(pred_dir: Path, case_info: dict) -> tuple[list, list, list, list]:
    """
    单次遍历所有 case，同时完成：
      - 指标计算（Dice / Recall / Precision 等）
      - 连通域分析（FP CC + TP CC）
    返回: has_tumor_cases, no_tumor_cases, fp_cc_info, tp_cc_sizes
    """
    all_cases = sorted(case_info.keys())
    has_tumor_cases: list[dict] = []
    no_tumor_cases:  list[dict] = []
    fp_cc_info:      list[dict] = []
    tp_cc_sizes:     list[int]  = []

    for case in tqdm(all_cases, desc="评估+CC分析", unit="case", ncols=80):
        pred_path = pred_dir  / f"{case}.nii.gz"
        gt_path   = LABEL_DIR / f"{case}.nii.gz"
        if not pred_path.exists() or not gt_path.exists():
            print(f"  [WARN] 跳过 {case}（缺预测或GT）")
            continue

        pred = _load(pred_path).astype("uint8")
        gt   = _load(gt_path).astype("uint8")
        vv   = _vox_vol(pred_path)

        gt_tumor   = int((gt == TUMOR_CLS).sum())
        pred_tumor = int((pred == TUMOR_CLS).sum())
        gt_liver   = int((gt >= LIVER_CLS).sum())

        liver_dice = _dice(pred, gt, LIVER_CLS)
        liver_fp   = int(((pred == LIVER_CLS) & (gt != LIVER_CLS)).sum())
        liver_fn   = int(((gt   == LIVER_CLS) & (pred != LIVER_CLS)).sum())

        # ── CC 分析（与指标共用同一次 nii 加载）────────────────────────
        gt_mask   = gt   == TUMOR_CLS
        pred_mask = pred == TUMOR_CLS
        labeled_pred, n_pred_cc = ndimage.label(pred_mask)
        for i in range(1, n_pred_cc + 1):
            cc_mask = labeled_pred == i
            vol     = int(cc_mask.sum())
            overlap = int((cc_mask & gt_mask).sum())
            if gt_tumor == 0:
                fp_cc_info.append({"case": case, "vol": vol})
            elif overlap > 0:
                tp_cc_sizes.append(vol)

        if gt_tumor == 0:
            no_tumor_cases.append({
                "case": case, "liver_dice": liver_dice,
                "pred_tumor": pred_tumor, "gt_liver": gt_liver,
                "tumor_fp": pred_tumor, "tumor_fn": 0,
                "liver_fp": liver_fp, "liver_fn": liver_fn, "vv": vv,
            })
        else:
            tp = int(((pred == TUMOR_CLS) & (gt == TUMOR_CLS)).sum())
            fp = int(((pred == TUMOR_CLS) & (gt != TUMOR_CLS)).sum())
            fn = int(((gt   == TUMOR_CLS) & (pred != TUMOR_CLS)).sum())
            denom_r = tp + fn
            denom_p = tp + fp
            dice_t    = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
            recall    = tp / denom_r if denom_r > 0 else 0.0
            precision = tp / denom_p if denom_p > 0 else 0.0
            fdr       = fp / denom_p if denom_p > 0 else 0.0
            fnr       = fn / denom_r if denom_r > 0 else 0.0
            jaccard   = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
            has_tumor_cases.append({
                "case": case, "liver_dice": liver_dice,
                "dice": dice_t, "recall": recall, "precision": precision,
                "fdr": fdr, "fnr": fnr, "jaccard": jaccard,
                "pred_tumor": pred_tumor, "gt_tumor": gt_tumor, "gt_liver": gt_liver,
                "tumor_fp": fp, "tumor_fn": fn,
                "liver_fp": liver_fp, "liver_fn": liver_fn, "vv": vv,
            })

    has_tumor_cases.sort(key=lambda x: x["dice"])
    return has_tumor_cases, no_tumor_cases, fp_cc_info, tp_cc_sizes


# ── 报告生成 ──────────────────────────────────────────────────────────────

def build_report(method: str, pred_dir: Path,
                 has_tumor_cases, no_tumor_cases,
                 fp_cc_info, tp_cc_sizes) -> str:
    all_cases_data  = has_tumor_cases + no_tumor_cases
    false_pos_cases = [r for r in no_tumor_cases if r["pred_tumor"] > 0]

    liver_dices = [r["liver_dice"] for r in all_cases_data]
    mean_liver  = float(np.mean(liver_dices)) if liver_dices else float("nan")

    agg = aggregate_liver_tumor_metrics(has_tumor_cases, no_tumor_cases)
    primary = agg["primary"]
    nnref = agg["nnunet_reference"]
    n_tn, n_fp = agg["n_negative_tn"], agg["n_negative_fp"]
    mean_tumor = primary["dice"]["mean"]
    overall = primary["overall"]

    def mm3(r, key): return r[key] * r["vv"]

    SIZE_BINS = [
        ("极小(<5k)",       lambda g: g < 5_000),
        ("小(5k-50k)",      lambda g: 5_000 <= g < 50_000),
        ("中等(50k-300k)",  lambda g: 50_000 <= g < 300_000),
        ("大(>=300k)",      lambda g: g >= 300_000),
    ]

    L = []   # lines
    L += [
        "nnUNet External Validation Report (IRCADb)",
        "=" * 40,
        f"method   : {method}",
        f"pred_dir : {pred_dir}",
        f"n_cases  : {len(all_cases_data)}",
        "",
        "Liver",
        f"  Dice: mean={mean_liver:.4f}  std={np.std(liver_dices):.4f}",
        "",
        f"Tumor (无肿瘤 case, n={len(no_tumor_cases)})",
    ]
    if no_tumor_cases:
        fp_rate  = len(false_pos_cases) / len(no_tumor_cases)
        no_t_vol = [r["pred_tumor"] for r in no_tumor_cases]
        L.append(f"  误报率(预测出肿瘤但GT无肿瘤): {fp_rate:.2%}  ({len(false_pos_cases)}/{len(no_tumor_cases)} cases)")
        L.append(f"  FP pred_tumor : mean={np.mean(no_t_vol):.1f}  std={np.std(no_t_vol):.1f}")
        if false_pos_cases:
            L.append("  误报 cases:")
            for r in false_pos_cases:
                L.append(f"    {r['case']:<22}  pred_tumor={fmt_n(r['pred_tumor'])}")
        else:
            L.append("  所有无肿瘤 case 均正确预测为阴性")
        L.append("\n  无肿瘤 case 列表:")
        for r in no_tumor_cases:
            flag = "  [误报]" if r["pred_tumor"] > 0 else ""
            L.append(f"    {r['case']:<22}  liver_dice={r['liver_dice']:.4f}"
                     f"  pred_tumor={fmt_n(r['pred_tumor'])}{flag}")
    else:
        L.append("  当前评估集不含无肿瘤 case，阴性误报率不可评估（N/A）")
    L += [
        "",
        "Tumor 综合指标（PMY-LT-v1 论文主口径；仅 GT 有肿瘤 case）",
        f"  无肿瘤 case (n={len(no_tumor_cases)}) 的 Tumor Dice/Recall/Precision 统一为 N/A，不进入均值",
        f"  Dice        : mean={primary['dice']['mean']:.4f}  std={primary['dice']['std']:.4f}  (有肿瘤 n={len(has_tumor_cases)})",
        f"  Jaccard     : mean={primary['jaccard']['mean']:.4f}  std={primary['jaccard']['std']:.4f}  (有肿瘤 n={len(has_tumor_cases)})",
        f"  Recall      : mean={primary['recall']['mean']:.4f}  std={primary['recall']['std']:.4f}  (有肿瘤 n={len(has_tumor_cases)})",
        f"  FNR         : mean={primary['fnr']['mean']:.4f}  std={primary['fnr']['std']:.4f}  (有肿瘤 n={len(has_tumor_cases)})",
        f"  Precision   : mean={primary['precision']['mean']:.4f}  std={primary['precision']['std']:.4f}  (有肿瘤 n={len(has_tumor_cases)})",
        f"  FDR         : mean={primary['fdr']['mean']:.4f}  std={primary['fdr']['std']:.4f}  (有肿瘤 n={len(has_tumor_cases)})",
        f"  Overall     : (all-case liver {mean_liver:.4f} + positive-only tumor {mean_tumor:.4f}) / 2 = {overall:.4f}",
        f"  构成        : 全部 n={agg['n_all']} | 有肿瘤 n={len(has_tumor_cases)} | 无肿瘤 n={len(no_tumor_cases)} (TN={n_tn}, FP={n_fp})",
        "",
        "nnUNet foreground_mean 参考（仅供 summary.json 追溯，不参与论文排名）",
        f"  Tumor Dice : mean={nnref['tumor_dice']['mean']:.4f}  std={nnref['tumor_dice']['std']:.4f}  (valid n={nnref['n_tumor_valid']})",
        f"  Overall    : {nnref['overall']:.4f}",
        "",
    ]

    # FPV / FNV
    t_fpv = [mm3(r, "tumor_fp") for r in all_cases_data]
    t_fnv = [mm3(r, "tumor_fn") for r in all_cases_data]
    l_fpv = [mm3(r, "liver_fp") for r in all_cases_data]
    l_fnv = [mm3(r, "liver_fn") for r in all_cases_data]
    L += [
        "FPV / FNV 体积误差（mm³，体素数 × spacing）",
        f"  {'':20} {'FPV总量(mm³)':>14} {'FPV均值/case':>14} {'FNV总量(mm³)':>14} {'FNV均值/case':>14}",
        f"  {'Tumor':20} {int(sum(t_fpv)):>14,} {np.mean(t_fpv):>14,.1f}"
        f" {int(sum(t_fnv)):>14,} {np.mean(t_fnv):>14,.1f}",
        f"  {'Liver':20} {int(sum(l_fpv)):>14,} {np.mean(l_fpv):>14,.1f}"
        f" {int(sum(l_fnv)):>14,} {np.mean(l_fnv):>14,.1f}",
        "",
        "  Per-case Tumor FPV（从高到低，前10）",
        f"  {'case':<22} {'FPV(mm³)':>12} {'FNV(mm³)':>12} {'gt_tumor(vox)':>14} {'size_cat':<16}",
    ]
    for r in sorted(all_cases_data, key=lambda r: mm3(r, "tumor_fp"), reverse=True)[:10]:
        sc = size_cat(r.get("gt_tumor", 0)) if r.get("gt_tumor", 0) > 0 else "无肿瘤"
        L.append(f"  {r['case']:<22} {int(mm3(r,'tumor_fp')):>12,}"
                 f" {int(mm3(r,'tumor_fn')):>12,}"
                 f" {fmt_n(r.get('gt_tumor',0)):>14} {sc:<16}")
    L.append("")

    # CC 分析
    fp_cases_cc = sorted({d["case"] for d in fp_cc_info})
    L += [
        "连通域阈值分析（后处理参考）",
        f"  Q1 无肿瘤 case 被预测出连通域： {len(fp_cases_cc)}/{len(no_tumor_cases)} cases",
    ]
    if fp_cases_cc:
        for fc in fp_cases_cc:
            vols = sorted([d["vol"] for d in fp_cc_info if d["case"] == fc])
            L.append(f"    {fc}: {len(vols)} 个假CC，体素数={vols}")
    elif no_tumor_cases:
        L.append("    所有无肿瘤 case 预测均为空 ✓")
    else:
        L.append("    当前评估集不含无肿瘤 case，该项不可评估（N/A）")
    if fp_cc_info:
        all_fp_vols = sorted([d["vol"] for d in fp_cc_info])
        L.append(f"  Q2 假连通域大小: min={min(all_fp_vols)}  max={max(all_fp_vols)}"
                 f"  mean={np.mean(all_fp_vols):.0f}  列表={all_fp_vols}")
    else:
        L.append("  Q2 无假连通域")
    if tp_cc_sizes:
        L.append(f"  Q3 真阳性CC最小体素数: {min(tp_cc_sizes)}"
                 f"  (共 {len(tp_cc_sizes)} 个TP CC，min={min(tp_cc_sizes)} max={max(tp_cc_sizes)})")
        L.append(f"     → 后处理阈值上限 < {min(tp_cc_sizes)} 体素")
    else:
        L.append("  Q3 未找到真阳性 CC")
    L.append("")

    # 按大小分组
    L += [
        "Tumor Dice 按大小分组 (有肿瘤 case)",
        f"  {'大小分类':<16} {'n':>4}  {'Dice mean':>10}  {'Dice std':>9}  {'Recall':>8}  {'Precision':>10}",
        "  " + "-" * 68,
    ]
    for label, cond in SIZE_BINS:
        subset = [r for r in has_tumor_cases if cond(r["gt_tumor"])]
        if subset:
            L.append(f"  {label:<16} {len(subset):>4}  "
                     f"{np.mean([r['dice'] for r in subset]):>10.4f}  "
                     f"{np.std([r['dice'] for r in subset]):>9.4f}  "
                     f"{np.mean([r['recall'] for r in subset]):>8.4f}  "
                     f"{np.mean([r['precision'] for r in subset]):>10.4f}")
        else:
            L.append(f"  {label:<16} {0:>4}  {'—':>10}")
    L.append("")

    # Per-case 分级
    SEP     = "-" * 104
    COL_HDR = (f"  {'case':<22} {'tumor_dice':>10} {'recall':>8} {'precision':>10}"
               f" {'FDR':>8} {'pred_tumor':>12} {'gt_tumor':>10} {'gt_liver':>10} {'size_cat':<18}")
    L += ["", "=" * 80, "Per-Case 分级(按 tumor_dice 从低到高)", "=" * 80]
    TIERS = [
        ("[严重失败] tumor_dice < 0.3",        lambda r: r["dice"] < 0.3),
        ("[需要改进] 0.3 <= tumor_dice < 0.7",  lambda r: 0.3 <= r["dice"] < 0.7),
        ("[没问题]   tumor_dice >= 0.7",         lambda r: r["dice"] >= 0.7),
    ]
    for tier_label, cond in TIERS:
        subset = [r for r in has_tumor_cases if cond(r)]
        L += [f"\n{tier_label}  (n={len(subset)})", SEP, COL_HDR, SEP]
        for r in subset:
            L.append(
                f"  {r['case']:<22} {r['dice']:>10.4f} {r['recall']:>8.4f} {r['precision']:>10.4f}"
                f" {r['fdr']:>8.4f} {fmt_n(r['pred_tumor']):>12} {fmt_n(r['gt_tumor']):>10}"
                f" {fmt_n(r['gt_liver']):>10} {size_cat(r['gt_tumor']):<18}"
            )
    if false_pos_cases:
        fp_col = (f"  {'case':<22} {'liver_dice':>10} {'pred_tumor':>12}"
                  f" {'gt_liver':>10}  说明")
        L += [f"\n[无肿瘤误报] tumor_dice=N/A，误报单独报告  (n={len(false_pos_cases)})",
              SEP, fp_col, SEP]
        for r in sorted(false_pos_cases, key=lambda x: x["pred_tumor"], reverse=True):
            L.append(f"  {r['case']:<22} {r['liver_dice']:>10.4f} {fmt_n(r['pred_tumor']):>12}"
                     f" {fmt_n(r['gt_liver']):>10}  GT无肿瘤,预测出{fmt_n(r['pred_tumor'])}体素")
    elif no_tumor_cases:
        L.append(f"\n[无肿瘤误报] 无  (全部 {len(no_tumor_cases)} 个无肿瘤 case 均正确预测为阴性 ✓)")
    else:
        L.append("\n[无肿瘤误报] N/A  (当前评估集无阴性 case，不能据此判断误报能力)")

    return "\n".join(L)


# ── 主函数 ────────────────────────────────────────────────────────────────

def run(method: str, no_vis: bool, predict: bool, trainer: str, fold: int, gpu: int,
        min_voxel: int, checkpoint: str, dataset: str, force_predict: bool = False):
    method_dir = EXT_RESULT_ROOT / method
    pred_dir   = method_dir / "predictions"   # nii.gz 所在目录
    result_dir = method_dir                   # 报告 + viz 输出目录
    result_dir.mkdir(parents=True, exist_ok=True)

    # 1. 可选：先推理
    if predict:
        if not trainer:
            raise ValueError("--predict 需要同时指定 --trainer")
        if str(dataset).lstrip("0") == "3" and checkpoint == "checkpoint_best.pth":
            # Source-only external evaluations must inherit the same checkpoint
            # provenance gate as the internal best-only report.
            from pumengyu.tools.run_internal_test_best_report import (
                validate_best_checkpoint_provenance,
            )
            validate_best_checkpoint_provenance(trainer)
        expected_cases = list(json.loads(INFO_FILE.read_text(encoding="utf-8")))
        predictions_complete = all(
            (pred_dir / f"{case}.nii.gz").is_file() for case in expected_cases
        )
        if predictions_complete and not force_predict:
            print(f"[reuse] {method}: reuse {len(expected_cases)} existing predictions")
        else:
            run_predict(method, trainer, fold, gpu, checkpoint, dataset)

    if not pred_dir.is_dir():
        raise FileNotFoundError(f"预测目录不存在: {pred_dir}")

    case_info: dict = json.load(open(INFO_FILE))

    # 2. summary.json（nnUNet 官方格式，写入 predictions/）
    summary_path = pred_dir / "summary.json"
    print(f"[summary.json] 计算中 → {summary_path}")
    from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder
    from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
    compute_metrics_on_folder(
        folder_ref=str(LABEL_DIR),
        folder_pred=str(pred_dir),
        output_file=str(summary_path),
        image_reader_writer=SimpleITKIO(),
        file_ending=".nii.gz",
        regions_or_labels=[1, 2],   # 1=liver  2=tumor
        ignore_label=None,
        chill=True,
    )
    print(f"[summary.json] 完成")

    # 3. 评估 + CC 分析（单次循环）
    has_tumor_cases, no_tumor_cases, fp_cc_info, tp_cc_sizes = evaluate(pred_dir, case_info)

    # 4. 生成报告 → method/report_custom.txt
    report_text = build_report(method, pred_dir,
                               has_tumor_cases, no_tumor_cases,
                               fp_cc_info, tp_cc_sizes)
    report_path = result_dir / "report_custom.txt"
    report_path.write_text(report_text, encoding="utf-8")

    agg = aggregate_liver_tumor_metrics(has_tumor_cases, no_tumor_cases)
    false_pos_cases = agg["false_positive_cases"]
    mean_tumor = agg["primary"]["dice"]["mean"]
    mean_liver = agg["liver"]["mean"]

    print(f"\n[报告] → {report_path}")
    print(f"  Overall={agg['primary']['overall']:.4f}  Liver={mean_liver:.4f}  Tumor={mean_tumor:.4f}")
    print(f"  Recall={agg['primary']['recall']['mean']:.4f}  Precision={agg['primary']['precision']['mean']:.4f}  FDR={agg['primary']['fdr']['mean']:.4f}")
    if no_tumor_cases:
        print(f"  无肿瘤误报: {len(false_pos_cases)}/{len(no_tumor_cases)}")
    else:
        print("  无肿瘤误报: N/A（当前评估集无阴性 case）")

    # 5. 可视化 → method/test_viz/（复用 mixin 的 _gen_viz_pngs_and_cleanup）
    if no_vis:
        print("[可视化] 跳过（--no_vis）")
        return

    from pumengyu.mixins import _gen_viz_pngs_and_cleanup
    viz_dir = result_dir / "test_viz"
    viz_dir.mkdir(exist_ok=True)
    print(f"[可视化] 生成中（仅 FP/FN >= {min_voxel} 体素的切片）→ {viz_dir}/")
    _gen_viz_pngs_and_cleanup(
        pred_folder=pred_dir,
        gt_dir=LABEL_DIR,
        img_dir=IMAGE_DIR,
        out_viz_dir=viz_dir,
        min_voxel=min_voxel,
        delete_nii=False,   # 外部验证保留预测文件
    )
    print(f"[完成] IRCADb/source_only/{method}/")


# ── CLI ───────────────────────────────────────────────────────────────────

def main():
    global EXT_RESULT_ROOT
    p = argparse.ArgumentParser()
    p.add_argument("--method",    required=True,
                   help="预测目录名，如 MoE_SizeOV4")
    p.add_argument("--no_vis",    action="store_true",
                   help="跳过可视化生成")
    p.add_argument("--min_voxel", type=int, default=20,
                   help="viz 最小 FP/FN 体素数（默认 20，与内部 test_viz 一致）")
    # pipeline 选项
    p.add_argument("--predict",   action="store_true",
                   help="先运行 nnUNetv2_predict 再生成报告")
    p.add_argument("--trainer",   default="",
                   help="trainer 名称，--predict 时必填，如 nnUNetTrainer_MLAUNet_MoE_SizeOversampleV4")
    p.add_argument("--fold",      type=int, default=0,
                   help="fold 编号（默认 0）")
    p.add_argument("--gpu",       type=int, default=1,
                   help="CUDA_VISIBLE_DEVICES（默认 1）")
    p.add_argument("--checkpoint", default="checkpoint_best.pth",
                   help="nnUNet checkpoint 文件名（默认且推荐：checkpoint_best.pth）")
    p.add_argument("--dataset", default="003",
                   help="预测使用的数据集 id/name，默认 003；HCC/MSD 混合模型应使用 013")
    p.add_argument("--force_predict", action="store_true",
                   help="即使预测文件齐全也强制重新推理；默认复用并只补报告/可视化")
    p.add_argument("--result_root", default=str(EXT_RESULT_ROOT),
                   help="外部验证输出根目录，默认 results_v2/IRCADb/source_only")
    args = p.parse_args()
    EXT_RESULT_ROOT = Path(args.result_root)
    run(args.method, args.no_vis, args.predict, args.trainer, args.fold, args.gpu,
        args.min_voxel, args.checkpoint, args.dataset, args.force_predict)


if __name__ == "__main__":
    main()
