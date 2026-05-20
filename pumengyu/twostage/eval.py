"""
端到端两阶段推理评估
  Stage1: 整图 CT → 肝脏分割（Dataset003）
  Stage2: Stage1 预测 mask 裁剪 ROI → 肿瘤分割（Dataset004）

这是论文中"真实推理流程"的评估，区别于训练时用 GT mask 裁剪的验证。
原始 CT 读取 Dataset003_Liver/imagesTr（需提前用 extract_images.sh 解压）。

用法：
  python pumengyu/twostage/eval.py --fold 0
  python pumengyu/twostage/eval.py --fold 0 --keep_tmp   # 保留中间文件
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


# ──────────────────────────────── helpers ────────────────────────────────────

def liver_bbox_sitk(mask_img: sitk.Image, margin_mm: float):
    """Stage1 预测 mask → bbox + margin（与 create_dataset.py 逻辑一致）
    mask 是强调"用来遮住/筛选某个区域",比如把肝脏区域提出来用

    """
    arr     = sitk.GetArrayViewFromImage(mask_img)  # numpy: (z, y, x)
    spacing = mask_img.GetSpacing()                  # SimpleITK: (sp_x, sp_y, sp_z)
    full_sz = mask_img.GetSize()                     # SimpleITK: (sx, sy, sz)

    nz = np.where(arr > 0)

    #nz[0]是z轴非零索引，nz[1]是y轴非零索引，nz[2]是x轴非零索引
    if len(nz[0]) == 0:
        print("  [WARN] Stage1 未预测出肝脏，使用全图")
        return [0, 0, 0], list(full_sz)

    z_min, z_max = int(nz[0].min()), int(nz[0].max()) + 1
    y_min, y_max = int(nz[1].min()), int(nz[1].max()) + 1
    x_min, x_max = int(nz[2].min()), int(nz[2].max()) + 1

    mx = int(np.ceil(margin_mm / spacing[0]))
    my = int(np.ceil(margin_mm / spacing[1]))
    mz = int(np.ceil(margin_mm / spacing[2]))

    x0 = max(0, x_min - mx); x1 = min(full_sz[0], x_max + mx)
    y0 = max(0, y_min - my); y1 = min(full_sz[1], y_max + my)
    z0 = max(0, z_min - mz); z1 = min(full_sz[2], z_max + mz)

    return [x0, y0, z0], [x1-x0, y1-y0, z1-z0]


def crop_sitk(img: sitk.Image, start_xyz: list, size_xyz: list) -> sitk.Image:
    roi = sitk.RegionOfInterestImageFilter()#创建过滤器对象
    roi.SetIndex(start_xyz)#配置参数
    roi.SetSize(size_xyz)#配置参数
    return roi.Execute(img)#执行


def compute_metrics(pred_arr: np.ndarray, gt_arr: np.ndarray,
                    pred_cls: int, gt_cls: int) -> dict:
    p  = pred_arr == pred_cls
    g  = gt_arr   == gt_cls
    #gt_cls是原始真实代表肿瘤的类别,2=肿瘤
    #pred_cls是预测代表肿瘤的类别,1=肿瘤
    #pred_arr 和 gt_arr 是 (z,y,x) 的 numpy 数组，p 和 g 是 bool 数组
    tp = int((p & g).sum())
    fp = int((p & ~g).sum())
    fn = int((~p & g).sum())
    pred_vol = tp + fp
    gt_vol   = tp + fn
    if tp == 0 and fp == 0 and fn == 0:
        return dict(dice=1.0, recall=float('nan'), precision=float('nan'),
                    fdr=0.0, fnr=0.0, pred_tumor=0, gt_tumor=0)
    dice      = 2*tp / (2*tp + fp + fn)
    recall    = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
    precision = tp / (tp + fp) if (tp + fp) > 0 else float('nan')
    fdr       = fp / (tp + fp) if (tp + fp) > 0 else 0.0
    fnr       = fn / (tp + fn) if (tp + fn) > 0 else 0.0
    return dict(dice=dice, recall=recall, precision=precision,
                fdr=fdr, fnr=fnr, pred_tumor=pred_vol, gt_tumor=gt_vol)


def make_predictor(model_folder: str, fold: int,
                   checkpoint: str, device: str) -> nnUNetPredictor:
    """
    model_folder:nnUNet的训练结果的文件夹路径,也是nnUNet的训练完后保存模型权重,配置和预处理参数的目录
    fold是交叉验证的折号
    checkpoint是决定加载训练过程中那个时间点的保存的权重
    nnUNetPredictor是nnUNet官方封装的推理类,负责完整的推理流水线
    """
    pred = nnUNetPredictor(
        tile_step_size=0.5,#滑动窗口的步长比例,0.5表示每次移动半个窗口大小,
        use_gaussian=True,
        use_mirroring=True,#推理时做翻转数据增强
        perform_everything_on_device=True,
        device=torch.device(device),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False,
    )
    pred.initialize_from_trained_model_folder(
        model_folder,#训练结果目录
        use_folds=(fold,),#加载哪几折的权重
        checkpoint_name=checkpoint,
    )
    return pred


def fmt_n(n) -> str:
    return f"{int(n):,}" if n is not None else "N/A"


def predict_with_progress(predictor, inputs: list, outputs: list, label: str = "推理") -> None:
    """逐 case 推理，打印 case 级进度 + 已用时 + ETA，替代 nnUNet 内部的 tile 进度条。"""
    import io, warnings, contextlib
    n  = len(inputs)
    t0 = time.time()
    for i, (inp, out) in enumerate(zip(inputs, outputs)):
        case_name = Path(inp[0]).name.replace("_0000.nii.gz", "")
        elapsed   = time.time() - t0
        if i > 0:
            eta_sec = elapsed / i * (n - i)
            eta_str = f"ETA {eta_sec/60:.1f}min"
        else:
            eta_str = "ETA ?"
        print(f"  [{i+1:2d}/{n}] {case_name:<20}  已用 {elapsed/60:.1f}min  {eta_str}", flush=True)
        with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
            warnings.simplefilter("ignore")
            predictor.predict_from_files(
                [inp], [out],
                save_probabilities=False, overwrite=True,
                num_processes_preprocessing=2,
                num_processes_segmentation_export=2,
            )
    total = time.time() - t0
    print(f"  ✓ {label} {n} 个 case 完成，总耗时 {total/60:.1f}min")

# ──────────────────────────────── main ───────────────────────────────────────

def main():
    """
    /home/PuMengYu/nnUNet/pumengyu/twostage/eval.py
    这个文件是整个两阶段流水线的端到端推理评估脚本,模拟真实部署场景:
    - stage1预测肝脏mask,
    - 用预测mask裁剪ROI
    - step2在裁剪图上预测肿瘤
    - 映射回原图空间评估Dice


    """
    pa = argparse.ArgumentParser()
    pa.add_argument("--fold",            type=int,   default=0)
    pa.add_argument("--workspace",       default="/home/PuMengYu/nnUNet_workspace")
    pa.add_argument("--stage1_trainer",  default="nnUNetTrainer")
    pa.add_argument("--stage2_trainer",  default="nnUNetTrainer_TwoStage")
    pa.add_argument("--margin_mm",       type=float, default=30.0)
    pa.add_argument("--checkpoint",      default="checkpoint_best.pth")
    pa.add_argument("--device",          default="cuda")
    pa.add_argument("--keep_tmp",        action="store_true")
    args = pa.parse_args()

    ws      = Path(args.workspace)
    results = ws / "results"
    preproc = ws / "preprocessed"
    raw     = ws / "raw"

    stage1_folder = str(results / "Dataset003_Liver" /
                        f"{args.stage1_trainer}__nnUNetPlans__3d_fullres")
    stage2_folder = str(results / "Dataset004_LiverTumor" /
                        f"{args.stage2_trainer}__nnUNetPlans__3d_fullres")
    ct_dir       = raw / "Dataset003_Liver" / "imagesTr"
    #ct_dir是原始的CT图像目录,存放liver_xxx_0000.nii.gz,是未经过任何处理的原始输入,stage1直接从这里读图推理
    #gt_dir是真值标签目录,
    gt_dir       = preproc / "Dataset003_Liver" / "gt_segmentations"
    splits_file  = preproc / "Dataset003_Liver" / "splits_final.json"
    s1_cache_dir = ws / "cache" / "stage1_pred"
    s1_cache_dir.mkdir(parents=True, exist_ok=True)

    if not ct_dir.exists():
        raise FileNotFoundError(
            f"找不到原始 CT 目录：{ct_dir}\n请先运行 extract_images.sh 解压 tar 包")

    splits    = json.loads(splits_file.read_text())
    val_cases = sorted(splits[args.fold]["val"])
    print(f"Fold {args.fold}：{len(val_cases)} 个验证 case")

    tmp_root    = Path(tempfile.mkdtemp(prefix="twostage_eval_"))
    cropped_dir = tmp_root / "cropped_ct"
    stage2_out  = tmp_root / "stage2_pred"
    for d in (cropped_dir, stage2_out):
        d.mkdir()

    try:
        # ── Stage 1：肝脏分割 ─────────────────────────────────────────────────
        print("\n===== Stage 1：肝脏分割 =====")
        missing = [c for c in val_cases if not (s1_cache_dir / f"{c}.nii.gz").exists()]
        if missing:
            print(f"  缓存缺失 {len(missing)} 个 case，开始推理并写入缓存")
            s1_inputs  = [[str(ct_dir / f"{c}_0000.nii.gz")] for c in missing]
            s1_outputs = [str(s1_cache_dir / f"{c}.nii.gz")  for c in missing]
            s1_pred = make_predictor(stage1_folder, args.fold, args.checkpoint, args.device)
            predict_with_progress(s1_pred, s1_inputs, s1_outputs, label="Stage1")
            del s1_pred   # 释放显存再加载 Stage2
        else:
            print(f"  全部 {len(val_cases)} 个 case 命中缓存，跳过 Stage1 推理")

        # ── 裁剪：Stage1 mask → 裁剪原始 CT ──────────────────────────────────
        print("\n===== 裁剪肝脏 ROI =====")
        crop_info: dict[str, dict] = {}
        for case in val_cases:
            ct_img     = sitk.ReadImage(str(ct_dir / f"{case}_0000.nii.gz"))
            s1_mask    = sitk.ReadImage(str(s1_cache_dir / f"{case}.nii.gz"))
            liver_mask = sitk.BinaryThreshold(
                s1_mask, lowerThreshold=1, upperThreshold=2,
                insideValue=1, outsideValue=0)

            start_xyz, size_xyz = liver_bbox_sitk(liver_mask, args.margin_mm)
            ct_crop = crop_sitk(ct_img, start_xyz, size_xyz)
            sitk.WriteImage(ct_crop, str(cropped_dir / f"{case}_0000.nii.gz"))
            crop_info[case] = {"start_xyz": start_xyz, "size_xyz": size_xyz,
                               "orig_size":  list(ct_img.GetSize())}
            print(f"  {case:<20} {list(ct_img.GetSize())} → {size_xyz}")

        # ── Stage 2：肿瘤分割（裁剪图）────────────────────────────────────────
        print("\n===== Stage 2：肿瘤分割 =====")
        s2_inputs  = [[str(cropped_dir / f"{c}_0000.nii.gz")] for c in val_cases]
        s2_outputs = [str(stage2_out  / f"{c}.nii.gz")        for c in val_cases]

        s2_pred = make_predictor(stage2_folder, args.fold, args.checkpoint, args.device)
        predict_with_progress(s2_pred, s2_inputs, s2_outputs, label="Stage2")
        del s2_pred

        # ── 评估：Stage2 预测映射回原图空间 ──────────────────────────────────
        print("\n===== 评估 =====")
        per_case = []

        for case in val_cases:
            gt_arr = sitk.GetArrayFromImage(
                sitk.ReadImage(str(gt_dir / f"{case}.nii.gz")))   # (z,y,x) label2=tumor
            s2_arr = sitk.GetArrayFromImage(
                sitk.ReadImage(str(stage2_out / f"{case}.nii.gz")))  # (z,y,x) label1=tumor

            x0, y0, z0 = crop_info[case]["start_xyz"]
            sx, sy, sz = crop_info[case]["size_xyz"]
            full_pred  = np.zeros(gt_arr.shape, dtype=np.uint8)
            full_pred[z0:z0+sz, y0:y0+sy, x0:x0+sx] = s2_arr

            gt_vol = int((gt_arr == 2).sum())
            m = compute_metrics(full_pred, gt_arr, pred_cls=1, gt_cls=2)
            m["case"]      = case
            m["has_tumor"] = gt_vol > 0
            per_case.append(m)

        # ── 汇总报告 ──────────────────────────────────────────────────────────
        has_t = [r for r in per_case if r["has_tumor"]]
        no_t  = [r for r in per_case if not r["has_tumor"]]

        lines = [
            "nnUNet 端到端两阶段推理评估报告",
            "=" * 50,
            f"fold          : {args.fold}",
            f"stage1_trainer: {args.stage1_trainer}",
            f"stage2_trainer: {args.stage2_trainer}",
            f"margin_mm     : {args.margin_mm}",
            f"checkpoint    : {args.checkpoint}",
            f"n_cases       : {len(per_case)}",
            "",
            "裁剪来源：Stage1 预测 mask（非 GT），真实推理流程",
            "",
        ]

        lines.append(f"Tumor (有肿瘤 case, n={len(has_t)})")
        if has_t:
            for metric, key in [("Dice", "dice"), ("Recall", "recall"),
                                 ("Precision", "precision"), ("FDR", "fdr"),
                                 ("FNR", "fnr")]:
                vals = [r[key] for r in has_t]
                lines.append(f"  {metric:<12}: mean={np.mean(vals):.4f}  std={np.std(vals):.4f}")
        lines.append("")

        lines.append(f"Tumor (无肿瘤 case, n={len(no_t)})")
        if no_t:
            fp_cases = [r for r in no_t if r["pred_tumor"] > 0]
            vols = [r["pred_tumor"] for r in no_t]
            lines.append(f"  误报率       : {len(fp_cases)}/{len(no_t)}")
            lines.append(f"  FP pred_tumor: mean={np.mean(vols):.1f}  std={np.std(vols):.1f}")
            for r in no_t:
                flag = "  [误报]" if r["pred_tumor"] > 0 else ""
                lines.append(f"    {r['case']:<20}  pred_tumor={fmt_n(r['pred_tumor'])}{flag}")
        lines.append("")

        sep = "-" * 100
        lines.append("\n" + "=" * 80)
        lines.append("Per-Case 详情（按 tumor_dice 从低到高）")
        lines.append("=" * 80)
        col = (f"  {'case':<20} {'dice':>8} {'recall':>8} {'precision':>10}"
               f" {'FDR':>8} {'pred_tumor':>12} {'gt_tumor':>10}")
        for label, cond in [
            ("[严重失败] dice < 0.3",       lambda r: r["dice"] < 0.3),
            ("[需要改进] 0.3 ≤ dice < 0.7", lambda r: 0.3 <= r["dice"] < 0.7),
            ("[没问题]   dice ≥ 0.7",       lambda r: r["dice"] >= 0.7),
        ]:
            subset = sorted([r for r in has_t if cond(r)], key=lambda r: r["dice"])
            lines.append(f"\n{label}  (n={len(subset)})")
            lines.append(sep)
            lines.append(col)
            lines.append(sep)
            for r in subset:
                lines.append(
                    f"  {r['case']:<20} {r['dice']:>8.4f} {r['recall']:>8.4f}"
                    f" {r['precision']:>10.4f} {r['fdr']:>8.4f}"
                    f" {fmt_n(r['pred_tumor']):>12} {fmt_n(r['gt_tumor']):>10}")

        margin_tag  = f"{int(args.margin_mm)}mm"
        ts          = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir     = Path(stage2_folder) / f"fold_{args.fold}" / f"{ts}_e2e_{margin_tag}"
        run_dir.mkdir(parents=True, exist_ok=True)

        report_txt  = run_dir / "eval_e2e.txt"
        report_txt.write_text("\n".join(lines), encoding="utf-8")

        report_json = run_dir / "eval_e2e.json"
        report_json.write_text(json.dumps({
            "fold": args.fold, "stage1_trainer": args.stage1_trainer,
            "stage2_trainer": args.stage2_trainer, "margin_mm": args.margin_mm,
            "checkpoint": args.checkpoint, "per_case": per_case,
            "summary_has_tumor": {
                "n": len(has_t),
                "dice_mean":   float(np.mean([r["dice"] for r in has_t])) if has_t else None,
                "dice_std":    float(np.std( [r["dice"] for r in has_t])) if has_t else None,
                "recall_mean": float(np.mean([r["recall"] for r in has_t])) if has_t else None,
                "fdr_mean":    float(np.mean([r["fdr"] for r in has_t])) if has_t else None,
            },
        }, indent=2, ensure_ascii=False))

        print("\n".join(lines))
        print(f"\n报告已保存 → {run_dir}/\n  {report_txt.name}\n  {report_json.name}")

    finally:
        if not args.keep_tmp:
            shutil.rmtree(tmp_root)
        else:
            print(f"临时目录保留：{tmp_root}")


if __name__ == "__main__":
    main()
