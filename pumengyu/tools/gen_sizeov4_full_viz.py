"""
SizeOversampleV4 全量 131-case 可视化生成脚本
=============================================
覆盖所有 train(92) + val(13) + test(26) = 131 个 case，
输出到 fold_0/train_viz/、val_viz/、test_viz/（完全替换旧内容）。

- 有 FP/FN 的 case：生成每张错误切片的 PNG（现有格式）
- 无误差的 clean case：生成 1-3 张代表切片，标题注明 ✅

用法（在 /home/PuMengYu/nnUNet 目录）：
    CUDA_VISIBLE_DEVICES=0 conda run -n medseg python pumengyu/tools/gen_sizeov4_full_viz.py

可选参数：
    --skip-train   跳过 train_viz（不重新推理 92 个 train case）
    --skip-val     跳过 val_viz
    --skip-test    跳过 test_viz
    --min-voxel    错误切片最小体素阈值，默认 1
    --clean-slices 无误差 case 展示的切片数，默认 3
"""

import argparse, json, os, shutil, subprocess, sys, tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import Patch
import nibabel as nib
import numpy as np

# ── 环境变量 ──────────────────────────────────────────────────────────────────
os.environ.setdefault("nnUNet_raw",          "/home/PuMengYu/nnUNet_workspace/raw")
os.environ.setdefault("nnUNet_preprocessed", "/home/PuMengYu/nnUNet_workspace/preprocessed")
os.environ.setdefault("nnUNet_results",      "/home/PuMengYu/nnUNet_workspace/results_v2")

NNUNET_ROOT = "/home/PuMengYu/nnUNet"
if NNUNET_ROOT not in sys.path:
    sys.path.insert(0, NNUNET_ROOT)

TRAINER   = "nnUNetTrainer_MLAUNet_MoE_SizeOversampleV4"
DATASET   = "Dataset003_Liver"
FOLD      = 0
CHECKPOINT = "checkpoint_best.pth"

SPLIT_FILE = Path("/home/PuMengYu/nnUNet_workspace/preprocessed/Dataset003_Liver/split_info_712.json")
IMG_DIR    = Path("/home/PuMengYu/nnUNet_workspace/raw/Dataset003_Liver/imagesTr")
GT_DIR     = Path("/home/PuMengYu/nnUNet_workspace/preprocessed/Dataset003_Liver/gt_segmentations")

# ── 字体 ─────────────────────────────────────────────────────────────────────
_CN_FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
_cn_prop = fm.FontProperties(fname=_CN_FONT) if os.path.exists(_CN_FONT) else None
matplotlib.rcParams["axes.unicode_minus"] = False

LEGEND_ENTRIES = [
    Patch(facecolor="green", alpha=0.25, label="TP"),
    Patch(facecolor="red",   alpha=0.75, label="FP"),
    Patch(facecolor="blue",  alpha=0.70, label="FN"),
]


def _title(s):
    kw = dict(fontproperties=_cn_prop) if _cn_prop else {}
    return dict(label=s, **kw)


def gen_case_viz(case: str, ct_path: Path, gt_path: Path, pred_path: Path,
                 out_dir: Path, min_voxel: int, clean_slices: int):
    """为单个 case 生成 PNG，返回生成的 PNG 数量。"""
    ct   = nib.load(ct_path).get_fdata()
    gt   = nib.load(gt_path).get_fdata()
    pred = nib.load(pred_path).get_fdata()

    gt_tumor   = (gt   == 2).astype(np.uint8)
    pred_tumor = (pred == 2).astype(np.uint8)

    combined = np.zeros_like(gt, dtype=np.uint8)
    combined[(gt_tumor == 1) & (pred_tumor == 1)] = 1  # TP
    combined[(gt_tumor == 0) & (pred_tumor == 1)] = 2  # FP
    combined[(gt_tumor == 1) & (pred_tumor == 0)] = 3  # FN

    error_per_slice = ((combined == 2) | (combined == 3)).sum(axis=(0, 1))
    error_slices    = np.where(error_per_slice >= min_voxel)[0]

    out_dir.mkdir(parents=True, exist_ok=True)

    if len(error_slices) == 0:
        # ── Clean case：展示代表切片 ────────────────────────────────────────
        tp_per_slice = (combined == 1).sum(axis=(0, 1))
        gt_per_slice = gt_tumor.sum(axis=(0, 1))

        if gt_per_slice.sum() > 0:
            # 有肿瘤但全部正确，取 TP 最多的几张
            top_zs = np.argsort(tp_per_slice)[::-1][:clean_slices]
            label  = "✅无误差-完美分割"
        else:
            # 无肿瘤且正确（TN）
            liver_mask = (gt >= 1).sum(axis=(0, 1))
            top_zs = np.argsort(liver_mask)[::-1][:clean_slices]
            label  = "✅无误差-无肿瘤TN"

        active_slices = np.sort(top_zs)
        is_clean = True
    else:
        active_slices = error_slices
        label = ""
        is_clean = False

    for z in active_slices:
        ct_sl   = ct[:, :, z].T
        tp_mask = (combined[:, :, z] == 1).T
        fp_mask = (combined[:, :, z] == 2).T
        fn_mask = (combined[:, :, z] == 3).T
        gt_vox  = int(tp_mask.sum()) + int(fn_mask.sum())

        overlay = np.zeros((*ct_sl.shape, 4), dtype=float)
        overlay[tp_mask] = [0, 1, 0, 0.18]
        overlay[fp_mask] = [1, 0, 0, 0.75]
        overlay[fn_mask] = [0, 0, 1, 0.70]

        # 病灶 bounding box 放大图
        lesion_mask = tp_mask | fp_mask | fn_mask
        has_zoom = lesion_mask.any()
        if has_zoom:
            rows = np.where(lesion_mask.any(axis=1))[0]
            cols = np.where(lesion_mask.any(axis=0))[0]
            H, W = ct_sl.shape
            pad = 32
            r0, r1 = max(0, rows[0]-pad), min(H-1, rows[-1]+pad)
            c0, c1 = max(0, cols[0]-pad), min(W-1, cols[-1]+pad)
            ct_zoom = ct_sl[r0:r1+1, c0:c1+1]
            ov_zoom = overlay[r0:r1+1, c0:c1+1]

        fig = plt.figure(figsize=(16, 5))
        gs  = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.6], wspace=0.02)

        clean_tag = f"  [{label}]" if is_clean else ""
        _fp = dict(fontproperties=_cn_prop) if _cn_prop else {}
        _leg_kw = dict(prop=_cn_prop, title_fontproperties=_cn_prop) if _cn_prop else dict(fontsize=8)

        ax_ct = fig.add_subplot(gs[0])
        ax_ct.imshow(ct_sl, cmap="gray", vmin=-150, vmax=250, origin="lower")
        ax_ct.set_title(f"原始CT  z={z}  GT={gt_vox}体素{clean_tag}", fontsize=8, **_fp)
        ax_ct.axis("off")
        ax_ct.legend(
            handles=LEGEND_ENTRIES, loc="upper left",
            frameon=True, framealpha=0.75, edgecolor="#888",
            title="颜色说明", fontsize=7,
            handlelength=1.2, handleheight=1.2, borderpad=0.6, labelspacing=0.5,
            **_leg_kw,
        )

        ax_ov = fig.add_subplot(gs[1])
        ax_ov.imshow(ct_sl, cmap="gray", vmin=-150, vmax=250, origin="lower")
        ax_ov.imshow(overlay, origin="lower")
        ax_ov.set_title(
            f"预测结果  z={z}   TP={tp_mask.sum()}  FP={fp_mask.sum()}  FN={fn_mask.sum()}",
            fontsize=8, **_fp,
        )
        ax_ov.axis("off")

        ax_zm = fig.add_subplot(gs[2])
        if has_zoom:
            ax_zm.imshow(ct_zoom, cmap="gray", vmin=-150, vmax=250, origin="lower")
            ax_zm.imshow(ov_zoom, origin="lower")
            ax_zm.set_title(f"病灶放大 (±{pad}px)", fontsize=8, **_fp)
        ax_zm.axis("off")

        plt.savefig(out_dir / f"{case}_z{z}_full.png", dpi=150,
                    bbox_inches="tight", pad_inches=0.05)
        plt.close()

    return len(active_slices)


def run_inference(cases, img_dir, trainer, dataset, fold, checkpoint, out_dir):
    """对指定 case 列表运行 nnUNetv2_predict，预测保存到 out_dir。"""
    tmp_in = Path(tempfile.mkdtemp(prefix="fullviz_in_"))
    try:
        linked = 0
        for c in cases:
            src = img_dir / f"{c}_0000.nii.gz"
            if src.exists():
                (tmp_in / src.name).symlink_to(src)
                linked += 1
            else:
                print(f"  [WARN] 找不到 {src}")
        print(f"  已链接 {linked}/{len(cases)} 个 case 到临时输入目录")

        ret = subprocess.run([
            "nnUNetv2_predict",
            "-i",   str(tmp_in),
            "-o",   str(out_dir),
            "-d",   dataset,
            "-c",   "3d_fullres",
            "-tr",  trainer,
            "-f",   str(fold),
            "-chk", checkpoint,
        ])
        return ret.returncode == 0
    finally:
        shutil.rmtree(tmp_in, ignore_errors=True)


def process_split(split_name: str, cases: list, pred_dir: Path,
                  gt_dir: Path, img_dir: Path, viz_dir: Path,
                  min_voxel: int, clean_slices: int):
    """为指定 split 的所有 case 生成 viz，输出到 viz_dir。"""
    if viz_dir.exists():
        shutil.rmtree(viz_dir)
        print(f"  已清空旧 {viz_dir.name}/")
    viz_dir.mkdir(parents=True, exist_ok=True)

    total_png = 0
    clean_count = 0
    error_count = 0

    for case in sorted(cases):
        pred_path = pred_dir / f"{case}.nii.gz"
        ct_path   = img_dir  / f"{case}_0000.nii.gz"
        gt_path   = gt_dir   / f"{case}.nii.gz"

        if not pred_path.exists():
            print(f"  [WARN] {case}: 预测文件不存在，跳过")
            continue
        if not ct_path.exists() or not gt_path.exists():
            print(f"  [WARN] {case}: CT 或 GT 不存在，跳过")
            continue

        case_dir = viz_dir / case
        n = gen_case_viz(case, ct_path, gt_path, pred_path,
                         case_dir, min_voxel, clean_slices)
        total_png += n

        # 判断是否有错误
        gt   = nib.load(gt_path).get_fdata()
        pred = nib.load(pred_path).get_fdata()
        has_error = (((gt == 2) != (pred == 2)).sum() > 0) or (((gt != 2) & (pred == 2)).sum() > 0)
        if has_error:
            error_count += 1
        else:
            clean_count += 1

    print(f"  [{split_name}] {len(cases)} cases: {error_count} 有误差, {clean_count} 无误差 → {total_png} 张 PNG")
    return total_png


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-train",    action="store_true")
    parser.add_argument("--skip-val",      action="store_true")
    parser.add_argument("--skip-test",     action="store_true")
    parser.add_argument("--min-voxel",     type=int, default=1)
    parser.add_argument("--clean-slices",  type=int, default=3)
    parser.add_argument("--checkpoint",    default=CHECKPOINT)
    args = parser.parse_args()

    split = json.loads(SPLIT_FILE.read_text())
    train_cases = split["train"]["cases"]
    val_cases   = split["val"]["cases"]
    test_cases  = split["test"]["cases"]

    fold_dir = (Path(os.environ["nnUNet_results"]) / DATASET /
                f"{TRAINER}__nnUNetPlans__3d_fullres" / f"fold_{FOLD}")
    print(f"Trainer: {TRAINER}")
    print(f"Fold dir: {fold_dir}")

    grand_total = 0

    # ── TEST ─────────────────────────────────────────────────────────────────
    if not args.skip_test:
        test_viz_dir = fold_dir / "test_viz"
        # 检查 test_prediction/ 是否有可用的 nii.gz（可能已被之前的脚本删除）
        test_pred_dir = fold_dir / "test_prediction"
        test_niis = list(test_pred_dir.glob("*.nii.gz")) if test_pred_dir.exists() else []
        if len(test_niis) >= len(test_cases):
            print(f"\n[test] 使用已有 test_prediction/ 预测（{len(test_niis)} 个文件）")
            n = process_split("test", test_cases, test_pred_dir,
                              GT_DIR, IMG_DIR, test_viz_dir,
                              args.min_voxel, args.clean_slices)
            grand_total += n
        else:
            # nii.gz 已被删除，重新推理
            tmp_pred_dir = Path(tempfile.mkdtemp(prefix="fullviz_test_"))
            print(f"\n[test] test_prediction/ 内 nii.gz 已被删除，重新推理 {len(test_cases)} 个 test case ...")
            print(f"  临时预测目录: {tmp_pred_dir}")
            try:
                ok = run_inference(
                    cases=test_cases,
                    img_dir=IMG_DIR,
                    trainer=TRAINER,
                    dataset=DATASET,
                    fold=FOLD,
                    checkpoint=args.checkpoint,
                    out_dir=tmp_pred_dir,
                )
                if not ok:
                    print("[ERROR] test 推理失败，跳过 test_viz")
                else:
                    n = process_split("test", test_cases, tmp_pred_dir,
                                      GT_DIR, IMG_DIR, test_viz_dir,
                                      args.min_voxel, args.clean_slices)
                    grand_total += n
            finally:
                shutil.rmtree(tmp_pred_dir, ignore_errors=True)
                print(f"  临时预测目录已清理: {tmp_pred_dir}")

    # ── VAL ──────────────────────────────────────────────────────────────────
    if not args.skip_val:
        val_pred_dir = fold_dir / "validation"
        val_viz_dir  = fold_dir / "val_viz"
        print(f"\n[val] 处理 {len(val_cases)} 个验证 case ...")
        if not val_pred_dir.exists():
            print(f"  [WARN] validation/ 不存在，跳过 val_viz")
        else:
            n = process_split("val", val_cases, val_pred_dir,
                              GT_DIR, IMG_DIR, val_viz_dir,
                              args.min_voxel, args.clean_slices)
            grand_total += n

    # ── TRAIN ────────────────────────────────────────────────────────────────
    if not args.skip_train:
        train_viz_dir = fold_dir / "train_viz"
        tmp_pred_dir  = Path(tempfile.mkdtemp(prefix="fullviz_train_"))
        print(f"\n[train] 对全量 {len(train_cases)} 个训练 case 推理 ...")
        print(f"  临时预测目录: {tmp_pred_dir}")
        try:
            ok = run_inference(
                cases=train_cases,
                img_dir=IMG_DIR,
                trainer=TRAINER,
                dataset=DATASET,
                fold=FOLD,
                checkpoint=args.checkpoint,
                out_dir=tmp_pred_dir,
            )
            if not ok:
                print("[ERROR] train 推理失败，跳过 train_viz")
            else:
                n = process_split("train", train_cases, tmp_pred_dir,
                                  GT_DIR, IMG_DIR, train_viz_dir,
                                  args.min_voxel, args.clean_slices)
                grand_total += n
        finally:
            shutil.rmtree(tmp_pred_dir, ignore_errors=True)
            print(f"  临时预测目录已清理: {tmp_pred_dir}")

    print("\n" + "=" * 60)
    print(f"全部完成！共生成 {grand_total} 张 PNG")
    for name in ("train_viz", "val_viz", "test_viz"):
        d = fold_dir / name
        if d.exists():
            n_cases = sum(1 for p in d.iterdir() if p.is_dir())
            n_pngs  = sum(1 for _ in d.rglob("*.png"))
            print(f"  {name}/: {n_cases} cases, {n_pngs} PNG")
    print("=" * 60)


if __name__ == "__main__":
    main()
