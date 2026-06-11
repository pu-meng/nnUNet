"""
对 SizeOversampleV2 全量 131 个 case + 26 个测试 case 推理，直接生成 PNG 可视化，不保留 nii.gz。

输出结构：
  analysis/SizeOversampleV2/viz/
    train/liver_XXX/liver_XXX_z{z}_full.png   (92 个训练 case)
    val/  liver_XXX/liver_XXX_z{z}_full.png   (13 个验证 case)
    test/ liver_XXX/liver_XXX_z{z}_full.png   (26 个测试 case，预测已存在)

用法：
    cd /home/PuMengYu/nnUNet
    CUDA_VISIBLE_DEVICES=0 python pumengyu/tools/run_full131_viz.py
"""

import os, sys, shutil, tempfile, json, subprocess
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path

# ── 环境变量 ─────────────────────────────────────────────────────────────────
os.environ.setdefault("nnUNet_raw",          "/home/PuMengYu/nnUNet_workspace/raw")
os.environ.setdefault("nnUNet_preprocessed", "/home/PuMengYu/nnUNet_workspace/preprocessed")
os.environ.setdefault("nnUNet_results",      "/home/PuMengYu/nnUNet_workspace/results_v2")

NNUNET_ROOT = "/home/PuMengYu/nnUNet"
if NNUNET_ROOT not in sys.path:
    sys.path.insert(0, NNUNET_ROOT)

# ── 路径配置 ─────────────────────────────────────────────────────────────────
INPUT_DIR    = Path("/home/PuMengYu/nnUNet_workspace/raw/Dataset003_Liver/imagesTr")
GT_DIR       = Path("/home/PuMengYu/nnUNet_workspace/preprocessed/Dataset003_Liver/gt_segmentations")
SPLIT_FILE   = Path("/home/PuMengYu/nnUNet_workspace/preprocessed/Dataset003_Liver/splits_final.json")
TEST_PRED_DIR = Path("/home/PuMengYu/nnUNet_workspace/results_v2/Dataset003_Liver"
                     "/nnUNetTrainer_SizeOversampleV2__nnUNetPlans__3d_fullres"
                     "/fold_0/test_prediction")
OUT_BASE     = Path("/home/PuMengYu/nnUNet_workspace/analysis/SizeOversampleV2/viz")

MIN_VOXEL  = 20   # 低于此体素数的切片跳过

# ── 读取 train/val 划分 ───────────────────────────────────────────────────────
splits    = json.loads(SPLIT_FILE.read_text())[0]
VAL_CASES = set(splits["val"])
TEST_CASES = {p.stem.replace(".nii", "") for p in TEST_PRED_DIR.glob("liver_*.nii.gz")}

# ── 图例（所有图复用同一份）──────────────────────────────────────────────────
LEGEND = [
    Patch(facecolor="yellow",    alpha=0.85, label="GT肿瘤  实际肿瘤范围"),
    Patch(facecolor="green",     alpha=0.7,  label="TP      预测=肿瘤  GT=肿瘤  (正确)"),
    Patch(facecolor="red",       alpha=0.7,  label="FP      预测=肿瘤  GT=背景  (误报)"),
    Patch(facecolor="royalblue", alpha=0.7,  label="FN      预测=背景  GT=肿瘤  (漏检)"),
]


def gen_png(case: str, pred_path: Path, out_dir: Path):
    ct_path = INPUT_DIR / f"{case}_0000.nii.gz"
    gt_path = GT_DIR / f"{case}.nii.gz"

    ct       = nib.load(ct_path).get_fdata()
    gt       = nib.load(gt_path).get_fdata()
    pred     = nib.load(pred_path).get_fdata()

    gt_tumor   = (gt   == 2).astype(np.uint8)
    pred_tumor = (pred == 2).astype(np.uint8)

    # combined: 1=TP 2=FP 3=FN
    combined = np.zeros_like(gt, dtype=np.uint8)
    combined[(gt_tumor == 1) & (pred_tumor == 1)] = 1
    combined[(gt_tumor == 0) & (pred_tumor == 1)] = 2
    combined[(gt_tumor == 1) & (pred_tumor == 0)] = 3

    total_per_slice = ((combined == 1) | (combined == 2) | (combined == 3)).sum(axis=(0, 1))
    active_slices   = np.where(total_per_slice >= MIN_VOXEL)[0]

    if len(active_slices) == 0:
        print(f"    无满足 MIN_VOXEL={MIN_VOXEL} 的切片，跳过")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)

    for z in active_slices:
        ct_slice = ct[:, :, z].T
        gt_tumor_sl = (gt[:, :, z] == 2).T
        tp_mask = (combined[:, :, z] == 1).T
        fp_mask = (combined[:, :, z] == 2).T
        fn_mask = (combined[:, :, z] == 3).T
        tp_n, fp_n, fn_n = tp_mask.sum(), fp_mask.sum(), fn_mask.sum()

        overlay_gt = np.zeros((*ct_slice.shape, 4), dtype=float)
        overlay_gt[gt_tumor_sl] = [1, 1, 0, 0.9]

        overlay_pred = np.zeros((*ct_slice.shape, 4), dtype=float)
        overlay_pred[gt_tumor_sl] = [1, 1, 0, 0.6]
        overlay_pred[tp_mask]     = [0,   1,   0, 0.55]
        overlay_pred[fp_mask]     = [1,   0,   0, 0.55]
        overlay_pred[fn_mask]     = [0, 0.4,   1, 0.65]

        fig = plt.figure(figsize=(16, 6))
        gs  = fig.add_gridspec(1, 3, width_ratios=[0.32, 1, 1], wspace=0.05)

        ax_legend = fig.add_subplot(gs[0])
        ax_legend.axis("off")
        ax_legend.legend(
            handles=LEGEND, loc="center", fontsize=11,
            frameon=True, framealpha=0.9, edgecolor="#888",
            title="颜色说明", title_fontsize=12,
            handlelength=2, handleheight=1.6, borderpad=1.2, labelspacing=1.2,
        )

        ax_ct = fig.add_subplot(gs[1])
        ax_ct.imshow(ct_slice, cmap="gray", vmin=-150, vmax=250, origin="lower")
        ax_ct.imshow(overlay_gt, origin="lower")
        ax_ct.set_title(f"CT + GT肿瘤  z={z}  ({gt_tumor_sl.sum()}体素)", fontsize=12)
        ax_ct.axis("off")

        ax_ov = fig.add_subplot(gs[2])
        ax_ov.imshow(ct_slice, cmap="gray", vmin=-150, vmax=250, origin="lower")
        ax_ov.imshow(overlay_pred, origin="lower")
        ax_ov.set_title(f"预测结果  z={z}   TP={tp_n}  FP={fp_n}  FN={fn_n}", fontsize=12)
        ax_ov.axis("off")

        out_path = out_dir / f"{case}_z{z}_full.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()

    print(f"    → {len(active_slices)} 张 PNG 保存到 {out_dir}")
    return len(active_slices)


# ── 主流程 ────────────────────────────────────────────────────────────────────
def main():
    # 临时目录存放预测 nii.gz（用完即删）
    tmp_pred_dir = Path(tempfile.mkdtemp(prefix="full131_pred_"))
    print(f"临时预测目录: {tmp_pred_dir}")

    # 第一步：批量推理
    print("\n" + "=" * 60)
    print("[1/2] nnUNetv2_predict 推理 131 个 case...")
    print("=" * 60)
    subprocess.run([
        "nnUNetv2_predict",
        "-i",   str(INPUT_DIR),
        "-o",   str(tmp_pred_dir),
        "-d",   "Dataset003_Liver",
        "-c",   "3d_fullres",
        "-tr",  "nnUNetTrainer_SizeOversampleV2",
        "-f",   "0",
        "-chk", "checkpoint_best.pth",
    ], check=True)

    # 第二步：逐 case 生成 PNG，按 train/val 分目录
    print("\n" + "=" * 60)
    print("[2/2] 生成 PNG 可视化...")
    print("=" * 60)

    pred_files = sorted(tmp_pred_dir.glob("liver_*.nii.gz"))
    total_png  = 0

    for pred_path in pred_files:
        case = pred_path.stem.replace(".nii", "")   # liver_XXX
        split = "val" if case in VAL_CASES else "train"
        out_dir = OUT_BASE / split / case

        print(f"  [{split}] {case}")
        n = gen_png(case, pred_path, out_dir)
        total_png += n

    # 删除临时预测目录
    shutil.rmtree(tmp_pred_dir)
    print(f"\n临时 nii.gz 已清理: {tmp_pred_dir}")

    # 第三步：处理已有的 26 个测试 case
    print("\n" + "=" * 60)
    print(f"[3/3] 生成 26 个测试 case 的 PNG（使用已有预测）...")
    print("=" * 60)

    for pred_path in sorted(TEST_PRED_DIR.glob("liver_*.nii.gz")):
        case = pred_path.stem.replace(".nii", "")
        out_dir = OUT_BASE / "test" / case
        print(f"  [test] {case}")
        n = gen_png(case, pred_path, out_dir)
        total_png += n

    print("\n" + "=" * 60)
    print(f"全部完成！共生成 {total_png} 张 PNG")
    print(f"输出目录: {OUT_BASE}/")
    for split in ("train", "val", "test"):
        d = OUT_BASE / split
        n = sum(1 for p in d.glob("liver_*") if p.is_dir()) if d.exists() else 0
        print(f"  {split}/ : {n} 个 case")
    print("=" * 60)


if __name__ == "__main__":
    main()
