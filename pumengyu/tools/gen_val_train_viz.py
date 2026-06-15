"""
为 val 和 train 反常 case 生成可视化 PNG。

- val_viz:   直接用 fold_0/validation/ 的已有预测，不重新推理
- train_viz: 只对「反常A/B」case 推理（14个），避免跑全量 92 个训练 case

用法（在 /home/PuMengYu/nnUNet 目录）：
    conda run -n medseg python pumengyu/tools/gen_val_train_viz.py
    conda run -n medseg python pumengyu/tools/gen_val_train_viz.py --skip-train
    conda run -n medseg python pumengyu/tools/gen_val_train_viz.py --skip-val

可选参数：
    --trainer     nnUNetTrainer_MLAUNet_MoE_SizeOversampleV4 (默认)
    --dataset     Dataset003_Liver (默认)
    --fold        0 (默认)
    --checkpoint  checkpoint_best.pth (默认)
    --skip-val    跳过 val_viz 生成
    --skip-train  跳过 train_viz 生成
    --min-voxel   FP/FN 最小体素数才出图，默认 1（显示所有误差）
"""

import argparse, os, shutil, subprocess, sys, tempfile
from pathlib import Path

os.environ.setdefault("nnUNet_raw",          "/home/PuMengYu/nnUNet_workspace/raw")
os.environ.setdefault("nnUNet_preprocessed", "/home/PuMengYu/nnUNet_workspace/preprocessed")
os.environ.setdefault("nnUNet_results",      "/home/PuMengYu/nnUNet_workspace/results_v2")

NNUNET_ROOT = "/home/PuMengYu/nnUNet"
if NNUNET_ROOT not in sys.path:
    sys.path.insert(0, NNUNET_ROOT)

# 反常 A: 有阴影但无肿瘤（易假阳性）
ANOMALY_A_TRAIN = [
    "liver_105", "liver_114", "liver_32",
    "liver_38",  "liver_47",  "liver_87",
]

# 反常 B: 无阴影但有肿瘤（易假阴性）
ANOMALY_B_TRAIN = [
    "liver_113", "liver_120", "liver_122",
    "liver_20",  "liver_25",  "liver_55",
    "liver_61",  "liver_67",
]

ALL_ANOMALY_TRAIN = ANOMALY_A_TRAIN + ANOMALY_B_TRAIN


def run_inference(cases, img_dir, fold_dir, trainer, dataset, fold, checkpoint, out_dir):
    tmp_in = Path(tempfile.mkdtemp(prefix="trainviz_in_"))
    try:
        for c in cases:
            src = img_dir / f"{c}_0000.nii.gz"
            if src.exists():
                (tmp_in / src.name).symlink_to(src)
            else:
                print(f"  [WARN] 找不到 {src}，跳过")

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
        if ret.returncode != 0:
            print("[ERROR] 推理失败")
            return False
        return True
    finally:
        shutil.rmtree(tmp_in, ignore_errors=True)


def gen_viz(pred_folder, gt_dir, img_dir, out_viz_dir, min_voxel, delete_nii):
    from pumengyu.mixins import _gen_viz_pngs_and_cleanup
    _gen_viz_pngs_and_cleanup(
        pred_folder=pred_folder,
        gt_dir=gt_dir,
        img_dir=img_dir,
        out_viz_dir=out_viz_dir,
        min_voxel=min_voxel,
        delete_nii=delete_nii,
    )
    n_cases = sum(1 for p in out_viz_dir.iterdir() if p.is_dir()) if out_viz_dir.exists() else 0
    n_pngs  = sum(1 for p in out_viz_dir.rglob("*.png")) if out_viz_dir.exists() else 0
    print(f"  → {n_cases} 个 case，共 {n_pngs} 张 PNG  [{out_viz_dir}]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainer",     default="nnUNetTrainer_MLAUNet_MoE_SizeOversampleV4")
    parser.add_argument("--dataset",     default="Dataset003_Liver")
    parser.add_argument("--fold",        type=int, default=0)
    parser.add_argument("--checkpoint",  default="checkpoint_best.pth")
    parser.add_argument("--skip-val",    action="store_true")
    parser.add_argument("--skip-train",  action="store_true")
    parser.add_argument("--min-voxel",   type=int, default=1)
    args = parser.parse_args()

    results_root = Path(os.environ["nnUNet_results"])
    raw_root     = Path(os.environ["nnUNet_raw"])
    pre_root     = Path(os.environ["nnUNet_preprocessed"])

    fold_dir = (results_root / args.dataset /
                f"{args.trainer}__nnUNetPlans__3d_fullres" /
                f"fold_{args.fold}")
    img_dir  = raw_root / args.dataset / "imagesTr"
    gt_dir   = pre_root / args.dataset / "gt_segmentations"

    if not fold_dir.exists():
        print(f"[ERROR] fold 目录不存在: {fold_dir}")
        sys.exit(1)

    # ── 1. val_viz ──────────────────────────────────────────────────────────
    if not args.skip_val:
        print("\n[val_viz] 使用已有 validation/ 预测 ...")
        val_pred_dir = fold_dir / "validation"
        val_viz_dir  = fold_dir / "val_viz"

        if not val_pred_dir.exists():
            print(f"  [WARN] validation/ 不存在，跳过 val_viz")
        else:
            val_nii = list(val_pred_dir.glob("*.nii.gz"))
            print(f"  找到 {len(val_nii)} 个 val 预测")
            if val_viz_dir.exists():
                shutil.rmtree(val_viz_dir)
                print(f"  已清空旧 val_viz")
            val_viz_dir.mkdir(parents=True, exist_ok=True)

            # delete_nii=False: 保留 validation/ 里的原始预测，不删除
            gen_viz(val_pred_dir, gt_dir, img_dir, val_viz_dir,
                    min_voxel=args.min_voxel, delete_nii=False)

    # ── 2. train_viz (只对反常 case) ────────────────────────────────────────
    if not args.skip_train:
        print(f"\n[train_viz] 对 {len(ALL_ANOMALY_TRAIN)} 个反常 train case 推理 ...")
        print(f"  反常A (有阴影无肿瘤): {ANOMALY_A_TRAIN}")
        print(f"  反常B (无阴影有肿瘤): {ANOMALY_B_TRAIN}")

        train_viz_dir = fold_dir / "train_viz"
        tmp_pred_dir  = Path(tempfile.mkdtemp(prefix="trainviz_pred_"))

        try:
            ok = run_inference(
                cases=ALL_ANOMALY_TRAIN,
                img_dir=img_dir,
                fold_dir=fold_dir,
                trainer=args.trainer,
                dataset=args.dataset,
                fold=args.fold,
                checkpoint=args.checkpoint,
                out_dir=tmp_pred_dir,
            )
            if not ok:
                sys.exit(1)

            if train_viz_dir.exists():
                shutil.rmtree(train_viz_dir)
                print(f"  已清空旧 train_viz")
            train_viz_dir.mkdir(parents=True, exist_ok=True)

            # delete_nii=True: 临时预测用完即删（已在 tmp_pred_dir，最终也会清理）
            gen_viz(tmp_pred_dir, gt_dir, img_dir, train_viz_dir,
                    min_voxel=args.min_voxel, delete_nii=True)
        finally:
            shutil.rmtree(tmp_pred_dir, ignore_errors=True)

    print("\n全部完成！")
    print(f"  val_viz:   {fold_dir}/val_viz")
    print(f"  train_viz: {fold_dir}/train_viz")


if __name__ == "__main__":
    main()
