"""
对指定实验的测试集重新推理并生成 viz PNG（应用最新配色/字体修复）。

用法（在 /home/PuMengYu/nnUNet 目录下）：
    python pumengyu/tools/regen_test_viz.py --trainer nnUNetTrainer_MLAUNet_MoE
    python pumengyu/tools/regen_test_viz.py --trainer nnUNetTrainer_MLAUNet_MoE_SizeOversampleV2
    python pumengyu/tools/regen_test_viz.py --trainer nnUNetTrainer_SizeOversampleV2

可选参数：
    --checkpoint  checkpoint_best.pth（默认）或 checkpoint_final.pth
    --dataset     Dataset003_Liver（默认）
    --fold        0（默认）
"""

import argparse, json, os, shutil, sys, tempfile
from pathlib import Path

os.environ.setdefault("nnUNet_raw",          "/home/PuMengYu/nnUNet_workspace/raw")
os.environ.setdefault("nnUNet_preprocessed", "/home/PuMengYu/nnUNet_workspace/preprocessed")
os.environ.setdefault("nnUNet_results",      "/home/PuMengYu/nnUNet_workspace/results_v2")

NNUNET_ROOT = "/home/PuMengYu/nnUNet"
if NNUNET_ROOT not in sys.path:
    sys.path.insert(0, NNUNET_ROOT)


def get_test_cases(dataset: str, fold: int) -> list[str]:
    pre_dir = Path(os.environ["nnUNet_preprocessed"]) / dataset
    sp = json.loads((pre_dir / "splits_final.json").read_text())
    all_cases = {f.name.replace(".nii.gz", "") for f in
                 (pre_dir / "gt_segmentations").glob("*.nii.gz")}
    train_val = set(sp[fold]["train"]) | set(sp[fold]["val"])
    return sorted(all_cases - train_val)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainer",    required=True)
    parser.add_argument("--dataset",    default="Dataset003_Liver")
    parser.add_argument("--fold",       type=int, default=0)
    parser.add_argument("--checkpoint", default="checkpoint_best.pth")
    args = parser.parse_args()

    results_root = Path(os.environ["nnUNet_results"])
    raw_root     = Path(os.environ["nnUNet_raw"])
    pre_root     = Path(os.environ["nnUNet_preprocessed"])

    fold_dir = (results_root / args.dataset /
                f"{args.trainer}__nnUNetPlans__3d_fullres" /
                f"fold_{args.fold}")
    img_dir  = raw_root / args.dataset / "imagesTr"
    gt_dir   = pre_root / args.dataset / "gt_segmentations"
    out_viz  = fold_dir / "test_viz"

    if not fold_dir.exists():
        print(f"[ERROR] 找不到 fold 目录: {fold_dir}")
        sys.exit(1)

    test_cases = get_test_cases(args.dataset, args.fold)
    print(f"测试集 {len(test_cases)} cases: {test_cases[:4]} ...")

    # 创建临时目录存放推理结果
    tmp_dir = Path(tempfile.mkdtemp(prefix="regen_viz_"))
    print(f"\n[1/2] 推理 → {tmp_dir}")

    # 只对测试 case 推理：先把测试图像软链或复制到临时输入目录
    tmp_input = tmp_dir / "input"
    tmp_input.mkdir()
    for case in test_cases:
        src = img_dir / f"{case}_0000.nii.gz"
        if src.exists():
            (tmp_input / src.name).symlink_to(src)
        else:
            print(f"  [WARN] 找不到 {src}，跳过")

    tmp_pred = tmp_dir / "pred"
    tmp_pred.mkdir()

    import subprocess
    ret = subprocess.run([
        "nnUNetv2_predict",
        "-i",   str(tmp_input),
        "-o",   str(tmp_pred),
        "-d",   args.dataset,
        "-c",   "3d_fullres",
        "-tr",  args.trainer,
        "-f",   str(args.fold),
        "-chk", args.checkpoint,
    ])
    if ret.returncode != 0:
        print("[ERROR] 推理失败")
        shutil.rmtree(tmp_dir)
        sys.exit(1)

    # 删除已有 viz，重新生成
    if out_viz.exists():
        shutil.rmtree(out_viz)
        print(f"  已删除旧 viz: {out_viz}")

    print(f"\n[2/2] 生成 viz → {out_viz}")

    from pumengyu.mixins import generate_viz_and_cleanup
    generate_viz_and_cleanup(
        pred_folder  = tmp_pred,
        out_viz_dir  = out_viz,
        gt_dir       = gt_dir,
        img_dir      = img_dir,
        delete_nii   = False,   # 不删，脚本结束后统一清理临时目录
        min_voxel    = 1,
        log_fn       = None,
    )

    shutil.rmtree(tmp_dir)
    print(f"\n完成！viz 保存到: {out_viz}")
    n_cases = sum(1 for p in out_viz.iterdir() if p.is_dir()) if out_viz.exists() else 0
    n_pngs  = sum(1 for p in out_viz.rglob("*.png")) if out_viz.exists() else 0
    print(f"  {n_cases} 个 case，共 {n_pngs} 张 PNG")


if __name__ == "__main__":
    main()
