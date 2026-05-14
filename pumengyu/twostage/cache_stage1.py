"""
Stage1 离线预处理：批量跑 5 折 Stage1 推理，结果持久化到 cache/stage1_pred/
后续 eval.py 直接从缓存读，跳过 Stage1，节省时间。

用法：
  python pumengyu/twostage/cache_stage1.py               # 全部 5 折
  python pumengyu/twostage/cache_stage1.py --folds 0 1   # 指定折
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


def make_predictor(model_folder: str, fold: int,
                   checkpoint: str, device: str) -> nnUNetPredictor:
    pred = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=torch.device(device),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True,
    )
    pred.initialize_from_trained_model_folder(
        model_folder,
        use_folds=(fold,),
        checkpoint_name=checkpoint,
    )
    return pred


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--folds",          type=int, nargs="+", default=[0, 1, 2, 3, 4])
    pa.add_argument("--workspace",      default="/home/PuMengYu/nnUNet_workspace")
    pa.add_argument("--stage1_trainer", default="nnUNetTrainer")
    pa.add_argument("--checkpoint",     default="checkpoint_best.pth")
    pa.add_argument("--device",         default="cuda")
    args = pa.parse_args()

    ws      = Path(args.workspace)
    results = ws / "results"
    preproc = ws / "preprocessed"
    raw     = ws / "raw"

    stage1_folder = str(results / "Dataset003_Liver" /
                        f"{args.stage1_trainer}__nnUNetPlans__3d_fullres")
    ct_dir      = raw     / "Dataset003_Liver" / "imagesTr"
    splits_file = preproc / "Dataset003_Liver" / "splits_final.json"
    cache_dir   = ws / "cache" / "stage1_pred"
    cache_dir.mkdir(parents=True, exist_ok=True)

    if not ct_dir.exists():
        raise FileNotFoundError(
            f"找不到原始 CT 目录：{ct_dir}\n请先运行 extract_images.sh 解压 tar 包")

    splits = json.loads(splits_file.read_text())

    for fold in args.folds:
        val_cases = sorted(splits[fold]["val"])
        missing   = [c for c in val_cases if not (cache_dir / f"{c}.nii.gz").exists()]

        print(f"\n===== Fold {fold}：{len(val_cases)} 个 case，"
              f"缓存缺失 {len(missing)} 个 =====")

        if not missing:
            print("  全部命中缓存，跳过")
            continue

        s1_inputs  = [[str(ct_dir / f"{c}_0000.nii.gz")] for c in missing]
        s1_outputs = [str(cache_dir / f"{c}.nii.gz")     for c in missing]
#make_predictor是完整的推理流水线
        pred = make_predictor(stage1_folder, fold, args.checkpoint, args.device)
        pred.predict_from_files(
            s1_inputs, s1_outputs,
            save_probabilities=False, overwrite=False,
            num_processes_preprocessing=2,
            num_processes_segmentation_export=2,
        )
        del pred  # 每折推理完释放显存

        print(f"  Fold {fold} 完成，缓存写入 {len(missing)} 个 case")

    print(f"\n全部完成，缓存目录：{cache_dir}")


if __name__ == "__main__":
    main()
