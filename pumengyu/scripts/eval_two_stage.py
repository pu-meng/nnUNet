"""
两阶段端到端评估 —— 输入原始 .nii.gz，输出 Dice。

pipeline（完全不用 GT，模拟真实推理）：
  原始 CT
    → Stage1 fold_k 肝脏模型  → liver mask
    → pred bbox + 20mm padding → crop CT
    → Stage2 fold_k 肿瘤模型  → tumor mask（裁剪空间）
    → 映射回原始空间
    → Dice(liver) + Dice(tumor)  vs  GT

交叉验证无泄露：
  fold_k 的验证 case 只用 fold_k 的肝脏/肿瘤模型推理

前提：
  - Dataset003 nnUNetTrainer 5折已训完
  - Dataset004 nnUNetTrainer 5折已训完

运行：
  python pumengyu/scripts/eval_two_stage.py [--fold 0]
"""

import os
import argparse
import json

import numpy as np
import nibabel as nib
import torch

LABELS_DIR      = "/home/PuMengYu/nnUNet_workspace/raw/Dataset003_Liver/labelsTr"
IMAGES_DIR      = "/home/PuMengYu/nnUNet_workspace/raw/Dataset003_Liver/imagesTr"
LIVER_MODEL_DIR = "/home/PuMengYu/nnUNet_workspace/results/Dataset003_Liver/nnUNetTrainer__nnUNetPlans__3d_fullres"
TUMOR_MODEL_DIR = "/home/PuMengYu/nnUNet_workspace/results/Dataset004_LiverTumor/nnUNetTrainer__nnUNetPlans__3d_fullres"
SPLITS_JSON     = "/home/PuMengYu/nnUNet_workspace/preprocessed/Dataset003_Liver/splits_final.json"
OUTPUT_JSON     = "/home/PuMengYu/nnUNet/pumengyu/scripts/eval_two_stage_results.json"
PADDING_MM      = 20.0


# ── 工具函数 ────────────────────────────────────────────────────────────────

def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    tp = int(np.sum((pred == 1) & (gt == 1)))
    fp = int(np.sum((pred == 1) & (gt == 0)))
    fn = int(np.sum((pred == 0) & (gt == 1)))
    denom = 2 * tp + fp + fn
    return float("nan") if denom == 0 else 2.0 * tp / denom


def bbox_from_mask(mask_xyz: np.ndarray, spacing_xyz: tuple, padding_mm: float) -> dict:
    """mask shape: (X,Y,Z)，返回 {x,y,z: [lo,hi]}"""
    idx = np.where(mask_xyz)
    if len(idx[0]) == 0:
        return {"x": [0, mask_xyz.shape[0]-1],
                "y": [0, mask_xyz.shape[1]-1],
                "z": [0, mask_xyz.shape[2]-1]}
    result = {}
    for ax, (indices, sp) in enumerate(zip(idx, spacing_xyz)):
        pad = int(np.ceil(padding_mm / sp))
        lo = max(0, int(indices.min()) - pad)
        hi = min(mask_xyz.shape[ax] - 1, int(indices.max()) + pad)
        result[["x", "y", "z"][ax]] = [lo, hi]
    return result


def load_predictor(model_dir: str, fold: int, device: torch.device):
    """加载 nnUNetPredictor，只使用指定的单折权重。"""
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,     # 关闭 TTA 以加快速度；开启可提升精度
        perform_everything_on_device=True,
        device=device,
        verbose=False,
    )
    predictor.initialize_from_trained_model_folder(
        model_dir,
        use_folds=(fold,),
        checkpoint_name="checkpoint_final.pth",
    )
    return predictor


def predict_nib(predictor, nib_img: nib.Nifti1Image) -> np.ndarray:
    """
    用 nnUNetPredictor 对一张 nibabel 图像做推理。

    nibabel 轴顺序: (X, Y, Z)
    nnUNet 期望:   (C, Z, Y, X) + spacing [sz, sy, sx]
    返回 numpy 预测结果，轴顺序 (X, Y, Z)。
    """
    data_xyz = np.asarray(nib_img.dataobj, dtype=np.float32)
    aff = nib_img.affine
    spacing_xyz = (abs(float(aff[0,0])), abs(float(aff[1,1])), abs(float(aff[2,2])))

    # nibabel (X,Y,Z) → nnUNet (Z,Y,X)，再加 channel 维
    data_czyx = data_xyz.transpose(2, 1, 0)[np.newaxis]        # (1, Z, Y, X)
    spacing_zyx = [spacing_xyz[2], spacing_xyz[1], spacing_xyz[0]]

    seg_zyx = predictor.predict_single_npy_array(
        data_czyx,
        {"spacing": spacing_zyx},
        segmentation_previous_stage=None,
        output_file_truncated=None,
        save_or_return_probabilities=False,
    )
    # (Z,Y,X) → (X,Y,Z)
    return seg_zyx.transpose(2, 1, 0)


def crop_nib(nib_img: nib.Nifti1Image, bbox: dict) -> nib.Nifti1Image:
    """按 bbox 裁剪 nibabel 图像，同时更新 affine。"""
    x0,x1 = bbox["x"]; y0,y1 = bbox["y"]; z0,z1 = bbox["z"]
    data = np.asarray(nib_img.dataobj)[x0:x1+1, y0:y1+1, z0:z1+1]
    aff = nib_img.affine.copy()
    aff[:3, 3] += aff[:3, :3] @ np.array([x0, y0, z0], dtype=float)
    return nib.Nifti1Image(data, aff, nib_img.header)


# ── 主流程 ──────────────────────────────────────────────────────────────────

def eval_fold(fold: int, liver_predictor, tumor_predictor, case_ids: list) -> list:
    results = []
    for case_id in case_ids:
        img_path = os.path.join(IMAGES_DIR, f"{case_id}_0000.nii.gz")
        lbl_path = os.path.join(LABELS_DIR, f"{case_id}.nii.gz")
        if not os.path.exists(img_path):
            print(f"  [SKIP] 找不到原图: {case_id}")
            continue

        ct_nib  = nib.load(img_path)
        gt_nib  = nib.load(lbl_path)
        gt_arr  = np.asarray(gt_nib.dataobj, dtype=np.uint8)
        aff     = ct_nib.affine
        spacing_xyz = (abs(float(aff[0,0])), abs(float(aff[1,1])), abs(float(aff[2,2])))

        # Stage 1：肝脏预测（在原始 CT 上）
        liver_pred_xyz = predict_nib(liver_predictor, ct_nib).astype(np.uint8)

        # bbox（只用肝脏预测，不用 GT）
        bbox = bbox_from_mask(liver_pred_xyz >= 1, spacing_xyz, PADDING_MM)

        # Stage 2：肿瘤预测（在 crop 后的 CT 上）
        ct_crop_nib   = crop_nib(ct_nib, bbox)
        tumor_pred_crop = predict_nib(tumor_predictor, ct_crop_nib).astype(np.uint8)

        # 映射回原始空间
        x0,x1 = bbox["x"]; y0,y1 = bbox["y"]; z0,z1 = bbox["z"]
        full_tumor_pred = np.zeros_like(gt_arr, dtype=np.uint8)
        full_tumor_pred[x0:x1+1, y0:y1+1, z0:z1+1] = (tumor_pred_crop == 1).astype(np.uint8)

        d_liver = dice((liver_pred_xyz >= 1).astype(np.uint8), (gt_arr >= 1).astype(np.uint8))
        d_tumor = dice(full_tumor_pred, (gt_arr == 2).astype(np.uint8))

        results.append({"case": case_id, "fold": fold, "dice_liver": d_liver, "dice_tumor": d_tumor})
        print(f"  {case_id}: liver={d_liver:.4f}  tumor={d_tumor:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=None,
                        help="只评估某一折（0-4）；默认全部 5 折")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    with open(SPLITS_JSON) as f:
        splits = json.load(f)

    folds_to_run = [args.fold] if args.fold is not None else list(range(5))
    all_results = []

    for fold in folds_to_run:
        val_cases = splits[fold]["val"]
        print(f"\n{'='*50}")
        print(f"Fold {fold}  ({len(val_cases)} cases)")
        print(f"{'='*50}")

        # 检查 tumor 模型是否存在
        tumor_ckpt = os.path.join(TUMOR_MODEL_DIR, f"fold_{fold}", "checkpoint_final.pth")
        if not os.path.exists(tumor_ckpt):
            print(f"  [SKIP] Dataset004 fold_{fold} 尚未训完，跳过")
            continue

        liver_predictor = load_predictor(LIVER_MODEL_DIR, fold, device)
        tumor_predictor = load_predictor(TUMOR_MODEL_DIR, fold, device)

        fold_results = eval_fold(fold, liver_predictor, tumor_predictor, val_cases)
        all_results.extend(fold_results)

        # 每折结束后清显存
        del liver_predictor, tumor_predictor
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if not all_results:
        print("\n无结果，请确认 Dataset004 已训完。")
        return

    liver_scores = [r["dice_liver"] for r in all_results if not np.isnan(r["dice_liver"])]
    tumor_scores = [r["dice_tumor"] for r in all_results if not np.isnan(r["dice_tumor"])]

    summary = {
        "n":               len(all_results),
        "dice_liver_mean": round(float(np.mean(liver_scores)), 4),
        "dice_liver_std":  round(float(np.std(liver_scores)),  4),
        "dice_tumor_mean": round(float(np.mean(tumor_scores)), 4),
        "dice_tumor_std":  round(float(np.std(tumor_scores)),  4),
        "mean_fg_dice":    round(float((np.mean(liver_scores) + np.mean(tumor_scores)) / 2), 4),
    }

    print(f"\n{'='*50}")
    print(f"总评估样本: {summary['n']}")
    print(f"Liver Dice : {summary['dice_liver_mean']:.4f} ± {summary['dice_liver_std']:.4f}")
    print(f"Tumor Dice : {summary['dice_tumor_mean']:.4f} ± {summary['dice_tumor_std']:.4f}")
    print(f"Mean Fg    : {summary['mean_fg_dice']:.4f}")

    with open(OUTPUT_JSON, "w") as f:
        json.dump({"summary": summary, "per_case": all_results}, f, indent=2)
    print(f"\n详细结果 → {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
