"""
Step 2: 根据 bbox_crops.json 裁剪原图和标签，生成 Dataset004_LiverTumor。

bbox 来源：肝脏模型预测（pred only）+ 20mm padding，与测试时完全一致。

标签映射：
  0 (background) → 0
  1 (liver)      → 0  ← 只保留肿瘤，肝脏归入背景
  2 (tumor)      → 1

注意：
  - 坐标是 nibabel 读出的 (X,Y,Z) 轴顺序
  - affine 更新：origin 随 crop 偏移
  - splits_final.json 直接复制，fold 定义不变

运行：
  python pumengyu/scripts/create_dataset004.py
"""

import os
import json
import shutil
import glob
import numpy as np
import nibabel as nib

# ── 路径配置 ────────────────────────────────────────────────────────────────
SRC_RAW      = "/home/PuMengYu/nnUNet_workspace/raw/Dataset003_Liver"
DST_RAW      = "/home/PuMengYu/nnUNet_workspace/raw/Dataset004_LiverTumor"
BBOX_JSON    = "/home/PuMengYu/nnUNet/pumengyu/scripts/bbox_crops.json"
SRC_SPLITS   = "/home/PuMengYu/nnUNet_workspace/preprocessed/Dataset003_Liver/splits_final.json"
DST_PREPROCESSED = "/home/PuMengYu/nnUNet_workspace/preprocessed/Dataset004_LiverTumor"

# ── 工具函数 ────────────────────────────────────────────────────────────────

def crop_nifti(nib_img: nib.Nifti1Image, bbox: dict) -> nib.Nifti1Image:
    """
    按 bbox 裁剪 nifti image，并正确更新 affine origin。
    bbox 格式: {"x": [lo,hi], "y": [lo,hi], "z": [lo,hi]}
    nibabel 数据排列: data[x, y, z]
    """
    x0, x1 = bbox["x"]
    y0, y1 = bbox["y"]
    z0, z1 = bbox["z"]

    data = np.asarray(nib_img.dataobj)
    cropped = data[x0:x1+1, y0:y1+1, z0:z1+1]

    # 更新 affine：origin 沿各轴偏移
    new_affine = nib_img.affine.copy()
    offset = nib_img.affine[:3, :3] @ np.array([x0, y0, z0], dtype=float)
    new_affine[:3, 3] += offset

    return nib.Nifti1Image(cropped, new_affine, nib_img.header)


def remap_labels(seg_arr: np.ndarray) -> np.ndarray:
    """liver(1)→0, tumor(2)→1, background(0)→0"""
    out = np.zeros_like(seg_arr, dtype=np.uint8)
    out[seg_arr == 2] = 1
    return out


# ── 主流程 ──────────────────────────────────────────────────────────────────

def main():
    with open(BBOX_JSON) as f:
        bboxes = json.load(f)
    print(f"读取 bbox: {len(bboxes)} 个 case")

    # 创建目录
    images_tr = os.path.join(DST_RAW, "imagesTr")
    labels_tr = os.path.join(DST_RAW, "labelsTr")
    os.makedirs(images_tr, exist_ok=True)
    os.makedirs(labels_tr, exist_ok=True)

    src_images = os.path.join(SRC_RAW, "imagesTr")
    src_labels = os.path.join(SRC_RAW, "labelsTr")

    shape_log = []

    for case_id, bbox in sorted(bboxes.items()):
        img_path = os.path.join(src_images, f"{case_id}_0000.nii.gz")
        lbl_path = os.path.join(src_labels, f"{case_id}.nii.gz")

        if not os.path.exists(img_path) or not os.path.exists(lbl_path):
            print(f"  [SKIP] 找不到原始文件: {case_id}")
            continue

        img_nib = nib.load(img_path)
        lbl_nib = nib.load(lbl_path)

        # 裁剪
        img_cropped = crop_nifti(img_nib, bbox)
        lbl_cropped = crop_nifti(lbl_nib, bbox)

        # label 重映射
        lbl_arr = np.asarray(lbl_cropped.dataobj, dtype=np.uint8)
        lbl_remapped = remap_labels(lbl_arr)
        lbl_out = nib.Nifti1Image(lbl_remapped, lbl_cropped.affine, lbl_cropped.header)

        # 保存
        dst_img = os.path.join(images_tr, f"{case_id}_0000.nii.gz")
        dst_lbl = os.path.join(labels_tr, f"{case_id}.nii.gz")
        nib.save(img_cropped, dst_img)
        nib.save(lbl_out, dst_lbl)

        orig_shape = bbox["shape_xyz"]
        crop_shape = list(img_cropped.shape)
        ratio = np.prod(crop_shape) / np.prod(orig_shape)
        shape_log.append(ratio)
        print(f"  {case_id}: {orig_shape} → {crop_shape}  ({ratio*100:.1f}%)")

    print(f"\n平均体积压缩至原来的 {np.mean(shape_log)*100:.1f}%")

    # ── dataset.json ────────────────────────────────────────────────────────
    dataset_json = {
        "name": "LiverTumor",
        "description": "Liver tumor segmentation on liver-cropped CT (from Dataset003)",
        "reference": "Derived from MSD Task03_Liver",
        "licence": "CC-BY-SA 4.0",
        "tensorImageSize": "3D",
        "labels": {
            "background": 0,
            "tumor": 1
        },
        "numTraining": len(bboxes),
        "file_ending": ".nii.gz",
        "channel_names": {"0": "CT"}
    }
    with open(os.path.join(DST_RAW, "dataset.json"), "w") as f:
        json.dump(dataset_json, f, indent=2)

    # ── 复制 splits_final.json，保持 fold 一致 ─────────────────────────────
    os.makedirs(DST_PREPROCESSED, exist_ok=True)
    dst_splits = os.path.join(DST_PREPROCESSED, "splits_final.json")
    shutil.copy(SRC_SPLITS, dst_splits)
    print(f"\nsplits_final.json 已复制 → {dst_splits}")
    print(f"Dataset004 已生成 → {DST_RAW}")
    print("\n下一步：")
    print("  nnUNetv2_plan_and_preprocess -d 4 --verify_dataset_integrity")
    print("  nnUNetv2_train 4 3d_fullres 0  (fold 0~4)")


if __name__ == "__main__":
    main()
