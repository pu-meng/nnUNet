"""
Step 1: 计算每个训练样本的肝脏 ROI bounding box，保存为 JSON。

逻辑：
  crop_mask = pred_liver（只用肝脏模型预测，不用 GT）
  bbox = 包围 crop_mask 的最小长方体 + padding_mm

训练和测试的 crop 策略完全一致（都只用 pred），分布对齐。

5折 cross-val 预测已经存在，直接使用，无 data leakage：
  fold_k 的验证集用 fold_k 的肝脏模型预测（训练时未见该样本）

输出：
  bbox_crops.json  —  每个 case 的 {x,y,z} 范围（原始空间体素坐标）

运行：
  python pumengyu/scripts/build_crop_json.py
"""

import os
import json
import glob
import numpy as np
import nibabel as nib

# ── 路径配置 ────────────────────────────────────────────────────────────────
RAW_DIR     = "/home/PuMengYu/nnUNet_workspace/raw/Dataset003_Liver"
IMAGES_DIR  = os.path.join(RAW_DIR, "imagesTr")
PRED_BASE   = "/home/PuMengYu/nnUNet_workspace/results/Dataset003_Liver/nnUNetTrainer__nnUNetPlans__3d_fullres"
OUTPUT_JSON = "/home/PuMengYu/nnUNet/pumengyu/scripts/bbox_crops.json"

# padding 单位：mm，会根据各轴 spacing 换算成体素数
PADDING_MM = 20.0

# ── 工具函数 ────────────────────────────────────────────────────────────────

def get_pred_path(case_id: str) -> str:
    """从5折 validation 目录中找到该 case 的预测文件。"""
    for fold in range(5):
        p = os.path.join(PRED_BASE, f"fold_{fold}", "validation", f"{case_id}.nii.gz")
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"找不到 {case_id} 的预测文件")


def compute_bbox(mask: np.ndarray, spacing_xyz: tuple, padding_mm: float) -> dict:
    """
    计算 mask 非零区域的 bbox，并在每个方向外扩 padding_mm。

    mask shape: (X, Y, Z)  —— nibabel 读出来的轴顺序
    spacing_xyz: (sx, sy, sz) 单位 mm/voxel

    返回体素坐标 {x, y, z} 各自的 [lo, hi]（含 hi，Python slice 用时 hi+1）
    """
    idx = np.where(mask)
    if len(idx[0]) == 0:
        # 极端情况：mask 全空，返回整图
        return {
            "x": [0, mask.shape[0] - 1],
            "y": [0, mask.shape[1] - 1],
            "z": [0, mask.shape[2] - 1],
        }

    result = {}
    for ax, (indices, spacing) in enumerate(zip(idx, spacing_xyz)):
        pad_vox = int(np.ceil(padding_mm / spacing))
        lo = max(0, int(indices.min()) - pad_vox)
        hi = min(mask.shape[ax] - 1, int(indices.max()) + pad_vox)
        result[["x", "y", "z"][ax]] = [lo, hi]
    return result


# ── 主流程 ──────────────────────────────────────────────────────────────────

def main():
    image_files = sorted(glob.glob(os.path.join(IMAGES_DIR, "*_0000.nii.gz")))
    print(f"共 {len(image_files)} 个训练样本")

    all_bboxes = {}
    failed = []

    for img_path in image_files:
        case_id = os.path.basename(img_path).replace("_0000.nii.gz", "")

        try:
            pred_path = get_pred_path(case_id)
        except FileNotFoundError as e:
            print(f"  [SKIP] {e}")
            failed.append(case_id)
            continue

        # 只读预测，不读 GT
        pred_nib = nib.load(pred_path)
        pred_arr = np.asarray(pred_nib.get_fdata(), dtype=np.uint8)
        pred_liver = pred_arr >= 1    # label1(liver) + label2(tumor) 都算肝脏区域

        # spacing：nibabel affine 对角线取绝对值，顺序 x,y,z
        aff = pred_nib.affine
        spacing_xyz = (
            abs(float(aff[0, 0])),
            abs(float(aff[1, 1])),
            abs(float(aff[2, 2])),
        )

        bbox = compute_bbox(pred_liver, spacing_xyz, PADDING_MM)
        bbox["spacing_xyz_mm"] = list(spacing_xyz)
        bbox["shape_xyz"]      = list(pred_arr.shape)
        bbox["padding_mm"]     = PADDING_MM

        all_bboxes[case_id] = bbox
        print(f"  {case_id}: x{bbox['x']} y{bbox['y']} z{bbox['z']}  "
              f"spacing={[round(s,3) for s in spacing_xyz]}")

    with open(OUTPUT_JSON, "w") as f:
        json.dump(all_bboxes, f, indent=2)

    print(f"\n已保存 → {OUTPUT_JSON}")
    print(f"成功: {len(all_bboxes)}, 跳过: {len(failed)}")
    if failed:
        print("跳过列表:", failed)


if __name__ == "__main__":
    main()
