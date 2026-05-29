"""
3D-IRCADb → nnUNet 格式转换脚本
将 20 个 patient 的 DICOM 数据转成 NIfTI，生成 label 0/1/2（背景/肝脏/肿瘤）

用法：python prepare_ircadb.py

输出：
  /home/PuMengYu/nnUNet_workspace/external_val/ircadb_full/images/ircadb_XXX_0000.nii.gz
  /home/PuMengYu/nnUNet_workspace/external_val/ircadb_full/labels/ircadb_XXX.nii.gz
  /home/PuMengYu/nnUNet_workspace/external_val/ircadb_full/case_info.json
"""

import os
import json
import numpy as np
import SimpleITK as sitk
from pathlib import Path

SRC_ROOT  = Path("/home/PuMengYu/8T/Datasets/3Dircadb1")
OUT_ROOT  = Path("/home/PuMengYu/nnUNet_workspace/external_val/ircadb_full")
IMG_DIR   = OUT_ROOT / "images"
LBL_DIR   = OUT_ROOT / "labels"

IMG_DIR.mkdir(parents=True, exist_ok=True)
LBL_DIR.mkdir(parents=True, exist_ok=True)

# 哪些 mask 目录名对应肝脏肿瘤（排除：囊肿、肾上腺瘤、术后瘢痕、其他器官肿瘤）
def is_liver_tumor(mask_name: str) -> bool:
    name = mask_name.lower()
    if name == "liver":                         # liver mask 本身
        return False
    if "metastasectomi" in name:                # 术后切除灶，无活体肿瘤
        return False
    if "liver" not in name:                     # 只取名字里含 liver 的 mask
        return False
    if "cyst" in name or "kyst" in name:        # 囊肿不算肿瘤（与LiTS一致）
        return False
    return True


def read_dicom_series(dicom_dir: Path) -> sitk.Image:
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(str(dicom_dir))
    if series_ids:
        files = reader.GetGDCMSeriesFileNames(str(dicom_dir), series_ids[0])
    else:
        # 回退：按文件名排序手动读取
        files = sorted(
            [str(f) for f in dicom_dir.iterdir() if f.is_file()],
            key=lambda x: int(Path(x).stem.split("_")[-1])
        )
    reader.SetFileNames(files)
    reader.MetaDataDictionaryArrayUpdateOn()
    return reader.Execute()


def main():
    case_info = {}

    for case_id in range(1, 21):
        case_name = f"3Dircadb1.{case_id}"
        case_dir  = SRC_ROOT / case_name

        out_id   = f"ircadb_{case_id:03d}"
        img_out  = IMG_DIR / f"{out_id}_0000.nii.gz"
        lbl_out  = LBL_DIR / f"{out_id}.nii.gz"

        if img_out.exists() and lbl_out.exists():
            print(f"[SKIP] {out_id} 已存在")
            # 从已有 label 文件直接读取 has_tumor，不依赖旧 case_info.json
            import SimpleITK as _sitk, numpy as _np
            _lbl = _np.asarray(_sitk.GetArrayFromImage(_sitk.ReadImage(str(lbl_out))))
            _tumor_vox = int(_np.sum(_lbl == 2))
            case_info[out_id] = {
                "has_tumor": _tumor_vox > 0,
                "tumor_voxels": _tumor_vox,
                "tumor_masks": [],   # skip 时不重新扫描原始目录
                "n_slices": int(_lbl.shape[0]),
            }
            continue

        print(f"[{case_id:02d}/20] 处理 {case_name} ...")

        patient_dir = case_dir / "PATIENT_DICOM" / "PATIENT_DICOM"
        masks_root  = case_dir / "MASKS_DICOM"   / "MASKS_DICOM"

        # ── 读取 CT 图像 ──────────────────────────────────────
        try:
            ct_img = read_dicom_series(patient_dir)
        except Exception as e:
            print(f"  [ERROR] 读取CT失败: {e}")
            continue

        ct_arr   = sitk.GetArrayFromImage(ct_img)        # (Z, Y, X)
        n_slices = ct_arr.shape[0]

        # ── 读取 liver mask ───────────────────────────────────
        liver_dir = masks_root / "liver"
        try:
            liver_img = read_dicom_series(liver_dir)
            liver_arr = sitk.GetArrayFromImage(liver_img).astype(np.uint8)
        except Exception as e:
            print(f"  [ERROR] 读取liver mask失败: {e}")
            continue

        # ── 读取所有肿瘤 mask，合并 ───────────────────────────
        tumor_arr   = np.zeros_like(liver_arr)
        tumor_masks = []
        for mask_name in sorted(os.listdir(masks_root)):
            if is_liver_tumor(mask_name):
                tumor_masks.append(mask_name)
                try:
                    t_img = read_dicom_series(masks_root / mask_name)
                    t_arr = sitk.GetArrayFromImage(t_img).astype(np.uint8)
                    if t_arr.shape == tumor_arr.shape:
                        tumor_arr = np.maximum(tumor_arr, t_arr)
                    else:
                        print(f"  [WARN] {mask_name} shape不匹配: {t_arr.shape} vs {tumor_arr.shape}")
                except Exception as e:
                    print(f"  [WARN] 读取{mask_name}失败: {e}")

        has_tumor = bool(np.any(tumor_arr > 0))
        tumor_voxels = int(np.sum(tumor_arr > 0))

        # ── 构建 label: 0=bg, 1=liver, 2=tumor ───────────────
        label_arr = np.zeros_like(liver_arr, dtype=np.uint8)
        label_arr[liver_arr > 0] = 1
        label_arr[tumor_arr > 0] = 2

        # ── 保存 NIfTI ────────────────────────────────────────
        ct_out_img = sitk.GetImageFromArray(ct_arr.astype(np.int16))
        ct_out_img.CopyInformation(ct_img)
        sitk.WriteImage(ct_out_img, str(img_out))

        lbl_out_img = sitk.GetImageFromArray(label_arr)
        lbl_out_img.CopyInformation(ct_img)
        sitk.WriteImage(lbl_out_img, str(lbl_out))

        case_info[out_id] = {
            "has_tumor": has_tumor,
            "tumor_voxels": tumor_voxels,
            "tumor_masks": tumor_masks,
            "n_slices": n_slices,
        }

        status = f"有肿瘤 ({tumor_voxels:,} vox, masks={tumor_masks})" if has_tumor else "无肿瘤"
        print(f"  → {status}")

    # ── 保存 case_info ────────────────────────────────────────
    with open(OUT_ROOT / "case_info.json", "w", encoding="utf-8") as f:
        json.dump(case_info, f, ensure_ascii=False, indent=2)

    # ── 汇总 ─────────────────────────────────────────────────
    tumor_cases    = [k for k, v in case_info.items() if v["has_tumor"]]
    no_tumor_cases = [k for k, v in case_info.items() if not v["has_tumor"]]
    print(f"\n转换完成：共 {len(case_info)} 个 case")
    print(f"  有肿瘤：{len(tumor_cases)} 个  → {sorted(tumor_cases)}")
    print(f"  无肿瘤：{len(no_tumor_cases)} 个 → {sorted(no_tumor_cases)}")
    print(f"\n输出目录：{OUT_ROOT}")


if __name__ == "__main__":
    main()
