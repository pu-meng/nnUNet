"""
3D-IRCADb → Dataset003_Liver 格式转换

将 IRCADb 的 DICOM 格式转为 nii.gz，输出到 staging 目录。
只处理无肿瘤 case（默认：5/7/11/14/20）。

用法：
  python pumengyu/tools/external_data/convert_ircad.py \
    --ircad_dir /home/PuMengYu/8T/Datasets/3Dircadb1 \
    --out_dir   /home/PuMengYu/nnUNet_workspace/external_staging/ircad \
    [--cases 5 7 11 14 20]

输出文件：
  ircad_005_0000.nii.gz  — CT 图像
  ircad_005.nii.gz       — 分割标注（label 1=liver, 2=tumor全0）
"""

from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import SimpleITK as sitk

# 默认处理无肿瘤 case
NO_TUMOR_CASES = [5, 7, 11, 14, 20]


def read_dicom_series(dicom_dir: Path) -> sitk.Image:
    reader = sitk.ImageSeriesReader()
    names  = reader.GetGDCMSeriesFileNames(str(dicom_dir))
    if not names:
        raise FileNotFoundError(f"未找到 DICOM 文件: {dicom_dir}")
    reader.SetFileNames(names)
    return reader.Execute()


def convert_case(case_num: int, ircad_root: Path, out_dir: Path) -> None:
    case_dir  = ircad_root / f"3Dircadb1.{case_num}"
    ct_dicom  = case_dir / "PATIENT_DICOM" / "PATIENT_DICOM"
    liver_dir = case_dir / "MASKS_DICOM" / "MASKS_DICOM" / "liver"

    if not ct_dicom.exists():
        raise FileNotFoundError(f"CT DICOM 不存在: {ct_dicom}")
    if not liver_dir.exists():
        raise FileNotFoundError(f"liver mask 不存在: {liver_dir}")

    print(f"[ircad_{case_num:03d}] 读取 CT...")
    ct_sitk  = read_dicom_series(ct_dicom)
    ct_arr   = sitk.GetArrayFromImage(ct_sitk)  # (Z, Y, X) int16

    print(f"[ircad_{case_num:03d}] 读取 liver mask...")
    liver_sitk = read_dicom_series(liver_dir)
    liver_arr  = sitk.GetArrayFromImage(liver_sitk)

    # 二值化 liver（mask 值通常为 0/255）
    seg_arr = np.zeros_like(ct_arr, dtype=np.int16)
    seg_arr[liver_arr > 0] = 1
    # label 2（tumor）全为 0（无肿瘤 case）

    # 用 CT 的空间信息创建 label image（保证 spacing/origin/direction 一致）
    seg_sitk = sitk.GetImageFromArray(seg_arr)
    seg_sitk.CopyInformation(ct_sitk)

    case_id = f"ircad_{case_num:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    ct_out  = out_dir / f"{case_id}_0000.nii.gz"
    seg_out = out_dir / f"{case_id}.nii.gz"

    sitk.WriteImage(ct_sitk,  str(ct_out),  useCompression=True)
    sitk.WriteImage(seg_sitk, str(seg_out), useCompression=True)

    liver_voxels = int((seg_arr == 1).sum())
    print(f"[ircad_{case_num:03d}] 完成  shape={ct_arr.shape}"
          f"  spacing={[round(s,3) for s in reversed(ct_sitk.GetSpacing())]}"
          f"  liver={liver_voxels:,} voxels")
    print(f"  CT  -> {ct_out}")
    print(f"  seg -> {seg_out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ircad_dir", required=True, help="3Dircadb1 根目录")
    p.add_argument("--out_dir",   required=True, help="staging 输出目录")
    p.add_argument("--cases", type=int, nargs="+", default=NO_TUMOR_CASES,
                   help="要处理的 case 编号（默认无肿瘤 case: 5 7 11 14 20）")
    args = p.parse_args()

    ircad_root = Path(args.ircad_dir)
    out_dir    = Path(args.out_dir)

    print(f"处理 {len(args.cases)} 个 IRCADb case: {args.cases}")
    for c in args.cases:
        convert_case(c, ircad_root, out_dir)

    print(f"\n全部完成，输出到: {out_dir}")


if __name__ == "__main__":
    main()
