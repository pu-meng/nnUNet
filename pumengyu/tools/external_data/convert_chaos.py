"""
CHAOS CT → Dataset003_Liver 格式转换

CHAOS CT 格式：
  Train_Sets/CT/{patient_id}/DICOM_anon/  — CT DICOM
  Train_Sets/CT/{patient_id}/Ground/      — 分割 PNG（灰度值 55=liver）

用法：
  python pumengyu/tools/external_data/convert_chaos.py \
    --chaos_dir /home/PuMengYu/8T/Datasets/CHAOS/Train_Sets \
    --out_dir   /home/PuMengYu/nnUNet_workspace/external_staging/chaos

输出文件：
  chaos_001_0000.nii.gz  — CT 图像
  chaos_001.nii.gz       — 分割标注（label 1=liver, 2=tumor全0）
"""

from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from PIL import Image




def read_dicom_series(dicom_dir: Path) -> sitk.Image:
    reader = sitk.ImageSeriesReader()
    names  = reader.GetGDCMSeriesFileNames(str(dicom_dir))
    if not names:
        raise FileNotFoundError(f"未找到 DICOM 文件: {dicom_dir}")
    reader.SetFileNames(names)
    return reader.Execute()


def read_ground_pngs(ground_dir: Path, n_slices: int) -> np.ndarray:
    """读取 Ground/ 目录下的 PNG 文件，按文件名数字排序，返回 (Z, Y, X) int16。"""
    pngs = sorted(ground_dir.glob("*.png"),
                  key=lambda p: int("".join(filter(str.isdigit, p.stem)) or "0"))
    if len(pngs) == 0:
        raise FileNotFoundError(f"未找到 PNG 文件: {ground_dir}")
    if len(pngs) != n_slices:
        raise ValueError(f"PNG 数量({len(pngs)}) 与 CT 层数({n_slices}) 不一致: {ground_dir}")

    slices = []
    for png_path in pngs:
        arr = np.array(Image.open(png_path).convert("L"))
        slices.append(arr)
    return np.stack(slices, axis=0)  # (Z, Y, X)


def convert_case(patient_id: str, chaos_ct_dir: Path, out_dir: Path) -> None:
    """
    chaos_ct_dir: Train_Sets/CT/
    patient_id:   '1', '2', ...
    """
    dicom_dir  = chaos_ct_dir / patient_id / "DICOM_anon"
    ground_dir = chaos_ct_dir / patient_id / "Ground"

    if not dicom_dir.exists():
        raise FileNotFoundError(f"DICOM 目录不存在: {dicom_dir}")
    if not ground_dir.exists():
        raise FileNotFoundError(f"Ground 目录不存在: {ground_dir}")

    print(f"[chaos_{int(patient_id):03d}] 读取 CT...")
    ct_sitk = read_dicom_series(dicom_dir)
    ct_arr  = sitk.GetArrayFromImage(ct_sitk)  # (Z, Y, X)

    print(f"[chaos_{int(patient_id):03d}] 读取 Ground PNG...")
    ground_arr = read_ground_pngs(ground_dir, ct_arr.shape[0])

    seg_arr = np.zeros_like(ct_arr, dtype=np.int16)
    seg_arr[ground_arr > 0] = 1  # label 1=liver，非零即肝脏
    # label 2（tumor）全为 0

    seg_sitk = sitk.GetImageFromArray(seg_arr)
    seg_sitk.CopyInformation(ct_sitk)

    case_id = f"chaos_{int(patient_id):03d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    ct_out  = out_dir / f"{case_id}_0000.nii.gz"
    seg_out = out_dir / f"{case_id}.nii.gz"

    sitk.WriteImage(ct_sitk,  str(ct_out),  useCompression=True)
    sitk.WriteImage(seg_sitk, str(seg_out), useCompression=True)

    liver_voxels = int((seg_arr == 1).sum())
    print(f"[chaos_{int(patient_id):03d}] 完成  shape={ct_arr.shape}"
          f"  liver={liver_voxels:,} voxels")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--chaos_dir", required=True, help="CHAOS Train_Sets 目录")
    p.add_argument("--out_dir",   required=True, help="staging 输出目录")
    args = p.parse_args()

    chaos_ct_dir = Path(args.chaos_dir) / "CT"
    out_dir      = Path(args.out_dir)

    if not chaos_ct_dir.exists():
        raise FileNotFoundError(f"未找到 CT 目录: {chaos_ct_dir}，请确认 CHAOS 解压路径")

    patient_ids = sorted(
        [p.name for p in chaos_ct_dir.iterdir() if p.is_dir()],
        key=lambda x: int(x)
    )
    print(f"找到 {len(patient_ids)} 个 CHAOS CT patient: {patient_ids}")

    for pid in patient_ids:
        convert_case(pid, chaos_ct_dir, out_dir)

    print(f"\n全部完成，输出到: {out_dir}")


if __name__ == "__main__":
    main()
