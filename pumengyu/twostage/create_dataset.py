"""
用 GT 肝脏+肿瘤 mask 裁剪 ROI，创建 Dataset004_LiverTumor。


Training data preparation:
  For each case, use the GT liver (class 1) + tumor (class 2) mask to
  define a bounding box + margin, then crop the image and label.
  Label remapping: background=0, liver=1 → 0, tumor=2 → 1.

Note on train/inference gap:
  Training uses GT crops (perfect boundaries).
  Inference uses Stage-1 predicted crops (see eval.py).
  The 30 mm margin compensates for Stage-1 prediction uncertainty.
  This is standard practice in two-stage segmentation pipelines.

Usage:
  python pumengyu/twostage/create_dataset.py \\
      --workspace /home/PuMengYu/nnUNet_workspace \\
      [--margin_mm 30]

After this script:
  nnUNetv2_plan_and_preprocess -d 4 --verify_dataset_integrity
"""

import argparse
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def liver_bbox_sitk(
    organ_mask: sitk.Image,
    margin_mm: float,
) -> tuple[list[int], list[int]]:
    """
    SimpleITK是ITK(Insight Toolkit)的简化封装,ITK是医学图像领域最主流的C++库,SimpleITK提供了Python接口,使得Python用户也能方便地使用ITK的功能。
    
    

    
    Return (start_xyz, size_xyz) covering the organ mask + margin.
    SimpleITK convention: (x, y, z).
    """
    arr      = sitk.GetArrayViewFromImage(organ_mask)  # (z, y, x)
    spacing  = organ_mask.GetSpacing()                 # (sp_x, sp_y, sp_z)
    full_sz  = organ_mask.GetSize()                    # (sx, sy, sz)

    nz = np.where(arr > 0)
    if len(nz[0]) == 0:
        return [0, 0, 0], list(full_sz)

    z_min, z_max = int(nz[0].min()), int(nz[0].max()) + 1
    y_min, y_max = int(nz[1].min()), int(nz[1].max()) + 1
    x_min, x_max = int(nz[2].min()), int(nz[2].max()) + 1

    mx = int(np.ceil(margin_mm / spacing[0]))
    my = int(np.ceil(margin_mm / spacing[1]))
    mz = int(np.ceil(margin_mm / spacing[2]))

    x0 = max(0,          x_min - mx);  x1 = min(full_sz[0], x_max + mx)
    y0 = max(0,          y_min - my);  y1 = min(full_sz[1], y_max + my)
    z0 = max(0,          z_min - mz);  z1 = min(full_sz[2], z_max + mz)

    return [x0, y0, z0], [x1 - x0, y1 - y0, z1 - z0]


def crop_case(
    case_id:    str,
    images_tr:  Path,
    labels_tr:  Path,
    out_images: Path,
    out_labels: Path,
    margin_mm:  float,
) -> dict:
    """Crop one case using GT mask. Returns crop metadata."""
    img   = sitk.ReadImage(str(images_tr / f"{case_id}_0000.nii.gz"))
    label = sitk.ReadImage(str(labels_tr / f"{case_id}.nii.gz"))

    # ROI = liver (1) ∪ tumor (2) — must include both so large peripheral
    # tumors are not clipped by a liver-only bounding box.
    organ_mask = sitk.BinaryThreshold(
        label, lowerThreshold=1, upperThreshold=2,
        insideValue=1, outsideValue=0)
    #逐像素对比,二值化阈值操作,逐voxel判断,像素值\in[lowerThreshold, upperThreshold]的像素值设为insideValue,其他设为outsideValue


    start_xyz, size_xyz = liver_bbox_sitk(organ_mask, margin_mm)
#start_xyz是列表,[x0,y0,z0]是organ_mask的边界框起点坐标, size_xyz是列表,[sx,sy,sz]是边界框的大小(单位:像素)
    roi = sitk.RegionOfInterestImageFilter()
    #.RegionOfInterestImageFilter()是SimpleITK内置得裁剪工具,从一个大图切出一个矩形子区域,保留原图得
    #spacing, origin, direction等元信息不变,只改变size和index(起点坐标)
    roi.SetIndex(start_xyz)#设置裁剪起点
    roi.SetSize(size_xyz)#设置裁剪区域得尺寸,也就是各轴得voxel数

    img_crop   = roi.Execute(img)
    label_crop = roi.Execute(label)

    # Remap: liver=1 → 0,  tumor=2 → 1
    label_arr = sitk.GetArrayFromImage(label_crop).astype(np.uint8)
    #这个label_crop转化为numpy.array,并且数据类型转为uint8
    tumor_arr = (label_arr == 2).astype(np.uint8)
    tumor_img = sitk.GetImageFromArray(tumor_arr)
    tumor_img.CopyInformation(label_crop)
    #.CopyInformation()是把label_crop得几何信息原封不动得复制过去


    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img_crop,  str(out_images / f"{case_id}_0000.nii.gz"))
    sitk.WriteImage(tumor_img, str(out_labels / f"{case_id}.nii.gz"))

    return {
        "original_size_xyz": list(img.GetSize()),
        "crop_start_xyz":    start_xyz,
        "crop_size_xyz":     size_xyz,
        "spacing_xyz":       list(img.GetSpacing()),
        "origin":            list(img.GetOrigin()),
        "direction":         list(img.GetDirection()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace",     required=True)
    parser.add_argument("--dataset003_id", type=int, default=3)
    parser.add_argument("--dataset004_id", type=int, default=4)
    parser.add_argument("--margin_mm",     type=float, default=30.0)
    args = parser.parse_args()

    ws   = Path(args.workspace)
    #先Path之后/"raw"在任何平台都不会出错,字符串拼接很容易出错
    raw3 = ws / "raw" / f"Dataset{args.dataset003_id:03d}_Liver"
    raw4 = ws / "raw" / f"Dataset{args.dataset004_id:03d}_LiverTumor"
#f-string本质是普通字符串,任何需要字符串得地方都可以用
#
    images_tr  = raw3 / "imagesTr"
    # gt_segmentations has original-space labels in nnUNet naming (liver_X.nii.gz).
    # Use it directly so we don't need to re-extract labelsTr from the tar.
    labels_tr  = ws / "preprocessed" / f"Dataset{args.dataset003_id:03d}_Liver" / "gt_segmentations"
    out_images = raw4 / "imagesTr"
    out_labels = raw4 / "labelsTr"
#类似得.name,.stem,.parent等不区分最后得是文件名还说文件夹,这些都只是字符串操作
    cases = sorted(p.name.replace("_0000.nii.gz", "")
                   for p in images_tr.glob("*_0000.nii.gz"))
    print(f"Found {len(cases)} cases, margin = {args.margin_mm} mm\n")

    crop_meta: dict[str, dict] = {}
    #这个类型注解只是给人得提示不是必须这样做
    for case_id in cases:
        meta = crop_case(case_id, images_tr, labels_tr,
                         out_images, out_labels, args.margin_mm)
        crop_meta[case_id] = meta
        orig = meta["original_size_xyz"]
        crop = meta["crop_size_xyz"]
        print(f"  {case_id}: {orig} → {crop}")

    dataset_json = {
        "name":            "LiverTumor",
        "description":     (
            "Tumor segmentation on liver-ROI-cropped CT. "
            "Crops defined by GT liver+tumor masks (training only). "
            f"Margin: {args.margin_mm} mm."
        ),
        "reference":       "Derived from MSD Task03_Liver",
        "licence":         "CC-BY-SA 4.0",
        "tensorImageSize": "3D",
        "channel_names":   {"0": "CT"},
        "labels":          {"background": 0, "tumor": 1},
        "numTraining":     len(cases),
        "file_ending":     ".nii.gz",
    }
    with open(raw4 / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=2)

    # crop_meta.json: used by eval.py to map Stage-2 predictions back to
    # original space (GT-crop bbox is NOT used at inference time).
    with open(raw4 / "crop_meta.json", "w") as f:
        json.dump(crop_meta, f, indent=2)

    print(f"\nDataset004 created at: {raw4}")
    print(f"Next: nnUNetv2_plan_and_preprocess -d {args.dataset004_id} --verify_dataset_integrity")


if __name__ == "__main__":
    main()
