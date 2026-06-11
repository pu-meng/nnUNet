#!/bin/bash
# 为关键失败 case 生成 ITK-SNAP 可视化文件
#
# 每个 case 生成三个文件：
#   {case}_ct.nii.gz         → 直接加载为主图像
#   {case}_gt.nii.gz         → GT标注（label1=肝脏 label2=肿瘤）
#   {case}_combined.nii.gz   → 对比图（label1=TP绿 label2=FP红 label3=FN蓝 label4=肝脏灰）
#
# ITK-SNAP 加载方式：
#   1. File → Open Main Image → 选 {case}_ct.nii.gz
#   2. Segmentation → Open Segmentation → 选 {case}_combined.nii.gz
#   绿=预测正确的肿瘤  红=误报  蓝=漏检  灰=肝脏轮廓
#
# 运行：bash pumengyu/notes/sh/gen_itksnap_viz.sh

set -e
cd /home/PuMengYu/nnUNet

OUT=/home/PuMengYu/viz_itksnap
mkdir -p "$OUT"

PRED_DIR=/home/PuMengYu/nnUNet_workspace/results_v2/Dataset003_Liver/nnUNetTrainer_SizeOversampleV3_NoMirror__nnUNetPlans__3d_fullres/fold_0/test_prediction
GT_DIR=/home/PuMengYu/nnUNet_workspace/preprocessed/Dataset003_Liver/gt_segmentations
CT_DIR=/home/PuMengYu/nnUNet_workspace/raw/Dataset003_Liver/imagesTr

# 要分析的关键 case（可自由增删）
CASES=(
    liver_41     # 无肿瘤 大FP误报 28352体素
    liver_91     # 无肿瘤 小FP误报 131体素
    liver_127    # 严重失败 极小肿瘤完全漏掉 Dice=0
    liver_130    # 大肿瘤低Recall=0.43 Dice=0.59
    liver_58     # 极小肿瘤 Dice=0.55
    liver_63     # 极小肿瘤 Dice=0.50
)

echo "输出目录: $OUT"
echo "使用模型: SizeOversampleV3_NoMirror（当前有test_prediction的最优模型）"
echo ""

python3 - <<'PYEOF'
import os, sys, numpy as np, nibabel as nib
from pathlib import Path

out      = Path(os.environ.get("OUT",      "/home/PuMengYu/viz_itksnap"))
pred_dir = Path(os.environ.get("PRED_DIR", ""))
gt_dir   = Path(os.environ.get("GT_DIR",   ""))
ct_dir   = Path(os.environ.get("CT_DIR",   ""))

cases_str = os.environ.get("CASES", "")
cases = [c.strip() for c in cases_str.split(",") if c.strip()]

PRED_DIR="/home/PuMengYu/nnUNet_workspace/results_v2/Dataset003_Liver/nnUNetTrainer_SizeOversampleV3_NoMirror__nnUNetPlans__3d_fullres/fold_0/test_prediction"
GT_DIR="/home/PuMengYu/nnUNet_workspace/preprocessed/Dataset003_Liver/gt_segmentations"
CT_DIR="/home/PuMengYu/nnUNet_workspace/raw/Dataset003_Liver/imagesTr"
OUT="/home/PuMengYu/viz_itksnap"
CASES=["liver_41","liver_91","liver_127","liver_130","liver_58","liver_63"]

pred_dir = Path(PRED_DIR)
gt_dir   = Path(GT_DIR)
ct_dir   = Path(CT_DIR)
out      = Path(OUT)
out.mkdir(parents=True, exist_ok=True)

for case in CASES:
    print(f"处理 {case}...")

    ct_path   = ct_dir  / f"{case}_0000.nii.gz"
    gt_path   = gt_dir  / f"{case}.nii.gz"
    pred_path = pred_dir / f"{case}.nii.gz"

    if not ct_path.exists():
        print(f"  [跳过] CT 不存在: {ct_path}")
        continue
    if not gt_path.exists():
        print(f"  [跳过] GT 不存在: {gt_path}")
        continue
    if not pred_path.exists():
        print(f"  [跳过] 预测不存在: {pred_path}")
        continue

    case_dir = out / case          # 每个 case 独立子文件夹
    case_dir.mkdir(parents=True, exist_ok=True)

    ct_nib   = nib.load(ct_path)
    gt_nib   = nib.load(gt_path)
    pred_nib = nib.load(pred_path)

    gt   = gt_nib.get_fdata().astype(np.uint8)
    pred = pred_nib.get_fdata().astype(np.uint8)

    gt_tumor   = (gt   == 2).astype(np.uint8)
    pred_tumor = (pred == 2).astype(np.uint8)
    gt_liver   = (gt   >= 1).astype(np.uint8)

    # combined label map
    # 0=背景  1=TP(绿)  2=FP(红)  3=FN(蓝)  4=肝脏轮廓(灰)
    combined = np.zeros_like(gt, dtype=np.uint8)
    combined[gt_liver == 1]                                  = 4  # 肝脏底色
    combined[(gt_tumor == 1) & (pred_tumor == 1)]            = 1  # TP
    combined[(gt_tumor == 0) & (pred_tumor == 1)]            = 2  # FP
    combined[(gt_tumor == 1) & (pred_tumor == 0)]            = 3  # FN

    # 保存到 case 子文件夹
    nib.save(nib.Nifti1Image(combined, gt_nib.affine, gt_nib.header),
             str(case_dir / f"{case}_combined.nii.gz"))
    nib.save(nib.Nifti1Image(gt, gt_nib.affine, gt_nib.header),
             str(case_dir / f"{case}_gt.nii.gz"))
    nib.save(nib.Nifti1Image(pred, pred_nib.affine, pred_nib.header),
             str(case_dir / f"{case}_pred.nii.gz"))

    import shutil
    shutil.copy(str(ct_path), str(case_dir / f"{case}_ct.nii.gz"))

    tp = int((combined == 1).sum())
    fp = int((combined == 2).sum())
    fn = int((combined == 3).sum())
    print(f"  TP={tp:,}  FP={fp:,}  FN={fn:,}  → 已保存到 {case_dir}")

print("\n全部完成！")
print(f"输出目录: {OUT}/")
print("\nITK-SNAP 加载步骤：")
print("  1. File → Open Main Image → 选 liver_XXX_ct.nii.gz")
print("  2. Segmentation → Open Segmentation → 选 liver_XXX_combined.nii.gz")
print("  颜色：绿=TP（正确）  红=FP（误报）  蓝=FN（漏检）  灰=肝脏")
PYEOF

echo ""
echo "===== 打包为 zip ====="
cd /home/PuMengYu
zip -r viz_itksnap.zip viz_itksnap/
echo "下载包: /home/PuMengYu/viz_itksnap.zip"
du -sh /home/PuMengYu/viz_itksnap.zip
