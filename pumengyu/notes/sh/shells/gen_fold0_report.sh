#!/bin/bash
set -e

FOLD_DIR=/home/PuMengYu/nnUNet_workspace/results/Dataset003_Liver/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_4
GT_DIR=/home/PuMengYu/nnUNet_workspace/preprocessed/Dataset003_Liver/gt_segmentations
IMG_DIR=/home/PuMengYu/nnUNet_workspace/raw/Dataset003_Liver/imagesTr

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "===== 生成 report_custom.json ====="
python "${SCRIPT_DIR}/../tools/gen_report_json.py" \
  --fold_dir "${FOLD_DIR}"

echo ""
echo "===== 生成 report_custom.txt + vis_png_custom ====="
python "${SCRIPT_DIR}/../tools/eval_fold_report.py" \
  --val_dir  "${FOLD_DIR}/validation" \
  --gt_dir   "${GT_DIR}" \
  --img_dir  "${IMG_DIR}" \
  --vis_slices 5

echo ""
echo "===== Done ====="
