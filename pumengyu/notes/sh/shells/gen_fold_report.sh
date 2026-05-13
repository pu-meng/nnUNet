#!/bin/bash
# 用法: bash pumengyu/scripts/gen_fold_report.sh <FOLD> [DATASET_ID] [CONFIG]
# 示例: bash pumengyu/scripts/gen_fold_report.sh 0
#       bash pumengyu/scripts/gen_fold_report.sh 1 3 3d_fullres
#fold_1
# bash pumengyu/scripts/gen_fold_report.sh 1 
set -e

FOLD=${1:?用法: $0 <FOLD> [DATASET_ID] [CONFIG]}
DATASET_ID=${2:-3}
CONFIG=${3:-3d_fullres}

RESULTS=${nnUNet_results:-/home/PuMengYu/nnUNet_workspace/results}
PREPROCESSED=${nnUNet_preprocessed:-/home/PuMengYu/nnUNet_workspace/preprocessed}
RAW=${nnUNet_raw:-/home/PuMengYu/nnUNet_workspace/raw}

DATASET_NAME="Dataset$(printf '%03d' ${DATASET_ID})_Liver"
FOLD_DIR="${RESULTS}/${DATASET_NAME}/nnUNetTrainer__nnUNetPlans__${CONFIG}/fold_${FOLD}"
GT_DIR="${PREPROCESSED}/${DATASET_NAME}/gt_segmentations"
IMG_DIR="${RAW}/${DATASET_NAME}/imagesTr"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "===== fold_${FOLD}  dataset=${DATASET_NAME}  config=${CONFIG} ====="

echo ""
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
