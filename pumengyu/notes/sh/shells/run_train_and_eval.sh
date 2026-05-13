#!/bin/bash
# 用法: bash pumengyu/scripts/run_train_and_eval.sh <DATASET_ID> <CONFIG> <FOLD> [GPU_ID]
# 示例: bash pumengyu/scripts/run_train_and_eval.sh 3 3d_fullres 0 0

DATASET_ID=${1:?need DATASET_ID}
CONFIG=${2:?need CONFIG}
FOLD=${3:?need FOLD}
GPU=${4:-0}

NNUNET_RESULTS=${nnUNet_results:-/home/PuMengYu/nnUNet_workspace/results}
GT_DIR=${nnUNet_preprocessed:-/home/PuMengYu/nnUNet_workspace/preprocessed}/Dataset${DATASET_ID}_Liver/gt_segmentations
IMG_DIR=${nnUNet_raw:-/home/PuMengYu/nnUNet_workspace/raw}/Dataset${DATASET_ID}_Liver/imagesTr

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "===== Training fold ${FOLD} on GPU ${GPU} ====="
CUDA_VISIBLE_DEVICES=${GPU} nnUNetv2_train ${DATASET_ID} ${CONFIG} ${FOLD}

if [ $? -ne 0 ]; then
  echo "[ERROR] Training failed, skipping eval"
  exit 1
fi

VAL_DIR="${NNUNET_RESULTS}/Dataset$(printf '%03d' ${DATASET_ID})_Liver/nnUNetTrainer__nnUNetPlans__${CONFIG}/fold_${FOLD}/validation"

echo ""
echo "===== Running eval report for fold ${FOLD} ====="
python "${SCRIPT_DIR}/../tools/eval_fold_report.py" \
  --val_dir  "${VAL_DIR}" \
  --gt_dir   "${GT_DIR}" \
  --img_dir  "${IMG_DIR}" \
  --vis_slices 5

echo "===== Done fold ${FOLD} ====="