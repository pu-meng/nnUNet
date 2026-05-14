#!/bin/bash
# 通用报告生成脚本，支持 Dataset003/Dataset004、任意 Trainer
#
# 用法：
#   bash pumengyu/notes/sh/shells/gen_report_any.sh \
#     <fold> <dataset_id> <dataset_suffix> <trainer> [config]
#
# 示例（Dataset003 单阶段，fold 0）：
#   bash ... 0 3 Liver nnUNetTrainer
#
# 示例（Dataset004 TwoStage，fold 0）：
#   bash ... 0 4 LiverTumor nnUNetTrainer_TwoStage
#
# 示例（Dataset004 BoundaryLoss，fold 0）：
#   bash ... 0 4 LiverTumor nnUNetTrainer_BoundaryLoss

set -e

FOLD=${1:?用法: $0 <fold> <dataset_id> <dataset_suffix> <trainer> [config]}
DATASET_ID=${2:?需要 dataset_id}
DATASET_SUFFIX=${3:?需要 dataset_suffix，如 Liver 或 LiverTumor}
TRAINER=${4:?需要 trainer，如 nnUNetTrainer}
CONFIG=${5:-3d_fullres}

RESULTS=${nnUNet_results:-/home/PuMengYu/nnUNet_workspace/results}
PREPROCESSED=${nnUNet_preprocessed:-/home/PuMengYu/nnUNet_workspace/preprocessed}
RAW=${nnUNet_raw:-/home/PuMengYu/nnUNet_workspace/raw}

DATASET_NAME="Dataset$(printf '%03d' ${DATASET_ID})_${DATASET_SUFFIX}"
FOLD_DIR="${RESULTS}/${DATASET_NAME}/${TRAINER}__nnUNetPlans__${CONFIG}/fold_${FOLD}"
GT_DIR="${PREPROCESSED}/${DATASET_NAME}/gt_segmentations"
IMG_DIR="${RAW}/${DATASET_NAME}/imagesTr"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "===== 生成报告 ====="
echo "  fold      : ${FOLD}"
echo "  dataset   : ${DATASET_NAME}"
echo "  trainer   : ${TRAINER}"
echo "  fold_dir  : ${FOLD_DIR}"
echo ""

if [ ! -f "${FOLD_DIR}/validation/summary.json" ]; then
    echo "[ERROR] 找不到 summary.json: ${FOLD_DIR}/validation/summary.json"
    exit 1
fi

echo "===== 生成 report_custom.json ====="
python "${SCRIPT_DIR}/../../../tools/analyasis/gen_report_json.py" \
  --fold_dir "${FOLD_DIR}"

echo ""
echo "===== 生成 report_custom.txt + vis_png_custom ====="
python "${SCRIPT_DIR}/../../../tools/analyasis/eval_fold_report.py" \
  --val_dir  "${FOLD_DIR}/validation" \
  --gt_dir   "${GT_DIR}" \
  --img_dir  "${IMG_DIR}" \
  --vis_slices 5

echo ""
echo "===== Done: ${FOLD_DIR}/report_custom.txt ====="
