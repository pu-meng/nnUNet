#!/bin/bash
# 用法: bash pumengyu/scripts/debug_batseg.sh <GPU_ID> <FOLD>
# 例如: bash pumengyu/scripts/debug_batseg.sh 1 4

GPU=${1:-1}
FOLD=${2:-4}
LOG=/tmp/batseg_gpu${GPU}_fold${FOLD}.txt

echo "Running fold ${FOLD} on GPU ${GPU}, log -> ${LOG}"

CUDA_VISIBLE_DEVICES=${GPU} nnUNetv2_train 3 3d_fullres ${FOLD} -tr nnUNetTrainer_BATseg 2>&1 | tee "${LOG}"

echo ""
echo "===== ERROR SUMMARY ====="
ERROR_LINE=$(grep -n "RuntimeError: One or more background" "${LOG}" | head -1 | cut -d: -f1)
if [ -n "${ERROR_LINE}" ]; then
    START=$(( ERROR_LINE > 60 ? ERROR_LINE - 60 : 1 ))
    echo "--- Context before worker crash (lines ${START}-${ERROR_LINE}) ---"
    sed -n "${START},${ERROR_LINE}p" "${LOG}"
else
    echo "No background worker error found, check ${LOG} directly."
fi
