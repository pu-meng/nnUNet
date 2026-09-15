#!/usr/bin/env bash
# Run the one non-paper Task02 smoke check. It must never share a result root
# with formal Task02 experiments and is deliberately not accepted by
# final_evaluation.py.

set -euo pipefail

TASK02_WORKSPACE="/home/PuMengYu/nnUNet_workspace"
export nnUNet_raw="${TASK02_WORKSPACE}/raw"
export nnUNet_preprocessed="${TASK02_WORKSPACE}/preprocessed"
export nnUNet_results="${TASK02_SMOKE_RESULTS_ROOT:-${TASK02_WORKSPACE}/results_task02_smoke}"
export CUDA_VISIBLE_DEVICES="${TASK02_CUDA_VISIBLE_DEVICES:-0}"
NUM_GPUS="${TASK02_NUM_GPUS:-1}"

TRAINER="nnUNetTrainer_MedNeXt_MHA_MoE_Task02Smoke_Replay_K03"
FOLD_DIR="${nnUNet_results}/Dataset003_Liver/${TRAINER}__nnUNetPlans__3d_fullres/fold_0"
if [[ -e "$FOLD_DIR" ]]; then
    echo "Refusing to overwrite existing Task02 smoke directory: $FOLD_DIR" >&2
    exit 1
fi

echo "[Task02 smoke] 3 epochs x 2 updates only; this is not a formal result."
echo "[Task02 smoke] result root=$nnUNet_results"
nnUNetv2_train 3 3d_fullres 0 -tr "$TRAINER" -num_gpus "$NUM_GPUS" -device cuda
