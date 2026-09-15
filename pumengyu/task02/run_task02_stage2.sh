#!/usr/bin/env bash
# Launch exactly one fresh Task02 Stage-2 training run.
#
# Examples (do not append --c, --val, or -pretrained_weights):
#   TASK02_CUDA_VISIBLE_DEVICES=0 bash pumengyu/task02/run_task02_stage2.sh \
#     nnUNetTrainer_MedNeXt_MHA_MoE_Task02_Replay_K03
#   TASK02_CUDA_VISIBLE_DEVICES=0,1 TASK02_NUM_GPUS=2 \
#     bash pumengyu/task02/run_task02_stage2.sh \
#     nnUNetTrainer_MedNeXt_MHA_MoE_Task02_Replay_K70
#
# This file deliberately starts training only. After the run finishes, first
# inspect task02_manifest.json, task02_validation_history.jsonl and
# task02_selection.json. The formal LiTS/IRCADb/HCC final evaluation is a
# separate operation on checkpoint_task02_best.pth.

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <one Task02 trainer name>" >&2
    exit 2
fi

TRAINER="$1"
case "$TRAINER" in
    nnUNetTrainer_MedNeXt_MHA_MoE_Task02_HCCOnly_K01|\
    nnUNetTrainer_MedNeXt_MHA_MoE_Task02_HCCOnly_K03|\
    nnUNetTrainer_MedNeXt_MHA_MoE_Task02_HCCOnly_K05|\
    nnUNetTrainer_MedNeXt_MHA_MoE_Task02_HCCOnly_K10|\
    nnUNetTrainer_MedNeXt_MHA_MoE_Task02_Replay_K01|\
    nnUNetTrainer_MedNeXt_MHA_MoE_Task02_Replay_K03|\
    nnUNetTrainer_MedNeXt_MHA_MoE_Task02_Replay_K05|\
    nnUNetTrainer_MedNeXt_MHA_MoE_Task02_Replay_K10|\
    nnUNetTrainer_MedNeXt_MHA_MoE_Task02_Replay_K70)
        ;;
    *)
        echo "Refusing non-Task02 trainer: $TRAINER" >&2
        exit 2
        ;;
esac

TASK02_WORKSPACE="/home/PuMengYu/nnUNet_workspace"
export nnUNet_raw="${TASK02_WORKSPACE}/raw"
export nnUNet_preprocessed="${TASK02_WORKSPACE}/preprocessed"
export nnUNet_results="${TASK02_RESULTS_ROOT:-${TASK02_WORKSPACE}/results_task02}"
export CUDA_VISIBLE_DEVICES="${TASK02_CUDA_VISIBLE_DEVICES:-0}"

NUM_GPUS="${TASK02_NUM_GPUS:-1}"
FOLD_DIR="${nnUNet_results}/Dataset003_Liver/${TRAINER}__nnUNetPlans__3d_fullres/fold_0"
if [[ -e "$FOLD_DIR" ]]; then
    echo "Refusing to reuse or overwrite existing Task02 result directory:" >&2
    echo "  $FOLD_DIR" >&2
    echo "Use a new TASK02_RESULTS_ROOT for a new run; --c is not part of the frozen protocol." >&2
    exit 1
fi

echo "[Task02] trainer=$TRAINER"
echo "[Task02] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, num_gpus=$NUM_GPUS"
echo "[Task02] result root=$nnUNet_results"
echo "[Task02] Stage-2 source checkpoint is loaded and SHA-checked by the trainer."

nnUNetv2_train 3 3d_fullres 0 -tr "$TRAINER" -num_gpus "$NUM_GPUS" -device cuda
