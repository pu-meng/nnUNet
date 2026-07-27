#!/usr/bin/env bash
# 在 GPU1 上顺序评估两个新方法的 HCCReferencedCT v2 source-only 结果。
#
# 方法：
#   1. MedNeXt_MHA_MoE
#   2. EfficientMedNeXt_L_Official
#
# 产物：
#   results_v2/Dataset013_HCCReferencedCT/source_only/<method>/
#     predictions/HCC_*.nii.gz
#     predictions/summary.json
#     report_custom.txt
#     test_viz/*.png
#
# 断点逻辑：默认不强制重跑；已有完整预测时复用预测，只重算报告/可视化。

set -uo pipefail

REPO_ROOT="/home/PuMengYu/nnUNet"
WORKSPACE="/home/PuMengYu/nnUNet_workspace"
PYTHON_BIN="/home/PuMengYu/anaconda3/envs/medseg/bin/python"
REPORT_PY="$REPO_ROOT/pumengyu/ext_val/05_gen_hcc_test_report.py"
SPLIT_INFO="$WORKSPACE/preprocessed/Dataset013_HCCReferencedCT/split_info_701020_stratified_v2.json"
EXT_VAL_ROOT="$WORKSPACE/external_val/hcc_referenced_ct_test_v2"
RESULT_ROOT="$WORKSPACE/results_v2/Dataset013_HCCReferencedCT/source_only"
DATASET003_ROOT="$WORKSPACE/results_v2/Dataset003_Liver"
GPU=1
EXPECTED_CASES=21

export PATH="/home/PuMengYu/anaconda3/envs/medseg/bin:$PATH"
export nnUNet_raw="$WORKSPACE/raw"
export nnUNet_preprocessed="$WORKSPACE/preprocessed"
export nnUNet_results="$WORKSPACE/results_v2"
export nnUNet_extTrainer="$REPO_ROOT/pumengyu"

METHODS=(
  "MedNeXt_MHA_MoE"
  "EfficientMedNeXt_L_Official"
)

TRAINERS=(
  "nnUNetTrainer_MedNeXt_MHA_MoE"
  "nnUNetTrainer_EfficientMedNeXt_L_Official"
)

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

if [ ! -x "$PYTHON_BIN" ]; then
  echo "Python environment not found: $PYTHON_BIN" >&2
  exit 1
fi

if [ ! -f "$REPORT_PY" ]; then
  echo "HCC report script not found: $REPORT_PY" >&2
  exit 1
fi

if [ ! -f "$SPLIT_INFO" ]; then
  echo "HCC split file not found: $SPLIT_INFO" >&2
  exit 1
fi

cd "$REPO_ROOT" || exit 1

SUCCESS=0
PARTIAL=0
FAILED=0
SUMMARY=()

log "HCC source-only evaluation start"
log "GPU=$GPU (physical GPU1)"
log "methods=${#METHODS[@]}"
log "result_root=$RESULT_ROOT"

for idx in "${!METHODS[@]}"; do
  METHOD="${METHODS[$idx]}"
  TRAINER="${TRAINERS[$idx]}"
  N=$((idx + 1))
  CHECKPOINT="$DATASET003_ROOT/${TRAINER}__nnUNetPlans__3d_fullres/fold_0/checkpoint_best.pth"
  METHOD_DIR="$RESULT_ROOT/$METHOD"
  PRED_DIR="$METHOD_DIR/predictions"

  log "[$N/${#METHODS[@]}] START method=$METHOD trainer=$TRAINER"

  if [ ! -s "$CHECKPOINT" ]; then
    log "[$N/${#METHODS[@]}] FAIL missing checkpoint: $CHECKPOINT"
    FAILED=$((FAILED + 1))
    SUMMARY+=("FAIL    $METHOD: missing checkpoint_best.pth")
    continue
  fi

  "$PYTHON_BIN" "$REPORT_PY" \
    --method "$METHOD" \
    --predict \
    --trainer "$TRAINER" \
    --fold 0 \
    --gpu "$GPU" \
    --checkpoint checkpoint_best.pth \
    --dataset 003 \
    --split_info "$SPLIT_INFO" \
    --ext_val_root "$EXT_VAL_ROOT" \
    --result_root "$RESULT_ROOT"

  RUN_STATUS=$?
  PRED_COUNT=$(find "$PRED_DIR" -maxdepth 1 -type f -name 'HCC_*.nii.gz' 2>/dev/null | wc -l)
  PNG_COUNT=$(find "$METHOD_DIR/test_viz" -type f -name '*.png' 2>/dev/null | wc -l)
  SUMMARY_OK=no
  REPORT_OK=no
  [ -s "$PRED_DIR/summary.json" ] && SUMMARY_OK=yes
  [ -s "$METHOD_DIR/report_custom.txt" ] && REPORT_OK=yes

  log "[$N/${#METHODS[@]}] AUDIT predictions=$PRED_COUNT/$EXPECTED_CASES summary=$SUMMARY_OK report=$REPORT_OK png=$PNG_COUNT exit=$RUN_STATUS"

  if [ "$RUN_STATUS" -eq 0 ] && [ "$PRED_COUNT" -eq "$EXPECTED_CASES" ] && \
     [ "$SUMMARY_OK" = yes ] && [ "$REPORT_OK" = yes ] && [ "$PNG_COUNT" -gt 0 ]; then
    SUCCESS=$((SUCCESS + 1))
    SUMMARY+=("DONE    $METHOD: predictions=$PRED_COUNT summary=yes report=yes png=$PNG_COUNT")
  elif [ "$PRED_COUNT" -gt 0 ] || [ "$SUMMARY_OK" = yes ] || [ "$REPORT_OK" = yes ] || [ "$PNG_COUNT" -gt 0 ]; then
    PARTIAL=$((PARTIAL + 1))
    SUMMARY+=("PARTIAL $METHOD: predictions=$PRED_COUNT summary=$SUMMARY_OK report=$REPORT_OK png=$PNG_COUNT exit=$RUN_STATUS")
  else
    FAILED=$((FAILED + 1))
    SUMMARY+=("FAIL    $METHOD: no usable artifacts exit=$RUN_STATUS")
  fi
done

log "HCC source-only evaluation finished"
log "success=$SUCCESS partial=$PARTIAL failed=$FAILED total=${#METHODS[@]}"
for line in "${SUMMARY[@]}"; do
  echo "  $line"
done

if [ "$SUCCESS" -ne "${#METHODS[@]}" ]; then
  exit 1
fi
