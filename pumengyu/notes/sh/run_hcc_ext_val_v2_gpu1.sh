#!/bin/bash
# 批量运行 HCCReferencedCT stratified v2 held-out test 外部验证。
#
# 口径：
#   - 模型：results_v2/Dataset003_Liver 里已训练好的 checkpoint_best.pth
#   - 测试集：Dataset013_HCCReferencedCT 的 split_info_701020_stratified_v2.json 中 test 21 cases
#   - 不使用 HCC train/val，不覆盖旧 ExternalVal_HCCReferencedCT 结果
#   - 输出：results_v2/Dataset013_HCCReferencedCT/source_only/<method>/
#   - 默认生成 test_viz，便于快速肉眼复核
#   - GPU：1
#
# 用法：
#   bash pumengyu/notes/sh/run_hcc_ext_val_v2_gpu1.sh
#   nohup bash pumengyu/notes/sh/run_hcc_ext_val_v2_gpu1.sh > /tmp/hcc_ext_val_v2_gpu1.log 2>&1 &

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
REPORT_PY="$REPO_ROOT/pumengyu/ext_val/05_gen_hcc_test_report.py"

SPLIT_INFO="/home/PuMengYu/nnUNet_workspace/preprocessed/Dataset013_HCCReferencedCT/split_info_701020_stratified_v2.json"
EXT_VAL_ROOT="/home/PuMengYu/nnUNet_workspace/external_val/hcc_referenced_ct_test_v2"
EXT_ROOT="/home/PuMengYu/nnUNet_workspace/results_v2/Dataset013_HCCReferencedCT/source_only"
GPU=1

METHODS=(
  "Baseline"
  "DeepDWIBMedConfig"
  "DeepDWIBResGN"
  "DeepPlainResGN"
  "DeepPlainResGN_SizeOV4"
  "DeepResGN_MLA"
  "DWSepRes4_MoE_SizeOV4"
  "MLAUNet"
  "MLAUNet_1500"
  "MoE"
  "MLAUNet_MoE_IB7_SizeOV4"
  "MoE_SizeOV2"
  "MoE_SizeOV4"
  "MoE_SizeOV5"
  "MLA_GK5_V4"
  "MedNeXt"
  "MedNeXt_MLA"
  "MedNeXt_MLA_MoE_FPSafe"
  "MedNeXt_MLA_MoE_SizeOV4"
  "MedNeXt_SizeOV4"
  "NoMirror"
  "SizeOV2"
  "SizeOV3"
  "SizeOV3_NoMirror"
  "SwinUNETR"
  "nnFormer"
)

TRAINERS=(
  "nnUNetTrainer_Baseline"
  "nnUNetTrainer_DeepDWIBMedConfig"
  "nnUNetTrainer_DeepDWIBResGN"
  "nnUNetTrainer_DeepPlainResGN"
  "nnUNetTrainer_DeepPlainResGN_SizeOV4"
  "nnUNetTrainer_DeepResGN_MLA"
  "nnUNetTrainer_DWSepRes4_MoE_SizeOV4"
  "nnUNetTrainer_MLAUNet"
  "nnUNetTrainer_MLAUNet_1500"
  "nnUNetTrainer_MLAUNet_MoE"
  "nnUNetTrainer_MLAUNet_MoE_IB7_SizeOversampleV4"
  "nnUNetTrainer_MLAUNet_MoE_SizeOversampleV2"
  "nnUNetTrainer_MLAUNet_MoE_SizeOversampleV4"
  "nnUNetTrainer_MLAUNet_MoE_SizeOversampleV5"
  "nnUNetTrainer_MLA_GK5_V4"
  "nnUNetTrainer_MedNeXt"
  "nnUNetTrainer_MedNeXt_MLA_MoE"
  "nnUNetTrainer_MedNeXt_MLA_MoE_FPSafe"
  "nnUNetTrainer_MedNeXt_MLA_MoE_SizeOV4"
  "nnUNetTrainer_MedNeXt_SizeOV4"
  "nnUNetTrainer_NoMirror"
  "nnUNetTrainer_SizeOversampleV2"
  "nnUNetTrainer_SizeOversampleV3"
  "nnUNetTrainer_SizeOversampleV3_NoMirror"
  "nnUNetTrainer_SwinUNETR"
  "nnUNetTrainer_nnFormer"
)

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

has_done_result() {
  local method="$1"
  local pred_dir="$EXT_ROOT/$method/predictions"
  local report="$EXT_ROOT/$method/report_custom.txt"
  [ -f "$report" ] && [ -d "$pred_dir" ] && find "$pred_dir" -maxdepth 1 -name '*.nii.gz' -print -quit | grep -q .
}

TOTAL=${#METHODS[@]}
SUCCESS=0
SKIPPED=0
FAILED=0
LAST_INDEX=0
SUMMARY=()

if [ ! -f "$SPLIT_INFO" ]; then
  echo "Missing split info: $SPLIT_INFO" >&2
  exit 1
fi

cd "$REPO_ROOT" || exit 1

log "HCC external validation v2 batch start"
log "GPU=$GPU"
log "split_info=$SPLIT_INFO"
log "external_val_root=$EXT_VAL_ROOT"
log "output root=$EXT_ROOT"
log "total experiments=$TOTAL"
echo ""

for idx in "${!METHODS[@]}"; do
  N=$((idx + 1))
  METHOD="${METHODS[$idx]}"
  TRAINER="${TRAINERS[$idx]}"
  LAST_INDEX=$N

  log "[$N/$TOTAL] START method=$METHOD trainer=$TRAINER"

  if has_done_result "$METHOD"; then
    log "[$N/$TOTAL] SKIP existing v2 report + predictions: $METHOD"
    SKIPPED=$((SKIPPED + 1))
    SUMMARY+=("[$N/$TOTAL] SKIP  $METHOD")
    echo ""
    continue
  fi

  CUDA_VISIBLE_DEVICES=$GPU python "$REPORT_PY" \
    --method "$METHOD" \
    --predict \
    --trainer "$TRAINER" \
    --fold 0 \
    --gpu "$GPU" \
    --checkpoint checkpoint_best.pth \
    --dataset 003 \
    --split_info "$SPLIT_INFO" \
    --ext_val_root "$EXT_VAL_ROOT" \
    --result_root "$EXT_ROOT"

  STATUS=$?
  if [ $STATUS -eq 0 ]; then
    log "[$N/$TOTAL] DONE method=$METHOD"
    SUCCESS=$((SUCCESS + 1))
    SUMMARY+=("[$N/$TOTAL] DONE  $METHOD")
  else
    log "[$N/$TOTAL] FAIL method=$METHOD exit_code=$STATUS; continue next"
    FAILED=$((FAILED + 1))
    SUMMARY+=("[$N/$TOTAL] FAIL  $METHOD exit_code=$STATUS")
  fi
  echo ""
done

log "=========================================="
log "HCC external validation v2 batch finished"
log "executed_until=$LAST_INDEX/$TOTAL"
log "success=$SUCCESS skipped=$SKIPPED failed=$FAILED total=$TOTAL"
log "summary:"
for line in "${SUMMARY[@]}"; do
  echo "  $line"
done
log "results root: $EXT_ROOT"
