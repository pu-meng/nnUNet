#!/usr/bin/env bash
# HCCReferencedCT v2 source-only 验证：
#   GPU 0 -> MedNeXt_MHA  (standard MHA + standard MLP)
#   GPU 1 -> MedNeXt_MLA  (low-rank KV MLA + standard MLP)
#
# 两个方法并行执行，均使用 Dataset003_Liver/fold_0/checkpoint_best.pth。
# 默认生成 21 例 NIfTI 预测、summary.json、report_custom.txt 和 test_viz PNG。
# 断点重跑时会复用已完整预测，补算 summary、报告和可视化。

set -uo pipefail

REPO_ROOT="/home/PuMengYu/nnUNet"
WORKSPACE="/home/PuMengYu/nnUNet_workspace"
PYTHON_BIN="/home/PuMengYu/anaconda3/envs/medseg/bin/python"
REPORT_PY="$REPO_ROOT/pumengyu/ext_val/05_gen_hcc_test_report.py"
SPLIT_INFO="$WORKSPACE/preprocessed/Dataset013_HCCReferencedCT/split_info_701020_stratified_v2.json"
RESULT_ROOT="$WORKSPACE/results_v2/Dataset013_HCCReferencedCT/source_only"
LOG_ROOT="$WORKSPACE/results_v2/_meta/logs/hcc_mha_mla_gpu01"

export nnUNet_raw="$WORKSPACE/raw"
export nnUNet_preprocessed="$WORKSPACE/preprocessed"
export nnUNet_results="$WORKSPACE/results_v2"

MHA_METHOD="MedNeXt_MHA"
MHA_TRAINER="nnUNetTrainer_MedNeXt_MHA"
MHA_GPU=0
MHA_EXT_ROOT="$WORKSPACE/external_val/hcc_referenced_ct_test_v2_mha"

MLA_METHOD="MedNeXt_MLA"
MLA_TRAINER="nnUNetTrainer_MedNeXt_MLA"
MLA_GPU=1
MLA_EXT_ROOT="$WORKSPACE/external_val/hcc_referenced_ct_test_v2_mla"

checkpoint_path() {
  local trainer="$1"
  printf '%s/results_v2/Dataset003_Liver/%s__nnUNetPlans__3d_fullres/fold_0/checkpoint_best.pth' \
    "$WORKSPACE" "$trainer"
}

fail_preflight() {
  echo "[preflight failed] $*" >&2
  exit 1
}

[ -x "$PYTHON_BIN" ] || fail_preflight "Python not executable: $PYTHON_BIN"
[ -f "$REPORT_PY" ] || fail_preflight "Missing report script: $REPORT_PY"
[ -f "$SPLIT_INFO" ] || fail_preflight "Missing HCC split: $SPLIT_INFO"
[ -f "$(checkpoint_path "$MHA_TRAINER")" ] || fail_preflight "Missing MHA checkpoint_best.pth"
[ -f "$(checkpoint_path "$MLA_TRAINER")" ] || fail_preflight "Missing MLA checkpoint_best.pth"

mkdir -p "$RESULT_ROOT" "$LOG_ROOT"
cd "$REPO_ROOT" || exit 1

run_method() {
  local method="$1"
  local trainer="$2"
  local gpu="$3"
  local ext_root="$4"
  local log_file="$5"

  echo "[start] method=$method trainer=$trainer physical_gpu=$gpu"
  echo "[log]   $log_file"

  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" "$REPORT_PY" \
    --method "$method" \
    --predict \
    --trainer "$trainer" \
    --fold 0 \
    --gpu "$gpu" \
    --checkpoint checkpoint_best.pth \
    --dataset 003 \
    --split_info "$SPLIT_INFO" \
    --ext_val_root "$ext_root" \
    --result_root "$RESULT_ROOT" \
    >"$log_file" 2>&1
}

verify_method() {
  local method="$1"
  local method_dir="$RESULT_ROOT/$method"
  local pred_dir="$method_dir/predictions"
  local pred_count=0
  local png_count=0

  if [ -d "$pred_dir" ]; then
    pred_count=$(find "$pred_dir" -maxdepth 1 -type f -name '*.nii.gz' | wc -l)
  fi
  if [ -d "$method_dir/test_viz" ]; then
    png_count=$(find "$method_dir/test_viz" -type f -name '*.png' | wc -l)
  fi

  echo "[artifacts] $method predictions=$pred_count/21 summary=$([ -f "$pred_dir/summary.json" ] && echo yes || echo no) report=$([ -f "$method_dir/report_custom.txt" ] && echo yes || echo no) viz_png=$png_count"

  [ "$pred_count" -eq 21 ] && \
    [ -f "$pred_dir/summary.json" ] && \
    [ -f "$method_dir/report_custom.txt" ] && \
    [ "$png_count" -gt 0 ] && \
    grep -q '^checkpoint: checkpoint_best.pth' "$method_dir/report_custom.txt" && \
    grep -q '^model_source: Dataset003_Liver source-only model' "$method_dir/report_custom.txt"
}

STAMP=$(date '+%Y%m%d_%H%M%S')
MHA_LOG="$LOG_ROOT/${STAMP}_gpu0_mednext_mha.log"
MLA_LOG="$LOG_ROOT/${STAMP}_gpu1_mednext_mla.log"

run_method "$MHA_METHOD" "$MHA_TRAINER" "$MHA_GPU" "$MHA_EXT_ROOT" "$MHA_LOG" &
MHA_PID=$!
run_method "$MLA_METHOD" "$MLA_TRAINER" "$MLA_GPU" "$MLA_EXT_ROOT" "$MLA_LOG" &
MLA_PID=$!

echo "[parallel] MHA pid=$MHA_PID on GPU $MHA_GPU"
echo "[parallel] MLA pid=$MLA_PID on GPU $MLA_GPU"
echo "[monitor]  tail -f '$MHA_LOG' '$MLA_LOG'"

wait "$MHA_PID"
MHA_STATUS=$?
wait "$MLA_PID"
MLA_STATUS=$?

echo
echo "================ process status ================"
echo "MedNeXt_MHA exit_code=$MHA_STATUS log=$MHA_LOG"
echo "MedNeXt_MLA exit_code=$MLA_STATUS log=$MLA_LOG"

VERIFY_FAILED=0
if [ "$MHA_STATUS" -eq 0 ] && verify_method "$MHA_METHOD"; then
  echo "[complete] $MHA_METHOD"
else
  echo "[incomplete] $MHA_METHOD; inspect $MHA_LOG" >&2
  VERIFY_FAILED=1
fi

if [ "$MLA_STATUS" -eq 0 ] && verify_method "$MLA_METHOD"; then
  echo "[complete] $MLA_METHOD"
else
  echo "[incomplete] $MLA_METHOD; inspect $MLA_LOG" >&2
  VERIFY_FAILED=1
fi

echo "[results] $RESULT_ROOT"
exit "$VERIFY_FAILED"
