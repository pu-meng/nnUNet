#!/bin/bash
# Regenerate HCCReferencedCT external-validation test_viz PNGs from existing predictions.
#
# This does NOT run nnUNet prediction. It only reads:
#   results_v2/ExternalVal_HCCReferencedCT/<method>/predictions/*.nii.gz
# and writes:
#   results_v2/ExternalVal_HCCReferencedCT/<method>/test_viz/
#
# Usage:
#   bash pumengyu/ext_val/regen_hcc_test_viz.sh
#   bash pumengyu/ext_val/regen_hcc_test_viz.sh --only NoMirror MedNeXt_MLA
#   bash pumengyu/ext_val/regen_hcc_test_viz.sh --force
#   nohup bash pumengyu/ext_val/regen_hcc_test_viz.sh > /tmp/hcc_test_viz.log 2>&1 &

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPORT_PY="$SCRIPT_DIR/05_gen_hcc_test_report.py"
EXT_ROOT="/home/PuMengYu/nnUNet_workspace/results_v2/ExternalVal_HCCReferencedCT"

GPU=0
MIN_VOXEL=20
FORCE=0
ONLY=()

usage() {
  sed -n '2,13p' "$0"
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --gpu)
      GPU="$2"
      shift 2
      ;;
    --min_voxel)
      MIN_VOXEL="$2"
      shift 2
      ;;
    --force)
      FORCE=1
      shift
      ;;
    --only)
      shift
      while [ "$#" -gt 0 ] && [[ "$1" != --* ]]; do
        ONLY+=("$1")
        shift
      done
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

has_prediction() {
  local method="$1"
  local pred_dir="$EXT_ROOT/$method/predictions"
  [ -d "$pred_dir" ] && find "$pred_dir" -maxdepth 1 -name '*.nii.gz' -print -quit | grep -q .
}

has_viz_png() {
  local method="$1"
  local viz_dir="$EXT_ROOT/$method/test_viz"
  [ -d "$viz_dir" ] && find "$viz_dir" -type f -name '*.png' -print -quit | grep -q .
}

is_requested() {
  local method="$1"
  if [ "${#ONLY[@]}" -eq 0 ]; then
    return 0
  fi
  local item
  for item in "${ONLY[@]}"; do
    if [ "$item" = "$method" ]; then
      return 0
    fi
  done
  return 1
}

METHODS=()
if [ ! -d "$EXT_ROOT" ]; then
  echo "[ERROR] External result root not found: $EXT_ROOT" >&2
  exit 1
fi

while IFS= read -r method_dir; do
  method="$(basename "$method_dir")"
  if is_requested "$method" && has_prediction "$method"; then
    METHODS+=("$method")
  fi
done < <(find "$EXT_ROOT" -mindepth 1 -maxdepth 1 -type d | sort)

if [ "${#METHODS[@]}" -eq 0 ]; then
  echo "[ERROR] No methods with existing predictions matched the request." >&2
  exit 1
fi

cd "$REPO_ROOT" || exit 1

TOTAL=${#METHODS[@]}
DONE=0
SKIPPED=0
FAILED=0
SUMMARY=()

log "HCC test_viz regeneration start"
log "GPU=$GPU min_voxel=$MIN_VOXEL force=$FORCE"
log "methods=$TOTAL"
log "output root=$EXT_ROOT"
echo ""

for idx in "${!METHODS[@]}"; do
  N=$((idx + 1))
  METHOD="${METHODS[$idx]}"

  log "[$N/$TOTAL] START method=$METHOD"

  if [ "$FORCE" -eq 0 ] && has_viz_png "$METHOD"; then
    log "[$N/$TOTAL] SKIP existing test_viz PNGs: $METHOD"
    SKIPPED=$((SKIPPED + 1))
    SUMMARY+=("[$N/$TOTAL] SKIP  $METHOD")
    echo ""
    continue
  fi

  CUDA_VISIBLE_DEVICES="$GPU" python "$REPORT_PY" \
    --method "$METHOD" \
    --gpu "$GPU" \
    --min_voxel "$MIN_VOXEL" \
    --checkpoint checkpoint_best.pth \
    --dataset 003

  STATUS=$?
  if [ "$STATUS" -eq 0 ]; then
    N_PNG=$(find "$EXT_ROOT/$METHOD/test_viz" -type f -name '*.png' 2>/dev/null | wc -l)
    log "[$N/$TOTAL] DONE method=$METHOD png=$N_PNG"
    DONE=$((DONE + 1))
    SUMMARY+=("[$N/$TOTAL] DONE  $METHOD png=$N_PNG")
  else
    log "[$N/$TOTAL] FAIL method=$METHOD exit_code=$STATUS; continue next"
    FAILED=$((FAILED + 1))
    SUMMARY+=("[$N/$TOTAL] FAIL  $METHOD exit_code=$STATUS")
  fi
  echo ""
done

log "=========================================="
log "HCC test_viz regeneration finished"
log "done=$DONE skipped=$SKIPPED failed=$FAILED total=$TOTAL"
log "summary:"
for line in "${SUMMARY[@]}"; do
  echo "  $line"
done
log "results root: $EXT_ROOT"
