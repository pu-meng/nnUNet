#!/usr/bin/env bash
# Missing IRCADb external validation, GPU0 parallel group A.
#
# Run together with:
#   bash pumengyu/notes/sh/run_missing_external_v2_gpu0_b.sh
#
# Usage:
#   bash pumengyu/notes/sh/run_missing_external_v2_gpu0_a.sh
#   NO_VIS=1 bash pumengyu/notes/sh/run_missing_external_v2_gpu0_a.sh
#   FORCE=1 bash pumengyu/notes/sh/run_missing_external_v2_gpu0_a.sh

set -u

GPU=${GPU:-0}
NO_VIS=${NO_VIS:-0}
FORCE=${FORCE:-0}
STOP_ON_FAIL=${STOP_ON_FAIL:-1}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
REPORT_PY="$REPO_ROOT/pumengyu/ext_val/03_gen_method_report.py"
RESULT_ROOT="/home/PuMengYu/nnUNet_workspace/results_v2"
INTERNAL_ROOT="$RESULT_ROOT/Dataset003_Liver"
EXTERNAL_ROOT="$RESULT_ROOT/ExternalVal_IRCADb"

log() {
    echo "[$(date '+%F %T')] $*"
}

has_report() {
    local method=$1
    [ -f "$EXTERNAL_ROOT/$method/report_custom.txt" ]
}

checkpoint_exists() {
    local trainer=$1
    local checkpoint=$2
    [ -f "$INTERNAL_ROOT/${trainer}__nnUNetPlans__3d_fullres/fold_0/$checkpoint" ]
}

run_method() {
    local method=$1
    local trainer=$2
    local checkpoint=${3:-checkpoint_final.pth}

    if [ "$FORCE" != "1" ] && has_report "$method"; then
        log "[SKIP] $method already has report_custom.txt"
        return 0
    fi

    if ! checkpoint_exists "$trainer" "$checkpoint"; then
        log "[SKIP] $method checkpoint missing: $trainer/fold_0/$checkpoint"
        return 0
    fi

    local cmd=(
        python "$REPORT_PY"
        --method "$method"
        --predict
        --trainer "$trainer"
        --fold 0
        --gpu "$GPU"
        --checkpoint "$checkpoint"
    )

    if [ "$NO_VIS" = "1" ]; then
        cmd+=(--no_vis)
    fi

    log "========== RUN $method =========="
    log "trainer=$trainer checkpoint=$checkpoint GPU=$GPU"
    CUDA_VISIBLE_DEVICES="$GPU" "${cmd[@]}"
    local status=$?

    if [ $status -ne 0 ]; then
        log "[FAIL] $method exit=$status"
        if [ "$STOP_ON_FAIL" = "1" ]; then
            exit $status
        fi
        return $status
    fi

    log "[DONE] $method"
    echo ""
}

print_summary() {
    log "External validation summary by Overall"
    python - <<'PY'
import math
import re
from pathlib import Path

root = Path("/home/PuMengYu/nnUNet_workspace/results_v2/ExternalVal_IRCADb")
rows = []
for report in sorted(root.glob("*/report_custom.txt")):
    text = report.read_text(errors="ignore")
    def val(pattern):
        m = re.search(pattern, text, re.S)
        return float(m.group(1)) if m else math.nan
    rows.append((
        report.parent.name,
        val(r"Overall\s*:.*?=\s*([0-9.]+)"),
        val(r"Liver\s*\n\s*Dice: mean=([0-9.]+)"),
        val(r"Tumor 综合指标.*?\n\s*无肿瘤.*?\n\s*Dice\s*: mean=([0-9.]+)"),
    ))

for rank, (method, overall, liver, tumor) in enumerate(
    sorted(rows, key=lambda x: x[1], reverse=True), 1
):
    print(f"{rank:02d}. {method:32s} overall={overall:.4f} liver={liver:.4f} tumor={tumor:.4f}")
PY
}

cd "$REPO_ROOT" || exit 1

log "Repo: $REPO_ROOT"
log "Group=A GPU=$GPU NO_VIS=$NO_VIS FORCE=$FORCE STOP_ON_FAIL=$STOP_ON_FAIL"
echo ""

run_method "MedNeXt_MLA"              "nnUNetTrainer_MedNeXt_MLA"                 "checkpoint_final.pth"
run_method "MLAUNet_1500"             "nnUNetTrainer_MLAUNet_1500"                "checkpoint_final.pth"
run_method "DWSepRes4_MoE_SizeOV4"    "nnUNetTrainer_DWSepRes4_MoE_SizeOV4"       "checkpoint_best.pth"

echo ""
print_summary
log "Group A finished."
