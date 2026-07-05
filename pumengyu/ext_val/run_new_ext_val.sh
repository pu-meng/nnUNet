#!/bin/bash
# 补跑新的外部验证（IRCADb）
#
# 任务：
#   GPU0（立即）: DeepPlainResGN, DeepPlainResGN_SizeOV4, MLA_GK5_V4
#   GPU1（等训练完）: DeepResGN_MLA（训练约 19:30 完成，自动等待）
#
# 用法：
#   bash pumengyu/ext_val/run_new_ext_val.sh
#   或后台运行：nohup bash pumengyu/ext_val/run_new_ext_val.sh > /tmp/ext_val_new.log 2>&1 &

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPORT_PY="$SCRIPT_DIR/03_gen_method_report.py"

DEEP_RES_GN_MLA_CKPT="$nnUNet_results/Dataset003_Liver/nnUNetTrainer_DeepResGN_MLA__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

run_ext_val() {
    local METHOD=$1
    local TRAINER=$2
    local GPU=$3
    log "========== 开始外部验证: $METHOD (GPU$GPU) =========="
    CUDA_VISIBLE_DEVICES=$GPU python "$REPORT_PY" \
        --method   "$METHOD"  \
        --predict             \
        --trainer  "$TRAINER" \
        --fold     0          \
        --gpu      $GPU
    log "========== 完成: $METHOD =========="
    echo ""
}

cd "$REPO_ROOT"

# ── GPU0：立即开始 ────────────────────────────────────────────────────────
log "=== GPU0 任务开始 ==="

run_ext_val "DeepPlainResGN"        "nnUNetTrainer_DeepPlainResGN"        0
run_ext_val "DeepPlainResGN_SizeOV4" "nnUNetTrainer_DeepPlainResGN_SizeOV4" 0
run_ext_val "MLA_GK5_V4"            "nnUNetTrainer_MLA_GK5_V4"            0

log "=== GPU0 全部完成 ==="
echo ""

# ── GPU1：等 DeepResGN_MLA 训练结束 ──────────────────────────────────────
log "等待 DeepResGN_MLA 训练完成（检测 checkpoint_final.pth）..."
log "路径: $DEEP_RES_GN_MLA_CKPT"

WAIT_INTERVAL=60   # 每 60s 轮询一次
WAITED=0
MAX_WAIT=14400     # 最多等 4 小时

while [ ! -f "$DEEP_RES_GN_MLA_CKPT" ]; do
    if [ $WAITED -ge $MAX_WAIT ]; then
        log "错误：等待超过 4 小时，未检测到 checkpoint_final.pth，退出。"
        exit 1
    fi
    log "训练中...已等待 ${WAITED}s，继续等待..."
    sleep $WAIT_INTERVAL
    WAITED=$((WAITED + WAIT_INTERVAL))
done

log "DeepResGN_MLA 训练完成！开始外部验证..."
run_ext_val "DeepResGN_MLA" "nnUNetTrainer_DeepResGN_MLA" 1

log "=========================================="
log "全部外部验证完成！"
log "查看结果："
log "  cat $nnUNet_results/ExternalVal_IRCADb/DeepPlainResGN/report_custom.txt"
log "  cat $nnUNet_results/ExternalVal_IRCADb/DeepResGN_MLA/report_custom.txt"
