#!/bin/bash
# 补跑缺失的外部验证（IRCADb）
# 用法：bash pumengyu/ext_val/run_missing_ext_val.sh
# 固定 GPU=1，一个接一个顺序执行

set -e
GPU=1
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

run_method() {
    local METHOD=$1
    local TRAINER=$2
    local PREDICT=${3:-1}   # 1=先推理, 0=只生成report（predictions已存在）

    log "========== 开始: $METHOD =========="
    if [ "$PREDICT" -eq 1 ]; then
        CUDA_VISIBLE_DEVICES=$GPU python "$SCRIPT_DIR/03_gen_method_report.py" \
            --method "$METHOD" \
            --predict \
            --trainer "$TRAINER" \
            --fold 0 \
            --gpu $GPU
    else
        CUDA_VISIBLE_DEVICES=$GPU python "$SCRIPT_DIR/03_gen_method_report.py" \
            --method "$METHOD" \
            --gpu $GPU
    fi
    log "========== 完成: $METHOD =========="
    echo ""
}

cd "$REPO_ROOT"

# ── 需要推理+报告 ──────────────────────────────────────────────────────────
run_method "MedNeXt_SizeOV4"      "nnUNetTrainer_MedNeXt_SizeOV4"                  1
run_method "MedNeXt"               "nnUNetTrainer_MedNeXt"                           1
run_method "MedNeXt_MLA_MoE_SizeOV4"  "nnUNetTrainer_MedNeXt_MLA_MoE_SizeOV4"      1
run_method "MLAUNet_MoE_IB7_SizeOV4" "nnUNetTrainer_MLAUNet_MoE_IB7_SizeOversampleV4" 1
run_method "SwinUNETR"             "nnUNetTrainer_SwinUNETR"                         1
run_method "nnFormer"              "nnUNetTrainer_nnFormer"                          1

# ── MLAUNet：predictions 目录存在但为空，需要重跑推理 ─────────────────────
run_method "MLAUNet"               "nnUNetTrainer_MLAUNet"                           1

log "全部完成！"
log "查看汇总: python pumengyu/ext_val/04_batch_ext_val.py --only MedNeXt_SizeOV4 MedNeXt MedNeXt_MLA_MoE_SizeOV4 MLAUNet_MoE_IB7_SizeOV4 SwinUNETR nnFormer MLAUNet"
