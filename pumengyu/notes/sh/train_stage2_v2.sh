#!/bin/bash
# Stage2 FP 抑制 v2 训练
# Loss: CE+Dice + NoTumorFPPenalty
# 热启动: Stage1 权重 (_orig_mod 已修复, 多key扩展已修复)
# 运行: bash pumengyu/notes/sh/train_stage2_v2.sh

set -e

LOG=/tmp/stage2_v2_$(date +%Y%m%d_%H%M%S).log

echo "日志: $LOG"
echo "开始时间: $(date)"

nnUNet_compile=0 \
CUDA_VISIBLE_DEVICES=1 \
RESULTS_FOLDER=/home/PuMengYu/nnUNet_workspace/results_v2 \
nnUNetv2_train Dataset003_Liver 3d_fullres 0 -tr Tr_Stage2_FPSup_v2 \
2>&1 | tee "$LOG"

echo "结束时间: $(date)"
