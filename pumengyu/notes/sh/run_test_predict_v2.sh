#!/bin/bash
# 对 SizeOversampleV2 补跑 test 推理（26个 held-out case）
# 结果保存到：
#   fold_0/test_prediction/        （标准位置）
#   /home/PuMengYu/nnUNet_workspace/analysis/SizeOversampleV2/test_report_custom.txt （汇总位置）
# 运行：bash pumengyu/notes/sh/run_test_predict_v2.sh

set -e
cd /home/PuMengYu/nnUNet

ANALYSIS_DIR=/home/PuMengYu/nnUNet_workspace/analysis/SizeOversampleV2
mkdir -p "$ANALYSIS_DIR"

echo "===== [1/2] 推理 test 26 cases ====="
CUDA_VISIBLE_DEVICES=0 \
RESULTS_FOLDER=/home/PuMengYu/nnUNet_workspace/results_v2 \
python pumengyu/tools/run_internal_test.py \
    --trainer nnUNetTrainer_SizeOversampleV2 \
    --dataset Dataset003_Liver \
    --fold    0 \
    --gpu     0

echo ""
echo "===== [2/2] 拷贝报告到 analysis 目录 ====="
cp /home/PuMengYu/nnUNet_workspace/results_v2/Dataset003_Liver/nnUNetTrainer_SizeOversampleV2__nnUNetPlans__3d_fullres/fold_0/test_report_custom.txt \
   "$ANALYSIS_DIR/test_report_custom.txt"

echo ""
echo "全部完成！报告位置："
echo "  $ANALYSIS_DIR/test_report_custom.txt"
