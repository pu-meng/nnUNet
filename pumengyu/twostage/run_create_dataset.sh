#!/bin/bash
set -e

WORKSPACE=/home/PuMengYu/nnUNet_workspace
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== 生成 Dataset004 (Stage-1 CV预测 → 肝脏ROI裁剪) ==="
python "$SCRIPT_DIR/create_dataset.py" \
    --workspace "$WORKSPACE" \
    --dataset003_id 3 \
    --dataset004_id 4 \
    --margin_mm 30 \
    --folds 0 1 2 3 4 \
    --trainer nnUNetTrainer \
    --plans nnUNetPlans \
    --config 3d_fullres

echo ""
echo "=== 预处理 Dataset004 ==="
nnUNetv2_plan_and_preprocess -d 4 --verify_dataset_integrity
