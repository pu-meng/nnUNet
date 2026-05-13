#!/bin/bash
set -e

WORKSPACE=/home/PuMengYu/nnUNet_workspace
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== 生成 Dataset004 (GT 裁剪, margin=30mm) ==="
python "$SCRIPT_DIR/create_dataset.py" \
    --workspace "$WORKSPACE" \
    --dataset003_id 3 \
    --dataset004_id 4 \
    --margin_mm 30

echo ""
echo "=== 预处理 Dataset004 ==="
nnUNetv2_plan_and_preprocess -d 4 --verify_dataset_integrity
