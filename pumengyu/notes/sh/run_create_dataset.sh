#!/bin/bash
set -e

WORKSPACE=/home/PuMengYu/nnUNet_workspace
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Step 1: 解压原始图像（gt_segmentations 有标签，只需解压图像）==="
bash "$SCRIPT_DIR/extract_images.sh"

echo ""
echo "=== Step 2: GT 裁剪生成 Dataset004 (margin=30mm) ==="
python "$SCRIPT_DIR/create_dataset.py" \
    --workspace "$WORKSPACE" \
    --dataset003_id 3 \
    --dataset004_id 4 \
    --margin_mm 30

echo ""
echo "=== Step 3: 预处理 Dataset004 ==="
nnUNetv2_plan_and_preprocess -d 4 --verify_dataset_integrity
