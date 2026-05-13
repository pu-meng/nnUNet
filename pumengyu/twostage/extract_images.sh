#!/bin/bash
# 只从 tar 解压图像（跳过 labelsTr，标签用 gt_segmentations）
set -e

TAR=/home/PuMengYu/8T/Task03_Liver.tar
OUT=/home/PuMengYu/nnUNet_workspace/raw/Dataset003_Liver/imagesTr

mkdir -p "$OUT"

echo "=== 解压 imagesTr（只解压图像，跳过标签）==="
tar -xf "$TAR" \
    --wildcards "Task03_Liver/imagesTr/*.nii.gz" \
    -C /tmp/

echo "=== 重命名：liver_X.nii.gz → liver_X_0000.nii.gz ==="
for f in /tmp/Task03_Liver/imagesTr/liver_*.nii.gz; do
    base=$(basename "$f" .nii.gz)
    mv "$f" "$OUT/${base}_0000.nii.gz"
done

rm -rf /tmp/Task03_Liver
echo "完成：$(ls $OUT | wc -l) 个图像文件"
