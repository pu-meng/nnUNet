#!/bin/bash
set -e

TAR=/home/PuMengYu/8T/Task03_Liver.tar
OUT=/home/PuMengYu/nnUNet_workspace/raw/Dataset003_Liver/imagesTr
TMP=/tmp/task03_extract

mkdir -p "$OUT"
rm -rf "$TMP" && mkdir -p "$TMP"

echo "=== 解压 imagesTr（只解压图像目录）==="
tar -xf "$TAR" -C "$TMP" Task03_Liver/imagesTr/

echo "=== 重命名：liver_X.nii.gz → liver_X_0000.nii.gz ==="
COUNT=0
for f in "$TMP/Task03_Liver/imagesTr/liver_"*.nii.gz; do
    [ -f "$f" ] || continue
    base=$(basename "$f" .nii.gz)
    cp "$f" "$OUT/${base}_0000.nii.gz"
    COUNT=$((COUNT + 1))
done

rm -rf "$TMP"
echo "完成：$COUNT 个图像文件 → $OUT"
