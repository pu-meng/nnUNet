#!/bin/bash
# 体积过滤后处理扫描
# 对已完成验证的各 fold 预测结果，扫描多个体积阈值，输出综合Dice/FP率/大小类别Dice
# 结果写入 pumengyu/notes/实验结果分析/volume_filter_scan/ 目录

set -e
cd /home/PuMengYu/nnUNet

RESULTS=/home/PuMengYu/nnUNet_workspace/results/Dataset003_Liver
GT_DIR=/home/PuMengYu/nnUNet_workspace/preprocessed/Dataset003_Liver/gt_segmentations
OUT_DIR=/home/PuMengYu/nnUNet/pumengyu/notes/实验结果分析/volume_filter_scan
SCRIPT=pumengyu/tools/analyasis/postprocess_volume_scan.py

mkdir -p "$OUT_DIR"

echo "=========================================="
echo "体积过滤阈值扫描"
echo "=========================================="

# ---------- Baseline ----------
echo "[1/4] Baseline fold_4..."
python $SCRIPT \
  --val_dir "$RESULTS/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_4/validation" \
  --gt_dir  "$GT_DIR" \
  --out_txt "$OUT_DIR/baseline_fold4.txt"

echo "[2/4] Baseline fold_0..."
python $SCRIPT \
  --val_dir "$RESULTS/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/validation" \
  --gt_dir  "$GT_DIR" \
  --out_txt "$OUT_DIR/baseline_fold0.txt"

# ---------- SizeOversample ----------
echo "[3/4] SizeOversample fold_4..."
python $SCRIPT \
  --val_dir "$RESULTS/nnUNetTrainer_SizeOversample__nnUNetPlans__3d_fullres/fold_4/validation" \
  --gt_dir  "$GT_DIR" \
  --out_txt "$OUT_DIR/sizeoversample_fold4.txt"

# ---------- UFL ----------
echo "[4/4] UFL fold_4..."
python $SCRIPT \
  --val_dir "$RESULTS/nnUNetTrainer_UFL__nnUNetPlans__3d_fullres/fold_4/validation" \
  --gt_dir  "$GT_DIR" \
  --out_txt "$OUT_DIR/ufl_fold4.txt"

echo ""
echo "完成，结果在 $OUT_DIR/"
ls -lh "$OUT_DIR/"
