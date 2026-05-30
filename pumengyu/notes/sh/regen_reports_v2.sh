#!/bin/bash
# 重新生成 results_v2 下所有实验的 validation + test 报告
# 用法：cd /home/PuMengYu/nnUNet && bash pumengyu/notes/sh/regen_reports_v2.sh
set -e

GT=/home/PuMengYu/nnUNet_workspace/preprocessed/Dataset003_Liver/gt_segmentations
IMG=/home/PuMengYu/nnUNet_workspace/raw/Dataset003_Liver/imagesTr
V2=/home/PuMengYu/nnUNet_workspace/results_v2/Dataset003_Liver
EVAL=pumengyu/tools/analyasis/eval_fold_report.py
GEN=pumengyu/tools/gen_test_report.py

TRAINERS=(
  nnUNetTrainer_Baseline
  nnUNetTrainer_SizeOversampleV2
  nnUNetTrainer_SizeOversampleV3
)

echo "====== validation reports ======"
for T in "${TRAINERS[@]}"; do
  VAL=$V2/${T}__nnUNetPlans__3d_fullres/fold_0/validation
  if [ ! -d "$VAL" ]; then
    echo "[SKIP] 目录不存在: $VAL"
    continue
  fi
  echo "--- $T ---"
  python $EVAL --val_dir "$VAL" --gt_dir $GT --img_dir $IMG --no_vis
done

echo ""
echo "====== test reports ======"
for T in "${TRAINERS[@]}"; do
  TEST=$V2/${T}__nnUNetPlans__3d_fullres/fold_0/test_prediction
  if [ ! -d "$TEST" ]; then
    echo "[SKIP] 目录不存在: $TEST"
    continue
  fi
  echo "--- $T ---"
  python $GEN --trainer "$T" --dataset Dataset003_Liver --fold 0
done

echo ""
echo "====== 完成 ======"
