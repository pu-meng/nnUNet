#!/bin/bash
GT=/home/PuMengYu/nnUNet_workspace/preprocessed/Dataset003_Liver/gt_segmentations
IMG=/home/PuMengYu/nnUNet_workspace/raw/Dataset003_Liver/imagesTr
RES=/home/PuMengYu/nnUNet_workspace/results/Dataset003_Liver
SCRIPT=/home/PuMengYu/nnUNet/pumengyu/tools/analyasis/eval_fold_report.py

for VAL_DIR in \
  $RES/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/validation \
  $RES/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_1/validation \
  $RES/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_2/validation \
  $RES/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_3/validation \
  $RES/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_4/validation \
  $RES/nnUNetTrainer_CopyPaste__nnUNetPlans__3d_fullres/fold_4/validation \
  $RES/nnUNetTrainer_UFL__nnUNetPlans__3d_fullres/fold_1/validation \
  $RES/nnUNetTrainer_UFL__nnUNetPlans__3d_fullres/fold_2/validation \
  $RES/nnUNetTrainer_UFL__nnUNetPlans__3d_fullres/fold_4/validation
do
  echo "=== $VAL_DIR ==="
  python $SCRIPT --val_dir $VAL_DIR --gt_dir $GT --img_dir $IMG --no_vis --min_tumor_size 100
done
