#!/bin/bash
# 打包3个关键case的CT/GT/预测文件，方便下载到Windows可视化
set -e

OUT=/home/PuMengYu/viz_download
RAW=/home/PuMengYu/nnUNet_workspace/raw/Dataset003_Liver/imagesTr
GT=/home/PuMengYu/nnUNet_workspace/preprocessed/Dataset003_Liver/gt_segmentations
PRED=/home/PuMengYu/nnUNet_workspace/results_v2/Dataset003_Liver/nnUNetTrainer_Baseline__nnUNetPlans__3d_fullres/fold_0/test_prediction

CASES="liver_41 liver_127 liver_89"

mkdir -p $OUT/ct $OUT/gt $OUT/pred_baseline

for case in $CASES; do
    echo "复制 $case ..."
    cp $RAW/${case}_0000.nii.gz  $OUT/ct/
    cp $GT/${case}.nii.gz        $OUT/gt/
    cp $PRED/${case}.nii.gz      $OUT/pred_baseline/
done

echo ""
echo "完成，文件在: $OUT"
echo "目录结构:"
find $OUT -name "*.nii.gz" | sort
