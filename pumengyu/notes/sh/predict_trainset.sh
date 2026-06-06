#!/bin/bash
# 对训练集跑推理，看训练 case 有没有被模型真正学会
# 注意：Stage2 训练跑完后再执行，不要和训练同时占 GPU

set -e

export CUDA_VISIBLE_DEVICES=1
export RESULTS_FOLDER=/home/PuMengYu/nnUNet_workspace/results_v2

INPUT=/home/PuMengYu/nnUNet_workspace/raw/Dataset003_Liver/imagesTr
OUTPUT=/home/PuMengYu/nnUNet_workspace/results_v2/Dataset003_Liver/nnUNetTrainer_SizeOversampleV2__nnUNetPlans__3d_fullres/fold_0/train_prediction

echo "开始对训练集推理，输出到: $OUTPUT"

nnUNetv2_predict \
    -i  $INPUT \
    -o  $OUTPUT \
    -d  Dataset003_Liver \
    -c  3d_fullres \
    -tr nnUNetTrainer_SizeOversampleV2 \
    -f  0

echo "推理完成，生成报告..."

python3 - <<'EOF'
from pathlib import Path
from pumengyu.tools.analyasis.eval_fold_report import run_eval_report

val_dir = Path("/home/PuMengYu/nnUNet_workspace/results_v2/Dataset003_Liver/nnUNetTrainer_SizeOversampleV2__nnUNetPlans__3d_fullres/fold_0/train_prediction")
gt_dir  = Path("/home/PuMengYu/nnUNet_workspace/preprocessed/Dataset003_Liver/gt_segmentations")
img_dir = Path("/home/PuMengYu/nnUNet_workspace/raw/Dataset003_Liver/imagesTr")

run_eval_report(
    val_dir=val_dir,
    gt_dir=gt_dir,
    img_dir=img_dir,
    no_vis=True,
    min_tumor_size=0,
    out_dir=val_dir.parent,
    report_name="train_report_custom.txt",
)
print("报告已写入 train_report_custom.txt")
EOF
