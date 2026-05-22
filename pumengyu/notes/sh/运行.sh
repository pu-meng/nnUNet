CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 4 3d_fullres 4 -tr nnUNetTrainer_ConvBoundaryLoss

# ── LoCo 消融实验（逐步加 BCE / ICE）────────────────────────────────────────
# Step 1: 只加 BCE（边界对比增强）
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 4 3d_fullres 4 -tr nnUNetTrainer_BCE

# Step 2: 只加 ICE（类间对比增强）
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 4 3d_fullres 4 -tr nnUNetTrainer_ICE

# Step 3: BCE + ICE 合用
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 4 3d_fullres 4 -tr nnUNetTrainer_BCE_ICE

#step4 two-stage 的扰动训练
  nnUNetv2_train 4 3d_fullres 0 -tr nnUNetTrainer_TwoStageJitter
# 手动补生成 report_custom.txt（训练结束后 conv_trainer 未自动触发时使用）
cd /home/PuMengYu/nnUNet && python - <<'EOF'
from pumengyu.tools.analyasis.auto_report import run_auto_report

fold_dir = "/home/PuMengYu/nnUNet_workspace/results/Dataset004_LiverTumor/nnUNetTrainer_ConvBoundaryLoss__nnUNetPlans__3d_fullres/fold_4"
gt_dir   = "/home/PuMengYu/nnUNet_workspace/preprocessed/Dataset004_LiverTumor/gt_segmentations"
img_dir  = "/home/PuMengYu/nnUNet_workspace/raw/Dataset004_LiverTumor/imagesTr"

run_auto_report(fold_dir, gt_dir, img_dir)
EOF


CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 3 3d_fullres 3 -tr nnUNetTrainer_UFL --npz
for i in 1 2 3 0; do
  CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 3 3d_fullres $i -tr nnUNetTrainer_UFL
done