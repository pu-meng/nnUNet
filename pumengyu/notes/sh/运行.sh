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


CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 3 3d_fullres 4 -tr nnUNetTrainer_UFL_v2

CUDA_VISIBLE_DEVICES=1  nnUNetv2_train 3 3d_fullres 4 -tr nnUNetTrainer_CopyPaste_v2


CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 3 3d_fullres 4 -tr nnUNetTrainer_CopyPaste_v2 2>&1 | tee /tmp/copypaste_v2_debug.txt


CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 3 3d_fullres 4 -tr nnUNetTrainer_CopyPaste_v2 



CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 3 3d_fullres 1 -tr nnUNetTrainer_CopyPaste_Diff
for i in 1 2 3 0; do
  CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 3 3d_fullres $i -tr nnUNetTrainer_CopyPaste_Diff
done

  nnUNetv2_train Dataset003_Liver 3d_fullres 4 -tr nnUNetTrainer_UFL_delta06


for i in 3 0; do
  CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 3 3d_fullres $i -tr nnUNetTrainer_UFL_delta06
done



  # 消融：仅 DOS（对照 baseline，看难度过采样单独效果）
  nnUNetv2_train 3 3d_fullres 4 -tr Tr_DOS

  # 消融：DOS + UFL（不带 CopyPaste）
  nnUNetv2_train 3 3d_fullres 4 -tr Tr_DOS_UFL

  # 消融：DOS + 难度 CopyPaste（不带 UFL）
  nnUNetv2_train 3 3d_fullres 4 -tr Tr_DOS_DCP

  # 主实验：全组合
  nnUNetv2_train 3 3d_fullres 4 -tr Tr_DOS_DCP_UFL

#我们补充UFL的实验
● # fold0
  nnUNetv2_train 003 3d_fullres 0 -tr nnUNetTrainer_UFL -p nnUNetPlans

  # fold3
  nnUNetv2_train 003 3d_fullres 3 -tr nnUNetTrainer_UFL -p nnUNetPlans


  CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 3 3d_fullres 4 -tr Tr_SOS
  CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 3 3d_fullres 4 -tr Tr_SOS_UFL


CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 3 3d_fullres 4 -tr nnUNetTrainer_SizeOversampleV2_NTFP_Ext25

CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 3 3d_fullres 4 -tr nnUNetTrainer_SizeOversampleV2_Ext25



  CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 3 3d_fullres 4 -p nnUNetPlans -tr nnUNetTrainer_Ext25


    python 02_eval_ircadb.py


 nnUNetv2_train 3 3d_fullres 0 -tr nnUNetTrainer_SizeOversampleV3 -num_gpus 2