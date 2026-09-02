  # 1. dataset_profile：从缓存出图+CSV（秒级，不重读CT）
  python -m pumengyu.analysis.dataset_profile \
    --dataset Dataset003_Liver --output_tag lits --from_cache


     # 2. validation 集：补 CSV + 重出图
  python -m pumengyu.analysis.run \
    --trainer nnUNetTrainer_Baseline \
    --dataset Dataset003_Liver --fold 0 --split validation


  # 3. test 集：重出图（字体已修复）
  python -m pumengyu.analysis.run \
    --trainer nnUNetTrainer_Baseline \
    --dataset Dataset003_Liver --fold 0 --split test

CUDA_VISIBLE_DEVICES=0,1 nnUNetv2_train Dataset003_Liver 3d_fullres 0 \
    -tr nnUNetTrainer_NoMirror -num_gpus 2