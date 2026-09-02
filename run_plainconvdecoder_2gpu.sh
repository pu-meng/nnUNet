#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1
export nnUNet_raw=/home/PuMengYu/nnUNet_workspace/raw
export nnUNet_preprocessed=/home/PuMengYu/nnUNet_workspace/preprocessed
export nnUNet_results=/home/PuMengYu/nnUNet_workspace/results_v2
export nnUNet_extTrainer=/home/PuMengYu/nnUNet/pumengyu

exec /home/PuMengYu/anaconda3/envs/medseg/bin/nnUNetv2_train \
  3 3d_fullres 0 \
  -tr nnUNetTrainer_MedNeXt_MLA_MoE_PlainConvDecoder \
  -num_gpus 2
