#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1

exec /home/PuMengYu/anaconda3/envs/medseg/bin/nnUNetv2_train \
  3 3d_fullres 0 \
  -tr nnUNetTrainer_MedNeXt_MLA_MoE_FixedPatch96 \
  -num_gpus 2
