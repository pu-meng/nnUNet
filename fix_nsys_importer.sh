#!/bin/bash
sudo ln -s /usr/lib/nsight-systems/host-linux-x64 /usr/lib/x86_64-linux-gnu/nsight-systems/host-linux-x64
ls /usr/lib/x86_64-linux-gnu/nsight-systems/


CUDA_VISIBLE_DEVICES=1 nsys profile \
    --output ~/nnUNet_workspace/profiling/ib7_v2 \
    --trace cuda,nvtx \
    nnUNetv2_train Dataset003_Liver 3d_fullres 0 \
    -tr nnUNetTrainer_MLAUNet_MoE_IB7_SizeOversampleV4



  CUDA_VISIBLE_DEVICES=1 nsys profile \
    --output ~/nnUNet_workspace/profiling/ib7 \
    --trace cuda,nvtx \
    --delay 120 \
    --duration 30 \
    nnUNetv2_train Dataset003_Liver 3d_fullres 0 \
    -tr nnUNetTrainer_MLAUNet_MoE_IB7_SizeOversampleV4