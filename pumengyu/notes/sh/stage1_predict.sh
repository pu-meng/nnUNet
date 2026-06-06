#!/bin/bash
# Stage1 全量推理：对 131 个训练 case 生成概率图，供 Stage2 使用
# 输出：stage1_softmax/{case}.npz（softmax 3通道，channel=2 为肿瘤概率）

RESULTS_FOLDER=/home/PuMengYu/nnUNet_workspace/results_v2 \
nnUNetv2_predict \
    -i /home/PuMengYu/nnUNet_workspace/raw/Dataset003_Liver/imagesTr \
    -o /home/PuMengYu/nnUNet_workspace/results_v2/Dataset003_Liver/Tr_Stage1_TumorOnly__nnUNetPlans__3d_fullres/fold_0/stage1_softmax \
    -d 3 \
    -c 3d_fullres \
    -tr Tr_Stage1_TumorOnly \
    -f 0 \
    -chk checkpoint_best.pth \
    --save_probabilities
