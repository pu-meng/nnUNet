#!/bin/bash
# 3D-IRCADb 外部验证推理脚本
# 用法：bash 01_run_inference.sh [GPU_ID]
#
# 对每个方法跑全部 5 折推理，然后做 ensemble 平均
# 输出目录：nnUNet_workspace/external_val/ircadb_full/predictions/<trainer>/
#   - fold_0/ fold_1/ ... fold_4/   各折独立预测
#   - ensemble/                      5折 softmax 平均后的最终预测

GPU=${1:-0}
INPUT="/home/PuMengYu/nnUNet_workspace/external_val/ircadb_full/images"
PRED_ROOT="/home/PuMengYu/nnUNet_workspace/external_val/ircadb_full/predictions"
RESULTS="/home/PuMengYu/nnUNet_workspace/results/Dataset003_Liver"

if [ ! -d "$INPUT" ] || [ -z "$(ls -A $INPUT 2>/dev/null)" ]; then
    echo "[ERROR] 输入目录为空: $INPUT"
    echo "请先运行: python prepare_ircadb.py"
    exit 1
fi

# ------------------------------------------------------------------ #
# 待测方法（trainer名）：跑所有已训练的折
# ------------------------------------------------------------------ #
TRAINERS=(
    "nnUNetTrainer"
    "nnUNetTrainer_UFL"
    "nnUNetTrainer_UFL_delta06"
    "nnUNetTrainer_SizeOversample"
    "nnUNetTrainer_SizeOversampleV2"
)

for TRAINER in "${TRAINERS[@]}"; do
    echo ""
    echo "=========================================="
    echo "方法: $TRAINER"
    echo "=========================================="

    FOLD_DIRS=()

    for FOLD in 0 1 2 3 4; do
        CKPT="${RESULTS}/${TRAINER}__nnUNetPlans__3d_fullres/fold_${FOLD}/checkpoint_best.pth"
        [ -f "$CKPT" ] || continue

        OUT_FOLD="${PRED_ROOT}/${TRAINER}/fold_${FOLD}"

        if [ -d "$OUT_FOLD" ] && [ "$(ls -A $OUT_FOLD/*.nii.gz 2>/dev/null)" ]; then
            echo "  [SKIP] fold_$FOLD 已有预测"
        else
            mkdir -p "$OUT_FOLD"
            echo "  [RUN] fold_$FOLD ..."
            CUDA_VISIBLE_DEVICES=$GPU nnUNetv2_predict \
                -i "$INPUT" \
                -o "$OUT_FOLD" \
                -d 3 -c 3d_fullres -p nnUNetPlans \
                -tr "$TRAINER" -f "$FOLD" \
                -chk checkpoint_best.pth \
                --save_probabilities
            echo "  [DONE] fold_$FOLD"
        fi

        FOLD_DIRS+=("$OUT_FOLD")
    done

    # ensemble：把所有完成折的概率图平均
    ENS_DIR="${PRED_ROOT}/${TRAINER}/ensemble"
    if [ ${#FOLD_DIRS[@]} -ge 2 ]; then
        echo "  [ENSEMBLE] 合并 ${#FOLD_DIRS[@]} 折 -> $ENS_DIR"
        mkdir -p "$ENS_DIR"
        # nnUNet 内置 ensemble 工具
        nnUNetv2_ensemble -i "${FOLD_DIRS[@]}" -o "$ENS_DIR" -np 4
        echo "  [DONE] ensemble"
    elif [ ${#FOLD_DIRS[@]} -eq 1 ]; then
        echo "  [INFO] 只有1折，直接用 fold 结果作为最终预测"
        ENS_DIR="${FOLD_DIRS[0]}"
    else
        echo "  [WARN] $TRAINER 没有任何已训练的折，跳过"
    fi
done

echo ""
echo "所有推理完成，运行 python 02_eval_ircadb.py 查看结果"
