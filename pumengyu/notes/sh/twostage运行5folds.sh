#!/usr/bin/env bash
# 训练 TwoStage 剩余三折（fold_1, fold_2, fold_3），全部使用 GPU 0
# 已完成：fold_0, fold_4
# Dataset: 4 (Dataset004_LiverTumor)，每折约 10 小时，总计约 30 小时

PYTHON=/home/PuMengYu/anaconda3/envs/medseg/bin/nnUNetv2_train
LOGDIR=/home/PuMengYu/nnUNet_workspace/logs
DATASET=4
CONFIG=3d_fullres
TRAINER=nnUNetTrainer_TwoStage

mkdir -p "$LOGDIR"
LOG="$LOGDIR/twostage_gpu0_fold123.log"

nohup bash -c '
    ts() { date "+[%Y-%m-%d %H:%M:%S]"; }

    for fold in 1 2 3; do
        echo ""
        echo "$(ts) ========== 开始 fold_${fold} =========="
        t0=$SECONDS

        CUDA_VISIBLE_DEVICES=0 '"$PYTHON"' '"$DATASET"' '"$CONFIG"' $fold -tr '"$TRAINER"'
        code=$?

        elapsed=$(( SECONDS - t0 ))
        h=$(( elapsed/3600 )); m=$(( elapsed%3600/60 )); s=$(( elapsed%60 ))

        if [ $code -ne 0 ]; then
            echo "$(ts) !!! fold_${fold} 失败 (exit code=$code)，训练中止 !!!"
            exit $code
        fi
        echo "$(ts) ========== fold_${fold} 完成，用时 ${h}h${m}m${s}s =========="
    done

    echo ""
    echo "$(ts) ✓ 全部三折训练完成"
' > "$LOG" 2>&1 &

PID=$!
echo "$PID" > "$LOGDIR/twostage_gpu0_fold123.pid"
echo "PID: $PID（已写入 $LOGDIR/twostage_gpu0_fold123.pid）"
echo ""
echo "查看日志：tail -f $LOG"
echo "查看状态：bash pumengyu/notes/sh/check_twostage.sh"
