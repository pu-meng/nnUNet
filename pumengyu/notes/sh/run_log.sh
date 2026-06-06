#!/bin/bash
# 用法: bash run_log.sh env CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 3 3d_fullres 0 -tr Tr_Stage2_FPSup
# 训练输出完全正常，进程退出后自动写崩溃报告

LOG_DIR="/home/PuMengYu/nnUNet/pumengyu/notes/log"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REPORT_FILE="${LOG_DIR}/crash_${TIMESTAMP}.txt"
CMD_STR="$*"

echo "命令: $CMD_STR"     > "$REPORT_FILE"
echo "启动: $(date)"     >> "$REPORT_FILE"
echo "报告: $REPORT_FILE"

# 直接运行，不拦截任何输出
"$@"
EXIT_CODE=$?

# 退出后写报告
{
echo ""
echo "======== 退出报告 $(date) ========"
echo "退出码: $EXIT_CODE"

case $EXIT_CODE in
  0)   echo "状态: 正常完成" ;;
  1)   echo "状态: Python 报错退出 (exception)" ;;
  2)   echo "状态: 命令参数错误" ;;
  137) echo "状态: SIGKILL — 极可能是 OOM 被系统杀死" ;;
  139) echo "状态: 段错误 (segfault)" ;;
  *)   echo "状态: 异常退出 (exit=$EXIT_CODE)" ;;
esac

echo ""
echo "--- OOM 检查 (dmesg) ---"
dmesg 2>/dev/null | grep -i "oom\|killed process\|out of memory" | tail -10 || echo "(无记录或需要 sudo)"

} | tee -a "$REPORT_FILE"

echo "崩溃报告: $REPORT_FILE"
