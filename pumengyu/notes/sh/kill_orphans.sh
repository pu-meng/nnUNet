#!/bin/bash
# 清理训练被 OOM Kill 后遗留的孤儿 DataLoader worker 进程
# 孤儿进程特征：PPID=1（被 init 领养）+ medseg 环境的 python
# 副作用：释放这些进程持有的 GPU CUDA context（nvtop 里显示 N/A 的那些）

PIDS=$(ps -eo pid,ppid,cmd | awk '$2==1 && /medseg.*python/ {print $1}')

if [ -z "$PIDS" ]; then
    echo "没有发现孤儿进程"
    exit 0
fi

COUNT=$(echo "$PIDS" | wc -w)
echo "发现 $COUNT 个孤儿进程，正在清理..."
echo "$PIDS" | xargs kill -9
echo "完成"
