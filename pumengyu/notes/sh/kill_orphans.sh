#!/bin/bash
# 清理训练被 OOM Kill 后遗留的孤儿 DataLoader worker 进程
# 孤儿进程特征：PPID=1（被 init 领养）+ medseg 环境的 python
# 副作用：释放这些进程持有的 GPU CUDA context（nvtop 里显示 N/A 的那些）

PIDS=$(ps -eo pid,ppid,cmd | awk '$2==1 && /medseg.*python/ {print $1}')
#ps是"process status"的缩写,列出当前所有进程
#-e=所有进程,不只是当前终端的,-o pid ppid,cmd是自定义输出列
#pid是进程自己的id,ppid是父进程id,cmd是进程的命令行
#Linux规则,父进程死亡,子进程被init(pid=1)领养,所以孤儿进程的ppid=1
#awk是文本处理工具,这里用来筛选出符合条件的进程id
#awk是逐行处理ps输出,$1 $2 $3分别是每行的第一列,第二列,第三列
#$2==1表示第二列(即ppid)等于1,/medseg.*python/表示第三列(即cmd)包含medseg和python的字符串
#两个条件都满足{print $1}就打印第一列(即pid),也就是孤儿进程的pid
#wc=word count,-w=统计单词数
#
if [ -z "$PIDS" ]; then
    echo "没有发现孤儿进程"
    exit 0
fi

COUNT=$(echo "$PIDS" | wc -w)
echo "发现 $COUNT 个孤儿进程，正在清理..."
echo "$PIDS" | xargs kill -9
echo "完成"
#xargs是把标准输入的内容转成后续命令的参数
#echo  "234 235" | xargs kill -9等价于kill -9 234 235
#-9是信号编号,SIGKILL,强制杀死进程,无法被捕获和忽略
#