#!/bin/bash
# 清理训练异常退出后遗留的 medseg Python/CUDA 进程。
#
# 安全设计：
# - 默认只预览，不杀进程；真正清理必须加 --kill。
# - 默认只处理当前用户、medseg 环境、python、并且出现在 nvidia-smi compute-apps
#   里的 GPU 进程。
# - 自动模式默认只清理 PPID=1 的孤儿进程；加 --all 才会包含 PPID 不是 1 的
#   当前用户 medseg GPU Python，例如这次的 4093418 -> PPID 1163644。
# - 手动传 PID 时也会校验 USER/CMD，避免误杀别人的进程或非 medseg 任务。
#
# 常用：
#   ./kill_orphans.sh                 # 预览 PPID=1 的 medseg GPU Python
#   ./kill_orphans.sh --all           # 预览所有当前用户 medseg GPU Python
#   ./kill_orphans.sh --kill          # 杀 PPID=1 的 medseg GPU Python
#   ./kill_orphans.sh --kill --all    # 杀所有当前用户 medseg GPU Python，谨慎使用
#   ./kill_orphans.sh --kill 4093418  # 杀指定 PID，仍会做安全校验

set -u

DO_KILL=0
INCLUDE_ALL=0
PIDS=""

usage() {
    sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'
}

for ARG in "$@"; do
    case "$ARG" in
        --kill)
            DO_KILL=1
            ;;
        --list|--dry-run)
            DO_KILL=0
            ;;
        --all)
            INCLUDE_ALL=1
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        ''|*[!0-9]*)
            echo "不认识的参数: $ARG"
            echo "用法: $0 [--kill] [--all] [PID ...]"
            exit 2
            ;;
        *)
            PIDS="$PIDS $ARG"
            ;;
    esac
done

current_user() {
    id -un
}

gpu_pids() {
    nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null \
        | awk -F, '/medseg.*python/ {gsub(/^[ \t]+|[ \t]+$/, "", $1); print $1}'
}

auto_pids() {
    local smi_pids
    smi_pids=$(gpu_pids)

    if [ -z "$smi_pids" ]; then
        return 0
    fi

    for pid in $smi_pids; do
        ps -o pid=,ppid=,user=,cmd= -p "$pid" 2>/dev/null \
            | awk -v me="$(current_user)" -v include_all="$INCLUDE_ALL" '
                $3 == me && /medseg/ && /python/ && (include_all == 1 || $2 == 1) {print $1}
            '
    done
}

validate_pid() {
    local pid="$1"
    local line user cmd

    if ! kill -0 "$pid" 2>/dev/null; then
        echo "跳过 $pid: 当前 shell 看不到、进程已退出，或没有权限"
        return 1
    fi

    line=$(ps -o user=,cmd= -p "$pid" 2>/dev/null)
    user=$(printf "%s\n" "$line" | awk '{print $1}')
    cmd=$(printf "%s\n" "$line" | cut -d' ' -f2-)

    if [ "$user" != "$(current_user)" ]; then
        echo "跳过 $pid: 进程用户是 $user，不是当前用户 $(current_user)"
        return 1
    fi

    case "$cmd" in
        *medseg*python*|*python*medseg*)
            return 0
            ;;
        *)
            echo "跳过 $pid: 命令不像 medseg python: $cmd"
            return 1
            ;;
    esac
}

if [ -z "$PIDS" ]; then
    PIDS=$(auto_pids | awk 'NF && !seen[$1]++ {print $1}')
else
    PIDS=$(printf "%s\n" $PIDS | awk 'NF && !seen[$1]++ {print $1}')
fi

if [ -z "$PIDS" ]; then
    if [ "$INCLUDE_ALL" -eq 1 ]; then
        echo "没有发现当前用户的 medseg GPU Python 目标进程"
    else
        echo "没有发现 PPID=1 的当前用户 medseg GPU Python 目标进程"
        echo "如果 nvidia-smi 里有类似 4093418 这种 PPID 不是 1 的进程，先用 --all 预览。"
    fi
    exit 0
fi

echo "候选目标："
VALID_PIDS=""
for PID in $PIDS; do
    if validate_pid "$PID"; then
        ps -o pid,ppid,stat,user,etime,cmd -p "$PID"
        VALID_PIDS="$VALID_PIDS $PID"
    fi
done

if [ -z "$VALID_PIDS" ]; then
    echo "没有通过安全校验的目标进程"
    exit 0
fi

if [ "$DO_KILL" -ne 1 ]; then
    echo
    echo "当前是预览模式，没有杀进程。确认无误后加 --kill 执行。"
    exit 0
fi

echo
echo "开始清理："
for PID in $VALID_PIDS; do
    echo "kill -9 $PID"
    kill -9 "$PID"
done
echo "完成"
