#!/usr/bin/env bash
set -uo pipefail

# 只读检查服务器重启、SSH、空闲电源策略、温度及关机触发记录。
# 不修改配置，不重启服务。

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPORT_FILE="${SCRIPT_DIR}/检查ssh断开_$(date '+%Y%m%d_%H%M%S').log"

section() {
    printf '\n========== %s ==========\n' "$1"
}

run_and_save() {
    exec > >(tee -a "${REPORT_FILE}") 2>&1

    section "检查时间"
    date --iso-8601=seconds
    printf '报告文件：%s\n' "${REPORT_FILE}"

    section "当前运行时间和最近重启"
    uptime
    who -b
    sudo journalctl --list-boots --no-pager
    last -x -F -n 60

    section "SSH 当前状态"
    sudo systemctl status ssh --no-pager -l
    sudo systemctl is-enabled ssh
    sudo ss -ltnp 'sport = :22'

    section "网卡和路由"
    ip -brief address
    ip route

    section "空闲关机、休眠和定时器"
    systemd-analyze cat-config systemd/logind.conf
    sudo systemctl status \
        sleep.target suspend.target hibernate.target hybrid-sleep.target \
        --no-pager
    sudo systemctl list-timers --all --no-pager

    section "今天的关机、重启、过热和硬件异常"
    sudo journalctl --since today --no-pager |
        grep -Ei \
        'shutdown|poweroff|power key|reboot|suspend|hibernate|thermal|overheat|critical temperature|watchdog|hardware error|mce|oom|out of memory' \
        || printf '没有匹配记录。\n'

    section "今天 SSH 和网卡异常"
    sudo journalctl --since today --no-pager |
        grep -Ei \
        'sshd|ssh\.service|link is down|link becomes ready|carrier|network.*disconnect|dhcp.*(fail|timeout|expire)|reset.*(usb|ethernet)|enx6c1ff704d75f' \
        || printf '没有匹配记录。\n'

    section "历史关机命令和 sudo 操作"
    sudo grep -Eih \
        'COMMAND=.*(shutdown|reboot|poweroff|halt|systemctl.*(poweroff|reboot|suspend|hibernate))' \
        /var/log/auth.log /var/log/auth.log.1 2>/dev/null \
        || printf '没有找到通过 sudo 执行的关机命令。\n'

    section "最近 8 次启动结束前的关键日志"
    for boot_index in -1 -2 -3 -4 -5 -6 -7 -8; do
        printf '\n----- boot %s -----\n' "${boot_index}"
        sudo journalctl -b "${boot_index}" -o short-iso --no-pager -n 200 |
            grep -Eiv 'mihomo.*\[(TCP|UDP)\]' |
            tail -n 80
    done

    section "本次启动的内核硬件异常"
    sudo journalctl -b 0 -k --no-pager |
        grep -Ei \
        'thermal|overheat|critical|watchdog|hardware error|mce|edac|oom|out of memory|nvrm|xid|pcie.*error|nvme.*error|reset|unclean|recover' \
        || printf '没有匹配记录。\n'

    section "当前温度和负载"
    uptime
    sensors
    nvidia-smi \
        --query-gpu=index,name,temperature.gpu,power.draw,power.limit \
        --format=csv,noheader
    ps -eo pid,user,comm,%cpu,%mem --sort=-%cpu | head -n 25

    section "自建 SSH 持续监控日志"
    if [[ -f /home/PuMengYu/SSH断开监控/监控.log ]]; then
        sudo tail -n 200 /home/PuMengYu/SSH断开监控/监控.log
    else
        printf '持续监控日志尚未安装或尚未生成。\n'
    fi

    section "检查结束"
    printf '完整报告已保存到：%s\n' "${REPORT_FILE}"
}

printf '这个脚本只读取日志和状态，不会修改系统配置。\n'
printf '接下来 sudo 会要求输入服务器密码。\n'

if ! sudo -v; then
    printf 'sudo 验证失败，检查未执行。\n' >&2
    exit 1
fi

run_and_save
