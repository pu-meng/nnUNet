#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="/home/PuMengYu/SSH断开监控"
LOG_FILE="${LOG_DIR}/监控.log"

printf '即将安装只读 SSH/网络/温度监控服务。\n'
printf '只需安装一次，以后每次重启都会自动运行。\n'
printf '监控日志：%s\n' "${LOG_FILE}"
printf '需要输入 sudo 密码。\n\n'

sudo -v
sudo install -d -o PuMengYu -g PuMengYu -m 0755 "${LOG_DIR}"
sudo install -o root -g root -m 0755 \
    "${SCRIPT_DIR}/ssh-disconnect-monitor" \
    /usr/local/sbin/ssh-disconnect-monitor
sudo install -o root -g root -m 0644 \
    "${SCRIPT_DIR}/ssh-disconnect-monitor.service" \
    /etc/systemd/system/ssh-disconnect-monitor.service
sudo install -o root -g root -m 0644 \
    "${SCRIPT_DIR}/ssh-disconnect-monitor.logrotate" \
    /etc/logrotate.d/ssh-disconnect-monitor

sudo systemctl daemon-reload
sudo systemctl enable --now ssh-disconnect-monitor.service

printf '\n安装完成。\n'
sudo systemctl status ssh-disconnect-monitor.service --no-pager -l
printf '\n最新日志：\n'
sudo tail -n 8 "${LOG_FILE}"
