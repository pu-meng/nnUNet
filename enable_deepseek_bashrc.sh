#!/usr/bin/env bash
set -euo pipefail

bashrc="/home/PuMengYu/.bashrc"
backup="${bashrc}.bak.$(date +%Y%m%d%H%M%S)"
cp "$bashrc" "$backup"

python3 - "$bashrc" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()

old = '''# # ── DeepSeek（API key 请改成新的，旧的已泄露务必作废）─
# # 自动检测直连/代理，优先直连 api.deepseek.com
# deepseek() {
#     local deepseek_api="https://api.deepseek.com"
#     local proxy_url="http://127.0.0.1:${PROXY_PORT:-7892}"

#     # 测试直连
#     if curl --noproxy '*' -s --max-time 5 \\
#          -o /dev/null -w "%{http_code}" "$deepseek_api" | grep -q '^[24]'; then
#         echo "🔗 DeepSeek: 直连可用" >&2
#         env -u ANTHROPIC_AUTH_TOKEN \\
#             ANTHROPIC_BASE_URL="https://api.deepseek.com/anthropic" \\
#             ANTHROPIC_API_KEY="sk-90e7f5f464a3429db6d3ff064b9d8bdb" \\
#             /home/PuMengYu/.local/bin/claude "$@"
#         return $?
#     fi

#     # 直连失败，试代理
#     echo "⚠️ DeepSeek: 直连失败，尝试代理..." >&2
#     if curl -s --max-time 5 --proxy "$proxy_url" \\
#          -o /dev/null -w "%{http_code}" "$deepseek_api" | grep -q '^[24]'; then
#         echo "🔗 DeepSeek: 代理可用" >&2
#         # 临时启用代理环境变量
#         local old_http="$http_proxy" old_https="$https_proxy"
#         export http_proxy="$proxy_url"
#         export https_proxy="$proxy_url"
#         env -u ANTHROPIC_AUTH_TOKEN \\
#             ANTHROPIC_BASE_URL="https://api.deepseek.com/anthropic" \\
#             ANTHROPIC_API_KEY="sk-90e7f5f464a3429db6d3ff064b9d8bdb" \\
#             /home/PuMengYu/.local/bin/claude "$@"
#         local ret=$?
#         # 恢复原代理状态
#         [ -n "$old_http" ] && export http_proxy="$old_http" || unset http_proxy
#         [ -n "$old_https" ] && export https_proxy="$old_https" || unset https_proxy
#         return $ret
#     fi

#     # 都不通
#     echo "❌ DeepSeek: 直连和代理均不可用！请检查网络或 mihomo 状态" >&2
#     echo "   代理状态: $(pgrep -x mihomo >/dev/null && echo 'mihomo 运行中' || echo 'mihomo 未运行')" >&2
#     return 1
# }
'''

new = '''# ── DeepSeek（使用 Anthropic 兼容接口）────────────────
# 先在当前 shell 设置：export DEEPSEEK_API_KEY="你的新 key"
deepseek() {
    if [ -z "${DEEPSEEK_API_KEY:-}" ]; then
        echo "DeepSeek: 请先设置 DEEPSEEK_API_KEY" >&2
        return 1
    fi

    env -u ANTHROPIC_AUTH_TOKEN \\
        ANTHROPIC_BASE_URL="https://api.deepseek.com/anthropic" \\
        ANTHROPIC_API_KEY="$DEEPSEEK_API_KEY" \\
        HTTPS_PROXY="${https_proxy:-}" \\
        /home/PuMengYu/.local/bin/claude "$@"
}
'''

if old not in text:
    raise SystemExit("没有找到预期的旧 DeepSeek 注释块，未修改 .bashrc")

path.write_text(text.replace(old, new))
PY

echo "已修改 $bashrc"
echo "备份在 $backup"
