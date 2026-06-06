#!/bin/bash
# 修复 GitHub SSH 连接被拒绝/reset 的问题
# 原理：将 GitHub SSH 流量改走 443 端口（ssh.github.com），绕过对 22 端口的封锁

set -e

SSH_CONFIG="$HOME/.ssh/config"
SSH_DIR="$HOME/.ssh"

echo "========== Step 1: 检查当前网络状况 =========="
echo "--- 测试 GitHub SSH 22 端口 ---"
timeout 5 ssh -T -o StrictHostKeyChecking=no -o ConnectTimeout=5 git@github.com 2>&1 || true

echo ""
echo "--- 测试 GitHub SSH 443 端口 ---"
timeout 5 ssh -T -o StrictHostKeyChecking=no -o ConnectTimeout=5 -p 443 git@ssh.github.com 2>&1 || true

echo ""
echo "========== Step 2: 确保 ~/.ssh 目录存在且权限正确 =========="
mkdir -p "$SSH_DIR"
chmod 700 "$SSH_DIR"
echo "~/.ssh 权限: $(stat -c '%a' $SSH_DIR)  (应为 700)"

echo ""
echo "========== Step 3: 检查 SSH Key 是否存在 =========="
if ls "$SSH_DIR"/id_*.pub 2>/dev/null; then
    echo "已有 SSH 公钥："
    for pub in "$SSH_DIR"/id_*.pub; do
        echo "  $pub"
        echo "  $(cat $pub)"
    done
else
    echo "未找到 SSH 公钥，正在生成 ed25519 密钥..."
    ssh-keygen -t ed25519 -C "pumengyu318@gmail.com" -f "$SSH_DIR/id_ed25519" -N ""
    echo ""
    echo ">>> 请将以下公钥添加到 GitHub: Settings > SSH and GPG keys > New SSH key <<<"
    echo ""
    cat "$SSH_DIR/id_ed25519.pub"
    echo ""
fi

echo ""
echo "========== Step 4: 写入 ~/.ssh/config（改用 443 端口）=========="

# 备份旧 config
if [ -f "$SSH_CONFIG" ]; then
    cp "$SSH_CONFIG" "${SSH_CONFIG}.bak.$(date +%Y%m%d_%H%M%S)"
    echo "已备份旧 config"
fi

# 检查是否已有 github.com 配置
if grep -q "Host github.com" "$SSH_CONFIG" 2>/dev/null; then
    echo "~/.ssh/config 已有 github.com 配置，跳过写入（请手动确认内容）："
    grep -A5 "Host github.com" "$SSH_CONFIG"
else
    cat >> "$SSH_CONFIG" << 'EOF'

Host github.com
    Hostname ssh.github.com
    Port 443
    User git
    IdentityFile ~/.ssh/id_ed25519
    ServerAliveInterval 60
    ServerAliveCountMax 3
EOF
    chmod 600 "$SSH_CONFIG"
    echo "已写入配置，config 权限: $(stat -c '%a' $SSH_CONFIG)  (应为 600)"
fi

echo ""
echo "========== Step 5: 验证 GitHub 连接 =========="
echo "正在通过 443 端口测试连接 GitHub..."
ssh -T git@github.com 2>&1 || true

echo ""
echo "========== 完成 =========="
echo "如果上面显示 'Hi <你的用户名>! You have successfully authenticated'，则配置成功。"
echo "现在可以重新执行: git push"