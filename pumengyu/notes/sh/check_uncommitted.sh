#!/bin/bash
REPO="/home/PuMengYu/nnUNet"
LOG="/home/PuMengYu/nnUNet/notes_pu/commit_reminders.log"

cd "$REPO" || exit 1

# 有未暂存、已暂存未提交、或新增未跟踪文件时记录
if ! git diff --quiet || ! git diff --cached --quiet || [ -n "$(git ls-files --others --exclude-standard)" ]; then
    echo "=== $(date '+%Y-%m-%d %H:%M:%S') 未提交改动 ===" >> "$LOG"
    git status --short >> "$LOG"
    echo "" >> "$LOG"
fi
