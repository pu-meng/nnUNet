du -sh /home/PuMengYu/nnUNet_workspace/* 2>/dev/null | sort -rh | head -20


  # Step 1: 迁移（预计 10~30 分钟）
rsync -ah /home/PuMengYu/nnUNet_workspace/preprocessed /media/8T/nnUNet_preprocessed

rm -rf /home/PuMengYu/nnUNet_workspace/preprocessed

  # Step 2: 建软链接
ln -s /media/8T/nnUNet_preprocessed /home/PuMengYu/nnUNet_workspace/preprocessed

  # Step 3: 验证
ls -la /home/PuMengYu/nnUNet_workspace/preprocessed
df -h /dev/nvme0n1p2
pip install nbstripout  #为了实现 Jupyter Notebook 的清理，去掉输出结果等元数据，减小文件体积
nbstripout --install

watch -n 5 du -sh /media/8T/nnUNet_preprocessed


# ============================================================
# notes 目录整理
# ============================================================

SRC=/home/PuMengYu/notes
DST=/home/PuMengYu/nnUNet/pumengyu/notes

# ── 第一步：删除过时文件 ──────────────────────────────────────
rm "$SRC/sh/restore_clash.sh"      # 老 clash 二进制，已换 mihomo
rm "$SRC/sh/start-clash.sh"        # 同上
rm "$SRC/sh/update_clash.sh"       # 老 clash 订阅更新，已有 update_my_clash.sh
rm "$SRC/sh/rsync.sh"              # 针对旧项目 MSD_LiverTumorSeg，已废弃
rm -rf "$SRC/代理/"                 # GOST 文档，与当前 mihomo 无关
rm -rf "$SRC/papers/"              # 旧版论文草稿，已被 draft_v1.md 替代

# ── 第二步：合并有价值的内容到目标目录 ──────────────────────────
# sh/ 里剩余脚本（proxy_fix, fix_github_ssh, dl_subconv, IP）
rsync -av "$SRC/sh/" "$DST/sh/"

# 代理命令参考
mv "$SRC/代理.md" "$DST/md/"

# nnUNet 代码笔记 → md/nnUNet/
mkdir -p "$DST/md/nnUNet"
mv "$SRC/nnUNet/"* "$DST/md/nnUNet/"

# 其余独立文件
mv "$SRC/ideas.md"          "$DST/md/"
mv "$SRC/tech.md"           "$DST/md/"
mv "$SRC/驿站.md"            "$DST/md/"
mv "$SRC/MED_VQA_READ.md"   "$DST/md/"
mv "$SRC/全世界可用数据集.md"  "$DST/md/"
mv "$SRC/全世界可用数据集.pdf" "$DST/md/"
mv "$SRC/INDEX.md"          "$DST/"
mv "$SRC/huawei"            "$DST/"

# ── 第三步：确认源目录已清空，删除 ──────────────────────────────
echo "=== 源目录剩余内容（应为空）==="
find "$SRC" -not -type d
# 确认无误后取消注释：
# rm -rf "$SRC"
