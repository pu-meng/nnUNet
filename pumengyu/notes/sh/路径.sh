from pathlib import Path
import shutil

p = Path("/data/huawei/videos/test.mp4")

# ── 路径信息 ──────────────────────────
p.name          # test.mp4
p.stem          # test
p.suffix        # .mp4
p.parent        # /data/huawei/videos
p.exists()      # 是否存在
p.is_file()     # 是否是文件
p.is_dir()      # 是否是目录

# ── 路径拼接 ──────────────────────────
p.parent / "new.mp4"          # /data/huawei/videos/new.mp4 #type:ignore
Path("/data") / "a" / "b.mp4" # /data/a/b.mp4 #type:ignore

# ── 目录操作 ──────────────────────────
p.parent.mkdir(parents=True, exist_ok=True)  # 创建目录
p.relative_to(Path("/data/huawei"))          # videos/test.mp4

# ── 搜索文件 ──────────────────────────
Path("/data").rglob("*.mp4")   # 递归找所有mp4，返回绝对路径
Path("/data").glob("*.mp4")    # 只找当前层

# ── 文件操作（shutil）─────────────────
shutil.copy2(src, dst)         # 复制文件
shutil.move(src, dst)          # 移动文件
shutil.rmtree(p)               # 删除整个目录
shutil.copytree(src, dst)      # 复制整个目录