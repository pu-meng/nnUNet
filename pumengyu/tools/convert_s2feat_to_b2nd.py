"""
将 s2feat/ 目录下的 .npy 文件批量转换为 .b2nd (blosc2) 格式。
转换后保留 .npy 文件，确认训练正常后可手动删除。

.npy 和 .b2nd 区别：
  1. 压缩存储：.npy 是原始 float32，原封不动写磁盘（630 MB/case）；
     .b2nd 将数据分块压缩（类似 zip 但极快），概率图压缩比约 8x（~80 MB/case），
     126 GB → 约 15-30 GB。
  2. 懒加载（lazy/mmap）：.npy 用 np.load 会把整个文件一次性读入内存；
     .b2nd 用 blosc2.open 几乎不占内存，只有真正访问的区域才按需解压，
     操作系统负责换入换出，12 个 dataloader worker 并发加载时内存从 8.4 GB 降到几百 MB。

用法:
    python pumengyu/tools/convert_s2feat_to_b2nd.py
"""

import os
import numpy as np
import blosc2
from tqdm import tqdm

S2FEAT_DIR = (
    "/home/PuMengYu/nnUNet_workspace/preprocessed/"
    "Dataset003_Liver/nnUNetPlans_3d_fullres/s2feat"
)

def main():
    npy_files = sorted(f for f in os.listdir(S2FEAT_DIR) if f.endswith('.npy'))
    print(f"共找到 {len(npy_files)} 个 .npy 文件，开始转换...")

    skipped, converted = 0, 0
    for fname in tqdm(npy_files, unit='file'):
        npy_path = os.path.join(S2FEAT_DIR, fname)
        b2nd_path = npy_path.replace('.npy', '.b2nd')

        if os.path.exists(b2nd_path):
            skipped += 1
            continue

        arr = np.load(npy_path)                          # (2, Z, Y, X) float32
        blosc2.asarray(arr, urlpath=b2nd_path, mode='w')
        converted += 1

    print(f"\n完成: 新转换 {converted} 个，已跳过 {skipped} 个")
    print(f"确认训练正常后可删除 .npy: rm {S2FEAT_DIR}/*.npy")

if __name__ == '__main__':
    main()
