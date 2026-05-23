#!/usr/bin/env python3
"""
离线 CopyPaste 预处理脚本
────────────────────────
把小肿瘤 ROI 粘贴到训练 case，生成增强后的 .b2nd 文件，
训练时用普通 nnUNetTrainer（+ SmallTumorOversampleMixin 可选），
完全不需要在线 CopyPaste，消除额外内存开销。

用法：
    cd /home/PuMengYu/nnUNet
    python pumengyu/tools/offline_copypaste.py --fold 4 --n_aug 3

输出：
    PREP_DIR/ 下新增 liver_XXX_cp0.b2nd / _seg.b2nd / .pkl
    splits_final.json 备份为 splits_final.json.bak，
    原文件更新为含增强 case 的新版本
"""

import argparse
import gc
import ctypes
import json
import pickle
import shutil
from pathlib import Path

import blosc2
import numpy as np
from scipy.ndimage import label as cc_label

# ─── 路径 ─────────────────────────────────────────────────────────────────────
PREP_BASE = Path("/home/PuMengYu/nnUNet_workspace/preprocessed/Dataset003_Liver")
PREP_DIR  = PREP_BASE / "nnUNetPlans_3d_fullres"
SPLITS_IN  = PREP_BASE / "splits_final.json"
SPLITS_OUT = PREP_BASE / "splits_final_cp.json"

# ─── 超参 ─────────────────────────────────────────────────────────────────────
TUMOR_CLS  = 2
LIVER_CLS  = 1
CP_MAX_LOCS = 5000   # 小肿瘤上限（class_locations 条目数），与 CopyPasteMixin 一致
CP_MARGIN   = 3      # ROI 提取时 bbox 扩边 voxel 数
MAX_TRIES   = 300    # 寻找合法粘贴位置最大尝试次数

_libc = ctypes.CDLL("libc.so.6")


def _trim():
    gc.collect()
    _libc.malloc_trim(0)


# ─── 1. 构建 ROI 库 ────────────────────────────────────────────────────────────
def build_library(tr_keys: list[str]) -> list[dict]:
    library = []
    for key in tr_keys:
        with open(PREP_DIR / f"{key}.pkl", "rb") as f:
            props = pickle.load(f)

        n_locs = len(props.get("class_locations", {}).get(TUMOR_CLS, []))
        if n_locs == 0:
            continue

        seg = blosc2.open(str(PREP_DIR / f"{key}_seg.b2nd"), mode="r")[:]  #type:ignore # (1,Z,Y,X)
        seg3d = seg[0]# (Z,Y,X)#type:ignore
        tmask = seg3d == TUMOR_CLS

        if not tmask.any():# type:ignore
            del seg, seg3d, tmask; _trim(); continue

        labeled, n_cc = cc_label(tmask)# type:ignore
        Z, Y, X = seg3d.shape# type:ignore
        m = CP_MARGIN

        small_ccs = [i for i in range(1, n_cc + 1)
                     if (labeled == i).sum() <= CP_MAX_LOCS]

        if not small_ccs:
            del seg, seg3d, tmask, labeled; _trim(); continue

        # 只开文件句柄，不做 [:]，让 blosc2 按需解压各 ROI 区域
        ct_file = blosc2.open(str(PREP_DIR / f"{key}.b2nd"), mode="r")

        for cc_id in small_ccs:
            cc_mask = labeled == cc_id
            coords  = np.where(cc_mask)
            z0, z1  = int(coords[0].min()), int(coords[0].max()) + 1
            y0, y1  = int(coords[1].min()), int(coords[1].max()) + 1
            x0, x1  = int(coords[2].min()), int(coords[2].max()) + 1
            z0m = max(0, z0 - m); z1m = min(Z, z1 + m)
            y0m = max(0, y0 - m); y1m = min(Y, y1 + m)
            x0m = max(0, x0 - m); x1m = min(X, x1 + m)
            library.append({
                "ct":    np.array(ct_file[:, z0m:z1m, y0m:y1m, x0m:x1m], dtype=np.float32),
                # 必须 .copy()：切片是视图，会锁住整卷 cc_mask(~82MB) 不释放
                "tmask": cc_mask[z0m:z1m, y0m:y1m, x0m:x1m].copy(),#这行器决定作用,决定了为什么copypaste的占据的内存不会爆炸
                "n_vox": int(cc_mask.sum()),
            })

        del ct_file, seg, seg3d, tmask, labeled; _trim()
        print(f"  {key}: +{len(small_ccs)} ROI(s), total={len(library)}")

    return library


# ─── 2. 原地粘贴：先复制文件，再只改 ROI 那一小块，不加载整卷 ───────────────────
def paste_into_files(
    new_ct_path:  Path,
    new_seg_path: Path,
    liver_coords: np.ndarray,   # (N,3) 肝脏体素 zyx，来自 pkl class_locations
    vol_shape:    tuple,        # (Z,Y,X)
    roi:          dict,
    rng:          np.random.Generator,
) -> bool:
    """
    new_ct_path / new_seg_path 已是原文件的磁盘副本，
    本函数用 mode='a' 打开，只读写 ROI 区域，峰值内存仅 ROI 大小。
    """
    dz, dy, dx = roi["tmask"].shape
    Z, Y, X = vol_shape
    max_z, max_y, max_x = Z - dz, Y - dy, X - dx
    if max_z < 0 or max_y < 0 or max_x < 0:
        return False

    # 随机选肝脏体素为中心，clamp 到合法范围
    idx = int(rng.integers(len(liver_coords)))
    cz, cy, cx = int(liver_coords[idx][0]), int(liver_coords[idx][1]), int(liver_coords[idx][2])
    pz = int(np.clip(cz - dz // 2, 0, max_z))
    py = int(np.clip(cy - dy // 2, 0, max_y))
    px = int(np.clip(cx - dx // 2, 0, max_x))

    mask = roi["tmask"]  # (dz,dy,dx) bool

    ct_disk  = blosc2.open(str(new_ct_path),  mode="a")
    seg_disk = blosc2.open(str(new_seg_path), mode="a")
    seg_dtype = seg_disk.dtype

    # 只读写 ROI 区块
    patch_ct = np.array(ct_disk[:, pz:pz+dz, py:py+dy, px:px+dx])
    patch_ct[:, mask] = roi["ct"][:, mask]
    ct_disk[:, pz:pz+dz, py:py+dy, px:px+dx] = patch_ct

    patch_seg = np.array(seg_disk[:, pz:pz+dz, py:py+dy, px:px+dx])
    patch_seg[0, mask] = TUMOR_CLS
    seg_disk[:, pz:pz+dz, py:py+dy, px:px+dx] = patch_seg.astype(seg_dtype)

    del ct_disk, seg_disk
    return True


# ─── 3. 主流程 ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold",  type=int, default=4,
                        help="目标 fold（默认 4）")
    parser.add_argument("--n_aug", type=int, default=3,
                        help="每个 case 生成几个增强版本（默认 3）")
    parser.add_argument("--seed",  type=int, default=12345)
    parser.add_argument("--skip_existing", action="store_true",
                        help="若增强文件已存在则跳过（断点续跑）")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    with open(SPLITS_IN) as f:
        splits = json.load(f)

    tr_keys  = splits[args.fold]["train"]
    val_keys = splits[args.fold]["val"]
    print(f"Fold {args.fold}: {len(tr_keys)} train, {len(val_keys)} val")

    # ── 构建库 ──
    print("\n=== 构建 ROI 库 ===")
    library = build_library(tr_keys)
    print(f"共 {len(library)} 个 ROI")
    if not library:
        print("库为空，退出")
        return

    # ── 生成增强 case ──
    print(f"\n=== 生成增强数据（每 case × {args.n_aug} 份）===")
    new_tr_keys = list(tr_keys)
    skipped = added = failed = 0

    for i, key in enumerate(tr_keys):
        print(f"[{i+1}/{len(tr_keys)}] {key}")

        ct_path  = PREP_DIR / f"{key}.b2nd"
        seg_path = PREP_DIR / f"{key}_seg.b2nd"
        pkl_path = PREP_DIR / f"{key}.pkl"

        # 从 pkl 拿肝脏坐标（class_locations[1] 的 zyx），不加载 seg/CT
        with open(pkl_path, "rb") as f:
            props = pickle.load(f)
        liver_locs = np.asarray(props.get("class_locations", {}).get(LIVER_CLS, []))
        if len(liver_locs) == 0:
            print(f"  无肝脏坐标，跳过")
            continue
        liver_coords = liver_locs[:, 1:]  # 去掉通道列，留 zyx

        # 从 blosc2 元数据拿 shape，不解压
        vol_shape = tuple(blosc2.open(str(ct_path), mode="r").shape[1:])  # (Z,Y,X)

        for aug_idx in range(args.n_aug):
            new_key      = f"{key}_cp{aug_idx}"
            new_ct_path  = PREP_DIR / f"{new_key}.b2nd"
            new_seg_path = PREP_DIR / f"{new_key}_seg.b2nd"
            new_pkl_path = PREP_DIR / f"{new_key}.pkl"

            if args.skip_existing and new_ct_path.exists():
                print(f"  cp{aug_idx}: 已存在，跳过")
                new_tr_keys.append(new_key)
                skipped += 1
                continue

            # 先复制文件（磁盘到磁盘，几乎不占内存）
            shutil.copy(ct_path,  new_ct_path)
            shutil.copy(seg_path, new_seg_path)

            roi    = library[int(rng.integers(len(library)))]
            pasted = paste_into_files(new_ct_path, new_seg_path,
                                      liver_coords, vol_shape, roi, rng)

            if not pasted:
                print(f"  cp{aug_idx}: 粘贴失败（ROI 比体积大），跳过")
                failed += 1
                new_ct_path.unlink(missing_ok=True)
                new_seg_path.unlink(missing_ok=True)
                continue

            shutil.copy(pkl_path, new_pkl_path)
            new_tr_keys.append(new_key)
            added += 1
            print(f"  cp{aug_idx}: 已保存 {new_key}  (ROI n_vox={roi['n_vox']})")

        _trim()

    # ── 写独立的 splits_final_cp.json，不动原始文件 ──
    splits[args.fold]["train"] = new_tr_keys
    with open(SPLITS_OUT, "w") as f:
        json.dump(splits, f, indent=2)

    print(f"\n=== 完成 ===")
    print(f"  新增 case：{added}，跳过：{skipped}，失败：{failed}")
    print(f"  训练集：{len(tr_keys)} → {len(new_tr_keys)}")
    print(f"  splits_final.json 不变（其他实验不受影响）")
    print(f"  cp splits 写入：{SPLITS_OUT}")
    print(f"\n训练命令：")
    print(f"  CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 3 3d_fullres {args.fold} -tr nnUNetTrainer_OfflineCopyPaste_v2")


if __name__ == "__main__":
    main()
