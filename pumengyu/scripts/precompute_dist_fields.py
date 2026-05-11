"""
离线预计算 BATseg 距离场。

对 nnUNetPlans_3d_fullres 里每个 case 的全图分割做 EDT，
结果保存为 <preprocessed_dir>/<case_id>_dist.npy，形状 (K, H, W, D)。

用法：
  python pumengyu/scripts/precompute_dist_fields.py \
      --dataset 3 \
      --config 3d_fullres \
      [--num_workers 8] \
      [--overwrite]
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import load_json

from nnunetv2.paths import nnUNet_preprocessed
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

from pumengyu.tools.dist_field import compute_surface_distance_field


def _load_seg(data_dir: str, case_id: str) -> np.ndarray:
    """Load segmentation for one case, supporting .b2nd and .npz formats."""
    import os
    seg_b2nd = os.path.join(data_dir, f'{case_id}_seg.b2nd')
    seg_npz  = os.path.join(data_dir, f'{case_id}.npz')
    seg_npy  = os.path.join(data_dir, f'{case_id}.npy')

    if os.path.isfile(seg_b2nd):
        import blosc2
        arr = blosc2.open(urlpath=seg_b2nd, mode='r')[:]
        return arr[0].astype(np.int32)   # (H, W, D)

    if os.path.isfile(seg_npz):
        arr = np.load(seg_npz)['data']
        return arr[-1].astype(np.int32)  # last channel is seg

    if os.path.isfile(seg_npy):
        arr = np.load(seg_npy, mmap_mode='r')
        return arr[-1].astype(np.int32)

    raise FileNotFoundError(f'No seg file found for case {case_id} in {data_dir}')


def _process_case(args):
    data_dir, case_id, out_path, num_classes, spacing = args
    seg = _load_seg(data_dir, case_id)
    dist = compute_surface_distance_field(seg, num_classes=num_classes, spacing=spacing)
    np.save(out_path, dist)
    return case_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True,
                        help='Dataset ID or name, e.g. 3 or Dataset003_Liver')
    parser.add_argument('--config', default='3d_fullres')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--overwrite', action='store_true',
                        help='Recompute even if _dist.npy already exists')
    args = parser.parse_args()

    dataset_name = maybe_convert_to_dataset_name(args.dataset)
    preprocessed_dir = Path(nnUNet_preprocessed) / dataset_name

    plans = PlansManager(str(preprocessed_dir / 'nnUNetPlans.json'))
    config = plans.get_configuration(args.config)
    spacing = tuple(config.spacing)

    dataset_json = load_json(str(preprocessed_dir / 'dataset.json'))
    num_classes = len(dataset_json['labels'])

    data_dir = preprocessed_dir / f'nnUNetPlans_{args.config}'

    # 用 .pkl 文件枚举所有 case（每种格式都有 .pkl）
    case_ids = sorted(p.stem for p in data_dir.glob('*.pkl'))

    tasks = []
    for case_id in case_ids:
        out_path = data_dir / f'{case_id}_dist.npy'
        if not args.overwrite and out_path.exists():
            continue
        tasks.append((str(data_dir), case_id, str(out_path), num_classes, spacing))

    print(f'Dataset : {dataset_name}')
    print(f'Config  : {args.config}  spacing={spacing}')
    print(f'Classes : {num_classes}')
    print(f'Cases   : {len(case_ids)} total, {len(tasks)} to compute')

    if not tasks:
        print('All distance fields already exist. Use --overwrite to recompute.')
        return

    with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
        futures = {pool.submit(_process_case, t): t for t in tasks}
        for i, f in enumerate(as_completed(futures), 1):
            name = f.result()
            print(f'[{i}/{len(tasks)}] {name}')

    print('Done.')


if __name__ == '__main__':
    main()
