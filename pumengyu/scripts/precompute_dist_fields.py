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
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import load_json, join

from nnunetv2.paths import nnUNet_preprocessed
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

from pumengyu.tools.dist_field import compute_surface_distance_field


def _process_case(args):
    npz_path, out_path, num_classes, spacing = args
    data = np.load(npz_path)
    # nnUNet v2: 'data' key contains image+seg concatenated; seg is last channel
    arr = data['data']
    seg = arr[-1].astype(np.int32)
    dist = compute_surface_distance_field(seg, num_classes=num_classes, spacing=spacing)
    np.save(out_path, dist)
    return os.path.basename(npz_path)


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
    num_classes = len(dataset_json['labels'])  # includes background

    data_dir = preprocessed_dir / f'nnUNetPlans_{args.config}'
    npz_files = sorted(data_dir.glob('*.npz'))

    tasks = []
    for npz in npz_files:
        case_id = npz.stem
        out_path = data_dir / f'{case_id}_dist.npy'
        if not args.overwrite and out_path.exists():
            continue
        tasks.append((str(npz), str(out_path), num_classes, spacing))

    print(f'Dataset : {dataset_name}')
    print(f'Config  : {args.config}  spacing={spacing}')
    print(f'Classes : {num_classes}')
    print(f'Cases   : {len(npz_files)} total, {len(tasks)} to compute')

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
