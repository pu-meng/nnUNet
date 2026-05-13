"""
验证预计算距离场文件完整性。
检查每个 _dist.npz 是否可加载、包含必要字段、形状合理。
"""

import sys
from pathlib import Path
import numpy as np

BASE_DIR = Path('/home/PuMengYu/nnUNet_workspace/preprocessed/Dataset003_Liver/nnUNetPlans_3d_fullres_dist')
DATA_DIR = BASE_DIR
DIST_DIR = BASE_DIR.parent / (BASE_DIR.name + '_dist')

dist_files = sorted(DIST_DIR.glob('*_dist.npz')) if DIST_DIR.exists() else sorted(DATA_DIR.glob('*_dist.npz'))
pkl_files  = sorted(DATA_DIR.glob('*.pkl'))

case_ids = [p.stem for p in pkl_files]
dist_map  = {p.stem.replace('_dist', ''): p for p in dist_files}

missing   = [c for c in case_ids if c not in dist_map]
corrupt   = []
ok        = []

print(f'Total cases : {len(case_ids)}')
print(f'Dist files  : {len(dist_files)}')
print(f'Missing     : {len(missing)}')
if missing:
    for m in missing:
        print(f'  MISSING  {m}')

print('\nValidating each file...')
for case_id in case_ids:
    if case_id not in dist_map:
        continue
    path = dist_map[case_id]
    try:
        d = np.load(path)
        assert 'data'  in d, 'missing key: data'
        assert 'bbox'  in d, 'missing key: bbox'
        assert 'shape' in d, 'missing key: shape'
        assert d['data'].ndim == 4,  f'data ndim={d["data"].ndim}, expected 4'
        assert d['bbox'].shape == (6,), f'bbox shape={d["bbox"].shape}'
        assert d['shape'].shape == (3,), f'shape shape={d["shape"].shape}'
        ok.append(case_id)
    except Exception as e:
        corrupt.append((case_id, str(e)))
        print(f'  CORRUPT  {case_id}: {e}')

print(f'\nResults: {len(ok)} OK, {len(missing)} missing, {len(corrupt)} corrupt')
if missing or corrupt:
    print('\nRe-run with --overwrite for affected cases, or add case IDs to a list.')
    sys.exit(1)
else:
    print('All distance fields look good.')
