#!/usr/bin/env python3
"""
分析流水线一键启动
──────────────────────────────────────────────────────────────
按顺序跑三步：
  Step 1  feature_extraction/extract_intrinsic_features.py
          → notes/实验结果分析/intrinsic_features.json
  Step 2  data_analysis/compute_difficulty.py [--check]
          → notes/实验结果分析/difficulty.json
  Step 3  data_analysis/feature_representativeness.py
          → notes/实验结果分析/feature_representativeness.txt

用法:
    python pumengyu/tools/run_analysis.py              # 全跑
    python pumengyu/tools/run_analysis.py --check      # Step2 额外做难度vs Dice体检
    python pumengyu/tools/run_analysis.py --skip 1     # 跳过 Step1（已有 intrinsic_features.json 时）
    python pumengyu/tools/run_analysis.py --only 3     # 只跑 Step3
    python pumengyu/tools/run_analysis.py --limit 5    # Step1 只处理5个case（调试用）
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent   # nnUNet/

STEPS = [
    (1, 'feature_extraction/extract_intrinsic_features.py', '提取内在特征 → intrinsic_features.json'),
    (2, 'data_analysis/compute_difficulty.py',              '计算难度权重 → difficulty.json'),
    (3, 'data_analysis/feature_representativeness.py',      '特征代表性排名 → feature_representativeness.txt'),
]


def run(script_rel, extra_args=()):
    script = ROOT / 'pumengyu' / 'tools' / script_rel
    cmd = [sys.executable, str(script)] + list(extra_args)
    print(f'\n{"="*60}')
    print(f'运行：{" ".join(cmd)}')
    print('='*60)
    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode != 0:
        print(f'\n[错误] 退出码 {result.returncode}，流水线中止。')
        sys.exit(result.returncode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--skip',  type=int, default=0,  help='跳过第 N 步及之前（如 --skip 1 跳过Step1）')
    ap.add_argument('--only',  type=int, default=0,  help='只跑第 N 步')
    ap.add_argument('--check', action='store_true',   help='Step2 额外做难度 vs Dice 体检')
    ap.add_argument('--limit', type=int, default=0,   help='Step1 只处理前 N 个 case（调试）')
    args = ap.parse_args()

    for step_id, script, desc in STEPS:
        if args.only and step_id != args.only:
            continue
        if not args.only and step_id <= args.skip:
            print(f'[跳过] Step{step_id}: {desc}')
            continue

        print(f'\n>>> Step{step_id}: {desc}')
        extra = []
        if step_id == 1 and args.limit:
            extra += ['--limit', str(args.limit)]
        if step_id == 2 and args.check:
            extra += ['--check']
        run(script, extra)

    print('\n\n全部完成。')
    print('  difficulty.json            → mixins.py CopyPaste 难度加权')
    print('  feature_representativeness.txt → 看哪个指标最能预测失败')


if __name__ == '__main__':
    main()
