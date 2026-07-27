"""
批量对 IRCADb 外部验证集运行 predict + report + viz，用于对比多个方法。

用法：
    # 运行全部（跳过已完成的）
    python pumengyu/ext_val/04_batch_ext_val.py

    # 只运行指定方法
    python pumengyu/ext_val/04_batch_ext_val.py --only MoE_SizeOV4 SizeOV2

    # 强制重新生成所有（即使已完成）
    python pumengyu/ext_val/04_batch_ext_val.py --force

    # 跳过 viz（更快）
    python pumengyu/ext_val/04_batch_ext_val.py --no_vis

    # 指定 GPU
    python pumengyu/ext_val/04_batch_ext_val.py --gpu 0
"""

from __future__ import annotations
import argparse
import re
import subprocess
import sys
from pathlib import Path

EXT_RESULT_ROOT = Path("/home/PuMengYu/nnUNet_workspace/results_v2/IRCADb/source_only")

# (method_name, trainer_name, internal_overall)  按内部 Overall 从高到低
METHODS = [
    ("MoE_SizeOV4",  "nnUNetTrainer_MLAUNet_MoE_SizeOversampleV4",  0.8330),
    ("SizeOV2",      "nnUNetTrainer_SizeOversampleV2",               0.8187),
    ("MoE_SizeOV5",  "nnUNetTrainer_MLAUNet_MoE_SizeOversampleV5",  0.8167),
    ("MoE",          "nnUNetTrainer_MLAUNet_MoE",                    0.8166),
    ("MoE_SizeOV2",  "nnUNetTrainer_MLAUNet_MoE_SizeOversampleV2",  0.8152),
    ("MLAUNet",      "nnUNetTrainer_MLAUNet",                        0.8148),
    ("SizeOV3",      "nnUNetTrainer_SizeOversampleV3",               0.8143),
    ("NoMirror",     "nnUNetTrainer_NoMirror",                       0.8133),
    ("Baseline",     "nnUNetTrainer_Baseline",                       0.7941),
]

SCRIPT = Path(__file__).parent / "03_gen_method_report.py"


def _has_predictions(method_dir: Path) -> bool:
    pred_dir = method_dir / "predictions"
    return pred_dir.exists() and any(pred_dir.glob("*.nii.gz"))


def _has_report(method_dir: Path) -> bool:
    return (method_dir / "report_custom.txt").exists()


def _read_ext_overall(method_dir: Path) -> str:
    report = method_dir / "report_custom.txt"
    if not report.exists():
        return "N/A"
    for line in report.read_text().splitlines():
        m = re.search(r"Overall\s*:.*?=\s*([0-9.]+)", line)
        if m:
            return m.group(1)
    return "N/A"


def run_method(method: str, trainer: str, gpu: int, no_vis: bool, force: bool) -> str:
    """返回 'done' / 'skip' / 'fail'"""
    method_dir = EXT_RESULT_ROOT / method
    has_pred = _has_predictions(method_dir)
    has_report = _has_report(method_dir)

    if not force and has_pred and has_report:
        return "skip"

    cmd = [sys.executable, str(SCRIPT), "--method", method, "--gpu", str(gpu)]

    if not has_pred or force:
        cmd += ["--predict", "--trainer", trainer]

    if no_vis:
        cmd.append("--no_vis")

    print(f"\n{'=' * 60}")
    print(f"[运行] {method}  trainer={trainer}")
    print(f"  cmd: {' '.join(cmd)}")
    print(f"{'=' * 60}")

    ret = subprocess.run(cmd, cwd=SCRIPT.parent.parent.parent)
    return "done" if ret.returncode == 0 else "fail"


def print_summary(results: list[tuple[str, float, str]]) -> None:
    print(f"\n{'=' * 70}")
    print(f"{'方法':<20}  {'内部Overall':>12}  {'外部Overall':>12}  {'Δ':>8}  状态")
    print(f"{'-' * 70}")
    for method, internal, status in results:
        method_dir = EXT_RESULT_ROOT / method
        ext = _read_ext_overall(method_dir)
        delta = ""
        if ext != "N/A":
            delta = f"{float(ext) - internal:+.4f}"
        print(f"{method:<20}  {internal:>12.4f}  {ext:>12}  {delta:>8}  [{status}]")
    print(f"{'=' * 70}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="+", metavar="METHOD",
                        help="只运行指定方法，默认全部")
    parser.add_argument("--force", action="store_true",
                        help="强制重新运行（即使已完成）")
    parser.add_argument("--no_vis", action="store_true",
                        help="跳过 viz 生成")
    parser.add_argument("--gpu", type=int, default=1,
                        help="CUDA 设备编号（默认 1）")
    args = parser.parse_args()

    method_set = set(args.only) if args.only else None
    results = []

    for method, trainer, internal in METHODS:
        if method_set and method not in method_set:
            continue

        status = run_method(method, trainer, args.gpu, args.no_vis, args.force)
        if status == "skip":
            print(f"[跳过] {method}  (已完成，使用 --force 重新运行)")
        results.append((method, internal, status))

    print_summary(results)


if __name__ == "__main__":
    main()
