"""
Batch-run HCCReferencedCT fixed-test external validation for Dataset003 trainers.

By default this scans:
    /home/PuMengYu/nnUNet_workspace/results_v2/Dataset003_Liver/*/fold_0/checkpoint_best.pth

and evaluates each trainer on the 21 held-out HCC test cases recorded in:
    preprocessed/Dataset013_HCCReferencedCT/split_info_701020_from_fold0.json
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


WORKSPACE = Path("/home/PuMengYu/nnUNet_workspace")
DATASET003_RESULTS = WORKSPACE / "results_v2" / "Dataset003_Liver"
EXT_RESULT_ROOT = WORKSPACE / "results_v2" / "ExternalVal_HCCReferencedCT"
SCRIPT = Path(__file__).with_name("05_gen_hcc_test_report.py")


ALIASES = {
    "nnUNetTrainer_Baseline": "Baseline",
    "nnUNetTrainer_DeepDWIBMedConfig": "DeepDWIBMedConfig",
    "nnUNetTrainer_DeepDWIBResGN": "DeepDWIBResGN",
    "nnUNetTrainer_DeepPlainResGN": "DeepPlainResGN",
    "nnUNetTrainer_DeepPlainResGN_SizeOV4": "DeepPlainResGN_SizeOV4",
    "nnUNetTrainer_DeepResGN_MLA": "DeepResGN_MLA",
    "nnUNetTrainer_DWSepRes4_MoE_SizeOV4": "DWSepRes4_MoE_SizeOV4",
    "nnUNetTrainer_MLAUNet": "MLAUNet",
    "nnUNetTrainer_MLAUNet_1500": "MLAUNet_1500",
    "nnUNetTrainer_MLAUNet_MoE": "MoE",
    "nnUNetTrainer_MLAUNet_MoE_IB7_SizeOversampleV4": "MLAUNet_MoE_IB7_SizeOV4",
    "nnUNetTrainer_MLAUNet_MoE_SizeOversampleV2": "MoE_SizeOV2",
    "nnUNetTrainer_MLAUNet_MoE_SizeOversampleV4": "MoE_SizeOV4",
    "nnUNetTrainer_MLAUNet_MoE_SizeOversampleV5": "MoE_SizeOV5",
    "nnUNetTrainer_MLA_GK5_V4": "MLA_GK5_V4",
    "nnUNetTrainer_MedNeXt": "MedNeXt",
    "nnUNetTrainer_MedNeXt_MLA": "MedNeXt_MLA",
    "nnUNetTrainer_MedNeXt_MLA_FPSafe": "MedNeXt_MLA_FPSafe",
    "nnUNetTrainer_MedNeXt_MLA_SizeOV4": "MedNeXt_MLA_SizeOV4",
    "nnUNetTrainer_MedNeXt_SizeOV4": "MedNeXt_SizeOV4",
    "nnUNetTrainer_NoMirror": "NoMirror",
    "nnUNetTrainer_SizeOversampleV2": "SizeOV2",
    "nnUNetTrainer_SizeOversampleV3": "SizeOV3",
    "nnUNetTrainer_SizeOversampleV3_NoMirror": "SizeOV3_NoMirror",
    "nnUNetTrainer_SwinUNETR": "SwinUNETR",
    "nnUNetTrainer_nnFormer": "nnFormer",
}


def method_from_trainer(trainer: str) -> str:
    if trainer in ALIASES:
        return ALIASES[trainer]
    method = trainer.removeprefix("nnUNetTrainer_")
    return method.replace("SizeOversample", "SizeOV")


def discover_trainers() -> list[str]:
    trainers: list[str] = []
    for checkpoint in sorted(DATASET003_RESULTS.glob("*/fold_0/checkpoint_best.pth")):
        result_name = checkpoint.parents[1].name
        if "__nnUNetPlans__" not in result_name:
            continue
        trainers.append(result_name.split("__nnUNetPlans__", 1)[0])
    return trainers


def has_predictions(method_dir: Path) -> bool:
    pred_dir = method_dir / "predictions"
    return pred_dir.exists() and any(pred_dir.glob("*.nii.gz"))


def has_report(method_dir: Path) -> bool:
    return (method_dir / "report_custom.txt").exists()


def read_overall(method_dir: Path) -> str:
    report = method_dir / "report_custom.txt"
    if not report.exists():
        return "N/A"
    for line in report.read_text(encoding="utf-8").splitlines():
        m = re.search(r"Overall\s*:.*?=\s*([0-9.]+)", line)
        if m:
            return m.group(1)
    return "N/A"


def run_method(method: str, trainer: str, gpu: int, no_vis: bool, force: bool) -> str:
    method_dir = EXT_RESULT_ROOT / method
    if not force and has_predictions(method_dir) and has_report(method_dir):
        return "skip"

    cmd = [
        sys.executable,
        str(SCRIPT),
        "--method",
        method,
        "--trainer",
        trainer,
        "--gpu",
        str(gpu),
        "--checkpoint",
        "checkpoint_best.pth",
    ]
    if force or not has_predictions(method_dir):
        cmd.append("--predict")
    if no_vis:
        cmd.append("--no_vis")

    print(f"\n{'=' * 72}")
    print(f"[运行] {method}  trainer={trainer}")
    print(f"  cmd: {' '.join(cmd)}")
    print(f"{'=' * 72}")
    ret = subprocess.run(cmd, cwd=SCRIPT.parent.parent.parent)
    return "done" if ret.returncode == 0 else "fail"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="+", help="只运行指定 method 或 trainer")
    parser.add_argument("--force", action="store_true", help="强制重跑")
    parser.add_argument("--no_vis", action="store_true", help="跳过可视化")
    parser.add_argument("--gpu", type=int, default=0, help="CUDA 设备编号")
    parser.add_argument("--dry_run", action="store_true", help="只打印将要运行的方法，不启动推理")
    args = parser.parse_args()

    requested = set(args.only) if args.only else None
    rows: list[tuple[str, str]] = []
    for trainer in discover_trainers():
        method = method_from_trainer(trainer)
        if requested and trainer not in requested and method not in requested:
            continue
        rows.append((method, trainer))

    if not rows:
        raise RuntimeError("没有找到可运行的 Dataset003_Liver checkpoint_best.pth")

    results = []
    for method, trainer in rows:
        if args.dry_run:
            print(f"[dry-run] {method}  trainer={trainer}")
            results.append((method, trainer, "dry-run", read_overall(EXT_RESULT_ROOT / method)))
            continue

        status = run_method(method, trainer, args.gpu, args.no_vis, args.force)
        if status == "skip":
            print(f"[跳过] {method} 已有 predictions + report，使用 --force 重跑")
        results.append((method, trainer, status, read_overall(EXT_RESULT_ROOT / method)))

    print(f"\n{'=' * 86}")
    print(f"{'method':<28} {'trainer':<44} {'HCC Overall':>11}  status")
    print("-" * 86)
    for method, trainer, status, overall in results:
        print(f"{method:<28} {trainer:<44} {overall:>11}  {status}")
    print("=" * 86)


if __name__ == "__main__":
    main()
