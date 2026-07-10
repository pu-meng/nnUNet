"""
Generate an external-test report on HCCReferencedCT fixed held-out test cases.

This script evaluates Dataset003_Liver-trained models on the HCC 70/10/21
split's held-out test set only. It does not use HCC train or validation cases.

Outputs:
    /home/PuMengYu/nnUNet_workspace/results_v2/ExternalVal_HCCReferencedCT/<method>/
        predictions/
        report_custom.txt
        test_viz/

Example:
    python pumengyu/ext_val/05_gen_hcc_test_report.py \\
        --method MedNeXt_MLA \\
        --predict \\
        --trainer nnUNetTrainer_MedNeXt_MLA \\
        --checkpoint checkpoint_best.pth \\
        --gpu 0 \\
        --no_vis
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path


NNUNET_ROOT = Path("/home/PuMengYu/nnUNet")
WORKSPACE = Path("/home/PuMengYu/nnUNet_workspace")
HCC_DATASET = "Dataset013_HCCReferencedCT"
HCC_RAW = WORKSPACE / "raw" / HCC_DATASET
HCC_PREPROCESSED = WORKSPACE / "preprocessed" / HCC_DATASET
SPLIT_INFO = HCC_PREPROCESSED / "split_info_701020_from_fold0.json"
EXT_VAL_ROOT = WORKSPACE / "external_val" / "hcc_referenced_ct_test"
IMAGE_DIR = EXT_VAL_ROOT / "images"
LABEL_DIR = EXT_VAL_ROOT / "labels"
INFO_FILE = EXT_VAL_ROOT / "case_info.json"
EXT_RESULT_ROOT = WORKSPACE / "results_v2" / "ExternalVal_HCCReferencedCT"


def _load_ircad_report_module():
    script = Path(__file__).with_name("03_gen_method_report.py")
    spec = importlib.util.spec_from_file_location("pmy_ircad_report", script)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load report module: {script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _safe_symlink(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    if dst.exists() or dst.is_symlink():
        if dst.resolve() == src.resolve():
            return
        raise FileExistsError(f"Refusing to overwrite existing path: {dst}")
    try:
        dst.symlink_to(src)
    except OSError:
        import shutil

        shutil.copy2(src, dst)


def load_test_cases() -> list[str]:
    split_info = json.loads(SPLIT_INFO.read_text(encoding="utf-8"))
    cases = split_info["test"]["cases"]
    if not cases:
        raise RuntimeError(f"No test cases in {SPLIT_INFO}")
    return list(cases)


def prepare_hcc_test_external_dir() -> list[str]:
    """Create a small external-val directory containing only the fixed HCC test cases."""
    cases = load_test_cases()
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    LABEL_DIR.mkdir(parents=True, exist_ok=True)

    for case in cases:
        _safe_symlink(HCC_RAW / "imagesTr" / f"{case}_0000.nii.gz", IMAGE_DIR / f"{case}_0000.nii.gz")
        _safe_symlink(HCC_RAW / "labelsTr" / f"{case}.nii.gz", LABEL_DIR / f"{case}.nii.gz")

    info = {
        case: {
            "source_dataset": HCC_DATASET,
            "split": "test",
            "protocol": "HCC fixed 70/10/21 held-out test; HCC train/val are not used.",
        }
        for case in cases
    }
    INFO_FILE.write_text(json.dumps(info, indent=4, ensure_ascii=False) + "\n", encoding="utf-8")
    return cases


def _patch_report_header(method: str, cases: list[str]) -> None:
    report_path = EXT_RESULT_ROOT / method / "report_custom.txt"
    if not report_path.exists():
        return
    text = report_path.read_text(encoding="utf-8")
    text = text.replace(
        "nnUNet External Validation Report (IRCADb)",
        "nnUNet External Validation Report (HCCReferencedCT fixed test)",
    )
    marker = f"method   : {method}\n"
    protocol = (
        marker
        + f"dataset  : {HCC_DATASET}\n"
        + "split    : fixed 70/10/21 test only\n"
        + f"n_test   : {len(cases)}\n"
        + f"split_info: {SPLIT_INFO}\n"
        + "checkpoint: external Dataset003_Liver model checkpoint; no HCC train/val used here\n"
    )
    if marker in text and "split    : fixed 70/10/21 test only" not in text:
        text = text.replace(marker, protocol, 1)
    report_path.write_text(text, encoding="utf-8")


def run(
    method: str,
    no_vis: bool,
    predict: bool,
    trainer: str,
    fold: int,
    gpu: int,
    min_voxel: int,
    checkpoint: str,
    dataset: str,
) -> None:
    cases = prepare_hcc_test_external_dir()

    report_module = _load_ircad_report_module()
    report_module.EXT_VAL_ROOT = EXT_VAL_ROOT
    report_module.LABEL_DIR = LABEL_DIR
    report_module.IMAGE_DIR = IMAGE_DIR
    report_module.INFO_FILE = INFO_FILE
    report_module.EXT_RESULT_ROOT = EXT_RESULT_ROOT
    report_module.NNUNET_RESULTS = WORKSPACE / "results_v2"

    report_module.run(
        method=method,
        no_vis=no_vis,
        predict=predict,
        trainer=trainer,
        fold=fold,
        gpu=gpu,
        min_voxel=min_voxel,
        checkpoint=checkpoint,
        dataset=dataset,
    )
    _patch_report_header(method, cases)
    print(f"[完成] ExternalVal_HCCReferencedCT/{method}/  test n={len(cases)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, help="输出方法名，如 MedNeXt_MLA")
    parser.add_argument("--no_vis", action="store_true", help="跳过可视化生成")
    parser.add_argument("--min_voxel", type=int, default=20, help="viz 最小 FP/FN 体素数")
    parser.add_argument("--predict", action="store_true", help="先运行 nnUNetv2_predict 再生成报告")
    parser.add_argument("--trainer", default="", help="Dataset003_Liver trainer 名称")
    parser.add_argument("--fold", type=int, default=0, help="fold 编号")
    parser.add_argument("--gpu", type=int, default=0, help="CUDA_VISIBLE_DEVICES")
    parser.add_argument("--checkpoint", default="checkpoint_best.pth", help="默认用 best checkpoint")
    parser.add_argument("--dataset", default="003", help="模型所属训练数据集，Dataset003_Liver 用 003")
    args = parser.parse_args()

    if args.predict and not args.trainer:
        raise ValueError("--predict 需要同时指定 --trainer")

    os.environ.setdefault("nnUNet_results", str(WORKSPACE / "results_v2"))
    run(
        method=args.method,
        no_vis=args.no_vis,
        predict=args.predict,
        trainer=args.trainer,
        fold=args.fold,
        gpu=args.gpu,
        min_voxel=args.min_voxel,
        checkpoint=args.checkpoint,
        dataset=args.dataset,
    )


if __name__ == "__main__":
    main()
