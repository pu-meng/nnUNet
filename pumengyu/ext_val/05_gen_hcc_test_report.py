"""
Generate a report on HCCReferencedCT fixed held-out test cases.

The evaluated model may be a Dataset003_Liver source-only model or a
Dataset013_HCCReferencedCT-adapted model. Metrics always include only the
fixed test partition; HCC train/validation cases are excluded from evaluation.

Outputs by default:
    /home/PuMengYu/nnUNet_workspace/results_v2/ExternalVal_HCCReferencedCT/<method>/
        predictions/
        report_custom.txt
        test_viz/

Example:
    python pumengyu/ext_val/05_gen_hcc_test_report.py \\
        --method MedNeXt_MLA_MoE \\
        --predict \\
        --trainer nnUNetTrainer_MedNeXt_MLA_MoE \\
        --checkpoint checkpoint_best.pth \\
        --gpu 0
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
SPLIT_INFO = HCC_PREPROCESSED / "split_info_701020_stratified_v2.json"
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

    expected_images = {f"{case}_0000.nii.gz" for case in cases}
    expected_labels = {f"{case}.nii.gz" for case in cases}
    for directory, expected in ((IMAGE_DIR, expected_images), (LABEL_DIR, expected_labels)):
        for path in directory.glob("*.nii.gz"):
            if path.name in expected:
                continue
            if not path.is_symlink():
                raise RuntimeError(f"Refusing to remove unexpected non-symlink input: {path}")
            path.unlink()
            print(f"[清理] 移除固定测试清单之外的旧链接: {path}")

    for case in cases:
        _safe_symlink(HCC_RAW / "imagesTr" / f"{case}_0000.nii.gz", IMAGE_DIR / f"{case}_0000.nii.gz")
        _safe_symlink(HCC_RAW / "labelsTr" / f"{case}.nii.gz", LABEL_DIR / f"{case}.nii.gz")

    info = {
        case: {
            "source_dataset": HCC_DATASET,
            "split": "test",
            "protocol": f"HCC fixed 70/10/21 held-out test from {SPLIT_INFO.name}; HCC train/val are excluded from metrics.",
        }
        for case in cases
    }
    INFO_FILE.write_text(json.dumps(info, indent=4, ensure_ascii=False) + "\n", encoding="utf-8")
    return cases


def _patch_report_header(
    method: str,
    cases: list[str],
    trainer: str,
    checkpoint: str,
    dataset: str,
) -> None:
    report_path = EXT_RESULT_ROOT / method / "report_custom.txt"
    if not report_path.exists():
        return
    text = report_path.read_text(encoding="utf-8")
    text = text.replace(
        "nnUNet External Validation Report (IRCADb)",
        "nnUNet External Validation Report (HCCReferencedCT fixed test)",
    )
    marker = f"method   : {method}\n"
    dataset_text = str(dataset)
    dataset_id = (
        int(dataset_text.removeprefix("Dataset")[:3])
        if dataset_text.startswith("Dataset")
        else int(dataset_text)
    )
    is_hcc_adapted = dataset_id == 13 or "HCCAdapter" in trainer
    is_msd_hcc_mixed = "MSDHCCPretrain" in trainer or "MSDHCCMix" in trainer
    if is_msd_hcc_mixed:
        provenance = (
            f"checkpoint: {checkpoint}\n"
            f"model_source: Dataset003_Liver-plan mixed-source model ({trainer}); "
            "MSD train and HCC train cases used for fitting; HCC val/test excluded from fitting and model selection\n"
        )
    elif is_hcc_adapted:
        provenance = (
            f"checkpoint: {checkpoint}\n"
            f"model_source: Dataset013_HCCReferencedCT-trained/adapted model ({trainer}); "
            "HCC train/val may be used for fitting/selection, never for test metrics\n"
        )
    else:
        provenance = (
            f"checkpoint: {checkpoint}\n"
            f"model_source: Dataset003_Liver source-only model ({trainer}); "
            "HCC train/val not used for fitting or model selection\n"
        )
    protocol = (
        marker
        + f"dataset  : {HCC_DATASET}\n"
        + f"split    : fixed 70/10/21 test only ({SPLIT_INFO.name})\n"
        + f"n_test   : {len(cases)}\n"
        + f"split_info: {SPLIT_INFO}\n"
        + provenance
    )
    removable_prefixes = ("dataset  :", "split    :", "n_test   :", "split_info:", "checkpoint:", "model_source:")
    text = "\n".join(
        line for line in text.splitlines()
        if not line.startswith(removable_prefixes)
    )
    if marker.strip() in text.splitlines():
        text = text.replace(marker.strip(), protocol.rstrip("\n"), 1)
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
    force_predict: bool = False,
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
        force_predict=force_predict,
    )
    _patch_report_header(method, cases, trainer, checkpoint, dataset)
    print(f"[完成] ExternalVal_HCCReferencedCT/{method}/  test n={len(cases)}")


def main() -> None:
    global SPLIT_INFO, EXT_VAL_ROOT, IMAGE_DIR, LABEL_DIR, INFO_FILE, EXT_RESULT_ROOT

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
    parser.add_argument("--force_predict", action="store_true",
                        help="即使预测文件齐全也强制重新推理；默认复用并只补报告/可视化")
    parser.add_argument("--split_info", default=str(SPLIT_INFO), help="HCC split_info json，默认兼容文件名；当前内容为 stratified v2")
    parser.add_argument("--ext_val_root", default=str(EXT_VAL_ROOT), help="HCC test images/labels symlink root")
    parser.add_argument("--result_root", default=str(EXT_RESULT_ROOT), help="HCC external validation output root")
    args = parser.parse_args()

    if args.predict and not args.trainer:
        raise ValueError("--predict 需要同时指定 --trainer")

    SPLIT_INFO = Path(args.split_info)
    EXT_VAL_ROOT = Path(args.ext_val_root)
    IMAGE_DIR = EXT_VAL_ROOT / "images"
    LABEL_DIR = EXT_VAL_ROOT / "labels"
    INFO_FILE = EXT_VAL_ROOT / "case_info.json"
    EXT_RESULT_ROOT = Path(args.result_root)

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
        force_predict=args.force_predict,
    )


if __name__ == "__main__":
    main()
