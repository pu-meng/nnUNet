"""
Run Dataset003_Liver fixed internal test with checkpoint_best.pth into a separate output root.

This does not touch the original results_v2 trainer folders. It predicts only the
26 cases listed in split_info_712.json and writes:
    <result_root>/Dataset003_Liver/<method>/predictions/
    <result_root>/Dataset003_Liver/<method>/test_report_custom.txt
    <result_root>/Dataset003_Liver/<method>/test_viz/

Predictions, summary, report and visualizations are treated as one complete
artifact set. If predictions already exist but a downstream artifact is missing,
the script reuses the NIfTI files and repairs the missing outputs without GPU
inference.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


NNUNET_ROOT = Path("/home/PuMengYu/nnUNet")
WORKSPACE = Path("/home/PuMengYu/nnUNet_workspace")
RAW = WORKSPACE / "raw" / "Dataset003_Liver"
PREPROCESSED = WORKSPACE / "preprocessed" / "Dataset003_Liver"
DEFAULT_RESULT_ROOT = WORKSPACE / "results_v2_best"

if str(NNUNET_ROOT) not in sys.path:
    sys.path.insert(0, str(NNUNET_ROOT))


ALIASES = {
    "nnUNetTrainer_Baseline": "Baseline",
    "nnUNetTrainer_DeepDWIBMedConfig": "DeepDWIBMedConfig",
    "nnUNetTrainer_DeepDWIBResGN": "DeepDWIBResGN",
    "nnUNetTrainer_DeepPlainResGN": "DeepPlainResGN",
    "nnUNetTrainer_DeepResGN_MLA": "DeepResGN_MLA",
    "nnUNetTrainer_MLAUNet": "MLAUNet",
    "nnUNetTrainer_MLAUNet_MoE_SizeOversampleV5": "MoE_SizeOV5",
    "nnUNetTrainer_MedNeXt": "MedNeXt",
    "nnUNetTrainer_MedNeXt_MLA_MoE": "MedNeXt_MLA_MoE",
    "nnUNetTrainer_MedNeXt_MLA_MoE_SizeOV4": "MedNeXt_MLA_MoE_SizeOV4",
    "nnUNetTrainer_MedNeXt_SizeOV4": "MedNeXt_SizeOV4",
    "nnUNetTrainer_SizeOversampleV2": "SizeOV2",
    "nnUNetTrainer_SizeOversampleV3": "SizeOV3",
    "nnUNetTrainer_SwinUNETR": "SwinUNETR",
    "nnUNetTrainer_nnFormer": "nnFormer",
}


def method_from_trainer(trainer: str) -> str:
    if trainer in ALIASES:
        return ALIASES[trainer]
    return trainer.removeprefix("nnUNetTrainer_").replace("SizeOversample", "SizeOV")


def load_test_cases() -> list[str]:
    split_info = json.loads((PREPROCESSED / "split_info_712.json").read_text(encoding="utf-8"))
    cases = list(split_info["test"]["cases"])
    if not cases:
        raise RuntimeError("Dataset003 fixed test split is empty")
    return cases


def prepare_test_images(result_root: Path) -> Path:
    cases = load_test_cases()
    out_dir = result_root / "_inputs" / "Dataset003_Liver_test26"
    out_dir.mkdir(parents=True, exist_ok=True)
    for case in cases:
        src = RAW / "imagesTr" / f"{case}_0000.nii.gz"
        dst = out_dir / f"{case}_0000.nii.gz"
        if dst.exists() or dst.is_symlink():
            continue
        dst.symlink_to(src)
    return out_dir


def predictions_are_complete(pred_dir: Path, cases: list[str]) -> bool:
    return all((pred_dir / f"{case}.nii.gz").is_file() for case in cases)


def validate_best_checkpoint_provenance(trainer: str) -> None:
    """Reject a best checkpoint that was overwritten by a later short run."""
    fold_dir = (
        WORKSPACE / "results_v2" / "Dataset003_Liver"
        / f"{trainer}__nnUNetPlans__3d_fullres" / "fold_0"
    )
    best_path = fold_dir / "checkpoint_best.pth"
    final_path = fold_dir / "checkpoint_final.pth"
    if not best_path.is_file():
        raise FileNotFoundError(f"Missing best checkpoint: {best_path}")
    if not final_path.is_file():
        return

    import torch

    best = torch.load(best_path, map_location="cpu", weights_only=False)
    final = torch.load(final_path, map_location="cpu", weights_only=False)
    best_epoch = int(best.get("current_epoch", -1))
    final_epoch = int(final.get("current_epoch", -1))
    overwritten_later = best_path.stat().st_mtime > final_path.stat().st_mtime
    epoch_regressed = best_epoch >= 0 and final_epoch >= 0 and best_epoch < final_epoch
    if overwritten_later and epoch_regressed:
        best_time = datetime.fromtimestamp(best_path.stat().st_mtime).isoformat(timespec="seconds")
        final_time = datetime.fromtimestamp(final_path.stat().st_mtime).isoformat(timespec="seconds")
        raise RuntimeError(
            "Suspicious checkpoint_best provenance: best was written after final "
            f"but regressed from epoch {final_epoch} to {best_epoch}. "
            f"best={best_path} ({best_time}); final={final_path} ({final_time}). "
            "Refusing to create a best-only report from a likely overwritten checkpoint."
        )


def ensure_summary(pred_dir: Path) -> Path:
    summary_path = pred_dir / "summary.json"
    from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder
    from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO

    compute_metrics_on_folder(
        folder_ref=str(PREPROCESSED / "gt_segmentations"),
        folder_pred=str(pred_dir),
        output_file=str(summary_path),
        image_reader_writer=SimpleITKIO(),
        file_ending=".nii.gz",
        regions_or_labels=[1, 2],
        ignore_label=None,
        chill=True,
    )
    if not summary_path.is_file():
        raise RuntimeError(f"Failed to generate summary: {summary_path}")
    return summary_path


def run_one(trainer: str, method: str, gpu: int, result_root: Path, force: bool,
            domain_dir: str = "Dataset003_Liver") -> None:
    os.environ.setdefault("nnUNet_raw", str(WORKSPACE / "raw"))
    os.environ.setdefault("nnUNet_preprocessed", str(WORKSPACE / "preprocessed"))
    os.environ.setdefault("nnUNet_results", str(WORKSPACE / "results_v2"))

    method_dir = result_root / domain_dir / method
    pred_dir = method_dir / "predictions"
    report = method_dir / "test_report_custom.txt"
    viz_dir = method_dir / "test_viz"
    method_dir.mkdir(parents=True, exist_ok=True)
    cases = load_test_cases()
    validate_best_checkpoint_provenance(trainer)

    complete_predictions = predictions_are_complete(pred_dir, cases)
    complete_artifacts = report.is_file() and viz_dir.is_dir() and any(viz_dir.rglob("*.png"))
    if complete_predictions and complete_artifacts and not force:
        print(f"[skip] {method}: predictions, report and test_viz are complete")
        return

    if force or not complete_predictions:
        input_dir = prepare_test_images(result_root)
        pred_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            "nnUNetv2_predict",
            "-i", str(input_dir),
            "-o", str(pred_dir),
            "-d", "003",
            "-c", "3d_fullres",
            "-tr", trainer,
            "-f", "0",
            "-chk", "checkpoint_best.pth",
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        print(f"[predict] CUDA_VISIBLE_DEVICES={gpu} {' '.join(cmd)}")
        subprocess.run(cmd, cwd=str(NNUNET_ROOT), env=env, check=True)
        if not predictions_are_complete(pred_dir, cases):
            missing = [case for case in cases if not (pred_dir / f"{case}.nii.gz").is_file()]
            raise RuntimeError(f"Incomplete predictions for {method}; missing={missing}")
    else:
        print(f"[reuse] {method}: reuse {len(cases)} existing predictions")

    ensure_summary(pred_dir)

    from pumengyu.tools.analyasis.eval_fold_report import run_eval_report

    run_eval_report(
        val_dir=pred_dir,
        gt_dir=PREPROCESSED / "gt_segmentations",
        img_dir=RAW / "imagesTr",
        no_vis=True,
        min_tumor_size=0,
        out_dir=method_dir,
        report_name="test_report_custom.txt",
    )
    if not report.is_file():
        raise RuntimeError(f"Missing report after evaluation: {report}")

    from pumengyu.mixins import _gen_viz_pngs_and_cleanup

    viz_dir.mkdir(exist_ok=True)
    _gen_viz_pngs_and_cleanup(
        pred_folder=pred_dir,
        gt_dir=PREPROCESSED / "gt_segmentations",
        img_dir=RAW / "imagesTr",
        out_viz_dir=viz_dir,
        min_voxel=20,
        delete_nii=False,
    )
    if not any(viz_dir.rglob("*.png")):
        raise RuntimeError(f"Missing visualizations after evaluation: {viz_dir}")
    print(f"[done] {method}: report={report} viz={viz_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainer", required=True)
    parser.add_argument("--method", default="")
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--result_root", default=str(DEFAULT_RESULT_ROOT))
    parser.add_argument("--domain_dir", default="Dataset003_Liver",
                        help="result_root 下的数据域目录名")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    trainer = args.trainer
    method = args.method or method_from_trainer(trainer)
    run_one(trainer, method, args.gpu, Path(args.result_root), args.force, args.domain_dir)


if __name__ == "__main__":
    main()
