"""
Run Dataset003_Liver fixed internal test with an explicit checkpoint into a separate output root.

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
import hashlib
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
    observed = {path.name.removesuffix(".nii.gz") for path in pred_dir.glob("*.nii.gz")} if pred_dir.is_dir() else set()
    return observed == set(cases)


def _checkpoint_identity(results_root: Path, trainer: str, checkpoint: str) -> dict:
    checkpoint_path = (
        results_root / "Dataset003_Liver"
        / f"{trainer}__nnUNetPlans__3d_fullres" / "fold_0" / checkpoint
    )
    import torch

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    digest = hashlib.sha256()
    with checkpoint_path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {
        "dataset": "Dataset003_Liver",
        "trainer": trainer,
        "configuration": "3d_fullres",
        "fold": 0,
        "checkpoint": checkpoint,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": digest.hexdigest(),
        "checkpoint_epoch": int(payload.get("current_epoch", -1)),
    }


def _write_evaluation_provenance(method_dir: Path, identity: dict) -> None:
    (method_dir / "evaluation_provenance.json").write_text(
        json.dumps(identity, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _require_matching_provenance(method_dir: Path, identity: dict) -> None:
    path = method_dir / "evaluation_provenance.json"
    if not path.is_file():
        raise RuntimeError(
            f"Existing LiTS predictions have no checkpoint provenance: {path}. "
            "Use --force to replace them explicitly."
        )
    recorded = json.loads(path.read_text(encoding="utf-8"))
    if recorded != identity:
        raise RuntimeError(
            "Existing LiTS predictions belong to a different checkpoint; "
            "use --force to replace them explicitly."
        )


def _append_report_provenance(report: Path, identity: dict) -> None:
    text = report.read_text(encoding="utf-8").rstrip()
    marker = "[Model Provenance]"
    if marker in text:
        text = text.split(marker, 1)[0].rstrip()
    lines = [marker, *(f"{key}: {value}" for key, value in identity.items())]
    report.write_text(text + "\n\n" + "\n".join(lines) + "\n", encoding="utf-8")


def validate_checkpoint_provenance(
    trainer: str,
    results_root: Path | None = None,
    dataset: str = "Dataset003_Liver",
    fold: int = 0,
    checkpoint: str = "checkpoint_best.pth",
) -> None:
    """Validate selected checkpoint identity; guard against a stale base best.

    The historical modification-time/epoch guard applies only to the base
    ``checkpoint_best.pth`` convention. Task02 intentionally uses the separate
    ``checkpoint_task02_best.pth`` selected from its dual-domain validation.
    """
    results_root = Path(results_root) if results_root is not None else WORKSPACE / "results_v2"
    fold_dir = (
        results_root / dataset
        / f"{trainer}__nnUNetPlans__3d_fullres" / f"fold_{fold}"
    )
    checkpoint_path = fold_dir / checkpoint
    final_path = fold_dir / "checkpoint_final.pth"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing requested checkpoint: {checkpoint_path}")

    import torch

    selected = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint == "checkpoint_best.pth" and selected.get("trainer_name") != trainer:
        raise RuntimeError(
            f"Checkpoint trainer mismatch: requested={trainer}, "
            f"checkpoint={selected.get('trainer_name')} ({checkpoint_path})"
        )
    if checkpoint != "checkpoint_best.pth" or not final_path.is_file():
        return

    best = selected
    final = torch.load(final_path, map_location="cpu", weights_only=False)
    best_epoch = int(best.get("current_epoch", -1))
    final_epoch = int(final.get("current_epoch", -1))
    overwritten_later = checkpoint_path.stat().st_mtime > final_path.stat().st_mtime
    epoch_regressed = best_epoch >= 0 and final_epoch >= 0 and best_epoch < final_epoch
    if overwritten_later and epoch_regressed:
        best_time = datetime.fromtimestamp(checkpoint_path.stat().st_mtime).isoformat(timespec="seconds")
        final_time = datetime.fromtimestamp(final_path.stat().st_mtime).isoformat(timespec="seconds")
        raise RuntimeError(
            "Suspicious checkpoint_best provenance: best was written after final "
            f"but regressed from epoch {final_epoch} to {best_epoch}. "
            f"best={checkpoint_path} ({best_time}); final={final_path} ({final_time}). "
            "Refusing to create a best-only report from a likely overwritten checkpoint."
        )


def validate_best_checkpoint_provenance(
    trainer: str,
    results_root: Path | None = None,
    dataset: str = "Dataset003_Liver",
    fold: int = 0,
) -> None:
    """Compatibility wrapper for existing source-only evaluation callers."""
    validate_checkpoint_provenance(
        trainer,
        results_root=results_root,
        dataset=dataset,
        fold=fold,
        checkpoint="checkpoint_best.pth",
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
            domain_dir: str = "Dataset003_Liver", checkpoint: str = "checkpoint_best.pth") -> None:
    os.environ.setdefault("nnUNet_raw", str(WORKSPACE / "raw"))
    os.environ.setdefault("nnUNet_preprocessed", str(WORKSPACE / "preprocessed"))
    os.environ.setdefault("nnUNet_results", str(WORKSPACE / "results_v2"))

    method_dir = result_root / domain_dir / method
    pred_dir = method_dir / "predictions"
    report = method_dir / "test_report_custom.txt"
    viz_dir = method_dir / "test_viz"
    method_dir.mkdir(parents=True, exist_ok=True)
    cases = load_test_cases()
    model_results_root = Path(os.environ["nnUNet_results"])
    validate_checkpoint_provenance(
        trainer,
        results_root=model_results_root,
        dataset="Dataset003_Liver",
        fold=0,
        checkpoint=checkpoint,
    )
    checkpoint_identity = _checkpoint_identity(model_results_root, trainer, checkpoint)

    complete_predictions = predictions_are_complete(pred_dir, cases)
    complete_artifacts = (
        (pred_dir / "summary.json").is_file()
        and report.is_file()
        and viz_dir.is_dir()
        and any(viz_dir.rglob("*.png"))
    )
    if complete_predictions and complete_artifacts and not force:
        print(f"[skip] {method}: predictions, report and test_viz are complete")
        return

    if complete_predictions and not force:
        _require_matching_provenance(method_dir, checkpoint_identity)

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
            "-chk", checkpoint,
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        print(f"[predict] CUDA_VISIBLE_DEVICES={gpu} {' '.join(cmd)}")
        subprocess.run(cmd, cwd=str(NNUNET_ROOT), env=env, check=True)
        if not predictions_are_complete(pred_dir, cases):
            observed = {path.name.removesuffix(".nii.gz") for path in pred_dir.glob("*.nii.gz")}
            missing = sorted(set(cases) - observed)
            stale = sorted(observed - set(cases))
            raise RuntimeError(
                f"Invalid fixed-test predictions for {method}; missing={missing or 'none'}, "
                f"stale={stale or 'none'}"
            )
        _write_evaluation_provenance(method_dir, checkpoint_identity)
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
    _append_report_provenance(report, checkpoint_identity)

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
    parser.add_argument("--checkpoint", default="checkpoint_best.pth",
                        help="checkpoint filename in the trainer fold directory")
    args = parser.parse_args()

    trainer = args.trainer
    method = args.method or method_from_trainer(trainer)
    run_one(trainer, method, args.gpu, Path(args.result_root), args.force, args.domain_dir, args.checkpoint)


if __name__ == "__main__":
    main()
