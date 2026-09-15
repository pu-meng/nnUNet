"""Formal three-domain evaluation for one completed Task02 Stage-2 run.

This program is deliberately separate from training. It accepts only the
selected ``checkpoint_task02_best.pth`` after it re-audits the training-time
selection record, then delegates prediction/report/viz generation to the
existing full-artifact evaluators for LiTS, IRCADb and HCC fixed test sets.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from pumengyu.task02.selection import audit_task02_selection
from pumengyu.task02.splits import HCC_SPLIT_INFO, LITS_SPLIT_INFO, sha256_file


REPO_ROOT = Path("/home/PuMengYu/nnUNet")
WORKSPACE_ROOT = Path("/home/PuMengYu/nnUNet_workspace")
CHECKPOINT_NAME = "checkpoint_task02_best.pth"


def _load_cases(split_path: Path, partition: str) -> list[str]:
    payload = json.loads(split_path.read_text(encoding="utf-8"))
    cases = list(payload[partition]["cases"])
    if not cases or len(cases) != len(set(cases)):
        raise RuntimeError(f"Invalid fixed {partition} case list in {split_path}")
    return cases


def _trainer_fold(model_results_root: Path, trainer: str) -> Path:
    fold_dir = (
        model_results_root / "Dataset003_Liver"
        / f"{trainer}__nnUNetPlans__3d_fullres" / "fold_0"
    )
    if not fold_dir.is_dir():
        raise FileNotFoundError(f"Task02 trainer fold is missing: {fold_dir}")
    return fold_dir


def _load_and_verify_selection(fold_dir: Path, trainer: str) -> dict[str, Any]:
    manifest_path = fold_dir / "task02_manifest.json"
    selection_path = fold_dir / "task02_selection.json"
    if not manifest_path.is_file() or not selection_path.is_file():
        raise RuntimeError("Task02 manifest or selection record is missing; formal evaluation is not permitted.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if manifest.get("run_kind") != "formal":
        raise RuntimeError("Task02 smoke runs are not eligible for formal three-domain evaluation")
    mode = manifest.get("mode")
    if mode not in {"hcc_only", "replay"} or selection.get("mode", mode) != mode:
        raise RuntimeError("Task02 mode in manifest/selection is missing or inconsistent")
    audit = audit_task02_selection(fold_dir=fold_dir, mode=mode)
    selected = fold_dir / CHECKPOINT_NAME
    if not selected.is_file():
        raise FileNotFoundError(f"Selected Task02 checkpoint is missing: {selected}")
    if Path(selection.get("selected_checkpoint", "")).resolve() != selected.resolve():
        raise RuntimeError("task02_selection.json points to a checkpoint outside this trainer fold")
    selected_sha = sha256_file(selected)
    if selected_sha != selection.get("selected_checkpoint_sha256"):
        raise RuntimeError("Selected Task02 checkpoint SHA differs from task02_selection.json")

    import torch

    checkpoint = torch.load(selected, map_location="cpu", weights_only=False)
    checkpoint_trainer = checkpoint.get("trainer_name")
    if selection.get("status") == "selected" and checkpoint_trainer != trainer:
        raise RuntimeError(
            f"Selected adapted checkpoint trainer mismatch: expected={trainer}, got={checkpoint_trainer}"
        )
    if selection.get("status") == "no_acceptable_adaptation":
        source = manifest.get("source", {})
        if checkpoint_trainer != source.get("trainer") or selected_sha != source.get("sha256"):
            raise RuntimeError("Task02 source fallback checkpoint does not match frozen source provenance")
    return {
        "manifest": manifest,
        "selection": selection,
        "selection_audit": audit,
        "checkpoint_path": str(selected),
        "checkpoint_sha256": selected_sha,
        "checkpoint_epoch": int(checkpoint.get("current_epoch", -1)),
    }


def _run(command: list[str], env: dict[str, str], dry_run: bool) -> None:
    print("[Task02 final] " + " ".join(command))
    if not dry_run:
        subprocess.run(command, cwd=str(REPO_ROOT), env=env, check=True)


def _audit_domain(
    *, domain: str, method_dir: Path, expected_cases: list[str], report_name: str
) -> dict[str, Any]:
    pred_dir = method_dir / "predictions"
    actual_cases = sorted(path.name.removesuffix(".nii.gz") for path in pred_dir.glob("*.nii.gz")) if pred_dir.is_dir() else []
    expected = sorted(expected_cases)
    png_count = len(list((method_dir / "test_viz").rglob("*.png"))) if (method_dir / "test_viz").is_dir() else 0
    complete = (
        actual_cases == expected
        and (pred_dir / "summary.json").is_file()
        and (method_dir / report_name).is_file()
        and png_count > 0
    )
    return {
        "domain": domain,
        "expected_case_count": len(expected),
        "prediction_case_count": len(actual_cases),
        "missing_cases": sorted(set(expected) - set(actual_cases)),
        "stale_cases": sorted(set(actual_cases) - set(expected)),
        "summary_json": str(pred_dir / "summary.json"),
        "summary_exists": (pred_dir / "summary.json").is_file(),
        "report": str(method_dir / report_name),
        "report_exists": (method_dir / report_name).is_file(),
        "test_viz": str(method_dir / "test_viz"),
        "test_viz_png_count": png_count,
        "status": "complete" if complete else "incomplete",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trainer", required=True, help="one concrete Task02 trainer class")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--model_results_root", default=str(WORKSPACE_ROOT / "results_task02"))
    parser.add_argument("--evaluation_root", default="",
                        help="separate three-domain artifact root; default is under model_results_root")
    parser.add_argument("--force_predict", action="store_true",
                        help="explicitly redo existing predictions; default is safe prediction reuse plus artifact repair")
    parser.add_argument("--dry_run", action="store_true", help="print exact commands after provenance checks only")
    args = parser.parse_args()

    model_root = Path(args.model_results_root).resolve()
    fold_dir = _trainer_fold(model_root, args.trainer)
    provenance = _load_and_verify_selection(fold_dir, args.trainer)
    method = args.trainer.removeprefix("nnUNetTrainer_")
    evaluation_root = (
        Path(args.evaluation_root).resolve()
        if args.evaluation_root
        else model_root / "Task02_FinalEvaluation" / method
    )
    evaluation_root.mkdir(parents=True, exist_ok=True)

    lits_cases = _load_cases(LITS_SPLIT_INFO, "test")
    hcc_cases = _load_cases(HCC_SPLIT_INFO, "test")
    ircad_info = WORKSPACE_ROOT / "external_val" / "ircadb_full" / "case_info.json"
    if not ircad_info.is_file():
        raise FileNotFoundError(f"IRCADb fixed-case manifest is missing: {ircad_info}")
    ircad_cases = sorted(json.loads(ircad_info.read_text(encoding="utf-8")).keys())
    if not ircad_cases:
        raise RuntimeError("IRCADb fixed-case manifest is empty")

    manifest = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "task": "Task02 formal three-domain evaluation",
        "trainer": args.trainer,
        "model_results_root": str(model_root),
        "fold_dir": str(fold_dir),
        "checkpoint_name": CHECKPOINT_NAME,
        "selection_provenance": provenance,
        "fixed_test_sets": {
            "lits": {"split": str(LITS_SPLIT_INFO), "case_ids": lits_cases},
            "ircadb": {"case_manifest": str(ircad_info), "case_ids": ircad_cases},
            "hcc": {"split": str(HCC_SPLIT_INFO), "case_ids": hcc_cases},
        },
    }
    (evaluation_root / "task02_final_evaluation_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    env = os.environ.copy()
    env["nnUNet_raw"] = str(WORKSPACE_ROOT / "raw")
    env["nnUNet_preprocessed"] = str(WORKSPACE_ROOT / "preprocessed")
    env["nnUNet_results"] = str(model_root)
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    force_args = ["--force_predict"] if args.force_predict else []
    internal_force_args = ["--force"] if args.force_predict else []
    commands = [
        [
            sys.executable, "pumengyu/tools/run_internal_test_best_report.py",
            "--trainer", args.trainer, "--method", method, "--gpu", str(args.gpu),
            "--result_root", str(evaluation_root), "--domain_dir", "LiTS",
            "--checkpoint", CHECKPOINT_NAME,
            *internal_force_args,
        ],
        [
            sys.executable, "pumengyu/ext_val/03_gen_method_report.py",
            "--method", method, "--predict", "--trainer", args.trainer,
            "--dataset", "003", "--checkpoint", CHECKPOINT_NAME, "--gpu", str(args.gpu),
            "--model_results_root", str(model_root), "--result_root", str(evaluation_root / "IRCADb"),
            *force_args,
        ],
        [
            sys.executable, "pumengyu/ext_val/05_gen_hcc_test_report.py",
            "--method", method, "--predict", "--trainer", args.trainer,
            "--dataset", "003", "--checkpoint", CHECKPOINT_NAME, "--gpu", str(args.gpu),
            "--model_results_root", str(model_root), "--result_root", str(evaluation_root / "HCC"),
            *force_args,
        ],
    ]
    for command in commands:
        _run(command, env, args.dry_run)
    if args.dry_run:
        return

    artifact_audit = {
        "schema_version": 1,
        "checkpoint_sha256": provenance["checkpoint_sha256"],
        "domains": {
            "LiTS": _audit_domain(
                domain="LiTS", method_dir=evaluation_root / "LiTS" / method,
                expected_cases=lits_cases, report_name="test_report_custom.txt",
            ),
            "IRCADb": _audit_domain(
                domain="IRCADb", method_dir=evaluation_root / "IRCADb" / method,
                expected_cases=ircad_cases, report_name="report_custom.txt",
            ),
            "HCC": _audit_domain(
                domain="HCC", method_dir=evaluation_root / "HCC" / method,
                expected_cases=hcc_cases, report_name="report_custom.txt",
            ),
        },
    }
    artifact_audit["status"] = (
        "complete" if all(item["status"] == "complete" for item in artifact_audit["domains"].values())
        else "incomplete"
    )
    audit_path = evaluation_root / "task02_final_artifact_audit.json"
    audit_path.write_text(json.dumps(artifact_audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if artifact_audit["status"] != "complete":
        raise RuntimeError(f"Task02 final artifacts are incomplete; inspect {audit_path}")
    print(f"[Task02 final] complete three-domain artifact bundle: {evaluation_root}")


if __name__ == "__main__":
    main()
