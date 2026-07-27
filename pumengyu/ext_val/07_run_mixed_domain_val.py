"""Run one routed MSD + IRCADb + HCC mixed-domain validation experiment.

Routing is explicit so preprocessing plans and adapters cannot be mixed up:
  - MSD test: Dataset003 MedNeXt_MLA_MoE base model (adapter absent)
  - IRCADb:   Dataset003 MedNeXt_MLA_MoE base model (adapter absent)
  - HCC test: Dataset013 HCCAdapter701020 model (adapter enabled)

All predictions use checkpoint_best.pth, always generate visualizations, and are
grouped below one result root.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path


REPO = Path("/home/PuMengYu/nnUNet")
WORKSPACE = Path("/home/PuMengYu/nnUNet_workspace")
DEFAULT_ROOT = WORKSPACE / "results_v2" / "Dataset013_HCCReferencedCT/mixed_adapter"

BASE_METHOD = "MedNeXt_MLA_MoE_Base"
BASE_TRAINER = "nnUNetTrainer_MedNeXt_MLA_MoE"
ADAPTER_METHOD = "MedNeXt_MLA_MoE_HCCAdapter701020"
ADAPTER_TRAINER = "nnUNetTrainer_MedNeXt_MLA_MoE_HCCAdapter701020"
CHECKPOINT = "checkpoint_best.pth"


def _run(cmd: list[str], env: dict[str, str], dry_run: bool) -> None:
    print("[run] " + " ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, cwd=str(REPO), env=env, check=True)


def _extract_metrics(report: Path) -> dict[str, float | str]:
    if not report.exists():
        return {"status": "missing"}
    text = report.read_text(encoding="utf-8")
    result: dict[str, float | str] = {"status": "complete"}
    patterns = {
        "liver_dice": r"Liver\s*\n\s*Dice:\s*mean=([0-9.]+)",
        "tumor_dice": r"Tumor \u7efc\u5408\u6307\u6807[\s\S]*?Dice\s*:\s*mean=([0-9.]+)",
        "overall": r"Overall\s*:.*?=\s*([0-9.]+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            result[key] = float(match.group(1))
    return result


def _write_manifest(root: Path) -> None:
    routes = {
        "experiment": root.name,
        "checkpoint": CHECKPOINT,
        "routes": {
            "MSD": {
                "dataset": "Dataset003_Liver",
                "trainer": BASE_TRAINER,
                "adapter": "disabled (base architecture has no adapter)",
                "plans": "Dataset003_Liver",
            },
            "IRCADb": {
                "dataset": "Dataset003_Liver",
                "trainer": BASE_TRAINER,
                "adapter": "disabled (base architecture has no adapter)",
                "plans": "Dataset003_Liver",
            },
            "HCC": {
                "dataset": "Dataset013_HCCReferencedCT",
                "trainer": ADAPTER_TRAINER,
                "adapter": "enabled (HCC-trained adapter)",
                "plans": "Dataset013_HCCReferencedCT",
            },
        },
    }
    (root / "routing_manifest.json").write_text(
        json.dumps(routes, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _write_summary(root: Path) -> None:
    reports = {
        "MSD": root / "MSD" / BASE_METHOD / "test_report_custom.txt",
        "IRCADb": root / "IRCADb" / BASE_METHOD / "report_custom.txt",
        "HCC": root / "HCC" / ADAPTER_METHOD / "report_custom.txt",
    }
    summary = {domain: {"report": str(path), **_extract_metrics(path)} for domain, path in reports.items()}
    (root / "mixed_validation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--result_root", default=str(DEFAULT_ROOT))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    root = Path(args.result_root)
    root.mkdir(parents=True, exist_ok=True)
    _write_manifest(root)

    env = os.environ.copy()
    env.setdefault("nnUNet_raw", str(WORKSPACE / "raw"))
    env.setdefault("nnUNet_preprocessed", str(WORKSPACE / "preprocessed"))
    env.setdefault("nnUNet_results", str(WORKSPACE / "results_v2"))
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    msd = [
        sys.executable, "pumengyu/tools/run_internal_test_best_report.py",
        "--trainer", BASE_TRAINER, "--method", BASE_METHOD,
        "--gpu", str(args.gpu), "--result_root", str(root), "--domain_dir", "MSD",
    ]
    if args.force:
        msd.append("--force")

    ircadb = [
        sys.executable, "pumengyu/ext_val/03_gen_method_report.py",
        "--method", BASE_METHOD, "--predict", "--trainer", BASE_TRAINER,
        "--dataset", "003", "--checkpoint", CHECKPOINT,
        "--gpu", str(args.gpu), "--result_root", str(root / "IRCADb"),
    ]
    hcc = [
        sys.executable, "pumengyu/ext_val/05_gen_hcc_test_report.py",
        "--method", ADAPTER_METHOD, "--predict", "--trainer", ADAPTER_TRAINER,
        "--dataset", "013", "--checkpoint", CHECKPOINT,
        "--gpu", str(args.gpu), "--result_root", str(root / "HCC"),
    ]
    for command in (msd, ircadb, hcc):
        _run(command, env, args.dry_run)

    _write_summary(root)
    print(f"[done] mixed validation root: {root}")


if __name__ == "__main__":
    main()
