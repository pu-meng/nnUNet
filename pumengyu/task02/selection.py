"""Offline audit for Task02 lightweight checkpoint selection.

The trainer itself selects ``checkpoint_task02_best.pth`` while it has the
candidate weights in memory. This module independently checks that the saved
JSONL history and final selection record obey the frozen rule. It never opens a
test set and never creates predictions, reports, or visualizations.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pumengyu.task02.splits import sha256_file


RETENTION_MAX_ABSOLUTE_DROP = 0.02
Mode = Literal["hcc_only", "replay"]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Task02 validation history is missing: {path}")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Malformed JSONL at {path}:{line_number}") from exc
        if not isinstance(record, dict):
            raise RuntimeError(f"Non-object JSONL record at {path}:{line_number}")
        records.append(record)
    if not records:
        raise RuntimeError(f"Task02 validation history is empty: {path}")
    return records


def audit_task02_selection(*, fold_dir: Path, mode: Mode) -> dict[str, Any]:
    """Verify that the saved lightweight selection is exactly reproducible.

    ``hcc_only`` chooses maximum HCC patch Dice. ``replay`` first requires
    LiTS patch Dice not to fall more than 0.02 from the recorded source
    baseline, then chooses maximum HCC patch Dice. Equal scores keep the first
    (earliest) epoch because records are processed chronologically.
    """
    if mode not in {"hcc_only", "replay"}:
        raise ValueError(f"Unknown Task02 mode: {mode}")
    records = _read_jsonl(fold_dir / "task02_validation_history.jsonl")
    baselines = [item for item in records if item.get("kind") == "source_baseline"]
    if len(baselines) != 1:
        raise RuntimeError(f"Expected exactly one source_baseline record; found {len(baselines)}")
    try:
        source_lits = float(baselines[0]["lits_val_mean_tumor_dice"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("Malformed Task02 source baseline record") from exc

    candidates = [item for item in records if item.get("kind") == "lightweight_patch_validation"]
    if not candidates:
        raise RuntimeError("No Task02 lightweight validation records found")
    epochs: set[int] = set()
    valid: list[dict[str, Any]] = []
    for item in candidates:
        try:
            epoch = int(item["epoch"])
            hcc = float(item["hcc_val_mean_tumor_dice"])
            lits = float(item["lits_val_mean_tumor_dice"])
            iterations = int(item["validation_steps_per_domain_global"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("Malformed Task02 lightweight validation record") from exc
        if epoch <= 0 or epoch % 3 != 0 or epoch in epochs:
            raise RuntimeError(f"Invalid or duplicate Task02 validation epoch: {epoch}")
        if iterations != 50:
            raise RuntimeError(f"Task02 validation iterations drift at epoch {epoch}: {iterations}")
        if not 0.0 <= hcc <= 1.0 or not 0.0 <= lits <= 1.0:
            raise RuntimeError(f"Task02 Dice outside [0, 1] at epoch {epoch}")
        epochs.add(epoch)
        if mode == "hcc_only" or lits >= source_lits - RETENTION_MAX_ABSOLUTE_DROP:
            valid.append(item)

    selection_path = fold_dir / "task02_selection.json"
    if not selection_path.is_file():
        raise FileNotFoundError(f"Task02 selection record is missing: {selection_path}")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    target = Path(selection.get("selected_checkpoint", ""))
    if not target.is_file():
        raise RuntimeError(f"Selected Task02 checkpoint is missing: {target}")
    if sha256_file(target) != selection.get("selected_checkpoint_sha256"):
        raise RuntimeError("Selected Task02 checkpoint SHA does not match task02_selection.json")

    if not valid:
        if selection.get("status") != "no_acceptable_adaptation":
            raise RuntimeError("Selection should have fallen back to source but did not")
        return {"status": "valid_source_fallback", "candidate_count": len(candidates)}

    chosen = sorted(
        valid,
        key=lambda item: (-float(item["hcc_val_mean_tumor_dice"]), int(item["epoch"])),
    )[0]
    if selection.get("status") != "selected" or int(selection.get("selected_epoch", -1)) != int(chosen["epoch"]):
        raise RuntimeError("Saved Task02 selection does not match the frozen rule")
    return {
        "status": "valid_selection",
        "selected_epoch": int(chosen["epoch"]),
        "candidate_count": len(candidates),
        "eligible_candidate_count": len(valid),
    }
