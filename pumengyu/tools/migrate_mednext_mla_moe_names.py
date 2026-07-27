"""One-time migration for historically misnamed MedNeXt MLA+MoE results.

The old ``nnUNetTrainer_MedNeXt_MLA*`` trainers unintentionally used MoE-FFN.
This utility renames existing result directories to ``*_MLA_MoE*`` and updates
the serialized ``trainer_name`` field in every affected checkpoint. It never
changes ``network_weights`` or optimizer contents.

Dry-run is the default. Use ``--apply`` only after reviewing the printed plan.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import torch


DEFAULT_ROOT = Path("/home/PuMengYu/nnUNet_workspace/results_v2")
OLD_TRAINER_PREFIX = "nnUNetTrainer_MedNeXt_MLA"
NEW_TRAINER_PREFIX = "nnUNetTrainer_MedNeXt_MLA_MoE"
OLD_METHOD_PREFIX = "MedNeXt_MLA"
NEW_METHOD_PREFIX = "MedNeXt_MLA_MoE"
TEXT_METADATA_NAMES = {
    "debug.json",
    "predict_from_raw_data_args.json",
    "report_custom.txt",
    "routing_manifest.json",
    "summary.json",
    "test_report_custom.txt",
    "mixed_validation_summary.json",
}


def corrected_name(name: str) -> str:
    if name.startswith(NEW_TRAINER_PREFIX) or name.startswith(NEW_METHOD_PREFIX):
        return name
    if name.startswith(OLD_TRAINER_PREFIX):
        return NEW_TRAINER_PREFIX + name[len(OLD_TRAINER_PREFIX) :]
    if name.startswith(OLD_METHOD_PREFIX):
        return NEW_METHOD_PREFIX + name[len(OLD_METHOD_PREFIX) :]
    if OLD_METHOD_PREFIX in name and NEW_METHOD_PREFIX not in name:
        return name.replace(OLD_METHOD_PREFIX, NEW_METHOD_PREFIX, 1)
    return name


def corrected_trainer_name(name: Any) -> Any:
    if not isinstance(name, str):
        return name
    if name.startswith(NEW_TRAINER_PREFIX):
        return name
    if name.startswith(OLD_TRAINER_PREFIX):
        return NEW_TRAINER_PREFIX + name[len(OLD_TRAINER_PREFIX) :]
    return name


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def corrected_text(text: str) -> str:
    """Correct stale names/paths without producing repeated ``_MoE_MoE``."""
    text = re.sub(
        r"nnUNetTrainer_MedNeXt_MLA(?!_MoE)",
        NEW_TRAINER_PREFIX,
        text,
    )
    return re.sub(
        r"(?<!nnUNetTrainer_)MedNeXt_MLA(?!_MoE)",
        NEW_METHOD_PREFIX,
        text,
    )


def discover_text_metadata_updates(root: Path) -> list[tuple[Path, str, str]]:
    updates = []
    for path in root.rglob("*"):
        if not path.is_file() or path.name not in TEXT_METADATA_NAMES:
            continue
        if NEW_METHOD_PREFIX not in str(path):
            continue
        old_text = path.read_text(encoding="utf-8")
        new_text = corrected_text(old_text)
        if new_text != old_text:
            updates.append((path, old_text, new_text))
    return sorted(updates, key=lambda item: str(item[0]))


def discover_directory_moves(root: Path) -> list[tuple[Path, Path]]:
    candidates = []
    for path in root.rglob("*"):
        if not path.is_dir():
            continue
        new_name = corrected_name(path.name)
        if new_name != path.name:
            candidates.append((path, path.with_name(new_name)))
    return sorted(candidates, key=lambda pair: len(pair[0].parts), reverse=True)


def validate_moves(moves: list[tuple[Path, Path]]) -> None:
    destinations = set()
    for source, destination in moves:
        if destination in destinations:
            raise RuntimeError(f"duplicate migration destination: {destination}")
        destinations.add(destination)
        if destination.exists() and destination != source:
            raise FileExistsError(f"migration destination already exists: {destination}")


def rewrite_checkpoint(path: Path) -> dict[str, Any] | None:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        return None
    old_name = checkpoint.get("trainer_name")
    new_name = corrected_trainer_name(old_name)
    if new_name == old_name:
        return None

    before_hash = sha256(path)
    weights = checkpoint.get("network_weights", {})
    weight_keys = len(weights) if isinstance(weights, dict) else None
    weight_numel = (
        sum(value.numel() for value in weights.values() if hasattr(value, "numel"))
        if isinstance(weights, dict)
        else None
    )
    checkpoint["trainer_name"] = new_name

    temporary = path.with_name(path.name + ".trainer_rename_tmp")
    torch.save(checkpoint, temporary)
    os.replace(temporary, path)
    after_hash = sha256(path)
    return {
        "path": str(path),
        "old_trainer_name": old_name,
        "new_trainer_name": new_name,
        "sha256_before": before_hash,
        "sha256_after": after_hash,
        "network_weight_keys": weight_keys,
        "network_weight_numel": weight_numel,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)

    moves = discover_directory_moves(root)
    validate_moves(moves)
    print(f"root={root}")
    print(f"directory_moves={len(moves)}")
    for source, destination in moves:
        print(f"  {source} -> {destination}")

    affected_checkpoints = []
    checkpoint_candidates = [
        path
        for path in root.rglob("*.pth")
        if OLD_TRAINER_PREFIX in str(path) and NEW_TRAINER_PREFIX not in str(path)
    ]
    for checkpoint_path in checkpoint_candidates:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        trainer_name = checkpoint.get("trainer_name") if isinstance(checkpoint, dict) else None
        new_name = corrected_trainer_name(trainer_name)
        if new_name != trainer_name:
            affected_checkpoints.append((checkpoint_path, trainer_name, new_name))
    print(f"checkpoint_metadata_updates={len(affected_checkpoints)}")
    for path, old_name, new_name in affected_checkpoints:
        print(f"  {path}: {old_name} -> {new_name}")

    text_updates = discover_text_metadata_updates(root)
    print(f"text_metadata_updates={len(text_updates)}")
    for path, _, _ in text_updates:
        print(f"  {path}")

    if not args.apply:
        print("dry-run only; pass --apply to perform the migration")
        return

    directory_records = []
    for source, destination in moves:
        source.rename(destination)
        directory_records.append({"source": str(source), "destination": str(destination)})

    checkpoint_records = []
    migrated_checkpoint_candidates = [
        path for path in root.rglob("*.pth") if NEW_TRAINER_PREFIX in str(path)
    ]
    for checkpoint_path in sorted(migrated_checkpoint_candidates):
        record = rewrite_checkpoint(checkpoint_path)
        if record is not None:
            checkpoint_records.append(record)

    text_records = []
    # Reports and JSON metadata must point at the renamed result paths. Original
    # training logs are deliberately excluded so they remain immutable evidence.
    for path, old_text, new_text in discover_text_metadata_updates(root):
        before_hash = hashlib.sha256(old_text.encode("utf-8")).hexdigest()
        path.write_text(new_text, encoding="utf-8")
        text_records.append(
            {
                "path": str(path),
                "sha256_before": before_hash,
                "sha256_after": sha256(path),
            }
        )

    manifest_dir = root / "_name_migrations"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%z")
    manifest_path = manifest_dir / f"mednext_mla_to_mla_moe_{timestamp}.json"
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "root": str(root),
        "reason": "Historical MedNeXt_MLA checkpoints used MoE-FFN; names corrected to MedNeXt_MLA_MoE.",
        "directory_moves": directory_records,
        "checkpoint_updates": checkpoint_records,
        "text_metadata_updates": text_records,
        "provenance_note": (
            "Original training logs were not modified. Checkpoint mtimes may reflect "
            "the metadata migration; sha256_before records the original serialized file."
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"migration complete: {manifest_path}")


if __name__ == "__main__":
    main()
