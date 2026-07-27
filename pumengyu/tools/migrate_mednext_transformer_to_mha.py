"""Rename the completed MedNeXt standard-MHA experiment consistently.

Canonical mapping:
    nnUNetTrainer_MedNeXt_Transformer -> nnUNetTrainer_MedNeXt_MHA
    MedNeXt_Transformer               -> MedNeXt_MHA

The old trainer class remains as a compatibility alias in ``trainer.py``.
This migration renames derived result directories, updates checkpoint
``trainer_name`` metadata, and rewrites derived report/JSON metadata. Original
training logs and earlier migration manifests remain untouched as provenance.

Dry-run is the default. Pass ``--apply`` only after reviewing the plan.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import torch


DEFAULT_ROOT = Path("/home/PuMengYu/nnUNet_workspace/results_v2")
OLD_TRAINER = "nnUNetTrainer_MedNeXt_Transformer"
NEW_TRAINER = "nnUNetTrainer_MedNeXt_MHA"
OLD_METHOD = "MedNeXt_Transformer"
NEW_METHOD = "MedNeXt_MHA"
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
    return name.replace(OLD_TRAINER, NEW_TRAINER).replace(OLD_METHOD, NEW_METHOD)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def network_sha256(weights: Any) -> str | None:
    if not isinstance(weights, dict):
        return None
    digest = hashlib.sha256()
    for name, value in sorted(weights.items()):
        digest.update(name.encode("utf-8"))
        if not torch.is_tensor(value):
            digest.update(repr(value).encode("utf-8"))
            continue
        tensor = value.detach().cpu().contiguous()
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def discover_directory_moves(root: Path) -> list[tuple[Path, Path]]:
    moves = []
    for path in root.rglob("*"):
        if not path.is_dir():
            continue
        new_name = corrected_name(path.name)
        if new_name != path.name:
            moves.append((path, path.with_name(new_name)))
    return sorted(moves, key=lambda pair: len(pair[0].parts), reverse=True)


def validate_moves(moves: list[tuple[Path, Path]]) -> None:
    destinations: set[Path] = set()
    for source, destination in moves:
        if destination in destinations:
            raise RuntimeError(f"duplicate destination: {destination}")
        destinations.add(destination)
        if destination.exists():
            raise FileExistsError(f"destination already exists: {destination}")


def discover_checkpoint_updates(root: Path) -> list[tuple[Path, str]]:
    updates = []
    for path in sorted(root.rglob("*.pth")):
        if OLD_TRAINER not in str(path):
            continue
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        trainer_name = checkpoint.get("trainer_name") if isinstance(checkpoint, dict) else None
        if trainer_name == OLD_TRAINER:
            updates.append((path, trainer_name))
        del checkpoint
        gc.collect()
    return updates


def discover_text_updates(root: Path) -> list[Path]:
    updates = []
    for path in root.rglob("*"):
        if not path.is_file() or path.name not in TEXT_METADATA_NAMES:
            continue
        if "_name_migrations" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        if OLD_TRAINER in text or OLD_METHOD in text:
            updates.append(path)
    return sorted(updates)


def rewrite_checkpoint(path: Path) -> dict[str, Any] | None:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or checkpoint.get("trainer_name") != OLD_TRAINER:
        return None

    old_mode = path.stat().st_mode
    before_file_hash = sha256(path)
    before_network_hash = network_sha256(checkpoint.get("network_weights"))
    epoch = checkpoint.get("current_epoch")
    checkpoint["trainer_name"] = NEW_TRAINER

    temporary = path.with_name(path.name + ".trainer_rename_tmp")
    torch.save(checkpoint, temporary)
    del checkpoint
    gc.collect()

    verification = torch.load(temporary, map_location="cpu", weights_only=False)
    after_network_hash = network_sha256(verification.get("network_weights"))
    new_trainer_name = verification.get("trainer_name")
    del verification
    gc.collect()
    if before_network_hash != after_network_hash or new_trainer_name != NEW_TRAINER:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"checkpoint verification failed: {path}")

    os.replace(temporary, path)
    os.chmod(path, old_mode)
    return {
        "path": str(path),
        "epoch": epoch,
        "old_trainer_name": OLD_TRAINER,
        "new_trainer_name": NEW_TRAINER,
        "sha256_before": before_file_hash,
        "sha256_after": sha256(path),
        "network_sha256": before_network_hash,
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
    checkpoints = discover_checkpoint_updates(root)
    text_updates = discover_text_updates(root)

    print(f"root={root}")
    print(f"directory_moves={len(moves)}")
    for source, destination in moves:
        print(f"  {source} -> {destination}")
    print(f"checkpoint_metadata_updates={len(checkpoints)}")
    for path, old_name in checkpoints:
        print(f"  {path}: {old_name} -> {NEW_TRAINER}")
    print(f"text_metadata_updates={len(text_updates)}")
    for path in text_updates:
        print(f"  {path}")

    if not args.apply:
        print("dry-run only; pass --apply to perform the migration")
        return

    directory_records = []
    for source, destination in moves:
        source.rename(destination)
        directory_records.append({"source": str(source), "destination": str(destination)})

    checkpoint_records = []
    for path in sorted(root.rglob("*.pth")):
        if NEW_TRAINER not in str(path):
            continue
        record = rewrite_checkpoint(path)
        if record is not None:
            checkpoint_records.append(record)

    text_records = []
    for path in discover_text_updates(root):
        old_text = path.read_text(encoding="utf-8")
        new_text = old_text.replace(OLD_TRAINER, NEW_TRAINER).replace(OLD_METHOD, NEW_METHOD)
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
    manifest_path = manifest_dir / f"mednext_transformer_to_mha_{timestamp}.json"
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "root": str(root),
        "reason": "Canonical attention naming: standard MHA + MLP is MedNeXt_MHA.",
        "directory_moves": directory_records,
        "checkpoint_updates": checkpoint_records,
        "text_metadata_updates": text_records,
        "provenance_note": (
            "Original training logs and older migration manifests were not modified. "
            "network_sha256 verifies unchanged model weights."
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"migration complete: {manifest_path}")


if __name__ == "__main__":
    main()
