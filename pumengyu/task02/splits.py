"""Immutable split loading and leakage checks for Task 02.

This module has no GPU dependency. Trainers call :func:`load_task02_split`
before constructing any data loader, so an incorrect case manifest fails before
an optimizer or augmentation worker can be created.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = Path("/home/PuMengYu/nnUNet_workspace")

HCC_DATASET = "Dataset013_HCCReferencedCT"
LITS_DATASET = "Dataset003_Liver"
HCC_SPLIT_INFO = (
    WORKSPACE_ROOT / "preprocessed" / HCC_DATASET / "split_info_701020_stratified_v2.json"
)
LITS_SPLIT_INFO = WORKSPACE_ROOT / "preprocessed" / LITS_DATASET / "split_info_712.json"
HCC_CASE_METADATA = REPO_ROOT / "pumengyu/notes/data/hcc_split_701020_stratified_v2_cases.csv"
TASK02_SPLIT_DIR = REPO_ROOT / "pumengyu/notes/data/task02_fewshot_splits"

EXPECTED_HCC_METADATA_SHA256 = "17aa097e3c3c21a783b46abf643a246074094ed061d0d8a8ce00a2c5c06d65cc"
EXPECTED_HCC_COUNTS = {"train": 70, "val": 10, "test": 21}
EXPECTED_LITS_COUNTS = {"train": 92, "val": 13, "test": 26}
EXPECTED_K = (1, 3, 5, 10)


def sha256_file(path: Path) -> str:
    """Return a SHA256 digest without loading a potentially large file at once."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_partition_file(path: Path, expected: dict[str, int], label: str) -> dict[str, tuple[str, ...]]:
    if not path.is_file():
        raise FileNotFoundError(f"Task02 {label} split file is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    partitions: dict[str, tuple[str, ...]] = {}
    for name, n_expected in expected.items():
        try:
            cases = tuple(payload[name]["cases"])
        except (KeyError, TypeError) as exc:
            raise RuntimeError(f"Malformed {label} split: missing {name}.cases in {path}") from exc
        if len(cases) != n_expected or len(set(cases)) != n_expected:
            raise RuntimeError(
                f"Task02 {label} {name} split must contain {n_expected} unique cases; "
                f"observed {len(cases)} total / {len(set(cases))} unique."
            )
        partitions[name] = cases
    all_cases = [case for cases in partitions.values() for case in cases]
    if len(all_cases) != len(set(all_cases)):
        raise RuntimeError(f"Task02 {label} split partitions overlap: {path}")
    return partitions


def load_hcc_partitions() -> dict[str, tuple[str, ...]]:
    """Load and verify the canonical HCC 70/10/21 partition."""
    return _read_partition_file(HCC_SPLIT_INFO, EXPECTED_HCC_COUNTS, "HCC")


def load_lits_partitions() -> dict[str, tuple[str, ...]]:
    """Load and verify the canonical LiTS 92/13/26 partition."""
    return _read_partition_file(LITS_SPLIT_INFO, EXPECTED_LITS_COUNTS, "LiTS")


def _metadata_by_case() -> dict[str, dict[str, str]]:
    if not HCC_CASE_METADATA.is_file():
        raise FileNotFoundError(f"Task02 HCC metadata CSV is missing: {HCC_CASE_METADATA}")
    observed_hash = sha256_file(HCC_CASE_METADATA)
    if observed_hash != EXPECTED_HCC_METADATA_SHA256:
        raise RuntimeError(
            "Task02 HCC metadata CSV SHA256 differs from the frozen protocol; "
            f"expected {EXPECTED_HCC_METADATA_SHA256}, observed {observed_hash}."
        )
    with HCC_CASE_METADATA.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    indexed = {row["case_id"]: row for row in rows}
    if len(indexed) != len(rows):
        raise RuntimeError(f"Duplicate HCC case IDs in metadata CSV: {HCC_CASE_METADATA}")
    return indexed


@dataclass(frozen=True)
class Task02Split:
    """Validated immutable k-shot manifest used by one Stage-2 run."""

    split_id: str
    k: int
    sampling_seed: int
    training_seed: int
    case_ids: tuple[str, ...]
    quota_by_bin: dict[str, int]
    manifest_path: Path
    manifest_sha256: str
    source_metadata_sha256: str


def _manifest_path(k: int) -> Path:
    if k not in EXPECTED_K:
        raise ValueError(f"Task02 only has frozen few-shot manifests for k={EXPECTED_K}; got k={k}")
    matches = sorted(TASK02_SPLIT_DIR.glob(f"task02_hcc_k{k:02d}_seed*.json"))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one frozen Task02 manifest for k={k} under {TASK02_SPLIT_DIR}; found {matches}"
        )
    return matches[0]


def load_task02_split(k: int) -> Task02Split:
    """Load one frozen k-shot manifest and reject leakage or metadata drift."""
    manifest_path = _manifest_path(k)
    payload: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    try:
        manifest_k = int(payload["k"])
        cases = list(payload["cases"])
        quota_by_bin = {str(name): int(value) for name, value in payload["quota_by_bin"].items()}
        metadata_sha = str(payload["candidate_pool"]["case_metadata_sha256"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"Malformed Task02 manifest: {manifest_path}") from exc
    if manifest_k != k or len(cases) != k:
        raise RuntimeError(f"Task02 manifest k/count mismatch in {manifest_path}: k={manifest_k}, cases={len(cases)}")
    if metadata_sha != EXPECTED_HCC_METADATA_SHA256:
        raise RuntimeError(f"Task02 manifest references an unexpected metadata SHA: {manifest_path}")

    hcc = load_hcc_partitions()
    metadata = _metadata_by_case()
    case_ids = tuple(str(item["case_id"]) for item in cases)
    if len(case_ids) != len(set(case_ids)):
        raise RuntimeError(f"Duplicate case IDs in Task02 manifest: {manifest_path}")
    forbidden = set(hcc["val"]) | set(hcc["test"])
    if set(case_ids) & forbidden:
        raise RuntimeError(f"HCC val/test leakage in Task02 manifest {manifest_path}: {sorted(set(case_ids) & forbidden)}")
    missing_from_train = sorted(set(case_ids) - set(hcc["train"]))
    if missing_from_train:
        raise RuntimeError(f"Task02 cases are outside the HCC 70-case pool: {missing_from_train}")

    observed_bins: Counter[str] = Counter()
    for item in cases:
        case_id = str(item["case_id"])
        if case_id not in metadata:
            raise RuntimeError(f"Task02 manifest case absent from HCC metadata: {case_id}")
        if metadata[case_id]["split"] != "train":
            raise RuntimeError(f"Task02 manifest case is not HCC train: {case_id}")
        manifest_bin = str(item["tumor_burden_bin"])
        if metadata[case_id]["bin"] != manifest_bin:
            raise RuntimeError(
                f"Task02 tumor-burden bin drift for {case_id}: "
                f"manifest={manifest_bin}, metadata={metadata[case_id]['bin']}"
            )
        observed_bins[manifest_bin] += 1
    if observed_bins != Counter(quota_by_bin):
        raise RuntimeError(
            f"Task02 manifest quota mismatch in {manifest_path}: quota={quota_by_bin}, observed={dict(observed_bins)}"
        )

    return Task02Split(
        split_id=str(payload["split_id"]),
        k=k,
        sampling_seed=int(payload["sampling_seed"]),
        training_seed=int(payload["training_seed"]),
        case_ids=case_ids,
        quota_by_bin=quota_by_bin,
        manifest_path=manifest_path,
        manifest_sha256=sha256_file(manifest_path),
        source_metadata_sha256=metadata_sha,
    )


def load_k70_hcc_cases() -> tuple[str, ...]:
    """Return all and only the canonical 70 HCC adaptation-pool cases."""
    return load_hcc_partitions()["train"]


def audit_task02_inputs() -> dict[str, Any]:
    """Return an auditable, JSON-serializable summary before a Stage-2 run."""
    hcc = load_hcc_partitions()
    lits = load_lits_partitions()
    manifests = [load_task02_split(k) for k in EXPECTED_K]
    return {
        "hcc_split_info": str(HCC_SPLIT_INFO),
        "hcc_counts": {name: len(cases) for name, cases in hcc.items()},
        "lits_split_info": str(LITS_SPLIT_INFO),
        "lits_counts": {name: len(cases) for name, cases in lits.items()},
        "fewshot_manifests": [
            {
                "k": item.k,
                "split_id": item.split_id,
                "case_ids": list(item.case_ids),
                "manifest_path": str(item.manifest_path),
                "manifest_sha256": item.manifest_sha256,
            }
            for item in manifests
        ],
    }
