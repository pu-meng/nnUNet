"""nnU-Net v2 adapter for the official EfficientMedNeXt-L implementation.

The architecture is loaded from a separate checkout of the authors' repository
instead of copying their research-licensed source into this repository.

Official repository: https://github.com/SLDGroup/EfficientMedNeXt
Pinned commit: 803f7efed9b728ac93ae4e0d8a2602501135241f
License: UT Austin Research License (academic/research use; see official checkout)

Only the network is reused. Data loading, augmentation, optimization, inference,
evaluation, reporting, and visualization remain in the local nnU-Net v2 pipeline.
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path
from typing import Dict

from torch import nn


OFFICIAL_COMMIT = "803f7efed9b728ac93ae4e0d8a2602501135241f"
OFFICIAL_CORE_SHA256: Dict[str, str] = {
    "networks/MedNeXt/mednextv1/EfficientMedNext_Full.py":
        "a7b8348534bddfcac949f56e03af2fa67e91fab45f4b5d388341d2dad456d7c8",
    "networks/MedNeXt/mednextv1/efficient_mednext_blocks.py":
        "3e12e9de5d85b2582c03a7d791902711085411eea18564c9aaf9a032bac85c4d",
    "networks/MedNeXt/mednextv1/create_efficient_mednext.py":
        "79810104a3b69f5bc5cf4c83165226dcad9ced6027b7a2c0d9090f89e29231a0",
}


def _official_checkout_root() -> Path:
    configured = os.environ.get("EFFICIENT_MEDNEXT_ROOT")
    candidates = []
    if configured:
        candidates.append(Path(configured).expanduser())

    # Normal layout used for this project:
    # /home/.../nnUNet and /home/.../EfficientMedNeXt are sibling checkouts.
    candidates.append(Path(__file__).resolve().parents[3] / "EfficientMedNeXt")

    for candidate in candidates:
        if (candidate / "networks/MedNeXt/mednextv1/create_efficient_mednext.py").is_file():
            return candidate.resolve()

    checked = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        "Official EfficientMedNeXt checkout not found. Clone commit "
        f"{OFFICIAL_COMMIT} and set EFFICIENT_MEDNEXT_ROOT. Checked: {checked}"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_official_core(root: Path) -> None:
    """Fail closed if any architecture-defining official source file changed."""
    errors = []
    for relative_path, expected in OFFICIAL_CORE_SHA256.items():
        path = root / relative_path
        if not path.is_file():
            errors.append(f"missing: {path}")
            continue
        actual = _sha256(path)
        if actual != expected:
            errors.append(
                f"hash mismatch: {path}\n  expected {expected}\n  actual   {actual}"
            )
    if errors:
        raise RuntimeError(
            "Official EfficientMedNeXt-L core does not match the pinned source "
            f"commit {OFFICIAL_COMMIT}:\n" + "\n".join(errors)
        )


def build_efficient_mednext_large_official(
    num_input_channels: int,
    num_output_channels: int,
    enable_deep_supervision: bool = True,
) -> nn.Module:
    """Build the paper's EfficientMedNeXt-L with its official configuration.

    Fixed architecture settings from the official training commands/factory:
    base channels 32, uniform decoder channels 32, receptive fields [1, 3, 5],
    residual blocks, and block counts [3, 4, 4, 4, 4, 4, 4, 4, 3].
    """
    root = _official_checkout_root()
    verify_official_core(root)

    root_string = str(root)
    if root_string not in sys.path:
        # The official source uses absolute imports rooted at `networks`.
        sys.path.insert(0, root_string)

    from networks.MedNeXt.mednextv1.create_efficient_mednext import (  # type: ignore
        create_efficient_mednext,
    )

    # Always instantiate all output heads so checkpoints remain loadable when
    # nnU-Net temporarily disables deep supervision for validation/inference.
    network = create_efficient_mednext(
        num_input_channels=num_input_channels,
        num_classes=num_output_channels,
        model_id="L",
        n_channels=32,
        kernel_sizes=[1, 3, 5],
        strides=[1, 1, 1],
        uniform_dec_channels=32,
        deep_supervision=True,
        mode="train",
    )
    network.do_ds = enable_deep_supervision
    network.official_source_commit = OFFICIAL_COMMIT
    network.official_source_root = root_string
    return network

