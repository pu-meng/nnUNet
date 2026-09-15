"""Core training mechanics for Task 02 Stage-2 adaptation.

The mixin intentionally keeps HCC and LiTS in separate augmented loaders.
Using one concatenated dataset would make the 1:1 replay loss only an average
sampling preference, not the frozen per-update objective.

No training is launched by importing this module.
"""

from __future__ import annotations

import hashlib
import json
import random
import shutil
from time import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from torch import autocast
from torch import distributed as dist
from torch._dynamo import OptimizedModule
from pumengyu.task02.routing import ReplayExpertCounts

from nnunetv2.paths import nnUNet_preprocessed
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from nnunetv2.utilities.helpers import dummy_context

from pumengyu.task02.splits import (
    HCC_DATASET,
    HCC_SPLIT_INFO,
    LITS_DATASET,
    Task02Split,
    audit_task02_inputs,
    load_hcc_partitions,
    load_k70_hcc_cases,
    load_lits_partitions,
    load_task02_split,
    sha256_file,
)


FROZEN_SOURCE_CHECKPOINT = Path(
    "/home/PuMengYu/nnUNet_workspace/results_v2_baseline_retrain_20260902/"
    "Dataset003_Liver/nnUNetTrainer_MedNeXt_MHA_MoE__nnUNetPlans__3d_fullres/"
    "fold_0/checkpoint_best.pth"
)
FROZEN_SOURCE_SHA256 = "3910ea7375656b84f14055bcd1cc2310ffd85e343d9dc0a33577aaf60eb13b81"
FROZEN_SOURCE_EPOCH = 951


class Task02Stage2Mixin:
    """Shared behavior for one auditable Task-2 Stage-2 trainer.

    Subclasses must only set ``TASK02_MODE`` and ``TASK02_K``. The active
    nnU-Net dataset remains Dataset003_Liver, so source plans, labels and model
    architecture stay compatible with the frozen MHA+MoE checkpoint.
    """

    TASK02_MODE: str = "replay"  # ``hcc_only`` or ``replay``
    # Task02 uses explicit selected-checkpoint evaluation, including after DDP.
    AUTO_VALIDATE_AFTER_TRAINING = False
    TASK02_K: int = 1             # 1/3/5/10, or 70 for the upper bound
    TASK02_TRAINING_SEED: int = 20260920
    TASK02_NUM_EPOCHS: int = 300
    TASK02_UPDATES_PER_EPOCH: int = 250
    TASK02_INITIAL_LR: float = 1e-3
    TASK02_REPLAY_LAMBDA: float = 1.0
    TASK02_VALIDATION_INTERVAL: int = 3
    TASK02_VALIDATION_ITERATIONS: int = 50
    TASK02_VALIDATION_SEED: int = 20260921
    TASK02_EXPECTED_SOURCE_TRAINER: str = "nnUNetTrainer_MedNeXt_MHA_MoE"
    TASK02_RUN_KIND: str = "formal"

    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)  # type: ignore[misc]
        if self.plans_manager.dataset_name != LITS_DATASET:  # type: ignore[attr-defined]
            raise RuntimeError(
                f"Task02 Stage-2 must start with -d {LITS_DATASET}; got {self.plans_manager.dataset_name}."  # type: ignore[attr-defined]
            )
        if self.configuration_name != "3d_fullres":  # type: ignore[attr-defined]
            raise RuntimeError("Task02 Stage-2 is frozen to the Dataset003 3d_fullres plan.")
        if int(fold) != 0:
            raise RuntimeError("Task02 Stage-2 is frozen to fold 0.")
        if self.TASK02_MODE not in {"hcc_only", "replay"}:
            raise ValueError(f"Unsupported Task02 mode: {self.TASK02_MODE}")
        if self.TASK02_K not in {1, 3, 5, 10, 70}:
            raise ValueError(f"Unsupported Task02 k: {self.TASK02_K}")
        if self.TASK02_RUN_KIND not in {"formal", "smoke"}:
            raise ValueError(f"Unsupported Task02 run kind: {self.TASK02_RUN_KIND}")

        self.num_epochs = self.TASK02_NUM_EPOCHS
        self.num_iterations_per_epoch = self.TASK02_UPDATES_PER_EPOCH
        self.initial_lr = self.TASK02_INITIAL_LR
        # Base nnU-Net's checkpoint_best is selected from patch pseudo-Dice.
        # Task02 selection is instead performed by its explicit dual-domain
        # lightweight patch validation below.
        self._best_ema = float("inf")
        self._task02_hcc_loader = None
        self._task02_hcc_val_loader = None
        self._task02_lits_val_loader = None
        self._task02_split: Task02Split | None = None
        self._task02_source_identity: dict[str, Any] | None = None
        self._task02_source_lits_val_dice: float | None = None
        self._task02_best_hcc_dice: float | None = None

    # ------------------------------------------------------------------
    # Source initialization and reproducibility manifest
    # ------------------------------------------------------------------
    def _task02_unwrapped_network(self):
        network = self.network  # type: ignore[attr-defined]
        if self.is_ddp:  # type: ignore[attr-defined]
            network = network.module
        if isinstance(network, OptimizedModule):
            network = network._orig_mod
        return network

    def _task02_load_source_weights_strict(self) -> dict[str, Any]:
        """Load every matching parameter, including segmentation heads, by SHA."""
        if not FROZEN_SOURCE_CHECKPOINT.is_file():
            raise FileNotFoundError(f"Task02 frozen source checkpoint is missing: {FROZEN_SOURCE_CHECKPOINT}")
        observed_sha = sha256_file(FROZEN_SOURCE_CHECKPOINT)
        if observed_sha != FROZEN_SOURCE_SHA256:
            raise RuntimeError(
                "Task02 source checkpoint SHA256 differs from the frozen protocol; "
                f"expected {FROZEN_SOURCE_SHA256}, observed {observed_sha}."
            )
        payload = torch.load(FROZEN_SOURCE_CHECKPOINT, map_location="cpu", weights_only=False)
        if payload.get("trainer_name") != self.TASK02_EXPECTED_SOURCE_TRAINER:
            raise RuntimeError(
                "Task02 source trainer mismatch: "
                f"expected {self.TASK02_EXPECTED_SOURCE_TRAINER}, got {payload.get('trainer_name')}"
            )
        if int(payload.get("current_epoch", -1)) != FROZEN_SOURCE_EPOCH:
            raise RuntimeError(
                f"Task02 source epoch mismatch: expected {FROZEN_SOURCE_EPOCH}, "
                f"got {payload.get('current_epoch')}"
            )
        network = self._task02_unwrapped_network()
        source_weights = payload.get("network_weights")
        if not isinstance(source_weights, dict):
            raise RuntimeError("Frozen source checkpoint has no network_weights state dict.")
        missing, unexpected = network.load_state_dict(source_weights, strict=True)
        if missing or unexpected:
            raise RuntimeError(f"Strict source load failed: missing={missing}, unexpected={unexpected}")
        return {
            "path": str(FROZEN_SOURCE_CHECKPOINT),
            "sha256": observed_sha,
            "trainer": payload["trainer_name"],
            "epoch": int(payload["current_epoch"]),
        }

    def _task02_set_seed(self) -> None:
        seed = int(self.TASK02_TRAINING_SEED) + int(self.local_rank)  # type: ignore[attr-defined]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _task02_active_hcc_cases(self) -> tuple[str, ...]:
        if self.TASK02_K == 70:
            return load_k70_hcc_cases()
        self._task02_split = load_task02_split(self.TASK02_K)
        if self._task02_split.training_seed != self.TASK02_TRAINING_SEED:
            raise RuntimeError(
                f"Task02 split training seed drift: manifest={self._task02_split.training_seed}, "
                f"trainer={self.TASK02_TRAINING_SEED}"
            )
        return self._task02_split.case_ids

    def _task02_write_manifest(self, hcc_cases: tuple[str, ...]) -> None:
        if self.local_rank != 0:  # type: ignore[attr-defined]
            return
        output = Path(self.output_folder)  # type: ignore[attr-defined]
        output.mkdir(parents=True, exist_ok=True)
        if self._task02_source_identity is None:
            raise RuntimeError("Task02 source identity was not initialized.")
        payload = {
            "schema_version": 1,
            "task": "Task02 Stage-2 shared-model adaptation",
            "run_kind": self.TASK02_RUN_KIND,
            "mode": self.TASK02_MODE,
            "k": self.TASK02_K,
            "hcc_train_case_ids": list(hcc_cases),
            "hcc_split_info": str(HCC_SPLIT_INFO),
            "source": self._task02_source_identity,
            "training": {
                "seed": self.TASK02_TRAINING_SEED,
                "epochs": self.TASK02_NUM_EPOCHS,
                "updates_per_epoch": self.TASK02_UPDATES_PER_EPOCH,
                "initial_lr": self.TASK02_INITIAL_LR,
                "patch_size": [int(value) for value in self.configuration_manager.patch_size],  # type: ignore[attr-defined]
                "batch_size_per_rank": int(self.batch_size),  # type: ignore[attr-defined]
                "ddp_world_size": dist.get_world_size() if self.is_ddp else 1,  # type: ignore[attr-defined]
                "optimizer": "SGD(momentum=0.99, nesterov=True, weight_decay=3e-5)",
                "scheduler": "PolyLR, restarted from epoch 0 for Stage-2",
                "augmentation": (
                    "source trainer get_training_transforms; HCC and LiTS use the same "
                    "3d_fullres patch/deep-supervision configuration"
                ),
                "replay_lambda": self.TASK02_REPLAY_LAMBDA if self.TASK02_MODE == "replay" else None,
                "optimizer_update": "one HCC batch" if self.TASK02_MODE == "hcc_only"
                else "one HCC batch plus one LiTS replay batch",
            },
            "checkpoint_selection": {
                "validation_interval_epochs": self.TASK02_VALIDATION_INTERVAL,
                "validation_steps_per_domain_global": self.TASK02_VALIDATION_ITERATIONS,
                "validation_seed": self.TASK02_VALIDATION_SEED,
                "metric": "patch-level tumor Dice; not final full-volume evaluation",
                "hcc_only_rule": "max HCC validation Dice; earliest epoch breaks ties",
                "replay_rule": "LiTS validation Dice >= source baseline - 0.02, then max HCC validation Dice; earliest epoch breaks ties",
                "fallback": "frozen source checkpoint if no replay candidate is eligible",
            },
            "split_audit": audit_task02_inputs(),
        }
        (output / "task02_manifest.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        config = {
            "schema_version": 1,
            "task": payload["task"],
            "run_kind": payload["run_kind"],
            "mode": payload["mode"],
            "k": payload["k"],
            "source": payload["source"],
            "training": payload["training"],
            "checkpoint_selection": payload["checkpoint_selection"],
        }
        (output / "task02_config.json").write_text(
            json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        if self._task02_split is not None:
            target = output / self._task02_split.manifest_path.name
            if target.exists() and sha256_file(target) != self._task02_split.manifest_sha256:
                raise RuntimeError(f"Refusing to overwrite divergent Task02 split copy: {target}")
            if not target.exists():
                shutil.copy2(self._task02_split.manifest_path, target)

    # ------------------------------------------------------------------
    # Separate HCC augmented loader
    # ------------------------------------------------------------------
    def _task02_hcc_dataset(self, hcc_cases: tuple[str, ...]):
        if int(self.num_input_channels) != 1:  # type: ignore[attr-defined]
            raise RuntimeError("Task02 HCCReferencedCT replay expects a one-channel Dataset003 source model.")
        hcc_folder = Path(nnUNet_preprocessed) / HCC_DATASET / self.configuration_manager.data_identifier  # type: ignore[attr-defined]
        if not hcc_folder.is_dir():
            raise FileNotFoundError(f"Task02 HCC preprocessed folder is missing: {hcc_folder}")
        dataset_class = infer_dataset_class(str(hcc_folder))
        available = set(dataset_class.get_identifiers(str(hcc_folder)))
        missing = sorted(set(hcc_cases) - available)
        if missing:
            raise RuntimeError(f"Task02 HCC preprocessed cases missing: {missing}")
        return dataset_class(str(hcc_folder), identifiers=list(hcc_cases))

    def _task02_make_training_augmenter(self, dataset):
        patch_size = self.configuration_manager.patch_size  # type: ignore[attr-defined]
        deep_supervision_scales = self._get_deep_supervision_scales()  # type: ignore[attr-defined]
        rotation, dummy_2d, initial_patch_size, mirror_axes = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()  # type: ignore[attr-defined]
        transforms = self.get_training_transforms(  # type: ignore[attr-defined]
            patch_size, rotation, deep_supervision_scales, mirror_axes, dummy_2d,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,  # type: ignore[attr-defined]
            is_cascaded=self.is_cascaded,  # type: ignore[attr-defined]
            foreground_labels=self.label_manager.foreground_labels,  # type: ignore[attr-defined]
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,  # type: ignore[attr-defined]
            ignore_label=self.label_manager.ignore_label,  # type: ignore[attr-defined]
        )
        loader = nnUNetDataLoader(
            dataset, self.batch_size, initial_patch_size, patch_size, self.label_manager,  # type: ignore[attr-defined]
            oversample_foreground_percent=self.oversample_foreground_percent,  # type: ignore[attr-defined]
            sampling_probabilities=None, pad_sides=None, transforms=transforms,
            probabilistic_oversampling=self.probabilistic_oversampling,  # type: ignore[attr-defined]
        )
        workers = get_allowed_n_proc_DA()
        if workers == 0:
            return SingleThreadedAugmenter(loader, None)
        return NonDetMultiThreadedAugmenter(
            data_loader=loader, transform=None, num_processes=workers,
            num_cached=max(6, workers // 2), seeds=None,
            pin_memory=self.device.type == "cuda", wait_time=0.002,  # type: ignore[attr-defined]
        )

    def _task02_make_validation_augmenter(self, dataset):
        """Build the same patch-level validation pipeline as nnU-Net, for HCC only."""
        patch_size = self.configuration_manager.patch_size  # type: ignore[attr-defined]
        deep_supervision_scales = self._get_deep_supervision_scales()  # type: ignore[attr-defined]
        transforms = self.get_validation_transforms(  # type: ignore[attr-defined]
            deep_supervision_scales,
            is_cascaded=self.is_cascaded,  # type: ignore[attr-defined]
            foreground_labels=self.label_manager.foreground_labels,  # type: ignore[attr-defined]
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,  # type: ignore[attr-defined]
            ignore_label=self.label_manager.ignore_label,  # type: ignore[attr-defined]
        )
        loader = nnUNetDataLoader(
            dataset, self.batch_size, patch_size, patch_size, self.label_manager,  # type: ignore[attr-defined]
            oversample_foreground_percent=self.oversample_foreground_percent,  # type: ignore[attr-defined]
            sampling_probabilities=None, pad_sides=None, transforms=transforms,
            probabilistic_oversampling=self.probabilistic_oversampling,  # type: ignore[attr-defined]
        )
        # Checkpoint ranking is only meaningful if every candidate sees the
        # same sampled validation patches. A single-threaded augmenter lets us
        # reset the sampling RNG immediately before each lightweight pass;
        # worker prefetching would otherwise make the next 50 batches depend on
        # timing and on the preceding epoch.
        return SingleThreadedAugmenter(loader, None)

    # ------------------------------------------------------------------
    # One optimizer update
    # ------------------------------------------------------------------
    def _task02_to_device(self, batch: dict):
        data = batch["data"].to(self.device, non_blocking=True)  # type: ignore[attr-defined]
        target = batch["target"]
        if isinstance(target, list):
            target = [item.to(self.device, non_blocking=True) for item in target]  # type: ignore[attr-defined]
        else:
            target = target.to(self.device, non_blocking=True)  # type: ignore[attr-defined]
        return data, target

    def _task02_replay_step(self, hcc_batch: dict, lits_batch: dict) -> dict[str, float]:
        hcc_data, hcc_target = self._task02_to_device(hcc_batch)
        lits_data, lits_target = self._task02_to_device(lits_batch)
        self.optimizer.zero_grad(set_to_none=True)  # type: ignore[attr-defined]
        expert_counts = ReplayExpertCounts(self.network)
        # Finish each domain's backward before starting the next forward.
        # Two live reentrant-checkpoint graphs for the same DDP parameters
        # otherwise fire the reducer's ready hook twice in one backward.
        # no_sync must include BOTH the HCC forward and backward. LiTS then
        # synchronizes the accumulated HCC + lambda * LiTS gradients once.
        sync_context = self.network.no_sync() if isinstance(self.network, torch.nn.parallel.DistributedDataParallel) else dummy_context()
        with sync_context:
            with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
                hcc_loss = self.loss(self.network(hcc_data), hcc_target)
                expert_counts.capture()
            if self.grad_scaler is not None:
                self.grad_scaler.scale(hcc_loss).backward()
            else:
                hcc_loss.backward()
        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            lits_loss = self.loss(self.network(lits_data), lits_target)  # type: ignore[attr-defined]
            expert_counts.capture()
            replay_loss = self.TASK02_REPLAY_LAMBDA * lits_loss
        loss = hcc_loss.detach() + replay_loss.detach()
        if self.grad_scaler is not None:  # type: ignore[attr-defined]
            self.grad_scaler.scale(replay_loss).backward()  # type: ignore[attr-defined]
            self.grad_scaler.unscale_(self.optimizer)  # type: ignore[attr-defined]
            # Backward may overwrite pending counts during checkpoint recomputation.
            expert_counts.restore()
            self._commit_deferred_network_updates()  # type: ignore[attr-defined]
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)  # type: ignore[attr-defined]
            self.grad_scaler.step(self.optimizer)  # type: ignore[attr-defined]
            self.grad_scaler.update()  # type: ignore[attr-defined]
        else:
            replay_loss.backward()
            expert_counts.restore()
            self._commit_deferred_network_updates()  # type: ignore[attr-defined]
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)  # type: ignore[attr-defined]
            self.optimizer.step()  # type: ignore[attr-defined]
        return {
            "loss": float(loss.detach().cpu().item()),
            "hcc_loss": float(hcc_loss.detach().cpu().item()),
            "lits_replay_loss": float(lits_loss.detach().cpu().item()),
        }

    # ------------------------------------------------------------------
    # Hooks used by the concrete trainer classes
    # ------------------------------------------------------------------
    def on_train_start(self):
        # This runs after any accidental CLI ``-pretrained_weights`` load and
        # deliberately overwrites it with the frozen, SHA-checked source state.
        self._task02_set_seed()
        super().on_train_start()  # type: ignore[misc]
        self._task02_source_identity = self._task02_load_source_weights_strict()
        self.optimizer, self.lr_scheduler = self.configure_optimizers()  # type: ignore[attr-defined]
        hcc_cases = self._task02_active_hcc_cases()
        self._task02_hcc_loader = self._task02_make_training_augmenter(self._task02_hcc_dataset(hcc_cases))
        _ = next(self._task02_hcc_loader)
        hcc_partitions = load_hcc_partitions()
        self._task02_hcc_val_loader = self._task02_make_validation_augmenter(
            self._task02_hcc_dataset(hcc_partitions["val"])
        )
        active_lits_train, active_lits_val = self.get_tr_and_val_datasets()  # type: ignore[attr-defined]
        canonical_lits = load_lits_partitions()
        if set(active_lits_train.identifiers) != set(canonical_lits["train"]):
            raise RuntimeError("Active Dataset003 train split differs from Task02 frozen LiTS train 92 cases.")
        if set(active_lits_val.identifiers) != set(canonical_lits["val"]):
            raise RuntimeError("Active Dataset003 val split differs from Task02 frozen LiTS val 13 cases.")
        # The base LiTS validation augmenter is prefetching/non-deterministic.
        # Build an isolated single-threaded equivalent so every Task02
        # candidate is ranked on the same fixed-seed validation patches.
        self._task02_lits_val_loader = self._task02_make_validation_augmenter(active_lits_val)
        self._task02_write_manifest(hcc_cases)
        self._task02_source_lits_val_dice = self._task02_lightweight_dice(
            self._task02_lits_val_loader, domain="lits"
        )
        self._task02_write_validation_history(
            {
                "epoch": 0,
                "kind": "source_baseline",
                "lits_val_mean_tumor_dice": self._task02_source_lits_val_dice,
                "lits_val_case_ids": list(canonical_lits["val"]),
            }
        )
        self.print_to_log_file(  # type: ignore[attr-defined]
            f"[Task02] mode={self.TASK02_MODE}, k={self.TASK02_K}, hcc_cases={list(hcc_cases)}, "
            f"updates/epoch={self.TASK02_UPDATES_PER_EPOCH}, replay_lambda="
            f"{self.TASK02_REPLAY_LAMBDA if self.TASK02_MODE == 'replay' else 'N/A'}"
        )

    def _task02_lightweight_dice(self, loader, *, domain: str) -> float:
        """Return a reproducible patch-level tumor Dice for one validation domain.

        This is intentionally the inexpensive training-time validation used to
        select checkpoints. It is not a final full-volume evaluation and never
        writes predictions, reports, or visualizations.
        """
        if domain not in {"hcc", "lits"}:
            raise ValueError(f"Unknown Task02 validation domain: {domain}")
        world_size = dist.get_world_size() if self.is_ddp else 1  # type: ignore[attr-defined]
        if self.TASK02_VALIDATION_ITERATIONS % world_size != 0:
            raise RuntimeError(
                "Task02 global validation step count must be divisible by the DDP world size: "
                f"steps={self.TASK02_VALIDATION_ITERATIONS}, world_size={world_size}"
            )
        local_iterations = self.TASK02_VALIDATION_ITERATIONS // world_size
        # Preserve training RNG state. Validation must not change which
        # augmentations the next training epoch receives.
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.random.get_rng_state()
        cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        domain_offset = 0 if domain == "hcc" else 10_000
        validation_seed = self.TASK02_VALIDATION_SEED + domain_offset + int(self.local_rank)  # type: ignore[attr-defined]
        random.seed(validation_seed)
        np.random.seed(validation_seed)
        torch.manual_seed(validation_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(validation_seed)
        outputs = []
        try:
            self.network.eval()  # type: ignore[attr-defined]
            with torch.no_grad():
                for _ in range(local_iterations):
                    outputs.append(self.validation_step(next(loader)))  # type: ignore[attr-defined]
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.random.set_rng_state(torch_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)
        tp = np.sum([item["tp_hard"] for item in outputs], axis=0)
        fp = np.sum([item["fp_hard"] for item in outputs], axis=0)
        fn = np.sum([item["fn_hard"] for item in outputs], axis=0)
        if self.is_ddp:  # type: ignore[attr-defined]
            gathered = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(gathered, (tp, fp, fn))
            tp = np.sum([item[0] for item in gathered], axis=0)
            fp = np.sum([item[1] for item in gathered], axis=0)
            fn = np.sum([item[2] for item in gathered], axis=0)
        # Dataset003/HCC labels are foreground [liver=1, tumor=2].
        if len(tp) != 2:
            raise RuntimeError(f"Task02 expects liver/tumor foreground metrics, got {len(tp)} classes.")
        denominator = 2 * float(tp[1]) + float(fp[1]) + float(fn[1])
        return 0.0 if denominator == 0 else 2 * float(tp[1]) / denominator

    def _task02_write_validation_history(self, record: dict[str, Any]) -> None:
        if self.local_rank != 0:  # type: ignore[attr-defined]
            return
        path = Path(self.output_folder) / "task02_validation_history.jsonl"  # type: ignore[attr-defined]
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    def _task02_maybe_update_best_checkpoint(self, epoch: int, hcc_dice: float, lits_dice: float) -> str:
        if self._task02_source_lits_val_dice is None:
            raise RuntimeError("Task02 source LiTS validation baseline was not recorded.")
        eligible = self.TASK02_MODE == "hcc_only" or lits_dice >= self._task02_source_lits_val_dice - 0.02
        improved = eligible and (
            self._task02_best_hcc_dice is None or hcc_dice > self._task02_best_hcc_dice
        )
        if not improved:
            return "ineligible" if not eligible else "not_improved"
        candidate = Path(self.output_folder) / "checkpoint_task02_candidate.pth"  # type: ignore[attr-defined]
        target = Path(self.output_folder) / "checkpoint_task02_best.pth"  # type: ignore[attr-defined]
        self.save_checkpoint(str(candidate))  # type: ignore[attr-defined]
        if self.local_rank == 0:  # type: ignore[attr-defined]
            shutil.copy2(candidate, target)
            (Path(self.output_folder) / "task02_selection.json").write_text(  # type: ignore[attr-defined]
                json.dumps(
                    {
                        "status": "selected",
                        "mode": self.TASK02_MODE,
                        "selected_epoch": epoch,
                        "selected_checkpoint": str(target),
                        "selected_checkpoint_sha256": sha256_file(target),
                        "hcc_val_patch_tumor_dice": hcc_dice,
                        "lits_val_patch_tumor_dice": lits_dice,
                        "source_lits_val_patch_tumor_dice": self._task02_source_lits_val_dice,
                        "retention_max_absolute_drop": 0.02 if self.TASK02_MODE == "replay" else None,
                    },
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
        self._task02_best_hcc_dice = hcc_dice
        return "updated_best"

    def run_training(self):
        """Run Stage-2 updates and lightweight dual-domain checkpoint selection."""
        self.on_train_start()
        if self._task02_hcc_loader is None:
            raise RuntimeError("Task02 HCC loader was not initialized.")
        for _ in range(self.current_epoch, self.num_epochs):  # type: ignore[attr-defined]
            self.on_epoch_start()  # type: ignore[attr-defined]
            self.on_train_epoch_start()  # type: ignore[attr-defined]
            train_outputs = []
            for _ in range(self.num_iterations_per_epoch):  # type: ignore[attr-defined]
                hcc_batch = next(self._task02_hcc_loader)
                if self.TASK02_MODE == "hcc_only":
                    train_outputs.append(self.train_step(hcc_batch))  # type: ignore[attr-defined]
                else:
                    train_outputs.append(self._task02_replay_step(hcc_batch, next(self.dataloader_train)))  # type: ignore[attr-defined]
            self.on_train_epoch_end(train_outputs)  # type: ignore[attr-defined]

            self.on_task02_validation_epoch_end()
            self.on_epoch_end()  # type: ignore[attr-defined]
        self.on_train_end()  # type: ignore[attr-defined]

    def on_task02_validation_epoch_end(self) -> None:
        epoch = int(self.current_epoch) + 1  # type: ignore[attr-defined]
        if epoch % self.TASK02_VALIDATION_INTERVAL != 0:
            return
        if self._task02_hcc_val_loader is None:
            raise RuntimeError("Task02 HCC validation loader was not initialized.")
        if self._task02_lits_val_loader is None:
            raise RuntimeError("Task02 LiTS validation loader was not initialized.")
        hcc_dice = self._task02_lightweight_dice(self._task02_hcc_val_loader, domain="hcc")
        lits_dice = self._task02_lightweight_dice(self._task02_lits_val_loader, domain="lits")
        selection = self._task02_maybe_update_best_checkpoint(epoch, hcc_dice, lits_dice)
        record = {
            "epoch": epoch,
            "kind": "lightweight_patch_validation",
            "hcc_val_mean_tumor_dice": hcc_dice,
            "lits_val_mean_tumor_dice": lits_dice,
            "selection": selection,
            "hcc_val_case_ids": list(load_hcc_partitions()["val"]),
            "lits_val_case_ids": list(load_lits_partitions()["val"]),
            "validation_steps_per_domain_global": self.TASK02_VALIDATION_ITERATIONS,
            "validation_seed": self.TASK02_VALIDATION_SEED,
        }
        self._task02_write_validation_history(record)
        self.print_to_log_file(  # type: ignore[attr-defined]
            f"[Task02][epoch {epoch}] HCC-val patch Tumor Dice={hcc_dice:.4f}; "
            f"LiTS-val patch Tumor Dice={lits_dice:.4f}; selection={selection}"
        )

    def on_epoch_end(self):
        """Task02 epoch bookkeeping without nnU-Net's patch-Dice best checkpoint."""
        self._synchronize_resource_device()  # type: ignore[attr-defined]
        self.logger.log("epoch_end_timestamps", time(), self.current_epoch)  # type: ignore[attr-defined]
        self._record_epoch_resource_usage()  # type: ignore[attr-defined]
        train_loss = self.logger.get_value("train_losses", step=-1)  # type: ignore[attr-defined]
        self.print_to_log_file("train_loss", np.round(train_loss, decimals=4), add_timestamp=False)  # type: ignore[attr-defined]
        epoch = int(self.current_epoch)  # type: ignore[attr-defined]
        if (epoch + 1) % self.save_every == 0 and epoch != (self.num_epochs - 1):  # type: ignore[attr-defined]
            self.save_checkpoint(str(Path(self.output_folder) / "checkpoint_latest.pth"))  # type: ignore[attr-defined]
        # Do not call nnU-Net's ``plot_progress_png`` here. That plot assumes
        # one base validation entry per epoch (val_losses, pseudo Dice and EMA),
        # whereas Task02 intentionally validates every three epochs and writes
        # its two-domain selection evidence to task02_validation_history.jsonl.
        # Calling it would create a misleading or malformed progress plot.
        self.current_epoch += 1  # type: ignore[attr-defined]

    def perform_actual_validation(self, save_probabilities: bool = False):
        """Prevent nnU-Net's default ``--val`` path from touching Task02 tests.

        The inherited MedNeXt trainer carries historical automatic internal and
        external test hooks. Task02 must instead use its explicit three-domain
        final-evaluation entrypoint after `checkpoint_task02_best.pth` exists.
        """
        raise RuntimeError(
            "Task02 does not permit nnU-Net --val / perform_actual_validation. "
            "Use the explicit Task02 three-domain final-evaluation command after checkpoint selection."
        )

    def on_train_end(self):
        hcc_loader = self._task02_hcc_loader
        if isinstance(hcc_loader, (NonDetMultiThreadedAugmenter, MultiThreadedAugmenter)):
            hcc_loader._finish()
        hcc_val_loader = self._task02_hcc_val_loader
        if isinstance(hcc_val_loader, (NonDetMultiThreadedAugmenter, MultiThreadedAugmenter)):
            hcc_val_loader._finish()
        lits_val_loader = self._task02_lits_val_loader
        if isinstance(lits_val_loader, (NonDetMultiThreadedAugmenter, MultiThreadedAugmenter)):
            lits_val_loader._finish()
        target = Path(self.output_folder) / "checkpoint_task02_best.pth"  # type: ignore[attr-defined]
        if self.local_rank == 0 and not target.is_file():  # type: ignore[attr-defined]
            # Replay can legitimately reject every adapted epoch under the
            # frozen LiTS retention constraint. Make that fallback explicit.
            shutil.copy2(FROZEN_SOURCE_CHECKPOINT, target)
            (Path(self.output_folder) / "task02_selection.json").write_text(  # type: ignore[attr-defined]
                json.dumps(
                    {
                        "status": "no_acceptable_adaptation",
                        "selected_checkpoint": str(target),
                        "selected_checkpoint_sha256": FROZEN_SOURCE_SHA256,
                        "reason": "No Stage-2 candidate met the frozen Task02 selection rule.",
                    },
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
        super().on_train_end()  # type: ignore[misc]
