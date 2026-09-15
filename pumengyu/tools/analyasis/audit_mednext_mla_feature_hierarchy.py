#!/usr/bin/env python3
"""Audit whether MedNeXt-MLA representations form a feature hierarchy.

This is a representation audit, not a segmentation evaluation. It freezes the
trained network, extracts encoder activations from case-level train/validation
splits, and fits deliberately small linear probes. The audit always writes a
JSON summary, a human-readable report, and PNG visualizations.

The tested targets are operational definitions rather than universal meanings:

* intensity: nnU-Net-normalized CT intensity;
* edge: 3-D gradient magnitude;
* texture: local standard deviation;
* local_structure: multi-scale Laplacian-of-Gaussian response;
* organ_shape: signed distance to the liver boundary;
* semantics: background/liver/tumor class.

Example (run only when a GPU is free):

    CUDA_VISIBLE_DEVICES=0 \
    /home/PuMengYu/anaconda3/envs/medseg/bin/python \
      pumengyu/tools/analyasis/audit_mednext_mla_feature_hierarchy.py \
      --device cuda:0 \
      --visualize-case liver_93

Omit ``--visualize-case`` to run the slower multi-case linear-probe audit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import warnings
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage, stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, f1_score, r2_score
from sklearn.preprocessing import StandardScaler


# External-trainer discovery imports unrelated optional trainer modules. Keep
# their Transformers compatibility chatter out of this focused visualization.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings(
    "ignore",
    message="Importing from timm.models.layers is deprecated.*",
    category=FutureWarning,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_DIR = Path(
    "/home/PuMengYu/nnUNet_workspace/results_v2/Dataset003_Liver/"
    "nnUNetTrainer_MedNeXt_MLA__nnUNetPlans__3d_fullres"
)
DEFAULT_PREPROCESSED_DIR = Path(
    "/home/PuMengYu/nnUNet_workspace/preprocessed/Dataset003_Liver"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "pumengyu/notes/实验结果分析/mednext_mla_feature_hierarchy"
)

LAYER_MODULES = OrderedDict(
    [
        ("input", None),
        ("input_projection", "stem"),
        ("encoder_0", "enc_block_0"),
        ("encoder_1", "enc_block_1"),
        ("encoder_2", "enc_block_2"),
        ("encoder_3", "enc_block_3"),
        ("conv_bottleneck", "bottleneck"),
        ("mla_output", "mla_bot"),
    ]
)
DECODER_HEADS = OrderedDict(
    [
        ("bottleneck_8", "out_4"),
        ("decoder_3_16", "out_3"),
        ("decoder_2_32", "out_2"),
        ("decoder_1_64", "out_1"),
        ("decoder_0_128", "out_0"),
    ]
)
CONTINUOUS_TARGETS = (
    "intensity",
    "edge",
    "texture",
    "local_structure",
    "organ_shape",
)
TARGET_ORDER = CONTINUOUS_TARGETS + ("semantics",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit the MedNeXt-MLA encoder feature hierarchy."
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--preprocessed-dir", type=Path, default=DEFAULT_PREPROCESSED_DIR)
    parser.add_argument("--checkpoint", default="checkpoint_best.pth")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--train-cases", type=int, default=24)
    parser.add_argument("--val-cases", type=int, default=13)
    parser.add_argument("--samples-per-case", type=int, default=2048)
    parser.add_argument("--pca-components", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--visualize-case",
        help=(
            "Run one tumor-centered patch from this preprocessed case and write "
            "one detailed PNG per observed layer; skip multi-case linear probes."
        ),
    )
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Check checkpoint, split and data provenance without running inference.",
    )
    return parser.parse_args()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def inspect_checkpoint(model_dir: Path, fold: int, checkpoint_name: str) -> dict[str, Any]:
    path = model_dir / f"fold_{fold}" / checkpoint_name
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    trainer = checkpoint.get("trainer_name")
    state = checkpoint.get("network_weights", {})
    mla_keys = [k for k in state if k.startswith("mla_bot.")]
    moe_keys = [k for k in state if "experts" in k or "router" in k]
    valid = trainer == "nnUNetTrainer_MedNeXt_MLA" and bool(mla_keys) and not moe_keys
    if not valid:
        raise RuntimeError(
            "Refusing hierarchy audit: checkpoint is not verified pure MedNeXt_MLA "
            f"(trainer={trainer!r}, MLA keys={len(mla_keys)}, MoE keys={len(moe_keys)})."
        )
    stat = path.stat()
    result = {
        "path": str(path.resolve()),
        "name": checkpoint_name,
        "trainer": trainer,
        "fold": fold,
        "current_epoch": checkpoint.get("current_epoch"),
        "best_ema": checkpoint.get("_best_ema"),
        "mtime": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        "size_bytes": stat.st_size,
        "sha256": _sha256(path),
        "mla_parameter_key_count": len(mla_keys),
        "moe_parameter_key_count": len(moe_keys),
        "pure_mla_verified": True,
    }
    del checkpoint, state
    final_path = model_dir / f"fold_{fold}" / "checkpoint_final.pth"
    if checkpoint_name == "checkpoint_best.pth" and final_path.is_file():
        final_checkpoint = torch.load(final_path, map_location="cpu", weights_only=False)
        result["final_checkpoint_crosscheck"] = {
            "path": str(final_path.resolve()),
            "current_epoch": final_checkpoint.get("current_epoch"),
            "best_ema": final_checkpoint.get("_best_ema"),
            "trainer": final_checkpoint.get("trainer_name"),
            "mtime": datetime.fromtimestamp(
                final_path.stat().st_mtime, tz=timezone.utc
            ).isoformat(),
            "best_precedes_final": stat.st_mtime < final_path.stat().st_mtime,
            "trainer_matches": final_checkpoint.get("trainer_name") == trainer,
        }
        del final_checkpoint
    return result


def load_split(
    preprocessed_dir: Path,
    fold: int,
    n_train: int,
    n_val: int,
    seed: int,
) -> dict[str, list[str]]:
    split_path = preprocessed_dir / "splits_final.json"
    if not split_path.is_file():
        raise FileNotFoundError(f"Split file not found: {split_path}")
    splits = json.loads(split_path.read_text(encoding="utf-8"))
    if fold >= len(splits):
        raise IndexError(f"Fold {fold} missing from {split_path}; available folds={len(splits)}")
    rng = np.random.default_rng(seed)
    train_pool = np.asarray(splits[fold]["train"], dtype=object)
    val_pool = np.asarray(splits[fold]["val"], dtype=object)
    selected = {
        "train": [str(v) for v in rng.permutation(train_pool)[:n_train]],
        "val": [str(v) for v in rng.permutation(val_pool)[:n_val]],
    }
    data_dir = preprocessed_dir / "nnUNetPlans_3d_fullres"
    missing = [
        case
        for cases in selected.values()
        for case in cases
        if not (data_dir / f"{case}.b2nd").is_file()
        or not (data_dir / f"{case}_seg.b2nd").is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Missing preprocessed data/seg for cases: {missing}")
    return selected


def write_inspection(
    output_dir: Path,
    checkpoint: dict[str, Any],
    split: dict[str, list[str]],
    args: argparse.Namespace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": "inspection_complete_inference_not_run",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": checkpoint,
        "data": {
            "preprocessed_dir": str(args.preprocessed_dir.resolve()),
            "fold": args.fold,
            "train_cases": split["train"],
            "val_cases": split["val"],
        },
        "planned_layers": list(LAYER_MODULES),
        "planned_targets": list(TARGET_ORDER),
        "artifacts": {
            "summary_json": True,
            "text_report": True,
            "visualization_png_count": 0,
            "reason_no_png": "inspect-only mode; no activations or probe scores were generated",
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = (
        "MedNeXt_MLA 表征层级审计：静态检查\n"
        "===================================\n"
        f"状态：{summary['status']}\n"
        f"模型：{checkpoint['trainer']}\n"
        f"checkpoint：{checkpoint['path']}\n"
        f"epoch：{checkpoint['current_epoch']}\n"
        f"final cross-check epoch：{checkpoint.get('final_checkpoint_crosscheck', {}).get('current_epoch', 'N/A')}\n"
        f"MLA 参数键：{checkpoint['mla_parameter_key_count']}\n"
        f"MoE 参数键：{checkpoint['moe_parameter_key_count']}\n"
        f"训练探针病例：{len(split['train'])}\n"
        f"独立验证探针病例：{len(split['val'])}\n"
        "可视化 PNG：0（inspect-only，尚未运行特征提取）\n"
        "结论：只验证了模型、checkpoint、split 和输入文件；尚无表征层级实验证据。\n"
    )
    (output_dir / "feature_hierarchy_report.txt").write_text(report, encoding="utf-8")


def crop_or_pad(array: np.ndarray, center: np.ndarray, patch_size: tuple[int, int, int], pad: float) -> np.ndarray:
    spatial = np.asarray(array.shape[-3:])
    patch = np.asarray(patch_size)
    start = center.astype(int) - patch // 2
    end = start + patch
    src_start = np.maximum(start, 0)
    src_end = np.minimum(end, spatial)
    dst_start = src_start - start
    dst_end = dst_start + (src_end - src_start)
    result = np.full((*array.shape[:-3], *patch), pad, dtype=array.dtype)
    result[(...,) + tuple(slice(a, b) for a, b in zip(dst_start, dst_end))] = array[
        (...,) + tuple(slice(a, b) for a, b in zip(src_start, src_end))
    ]
    return result


def make_targets(image: np.ndarray, seg: np.ndarray) -> dict[str, np.ndarray]:
    image = image.astype(np.float32, copy=False)
    valid = seg >= 0
    safe_seg = np.where(valid, seg, 0).astype(np.int16)
    gradients = np.gradient(image)
    edge = np.sqrt(sum(g.astype(np.float32) ** 2 for g in gradients))
    mean = ndimage.gaussian_filter(image, sigma=1.5)
    mean_sq = ndimage.gaussian_filter(image * image, sigma=1.5)
    texture = np.sqrt(np.maximum(mean_sq - mean * mean, 0.0))
    local_structure = np.abs(ndimage.gaussian_laplace(image, sigma=2.0))
    liver = safe_seg > 0
    if liver.any():
        signed = ndimage.distance_transform_edt(liver) - ndimage.distance_transform_edt(~liver)
        organ_shape = np.clip(signed, -32.0, 32.0) / 32.0
    else:
        organ_shape = np.zeros_like(image, dtype=np.float32)
    return {
        "intensity": image,
        "edge": edge,
        "texture": texture,
        "local_structure": local_structure,
        "organ_shape": organ_shape.astype(np.float32),
        "semantics": safe_seg,
        "valid": valid,
    }


def resize_target(array: np.ndarray, shape: tuple[int, int, int], nearest: bool = False) -> np.ndarray:
    tensor = torch.from_numpy(np.asarray(array, dtype=np.float32))[None, None]
    if not nearest and all(src >= dst for src, dst in zip(array.shape, shape)):
        return F.adaptive_avg_pool3d(tensor, shape)[0, 0].numpy()
    mode = "nearest" if nearest else "trilinear"
    kwargs = {} if nearest else {"align_corners": False}
    return F.interpolate(tensor, size=shape, mode=mode, **kwargs)[0, 0].numpy()


def resize_semantics(seg: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    """Label a coarse cell by whether it contains tumor, then liver.

    Plain nearest-neighbor sampling can erase a small tumor at the 8^3
    bottleneck. Presence pooling matches the question a coarse receptive cell
    can answer, but its class labels are therefore explicitly cell-level.
    """
    tensor = torch.from_numpy(np.asarray(seg))[None, None]
    tumor = F.adaptive_max_pool3d((tensor == 2).float(), shape)[0, 0].bool()
    organ = F.adaptive_max_pool3d((tensor > 0).float(), shape)[0, 0].bool()
    result = torch.zeros(shape, dtype=torch.int16)
    result[organ] = 1
    result[tumor] = 2
    return result.numpy()


def sample_indices(seg: np.ndarray, valid: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    flat_valid = np.flatnonzero(valid.reshape(-1))
    if len(flat_valid) == 0:
        raise RuntimeError("Patch has no valid voxels after crop/pad")
    pieces = []
    uniform_n = count // 2
    pieces.append(rng.choice(flat_valid, size=uniform_n, replace=len(flat_valid) < uniform_n))
    class_n = max(1, (count - uniform_n) // 3)
    flat_seg = seg.reshape(-1)
    for label in (0, 1, 2):
        candidates = np.flatnonzero((flat_seg == label) & valid.reshape(-1))
        if len(candidates):
            pieces.append(rng.choice(candidates, size=class_n, replace=len(candidates) < class_n))
    indices = np.concatenate(pieces)
    if len(indices) < count:
        extra = rng.choice(flat_valid, size=count - len(indices), replace=True)
        indices = np.concatenate([indices, extra])
    rng.shuffle(indices)
    return indices[:count]


def load_network(model_dir: Path, fold: int, checkpoint: str, device: torch.device):
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    predictor = nnUNetPredictor(
        perform_everything_on_device=device.type == "cuda",
        device=device,
        verbose=False,
        allow_tqdm=False,
    )
    predictor.initialize_from_trained_model_folder(
        str(model_dir), use_folds=(fold,), checkpoint_name=checkpoint
    )
    network = predictor.network
    if hasattr(network, "outside_block_checkpointing"):
        network.outside_block_checkpointing = False
    network.eval().to(device)
    return network, predictor.configuration_manager.patch_size


def register_hooks(
    network: torch.nn.Module,
    storage: dict[str, torch.Tensor],
    layer_modules: OrderedDict[str, str | None] = LAYER_MODULES,
):
    handles = []
    modules = dict(network.named_modules())
    for layer, module_name in layer_modules.items():
        if module_name is None:
            continue
        if module_name not in modules:
            raise KeyError(f"Expected module {module_name!r} for layer {layer!r} is missing")

        def hook(_module, _inputs, output, layer_name=layer):
            if isinstance(output, (list, tuple)):
                output = output[0]
            storage[layer_name] = output.detach()

        handles.append(modules[module_name].register_forward_hook(hook))
    return handles


def extract_case_samples(
    network: torch.nn.Module,
    data_dir: Path,
    case: str,
    patch_size: tuple[int, int, int],
    device: torch.device,
    samples_per_case: int,
    rng: np.random.Generator,
    keep_maps: bool,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any] | None]:
    from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2

    dataset = nnUNetDatasetBlosc2(str(data_dir), identifiers=[case])
    data_obj, seg_obj, _, _ = dataset.load_case(case)
    data = np.asarray(data_obj[:], dtype=np.float32)
    seg = np.asarray(seg_obj[:], dtype=np.int16)
    seg0 = seg[0]
    foreground = np.argwhere(seg0 == 2)
    if len(foreground) == 0:
        foreground = np.argwhere(seg0 > 0)
    center = (
        np.rint(np.median(foreground, axis=0)).astype(int)
        if len(foreground)
        else np.asarray(seg0.shape) // 2
    )
    patch_data = crop_or_pad(data, center, patch_size, 0.0)
    patch_seg = crop_or_pad(seg, center, patch_size, -1)[0]
    targets = make_targets(patch_data[0], patch_seg)

    activations: dict[str, torch.Tensor] = {}
    decoder_logits: dict[str, torch.Tensor] = {}
    handles = register_hooks(network, activations)
    if keep_maps:
        handles.extend(register_hooks(network, decoder_logits, DECODER_HEADS))
    tensor = torch.from_numpy(patch_data[None]).to(device)
    activations["input"] = tensor.detach()
    previous_do_ds = bool(network.do_ds)
    if keep_maps:
        network.do_ds = True
    try:
        with torch.inference_mode():
            _ = network(tensor)
    finally:
        network.do_ds = previous_do_ds
        for handle in handles:
            handle.remove()

    sampled: dict[str, dict[str, np.ndarray]] = {}
    maps: dict[str, Any] | None = {} if keep_maps else None
    for layer in LAYER_MODULES:
        feature = activations[layer][0].float().cpu()
        shape = tuple(int(v) for v in feature.shape[-3:])
        resized = {
            name: resize_target(targets[name], shape)
            for name in CONTINUOUS_TARGETS
        }
        resized["semantics"] = resize_semantics(targets["semantics"], shape)
        resized["valid"] = resize_target(targets["valid"], shape, nearest=True)
        resized["valid"] = resized["valid"] > 0.5
        indices = sample_indices(
            resized["semantics"], resized["valid"], samples_per_case, rng
        )
        x = feature.reshape(feature.shape[0], -1).T.numpy()[indices]
        sampled[layer] = {
            "x": x.astype(np.float32),
            **{name: resized[name].reshape(-1)[indices] for name in TARGET_ORDER},
        }
        if maps is not None:
            energy = torch.sqrt(torch.mean(feature * feature, dim=0) + 1e-8)
            maps[layer] = resize_target(energy.numpy(), patch_size)
            channel_std = feature.reshape(feature.shape[0], -1).std(dim=1)
            top_count = min(8, feature.shape[0])
            top_indices = torch.topk(channel_std, k=top_count).indices
            maps[f"{layer}__top_channels"] = feature[top_indices].numpy()
            maps[f"{layer}__top_indices"] = top_indices.numpy()
            maps[f"{layer}__shape"] = [int(v) for v in feature.shape]
    if maps is not None:
        maps.update({name: targets[name] for name in (*TARGET_ORDER, "valid")})
        labels, counts = np.unique(targets["semantics"][targets["valid"]], return_counts=True)
        maps["label_voxel_counts"] = {
            int(label): int(count) for label, count in zip(labels, counts)
        }
        decoder_probabilities = {}
        decoder_native_shapes = {}
        for stage in DECODER_HEADS:
            logits = decoder_logits[stage][0].float().cpu()
            decoder_native_shapes[stage] = [int(v) for v in logits.shape]
            probability = torch.softmax(logits, dim=0)[None]
            probability = F.interpolate(
                probability,
                size=patch_size,
                mode="trilinear",
                align_corners=False,
            )[0]
            decoder_probabilities[stage] = probability.numpy()
        maps["decoder_probabilities"] = decoder_probabilities
        maps["decoder_native_shapes"] = decoder_native_shapes
    return sampled, maps


def fit_probes(
    train: dict[str, list[dict[str, np.ndarray]]],
    val: dict[str, list[dict[str, np.ndarray]]],
    pca_components: int,
    seed: int,
) -> dict[str, dict[str, Any]]:
    scores: dict[str, dict[str, Any]] = {}
    for layer in LAYER_MODULES:
        x_train = np.concatenate([item["x"] for item in train[layer]], axis=0)
        x_val = np.concatenate([item["x"] for item in val[layer]], axis=0)
        scaler = StandardScaler().fit(x_train)
        x_train_s = scaler.transform(x_train)
        x_val_s = scaler.transform(x_val)
        components = min(pca_components, x_train_s.shape[1], x_train_s.shape[0] - 1)
        pca = PCA(n_components=components, svd_solver="randomized", random_state=seed)
        z_train = pca.fit_transform(x_train_s)
        z_val = pca.transform(x_val_s)
        layer_scores: dict[str, Any] = {
            "feature_channels": int(x_train.shape[1]),
            "probe_dimensions": int(components),
            "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
        }
        for target in CONTINUOUS_TARGETS:
            y_train = np.concatenate([item[target] for item in train[layer]])
            y_val = np.concatenate([item[target] for item in val[layer]])
            probe = Ridge(alpha=10.0).fit(z_train, y_train)
            prediction = probe.predict(z_val)
            corr = stats.pearsonr(y_val, prediction).statistic if np.std(prediction) > 0 else 0.0
            layer_scores[target] = {
                "r2": float(r2_score(y_val, prediction)),
                "pearson_r": float(corr),
            }
        y_train = np.concatenate([item["semantics"] for item in train[layer]]).astype(int)
        y_val = np.concatenate([item["semantics"] for item in val[layer]]).astype(int)
        classifier = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=400,
            random_state=seed,
            solver="lbfgs",
        ).fit(z_train, y_train)
        pred = classifier.predict(z_val)
        layer_scores["semantics"] = {
            "balanced_accuracy": float(balanced_accuracy_score(y_val, pred)),
            "macro_f1": float(f1_score(y_val, pred, average="macro", zero_division=0)),
            "classes_present": sorted(int(v) for v in np.unique(y_val)),
        }
        scores[layer] = layer_scores
    return scores


def primary_score(scores: dict[str, dict[str, Any]], layer: str, target: str) -> float:
    if target == "semantics":
        return scores[layer][target]["balanced_accuracy"]
    return scores[layer][target]["r2"]


def summarize_hierarchy(scores: dict[str, dict[str, Any]]) -> dict[str, Any]:
    layers = list(LAYER_MODULES)
    peaks = {}
    peak_indices = []
    for target in TARGET_ORDER:
        values = [primary_score(scores, layer, target) for layer in layers]
        index = int(np.nanargmax(values))
        peaks[target] = {"layer": layers[index], "layer_index": index, "score": values[index]}
        peak_indices.append(index)
    rho = stats.spearmanr(np.arange(len(TARGET_ORDER)), peak_indices).statistic
    conv_to_mla = {
        target: primary_score(scores, "mla_output", target)
        - primary_score(scores, "conv_bottleneck", target)
        for target in TARGET_ORDER
    }
    return {
        "peak_layer_by_target": peaks,
        "spearman_target_order_vs_peak_depth": float(rho),
        "mla_minus_conv_bottleneck": conv_to_mla,
        "interpretation_boundary": (
            "A positive ordered trend supports decodability of the operational targets; "
            "it does not prove that a layer exclusively represents one named concept."
        ),
    }


def plot_score_matrix(scores: dict[str, dict[str, Any]], output: Path) -> None:
    layers = list(LAYER_MODULES)
    raw = np.asarray(
        [[primary_score(scores, layer, target) for layer in layers] for target in TARGET_ORDER]
    )
    row_min = np.nanmin(raw, axis=1, keepdims=True)
    row_span = np.maximum(np.nanmax(raw, axis=1, keepdims=True) - row_min, 1e-8)
    normalized = (raw - row_min) / row_span
    fig, ax = plt.subplots(figsize=(12.5, 6.2))
    image = ax.imshow(normalized, cmap="YlGnBu", aspect="auto", vmin=0, vmax=1)
    for row in range(raw.shape[0]):
        for col in range(raw.shape[1]):
            ax.text(col, row, f"{raw[row, col]:.3f}", ha="center", va="center", fontsize=8)
    ax.set_xticks(range(len(layers)), [name.replace("_", "\n") for name in layers])
    ax.set_yticks(range(len(TARGET_ORDER)), TARGET_ORDER)
    ax.set_title("Frozen linear-probe scores across MedNeXt-MLA encoder depth")
    ax.set_xlabel("Layer (left to right = deeper encoder representation)")
    ax.set_ylabel("Operational target (R2; semantics uses balanced accuracy)")
    cbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Within-target normalized score; cells show raw score")
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _draw_contours(ax, semantics_slice: np.ndarray) -> None:
    if np.any(semantics_slice > 0) and np.any(semantics_slice == 0):
        ax.contour(
            semantics_slice > 0,
            levels=[0.5],
            colors=["#48BFE3"],
            linewidths=0.7,
        )
    if np.any(semantics_slice == 2) and np.any(semantics_slice != 2):
        ax.contour(
            semantics_slice == 2,
            levels=[0.5],
            colors=["#F72585"],
            linewidths=1.0,
        )


def plot_case_maps(case: str, maps: dict[str, Any], output: Path) -> None:
    tumor_per_slice = np.sum(maps["semantics"] == 2, axis=(1, 2))
    liver_per_slice = np.sum(maps["semantics"] > 0, axis=(1, 2))
    z = int(np.argmax(tumor_per_slice if tumor_per_slice.max() else liver_per_slice))
    layers = list(LAYER_MODULES)
    fig, axes = plt.subplots(2, 4, figsize=(14, 7.2))
    base = maps["intensity"][z]
    lo, hi = np.percentile(base[maps["valid"][z]], [1, 99])
    for ax, layer in zip(axes.flat, layers):
        heat = maps[layer][z]
        heat = (heat - np.percentile(heat, 2)) / (
            np.percentile(heat, 98) - np.percentile(heat, 2) + 1e-8
        )
        ax.imshow(base, cmap="gray", vmin=lo, vmax=hi)
        ax.imshow(np.clip(heat, 0, 1), cmap="magma", alpha=0.52, vmin=0, vmax=1)
        _draw_contours(ax, maps["semantics"][z])
        ax.set_title(layer.replace("_", " "), fontsize=10)
        ax.axis("off")
    fig.suptitle(
        f"{case}, axial z={z}: channel-RMS activation energy (exploratory, not proof)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_layer_channels(
    case: str,
    layer: str,
    maps: dict[str, Any],
    output: Path,
) -> None:
    """Write one interpretable page for one layer.

    The first panel is channel-RMS energy. Remaining panels are the channels
    with the largest spatial standard deviation. Channel selection is stated
    explicitly because a feature tensor does not have one canonical image.
    """
    tumor_per_slice = np.sum(maps["semantics"] == 2, axis=(1, 2))
    liver_per_slice = np.sum(maps["semantics"] > 0, axis=(1, 2))
    z = int(np.argmax(tumor_per_slice if tumor_per_slice.max() else liver_per_slice))
    base = maps["intensity"][z]
    valid_slice = maps["valid"][z]
    lo, hi = np.percentile(base[valid_slice], [1, 99])
    semantics_slice = maps["semantics"][z]
    channels = maps[f"{layer}__top_channels"]
    indices = maps[f"{layer}__top_indices"]
    fig, axes = plt.subplots(3, 3, figsize=(10.8, 10.3))

    energy = maps[layer][z]
    e_lo, e_hi = np.percentile(energy, [2, 98])
    axes.flat[0].imshow(base, cmap="gray", vmin=lo, vmax=hi)
    axes.flat[0].imshow(
        np.clip((energy - e_lo) / (e_hi - e_lo + 1e-8), 0, 1),
        cmap="magma",
        alpha=0.55,
        vmin=0,
        vmax=1,
    )
    _draw_contours(axes.flat[0], semantics_slice)
    axes.flat[0].set_title("Channel-RMS activation energy")
    axes.flat[0].axis("off")

    for panel, (channel, index) in enumerate(zip(channels, indices), start=1):
        upsampled = resize_target(channel, maps["intensity"].shape)
        view = upsampled[z]
        scale = max(float(np.percentile(np.abs(view), 98)), 1e-8)
        axes.flat[panel].imshow(view, cmap="coolwarm", vmin=-scale, vmax=scale)
        _draw_contours(axes.flat[panel], semantics_slice)
        axes.flat[panel].set_title(f"Channel {int(index)} (top spatial variation)")
        axes.flat[panel].axis("off")
    for panel in range(1 + len(channels), len(axes.flat)):
        axes.flat[panel].axis("off")
    tensor_shape = maps[f"{layer}__shape"]
    fig.suptitle(
        f"{case} | {layer} | feature shape {tensor_shape} | axial z={z}",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _segmentation_rgba(segmentation: np.ndarray) -> np.ndarray:
    rgba = np.zeros((*segmentation.shape, 4), dtype=np.float32)
    liver = segmentation == 1
    tumor = segmentation == 2
    rgba[liver] = (0.10, 0.75, 0.88, 0.16)
    rgba[tumor] = (0.97, 0.15, 0.52, 0.75)
    return rgba


def _tumor_error(gt: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    result = np.zeros(gt.shape, dtype=np.uint8)
    gt_tumor = gt == 2
    pred_tumor = prediction == 2
    result[gt_tumor & pred_tumor] = 1  # true positive
    result[~gt_tumor & pred_tumor] = 2  # false positive
    result[gt_tumor & ~pred_tumor] = 3  # false negative
    return result


def _display_base(ax, base: np.ndarray, lo: float, hi: float) -> None:
    ax.imshow(base, cmap="gray", vmin=lo, vmax=hi)
    ax.axis("off")


def plot_decoder_stage(
    case: str,
    stage: str,
    maps: dict[str, Any],
    output: Path,
) -> None:
    probabilities = maps["decoder_probabilities"][stage]
    if probabilities.shape[0] < 3:
        raise RuntimeError(
            f"Expected background/liver/tumor logits, got {probabilities.shape[0]} channels"
        )
    gt = maps["semantics"]
    tumor_per_slice = np.sum(gt == 2, axis=(1, 2))
    liver_per_slice = np.sum(gt > 0, axis=(1, 2))
    z = int(np.argmax(tumor_per_slice if tumor_per_slice.max() else liver_per_slice))
    base = maps["intensity"][z]
    valid = maps["valid"][z]
    lo, hi = np.percentile(base[valid], [1, 99])
    gt_slice = gt[z]
    prediction = np.argmax(probabilities, axis=0).astype(np.int16)
    pred_slice = prediction[z]
    organ_probability = probabilities[1, z] + probabilities[2, z]
    tumor_probability = probabilities[2, z]
    confidence = np.max(probabilities[:, z], axis=0)
    error = _tumor_error(gt_slice, pred_slice)

    fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.8))
    for ax in axes.flat:
        _display_base(ax, base, lo, hi)

    axes[0, 0].imshow(_segmentation_rgba(gt_slice))
    axes[0, 0].set_title(
        f"Ground truth: liver=cyan, tumor=pink (tumor voxels={int((gt_slice == 2).sum())})"
    )

    liver_im = axes[0, 1].imshow(
        organ_probability, cmap="Blues", vmin=0, vmax=1, alpha=0.75
    )
    _draw_contours(axes[0, 1], gt_slice)
    axes[0, 1].set_title("Organ probability P(liver or tumor)")
    fig.colorbar(liver_im, ax=axes[0, 1], fraction=0.046, pad=0.03)

    tumor_im = axes[0, 2].imshow(
        tumor_probability, cmap="magma", vmin=0, vmax=1, alpha=0.75
    )
    _draw_contours(axes[0, 2], gt_slice)
    axes[0, 2].set_title("Tumor probability P(tumor)")
    fig.colorbar(tumor_im, ax=axes[0, 2], fraction=0.046, pad=0.03)

    axes[1, 0].imshow(_segmentation_rgba(pred_slice))
    axes[1, 0].set_title("Stage prediction: liver=cyan, tumor=pink")

    error_cmap = ListedColormap(
        [(0, 0, 0, 0), (0.10, 0.75, 0.25, 0.75), (1.0, 0.55, 0.0, 0.85), (0.85, 0.0, 0.65, 0.85)]
    )
    error_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], error_cmap.N)
    axes[1, 1].imshow(np.ma.masked_where(error == 0, error), cmap=error_cmap, norm=error_norm)
    axes[1, 1].set_title(
        "Tumor on this slice: "
        f"TP={int((error == 1).sum())}, FP={int((error == 2).sum())}, "
        f"FN={int((error == 3).sum())}"
    )

    conf_im = axes[1, 2].imshow(confidence, cmap="viridis", vmin=1 / 3, vmax=1)
    axes[1, 2].set_title("Maximum class confidence")
    fig.colorbar(conf_im, ax=axes[1, 2], fraction=0.046, pad=0.03)

    native_shape = maps["decoder_native_shapes"][stage]
    fig.suptitle(
        f"{case} | {stage} | native logits {native_shape} | displayed at 128^3 | axial z={z}",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_decoder_reconstruction(
    case: str,
    maps: dict[str, Any],
    output: Path,
) -> None:
    gt = maps["semantics"]
    tumor_per_slice = np.sum(gt == 2, axis=(1, 2))
    liver_per_slice = np.sum(gt > 0, axis=(1, 2))
    z = int(np.argmax(tumor_per_slice if tumor_per_slice.max() else liver_per_slice))
    base = maps["intensity"][z]
    valid = maps["valid"][z]
    lo, hi = np.percentile(base[valid], [1, 99])
    fig, axes = plt.subplots(len(DECODER_HEADS), 4, figsize=(13.2, 16.0))
    for row, stage in enumerate(DECODER_HEADS):
        probabilities = maps["decoder_probabilities"][stage]
        prediction = np.argmax(probabilities, axis=0).astype(np.int16)
        organ_probability = probabilities[1, z] + probabilities[2, z]
        tumor_probability = probabilities[2, z]
        error = _tumor_error(gt[z], prediction[z])
        for col in range(4):
            _display_base(axes[row, col], base, lo, hi)
        axes[row, 0].imshow(
            organ_probability, cmap="Blues", vmin=0, vmax=1, alpha=0.75
        )
        axes[row, 1].imshow(
            tumor_probability, cmap="magma", vmin=0, vmax=1, alpha=0.75
        )
        _draw_contours(axes[row, 0], gt[z])
        _draw_contours(axes[row, 1], gt[z])
        axes[row, 2].imshow(_segmentation_rgba(prediction[z]))
        error_cmap = ListedColormap(
            [(0, 0, 0, 0), (0.10, 0.75, 0.25, 0.75), (1.0, 0.55, 0.0, 0.85), (0.85, 0.0, 0.65, 0.85)]
        )
        error_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], error_cmap.N)
        axes[row, 3].imshow(
            np.ma.masked_where(error == 0, error), cmap=error_cmap, norm=error_norm
        )
        axes[row, 3].text(
            0.02,
            0.98,
            f"TP {int((error == 1).sum())}  FP {int((error == 2).sum())}  FN {int((error == 3).sum())}",
            transform=axes[row, 3].transAxes,
            ha="left",
            va="top",
            fontsize=8,
            color="black",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 2},
        )
        native = maps["decoder_native_shapes"][stage]
        axes[row, 0].text(
            0.02,
            0.98,
            f"{stage}\n{native[-3:]} -> 128^3",
            transform=axes[row, 0].transAxes,
            ha="left",
            va="top",
            fontsize=9,
            color="black",
            bbox={"facecolor": "white", "edgecolor": "#555555", "alpha": 0.88, "pad": 3},
        )
    for col, title in enumerate(
        (
            "Organ probability",
            "Tumor probability",
            "Predicted mask",
            "Tumor error: TP green / FP orange / FN magenta",
        )
    ):
        axes[0, col].set_title(title, fontsize=10)
    fig.suptitle(
        f"{case}: semantic reconstruction through the decoder (same axial slice z={z})",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_visualization_only_report(
    output_dir: Path,
    checkpoint: dict[str, Any],
    case: str,
    maps: dict[str, Any],
    pngs: list[Path],
    args: argparse.Namespace,
) -> None:
    layer_shapes = {
        layer: maps[f"{layer}__shape"] for layer in LAYER_MODULES
    }
    decoder_shapes = maps["decoder_native_shapes"]
    summary = {
        "status": "complete_visualization_only",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": checkpoint,
        "data": {
            "preprocessed_dir": str(args.preprocessed_dir.resolve()),
            "case": case,
            "patch_policy": "one tumor-centered 128^3 training-space patch",
            "patch_label_voxel_counts": maps["label_voxel_counts"],
        },
        "layers": layer_shapes,
        "decoder_semantic_heads": decoder_shapes,
        "visualization_definition": {
            "first_panel": "channel RMS activation energy over CT",
            "remaining_panels": "up to 8 channels with largest spatial standard deviation",
            "cyan_contour": "liver including tumor",
            "pink_contour": "tumor",
            "decoder_pages": (
                "deep-supervision softmax probabilities, predicted mask, confidence, "
                "and tumor TP/FP/FN at each reconstruction scale"
            ),
        },
        "artifacts": {
            "summary_json": True,
            "text_report": True,
            "visualization_png_count": len(pngs),
            "visualizations": [str(p.resolve()) for p in pngs],
        },
        "evidence_boundary": (
            "These images show activation patterns from one case. They do not by "
            "themselves prove a universal or strictly ordered feature hierarchy."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "MedNeXt_MLA 单病例逐层特征可视化",
        "================================",
        "状态：complete_visualization_only",
        f"病例：{case}",
        f"checkpoint：{checkpoint['path']}",
        f"trainer / fold / epoch：{checkpoint['trainer']} / {args.fold} / {checkpoint['current_epoch']}",
        f"逐层 PNG：{len(pngs)}",
        f"裁剪块标签体素数：{maps['label_voxel_counts']}",
        "",
        "层与实际特征形状：",
        *[f"- {layer}: {shape}" for layer, shape in layer_shapes.items()],
        "",
        "Decoder 逐级语义输出：",
        *[f"- {stage}: logits {shape}" for stage, shape in decoder_shapes.items()],
        "",
        "每张图左上为该层所有通道的 RMS 激活能量，其余为该层空间变化最大的通道。",
        "Decoder 页面直接显示肝脏/肿瘤概率、预测掩膜、置信度和肿瘤 TP/FP/FN。",
        "青色为肝脏，粉色为肿瘤；错误图中绿色=TP、橙色=FP、洋红色=FN。",
        "原始通道热图不能单独证明严格层级，Decoder 深监督概率具有明确类别语义。",
    ]
    (output_dir / "feature_hierarchy_report.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def write_full_report(
    output_dir: Path,
    checkpoint: dict[str, Any],
    split: dict[str, list[str]],
    scores: dict[str, dict[str, Any]],
    hierarchy: dict[str, Any],
    args: argparse.Namespace,
    pngs: list[Path],
) -> None:
    summary = {
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": checkpoint,
        "data": {
            "preprocessed_dir": str(args.preprocessed_dir.resolve()),
            "fold": args.fold,
            "train_cases": split["train"],
            "val_cases": split["val"],
            "samples_per_case_per_layer": args.samples_per_case,
            "patch_policy": "one tumor-centered 128^3 training-space patch per case",
        },
        "target_definitions": {
            "intensity": "nnU-Net-normalized CT intensity",
            "edge": "3-D gradient magnitude",
            "texture": "local standard deviation, Gaussian sigma=1.5",
            "local_structure": "absolute 3-D Laplacian-of-Gaussian response, sigma=2",
            "organ_shape": "signed distance to liver boundary, clipped to +/-32 voxels",
            "semantics": "background/liver/tumor class",
        },
        "probe": {
            "frozen_network": True,
            "case_level_train_val_separation": True,
            "pca_components_max": args.pca_components,
            "continuous_model": "standardize + PCA + Ridge(alpha=10)",
            "semantic_model": "standardize + PCA + class-balanced multinomial logistic regression",
        },
        "scores": scores,
        "hierarchy_summary": hierarchy,
        "artifacts": {
            "summary_json": True,
            "text_report": True,
            "visualization_png_count": len(pngs),
            "visualizations": [str(p.resolve()) for p in pngs],
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "MedNeXt_MLA 表征层级审计",
        "=========================",
        "状态：complete（表征审计完成，不等同于分割测试完成）",
        f"checkpoint：{checkpoint['path']}",
        f"trainer / fold / epoch：{checkpoint['trainer']} / {args.fold} / {checkpoint['current_epoch']}",
        f"探针训练病例 / 独立验证病例：{len(split['train'])} / {len(split['val'])}",
        f"可视化 PNG：{len(pngs)}",
        "",
        "每个目标的最佳可解码层：",
    ]
    for target in TARGET_ORDER:
        peak = hierarchy["peak_layer_by_target"][target]
        lines.append(f"- {target}: {peak['layer']} ({peak['score']:.4f})")
    lines.extend(
        [
            "",
            "MLA 相对卷积 bottleneck 的分数变化：",
            *[
                f"- {target}: {hierarchy['mla_minus_conv_bottleneck'][target]:+.4f}"
                for target in TARGET_ORDER
            ],
            "",
            "证据边界：线性探针只说明信息能否从冻结特征中被简单解码；它不证明某层只表示某概念，",
            "也不证明网络内部严格按像素→边缘→纹理→结构→形态→语义单向运行。skip connection",
            "会把低层细节送入 decoder，MLA 后仍保留残差输入，因此多类信息通常会共存。",
        ]
    )
    (output_dir / "feature_hierarchy_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.model_dir = args.model_dir.resolve()
    args.preprocessed_dir = args.preprocessed_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    checkpoint = inspect_checkpoint(args.model_dir, args.fold, args.checkpoint)
    split = load_split(
        args.preprocessed_dir, args.fold, args.train_cases, args.val_cases, args.seed
    )
    if args.inspect_only:
        write_inspection(args.output_dir, checkpoint, split, args)
        print(f"Inspection complete: {args.output_dir}")
        return 0

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable: {args.device}")
    output_dir = args.output_dir
    viz_dir = output_dir / "test_viz"
    viz_dir.mkdir(parents=True, exist_ok=True)
    network, configured_patch = load_network(
        args.model_dir, args.fold, args.checkpoint, device
    )
    patch_size = tuple(int(v) for v in configured_patch)
    if patch_size != (128, 128, 128):
        raise RuntimeError(f"Expected audited 128^3 plan, got patch_size={patch_size}")
    data_dir = args.preprocessed_dir / "nnUNetPlans_3d_fullres"
    rng = np.random.default_rng(args.seed)
    if args.visualize_case:
        available_cases = set(split["train"]) | set(split["val"])
        case_data = data_dir / f"{args.visualize_case}.b2nd"
        case_seg = data_dir / f"{args.visualize_case}_seg.b2nd"
        if not case_data.is_file() or not case_seg.is_file():
            raise FileNotFoundError(
                f"Missing preprocessed data/seg for visualization case: {args.visualize_case}"
            )
        _, maps = extract_case_samples(
            network,
            data_dir,
            args.visualize_case,
            patch_size,
            device,
            args.samples_per_case,
            rng,
            keep_maps=True,
        )
        if maps is None:
            raise RuntimeError("Internal error: visualization maps were not retained")
        if maps["label_voxel_counts"].get(2, 0) == 0:
            print(
                "WARNING: the selected 128^3 patch contains no tumor label (class 2); "
                "tumor TP/FP/FN panels may be empty.",
                file=sys.stderr,
                flush=True,
            )
        pngs = [viz_dir / "00_all_layers_activation_energy.png"]
        plot_case_maps(args.visualize_case, maps, pngs[0])
        for index, layer in enumerate(LAYER_MODULES, start=1):
            output = viz_dir / f"{index:02d}_{layer}.png"
            plot_layer_channels(args.visualize_case, layer, maps, output)
            pngs.append(output)
        decoder_overview = viz_dir / "09_decoder_reconstruction_overview.png"
        plot_decoder_reconstruction(args.visualize_case, maps, decoder_overview)
        pngs.append(decoder_overview)
        for index, stage in enumerate(DECODER_HEADS, start=10):
            output = viz_dir / f"{index:02d}_{stage}_prediction.png"
            plot_decoder_stage(args.visualize_case, stage, maps, output)
            pngs.append(output)
        write_visualization_only_report(
            output_dir, checkpoint, args.visualize_case, maps, pngs, args
        )
        split_membership = (
            "selected probe subset"
            if args.visualize_case in available_cases
            else "available preprocessed case outside selected probe subset"
        )
        print(
            f"Layer visualization complete ({split_membership}): {output_dir}",
            flush=True,
        )
        return 0
    samples = {
        subset: {layer: [] for layer in LAYER_MODULES} for subset in ("train", "val")
    }
    first_maps = None
    first_val_case = None
    for subset in ("train", "val"):
        for case in split[subset]:
            print(f"[{subset}] extracting {case}", flush=True)
            case_samples, maps = extract_case_samples(
                network,
                data_dir,
                case,
                patch_size,
                device,
                args.samples_per_case,
                rng,
                keep_maps=subset == "val" and first_maps is None,
            )
            for layer in LAYER_MODULES:
                samples[subset][layer].append(case_samples[layer])
            if maps is not None:
                first_maps = maps
                first_val_case = case
    scores = fit_probes(samples["train"], samples["val"], args.pca_components, args.seed)
    hierarchy = summarize_hierarchy(scores)
    pngs = [viz_dir / "probe_score_matrix.png"]
    plot_score_matrix(scores, pngs[0])
    if first_maps is not None and first_val_case is not None:
        pngs.append(viz_dir / f"{first_val_case}_activation_energy.png")
        plot_case_maps(first_val_case, first_maps, pngs[-1])
    write_full_report(output_dir, checkpoint, split, scores, hierarchy, args, pngs)
    print(f"Audit complete: {output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
