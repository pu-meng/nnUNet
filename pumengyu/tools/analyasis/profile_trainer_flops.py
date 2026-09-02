#!/usr/bin/env python3
"""Measure FLOPs for any nnU-Net trainer without modifying a training checkpoint.

This is deliberately an offline command. Running an extra 3D dummy forward in
the live training process can add substantial initialization cost or trigger an
OOM. The result is stored beside the trainer fold as ``offline_complexity.json``
and is also merged into ``resource_usage.json`` when that file already exists.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import torch
from torch._dynamo import OptimizedModule
from torch.utils.flop_counter import FlopCounterMode

from nnunetv2.run.run_training import get_trainer_from_args


def unwrap_network(network: torch.nn.Module) -> torch.nn.Module:
    if hasattr(network, "module"):
        network = network.module
    if isinstance(network, OptimizedModule):
        network = network._orig_mod
    return network


def disable_checkpointing(model: torch.nn.Module) -> None:
    if hasattr(model, "outside_block_checkpointing"):
        model.outside_block_checkpointing = False
    if hasattr(model, "checkpoint_style"):
        model.checkpoint_style = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("configuration")
    parser.add_argument("fold")
    parser.add_argument("-tr", "--trainer", required=True)
    parser.add_argument("-p", "--plans", default="nnUNetPlans")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--meta",
        action="store_true",
        help="Count the execution graph on meta tensors without occupying a GPU",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fold = int(args.fold) if args.fold.isdigit() else args.fold
    if not args.meta and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the standardized 3D FLOP measurement")

    # Complexity is measured on the eager graph for consistency across trainers.
    os.environ["nnUNet_compile"] = "0"
    if args.meta:
        device = torch.device("cpu")
    else:
        torch.cuda.set_device(args.device)
        device = torch.device(f"cuda:{args.device}")
    trainer = get_trainer_from_args(
        args.dataset,
        args.configuration,
        fold,
        args.trainer,
        args.plans,
        continue_training=False,
        device=device,
    )
    trainer.initialize()
    trainer.set_deep_supervision_enabled(False)

    model = unwrap_network(trainer.network).eval()
    disable_checkpointing(model)
    patch_size = [int(i) for i in trainer.configuration_manager.patch_size]
    execution_device = torch.device("meta") if args.meta else device
    model = model.to(execution_device)
    x = torch.empty((1, int(trainer.num_input_channels), *patch_size), device=execution_device)

    with torch.no_grad(), FlopCounterMode(display=False) as counter:
        model(x)
    flops = int(counter.get_total_flops())
    params_total = int(sum(p.numel() for p in model.parameters()))
    params_trainable = int(sum(p.numel() for p in model.parameters() if p.requires_grad))

    payload = {
        "measured_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "dataset": trainer.plans_manager.dataset_name,
        "trainer": trainer.__class__.__name__,
        "configuration": trainer.configuration_name,
        "fold": trainer.fold,
        "checkpoint_loaded": False,
        "input_shape": [1, int(trainer.num_input_channels), *patch_size],
        "deep_supervision": False,
        "amp": False,
        "torch_compile": False,
        "counter": "torch.utils.flop_counter.FlopCounterMode",
        "execution_device": execution_device.type,
        "gpu": None if args.meta else torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "parameters_total": params_total,
        "parameters_trainable": params_trainable,
        "flops": flops,
        "flops_g": flops / 1e9,
    }

    output_folder = Path(trainer.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    output_path = output_folder / "offline_complexity.json"
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")

    resource_path = output_folder / "resource_usage.json"
    if resource_path.is_file():
        resource = json.loads(resource_path.read_text())
        resource.setdefault("model", {})["flops"] = flops
        resource["model"]["flops_status"] = "measured by offline_complexity.json"
        resource["model"]["flops_protocol"] = {
            key: payload[key]
            for key in (
                "measured_at",
                "input_shape",
                "deep_supervision",
                "amp",
                "torch_compile",
                "counter",
                "execution_device",
                "gpu",
                "torch_version",
                "cuda_version",
            )
        }
        resource_path.write_text(json.dumps(resource, ensure_ascii=False, indent=2) + "\n")

    print(output_path)


if __name__ == "__main__":
    main()
