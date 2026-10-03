#!/usr/bin/env python3
"""Read-only structural audit for the MedNeXt-L MHA+MoE network.

This tool does not create a trainer, load data, start training, or touch a
checkpoint. It builds the same architecture used by
``nnUNetTrainer_MedNeXt_MHA_MoE`` and writes an auditable module inventory.
Use ``--forward`` only when a controlled shape trace is desired.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch import nn

from pumengyu.architectures.mednext import build_mednext_large_mha


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("pumengyu/notes/模型结构图"))
    parser.add_argument("--forward", action="store_true", help="Run one CPU shape trace; can be slow for a 128^3 patch.")
    parser.add_argument("--patch-size", nargs=3, type=int, default=[128, 128, 128])
    return parser.parse_args()


def shape_of(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return list(value.shape)
    if isinstance(value, (list, tuple)):
        return [shape_of(item) for item in value]
    if isinstance(value, dict):
        return {str(key): shape_of(item) for key, item in value.items()}
    return type(value).__name__


def module_row(name: str, module: nn.Module) -> dict[str, Any]:
    own_parameters = sum(parameter.numel() for parameter in module.parameters(recurse=False))
    total_parameters = sum(parameter.numel() for parameter in module.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
    return {
        "name": name or "<root>",
        "class": module.__class__.__name__,
        "depth": 0 if not name else name.count(".") + 1,
        "children": sum(1 for _ in module.children()),
        "own_parameters": own_parameters,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "has_checkpoint_attribute": any(
            hasattr(module, attribute) for attribute in ("checkpoint_style", "outside_block_checkpointing")
        ),
    }


def build_audit(args: argparse.Namespace) -> dict[str, Any]:
    model = build_mednext_large_mha(
        num_input_channels=1,
        num_output_channels=3,
        enable_deep_supervision=False,
        mha_num_heads=8,
        mha_num_blocks=2,
        mha_mlp_ratio=4,
        mha_use_moe=True,
    ).cpu().eval()

    rows = [module_row(name, module) for name, module in model.named_modules()]
    norm_rows = [row for row in rows if "norm" in row["class"].lower()]
    checkpoint_rows = [row for row in rows if row["has_checkpoint_attribute"]]
    class_counts = Counter(row["class"] for row in rows)

    payload: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "trainer_architecture": "nnUNetTrainer_MedNeXt_MHA_MoE",
        "builder": "pumengyu.architectures.mednext.build_mednext_large_mha",
        "input_channels": 1,
        "output_channels": 3,
        "mha_num_heads": 8,
        "mha_num_blocks": 2,
        "mha_mlp_ratio": 4,
        "mha_use_moe": True,
        "forward_executed": False,
        "parameters_total": sum(parameter.numel() for parameter in model.parameters()),
        "parameters_trainable": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "module_class_counts": dict(sorted(class_counts.items())),
        "normalization_modules": norm_rows,
        "checkpoint_related_modules": checkpoint_rows,
        "modules": rows,
        "improvement_prompts": [
            {
                "question": "哪些模块占据最多参数和计算？",
                "evidence_to_collect": "controlled forward/backward profiler, not parameter count alone",
            },
            {
                "question": "checkpoint 重计算是否值得保留？",
                "evidence_to_collect": "same 20-update benchmark with checkpoint on/off, peak memory and sec/update",
            },
            {
                "question": "MoE 是否真的跳过了未选专家？",
                "evidence_to_collect": "expert forward call counts and per-module profiler time",
            },
            {
                "question": "归一化层保留了还是抹除了域强度信息？",
                "evidence_to_collect": "feature hooks before/after norm plus intensity probe on fixed HCC/LiTS cases",
            },
        ],
    }

    if args.forward:
        trace: list[dict[str, Any]] = []

        def hook(name: str):
            def record(module: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
                trace.append({
                    "name": name or "<root>",
                    "class": module.__class__.__name__,
                    "input_shape": shape_of(inputs),
                    "output_shape": shape_of(output),
                })
            return record

        handles = [module.register_forward_hook(hook(name)) for name, module in model.named_modules()]
        try:
            with torch.inference_mode():
                model(torch.zeros((1, 1, *args.patch_size), dtype=torch.float32))
        finally:
            for handle in handles:
                handle.remove()
        payload["forward_executed"] = True
        payload["forward_input_shape"] = [1, 1, *args.patch_size]
        payload["shape_trace"] = trace
    return payload


def write_outputs(payload: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "model_architecture.json"
    existing = {}
    if json_path.exists():
        try:
            existing = json.loads(json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}
    if "layer_coverage" in existing:
        merged = dict(existing)
        merged["static_audit"] = payload
    else:
        merged = payload
    json_path.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    markdown_path = output_dir / "model_architecture.md"
    if markdown_path.exists() and "layer_coverage" in existing:
        base = markdown_path.read_text(encoding="utf-8").split("\n## 最新静态审计", 1)[0].rstrip()
        audit_markdown = (
            "\n\n## 最新静态审计\n\n"
            f"- 总参数量：`{payload['parameters_total']:,}`\n"
            f"- 可训练参数量：`{payload['parameters_trainable']:,}`\n"
            f"- Forward shape trace：`{'已执行' if payload['forward_executed'] else '未执行'}`\n"
        )
        markdown_path.write_text(base + audit_markdown + "\n", encoding="utf-8")
        return
    lines = [
        "# MedNeXt-MHA-MoE 模型审计",
        "",
        f"- Trainer 架构：`{payload['trainer_architecture']}`",
        f"- 总参数量：`{payload['parameters_total']:,}`",
        f"- 可训练参数量：`{payload['parameters_trainable']:,}`",
        f"- Forward shape trace：`{'已执行' if payload['forward_executed'] else '未执行'}`",
        "",
        "## 归一化层",
        "",
        "| 名称 | 类型 | 参数量 | checkpoint 属性 |",
        "|---|---|---:|---|",
    ]
    for row in payload["normalization_modules"]:
        lines.append(f"| `{row['name']}` | `{row['class']}` | {row['total_parameters']:,} | {row['has_checkpoint_attribute']} |")
    lines.extend(["", "## 可验证的研究问题", ""])
    for item in payload["improvement_prompts"]:
        lines.append(f"- **{item['question']}**：{item['evidence_to_collect']}")
    (output_dir / "model_architecture.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    payload = build_audit(args)
    write_outputs(payload, args.output_dir)
    print(args.output_dir / "mednext_mha_moe_audit.md")


if __name__ == "__main__":
    main()
