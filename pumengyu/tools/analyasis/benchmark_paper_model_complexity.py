#!/usr/bin/env python3
"""Benchmark the paper's core architectures under one controlled protocol.

Reports exact parameter counts, PyTorch-supported FLOPs, AMP forward latency,
and peak allocated CUDA memory for a single 3D patch. This is an architecture
benchmark: it excludes preprocessing, sliding-window stitching, NIfTI I/O, and
postprocessing.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.flop_counter import FlopCounterMode

from pumengyu.architectures.efficient_mednext_official import (
    build_efficient_mednext_large_official,
)
from pumengyu.architectures.mednext import (
    build_mednext_large,
    build_mednext_large_mha,
    build_mednext_large_mla,
)


def build_models():
    common = dict(
        num_input_channels=1,
        num_output_channels=3,
        enable_deep_supervision=False,
    )
    return {
        "EfficientMedNeXt-L": lambda: build_efficient_mednext_large_official(**common),
        "MedNeXt": lambda: build_mednext_large(**common),
        "MedNeXt_MLA": lambda: build_mednext_large_mla(
            **common, mla_use_moe=False
        ),
        "MedNeXt_MHA": lambda: build_mednext_large_mha(
            **common, mha_use_moe=False
        ),
        "MedNeXt_MLA_MoE": lambda: build_mednext_large_mla(
            **common, mla_use_moe=True
        ),
        "MedNeXt_MHA_MoE": lambda: build_mednext_large_mha(
            **common, mha_use_moe=True
        ),
    }


def disable_checkpointing(model: torch.nn.Module) -> None:
    if hasattr(model, "outside_block_checkpointing"):
        model.outside_block_checkpointing = False
    if hasattr(model, "checkpoint_style"):
        model.checkpoint_style = None


def count_flops(model: torch.nn.Module, x: torch.Tensor) -> int:
    # inference_mode suppresses TorchDispatch in torch 2.3 and would yield zero.
    with torch.no_grad():
        with FlopCounterMode(display=False) as counter:
            model(x)
    return int(counter.get_total_flops())


def benchmark_latency(
    model: torch.nn.Module,
    x: torch.Tensor,
    warmup: int,
    repeats: int,
) -> tuple[list[float], int]:
    with torch.inference_mode():
        for _ in range(warmup):
            with torch.autocast("cuda", dtype=torch.float16):
                model(x)
        torch.cuda.synchronize()

        torch.cuda.reset_peak_memory_stats()
        elapsed_ms = []
        for _ in range(repeats):
            start = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.float16):
                model(x)
            torch.cuda.synchronize()
            elapsed_ms.append((time.perf_counter() - start) * 1000)
        peak_bytes = int(torch.cuda.max_memory_allocated())
    return elapsed_ms, peak_bytes


def add_rank(rows: list[dict], field: str, rank_field: str) -> None:
    ordered = sorted(rows, key=lambda row: row[field])
    for rank, row in enumerate(ordered, start=1):
        row[rank_field] = rank


def render_markdown(payload: dict) -> str:
    rows = payload["results"]
    lines = [
        "# 核心模型复杂度与单 Patch 前向 Benchmark",
        "",
        f"> 测量时间：{payload['measured_at']}",
        f"> GPU：{payload['gpu']}",
        f"> PyTorch/CUDA：{payload['torch_version']} / {payload['cuda_version']}",
        f"> 输入：`{payload['input_shape']}`；deep supervision=False；AMP=FP16；torch.compile=False。",
        "> 按当前 Trainer 架构构建网络，不加载 checkpoint；权重数值不改变本表的参数量、执行图和当前全专家 MoE 路径。",
        f"> 延迟：warm-up {payload['warmup']} 次后重复 {payload['repeats']} 次；数值为均值±标准差。",
        "> FLOPs 由 PyTorch FlopCounterMode 统计其支持的 Conv/Linear/MatMul 运算；不等同于实际运行时间。",
        "> 峰值显存仅为网络单 patch 前向的 CUDA allocated memory，不包含预处理、滑窗拼接、NIfTI I/O 或训练反向传播。",
        "",
        "| 方法 | 参数量（排名） | FLOPs（排名） | 单Patch前向时间（排名） | 峰值显存（排名） |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda item: item["latency_rank"]):
        lines.append(
            f"| {row['method']} | {row['params_m']:.3f}M（#{row['params_rank']}） | "
            f"{row['flops_g']:.2f}G（#{row['flops_rank']}） | "
            f"{row['latency_mean_ms']:.2f}±{row['latency_std_ms']:.2f} ms（#{row['latency_rank']}） | "
            f"{row['peak_memory_gib']:.2f} GiB（#{row['peak_memory_rank']}） |"
        )
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "- 参数量是精确可训练参数总数；排名越小，参数越少。",
            "- FLOPs 和前向时间排名越小，理论计算或当前环境中的单 patch 前向越低。",
            "- 该时间不是完整 CT 病例的端到端推理时间；nnU-Net sliding-window 的窗口数随病例尺寸变化。",
            "- 当前 MoE 实现会先计算全部 routed experts 再选择 Top-2，因此参数容量增加并未转化为官方稀疏 MoE 式的计算节省。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(
            "pumengyu/notes/paper/statistics/core_model_complexity_benchmark.json"
        ),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path(
            "pumengyu/notes/paper/statistics/core_model_complexity_benchmark.md"
        ),
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the standardized latency benchmark")
    torch.cuda.set_device(args.device)
    device = torch.device(f"cuda:{args.device}")
    torch.backends.cudnn.benchmark = True
    x = torch.randn((1, 1, 128, 128, 128), device=device)

    results = []
    for name, builder in build_models().items():
        model = builder().eval()
        disable_checkpointing(model)
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model = model.to(device)

        flops = count_flops(model, x)
        timings, peak_bytes = benchmark_latency(
            model, x, args.warmup, args.repeats
        )
        results.append(
            {
                "method": name,
                "params": params,
                "params_m": params / 1e6,
                "flops": flops,
                "flops_g": flops / 1e9,
                "latency_mean_ms": statistics.fmean(timings),
                "latency_std_ms": statistics.stdev(timings),
                "latency_median_ms": statistics.median(timings),
                "peak_memory_bytes": peak_bytes,
                "peak_memory_gib": peak_bytes / (1024**3),
            }
        )
        del model
        torch.cuda.empty_cache()

    add_rank(results, "params", "params_rank")
    add_rank(results, "flops", "flops_rank")
    add_rank(results, "latency_mean_ms", "latency_rank")
    add_rank(results, "peak_memory_bytes", "peak_memory_rank")

    payload = {
        "measured_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "gpu": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "input_shape": "1×1×128×128×128",
        "checkpoint_loaded": False,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "results": results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    )
    args.output_md.write_text(render_markdown(payload))
    print(args.output_md)


if __name__ == "__main__":
    main()
