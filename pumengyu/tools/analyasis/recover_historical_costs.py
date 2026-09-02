#!/usr/bin/env python3
"""Recover trustworthy cost fields from old nnU-Net checkpoints and logs."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from datetime import datetime
from pathlib import Path

import torch

from pumengyu.tools.analyasis.experiment_cost_report import upsert_cost_section


TIMESTAMP = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}): (.*)$")
INTERNAL_START = re.compile(r"\[InternalTest\] 开始推理 (\d+) 个 test cases")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _parse_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() == "true"
    return None


def recover_inference(fold_dir: Path, gpu_count: int | None, gpu_names: list[str]) -> list[dict]:
    recovered = []
    for log_path in sorted(fold_dir.glob("training_log_*.txt")):
        start = end = None
        n_cases = None
        for raw_line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
            match = TIMESTAMP.match(raw_line.strip())
            if not match:
                continue
            timestamp = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S")
            message = match.group(2)
            start_match = INTERNAL_START.search(message)
            if start_match:
                start = timestamp
                n_cases = int(start_match.group(1))
            if start is not None and "[InternalTest] 推理完成" in message:
                end = timestamp
        if start is not None and end is not None and n_cases:
            duration = (end - start).total_seconds()
            recovered.append({
                "scope": "historical_internal_test",
                "source": f"{log_path.name}: InternalTest start -> inference complete",
                "n_cases": n_cases,
                "duration_s": duration,
                "seconds_per_case": duration / n_cases,
                "gpu_count": gpu_count,
                "gpu_names": gpu_names,
                "gpu_evidence": "GPU 型号来自同目录后续 debug.json；该次推理日志未单独记录型号",
                "peak_allocated": None,
                "peak_reserved": None,
                "protocol": "preprocessed case load + sliding-window inference + asynchronous NIfTI export; excludes metrics and visualization",
            })
    return recovered


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("fold_dir", type=Path)
    parser.add_argument("--parameters-total", type=int)
    parser.add_argument("--flops", type=int)
    parser.add_argument("--flops-input-shape", type=int, nargs="+")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--scope", default="validation")
    args = parser.parse_args()

    fold_dir = args.fold_dir.resolve()
    final_path = fold_dir / "checkpoint_final.pth"
    if not final_path.is_file():
        raise FileNotFoundError(f"历史训练时间回收要求 checkpoint_final.pth: {final_path}")

    checkpoint = torch.load(final_path, map_location="cpu", weights_only=False)
    logging = checkpoint.get("logging", {})
    starts = logging.get("epoch_start_timestamps", [])
    ends = logging.get("epoch_end_timestamps", [])
    durations = [float(end - start) for start, end in zip(starts, ends)]
    if not durations:
        raise RuntimeError(f"checkpoint_final.pth 没有 epoch 起止时间: {final_path}")

    debug = _read_json(fold_dir / "debug.json")
    debug_is_later = (fold_dir / "debug.json").is_file() and (fold_dir / "debug.json").stat().st_mtime > final_path.stat().st_mtime
    is_ddp = _parse_bool(debug.get("is_ddp"))
    gpu_count = 1 if is_ddp is False else None
    gpu_name = debug.get("gpu_name")
    gpu_names = [gpu_name] if gpu_name else []
    compile_enabled = None if debug_is_later else debug.get("network") == "OptimizedModule"

    warnings = []
    if debug_is_later:
        warnings.append(
            "debug.json 晚于 checkpoint_final.pth，GPU 型号/数量来自同目录后续短试跑；原始 1000-epoch 训练未单独保存该硬件字段"
        )
    best_path = fold_dir / "checkpoint_best.pth"
    if best_path.is_file():
        best = torch.load(best_path, map_location="cpu", weights_only=False)
        if int(best.get("current_epoch", -1)) < int(checkpoint.get("current_epoch", -1)):
            warnings.append(
                f"checkpoint_best.pth current_epoch={best.get('current_epoch')} 早于 "
                f"checkpoint_final.pth current_epoch={checkpoint.get('current_epoch')}；历史训练成本以 final 为准，best 疑似被短试跑覆盖"
            )

    offline = _read_json(fold_dir / "offline_complexity.json")
    parameters_total = args.parameters_total or offline.get("parameters_total")
    flops = args.flops or offline.get("flops")
    payload = {
        "schema_version": 1,
        "recovered_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "model": {
            "parameters_total": parameters_total,
            "flops": flops,
            "input_shape": args.flops_input_shape or offline.get("input_shape"),
            "source": (
                "当前 trainer 架构 + PyTorch FlopCounterMode meta 执行图；未加载 checkpoint"
                if args.flops else offline.get("counter", "historical recovery")
            ),
            "status": "parameters/FLOPs are architecture properties; FLOPs require a separate fixed-input benchmark",
        },
        "training": {
            "checkpoint": "checkpoint_final.pth logging.epoch_start_timestamps/epoch_end_timestamps",
            "epochs": len(durations),
            "epoch_time_total_s": sum(durations),
            "wall_span_s": float(ends[-1] - starts[0]),
            "epoch_time_median_s": statistics.median(durations),
            "epoch_time_mean_s": statistics.fmean(durations),
            "gpu_count": gpu_count,
            "gpu_names": gpu_names,
            "gpu_evidence": "同目录后续 debug.json；原始完整训练未单独保存" if debug_is_later else "debug.json",
            "torch_compile": compile_enabled,
            "peak_allocated": None,
            "peak_reserved": None,
        },
        "inference": recover_inference(fold_dir, gpu_count, gpu_names),
        "warnings": warnings,
    }
    output_path = fold_dir / "resource_usage_recovered.json"
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output_path)

    if args.report is not None:
        upsert_cost_section(args.report, fold_dir, args.scope)
        print(args.report)


if __name__ == "__main__":
    main()
