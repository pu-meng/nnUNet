"""Render theoretical and measured experiment cost into text reports."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any


SECTION_START = "# >>> EXPERIMENT_COSTS_V1"
SECTION_END = "# <<< EXPERIMENT_COSTS_V1"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _duration(seconds: float | None) -> str:
    if seconds is None:
        return "N/A"
    seconds = float(seconds)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    if hours:
        return f"{hours} h {minutes:02d} min {secs:04.1f} s"
    if minutes:
        return f"{minutes} min {secs:04.1f} s"
    return f"{secs:.2f} s"


def _gib(value: int | float | None) -> str:
    return "N/A（历史未记录）" if value is None else f"{float(value) / 1024**3:.2f} GiB"


def _millions(value: int | float | None) -> str:
    return "N/A（尚未离线统计）" if value is None else f"{float(value) / 1e6:.3f} M"


def _flops(value: int | float | None) -> str:
    return "N/A（尚未离线测量）" if value is None else f"{float(value) / 1e9:.2f} G"


def _load_training(fold_dir: Path) -> tuple[dict[str, Any], str]:
    live = _read_json(fold_dir / "resource_usage.json")
    if live:
        epochs = live.get("epochs", [])
        durations = [float(item["epoch_duration_s"]) for item in epochs if item.get("epoch_duration_s") is not None]
        run = live.get("run", {})
        summary = live.get("summary", {})
        return {
            "checkpoint": "resource_usage.json（训练时自动记录）",
            "epochs": len(epochs),
            "epoch_time_total_s": summary.get("total_epoch_time_s"),
            "wall_span_s": None,
            "epoch_time_median_s": statistics.median(durations) if durations else None,
            "epoch_time_mean_s": statistics.fmean(durations) if durations else None,
            "gpu_count": run.get("world_size"),
            "gpu_names": run.get("gpu_names", []),
            "torch_compile": run.get("torch_compile"),
            "peak_allocated": summary.get("max_peak_memory_allocated_bytes_per_gpu"),
            "peak_reserved": summary.get("max_peak_memory_reserved_bytes_per_gpu"),
        }, "live"

    recovered = _read_json(fold_dir / "resource_usage_recovered.json")
    return recovered.get("training", {}), "recovered" if recovered else "missing"


def _load_model(fold_dir: Path) -> dict[str, Any]:
    live = _read_json(fold_dir / "resource_usage.json").get("model", {})
    offline = _read_json(fold_dir / "offline_complexity.json")
    recovered = _read_json(fold_dir / "resource_usage_recovered.json").get("model", {})
    return {
        "parameters_total": offline.get("parameters_total", live.get("parameters_total", recovered.get("parameters_total"))),
        "flops": offline.get("flops", live.get("flops", recovered.get("flops"))),
        "input_shape": (
            offline.get("input_shape")
            or live.get("flops_protocol", {}).get("input_shape")
            or recovered.get("input_shape")
        ),
        "source": offline.get("counter") or recovered.get("source") or "resource_usage.json",
    }


def _load_inference(fold_dir: Path, scope: str) -> dict[str, Any]:
    entries = _read_json(fold_dir / "inference_usage.json").get("entries", [])
    matching = [entry for entry in entries if entry.get("scope") == scope]
    if matching:
        return matching[-1]
    recovered = _read_json(fold_dir / "resource_usage_recovered.json")
    recovered_entries = recovered.get("inference", [])
    matching = [entry for entry in recovered_entries if entry.get("scope") == scope]
    if matching:
        return matching[-1]
    # Old validation runs often lack timing, while an exact internal-test log
    # still provides a useful deployment-cost observation. Keep its scope clear.
    return recovered_entries[-1] if recovered_entries else {}


def build_cost_section(fold_dir: str | Path, scope: str) -> list[str]:
    fold_dir = Path(fold_dir)
    model = _load_model(fold_dir)
    training, training_source = _load_training(fold_dir)
    inference = _load_inference(fold_dir, scope)

    gpu_names = training.get("gpu_names") or []
    gpu_count = training.get("gpu_count")
    training_gpu = "N/A（历史未记录）"
    if gpu_count:
        name = gpu_names[0] if gpu_names else "GPU 型号未记录"
        training_gpu = f"{gpu_count} × {name}"
        if training.get("gpu_evidence"):
            training_gpu += f"（{training['gpu_evidence']}）"

    infer_gpu_names = inference.get("gpu_names") or []
    infer_gpu_count = inference.get("gpu_count")
    inference_gpu = "N/A（历史未记录）"
    if infer_gpu_count:
        name = infer_gpu_names[0] if infer_gpu_names else "GPU 型号未记录"
        inference_gpu = f"{infer_gpu_count} × {name}"
        if inference.get("gpu_evidence"):
            inference_gpu += f"（{inference['gpu_evidence']}）"

    inference_scope_raw = inference.get("scope", scope)
    inference_scope = {
        "validation": "validation 验证集",
        "internal_test": "固定内部测试集",
        "historical_internal_test": "历史固定内部测试集",
    }.get(inference_scope_raw, inference_scope_raw)
    inference_cases = inference.get("n_cases")
    inference_seconds = inference.get("duration_s")
    seconds_per_case = inference.get("seconds_per_case")
    compile_value = training.get("torch_compile")
    compile_text = str(compile_value) if isinstance(compile_value, bool) else "N/A（历史未验证）"

    lines = [
        "",
        SECTION_START,
        "=" * 80,
        "模型成本：理论量与实际运行量",
        "=" * 80,
        "口径说明：Params/FLOPs 是固定网络与输入下的理论量；训练和推理 GPU、时间、峰值显存是实际运行量，二者不能互相替代。",
        "",
        "理论模型成本",
        f"  Params             : {_millions(model.get('parameters_total'))}",
        f"  FLOPs              : {_flops(model.get('flops'))}",
        f"  FLOPs 输入         : {model.get('input_shape') or 'N/A'}",
        f"  理论量来源         : {model.get('source')}",
        "",
        "实际训练成本",
        f"  证据来源           : {training.get('checkpoint', 'N/A（未找到训练资源记录）')}",
        f"  GPU                : {training_gpu}",
        f"  torch.compile      : {compile_text}",
        f"  已记录 epoch       : {training.get('epochs', 'N/A')}",
        f"  epoch 时间累计     : {_duration(training.get('epoch_time_total_s'))}",
        f"  首尾 wall-clock    : {_duration(training.get('wall_span_s'))}",
        f"  每轮中位时间       : {_duration(training.get('epoch_time_median_s'))}",
        f"  每轮平均时间       : {_duration(training.get('epoch_time_mean_s'))}",
        f"  训练峰值显存/卡    : {_gib(training.get('peak_allocated'))}（allocated）",
        f"  CUDA 保留峰值/卡   : {_gib(training.get('peak_reserved'))}（reserved）",
        "",
        "实际推理/部署成本",
        f"  测量范围           : {inference_scope}",
        f"  证据来源           : {inference.get('source', 'N/A（未找到推理资源记录）')}",
        f"  GPU                : {inference_gpu}",
        f"  病例数             : {inference_cases if inference_cases is not None else 'N/A'}",
        f"  总推理时间         : {_duration(inference_seconds)}",
        f"  平均时间/case      : {_duration(seconds_per_case)}",
        f"  推理峰值显存/卡    : {_gib(inference.get('peak_allocated'))}（allocated）",
        f"  CUDA 保留峰值/卡   : {_gib(inference.get('peak_reserved'))}（reserved）",
        "",
        "边界：历史未记录的峰值显存不能从 checkpoint 反推；必须用同 checkpoint、同病例清单和同滑窗协议重新实测。",
    ]
    warnings = _read_json(fold_dir / "resource_usage_recovered.json").get("warnings", [])
    for warning in warnings:
        lines.append(f"  警告               : {warning}")
    lines.extend([SECTION_END, ""])
    return lines


def upsert_cost_section(report_path: str | Path, fold_dir: str | Path, scope: str) -> None:
    report_path = Path(report_path)
    text = report_path.read_text(encoding="utf-8") if report_path.is_file() else ""
    if SECTION_START in text and SECTION_END in text:
        prefix = text.split(SECTION_START, 1)[0].rstrip()
        suffix = text.split(SECTION_END, 1)[1].lstrip("\n")
        text = prefix + "\n" + suffix
    section = "\n".join(build_cost_section(fold_dir, scope)).rstrip() + "\n"
    report_path.write_text(text.rstrip() + "\n" + section, encoding="utf-8")
