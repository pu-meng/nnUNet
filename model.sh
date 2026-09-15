#!/usr/bin/env bash
# One-command entrypoint for the read-only MedNeXt model audit.
# Usage: ./model.sh [check|trace|graph|pipeline|help]

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="/home/PuMengYu/anaconda3/envs/medseg/bin/python"
AUDIT_SCRIPT="${REPO_ROOT}/pumengyu/tools/model_audit/trace_mednext_mha_moe.py"
GRAPH_SCRIPT="${REPO_ROOT}/pumengyu/tools/model_audit/draw_model_architecture.py"
PIPELINE_SCRIPT="${REPO_ROOT}/pumengyu/tools/model_audit/draw_model_pipeline.py"
OUTPUT_DIR="${REPO_ROOT}/pumengyu/tools/model_audit/output"
ACTION="${1:-help}"

case "${ACTION}" in
    check)
        "${PYTHON_BIN}" "${AUDIT_SCRIPT}" --output-dir "${OUTPUT_DIR}" >/dev/null
        "${PYTHON_BIN}" - "${OUTPUT_DIR}/model_architecture.json" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
audit = payload.get("static_audit", payload)
norms = audit["normalization_modules"]
classes = sorted({row["class"] for row in norms})
print("模型审计完成")
print(f"参数量: {audit['parameters_total'] / 1e6:.2f}M")
print(f"归一化: {', '.join(classes)}")
print(f"研究问题: {len(audit['improvement_prompts'])} 个")
print("详细结果: " + str(Path(sys.argv[1]).parent))
PY
        ;;
    trace)
        "${PYTHON_BIN}" "${AUDIT_SCRIPT}" --forward --output-dir "${OUTPUT_DIR}" >/dev/null
        echo "模型 shape trace 完成"
        echo "详细结果: ${OUTPUT_DIR}"
        ;;
    graph)
        "${PYTHON_BIN}" "${GRAPH_SCRIPT}" --output-dir "${OUTPUT_DIR}" >/dev/null
        echo "模型结构图完成"
        echo "图像: ${OUTPUT_DIR}/model_architecture.png"
        echo "矢量图: ${OUTPUT_DIR}/model_architecture.svg"
        ;;
    pipeline)
        "${PYTHON_BIN}" "${PIPELINE_SCRIPT}" --output-dir "${OUTPUT_DIR}" >/dev/null
        echo "case-训练-推理流程图完成"
        echo "图像: ${OUTPUT_DIR}/model_pipeline.png"
        echo "矢量图: ${OUTPUT_DIR}/model_pipeline.svg"
        ;;
    help|-h|--help)
        echo "MedNeXt 模型分析工具"
        echo
        echo "用法: /home/PuMengYu/model.sh <命令>"
        echo
        echo "可用命令："
        echo "  check  检查模型结构、参数量、归一化层，并列出可研究问题"
        echo "  trace  在 CPU 上执行一个 128^3 patch，记录每层真实输入/输出 shape"
        echo "  graph  生成 Encoder-Decoder 总览图和 MHA/MoE 细节图"
        echo "  pipeline 生成 case提取、预处理、训练损失、推理后处理流程图"
        echo "  help   显示这份命令说明"
        echo
        echo "示例："
        echo "  /home/PuMengYu/model.sh check"
        echo "  /home/PuMengYu/model.sh trace"
        echo "  /home/PuMengYu/model.sh graph"
        echo "  /home/PuMengYu/model.sh pipeline"
        echo
        echo "功能实现："
        echo "  静态审计  -> pumengyu/tools/model_audit/trace_mednext_mha_moe.py"
        echo "  shape追踪 -> 同一脚本的 --forward"
        echo "  结构绘图  -> pumengyu/tools/model_audit/draw_model_architecture.py"
        echo "  流程绘图  -> pumengyu/tools/model_audit/draw_model_pipeline.py"
        echo "  输出目录  -> ${OUTPUT_DIR}"
        echo
        echo "说明：只读分析，不训练、不加载数据、不修改 checkpoint。"
        ;;
    *)
        echo "未知操作: ${ACTION}（可用: check, trace, graph, pipeline, help）" >&2
        exit 2
        ;;
esac
