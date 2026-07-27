"""Generate cross-trainer hard-case statistics for MSD, IRCADb, and HCC.

This script is read-only with respect to experiment artifacts. It reads existing
``summary.json`` files (falling back to the human-readable report when a legacy
internal run has no summary) and rewrites the paper-facing Markdown analysis.

Definitions:
  * positive severe failure: Tumor Dice < 0.3
  * negative false positive: predicted tumor voxels > 0

Only the 30 source-only methods used by the fair comparison are
eligible. HCC-adapted, HCC-only, and mixed-domain trainers are excluded.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import median


WORKSPACE = Path("/home/PuMengYu/nnUNet_workspace")
REPO = Path("/home/PuMengYu/nnUNet")
OUTPUT = REPO / "pumengyu/notes/md/02_实验结果/三个数据集失败案例分析.md"

FAIR_METHODS = (
    "Baseline",
    "SizeOV2",
    "SizeOV3",
    "NoMirror",
    "SizeOV3_NoMirror",
    "MoE",
    "MoE_SizeOV2",
    "MoE_SizeOV4",
    "MoE_SizeOV5",
    "MLAUNet",
    "MLAUNet_1500",
    "MLAUNet_MoE_IB7_SizeOV4",
    "MLA_GK5_V4",
    "DeepPlainResGN",
    "DeepPlainResGN_SizeOV4",
    "DeepDWIBResGN",
    "DeepDWIBMedConfig",
    "DeepResGN_MLA",
    "DWSepRes4_MoE_SizeOV4",
    "MedNeXt",
    "MedNeXt_SizeOV4",
    "MedNeXt_MHA",
    "MedNeXt_MHA_MoE",
    "MedNeXt_MLA",
    "MedNeXt_MLA_MoE",
    "MedNeXt_MLA_MoE_SizeOV4",
    "MedNeXt_MLA_MoE_FPSafe",
    "SwinUNETR",
    "nnFormer",
    "EfficientMedNeXt_L_Official",
)

TRAINER_ALIASES = {
    "nnUNetTrainer_Baseline": "Baseline",
    "nnUNetTrainer_SizeOversampleV2": "SizeOV2",
    "nnUNetTrainer_SizeOversampleV3": "SizeOV3",
    "nnUNetTrainer_NoMirror": "NoMirror",
    "nnUNetTrainer_SizeOversampleV3_NoMirror": "SizeOV3_NoMirror",
    "nnUNetTrainer_MLAUNet_MoE": "MoE",
    "nnUNetTrainer_MLAUNet_MoE_SizeOversampleV2": "MoE_SizeOV2",
    "nnUNetTrainer_MLAUNet_MoE_SizeOversampleV4": "MoE_SizeOV4",
    "nnUNetTrainer_MLAUNet_MoE_SizeOversampleV5": "MoE_SizeOV5",
    "nnUNetTrainer_MLAUNet": "MLAUNet",
    "nnUNetTrainer_MLAUNet_1500": "MLAUNet_1500",
    "nnUNetTrainer_MLAUNet_MoE_IB7_SizeOversampleV4": "MLAUNet_MoE_IB7_SizeOV4",
    "nnUNetTrainer_MLA_GK5_V4": "MLA_GK5_V4",
    "nnUNetTrainer_DeepPlainResGN": "DeepPlainResGN",
    "nnUNetTrainer_DeepPlainResGN_SizeOV4": "DeepPlainResGN_SizeOV4",
    "nnUNetTrainer_DeepDWIBResGN": "DeepDWIBResGN",
    "nnUNetTrainer_DeepDWIBMedConfig": "DeepDWIBMedConfig",
    "nnUNetTrainer_DeepResGN_MLA": "DeepResGN_MLA",
    "nnUNetTrainer_DWSepRes4_MoE_SizeOV4": "DWSepRes4_MoE_SizeOV4",
    "nnUNetTrainer_MedNeXt": "MedNeXt",
    "nnUNetTrainer_MedNeXt_SizeOV4": "MedNeXt_SizeOV4",
    "nnUNetTrainer_MedNeXt_MHA": "MedNeXt_MHA",
    "nnUNetTrainer_MedNeXt_MHA_MoE": "MedNeXt_MHA_MoE",
    "nnUNetTrainer_MedNeXt_MLA": "MedNeXt_MLA",
    "nnUNetTrainer_MedNeXt_MLA_MoE": "MedNeXt_MLA_MoE",
    "nnUNetTrainer_MedNeXt_MLA_MoE_SizeOV4": "MedNeXt_MLA_MoE_SizeOV4",
    "nnUNetTrainer_MedNeXt_MLA_MoE_FPSafe": "MedNeXt_MLA_MoE_FPSafe",
    "nnUNetTrainer_SwinUNETR": "SwinUNETR",
    "nnUNetTrainer_nnFormer": "nnFormer",
    "nnUNetTrainer_EfficientMedNeXt_L_Official": "EfficientMedNeXt_L_Official",
}


@dataclass(frozen=True)
class CaseMetric:
    dice: float | None
    recall: float | None
    precision: float | None
    pred_tumor: int
    gt_tumor: int

    @property
    def positive(self) -> bool:
        return self.gt_tumor > 0


@dataclass
class DomainData:
    name: str
    label: str
    methods: dict[str, dict[str, CaseMetric]]
    sources: dict[str, str]
    missing: list[str]


def _case_name(path_value: str) -> str:
    name = Path(path_value).name
    return name[:-7] if name.endswith(".nii.gz") else Path(name).stem


def _read_summary(path: Path) -> dict[str, CaseMetric]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows: dict[str, CaseMetric] = {}
    for item in data["metric_per_case"]:
        tumor = item["metrics"]["2"]
        tp, fp, fn = (int(tumor[key]) for key in ("TP", "FP", "FN"))
        gt_tumor = tp + fn
        pred_tumor = tp + fp
        if gt_tumor:
            rows[_case_name(item["reference_file"])] = CaseMetric(
                dice=float(tumor["Dice"]),
                recall=tp / gt_tumor,
                precision=tp / pred_tumor if pred_tumor else 0.0,
                pred_tumor=pred_tumor,
                gt_tumor=gt_tumor,
            )
        else:
            rows[_case_name(item["reference_file"])] = CaseMetric(
                dice=None,
                recall=None,
                precision=None,
                pred_tumor=pred_tumor,
                gt_tumor=0,
            )
    return rows


POSITIVE_ROW = re.compile(
    r"^\s+((?:liver|ircadb|HCC)_\d+)\s+"
    r"(\d+\.\d{4})\s+(\d+\.\d{4})\s+(\d+\.\d{4})\s+"
    r"\d+\.\d{4}\s+([\d,]+)\s+([\d,]+)\s+[\d,]+\s+",
    re.MULTILINE,
)
NEGATIVE_ROW = re.compile(
    r"^\s+((?:liver|ircadb|HCC)_\d+)\s+\d+\.\d{4}\s+([\d,]+)\s+"
    r"[\d,]+\s+GT无肿瘤",
    re.MULTILINE,
)
NEGATIVE_LIST_ROW = re.compile(
    r"^\s+((?:liver|ircadb|HCC)_\d+)\s+liver_dice=\d+\.\d{4}\s+"
    r"pred_tumor=([\d,]+)(?:\s+\[误报\])?\s*$",
    re.MULTILINE,
)


def _read_report(path: Path) -> dict[str, CaseMetric]:
    text = path.read_text(encoding="utf-8")
    rows: dict[str, CaseMetric] = {}
    for case, dice, recall, precision, pred, gt in POSITIVE_ROW.findall(text):
        rows[case] = CaseMetric(
            dice=float(dice),
            recall=float(recall),
            precision=float(precision),
            pred_tumor=int(pred.replace(",", "")),
            gt_tumor=int(gt.replace(",", "")),
        )
    for case, pred in NEGATIVE_ROW.findall(text):
        rows[case] = CaseMetric(None, None, None, int(pred.replace(",", "")), 0)
    # Legacy internal reports list correctly negative cases only in the summary block.
    for case, pred in NEGATIVE_LIST_ROW.findall(text):
        rows.setdefault(case, CaseMetric(None, None, None, int(pred.replace(",", "")), 0))
    return rows


def _internal_paths() -> dict[str, tuple[Path, Path]]:
    root = WORKSPACE / "results_v2/Dataset003_Liver"
    found: dict[str, tuple[Path, Path]] = {}
    for report in sorted(root.glob("*/fold_0/test_report_custom.txt")):
        trainer = report.parents[1].name.split("__nnUNetPlans__", 1)[0]
        method = TRAINER_ALIASES.get(trainer)
        if method in FAIR_METHODS:
            found[method] = (report, report.parent / "test_prediction/summary.json")
    return found


def _external_paths(relative_root: str) -> dict[str, tuple[Path, Path]]:
    root = WORKSPACE / relative_root
    return {
        method: (root / method / "report_custom.txt", root / method / "predictions/summary.json")
        for method in FAIR_METHODS
        if (root / method / "report_custom.txt").exists()
    }


def _load_domain(name: str, label: str, paths: dict[str, tuple[Path, Path]]) -> DomainData:
    methods: dict[str, dict[str, CaseMetric]] = {}
    sources: dict[str, str] = {}
    for method in FAIR_METHODS:
        pair = paths.get(method)
        if pair is None:
            continue
        report, summary = pair
        report_rows = _read_report(report)
        if summary.exists():
            summary_rows = _read_summary(summary)
            if len(summary_rows) >= len(report_rows):
                rows = summary_rows
                sources[method] = "summary.json"
            else:
                rows = report_rows
                sources[method] = "report fallback (incomplete summary.json)"
        else:
            rows = report_rows
            sources[method] = "report fallback (missing summary.json)"
        if not rows:
            raise RuntimeError(f"No per-case rows parsed: {name}/{method}")
        methods[method] = rows
    missing = [method for method in FAIR_METHODS if method not in methods]
    return DomainData(name, label, methods, sources, missing)


def _validate(domain: DomainData) -> None:
    case_sets = {method: set(rows) for method, rows in domain.methods.items()}
    first_method = next(iter(case_sets))
    expected = case_sets[first_method]
    mismatches = {method: sorted(expected ^ cases) for method, cases in case_sets.items() if cases != expected}
    if mismatches:
        raise RuntimeError(f"Case-list mismatch in {domain.name}: {mismatches}")
    for case in expected:
        gt_values = {rows[case].gt_tumor for rows in domain.methods.values()}
        if len(gt_values) != 1:
            raise RuntimeError(f"GT mismatch in {domain.name}/{case}: {sorted(gt_values)}")


def _tier(failed: int, total: int) -> str:
    if failed == total:
        return "全部严重失败"
    ratio = failed / total
    if ratio >= 0.8:
        return "近全部失败(≥80%)"
    if ratio >= 0.5:
        return "多数失败(≥50%)"
    if ratio >= 0.2:
        return "部分 Trainer 失败(20%–<50%)"
    return "少数 Trainer 失败(<20%)"


def _fmt_methods(methods: list[str], total: int) -> str:
    if len(methods) == total:
        return "全部纳入方法"
    return "<br>".join(f"`{method}`" for method in methods)


def _positive_table(domain: DomainData) -> tuple[list[str], list[dict]]:
    method_names = list(domain.methods)
    cases = sorted(next(iter(domain.methods.values())))
    rows = []
    for case in cases:
        values = [(method, domain.methods[method][case]) for method in method_names]
        if not values[0][1].positive:
            continue
        failed = [method for method, metric in values if metric.dice is not None and metric.dice < 0.3]
        if len(failed) < 2:
            continue
        dices = [metric.dice for _, metric in values if metric.dice is not None]
        ranked = sorted((metric.dice, method) for method, metric in values if metric.dice is not None)
        median_dice = median(dices)
        median_item = min(ranked, key=lambda item: (abs(item[0] - median_dice), item[1]))
        rows.append({
            "case": case,
            "failed": failed,
            "n": len(failed),
            "total": len(values),
            "min": min(dices),
            "median": median_dice,
            "max": max(dices),
            "gt": values[0][1].gt_tumor,
            "worst": ranked[0],
            "median_item": median_item,
            "best": ranked[-1],
        })
    rows.sort(key=lambda row: (-row["n"], row["median"], row["case"]))
    lines = [
        "| Case | 严重失败 Trainer | 失败率 | Dice min / median / max | GT tumor voxels | 分级 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['case']}` | {row['n']}/{row['total']} | {row['n'] / row['total']:.1%} | "
            f"{row['min']:.4f} / {row['median']:.4f} / {row['max']:.4f} | {row['gt']:,} | "
            f"{_tier(row['n'], row['total'])} |"
        )
    if not rows:
        lines.append("| 无 | 0 | 0% | — | — | 无重复严重失败 |")
    return lines, rows


def _negative_table(domain: DomainData) -> tuple[list[str], list[dict]]:
    method_names = list(domain.methods)
    cases = sorted(next(iter(domain.methods.values())))
    rows = []
    for case in cases:
        values = [(method, domain.methods[method][case]) for method in method_names]
        if values[0][1].positive:
            continue
        failed = [method for method, metric in values if metric.pred_tumor > 0]
        pred_values = [metric.pred_tumor for _, metric in values]
        rows.append({
            "case": case,
            "failed": failed,
            "n": len(failed),
            "total": len(values),
            "median": median(pred_values),
            "max": max(pred_values),
        })
    rows.sort(key=lambda row: (-row["n"], -row["max"], row["case"]))
    lines = [
        "| Case | 误报 Trainer | 误报率 | 预测 tumor voxels median / max | 分级 |",
        "|---|---:|---:|---:|---|",
    ]
    for row in rows:
        if row["n"] == 0:
            classification = "全部正确阴性"
        elif row["n"] == row["total"]:
            classification = "全部误报"
        else:
            classification = "多模型误报"
        lines.append(
            f"| `{row['case']}` | {row['n']}/{row['total']} | {row['n'] / row['total']:.1%} | "
            f"{row['median']:,.1f} / {row['max']:,} | "
            f"{classification} |"
        )
    if not rows:
        lines.append("| N/A | 0 | N/A | N/A | 该数据集无无肿瘤病例 |")
    return lines, rows


def _details(domain: DomainData, positive_rows: list[dict], negative_rows: list[dict]) -> list[str]:
    selected = [row for row in positive_rows if row["n"] / row["total"] >= 0.5]
    selected += [row for row in negative_rows if row["n"] / row["total"] >= 0.5]
    if not selected:
        return ["无失败率达到 50% 的病例。"]
    lines = []
    for row in selected:
        kind = "严重失败" if "min" in row else "无肿瘤误报"
        lines.append(
            f"#### `{row['case']}`：{kind} {row['n']}/{row['total']}",
        )
        lines.append("")
        if "min" in row:
            worst_dice, worst_method = row["worst"]
            median_dice, median_method = row["median_item"]
            best_dice, best_method = row["best"]
            lines += [
                f"- 最差代表：`{worst_method}`，Dice={worst_dice:.4f}。",
                f"- 中位代表：`{median_method}`，Dice={median_dice:.4f}。",
                f"- 最佳代表：`{best_method}`，Dice={best_dice:.4f}。",
                f"- 达到失败条件的方法：{_fmt_methods(row['failed'], row['total'])}。",
                "",
            ]
        else:
            lines += [
                f"达到失败条件的方法：{_fmt_methods(row['failed'], row['total'])}。",
                "",
            ]
    return lines


def _domain_section(domain: DomainData) -> list[str]:
    positive_table, positive_rows = _positive_table(domain)
    negative_table, negative_rows = _negative_table(domain)
    case_values = next(iter(domain.methods.values())).values()
    n_positive = sum(metric.positive for metric in case_values)
    n_negative = sum(not metric.positive for metric in case_values)
    universal_positive = sum(row["n"] == row["total"] for row in positive_rows)
    majority_positive = sum(row["n"] / row["total"] >= 0.5 for row in positive_rows)
    universal_negative = sum(row["n"] == row["total"] for row in negative_rows)
    fallback = sum(source != "summary.json" for source in domain.sources.values())
    missing_text = "无" if not domain.missing else "、".join(f"`{item}`" for item in domain.missing)
    fallback_methods = [method for method, source in domain.sources.items() if source != "summary.json"]
    fallback_text = "无" if not fallback_methods else "、".join(f"`{item}`" for item in fallback_methods)
    lines = [
        f"## {domain.label}",
        "",
        f"- 纳入方法：{len(domain.methods)}/{len(FAIR_METHODS)}。",
        f"- 缺失方法：{missing_text}。",
        f"- 病例构成：有肿瘤 {n_positive} 例，无肿瘤 {n_negative} 例。",
        f"- 数据来源：{len(domain.methods) - fallback} 个方法使用 `summary.json`，{fallback} 个方法使用报告回退解析（{fallback_text}）。",
        f"- 共识结果：{universal_positive} 个有肿瘤病例被全部方法严重分割失败，"
        f"{majority_positive} 个被至少一半方法严重失败；"
        f"{universal_negative} 个无肿瘤病例被全部方法误报。",
        "",
        "### 有肿瘤病例：多 Trainer 严重失败",
        "",
        *positive_table,
        "",
        "> `n/total` 的分子是该病例上 Tumor Dice < 0.3 的 Trainer 数，分母是纳入统计的 Trainer 总数，不是病例数。表中保留至少 2 个 Trainer 失败的病例作为完整记录；失败率 <20% 仅代表少数 Trainer 失败，不应解读为跨模型共同难例。`max` 仍小于 0.3 时，才是“全部方法严重失败”。",
        "",
        "### 无肿瘤病例：跨 Trainer 误报",
        "",
        *negative_table,
        "",
        "### 高共识失败病例的方法明细",
        "",
        *_details(domain, positive_rows, negative_rows),
    ]
    return lines


def _priority_case_line(domain: DomainData, case: str) -> str:
    values = [rows[case] for rows in domain.methods.values()]
    total = len(values)
    if values[0].positive:
        failed = sum(metric.dice is not None and metric.dice < 0.3 for metric in values)
        kind = "有肿瘤严重失败"
    else:
        failed = sum(metric.pred_tumor > 0 for metric in values)
        kind = "无肿瘤误报"
    return f"- `{case}` — {kind}：{failed}/{total}（{failed / total:.1%}）"


def _priority_summary(domains: list[DomainData]) -> list[str]:
    by_name = {domain.name: domain for domain in domains}
    msd = by_name["MSD"]
    ircadb = by_name["IRCADb"]
    hcc = by_name["HCC"]
    _, hcc_rows = _positive_table(hcc)
    hcc_majority = [row for row in hcc_rows if row["n"] / row["total"] >= 0.5]
    hcc_case_total = len(next(iter(hcc.methods.values())))

    lines = [
        "## 重点病例结论列表",
        "",
        "### MSD internal",
        "",
    ]
    lines += [_priority_case_line(msd, case) for case in ("liver_127", "liver_41", "liver_91", "liver_89")]
    lines += [
        "",
        "### 3D-IRCADb",
        "",
    ]
    lines += [
        _priority_case_line(ircadb, case)
        for case in ("ircadb_018", "ircadb_008", "ircadb_016", "ircadb_014", "ircadb_007", "ircadb_005")
    ]
    lines += [
        "",
        "### HCCReferencedCT v2",
        "",
        f"- Trainer 严重失败率 ≥50% 的病例共 {len(hcc_majority)}/{hcc_case_total}"
        f"（{len(hcc_majority) / hcc_case_total:.1%}）。",
    ]
    lines += [_priority_case_line(hcc, row["case"]) for row in hcc_majority]
    universal = sum(row["n"] == row["total"] for row in hcc_majority)
    lines += [
        "",
        f"> HCC 中有 {universal} 例为全部 Trainer 严重失败，说明其存在明显的系统性跨域肿瘤分割困难，不是少数模型的偶发异常。",
        "",
        "> 上述 `n/total` 均指“对同一病例达到失败或误报条件的 Trainer 数 / 纳入统计的 Trainer 总数”，不是失败病例数 / 总病例数。",
    ]
    return lines


def build_markdown(domains: list[DomainData]) -> str:
    method_counts = ", ".join(f"{domain.name}={len(domain.methods)}" for domain in domains)
    lines = [
        "# MSD、IRCADb 与 HCC 跨 Trainer 失败病例分析",
        "",
        "> 更新：2026-07-22  ",
        "> 文档定位：从“某个模型为什么失败”转为“哪些病例对多种架构与训练策略都困难”。  ",
        f"> 公平 source-only 方法覆盖：{method_counts}。",
        "",
        *_priority_summary(domains),
        "",
        "## 1. 统计口径",
        "",
        "1. MSD 指 `Dataset003_Liver` 固定 26 例 internal test；IRCADb 为 20 例 source-only；HCC 为 `HCCReferencedCT v2` 固定 21 例 source-only held-out test。",
        "2. 有肿瘤病例的“严重失败”定义为 `Tumor Dice < 0.3`。这与现有三域报告分级一致。",
        "3. 无肿瘤病例不计算 Tumor Dice；只要 `pred_tumor > 0` 就记为 case-level 误报。",
        "4. “全部 Trainer 失败”指当前该数据域所有纳入方法都达到失败条件，不是指 Dice 必须等于 0。",
        "5. 仅纳入公平表中的 30 种 Dataset003 source-only 方法；各数据域按实际已有报告的方法计数。HCC Adapter、HCC-only 和 MSD/HCC mix 改变了训练数据或路由，不进入共识失败计数。",
        "",
        "## 2. 如何解读",
        "",
        "- 表中的 `3/27`、`2/27` 等数值表示：对同一个病例，27 个 Trainer 中分别有 3 个、2 个达到失败条件；它不表示 27 个病例中有 3 个或 2 个失败。",
        "- 单个 Trainer 失败：可能是架构、采样、增强或 loss 的特定问题。",
        "- 失败率 <20%：只能说明少数 Trainer 在该病例上失败，更像模型特定异常，不足以将该病例定义为跨模型共同难例。",
        "- 失败率 20%–<50%：表示部分 Trainer 重复失败，可作为次要难例追踪，但不能代表多数模型存在共性问题。",
        "- 失败率 ≥50%：才作为高共识难例重点分析；此时更应考虑小病灶、低对比度、边界模糊、标注差异或显著域偏移。",
        "- 全部 Trainer 都失败：优先做原图、GT、多模型预测叠加和标注质检，而不应归因于某一个模块（例如 FP-Safe）。",
        "",
    ]
    for index, domain in enumerate(domains, start=3):
        section = _domain_section(domain)
        section[0] = f"## {index}. {domain.label}"
        lines += section + [""]
    lines += [
        "## 6. 结论与可视化优先级",
        "",
        "1. 首先可视化“全部严重失败”病例，其次是失败率≥80%和失败率≥50%的病例。",
        "2. 每个病例应至少同时展示 CT、GT、最佳 Dice 模型、中位 Dice 模型和最差 Dice 模型，不再只展示 FP-Safe。",
        "3. 对于所有模型都失败的病例，先核对标注质量、CT 相位/强度分布、病灶体积和是否位于肝边界或血管附近，再讨论机制归因。",
        "4. 本表回答的是“失败共识度”，不替代每个数据域的总体 Dice、Recall、Precision 和 FP 率统计。",
        "",
        "## 7. 重算方式",
        "",
        "```bash",
        "python pumengyu/tools/analyasis/generate_cross_trainer_case_analysis.py",
        "```",
        "",
        ">该命令不运行推理，只读取现有 `summary.json`/报告并更新本文档。",
    ]
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--check", action="store_true", help="Validate and print Markdown without writing")
    args = parser.parse_args()

    domains = [
        _load_domain("MSD", "MSD internal（Dataset003_Liver）", _internal_paths()),
        _load_domain("IRCADb", "3D-IRCADb source-only", _external_paths("results_v2/IRCADb/source_only")),
        _load_domain(
            "HCC",
            "HCCReferencedCT v2 source-only",
            _external_paths("results_v2/Dataset013_HCCReferencedCT/source_only"),
        ),
    ]
    for domain in domains:
        _validate(domain)
        print(
            f"[{domain.name}] methods={len(domain.methods)}/{len(FAIR_METHODS)} "
            f"cases={len(next(iter(domain.methods.values())))} missing={domain.missing or 'none'}"
        )
    markdown = build_markdown(domains)
    if args.check:
        print(markdown)
        return
    args.output.write_text(markdown, encoding="utf-8")
    print(f"WROTE {args.output}")


if __name__ == "__main__":
    main()
