"""
从 fold_X/validation/summary.json 生成 report_custom.json

自动检测数据集模式：
  liver_tumor  — label 1=肝脏, label 2=肿瘤（Dataset003）
  tumor_only   — label 1=肿瘤（Dataset004，已裁剪肝脏 ROI）

用法：
  python pumengyu/tools/analyasis/gen_report_json.py \
    --fold_dir <fold_X 目录>

输出：fold_dir/report_custom.json
"""

import argparse
import json
import math
from pathlib import Path


def _detect_mode(summary: dict) -> str:
    """从 metric_per_case 检测是 'liver_tumor' 还是 'tumor_only'。"""
    for c in summary.get("metric_per_case", []):
        if "2" in c.get("metrics", {}):
            return "liver_tumor"
    return "tumor_only"


def generate_report_json(fold_dir: Path) -> Path | None:
    """从 fold_dir/validation/summary.json 生成 fold_dir/report_custom.json，返回输出路径。"""
    summary_path = fold_dir / "validation" / "summary.json"
    if not summary_path.exists():
        print(f"[gen_report_json] summary.json 找不到: {summary_path}")
        return None

    summary = json.load(open(summary_path))
    mode = _detect_mode(summary)
    tumor_label = "2" if mode == "liver_tumor" else "1"
    liver_label = "1" if mode == "liver_tumor" else None
    print(f"[gen_report_json] 检测到模式: {mode}  (tumor=label{tumor_label})")

    records = []
    for c in summary["metric_per_case"]:
        case = Path(c["reference_file"]).stem.replace(".nii", "")
        m_tumor = c["metrics"].get(tumor_label, {})
        m_liver = c["metrics"].get(liver_label, {}) if liver_label else {}

        # 肝脏 dice（tumor_only 模式为 null）
        dice_liver = m_liver.get("Dice") if liver_label else None
        if dice_liver is not None and math.isnan(dice_liver):
            dice_liver = None
        if dice_liver is not None:
            dice_liver = round(dice_liver, 4)

        # GT 无肿瘤时 dice_cancer 为 null
        gt_tumor = int(m_tumor.get("TP", 0)) + int(m_tumor.get("FN", 0))
        dice_cancer = m_tumor.get("Dice")
        if gt_tumor == 0:
            dice_cancer = None
        elif dice_cancer is not None and math.isnan(dice_cancer):
            dice_cancer = None
        else:
            if dice_cancer is not None:
                dice_cancer = round(dice_cancer, 4)

        records.append({
            "case":        case,
            "mode":        mode,
            "dice_liver":  dice_liver,
            "dice_cancer": dice_cancer,
        })

    records.sort(key=lambda x: x["case"])

    out_path = fold_dir / "report_custom.json"
    out_path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[gen_report_json] {out_path}  ({len(records)} cases)")
    return out_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fold_dir", required=True, help="fold_X 目录路径")
    args = p.parse_args()
    generate_report_json(Path(args.fold_dir))


if __name__ == "__main__":
    main()
