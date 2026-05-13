"""
从 fold_X/validation/summary.json 生成 report_custom.json

用法：
  python pumengyu/tools/gen_report_json.py \
    --fold_dir <fold_X 目录>

输出：fold_dir/report_custom.json
"""

import argparse
import json
import math
from pathlib import Path


def generate_report_json(fold_dir: Path) -> Path | None:
    """从 fold_dir/validation/summary.json 生成 fold_dir/report_custom.json，返回输出路径。"""
    summary_path = fold_dir / "validation" / "summary.json"
    if not summary_path.exists():
        print(f"[gen_report_json] summary.json 找不到: {summary_path}")
        return None

    summary = json.load(open(summary_path))
    records = []

    for c in summary["metric_per_case"]:
        case = Path(c["reference_file"]).stem.replace(".nii", "")
        m1 = c["metrics"].get("1", {})
        m2 = c["metrics"].get("2", {})

        dice_liver = m1.get("Dice")
        if dice_liver is not None and math.isnan(dice_liver):
            dice_liver = None
        if dice_liver is not None:
            dice_liver = round(dice_liver, 4)

        # GT 无肿瘤时 dice_cancer 为 null
        gt_tumor = int(m2.get("TP", 0)) + int(m2.get("FN", 0))
        dice_cancer = m2.get("Dice")
        if gt_tumor == 0:
            dice_cancer = None
        elif dice_cancer is not None and math.isnan(dice_cancer):
            dice_cancer = None
        else:
            if dice_cancer is not None:
                dice_cancer = round(dice_cancer, 4)

        records.append({
            "case": case,
            "dice_liver": dice_liver,
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
