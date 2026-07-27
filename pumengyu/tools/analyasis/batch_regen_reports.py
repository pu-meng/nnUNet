"""批量重新生成所有 fold 的报告，默认同时生成可视化。"""

import argparse
from pathlib import Path
from pumengyu.tools.analyasis.eval_fold_report import run_eval_report

RESULTS_ROOT      = Path("/home/PuMengYu/nnUNet_workspace/results_v2")
PREPROCESSED_ROOT = Path("/home/PuMengYu/nnUNet_workspace/preprocessed")
RAW_ROOT          = Path("/home/PuMengYu/nnUNet_workspace/raw")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="仅在既有可视化已通过审计时使用；不生成新 PNG",
    )
    args = parser.parse_args()
    ok, fail = [], []

    for summary in sorted(RESULTS_ROOT.rglob("validation/summary.json")):
        val_dir  = summary.parent
        fold_dir = val_dir.parent
        dataset  = fold_dir.parts[fold_dir.parts.index("results") + 1]

        gt_dir  = PREPROCESSED_ROOT / dataset / "gt_segmentations"
        img_dir = RAW_ROOT           / dataset / "imagesTr"

        if not gt_dir.exists() or not img_dir.exists():
            print(f"[SKIP] {fold_dir.name} — 找不到 gt_dir 或 img_dir")
            fail.append(str(fold_dir))
            continue

        print(f"\n{'='*60}")
        print(f"[报告] {dataset} / {fold_dir.parent.name} / {fold_dir.name}")
        try:
            run_eval_report(
                val_dir=val_dir,
                gt_dir=gt_dir,
                img_dir=img_dir,
                no_vis=args.report_only,
            )
            ok.append(str(fold_dir))
        except Exception as e:
            print(f"[FAIL] {e}")
            fail.append(str(fold_dir))

    print(f"\n{'='*60}")
    print(f"完成 {len(ok)}/{len(ok)+len(fail)} 个 fold")
    if fail:
        print("失败：")
        for f in fail:
            print(f"  {f}")


if __name__ == "__main__":
    main()
