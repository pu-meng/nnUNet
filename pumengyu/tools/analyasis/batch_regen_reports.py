"""
批量重新生成所有 fold 的 report_custom.txt。
用法（在 nnUNet/ 根目录下执行）：
    python pumengyu/tools/analyasis/batch_regen_reports.py
"""

from pathlib import Path
from pumengyu.tools.analyasis.eval_fold_report import run_eval_report

RESULTS_ROOT      = Path("/home/PuMengYu/nnUNet_workspace/results")
PREPROCESSED_ROOT = Path("/home/PuMengYu/nnUNet_workspace/preprocessed")
RAW_ROOT          = Path("/home/PuMengYu/nnUNet_workspace/raw")

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
            no_vis=True,   # 跳过可视化，只更新 report_custom.txt
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
