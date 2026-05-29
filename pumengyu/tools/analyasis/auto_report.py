"""
训练结束后自动生成报告的统一出口。

数据集模式自动检测（无需手动指定）：
  liver_tumor  — label 1=肝脏, label 2=肿瘤（Dataset003）
  tumor_only   — label 1=肿瘤（Dataset004，已裁剪肝脏 ROI）

Trainer 中调用方式：
    from pumengyu.tools.analyasis.auto_report import run_auto_report
    run_auto_report(fold_dir, gt_dir, img_dir)

生成文件：
  fold_dir/report_custom.json   — 每 case 的 dice_liver / dice_cancer
  fold_dir/report_custom.txt    — 无肿瘤误报 + 综合指标（全 cases）+ per-case 分级
  fold_dir/vis_png_custom/      — 每 case 可视化 PNG（GT / Pred / Diff）
"""

from pathlib import Path

from pumengyu.tools.analyasis.gen_report_json import generate_report_json
from pumengyu.tools.analyasis.eval_fold_report import run_eval_report


def run_auto_report(
    fold_dir,
    gt_dir,
    img_dir,
    vis_slices: int = 5,
    no_vis: bool = False,
    min_tumor_size: int = 0,
    out_dir=None,
    pred_subdir: str = "validation",
):
    """
    参数：
      fold_dir       fold_X 目录
      gt_dir         preprocessed/DatasetXXX/gt_segmentations
      img_dir        raw/DatasetXXX/imagesTr
      vis_slices     每 case 可视化切片数（默认 5）
      no_vis         True 时跳过可视化，加速收尾
      min_tumor_size 后处理对比阈值，0 = 关闭
      out_dir        输出目录，None 时写到 fold_dir 根（向后兼容）
      pred_subdir    预测结果子目录名，默认 "validation"；内部测试传 "test_prediction"
    """
    fold_dir = Path(fold_dir)
    out_dir  = Path(out_dir) if out_dir is not None else fold_dir
    print(f"\n[auto_report] ===== 开始生成报告: {fold_dir.name}/{pred_subdir} → {out_dir} =====")

    try:
        generate_report_json(fold_dir, out_dir=out_dir)
    except Exception as e:
        print(f"[auto_report] gen_report_json 失败: {e}")

    try:
        run_eval_report(
            val_dir=fold_dir / pred_subdir,
            gt_dir=gt_dir,
            img_dir=img_dir,
            vis_slices=vis_slices,
            no_vis=no_vis,
            min_tumor_size=min_tumor_size,
            out_dir=out_dir,
        )
    except Exception as e:
        print(f"[auto_report] eval_fold_report 失败: {e}")

    print(f"[auto_report] ===== 报告完成 =====\n")
