"""
对 SizeOversampleV2 的全量 131 个训练 case 跑推理，分析哪些 case 在训练集里也表现差。

结果保存到：
  /home/PuMengYu/nnUNet_workspace/analysis/SizeOversampleV2/full131_prediction/  (预测 nii.gz)
  /home/PuMengYu/nnUNet_workspace/analysis/SizeOversampleV2/full131_report_custom.txt

用法：
    cd /home/PuMengYu/nnUNet
    CUDA_VISIBLE_DEVICES=0 python pumengyu/tools/run_full131_predict.py
"""

import os
import sys
import subprocess
from pathlib import Path

NNUNET_ROOT = "/home/PuMengYu/nnUNet"
if NNUNET_ROOT not in sys.path:
    sys.path.insert(0, NNUNET_ROOT)

os.environ.setdefault("nnUNet_raw",          "/home/PuMengYu/nnUNet_workspace/raw")
os.environ.setdefault("nnUNet_preprocessed", "/home/PuMengYu/nnUNet_workspace/preprocessed")
os.environ.setdefault("nnUNet_results",      "/home/PuMengYu/nnUNet_workspace/results_v2")

INPUT_DIR   = "/home/PuMengYu/nnUNet_workspace/raw/Dataset003_Liver/imagesTr"
OUTPUT_DIR  = "/home/PuMengYu/nnUNet_workspace/analysis/SizeOversampleV2/full131_prediction"
GT_DIR      = "/home/PuMengYu/nnUNet_workspace/preprocessed/Dataset003_Liver/gt_segmentations"
IMG_DIR     = "/home/PuMengYu/nnUNet_workspace/raw/Dataset003_Liver/imagesTr"
ANALYSIS_DIR = "/home/PuMengYu/nnUNet_workspace/analysis/SizeOversampleV2"

if __name__ == "__main__":
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # ── 第一步：nnUNetv2_predict ─────────────────────────────────────────────────
    print("=" * 60)
    print("[1/3] 推理 131 个 case...")
    print("=" * 60)
    cmd = [
        "nnUNetv2_predict",
        "-i",  INPUT_DIR,
        "-o",  OUTPUT_DIR,
        "-d",  "Dataset003_Liver",
        "-c",  "3d_fullres",
        "-tr", "nnUNetTrainer_SizeOversampleV2",
        "-f",  "0",
        "-chk", "checkpoint_best.pth",
    ]
    subprocess.run(cmd, check=True)

    # ── 第二步：compute_metrics_on_folder 生成 summary.json ──────────────────────
    print("\n" + "=" * 60)
    print("[2/3] 计算指标，生成 summary.json...")
    print("=" * 60)

    from batchgenerators.utilities.file_and_folder_operations import load_json, join
    from nnunetv2.configuration import default_num_processes
    from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder
    from nnunetv2.paths import nnUNet_results
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    from nnunetv2.utilities.label_handling.label_handling import LabelManager

    result_dir = join(nnUNet_results, "Dataset003_Liver",
                      "nnUNetTrainer_SizeOversampleV2__nnUNetPlans__3d_fullres")
    plans        = load_json(join(result_dir, "plans.json"))
    dataset_json = load_json(join(result_dir, "dataset.json"))

    plans_manager  = PlansManager(plans)
    config_manager = plans_manager.get_configuration("3d_fullres")
    label_manager  = plans_manager.get_label_manager(dataset_json)

    summary_path = join(OUTPUT_DIR, "summary.json")
    compute_metrics_on_folder(
        folder_ref=GT_DIR,
        folder_pred=OUTPUT_DIR,
        output_file=summary_path,
        image_reader_writer=plans_manager.image_reader_writer_class(),
        file_ending=dataset_json["file_ending"],
        regions_or_labels=(
            label_manager.foreground_regions
            if label_manager.has_regions
            else label_manager.foreground_labels
        ),
        ignore_label=label_manager.ignore_label,
        chill=True,
        num_processes=default_num_processes,
    )
    print(f"summary.json 已写入: {summary_path}")

    # ── 第三步：生成可读报告 ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("[3/3] 生成报告...")
    print("=" * 60)

    from pumengyu.tools.analyasis.eval_fold_report import run_eval_report

    run_eval_report(
        val_dir=Path(OUTPUT_DIR),
        gt_dir=Path(GT_DIR),
        img_dir=Path(IMG_DIR),
        no_vis=True,
        min_tumor_size=0,
        out_dir=Path(ANALYSIS_DIR),
        report_name="full131_report_custom.txt",
    )

    print("\n全部完成！结果位置：")
    print(f"  预测文件: {OUTPUT_DIR}/")
    print(f"  报告:     {ANALYSIS_DIR}/full131_report_custom.txt")
