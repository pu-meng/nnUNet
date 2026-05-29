"""
对已存在的 test_prediction/ 预测结果补生成 summary.json + report_custom.txt。
预测文件已存在时用这个，不需要重新推理。

用法：
    python pumengyu/tools/gen_test_report.py \
        --trainer nnUNetTrainer_Baseline \
        --dataset Dataset003_Liver \
        --fold    0
"""

import argparse
import os
import sys
from pathlib import Path

NNUNET_ROOT = "/home/PuMengYu/nnUNet"
if NNUNET_ROOT not in sys.path:
    sys.path.insert(0, NNUNET_ROOT)

os.environ.setdefault("nnUNet_raw",          "/home/PuMengYu/nnUNet_workspace/raw")
os.environ.setdefault("nnUNet_preprocessed", "/home/PuMengYu/nnUNet_workspace/preprocessed")
os.environ.setdefault("nnUNet_results",      "/home/PuMengYu/nnUNet_workspace/results_v2")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainer", default="nnUNetTrainer_Baseline")
    parser.add_argument("--dataset", default="Dataset003_Liver")
    parser.add_argument("--fold",    type=int, default=0)
    parser.add_argument("--plans",   default="nnUNetPlans")
    parser.add_argument("--config",  default="3d_fullres")
    args = parser.parse_args()

    from batchgenerators.utilities.file_and_folder_operations import join, load_json
    from nnunetv2.configuration import default_num_processes
    from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder
    from nnunetv2.paths import nnUNet_preprocessed, nnUNet_raw, nnUNet_results
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    from nnunetv2.utilities.label_handling.label_handling import LabelManager
    from pumengyu.tools.analyasis.eval_fold_report import run_eval_report

    result_dir = join(
        nnUNet_results, args.dataset,
        f"{args.trainer}__{args.plans}__{args.config}",
    )
    test_pred_dir = join(result_dir, f"fold_{args.fold}", "test_prediction")
    assert os.path.isdir(test_pred_dir), f"目录不存在: {test_pred_dir}"

    plans        = load_json(join(result_dir, "plans.json"))
    dataset_json = load_json(join(result_dir, "dataset.json"))

    plans_manager = PlansManager(plans)
    config_manager = plans_manager.get_configuration(args.config)
    label_manager  = plans_manager.get_label_manager(dataset_json)

    gt_dir  = Path(nnUNet_preprocessed) / args.dataset / "gt_segmentations"
    img_dir = Path(nnUNet_raw)          / args.dataset / "imagesTr"

    summary_path = join(test_pred_dir, "summary.json")
    print(f"[gen_test_report] 计算指标 → {summary_path}")
    compute_metrics_on_folder(
        folder_ref=str(gt_dir),
        folder_pred=test_pred_dir,
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

    fold_dir = Path(test_pred_dir).parent
    print(f"[gen_test_report] 生成报告 → {fold_dir}/test_report_custom.txt")
    run_eval_report(
        val_dir=Path(test_pred_dir),
        gt_dir=gt_dir,
        img_dir=img_dir,
        no_vis=True,
        min_tumor_size=0,
        out_dir=fold_dir,
        report_name="test_report_custom.txt",
    )
    print("[gen_test_report] 完成！")


if __name__ == "__main__":
    main()
