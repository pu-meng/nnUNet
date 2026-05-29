"""
对已训练好的模型补跑 internal test 推理并生成报告。

用法：
    python pumengyu/tools/run_internal_test.py \
        --trainer  nnUNetTrainer_Baseline \
        --dataset  Dataset003_Liver \
        --fold     0 \
        --gpu      0

生成结果：
    results_v2/.../fold_0/test_prediction/*.nii.gz
    results_v2/.../fold_0/test_prediction/summary.json
    results_v2/.../fold_0/test_prediction/report_custom.txt
"""

import argparse
import os
import sys

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
    parser.add_argument("--gpu",     default="0")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    import torch
    from batchgenerators.utilities.file_and_folder_operations import join, load_json
    from nnunetv2.paths import nnUNet_results
    from nnunetv2.utilities.find_class_by_name import recursive_find_python_class

    trainer_class = recursive_find_python_class(
        join(NNUNET_ROOT, "pumengyu", "trainers"),
        args.trainer,
        current_module="pumengyu.trainers",
    )
    if trainer_class is None:
        trainer_class = recursive_find_python_class(
            join(NNUNET_ROOT, "nnunetv2", "training", "nnUNetTrainer"),
            args.trainer,
            current_module="nnunetv2.training.nnUNetTrainer",
        )
    if trainer_class is None:
        raise RuntimeError(f"找不到 Trainer 类: {args.trainer}")

    plans_file = join(
        nnUNet_results, args.dataset,
        f"{args.trainer}__{args.plans}__{args.config}", "plans.json",
    )
    dataset_json_file = join(
        nnUNet_results, args.dataset,
        f"{args.trainer}__{args.plans}__{args.config}", "dataset.json",
    )

    plans        = load_json(plans_file)
    dataset_json = load_json(dataset_json_file)
    plans["continue_training"] = False  # run_training.py 正常注入，离线需手动补

    trainer = trainer_class(
        plans=plans,
        configuration=args.config,
        fold=args.fold,
        dataset_json=dataset_json,
        device=torch.device("cuda", 0),
    )
    trainer.initialize()

    ckpt_path = join(trainer.output_folder, "checkpoint_best.pth")
    print(f"[run_internal_test] 加载 checkpoint: {ckpt_path}")
    trainer.load_checkpoint(ckpt_path)

    print("[run_internal_test] 开始推理...")
    trainer._run_internal_test_prediction(save_probabilities=False)
    print("[run_internal_test] 全部完成。")


if __name__ == "__main__":
    main()
