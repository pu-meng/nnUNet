from pathlib import Path
import os
import subprocess
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from pumengyu.mixins import SmallTumorOversampleMixin, BboxJitterMixin, AutoReportMixin


class nnUNetTrainer_TwoStage(SmallTumorOversampleMixin, AutoReportMixin, nnUNetTrainer):
    """
    Stage-2 baseline trainer (liver ROI crop → tumor segmentation).
    Identical to nnUNetTrainer; separate class so the results folder is
    named nnUNetTrainer_TwoStage__ instead of nnUNetTrainer__, making
    it easy to distinguish from single-stage and boundary-loss runs.
    """

    def perform_actual_validation(self, save_probabilities: bool = False):
        super().perform_actual_validation(save_probabilities)  # AutoReportMixin 已包含报告生成
        self._run_e2e_eval()

    def _run_e2e_eval(self):
        """训练结束后自动跑端到端两阶段推理评估（Stage1预测裁剪 → Stage2推理）"""
        eval_script = (Path(__file__).parent / "eval.py").resolve()
        if not eval_script.exists():
            self.print_to_log_file(f"[e2e_eval] 找不到 eval.py: {eval_script}")
            return

        # workspace = nnUNet_results 的上一级目录
        nnunet_results = os.environ.get(
            "nnUNet_results",
            str(Path(self.output_folder_base).parent.parent))  #type:ignore
        workspace = str(Path(nnunet_results).parent)

        device = str(self.device).split(":")[0]   # "cuda" or "cpu"

        cmd = [
            "python", str(eval_script),
            "--fold",           str(self.fold),
            "--workspace",      workspace,
            "--stage2_trainer", type(self).__name__,
            "--device",         device,
        ]
        self.print_to_log_file(f"[e2e_eval] 运行: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.stdout:
            self.print_to_log_file(result.stdout)
        if result.returncode != 0:
            self.print_to_log_file(
                f"[e2e_eval] 失败 (rc={result.returncode}):\n{result.stderr}")
        else:
            self.print_to_log_file("[e2e_eval] 端到端评估完成 → eval_e2e.txt")


class nnUNetTrainer_TwoStageJitter(SmallTumorOversampleMixin, BboxJitterMixin, AutoReportMixin, nnUNetTrainer):
    """
    Stage2 + Stage-aware Crop Jitter。

    在 nnUNetTrainer_TwoStage 基础上加入 BboxJitterMixin：
    训练时随机对图像边界置零（模拟 Stage1 预测框偏差），
    弥合 GT-crop 训练与 Stage1-predicted-crop 推理之间的 distribution gap。

    结果目录：nnUNetTrainer_TwoStageJitter__nnUNetPlans__3d_fullres/
    """

    def perform_actual_validation(self, save_probabilities: bool = False):
        super().perform_actual_validation(save_probabilities)
        self._run_e2e_eval()

    def _run_e2e_eval(self):
        eval_script = (Path(__file__).parent / "eval.py").resolve()
        if not eval_script.exists():
            self.print_to_log_file(f"[e2e_eval] 找不到 eval.py: {eval_script}")
            return

        nnunet_results = os.environ.get(
            "nnUNet_results",
            str(Path(self.output_folder_base).parent.parent))  # type: ignore
        workspace = str(Path(nnunet_results).parent)
        device    = str(self.device).split(":")[0]

        cmd = [
            "python", str(eval_script),
            "--fold",           str(self.fold),
            "--workspace",      workspace,
            "--stage2_trainer", type(self).__name__,
            "--device",         device,
        ]
        self.print_to_log_file(f"[e2e_eval] 运行: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.stdout:
            self.print_to_log_file(result.stdout)
        if result.returncode != 0:
            self.print_to_log_file(
                f"[e2e_eval] 失败 (rc={result.returncode}):\n{result.stderr}")
        else:
            self.print_to_log_file("[e2e_eval] 端到端评估完成 → eval_e2e.txt")

#nnUNet得结果目录名是自动按照{Trainer}_{Plans}_{Configuration}命名得