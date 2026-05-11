import os
import sys
from datetime import datetime

import torch
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p

from nnunetv2.paths import nnUNet_raw
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

EXP_ROOT = "/home/PuMengYu/nnUNet_workspace/experiments"


class nnUNetTrainer_Exp(nnUNetTrainer):
    """
    Saves each training run to a timestamped subdirectory so runs never overwrite each other.

    Directory layout:
      <EXP_ROOT>/<dataset>/<trainer>__<plans>__<config>/fold_<n>/<timestamp>/
        checkpoint_best.pth
        checkpoint_latest.pth
        training_log_*.txt
        progress.png
        debug.json
        cmd.txt   <- full command used to launch this run

    New run:
      nnUNetv2_train 3 3d_fullres 1 -tr nnUNetTrainer_Exp

    Resume a specific run (set NNUNET_EXP_TS to the timestamp you want to continue):
      NNUNET_EXP_TS=05-04-17-36 nnUNetv2_train 3 3d_fullres 1 -tr nnUNetTrainer_Exp --c
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)

        resume_ts = os.environ.get("NNUNET_EXP_TS", "").strip()
        timestamp = resume_ts if resume_ts else datetime.now().strftime("%m-%d-%H-%M")

        exp_dir = join(
            EXP_ROOT,
            self.plans_manager.dataset_name,
            f"nnUNetTrainer_Exp__{self.plans_manager.plans_name}__{configuration}",
            f"fold_{fold}",
            timestamp,
        )
        maybe_mkdir_p(exp_dir)

        self.output_folder_base = exp_dir
        self.output_folder = exp_dir

        if not resume_ts:
            with open(join(exp_dir, "cmd.txt"), "w") as f:
                f.write(" ".join(sys.argv) + "\n")

    def perform_actual_validation(self, save_probabilities: bool = False):
        super().perform_actual_validation(save_probabilities)
        if self.local_rank == 0:
            from pumengyu.tools.analyasis.auto_report import run_auto_report
            run_auto_report(
                fold_dir=self.output_folder,
                gt_dir=join(self.preprocessed_dataset_folder_base, "gt_segmentations"),
                img_dir=join(str(nnUNet_raw), self.plans_manager.dataset_name, "imagesTr"),
            )
