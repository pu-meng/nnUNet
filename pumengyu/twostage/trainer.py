from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainer_TwoStage(nnUNetTrainer):
    """
    Stage-2 baseline trainer (liver ROI crop → tumor segmentation).
    Identical to nnUNetTrainer; separate class so the results folder is
    named nnUNetTrainer_TwoStage__ instead of nnUNetTrainer__, making
    it easy to distinguish from single-stage and boundary-loss runs.
    """
    pass
