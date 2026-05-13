from pumengyu.trainers.nnUNetTrainer_BATseg import nnUNetTrainer_BATseg


class BATseg_twostage(nnUNetTrainer_BATseg):
    """
    BATseg trainer for Dataset004 (two-stage liver-tumor pipeline).
    Identical to nnUNetTrainer_BATseg; separate class so the name resolves.
    """
    pass
