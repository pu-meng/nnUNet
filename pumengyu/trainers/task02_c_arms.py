"""Task 02 Stage-2 C arms: C1 (frozen encoder) and C2 (LoRA). NEW FILE.

Nothing existing is modified. These two classes differ from the reference arm

    nnUNetTrainer_MedNeXt_MHA_MoE_Task02_HCCOnly_K03        (C0, full fine-tune)

in exactly one respect -- which parameters the optimizer trains:

    nnUNetTrainer_MedNeXt_MHA_MoE_Task02_HCCOnly_K03_FrozenEncoder   (C1)
    nnUNetTrainer_MedNeXt_MHA_MoE_Task02_HCCOnly_K03_LoRA            (C2)

Everything else is inherited unchanged from ``Task02Stage2Mixin``: mode
``hcc_only``, ``TASK02_K = 3``, the same K=3 support manifest
(``task02_splits_k03``), ``TASK02_TRAINING_SEED = 20260920``, ``lr = 1e-3``,
PolyLR over 300 epochs, 250 updates/epoch, the SHA-checked frozen source
checkpoint, the HCC-only single-batch update, and the dual-domain lightweight
checkpoint selection. The only further difference is that the optimizer now
receives only parameters with ``requires_grad=True`` (see
``pumengyu.task02.c_arm_mixin``); the SGD/PolyLR hyper-parameters themselves are
byte-identical to ``nnUNetTrainer.configure_optimizers``.

Discovery: ``~/.bashrc`` exports ``nnUNet_extTrainer=/home/PuMengYu/nnUNet/pumengyu``,
so ``nnunetv2.utilities.find_objects.recursive_find_trainer_class_by_name``
scans ``pumengyu/trainers/*.py`` recursively. No ``__init__.py`` change is
needed (``pumengyu/trainers/__init__.py`` is empty and stays empty).

Launch with ``-num_gpus 1``: the arm is applied after
``nnUNetTrainer.initialize()``, so a DDP wrap built before it would miss the
injected / re-enabled parameters (the mixin raises a clear error otherwise).
``--c`` / ``-pretrained_weights`` are not part of the frozen Task 02 protocol
and are unsupported for these arms.
"""

from pumengyu.task02.c_arm_mixin import (
    Task02C1FrozenEncoderMixin,
    Task02C2LoRAMixin,
)
from pumengyu.task02.stage2_mixin import Task02Stage2Mixin
from pumengyu.trainers.trainer import nnUNetTrainer_MedNeXt_MHA_MoE


class nnUNetTrainer_MedNeXt_MHA_MoE_Task02_HCCOnly_K03_FrozenEncoder(
    Task02C1FrozenEncoderMixin, Task02Stage2Mixin, nnUNetTrainer_MedNeXt_MHA_MoE
):
    """Task02 k=3 HCC-only, encoder frozen; decoder + segmentation heads trained.

    C1 of the trainable-scope study. Reference arm (full fine-tuning) is
    ``nnUNetTrainer_MedNeXt_MHA_MoE_Task02_HCCOnly_K03``.
    """

    TASK02_MODE = "hcc_only"
    TASK02_K = 3


class nnUNetTrainer_MedNeXt_MHA_MoE_Task02_HCCOnly_K03_LoRA(
    Task02C2LoRAMixin, Task02Stage2Mixin, nnUNetTrainer_MedNeXt_MHA_MoE
):
    """Task02 k=3 HCC-only, rank-8 LoRA on encoder+decoder Conv/Linear.

    C2 of the trainable-scope study: pure LoRA. All backbone weights, biases,
    normalization affines and the 5 deep-supervision heads are frozen; the MoE
    gating ``router`` linears are excluded from the LoRA coverage because their
    hard top-k readout delivers exactly zero gradient. ``A ~ N(0, 1/fan_in)``
    and ``B == 0``, so the arm starts bit-identical to the frozen source
    checkpoint.
    """

    TASK02_MODE = "hcc_only"
    TASK02_K = 3
