#!/usr/bin/env python
"""CPU-only verification for the Task 02 C arms (C1 frozen encoder, C2 LoRA).

Never trains, never initialises CUDA, never touches an existing results root and
never modifies an existing file. It exercises exactly the production code path in
``pumengyu.task02.c_arm_mixin`` plus nnU-Net's own trainer-discovery and
trainer-construction machinery.

C1 trains only the decoder + segmentation head. C2 is the pure LoRA arm: rank-8
bypasses on every encoder+decoder Conv3d/ConvTranspose3d/Linear except the MoE
gating ``router`` (hard top-k readout, provably zero gradient), with every base
weight -- including the segmentation heads -- frozen.

Run (all three paths are read-only; ``nnUNet_results`` points at a fresh scratch
root so that nothing existing is written):

    source ~/anaconda3/etc/profile.d/conda.sh && conda activate medseg
    export nnUNet_raw=/home/PuMengYu/nnUNet_workspace/raw
    export nnUNet_preprocessed=/home/PuMengYu/nnUNet_workspace/preprocessed
    export nnUNet_results=/home/PuMengYu/c_arm_scratch/verify_results
    export nnUNet_extTrainer=/home/PuMengYu/nnUNet/pumengyu
    export CUDA_VISIBLE_DEVICES=""
    cd /home/PuMengYu/nnUNet && python -m pumengyu.task02.verify_c_arms

Optional env: ``C_ARM_SMOKE_PATCH=16,16,16`` to shrink the synthetic patch.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import sys
import traceback

import torch

C0_NAME = "nnUNetTrainer_MedNeXt_MHA_MoE_Task02_HCCOnly_K03"
C1_NAME = "nnUNetTrainer_MedNeXt_MHA_MoE_Task02_HCCOnly_K03_FrozenEncoder"
C2_NAME = "nnUNetTrainer_MedNeXt_MHA_MoE_Task02_HCCOnly_K03_LoRA"
DATASET = "Dataset003_Liver"
CONFIGURATION = "3d_fullres"
FOLD = 0

FAILURES: list[str] = []


def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78, flush=True)


def check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"[{'PASS' if ok else 'FAIL'}] {label}" + (f" | {detail}" if detail else ""), flush=True)
    if not ok:
        FAILURES.append(label)
    return ok


def smoke_patch() -> tuple[int, int, int]:
    raw = os.environ.get("C_ARM_SMOKE_PATCH", "32,32,32")
    return tuple(int(v) for v in raw.split(","))  # type: ignore[return-value]


def print_table(stats: dict, title: str) -> None:
    print(f"\n--- {title}")
    print(f"{'region':<22}{'tensors':>9}{'params':>15}{'trainable tensors':>19}{'trainable params':>18}")
    for region, entry in sorted(stats["per_region"].items()):
        print(
            f"{region:<22}{entry['tensors']:>9}{entry['params']:>15,}"
            f"{entry['trainable_tensors']:>19}{entry['trainable_params']:>18,}"
        )
    print(
        f"{'TOTAL':<22}{'':>9}{stats['total_params']:>15,}"
        f"{'':>19}{stats['trainable_params']:>18,}"
    )
    print(
        f"trainable/total = {stats['trainable_params']:,} / {stats['total_params']:,} "
        f"= {100.0 * stats['trainable_ratio']:.4f}%   (frozen {stats['frozen_params']:,})"
    )


# ------------------------------------------------------------------------- #
# 0. environment + real plans
# ------------------------------------------------------------------------- #
def load_plans_bundle():
    from nnunetv2.paths import nnUNet_preprocessed
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    from batchgenerators.utilities.file_and_folder_operations import join, load_json

    folder = join(str(nnUNet_preprocessed), DATASET)
    plans_manager = PlansManager(join(folder, "nnUNetPlans.json"))
    dataset_json = load_json(join(folder, "dataset.json"))
    configuration_manager = plans_manager.get_configuration(CONFIGURATION)
    return plans_manager, configuration_manager, dataset_json, folder


def build_network(trainer_cls, plans_manager, configuration_manager, dataset_json, seed: int = 20260920):
    from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels

    label_manager = plans_manager.get_label_manager(dataset_json)
    num_input_channels = determine_num_input_channels(plans_manager, configuration_manager, dataset_json)
    torch.manual_seed(seed)
    network = trainer_cls.build_network_architecture(
        plans_manager,
        configuration_manager,
        num_input_channels,
        label_manager.num_segmentation_heads,
        True,
    )
    return network, num_input_channels, label_manager.num_segmentation_heads


# ------------------------------------------------------------------------- #
# 1. discovery
# ------------------------------------------------------------------------- #
def section_discovery() -> dict:
    section("1. nnU-Net trainer discovery (the exact lookup nnUNetv2_train uses)")
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
    from nnunetv2.utilities.find_objects import recursive_find_trainer_class_by_name

    print(f"nnUNet_extTrainer={os.environ.get('nnUNet_extTrainer')!r}")
    found = {}
    for name in (C0_NAME, C1_NAME, C2_NAME):
        cls = recursive_find_trainer_class_by_name(name)
        found[name] = cls
        check(
            f"recursive_find_trainer_class_by_name({name!r})",
            cls is not None and issubclass(cls, nnUNetTrainer),
            f"-> {cls.__module__}.{cls.__name__}  file={inspect.getfile(cls)}",
        )
        print(f"      MRO: {' -> '.join(c.__name__ for c in cls.__mro__[:5])}")

    # The external live path is <nnUNet_extTrainer>/trainers (module prefix
    # "pumengyu.trainers"), so the trainers copy is the authoritative one. Assert
    # that it is the file nnU-Net actually imported, and that the companion copy
    # under pumengyu/task02 is byte-identical (no stale duplicate can exist).
    discovered_file = os.path.realpath(inspect.getfile(found[C1_NAME]))
    authoritative = os.path.realpath("/home/PuMengYu/nnUNet/pumengyu/trainers/task02_c_arms.py")
    companion = os.path.realpath("/home/PuMengYu/nnUNet/pumengyu/task02/task02_c_arms.py")
    check(
        "the discovered C classes are defined in pumengyu/trainers/task02_c_arms.py",
        discovered_file == authoritative and found[C1_NAME].__module__ == "pumengyu.trainers.task02_c_arms",
        f"module={found[C1_NAME].__module__} file={discovered_file}",
    )
    if os.path.exists(companion):
        hashes = {}
        for path in (authoritative, companion):
            with open(path, "rb") as handle:
                hashes[path] = hashlib.sha256(handle.read()).hexdigest()
        check(
            "the two copies of task02_c_arms.py are byte-identical",
            len(set(hashes.values())) == 1,
            " ".join(f"{os.path.basename(os.path.dirname(p))}={h[:16]}" for p, h in hashes.items()),
        )
    return found


# ------------------------------------------------------------------------- #
# 2. parameter accounting on the network nnU-Net actually builds
# ------------------------------------------------------------------------- #
def section_accounting(plans_manager, configuration_manager, dataset_json):
    section("2. Parameter accounting on the real Dataset003 3d_fullres network")
    from pumengyu.task02.c_arm_mixin import (
        apply_frozen_encoder_arm,
        apply_lora_arm,
        collect_trainable_stats,
        describe_excluded_lora_candidates,
        describe_lora_targets,
        parameter_region,
        LORA_EXCLUDED_LEAF_NAMES,
        Task02C1FrozenEncoderMixin,
        Task02C2LoRAMixin,
    )
    from pumengyu.trainers.trainer import nnUNetTrainer_MedNeXt_MHA_MoE

    net_c0, nic, noc = build_network(
        nnUNetTrainer_MedNeXt_MHA_MoE, plans_manager, configuration_manager, dataset_json
    )
    print(f"network={type(net_c0).__name__}  num_input_channels={nic}  num_output_channels={noc}")
    stats_c0 = collect_trainable_stats(net_c0)
    print_table(stats_c0, "C0 reference (full fine-tuning, requires_grad all True)")

    # ---------------- C1 ----------------
    # The mixin methods are called unbound with the mixin class as ``self`` so the
    # verification exercises the *production* class attributes verbatim.
    c1_mixin_attrs = {
        "trainable_regions": Task02C1FrozenEncoderMixin.C_ARM_TRAINABLE_REGIONS,
    }
    print(f"\nC1 production config: {c1_mixin_attrs}")
    net_c1, _, _ = build_network(
        nnUNetTrainer_MedNeXt_MHA_MoE, plans_manager, configuration_manager, dataset_json
    )
    stats_c1 = Task02C1FrozenEncoderMixin._c_arm_transform(Task02C1FrozenEncoderMixin, net_c1)
    print_table(stats_c1, "C1 frozen encoder (production mixin _c_arm_transform)")

    encoder_offenders = [
        n for n, p in net_c1.named_parameters() if parameter_region(n) == "encoder" and p.requires_grad
    ]
    check(
        "C1 assert: every encoder parameter has requires_grad == False",
        not encoder_offenders,
        f"{stats_c1['per_region']['encoder']['params']:,} encoder params, 0 trainable",
    )
    c1_dec = stats_c1["per_region"]["decoder"]["trainable_params"]
    c1_head = stats_c1["per_region"]["head"]["trainable_params"]
    check(
        "C1 trainable == decoder + head + 1-element checkpoint sentinel",
        stats_c1["trainable_params"] == c1_dec + c1_head + 1,
        f"{stats_c1['trainable_params']:,} == {c1_dec:,} + {c1_head:,} + 1",
    )
    print(
        "  note: dummy_tensor is the 1-element reentrant-checkpoint sentinel; it must stay "
        "requires_grad=True (see c_arm_mixin docstring) and is never read in forward()."
    )

    # ---------------- C2 ----------------
    c2_mixin_attrs = {
        "rank": Task02C2LoRAMixin.C_ARM_LORA_RANK,
        "scale": Task02C2LoRAMixin.C_ARM_LORA_SCALE,
        "regions": Task02C2LoRAMixin.C_ARM_TARGET_REGIONS,
        "train_seg_heads": Task02C2LoRAMixin.C_ARM_TRAIN_SEG_HEADS,
        "exclude_leaf_names": Task02C2LoRAMixin.C_ARM_EXCLUDE_LEAF_NAMES,
    }
    print(f"\nC2 production config: {c2_mixin_attrs}")
    check(
        "C2 production config is the pure LoRA arm (segmentation heads frozen)",
        Task02C2LoRAMixin.C_ARM_TRAIN_SEG_HEADS is False,
        f"train_seg_heads={Task02C2LoRAMixin.C_ARM_TRAIN_SEG_HEADS}",
    )
    check(
        "C2 production config excludes exactly the MoE router leaf",
        tuple(Task02C2LoRAMixin.C_ARM_EXCLUDE_LEAF_NAMES) == ("router",),
        f"exclude_leaf_names={Task02C2LoRAMixin.C_ARM_EXCLUDE_LEAF_NAMES}",
    )

    net_c2, _, _ = build_network(
        nnUNetTrainer_MedNeXt_MHA_MoE, plans_manager, configuration_manager, dataset_json
    )
    targets = describe_lora_targets(net_c2)
    excluded = describe_excluded_lora_candidates(net_c2)
    enc_targets = [t for t in targets if parameter_region(t[0]) == "encoder"]
    dec_targets = [t for t in targets if parameter_region(t[0]) == "decoder"]
    print(
        f"\nLoRA candidates in encoder+decoder = {len(targets) + len(excluded)} "
        f"(wrapped {len(targets)}; excluded {len(excluded)})"
    )
    print(f"  wrapped: encoder {len(enc_targets)}, decoder {len(dec_targets)}")
    kinds = {}
    for name, _, _ in targets:
        kind = name.rsplit(".", 1)[-1]
        kinds[kind] = kinds.get(kind, 0) + 1
    print(f"  wrapped by attribute name: {kinds}")
    print("  EXCLUDED from LoRA coverage (explicit, not silent):")
    for item in excluded:
        print(f"    {item['layer']}  shape=({item['rows']}, {item['cols']})  reason={item['reason'][:60]}...")
    min_dim = min(min(rows, cols) for _, rows, cols in targets)
    print(f"  min(rows, cols) over all wrapped targets = {min_dim}")

    stats_c2 = Task02C2LoRAMixin._c_arm_transform(Task02C2LoRAMixin, net_c2)
    print_table(stats_c2, "C2 LoRA rank-8 (production mixin _c_arm_transform)")
    check(
        "C2 wrapped every non-excluded encoder+decoder Conv/Linear target",
        stats_c2["lora_wrapped_layers"] == len(targets),
        f"wrapped={stats_c2['lora_wrapped_layers']} discovered={len(targets)}",
    )
    check(
        "C2 coverage count is 223 wrapped layers (225 candidates - 2 MoE routers)",
        stats_c2["lora_wrapped_layers"] == 223 and len(excluded) == 2,
        f"wrapped={stats_c2['lora_wrapped_layers']}, excluded={[e['layer'] for e in excluded]}",
    )
    shape_limit = {name: (rows, cols) for name, rows, cols in targets}
    bad_rank = {
        name: (stats_c2["lora_effective_ranks"][name], min(rows, cols))
        for name, rows, cols in targets
        if stats_c2["lora_effective_ranks"][name] != min(8, rows, cols)
    }
    check(
        "C2 effective rank == min(8, rows, cols) for every target (no silent truncation)",
        not bad_rank,
        f"shape-limited layers (rank < 8 is the matrix maximum, not a truncation): "
        f"{ {n: min(*shape_limit[n]) for n in stats_c2['lora_rank_shape_limited_layers']} }",
    )
    check(
        "C2 no base (non-LoRA) parameter anywhere is trainable",
        all(
            not p.requires_grad
            for n, p in net_c2.named_parameters()
            if not (n.endswith(".lora_A") or n.endswith(".lora_B")) and n != "dummy_tensor"
        ),
        "encoder + decoder + heads + everything else frozen",
    )
    check(
        "C2 every segmentation head parameter has requires_grad == False",
        all(not p.requires_grad for n, p in net_c2.named_parameters() if parameter_region(n) == "head"),
        f"{stats_c2['per_region']['head']['params']:,} head params, "
        f"{stats_c2['per_region']['head']['trainable_params']:,} trainable",
    )
    lora_params = sum(
        p.numel() for n, p in net_c2.named_parameters() if n.endswith((".lora_A", ".lora_B"))
    )
    check(
        "C2 trainable == LoRA bypass tensors + 1-element checkpoint sentinel",
        stats_c2["trainable_params"] == lora_params + 1,
        f"{stats_c2['trainable_params']:,} == {lora_params:,} + 1 (train_seg_heads="
        f"{stats_c2['train_seg_heads']})",
    )

    # ---------------- checkpoint compatibility shim ----------------
    section("2a2. Reentrant-checkpoint bypass for fully frozen blocks (C1 blocker)")
    from pumengyu.task02.c_arm_mixin import should_bypass_checkpoint

    dummy = net_c1.dummy_tensor
    no_grad_x = torch.zeros(1)
    samples = {
        "encoder block enc_block_0[0] (frozen)": (net_c1.enc_block_0[0], True),
        "encoder down block down_0 (frozen)": (net_c1.down_0, True),
        "bottleneck block[0] (frozen)": (net_c1.bottleneck[0], True),
        "decoder up block up_0 (trainable)": (net_c1.up_0, False),
        "head out_0 (trainable)": (net_c1.out_0, False),
        "bound method _apply_bottleneck_context (owner has trainable params)": (
            net_c1._apply_bottleneck_context,
            False,
        ),
    }
    for label, (module, expected) in samples.items():
        observed = should_bypass_checkpoint(module, (no_grad_x, dummy))
        check(f"bypass decision for {label}", observed is expected, f"bypass={observed}")
    check(
        "checkpoint bypass shim installed",
        bool(stats_c1.get("checkpoint_bypass_installed")) and bool(stats_c2.get("checkpoint_bypass_installed")),
    )

    # ---------------- C2 zero-init identity + gradient flow ----------------
    section("2b. C2 identity at injection (B == 0 => bit-identical to the frozen source net)")
    patch = smoke_patch()
    x = torch.randn(1, nic, *patch)
    net_ref, _, _ = build_network(
        nnUNetTrainer_MedNeXt_MHA_MoE, plans_manager, configuration_manager, dataset_json
    )
    net_lora, _, _ = build_network(
        nnUNetTrainer_MedNeXt_MHA_MoE, plans_manager, configuration_manager, dataset_json
    )
    Task02C2LoRAMixin._c_arm_transform(Task02C2LoRAMixin, net_lora)
    net_ref.eval()
    net_lora.eval()
    with torch.no_grad():
        out_ref = net_ref(x)
        out_lora = net_lora(x)
    out_ref = out_ref if isinstance(out_ref, (list, tuple)) else [out_ref]
    out_lora = out_lora if isinstance(out_lora, (list, tuple)) else [out_lora]
    check(
        "C2 output is bit-identical to C0 before any optimizer step",
        len(out_ref) == len(out_lora)
        and all(torch.equal(a, b) for a, b in zip(out_ref, out_lora)),
        f"{len(out_ref)} deep-supervision outputs, shapes {[tuple(o.shape) for o in out_ref]}",
    )

    section("2c. gradient flow: exactly the intended parameters receive gradients")
    for tag, net in (("C1", net_c1), ("C2", net_c2)):
        net.train()
        # C1 froze its encoder, so C0's reentrant-checkpoint sentinel policy must apply identically.
        net.zero_grad(set_to_none=True)
        outs = net(x)
        outs = outs if isinstance(outs, (list, tuple)) else [outs]
        loss = sum(o.float().pow(2).mean() for o in outs)
        loss.backward()
        with_grad = {
            n for n, p in net.named_parameters() if p.grad is not None and float(p.grad.abs().sum()) != 0.0
        }
        regions = {}
        for n in with_grad:
            r = parameter_region(n)
            regions[r] = regions.get(r, 0) + 1
        print(f"  [{tag}] loss={float(loss):.6f}  params with non-zero grad = {len(with_grad)}  by region {regions}")
        if tag == "C1":
            check(
                "C1 backward: zero encoder parameters receive a gradient",
                not [n for n in with_grad if parameter_region(n) == "encoder"],
            )
            check(
                "C1 backward: decoder and head parameters do receive gradients",
                regions.get("decoder", 0) > 0 and regions.get("head", 0) > 0,
            )
        else:
            n_a = sum(1 for n, _ in net.named_parameters() if n.endswith(".lora_A"))
            n_b = sum(1 for n, _ in net.named_parameters() if n.endswith(".lora_B"))
            a_grad = [n for n in with_grad if n.endswith(".lora_A")]
            b_grad = [n for n in with_grad if n.endswith(".lora_B")]
            non_lora_grad = [n for n in with_grad if not n.endswith((".lora_A", ".lora_B"))]
            check(
                "C2 backward: every lora_A gradient is exactly zero at step 1 (B == 0 by construction)",
                not a_grad,
                f"{len(a_grad)}/{n_a} lora_A tensors have a non-zero gradient",
            )
            check(
                "C2 backward: EVERY wrapped lora_B tensor receives a non-zero gradient",
                len(b_grad) == n_b and n_b == 223,
                f"{len(b_grad)}/{n_b} lora_B tensors non-zero",
            )
            check(
                "C2 backward: the non-zero-gradient set is exactly the lora_B set (no frozen "
                "parameter anywhere, heads included, receives a gradient)",
                set(with_grad) == {n for n, _ in net.named_parameters() if n.endswith(".lora_B")},
                f"{len(non_lora_grad)} non-LoRA tensors had a gradient",
            )
            check(
                "C2 backward: the frozen segmentation heads receive no gradient",
                not [n for n in with_grad if parameter_region(n) == "head"],
            )
            head_grads = [
                n
                for n, p in net_c2.named_parameters()
                if parameter_region(n) == "head" and p.grad is not None
            ]
            check(
                "C2 backward: every head parameter has grad None",
                not head_grads,
                f"{len(head_grads)} head tensors with a non-None grad",
            )
    return stats_c0, stats_c1, stats_c2


# ------------------------------------------------------------------------- #
# 3. real trainer object + one CPU optimizer step
# ------------------------------------------------------------------------- #
def section_trainer_smoke(trainer_cls_names: dict):
    section("3. Real trainer construction + ONE CPU optimizer step (best effort)")
    from nnunetv2.run.run_training import get_trainer_from_args
    from pumengyu.task02.c_arm_mixin import parameter_region

    for name in (C1_NAME, C2_NAME):
        print(f"\n--- {name}")
        try:
            trainer = get_trainer_from_args(
                DATASET, CONFIGURATION, FOLD, name, "nnUNetPlans", False, torch.device("cpu")
            )
        except Exception as exc:
            check(f"{name}: get_trainer_from_args", False, repr(exc))
            continue
        check(
            f"{name}: nnU-Net builds the trainer object on CPU",
            True,
            f"device={trainer.device} mode={trainer.TASK02_MODE} k={trainer.TASK02_K} "
            f"epochs={trainer.num_epochs} updates/epoch={trainer.num_iterations_per_epoch} "
            f"lr={trainer.initial_lr}",
        )
        check(
            "protocol equality with C0 (mode/k/seed/lr/epochs/updates/val interval)",
            trainer.TASK02_MODE == "hcc_only"
            and trainer.TASK02_K == 3
            and trainer.TASK02_TRAINING_SEED == 20260920
            and trainer.initial_lr == 1e-3
            and trainer.num_epochs == 300
            and trainer.num_iterations_per_epoch == 250
            and trainer.TASK02_VALIDATION_INTERVAL == 3
            and trainer.TASK02_VALIDATION_ITERATIONS == 50,
            f"seed={trainer.TASK02_TRAINING_SEED} val_interval={trainer.TASK02_VALIDATION_INTERVAL} "
            f"val_steps={trainer.TASK02_VALIDATION_ITERATIONS} "
            f"source={trainer.TASK02_EXPECTED_SOURCE_TRAINER}",
        )
        try:
            from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p

            maybe_mkdir_p(trainer.output_folder)
            trainer.initialize()  # builds the network from the real plans; no data loading, no GPU
            check("trainer.initialize() on CPU", True, f"network={type(trainer.network).__name__}")

            # This is the production hook: SHA-checked strict source load, then arm.
            identity = trainer._task02_load_source_weights_strict()
            stats = trainer._c_arm_stats
            print(f"      frozen source: epoch={identity['epoch']} sha256={identity['sha256'][:16]}...")
            print_table(stats, f"{name}: after the real _task02_load_source_weights_strict hook")

            optimizer, scheduler = trainer.configure_optimizers()
            n_opt = sum(p.numel() for group in optimizer.param_groups for p in group["params"])
            check(
                "optimizer receives only trainable parameters",
                n_opt == stats["trainable_params"],
                f"optimizer holds {n_opt:,} == trainable {stats['trainable_params']:,}",
            )
            print(
                f"      optimizer=SGD(momentum={optimizer.param_groups[0]['momentum']}, "
                f"nesterov={optimizer.param_groups[0]['nesterov']}, "
                f"lr={optimizer.param_groups[0]['lr']}, "
                f"weight_decay={optimizer.param_groups[0]['weight_decay']}) "
                f"scheduler={type(scheduler).__name__}"
            )

            trainer.optimizer = optimizer
            trainer.lr_scheduler = scheduler
            patch = smoke_patch()
            data = torch.randn(1, trainer.num_input_channels, *patch)
            with torch.no_grad():
                probe = trainer.network(data)
            probe = probe if isinstance(probe, (list, tuple)) else [probe]
            target = [
                torch.randint(0, trainer.label_manager.num_segmentation_heads, (1, 1, *o.shape[2:])).float()
                for o in probe
            ]
            before = {n: p.detach().clone() for n, p in trainer.network.named_parameters() if p.requires_grad}
            frozen_before = {
                n: p.detach().clone() for n, p in trainer.network.named_parameters() if not p.requires_grad
            }
            out = trainer.train_step({"data": data, "target": target})
            print(f"      train_step -> {out}   (patch={patch}, {len(target)} deep-supervision targets)")
            changed = [
                n
                for n, p in trainer.network.named_parameters()
                if p.requires_grad and not torch.equal(before[n], p.detach())
            ]
            frozen_changed = [
                n
                for n, p in trainer.network.named_parameters()
                if not p.requires_grad and not torch.equal(frozen_before[n], p.detach())
            ]
            check(
                "one optimizer step updated trainable parameters",
                len(changed) > 0,
                f"{len(changed)}/{len(before)} trainable tensors changed",
            )
            check("no frozen parameter was modified by the optimizer step", not frozen_changed)
            if name.endswith("LoRA"):
                delta = max(
                    float(p.detach().abs().max())
                    for n, p in trainer.network.named_parameters()
                    if n.endswith(".lora_B")
                )
                check("LoRA lora_B became non-zero after one step", delta > 0.0, f"max|B|={delta:.3e}")
                check(
                    "C2: exactly the LoRA bypass tensors were updated",
                    all(n.endswith((".lora_A", ".lora_B")) for n in changed)
                    and len(changed) > 0,
                    f"{len(changed)} tensors changed, all of them LoRA bypasses",
                )
                head_changed = [
                    n
                    for n, p in trainer.network.named_parameters()
                    if parameter_region(n) == "head" and not torch.equal(frozen_before[n], p.detach())
                ]
                check(
                    "C2: the frozen segmentation heads were not modified by the optimizer step",
                    not head_changed,
                    f"{len(head_changed)} head tensors changed",
                )
                head_grads = [
                    n
                    for n, p in trainer.network.named_parameters()
                    if parameter_region(n) == "head" and p.grad is not None
                ]
                check("C2: the frozen segmentation heads have grad None", not head_grads)
        except Exception:
            print("      train_step smoke FAILED:")
            traceback.print_exc()
            FAILURES.append(f"{name}: CPU 1-iteration smoke")


def main() -> int:
    print(f"torch={torch.__version__}  cuda_available={torch.cuda.is_available()}")
    print(
        f"nnUNet_raw={os.environ.get('nnUNet_raw')}\n"
        f"nnUNet_preprocessed={os.environ.get('nnUNet_preprocessed')}\n"
        f"nnUNet_results={os.environ.get('nnUNet_results')}"
    )
    init_file = "/home/PuMengYu/nnUNet/pumengyu/trainers/__init__.py"
    print(
        f"pumengyu/trainers/__init__.py size = {os.path.getsize(init_file)} bytes "
        f"(0 == untouched; discovery needs no import there)"
    )
    print("pre-existing dirty files (NOT touched by this work; mtimes predate the verification):")
    for rel in (
        "pumengyu/trainers/trainer.py",
        "pumengyu/trainers/__init__.py",
        "pumengyu/task02/run_task02_stage2.sh",
        "README.md",
    ):
        path = os.path.join("/home/PuMengYu/nnUNet", rel)
        if os.path.exists(path):
            stat = os.stat(path)
            print(f"  {rel:<44} mtime={stat.st_mtime:.3f} size={stat.st_size}")
    plans_manager, configuration_manager, dataset_json, folder = load_plans_bundle()
    print(f"plans: {folder}/nnUNetPlans.json")

    section_discovery()
    section_accounting(plans_manager, configuration_manager, dataset_json)

    if os.environ.get("C_ARM_SKIP_TRAINER_SMOKE", "0") != "1":
        section_trainer_smoke({})
    else:
        print("\n[skipped] trainer smoke (C_ARM_SKIP_TRAINER_SMOKE=1)")

    section("SUMMARY")
    if FAILURES:
        print(f"{len(FAILURES)} CHECK(S) FAILED:")
        for item in FAILURES:
            print(f"  - {item}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
