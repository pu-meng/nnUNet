import torch
import torch.nn as nn
from copy import deepcopy
from torch.utils.checkpoint import checkpoint
from unittest.mock import patch

from pumengyu.architectures.mla_unetr import MoEFFN
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


def test_forward_records_load_without_mutating_routing_state():
    moe = MoEFFN(d_model=8, mlp_ratio=2, num_routed_experts=4, top_k=2)
    moe.train()
    x = torch.randn(2, 5, 8, requires_grad=True)
    bias_before = moe.expert_bias.clone()
    ema_before = moe.expert_load_ema.clone()

    moe(x)

    torch.testing.assert_close(moe.expert_bias, bias_before)
    torch.testing.assert_close(moe.expert_load_ema, ema_before)
    assert moe._pending_expert_counts is not None
    assert moe._pending_expert_counts.sum().item() == 2 * 5 * 2


def test_checkpoint_replay_keeps_state_fixed_until_single_commit():
    torch.manual_seed(0)
    moe = MoEFFN(d_model=8, mlp_ratio=2, num_routed_experts=4, top_k=2)
    moe.train()
    x = torch.randn(2, 5, 8, requires_grad=True)
    bias_before = moe.expert_bias.clone()
    ema_before = moe.expert_load_ema.clone()

    checkpoint(moe, x, use_reentrant=False).sum().backward()

    # Both the original forward and checkpoint replay saw exactly the same
    # immutable routing state. The replay only overwrote transient counts.
    torch.testing.assert_close(moe.expert_bias, bias_before)
    torch.testing.assert_close(moe.expert_load_ema, ema_before)
    pending = moe._pending_expert_counts.clone()

    moe.commit_expert_bias_update()
    expected_load = pending / pending.sum()
    expected_ema = ema_before * 0.99 + expected_load * 0.01
    expected_bias = bias_before + (0.25 - expected_ema) * 1e-3
    torch.testing.assert_close(moe.expert_load_ema, expected_ema)
    torch.testing.assert_close(moe.expert_bias, expected_bias)
    assert moe._pending_expert_counts is None

    # A second commit in the same iteration is a no-op.
    moe.commit_expert_bias_update()
    torch.testing.assert_close(moe.expert_load_ema, expected_ema)
    torch.testing.assert_close(moe.expert_bias, expected_bias)


def test_checkpointed_and_direct_training_match():
    torch.manual_seed(1)
    direct = MoEFFN(d_model=8, mlp_ratio=2, num_routed_experts=4, top_k=2)
    replayed = deepcopy(direct)
    x_direct = torch.randn(2, 5, 8, requires_grad=True)
    x_replayed = x_direct.detach().clone().requires_grad_(True)

    direct(x_direct).sum().backward()
    checkpoint(replayed, x_replayed, use_reentrant=False).sum().backward()
    direct.commit_expert_bias_update()
    replayed.commit_expert_bias_update()

    torch.testing.assert_close(x_direct.grad, x_replayed.grad)
    for direct_param, replayed_param in zip(direct.parameters(), replayed.parameters()):
        torch.testing.assert_close(direct_param.grad, replayed_param.grad)
    torch.testing.assert_close(direct.expert_load_ema, replayed.expert_load_ema)
    torch.testing.assert_close(direct.expert_bias, replayed.expert_bias)


def test_eval_forward_neither_records_nor_updates_routing_state():
    moe = MoEFFN(d_model=8, mlp_ratio=2, num_routed_experts=4, top_k=2)
    moe.eval()
    bias_before = moe.expert_bias.clone()
    ema_before = moe.expert_load_ema.clone()

    with torch.no_grad():
        moe(torch.randn(2, 5, 8))

    assert moe._pending_expert_counts is None
    torch.testing.assert_close(moe.expert_bias, bias_before)
    torch.testing.assert_close(moe.expert_load_ema, ema_before)


def test_trainer_commits_deferred_moe_state_after_backward():
    moe = MoEFFN(d_model=8, mlp_ratio=2, num_routed_experts=4, top_k=2)
    network = nn.Sequential(moe)
    trainer = object.__new__(nnUNetTrainer)
    trainer.network = network

    network(torch.randn(2, 5, 8, requires_grad=True)).sum().backward()
    ema_before = moe.expert_load_ema.clone()
    trainer._commit_deferred_network_updates()

    assert moe._pending_expert_counts is None
    assert not torch.equal(moe.expert_load_ema, ema_before)


def test_ddp_commit_uses_global_expert_counts():
    moe = MoEFFN(d_model=8, mlp_ratio=2, num_routed_experts=4, top_k=2)
    moe._pending_expert_counts = torch.tensor([8.0, 4.0, 0.0, 0.0])
    remote_counts = torch.tensor([0.0, 0.0, 4.0, 8.0])

    def fake_all_reduce(counts, op):
        counts.add_(remote_counts)

    with (
        patch('pumengyu.architectures.mla_unetr.dist.is_available', return_value=True),
        patch('pumengyu.architectures.mla_unetr.dist.is_initialized', return_value=True),
        patch('pumengyu.architectures.mla_unetr.dist.all_reduce', side_effect=fake_all_reduce),
    ):
        moe.commit_expert_bias_update()

    expected_global_load = torch.tensor([8.0, 4.0, 4.0, 8.0]) / 24.0
    expected_ema = torch.full((4,), 0.25) * 0.99 + expected_global_load * 0.01
    torch.testing.assert_close(moe.expert_load_ema, expected_ema)
