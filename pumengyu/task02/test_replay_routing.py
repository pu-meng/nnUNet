"""CPU regressions: python -m unittest pumengyu.task02.test_replay_routing -v."""

import copy
import unittest
import tempfile
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint, set_checkpoint_early_stop

from pumengyu.architectures.mla_unetr import MoEFFN
from pumengyu.task02.stage2_mixin import Task02Stage2Mixin


class Network(nn.Module):
    def __init__(self, checkpoint_mode=None):
        super().__init__()
        self.moe = MoEFFN(4, mlp_ratio=2)
        self.checkpoint_mode = checkpoint_mode
        with torch.no_grad():
            self.moe.router.weight.copy_(torch.eye(4))

    def forward(self, x):
        if self.checkpoint_mode is None:
            return self.moe(x)
        # Force full recomputation to exercise the pending-count overwrite.
        with set_checkpoint_early_stop(False):
            return checkpoint(self.moe, x, use_reentrant=self.checkpoint_mode)


class Harness(Task02Stage2Mixin):
    def __init__(self, network):
        self.network = network
        self.device = torch.device('cpu')
        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.SGD(network.parameters(), lr=0.01)
        self.grad_scaler = None
        self.commits = 0
        self.submitted = None

    def _commit_deferred_network_updates(self):
        self.commits += 1
        network = self.network.module if hasattr(self.network, 'module') else self.network
        self.submitted = network.moe._pending_expert_counts.clone()
        network.moe.commit_expert_bias_update()


class MixedCheckpointNetwork(Network):
    """MedNeXt-like mixture of ordinary, nonreentrant and reentrant layers."""

    def __init__(self):
        super().__init__()
        self.stem = nn.Linear(4, 4)
        self.head = nn.Linear(4, 4)
        self.checkpoint_mode = True
        with torch.no_grad():
            self.stem.weight.copy_(torch.eye(4))
            self.stem.bias.zero_()

    def forward(self, x):
        x = self.stem(x)
        if self.checkpoint_mode is None:
            return self.head(self.moe(x))
        x = checkpoint(self.moe, x, use_reentrant=False)
        return checkpoint(self.head, x, use_reentrant=True)


def ddp_worker(rank, init_file, old_path):
    from datetime import timedelta
    from torch import distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + init_file,
                            rank=rank, world_size=2, timeout=timedelta(seconds=45))
    try:
        torch.manual_seed(123)
        network = MixedCheckpointNetwork()
        reference = copy.deepcopy(network)
        reference.checkpoint_mode = None
        wrapped = DDP(network)
        trainer = Harness(wrapped)
        hcc = torch.tensor([[[4., 3., 0., 0.]] * (rank + 2)], requires_grad=True)
        lits = torch.tensor([[[0., 0., 4., 3.]] * (rank + 3)], requires_grad=True)
        hcc_target = torch.ones_like(hcc) * rank
        lits_target = torch.ones_like(lits) * (rank + 1)
        if old_path:
            loss = trainer.loss(wrapped(hcc), hcc_target) + trainer.loss(wrapped(lits), lits_target)
            try:
                loss.backward()
            except RuntimeError as error:
                assert 'mark a variable ready only once' in str(error), str(error)
            else:
                raise AssertionError('Old two-forward replay did not reproduce the DDP error')
            return
        reference_optimizer = torch.optim.SGD(reference.parameters(), lr=.01)
        for _ in range(2):
            reference_optimizer.zero_grad(set_to_none=True)
            loss = trainer.loss(reference(hcc), hcc_target) + trainer.loss(reference(lits), lits_target)
            loss.backward()
            for p in reference.parameters():
                dist.all_reduce(p.grad)
                p.grad.div_(2)
            torch.nn.utils.clip_grad_norm_(reference.parameters(), 12)
            reference_optimizer.step()
            trainer._task02_replay_step({'data': hcc, 'target': hcc_target},
                                       {'data': lits, 'target': lits_target})
            torch.testing.assert_close(trainer.submitted,
                                       torch.tensor([rank+2., rank+2., rank+3., rank+3.]))
            assert network.moe._pending_expert_counts is None
            for p, ref in zip(network.parameters(), reference.parameters()):
                torch.testing.assert_close(p.grad, ref.grad)
                torch.testing.assert_close(p, ref)
            # Align reference routing buffers for the next update.
            reference.moe.expert_bias.copy_(network.moe.expert_bias)
            reference.moe.expert_load_ema.copy_(network.moe.expert_load_ema)
        assert trainer.commits == 2
    finally:
        dist.destroy_process_group()


class ReplayRoutingTest(unittest.TestCase):
    def test_real_two_process_ddp_old_failure_and_fixed_gradients(self):
        for old_path in (True, False):
            with self.subTest(old_path=old_path), tempfile.TemporaryDirectory(prefix='task02-ddp-') as folder:
                torch.multiprocessing.spawn(ddp_worker, args=(folder + '/init', old_path),
                                            nprocs=2, join=True)

    def test_replay_counts_gradients_and_checkpoint_recomputation(self):
        torch.manual_seed(42)
        original = Network()
        # Unequal token counts and disjoint routes detect last-domain-only counts.
        hcc = torch.tensor([[[4., 3., 0., 0.]] * 3], requires_grad=True)
        lits = torch.tensor([[[0., 0., 4., 3.]] * 5], requires_grad=True)
        expected = torch.tensor([3., 3., 5., 5.])
        reference = copy.deepcopy(original)
        loss = reference(hcc).square().mean() + reference(lits).square().mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(reference.parameters(), 12)
        reference_grads = [p.grad.clone() for p in reference.parameters()]
        for mode in (None, False, True):
            with self.subTest(checkpoint_mode=mode):
                network = copy.deepcopy(original)
                network.checkpoint_mode = mode
                trainer = Harness(network)
                trainer._task02_replay_step(
                    {'data': hcc, 'target': torch.zeros_like(hcc)},
                    {'data': lits, 'target': torch.zeros_like(lits)},
                )
                torch.testing.assert_close(trainer.submitted, expected)
                self.assertEqual(trainer.commits, 1)
                self.assertIsNone(network.moe._pending_expert_counts)
                ema = torch.full((4,), .25) * .99 + expected / expected.sum() * .01
                torch.testing.assert_close(network.moe.expert_load_ema, ema)
                torch.testing.assert_close(network.moe.expert_bias, (.25 - ema) * .001)
                for p, grad in zip(network.parameters(), reference_grads):
                    torch.testing.assert_close(p.grad, grad)
                bias = network.moe.expert_bias.clone()
                network.moe.commit_expert_bias_update()
                torch.testing.assert_close(network.moe.expert_bias, bias)
                trainer._task02_replay_step(
                    {'data': hcc, 'target': torch.zeros_like(hcc)},
                    {'data': lits, 'target': torch.zeros_like(lits)},
                )
                torch.testing.assert_close(trainer.submitted, expected)

    def test_eval_and_single_domain_unchanged(self):
        network = Network().eval()
        x = torch.randn(1, 3, 4)
        network(x)
        self.assertIsNone(network.moe._pending_expert_counts)
        network.train()
        network(x).sum().backward()
        self.assertEqual(network.moe._pending_expert_counts.sum().item(), 6)
        network.moe.commit_expert_bias_update()
        self.assertIsNone(network.moe._pending_expert_counts)

    def test_commit_aggregates_rank_counts_once(self):
        network = Network()
        trainer = Harness(network)
        hcc = torch.tensor([[[4., 3., 0., 0.]]])
        lits = torch.tensor([[[0., 0., 4., 3.]]])
        remote = torch.tensor([2., 2., 4., 4.])
        with patch('pumengyu.architectures.mla_unetr.dist.is_initialized', return_value=True), \
             patch('pumengyu.architectures.mla_unetr.dist.all_reduce',
                   side_effect=lambda counts, op: counts.add_(remote)) as reduce:
            trainer._task02_replay_step(
                {'data': hcc, 'target': torch.zeros_like(hcc)},
                {'data': lits, 'target': torch.zeros_like(lits)},
            )
        reduce.assert_called_once()
        total = torch.ones(4) + remote
        torch.testing.assert_close(network.moe.expert_load_ema,
                                   torch.full((4,), .25) * .99 + total / total.sum() * .01)


if __name__ == '__main__':
    unittest.main()
