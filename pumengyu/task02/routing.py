"""Collect original replay forwards without counting backward recomputation."""

from pumengyu.architectures.mla_unetr import MoEFFN


class ReplayExpertCounts:
    """One instance per optimizer update; counts remain local until commit."""

    def __init__(self, network):
        self.modules = [m for m in network.modules() if isinstance(m, MoEFFN)]
        self.counts = {}

    def capture(self):
        """Call immediately after each original domain forward, before backward."""
        for module in self.modules:
            counts = module._pending_expert_counts
            if counts is None:
                raise RuntimeError("Replay forward did not record MoE expert counts.")
            if module in self.counts:
                self.counts[module].add_(counts)
            else:
                self.counts[module] = counts.detach().clone()
            module._pending_expert_counts = None

    def restore(self):
        """Replace any recomputation counts with the original-forward totals."""
        for module in self.modules:
            module._pending_expert_counts = self.counts[module]

