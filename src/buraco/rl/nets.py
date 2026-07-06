"""Policy/value network and the masked action distribution."""

from __future__ import annotations

import torch
from torch import Tensor, nn

# Finite fill instead of -inf: -inf through softmax NaNs on MPS, and -inf - -inf
# in ratio/entropy terms is NaN even on CPU. exp(-1e9) underflows to exactly 0.
MASK_FILL = -1e9


def masked_dist(logits: Tensor, mask: Tensor) -> torch.distributions.Categorical:
    """Categorical over legal actions only; illegal ids get probability 0."""
    masked = logits.masked_fill(mask == 0, MASK_FILL)
    return torch.distributions.Categorical(logits=masked)


def _orthogonal(layer: nn.Linear, gain: float) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)
    return layer


class PolicyValueNet(nn.Module):
    """Flat-concat MLP baseline: LayerNorm+GELU trunk, policy and value heads.

    The obs is already hand-featurized (meld slots carry engineered features),
    so an MLP over the canonical flat vector is the defensible PPO baseline.
    Upgrade path: swap the first layer for structured per-meld-slot / history
    encoders; trunk and heads stay unchanged.
    """

    def __init__(self, flat_dim: int, num_actions: int, hidden: int = 512, layers: int = 2):
        super().__init__()
        blocks: list[nn.Module] = []
        width = flat_dim
        for _ in range(layers):
            blocks += [
                _orthogonal(nn.Linear(width, hidden), gain=2**0.5),
                nn.LayerNorm(hidden),
                nn.GELU(),
            ]
            width = hidden
        self.trunk = nn.Sequential(*blocks)
        # Tiny policy gain gives a near-uniform initial masked policy.
        self.policy_head = _orthogonal(nn.Linear(hidden, num_actions), gain=0.01)
        self.value_head = _orthogonal(nn.Linear(hidden, 1), gain=1.0)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        h = self.trunk(x)
        return self.policy_head(h), self.value_head(h).squeeze(-1)
