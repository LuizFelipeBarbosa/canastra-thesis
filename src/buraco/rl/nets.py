"""Policy/value networks (flat MLP and structured encoder) and the masked
action distribution. `build_net`/`net_config` are the only construction path
callers should use: they keep the architecture choice, its hyperparameters,
and the ObsSpec field layout together in one serializable dict, so
checkpoints, pool members, and parallel workers all rebuild identical nets."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from buraco.rl.obs import ObsSpec

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


# Fields the structured encoder handles specially; everything else in the
# ObsSpec is concatenated into the scalar block, so new observation fields
# degrade gracefully instead of being dropped.
_CARD_COUNT_FIELDS = ("hand", "trash_counts")
_MELD_FIELDS = ("own_melds", "opp_melds")


class StructuredPolicyValueNet(nn.Module):
    """Structured first layer over the same flat observation vector.

    The flat vector stays the transport format (collector/buffer/workers are
    untouched); this module slices it by ObsSpec offsets and encodes each
    group with the right inductive bias: one shared card-embedding table for
    every card-identity feature (hand counts, trash counts, trash top-k,
    pending card, history cards), a weight-shared per-slot MLP with
    occupancy-masked pooling for the meld blocks, and a small MLP for the
    scalar remainder. Trunk and heads match the MLP baseline.
    """

    def __init__(
        self,
        spec: ObsSpec,
        num_actions: int,
        hidden: int = 512,
        layers: int = 2,
        embed_dim: int = 64,
    ):
        super().__init__()
        self._fields = {f.name: f for f in spec.fields}
        for name in (*_CARD_COUNT_FIELDS, *_MELD_FIELDS, "trash_top_k",
                     "pending_pile_card", "history"):
            if name not in self._fields:
                raise ValueError(f"ObsSpec is missing field {name!r}")
        d = embed_dim
        # pending_pile_card is one card one-hot: its flat size IS card space.
        self.card_space = self._fields["pending_pile_card"].flat_size
        self.top_k = self._fields["trash_top_k"].shape[0]
        hist_len, hist_width = self._fields["history"].shape
        self.hist_meta = hist_width - self.card_space
        slots, slot_width = self._fields["own_melds"].shape

        self.card_embed = nn.Parameter(torch.randn(self.card_space, d) * 0.02)
        self.count_proj = nn.ModuleDict(
            {name: _orthogonal(nn.Linear(d, d), 2**0.5) for name in _CARD_COUNT_FIELDS}
        )
        self.pending_proj = _orthogonal(nn.Linear(d, d), 2**0.5)
        self.topk_pos = nn.Parameter(torch.zeros(self.top_k, d))
        self.topk_proj = _orthogonal(nn.Linear(d, d), 2**0.5)
        self.hist_meta_proj = _orthogonal(nn.Linear(self.hist_meta, d), 2**0.5)
        self.hist_pos = nn.Parameter(torch.zeros(hist_len, d))
        self.hist_proj = _orthogonal(nn.Linear(d, d), 2**0.5)
        self.slot_mlp = nn.Sequential(
            _orthogonal(nn.Linear(slot_width, d), 2**0.5), nn.GELU(),
            _orthogonal(nn.Linear(d, d), 2**0.5),
        )
        self.side_proj = nn.ModuleDict(
            {name: _orthogonal(nn.Linear(2 * d, d), 2**0.5) for name in _MELD_FIELDS}
        )
        special = {*_CARD_COUNT_FIELDS, *_MELD_FIELDS,
                   "trash_top_k", "pending_pile_card", "history"}
        self._scalar_fields = [f for f in spec.fields if f.name not in special]
        scalar_in = sum(f.flat_size for f in self._scalar_fields)
        scalar_out = min(hidden, 64)
        self.scalar_mlp = _orthogonal(nn.Linear(scalar_in, scalar_out), 2**0.5)

        concat = 7 * d + scalar_out  # hand+trash+pending+topk+history+2 meld sides
        blocks: list[nn.Module] = []
        width = concat
        for _ in range(layers):
            blocks += [
                _orthogonal(nn.Linear(width, hidden), gain=2**0.5),
                nn.LayerNorm(hidden),
                nn.GELU(),
            ]
            width = hidden
        self.trunk = nn.Sequential(*blocks)
        self.policy_head = _orthogonal(nn.Linear(hidden, num_actions), gain=0.01)
        self.value_head = _orthogonal(nn.Linear(hidden, 1), gain=1.0)
        self.act = nn.GELU()

    def _slice(self, x: Tensor, name: str) -> Tensor:
        f = self._fields[name]
        return x[:, f.offset : f.offset + f.flat_size]

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        e = self.card_embed
        parts = [
            self.act(self.count_proj[n](self._slice(x, n) @ e))
            for n in _CARD_COUNT_FIELDS
        ]
        parts.append(self.act(self.pending_proj(self._slice(x, "pending_pile_card") @ e)))

        topk = self._slice(x, "trash_top_k").view(-1, self.top_k, self.card_space)
        topk = self.act(self.topk_proj(topk @ e + self.topk_pos))
        parts.append(topk.mean(dim=1))

        hist = self._slice(x, "history").view(-1, *self._fields["history"].shape)
        rows = self.hist_meta_proj(hist[..., : self.hist_meta])
        rows = rows + hist[..., self.hist_meta :] @ e + self.hist_pos
        parts.append(self.act(self.hist_proj(rows)).mean(dim=1))

        for name in _MELD_FIELDS:
            block = self._slice(x, name).view(-1, *self._fields[name].shape)
            slot_h = self.slot_mlp(block)
            occupied = block[..., :1]  # slot feature 0 is the occupancy flag
            masked = slot_h * occupied
            mean = masked.sum(dim=1) / occupied.sum(dim=1).clamp(min=1.0)
            parts.append(self.act(self.side_proj[name](
                torch.cat([mean, masked.max(dim=1).values], dim=-1)
            )))

        scalars = torch.cat(
            [self._slice(x, f.name) for f in self._scalar_fields], dim=-1
        )
        parts.append(self.act(self.scalar_mlp(scalars)))

        h = self.trunk(torch.cat(parts, dim=-1))
        return self.policy_head(h), self.value_head(h).squeeze(-1)


def net_config(
    arch: str,
    spec: ObsSpec,
    num_actions: int,
    hidden: int,
    layers: int,
    embed_dim: int = 64,
) -> dict[str, Any]:
    """Serializable recipe for `build_net`; travels in checkpoints, pool
    member files, and parallel-worker initargs."""
    base = {"arch": arch, "num_actions": num_actions, "hidden": hidden, "layers": layers}
    if arch == "mlp":
        return {**base, "flat_dim": spec.flat_dim}
    if arch == "structured":
        return {**base, "embed_dim": embed_dim, "obs_spec": spec.to_dict()}
    raise ValueError(f"unknown arch {arch!r}; expected 'mlp' or 'structured'")


def build_net(config: dict[str, Any]) -> nn.Module:
    if config["arch"] == "mlp":
        return PolicyValueNet(
            config["flat_dim"], config["num_actions"],
            hidden=config["hidden"], layers=config["layers"],
        )
    if config["arch"] == "structured":
        return StructuredPolicyValueNet(
            ObsSpec.from_dict(config["obs_spec"]), config["num_actions"],
            hidden=config["hidden"], layers=config["layers"],
            embed_dim=config["embed_dim"],
        )
    raise ValueError(f"unknown arch {config['arch']!r}")
