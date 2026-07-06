"""Canonical observation flattening (numpy-only; no torch here).

`encode_observation` already produces fixed-shape numpy fields, but a network
input needs three fixes on top of it: card-id fields (`trash_top_k`,
`pending_pile_card`) must be one-hot rather than scalar ids, raw count fields
must be normalized, and the field order must be pinned and serialized so a
checkpoint can never be applied with silently permuted features. Shapes are
identical across profiles and player counts, so one ObsSpec serves 2p and 4p.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Any

import numpy as np

from buraco.cards import CARD_SPACE
from buraco.config import RulesConfig

# Fields holding card ids in [0, CARD_SPACE) that must be one-hot encoded.
_ONEHOT_CARD = "onehot_card"
_SCALE = "scale"
_AS_FLOAT = "as_float"

_CARD_ID_FIELDS = frozenset({"trash_top_k", "pending_pile_card"})


def _scale_constant(name: str, cfg: RulesConfig) -> float | None:
    total = float(cfg.deck.total_cards)
    return {
        "hand_size": total,
        "all_hand_sizes": total,
        "trash_size": total,
        "deck_size": total,
        "mortos_remaining": float(max(cfg.morto.count, 1)),
        "melds_this_turn": 8.0,
    }.get(name)


@dataclass(frozen=True)
class FieldSpec:
    name: str
    shape: tuple[int, ...]
    kind: str  # one of _ONEHOT_CARD / _SCALE / _AS_FLOAT
    scale: float
    offset: int
    flat_size: int


class ObsSpec:
    """Pinned field order + per-field transforms for dict-obs flattening."""

    def __init__(self, fields: tuple[FieldSpec, ...]):
        self.fields = fields
        self.flat_dim = sum(f.flat_size for f in fields)

    @classmethod
    def from_cfg(
        cls, cfg: RulesConfig, history_len: int = 8, trash_top_k: int = 8
    ) -> ObsSpec:
        from buraco.env.env import BuracoEnv

        env = BuracoEnv(cfg, history_len=history_len, trash_top_k=trash_top_k)
        obs, _ = env.reset(seed=0)
        fields: list[FieldSpec] = []
        offset = 0
        for name in sorted(obs):
            shape = obs[name].shape
            n = prod(shape)
            scale = _scale_constant(name, cfg)
            if name in _CARD_ID_FIELDS:
                kind, scale, size = _ONEHOT_CARD, 1.0, n * CARD_SPACE
            elif scale is not None:
                kind, size = _SCALE, n
            else:
                kind, scale, size = _AS_FLOAT, 1.0, n
            fields.append(FieldSpec(name, shape, kind, scale, offset, size))
            offset += size
        return cls(tuple(fields))

    def flatten(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        out = np.zeros(self.flat_dim, dtype=np.float32)
        self._fill(out, obs)
        return out

    def flatten_batch(self, obs_list: list[dict[str, np.ndarray]]) -> np.ndarray:
        out = np.zeros((len(obs_list), self.flat_dim), dtype=np.float32)
        for row, obs in zip(out, obs_list):
            self._fill(row, obs)
        return out

    def _fill(self, out: np.ndarray, obs: dict[str, np.ndarray]) -> None:
        for f in self.fields:
            arr = obs[f.name]
            if f.kind == _ONEHOT_CARD:
                for i, card in enumerate(arr.reshape(-1)):
                    out[f.offset + i * CARD_SPACE + int(card)] = 1.0
            elif f.kind == _SCALE:
                out[f.offset : f.offset + f.flat_size] = arr.reshape(-1) / f.scale
            else:
                out[f.offset : f.offset + f.flat_size] = arr.reshape(-1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "flat_dim": self.flat_dim,
            "fields": [
                {
                    "name": f.name,
                    "shape": list(f.shape),
                    "kind": f.kind,
                    "scale": f.scale,
                    "offset": f.offset,
                    "flat_size": f.flat_size,
                }
                for f in self.fields
            ],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ObsSpec:
        spec = cls(
            tuple(
                FieldSpec(
                    f["name"], tuple(f["shape"]), f["kind"], f["scale"],
                    f["offset"], f["flat_size"],
                )
                for f in d["fields"]
            )
        )
        assert spec.flat_dim == d["flat_dim"]
        return spec

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ObsSpec) and self.fields == other.fields
