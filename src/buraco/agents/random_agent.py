"""Uniform-random legal-action agent."""

from __future__ import annotations

import random
from typing import Any

from buraco.config import RulesConfig


class RandomAgent:
    """Picks uniformly among legal action ids. The floor baseline."""

    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)

    def act(self, raw_obs: dict[str, Any], legal_ids: list[int], cfg: RulesConfig) -> int:
        return self.rng.choice(legal_ids)
