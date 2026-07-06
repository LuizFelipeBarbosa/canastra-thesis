"""Rule-based baseline agent.

Greedy priorities: go out when possible; meld naturals aggressively; spend
wilds reluctantly (more willingly to complete a canastra); take the trash pile
when it connects to the hand or an own meld; discard the least connected,
cheapest card. Decides from the hidden-information view only (`raw_obs` from
`observations.observe`) — it can never peek at hidden zones.
"""

from __future__ import annotations

import random
from typing import Any

from buraco.cards import JOKER, CardId, id_rank, id_suit, positions_of
from buraco.config import RulesConfig
from buraco.engine.actions import (
    Add,
    CreateSeq,
    CreateSet,
    Discard,
    DrawDeck,
    DrawTrash,
    EndRound,
    GoOut,
    decode,
)


class HeuristicAgent:
    """Deterministic greedy policy with seeded tie-breaking."""

    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)

    def act(self, raw_obs: dict[str, Any], legal_ids: list[int], cfg: RulesConfig) -> int:
        slots = cfg.meld.max_meld_slots
        scored = [
            (self._score(decode(a, slots), raw_obs, cfg), self.rng.random(), a)
            for a in legal_ids
        ]
        return max(scored)[2]

    def _score(self, action: Any, raw: dict[str, Any], cfg: RulesConfig) -> float:
        hand: dict[CardId, int] = raw["hand"]
        match action:
            case GoOut():
                return 10_000.0
            case CreateSeq(wild=w):
                return 900.0 if w == 0 else 500.0
            case CreateSet(wild=w):
                return 850.0 if w == 0 else 450.0
            case Add(slot=slot, ct=ct):
                target = raw["own_melds"][slot] if slot < len(raw["own_melds"]) else None
                completes = target is not None and target["size"] == cfg.meld.canastra_min_size - 1
                if cfg.is_wild_card(ct):
                    return 300.0 + (250.0 if completes else 0.0)
                return 700.0 + (150.0 if completes else 0.0)
            case DrawTrash():
                return 600.0 if self._pile_connects(raw, cfg) else 100.0
            case DrawDeck():
                return 400.0
            case Discard(ct=ct):
                return 200.0 - 10.0 * self._usefulness(ct, hand, cfg) - cfg.card_value(ct) / 20.0
            case EndRound():
                return 50.0
        return 0.0

    def _pile_connects(self, raw: dict[str, Any], cfg: RulesConfig) -> bool:
        """Does any trash card pair with the hand or extend an own meld?"""
        hand = raw["hand"]
        ranks_in_hand = {id_rank(c) for c in hand if c != JOKER}
        for card in raw["trash"]:
            if card == JOKER or cfg.is_wild_card(card):
                return True  # a free wild is always worth taking
            if id_rank(card) in ranks_in_hand:
                return True
            for meld in raw["own_melds"]:
                if meld["is_sequence"] and meld["suit"] == int(id_suit(card)):
                    start, end = meld["start_pos"], meld["start_pos"] + meld["size"] - 1
                    if any(p in (start - 1, end + 1) for p in positions_of(id_rank(card))):
                        return True
                if not meld["is_sequence"] and meld["rank"] == int(id_rank(card)):
                    return True
        return False

    def _usefulness(self, ct: CardId, hand: dict[CardId, int], cfg: RulesConfig) -> float:
        """How connected a card is to the rest of the hand (higher = keep)."""
        if cfg.is_wild_card(ct):
            return 12.0  # never throw wilds if there is any alternative
        rank, suit = id_rank(ct), id_suit(ct)
        assert rank is not None and suit is not None
        same_rank = sum(
            n for c, n in hand.items() if c != ct and c != JOKER and id_rank(c) == rank
        )
        neighbors = 0
        for other, n in hand.items():
            if other == ct or other == JOKER or id_suit(other) != suit:
                continue
            other_rank = id_rank(other)
            assert other_rank is not None
            gap = min(
                abs(p - q)
                for p in positions_of(rank)
                for q in positions_of(other_rank)
            )
            if 1 <= gap <= 2:
                neighbors += n
        return 2.0 * same_rank + float(neighbors) + (hand.get(ct, 0) - 1)
