"""Regression tests for the Codex review findings (2026-07-05).

1. Canasta's forced pile-card obligation (SPEC 06 G1) must consume the taken
   top card *itself* — rank-only checks let a player meld another copy of the
   rank and keep the top card in hand.
2. Configurations with more than two sides must be rejected before reaching
   the two-side env/observation/GUI model (`1 - side` arithmetic).
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from buraco.cards import Rank, Suit, card_id
from buraco.engine.actions import Add, CreateSet, Discard
from buraco.engine.legal import legal_actions
from buraco.engine.melds import SET_WILD_NONE, Meld, MeldKind, Slot, SlotRole
from buraco.engine.state import Phase
from buraco.engine.turns import IllegalAction, apply_action
from buraco.env.env import BuracoEnv
from buraco.profiles import canasta, rummy
from tests.helpers import ct, make_state

QC, QD, QH, QS = (card_id(Rank.QUEEN, s) for s in Suit)
FILLER = (ct(Rank.EIGHT, Suit.CLUBS), ct(Rank.SIX, Suit.DIAMONDS))


def _queen_set(owner: int = 0) -> Meld:
    return Meld(
        meld_id=0,
        owner=owner,
        kind=MeldKind.SET,
        rank=Rank.QUEEN,
        slots=[
            Slot(QC, SlotRole.NATURAL),
            Slot(QD, SlotRole.NATURAL),
            Slot(QH, SlotRole.NATURAL),
        ],
    )


def _pending_state(hand0, melds=(), pair_only=False):
    state = make_state(
        cfg=canasta(2),
        hands=[tuple(hand0), (ct(Rank.NINE, Suit.CLUBS),) * 3],
        stock=[ct(Rank.FIVE, Suit.CLUBS)] * 6,
        melds=list(melds),
        phase=Phase.PLAY,
    )
    state.pending_pile_card = QS
    state.pending_pile_pair_only = pair_only
    return state


def test_create_set_consumes_the_pending_card_itself():
    state = _pending_state((QC, QD, QH, QS) + FILLER)
    legal = legal_actions(state)
    assert all(not isinstance(a, Discard) for a in legal)  # obligation blocks discard
    apply_action(state, CreateSet(rank=Rank.QUEEN, wild=SET_WILD_NONE))
    assert state.pending_pile_card is None
    # Canonical lowest-suit-first selection would have melded ♣♦♥ and left the
    # taken Q♠ in hand; the forced meld must consume Q♠ and spare a queen.
    assert state.hands[0].get(QS, 0) == 0
    assert state.hands[0][QH] == 1


def test_add_must_use_the_pending_card_not_just_its_rank():
    state = _pending_state((QC, QS) + FILLER, melds=[_queen_set()])
    adds = [a for a in legal_actions(state) if isinstance(a, Add)]
    assert Add(slot=0, ct=QS) in adds
    assert all(a.ct == QS for a in adds)
    with pytest.raises(IllegalAction):
        apply_action(state, Add(slot=0, ct=QC))
    apply_action(state, Add(slot=0, ct=QS))
    assert state.pending_pile_card is None
    assert state.hands[0][QC] == 1


def test_frozen_take_forced_set_consumes_the_pending_card():
    state = _pending_state((QC, QD, QH, QS) + FILLER, pair_only=True)
    apply_action(state, CreateSet(rank=Rank.QUEEN, wild=SET_WILD_NONE))
    assert state.pending_pile_card is None
    assert state.hands[0].get(QS, 0) == 0


def test_multi_side_configs_are_rejected():
    with pytest.raises(ValueError, match="2-player"):
        rummy(3)
    with pytest.raises(ValueError, match="2-player"):
        rummy(4)
    three_sides = replace(rummy(2), table=replace(rummy(2).table, num_players=3))
    with pytest.raises(ValueError, match="two sides"):
        BuracoEnv(three_sides)
