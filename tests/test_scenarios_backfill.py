"""Backfill lock-in tests for scenarios confirmed correct by review probes.

Scenario numbers refer to docs/specs/test-scenarios-rules.md.
"""

from collections import Counter
from dataclasses import replace

import pytest

from buraco.cards import JOKER, Rank, Suit
from buraco.engine.actions import Add, Discard, DrawDeck, DrawTrash
from buraco.engine.legal import bater_ready, legal_actions
from buraco.engine.melds import Meld, MeldKind, Slot, SlotRole, validate_meld
from buraco.engine.scoring import canastra_bonus, meld_points, round_scores
from buraco.engine.state import EndReason, Phase
from buraco.engine.turns import IllegalAction, apply_action
from buraco.profiles import buraco
from tests.helpers import ct, make_state, natural_run

CFG = buraco(2)

C9 = ct(Rank.NINE, Suit.CLUBS)
C10 = ct(Rank.TEN, Suit.CLUBS)


def test_scenario10_2p_mortos_are_independent():
    state = make_state(
        hands=[[C9], [C10]],
        morto=[(ct(Rank.FOUR, Suit.DIAMONDS), ct(Rank.FIVE, Suit.DIAMONDS)), None],
        morto_taken=[False, True],
        current_player=0,
    )
    apply_action(state, Discard(C9))
    assert state.morto_taken == [True, True]
    assert state.hands[0] == Counter(
        [ct(Rank.FOUR, Suit.DIAMONDS), ct(Rank.FIVE, Suit.DIAMONDS)]
    )
    assert not state.round_over


def test_scenario14_clean_canastra_profile_rejects_dirty_only():
    cfg = replace(CFG, going_out=replace(CFG.going_out, require_clean_canastra=True))
    dirty = Meld(
        0,
        0,
        MeldKind.SEQUENCE,
        suit=Suit.SPADES,
        start_pos=3,
        slots=[
            Slot(ct(Rank.THREE, Suit.SPADES), SlotRole.NATURAL),
            Slot(ct(Rank.FOUR, Suit.SPADES), SlotRole.NATURAL),
            Slot(ct(Rank.FIVE, Suit.SPADES), SlotRole.NATURAL),
            Slot(ct(Rank.SIX, Suit.SPADES), SlotRole.NATURAL),
            Slot(ct(Rank.SEVEN, Suit.SPADES), SlotRole.NATURAL),
            Slot(ct(Rank.EIGHT, Suit.SPADES), SlotRole.NATURAL),
            Slot(JOKER, SlotRole.WILD),
        ],
    )
    validate_meld(cfg, dirty)  # well-formed as a meld

    state = make_state(cfg=cfg, hands=[[C9], [C10]], melds=[dirty])
    assert Discard(C9) not in legal_actions(state)
    with pytest.raises(IllegalAction):
        apply_action(state, Discard(C9))

    clean = natural_run(cfg, Suit.HEARTS, 3, 7, owner=0, meld_id=1)
    state2 = make_state(cfg=cfg, hands=[[C9], [C10]], melds=[dirty, clean])
    apply_action(state2, Discard(C9))
    assert state2.round_over and state2.went_out_side == 0


def test_scenario28_empty_pile_take_illegal():
    state = make_state(hands=[[C9], [C10]], stock=[C10], trash=[], phase=Phase.DRAW)
    assert legal_actions(state) == [DrawDeck()]


def test_scenario29_wild_on_top_of_pile_still_takeable():
    two_s = ct(Rank.TWO, Suit.SPADES)
    state = make_state(hands=[[C9], [C10]], stock=[C10], trash=[two_s], phase=Phase.DRAW)
    assert DrawTrash() in legal_actions(state)
    apply_action(state, DrawTrash())
    assert state.hands[0][two_s] >= 1
    assert state.trash == []


def test_scenario33_fourteen_card_run_scores_one_bonus():
    meld = natural_run(CFG, Suit.HEARTS, 1, 14)
    assert meld.is_canastra(7) and meld.is_clean
    assert canastra_bonus(CFG, meld) == 200
    assert meld_points(CFG, meld) == 125


def test_scenario34_add_reaching_seven_enables_bater():
    nine_s = ct(Rank.NINE, Suit.SPADES)
    run = natural_run(CFG, Suit.SPADES, 3, 6)
    state = make_state(hands=[[nine_s, C9, C10], [C10]], melds=[run])
    assert not bater_ready(state, 0)
    apply_action(state, Add(slot=0, ct=nine_s))
    assert bater_ready(state, 0)


def test_scenario38_morto_penalty_alongside_melds():
    state = make_state(
        hands=[[], []],
        melds=[natural_run(CFG, Suit.HEARTS, 3, 7)],
        morto=[(C9,), (C10,)],
        morto_taken=[False, False],
    )
    state.round_over = True
    state.end_reason = EndReason.STOCK_EXHAUSTED
    scores = round_scores(state)
    assert scores[0] == 45 + 200 - 100 == 145
    assert scores[1] == -100
