"""4-player two-teams-of-two: shared melds, team morto, team scoring (M5).

Seats 0,2 form side 0; seats 1,3 form side 1 (partners sit opposite).
"""

import random
from dataclasses import replace

import pytest

from buraco.cards import Rank, Suit
from buraco.engine.actions import Add, Discard, DrawDeck
from buraco.engine.legal import legal_actions
from buraco.engine.melds import validate_meld
from buraco.engine.scoring import round_scores
from buraco.engine.state import EndReason, Phase, deal_round, dealt_multiset
from buraco.engine.turns import IllegalAction, apply_action
from buraco.profiles import buraco
from tests.helpers import ct, make_state, natural_run

CFG4 = buraco(4)
C9 = ct(Rank.NINE, Suit.CLUBS)
C10 = ct(Rank.TEN, Suit.CLUBS)


def four_hands(*hands):
    return [list(h) for h in hands]


def test_turn_order_rotates_through_both_teams():
    stock = [ct(Rank(i % 13), Suit(i % 4)) for i in range(12)]
    state = make_state(
        cfg=CFG4,
        hands=four_hands([C9, C10], [C9, C10], [C9, C10], [C9, C10]),
        stock=stock,
        phase=Phase.DRAW,
    )
    seen = []
    for _ in range(4):
        seen.append(state.current_player)
        apply_action(state, DrawDeck())
        apply_action(state, Discard(C9))
    assert seen == [0, 1, 2, 3]
    assert state.current_player == 0
    assert [CFG4.table.side(p) for p in seen] == [0, 1, 0, 1]


def test_partner_adds_to_shared_meld_opponent_cannot():
    meld = natural_run(CFG4, Suit.HEARTS, 3, 5, owner=0)  # 3♥..7♥ owned by side 0
    state = make_state(
        cfg=CFG4,
        hands=four_hands([C9], [C9], [ct(Rank.EIGHT, Suit.HEARTS), C9, C10], [C9]),
        melds=[meld],
        current_player=2,  # side 0's partner
    )
    add = Add(slot=0, ct=ct(Rank.EIGHT, Suit.HEARTS))
    assert add in legal_actions(state)
    apply_action(state, add)
    assert meld.size == 6

    # Opponent (seat 1, side 1) has no slot 0: the shared meld is not theirs.
    state2 = make_state(
        cfg=CFG4,
        hands=four_hands([C9], [ct(Rank.EIGHT, Suit.HEARTS), C9, C10], [C9], [C9]),
        melds=[natural_run(CFG4, Suit.HEARTS, 3, 5, owner=0)],
        current_player=1,
    )
    assert not any(isinstance(a, Add) for a in legal_actions(state2))
    with pytest.raises(IllegalAction):
        apply_action(state2, Add(slot=0, ct=ct(Rank.EIGHT, Suit.HEARTS)))


def test_partner_pickup_of_team_morto():
    state = make_state(
        cfg=CFG4,
        hands=four_hands([C9], [C9, C10], [C9], [C9, C10]),
        morto=[(C10, C10), (C9, C9)],
        morto_taken=[False, False],
        current_player=2,
    )
    apply_action(state, Discard(C9))  # partner (seat 2) empties first for side 0
    assert state.morto_taken[0] and state.morto[0] is None
    assert state.hands[2][C10] == 2
    assert not state.round_over


def test_scenario11_second_team_emptying_is_bater_attempt():
    # Side 0's morto was already taken by seat 0; seat 2 emptying is now a
    # bater attempt and needs a canastra.
    base = dict(
        cfg=CFG4,
        hands=four_hands([C9, C10], [C9, C10], [C9], [C9, C10]),
        morto=[None, (C9, C9)],
        morto_taken=[True, False],
        current_player=2,
    )
    state = make_state(**base)
    assert Discard(C9) not in legal_actions(state)  # no canastra yet

    state2 = make_state(**base, melds=[natural_run(CFG4, Suit.SPADES, 3, 7, owner=0)])
    apply_action(state2, Discard(C9))
    assert state2.round_over and state2.end_reason is EndReason.BATER
    assert state2.went_out_side == 0


def test_team_scores_pool_melds_and_hands():
    melds = [
        natural_run(CFG4, Suit.HEARTS, 3, 7, owner=0),  # 45 pts + 200 limpa
        natural_run(CFG4, Suit.CLUBS, 5, 3, owner=1),  # 15 pts
    ]
    state = make_state(
        cfg=CFG4,
        hands=four_hands(
            [ct(Rank.FOUR, Suit.SPADES)],  # seat 0 (side 0): 5
            [ct(Rank.ACE, Suit.CLUBS)],  # seat 1 (side 1): 15
            [ct(Rank.KING, Suit.SPADES)],  # seat 2 (side 0): 10
            [],  # seat 3 (side 1)
        ),
        melds=melds,
    )
    state.round_over = True
    state.end_reason = EndReason.STOCK_EXHAUSTED
    scores = round_scores(state)
    assert len(scores) == 2
    assert scores[0] == 45 + 200 - 5 - 10  # both seats' hands count against side 0
    assert scores[1] == 15 - 15


def test_random_play_smoke_4p():
    rng = random.Random(4242)
    state = deal_round(CFG4, rng)
    full = dealt_multiset(state)
    for step in range(3000):
        if state.round_over:
            break
        acts = legal_actions(state)
        assert acts, f"empty mask mid-episode at step {step}"
        apply_action(state, rng.choice(acts))
        if step % 25 == 0:
            assert dealt_multiset(state) == full
            for meld in state.melds:
                validate_meld(CFG4, meld)
    assert dealt_multiset(state) == full


def test_side_canastra_shared_for_going_out():
    # A canastra melded by one partner unlocks bater for the other.
    state = make_state(
        cfg=CFG4,
        hands=four_hands([C9, C10], [C9, C10], [C9], [C9, C10]),
        melds=[natural_run(CFG4, Suit.DIAMONDS, 4, 7, owner=0)],
        morto=[None, None],
        morto_taken=[True, True],
        current_player=2,
    )
    apply_action(state, Discard(C9))
    assert state.round_over and state.went_out_side == 0


def test_morto_config_flexibility_still_validated():
    bad = replace(CFG4, morto=replace(CFG4.morto, count=4))
    with pytest.raises(ValueError):
        deal_round(bad, random.Random(0))
