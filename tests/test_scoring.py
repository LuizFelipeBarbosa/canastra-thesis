"""Round scoring: meld points, canastra bonuses, penalties (scenarios 25, 36–40)."""

from collections import Counter
from dataclasses import replace

from buraco.cards import JOKER, Rank, Suit
from buraco.config import HAND_PENALTY_OPPONENT_POSITIVE
from buraco.engine.melds import SET_WILD_NONE, create_set
from buraco.engine.scoring import canastra_bonus, hand_points, meld_points, round_scores
from buraco.engine.state import EndReason, Phase
from buraco.profiles import buraco
from tests.helpers import ct, make_state, natural_run

CFG = buraco(2)


def test_scenario25_joker_scores_its_own_value():
    hand = Counter([ct(Rank.QUEEN, Suit.HEARTS), ct(Rank.QUEEN, Suit.SPADES),
                    ct(Rank.QUEEN, Suit.DIAMONDS)])
    meld = create_set(CFG, hand, 0, 0, Rank.QUEEN, SET_WILD_NONE)
    hand[JOKER] += 1
    from buraco.engine.melds import apply_add

    apply_add(CFG, hand, meld, JOKER)
    assert meld_points(CFG, meld) == 10 + 10 + 10 + 20


def test_canastra_bonus_tiers():
    limpa = natural_run(CFG, Suit.HEARTS, 3, 7)  # 3..9
    assert canastra_bonus(CFG, limpa) == 200
    short = natural_run(CFG, Suit.HEARTS, 3, 6)
    assert canastra_bonus(CFG, short) == 0


def test_scenario36_negative_round_score():
    hand0 = [ct(Rank.ACE, Suit.SPADES), ct(Rank.KING, Suit.HEARTS),
             ct(Rank.KING, Suit.DIAMONDS), ct(Rank.TEN, Suit.CLUBS)]  # 45 pts
    state = make_state(
        hands=[hand0, []],
        morto=[(ct(Rank.FOUR, Suit.CLUBS),), (ct(Rank.FIVE, Suit.CLUBS),)],
        morto_taken=[False, False],
        phase=Phase.DRAW,
    )
    state.round_over = True
    state.end_reason = EndReason.STOCK_EXHAUSTED
    # morto cards left on the table do not count against anyone; only hand + penalty
    scores = round_scores(state)
    assert scores[0] == -45 - 100 == -145
    assert scores[1] == -100


def test_scenario39_bater_bonus_additive():
    melds = [
        natural_run(CFG, Suit.HEARTS, 3, 7, owner=0),  # 5*5+10+10=45 pts + 200 limpa
        natural_run(CFG, Suit.SPADES, 9, 3, owner=0),  # 9,10,J = 30 pts
    ]
    state = make_state(hands=[[], [ct(Rank.KING, Suit.HEARTS)]], melds=melds)
    state.round_over = True
    state.end_reason = EndReason.BATER
    state.went_out_side = 0
    scores = round_scores(state)
    assert scores[0] == 45 + 200 + 30 + 100  # melds + bonus + bater
    assert scores[1] == -10  # king left in hand


def test_scenario37_exhaustion_scores_both_sides():
    melds = [natural_run(CFG, Suit.HEARTS, 3, 7, owner=0),
             natural_run(CFG, Suit.CLUBS, 5, 3, owner=1)]  # 5,6,7 = 15 pts
    state = make_state(hands=[[ct(Rank.FOUR, Suit.SPADES)], [ct(Rank.ACE, Suit.CLUBS)]],
                       melds=melds)
    state.round_over = True
    state.end_reason = EndReason.STOCK_EXHAUSTED
    scores = round_scores(state)
    assert scores[0] == 45 + 200 - 5  # no bater bonus for anyone
    assert scores[1] == 15 - 15


def test_scenario40_rummy_opponent_positive():
    cfg = replace(
        CFG,
        morto=replace(CFG.morto, count=0),
        scoring=replace(CFG.scoring, hand_penalty_mode=HAND_PENALTY_OPPONENT_POSITIVE),
        going_out=replace(CFG.going_out, go_out_bonus=0, require_canastra=False,
                          require_morto_taken=False),
    )
    state = make_state(
        cfg=cfg,
        hands=[[], [ct(Rank.KING, Suit.HEARTS), ct(Rank.SEVEN, Suit.SPADES),
                    ct(Rank.ACE, Suit.DIAMONDS)]],
        morto=[None, None],
    )
    state.round_over = True
    state.end_reason = EndReason.BATER
    state.went_out_side = 0
    scores = round_scores(state)
    # Buraco point values here (10+5+15); the Rummy profile swaps the table in M8.
    assert scores[0] == 30
    assert scores[1] == 0


def test_hand_points():
    state = make_state(hands=[[ct(Rank.THREE, Suit.CLUBS), JOKER], []])
    assert hand_points(CFG, state, 0) == 25
