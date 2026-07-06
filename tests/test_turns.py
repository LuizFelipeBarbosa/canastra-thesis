"""Turn state machine: morto flows, bater, exhaustion, guards (M4).

Scenario numbers refer to docs/specs/test-scenarios-rules.md.
"""

import random
from collections import Counter
from dataclasses import replace

import pytest

from buraco.cards import Rank, Suit
from buraco.config import DISCARD_OUT_OPTIONAL, DRAW_TOP_CARD
from buraco.engine.actions import (
    Add,
    CreateSeq,
    Discard,
    DrawDeck,
    DrawTrash,
    EndRound,
    GoOut,
)
from buraco.engine.legal import legal_actions
from buraco.engine.melds import SEQ_WILD_NONE, validate_meld
from buraco.engine.state import EndReason, Phase, deal_round, dealt_multiset
from buraco.engine.turns import IllegalAction, apply_action
from buraco.profiles import buraco
from tests.helpers import ct, make_state, natural_run

CFG = buraco(2)

H456 = [ct(Rank.FOUR, Suit.HEARTS), ct(Rank.FIVE, Suit.HEARTS), ct(Rank.SIX, Suit.HEARTS)]
C9 = ct(Rank.NINE, Suit.CLUBS)
C10 = ct(Rank.TEN, Suit.CLUBS)


def canastra_melds(owner=0):
    return [natural_run(CFG, Suit.SPADES, 3, 7, owner=owner)]


# --- morto flows -----------------------------------------------------------------


def test_scenario1_batida_direta_continues_turn():
    state = make_state(
        hands=[H456, [C10]],
        morto=[(C9, C10), (ct(Rank.SEVEN, Suit.DIAMONDS),)],
        morto_taken=[False, False],
    )
    apply_action(state, CreateSeq(Suit.HEARTS, 4, SEQ_WILD_NONE))
    assert state.morto_taken[0] and state.morto[0] is None
    assert state.hands[0] == Counter([C9, C10])
    assert state.phase is Phase.PLAY and state.current_player == 0
    assert not state.round_over
    apply_action(state, Discard(C9))  # normal discard ends the turn
    assert state.current_player == 1 and state.phase is Phase.DRAW


def test_scenario2_batida_indireta_ends_turn():
    state = make_state(
        hands=[[C9], [C10]],
        morto=[(ct(Rank.FOUR, Suit.DIAMONDS), ct(Rank.FIVE, Suit.DIAMONDS)), (C10,)],
        morto_taken=[False, False],
    )
    apply_action(state, Discard(C9))
    assert state.trash == [C9]
    assert state.morto_taken[0]
    assert state.hands[0] == Counter(
        [ct(Rank.FOUR, Suit.DIAMONDS), ct(Rank.FIVE, Suit.DIAMONDS)]
    )
    assert state.current_player == 1 and state.phase is Phase.DRAW
    assert not state.round_over


def test_scenario3_bater_by_discard():
    state = make_state(hands=[[C9], [C10]], melds=canastra_melds())
    apply_action(state, Discard(C9))
    assert state.round_over and state.end_reason is EndReason.BATER
    assert state.went_out_side == 0
    assert state.trash == [C9]


def test_scenario4_bater_same_turn_as_morto():
    state = make_state(
        hands=[H456, [C10]],
        melds=canastra_melds(),
        morto=[(C9,), (C10,)],
        morto_taken=[False, False],
    )
    apply_action(state, CreateSeq(Suit.HEARTS, 4, SEQ_WILD_NONE))  # direta → hand {9♣}
    assert state.morto_taken[0] and not state.round_over
    apply_action(state, Discard(C9))  # empties again; morto gone; canastra → bater
    assert state.round_over and state.went_out_side == 0


def test_scenario5_meld_out_blocked_when_discard_required():
    state = make_state(hands=[H456, [C10]], melds=canastra_melds())
    with pytest.raises(IllegalAction):
        apply_action(state, CreateSeq(Suit.HEARTS, 4, SEQ_WILD_NONE))
    acts = legal_actions(state)
    assert not any(isinstance(a, CreateSeq) for a in acts)
    assert any(isinstance(a, Discard) for a in acts)


def test_scenario6_meld_out_and_go_out_when_optional():
    cfg = replace(CFG, going_out=replace(CFG.going_out, discard_to_go_out=DISCARD_OUT_OPTIONAL))
    state = make_state(cfg=cfg, hands=[H456, [C10]],
                       melds=[natural_run(cfg, Suit.SPADES, 3, 7)])
    apply_action(state, CreateSeq(Suit.HEARTS, 4, SEQ_WILD_NONE))
    assert state.hand_size(0) == 0 and not state.round_over
    assert legal_actions(state) == [GoOut()]
    apply_action(state, GoOut())
    assert state.round_over and state.went_out_side == 0


def test_scenario12_first_emptying_needs_no_canastra():
    state = make_state(
        hands=[[C9], [C10]],
        morto=[(C10, C10), (C9,)],
        morto_taken=[False, False],
    )
    assert any(isinstance(a, Discard) for a in legal_actions(state))
    apply_action(state, Discard(C9))  # no canastra needed for morto pickup
    assert state.morto_taken[0] and not state.round_over


def test_scenario13_emptying_discard_blocked_without_canastra():
    state = make_state(hands=[[C9], [C10]])  # mortos taken, no canastra
    acts = legal_actions(state)
    assert Discard(C9) not in acts
    with pytest.raises(IllegalAction):
        apply_action(state, Discard(C9))


def test_anti_stranding_guard_blocks_meld_to_one_card():
    # Melding to exactly 1 card is illegal when the forced final discard
    # could not go out (no canastra).
    state = make_state(hands=[H456 + [C9], [C10]])  # mortos taken, no canastra
    with pytest.raises(IllegalAction):
        apply_action(state, CreateSeq(Suit.HEARTS, 4, SEQ_WILD_NONE))
    # With a canastra the same meld is fine: the final discard baters.
    state2 = make_state(hands=[H456 + [C9], [C10]], melds=canastra_melds())
    apply_action(state2, CreateSeq(Suit.HEARTS, 4, SEQ_WILD_NONE))
    apply_action(state2, Discard(C9))
    assert state2.round_over and state2.went_out_side == 0


def test_scenario15_add_completing_canastra_enables_same_turn_bater():
    # Reviewer blocker: the ADD that grows a meld 6→7 IS the side's first
    # canastra; it must be legal even though it leaves the hand at 1 card,
    # because the forced final discard now baters.
    nine_s = ct(Rank.NINE, Suit.SPADES)
    state = make_state(
        hands=[[nine_s, C9], [C10]],
        melds=[natural_run(CFG, Suit.SPADES, 3, 6)],  # 3♠..8♠, one short
    )
    add = Add(slot=0, ct=nine_s)
    assert add in legal_actions(state)
    apply_action(state, add)
    assert state.side_melds(0)[0].is_canastra(CFG.meld.canastra_min_size)
    apply_action(state, Discard(C9))
    assert state.round_over and state.went_out_side == 0


def test_scenario15_add_completing_canastra_meld_out_optional():
    cfg = replace(CFG, going_out=replace(CFG.going_out, discard_to_go_out=DISCARD_OUT_OPTIONAL))
    nine_s = ct(Rank.NINE, Suit.SPADES)
    state = make_state(
        cfg=cfg,
        hands=[[nine_s], [C10]],
        melds=[natural_run(cfg, Suit.SPADES, 3, 6)],
    )
    apply_action(state, Add(slot=0, ct=nine_s))  # empties hand, completes canastra
    assert legal_actions(state) == [GoOut()]
    apply_action(state, GoOut())
    assert state.round_over and state.went_out_side == 0


def test_wild_to_hand_swap_from_last_card_then_bater():
    # WILD_TO_HAND swap is hand-size neutral: swapping with the last card
    # leaves the freed wild in hand, and its discard goes out.
    from buraco.cards import JOKER
    from buraco.config import WILD_TO_HAND
    from buraco.engine.melds import Meld, MeldKind, Slot, SlotRole

    cfg = replace(CFG, wildcard=replace(CFG.wildcard, wild_relocation=WILD_TO_HAND))
    six_d = ct(Rank.SIX, Suit.DIAMONDS)
    dirty = Meld(1, 0, MeldKind.SEQUENCE, suit=Suit.DIAMONDS, start_pos=4, slots=[
        Slot(ct(Rank.FOUR, Suit.DIAMONDS), SlotRole.NATURAL),
        Slot(ct(Rank.FIVE, Suit.DIAMONDS), SlotRole.NATURAL),
        Slot(JOKER, SlotRole.WILD),
        Slot(ct(Rank.SEVEN, Suit.DIAMONDS), SlotRole.NATURAL),
    ])
    state = make_state(
        cfg=cfg,
        hands=[[six_d], [C10]],
        melds=[natural_run(cfg, Suit.HEARTS, 3, 7), dirty],  # canastra + swap target
    )
    add = Add(slot=1, ct=six_d)
    assert add in legal_actions(state)
    apply_action(state, add)
    assert state.hands[0][JOKER] == 1  # freed wild back in hand
    apply_action(state, Discard(JOKER))
    assert state.round_over and state.went_out_side == 0


def test_add_to_one_card_still_blocked_without_post_add_canastra():
    # The relaxation is surgical: an ADD that does NOT produce a qualifying
    # canastra still may not strand the hand at 1 card.
    eight_s = ct(Rank.EIGHT, Suit.SPADES)
    state = make_state(
        hands=[[eight_s, C9], [C10]],
        melds=[natural_run(CFG, Suit.SPADES, 3, 5)],  # 3♠..7♠ → grows to 6 only
    )
    assert Add(slot=0, ct=eight_s) not in legal_actions(state)
    with pytest.raises(IllegalAction):
        apply_action(state, Add(slot=0, ct=eight_s))


# --- draw phase and exhaustion ------------------------------------------------------


def test_draw_deck_flow():
    state = make_state(hands=[[C9], [C10]], stock=[C10, C9], phase=Phase.DRAW)
    apply_action(state, DrawDeck())
    assert state.hand_size(0) == 2 and len(state.stock) == 1
    assert state.phase is Phase.PLAY
    apply_action(state, Discard(C9))
    assert state.current_player == 1 and state.turn_number == 1


def test_scenario9_whole_pile_take():
    pile = [ct(Rank(i % 13), Suit(i % 4)) for i in range(20)]
    state = make_state(hands=[[C9], [C10]], stock=[C10], trash=pile, phase=Phase.DRAW)
    apply_action(state, DrawTrash())
    assert state.hand_size(0) == 21
    assert state.trash == []
    assert state.phase is Phase.PLAY


def test_scenario26_end_round_choice_when_stock_empty():
    state = make_state(hands=[[C9], [C10]], stock=[], trash=[C10], phase=Phase.DRAW)
    acts = legal_actions(state)
    assert acts == [DrawTrash(), EndRound()]
    apply_action(state, EndRound())
    assert state.round_over and state.end_reason is EndReason.STOCK_EXHAUSTED
    assert state.went_out_side is None


def test_scenario7_forced_end_round_when_nothing_to_draw():
    state = make_state(hands=[[C9], [C10]], stock=[], trash=[], phase=Phase.DRAW)
    assert legal_actions(state) == [EndRound()]
    with pytest.raises(IllegalAction):
        apply_action(state, DrawDeck())
    apply_action(state, EndRound())
    assert state.round_over


def test_end_round_illegal_while_stock_remains():
    state = make_state(hands=[[C9], [C10]], stock=[C10], phase=Phase.DRAW)
    with pytest.raises(IllegalAction):
        apply_action(state, EndRound())


def test_scenario27_redraw_discard_legal_in_buraco():
    state = make_state(hands=[[C9], [C10]], stock=[C10], trash=[ct(Rank.FIVE, Suit.DIAMONDS)],
                       phase=Phase.DRAW)
    apply_action(state, DrawTrash())
    apply_action(state, Discard(ct(Rank.FIVE, Suit.DIAMONDS)))  # same card back
    assert state.trash == [ct(Rank.FIVE, Suit.DIAMONDS)]


def test_scenario30_rummy_top_card_and_no_redraw():
    cfg = replace(
        CFG,
        discard_pile=replace(
            CFG.discard_pile, draw_rule=DRAW_TOP_CARD, no_immediate_redraw_discard=True
        ),
    )
    five = ct(Rank.FIVE, Suit.DIAMONDS)
    state = make_state(cfg=cfg, hands=[[C9], [C10]], stock=[C10], trash=[C10, five],
                       phase=Phase.DRAW)
    apply_action(state, DrawTrash())
    assert state.hands[0][five] == 1 and state.trash == [C10]
    discards = [a for a in legal_actions(state) if isinstance(a, Discard)]
    assert Discard(five) not in discards and Discard(C9) in discards


def test_out_of_phase_actions_rejected():
    state = make_state(hands=[[C9], [C10]], stock=[C10], phase=Phase.DRAW)
    with pytest.raises(IllegalAction):
        apply_action(state, Discard(C9))
    apply_action(state, DrawDeck())
    with pytest.raises(IllegalAction):
        apply_action(state, DrawDeck())


# --- mini random smoke (full soak is M7) --------------------------------------------


def test_random_play_smoke_2p():
    rng = random.Random(2024)
    cfg = buraco(2)
    state = deal_round(cfg, rng)
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
                validate_meld(cfg, meld)
    assert dealt_multiset(state) == full
