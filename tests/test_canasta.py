"""Classic Canasta mechanics (SPEC 06): frozen/conditional pile, red/black
threes, initial-meld staging, canasta scoring."""

import random
from collections import Counter

import pytest

from buraco.cards import JOKER, Rank, Suit, card_id
from buraco.engine.actions import CreateSet, Discard, DrawDeck, DrawTrash
from buraco.engine.legal import (
    can_take_conditional_pile,
    legal_actions,
    max_stageable_points,
)
from buraco.engine.scoring import round_scores
from buraco.engine.state import RED_THREE_IDS, EndReason, Phase, deal_round, dealt_multiset
from buraco.engine.turns import IllegalAction, apply_action
from buraco.profiles import canasta
from tests.helpers import ct, make_state
from tests.test_soak import play_engine_game

CFG = canasta(4)
C9 = ct(Rank.NINE, Suit.CLUBS)
QC, QD, QH, QS = (card_id(Rank.QUEEN, s) for s in Suit)
BLACK3_C = ct(Rank.THREE, Suit.CLUBS)
BLACK3_S = ct(Rank.THREE, Suit.SPADES)
RED3_D = RED_THREE_IDS[0]


def queens_hand(n=2, extra=()):
    return [QC, QD][:n] + list(extra)


def natural_set(rank, size, owner=0, meld_id=0, cfg=CFG):
    from buraco.engine.melds import Meld, MeldKind, Slot, SlotRole

    suits = [Suit.CLUBS, Suit.DIAMONDS, Suit.HEARTS, Suit.SPADES]
    slots = [Slot(card_id(rank, suits[i % 4]), SlotRole.NATURAL) for i in range(size)]
    return Meld(meld_id, owner, MeldKind.SET, rank=rank, slots=slots)


# --- deal, red threes -----------------------------------------------------------


def test_deal_resolves_red_threes_and_conserves_cards():
    for seed in range(12):
        state = deal_round(CFG, random.Random(seed))
        for player in range(4):
            for red in RED_THREE_IDS:
                assert state.hands[player].get(red, 0) == 0
        assert sum(dealt_multiset(state).values()) == 108
        assert len(state.trash) == 1  # upcard


def test_red_three_as_last_stock_card_ends_round():
    # Soak-found deadlock: unreplaced red 3 shrinks the hand entering PLAY.
    # Pagat: a red 3 drawn as the last stock card ends the round immediately.
    state = make_state(
        cfg=CFG,
        hands=[[C9], [C9], [C9], [C9]],
        stock=[RED3_D],  # nothing beneath it
        phase=Phase.DRAW,
    )
    apply_action(state, DrawDeck())
    assert state.round_over and state.end_reason is EndReason.STOCK_EXHAUSTED
    assert state.red_threes[0] == [RED3_D]


def test_red_three_drawn_goes_to_tray_with_replacement():
    state = make_state(
        cfg=CFG,
        hands=[[C9], [C9], [C9], [C9]],
        stock=[QH, RED3_D],  # red 3 on top, queen beneath as replacement
        phase=Phase.DRAW,
    )
    apply_action(state, DrawDeck())
    assert state.red_threes[0] == [RED3_D]
    assert state.hands[0][QH] == 1
    assert state.hands[0].get(RED3_D, 0) == 0


# --- conditional / frozen pile ----------------------------------------------------


def test_take_pile_with_pair_forces_top_card_meld():
    aces = [card_id(Rank.ACE, Suit.CLUBS), card_id(Rank.ACE, Suit.DIAMONDS)]
    state = make_state(
        cfg=CFG,
        hands=[queens_hand(2, extra=aces + [JOKER, C9]), [C9], [C9], [C9]],
        stock=[QH] * 3,
        trash=[ct(Rank.FOUR, Suit.HEARTS), QS],  # natural queen on top
        phase=Phase.DRAW,
        initial_meld_done=[False, False],
    )
    # threshold reachable: forced queens 30 + (A,A,joker) 90
    assert can_take_conditional_pile(state)
    apply_action(state, DrawTrash())
    assert state.pending_pile_card == QS
    assert state.pending_pile_pair_only  # unopened side → pile frozen for it
    acts = legal_actions(state)
    assert acts == [CreateSet(rank=Rank.QUEEN, wild=0)]
    apply_action(state, acts[0])
    assert state.pending_pile_card is None
    # 3 queens = 30 points < 50 threshold → still staging; discard masked
    assert not state.initial_meld_done[0]
    assert not any(isinstance(a, Discard) for a in legal_actions(state))
    apply_action(state, CreateSet(rank=Rank.ACE, wild=1))  # +90 → opened
    assert state.initial_meld_done[0]
    assert any(isinstance(a, Discard) for a in legal_actions(state))


def test_take_pile_rejected_when_threshold_unreachable():
    # A queen pair but nothing else meldable: forced 30 < 50 → take masked.
    state = make_state(
        cfg=CFG,
        hands=[queens_hand(2, extra=[C9, ct(Rank.EIGHT, Suit.HEARTS)]), [C9], [C9], [C9]],
        stock=[QH] * 3,
        trash=[ct(Rank.FOUR, Suit.HEARTS), QS],
        phase=Phase.DRAW,
        initial_meld_done=[False, False],
    )
    assert not can_take_conditional_pile(state)
    assert DrawTrash() not in legal_actions(state)


def test_take_pile_rejected_without_pair_when_frozen():
    state = make_state(
        cfg=CFG,
        hands=[[QC, JOKER, C9], [C9], [C9], [C9]],  # one queen + wild only
        stock=[QH],
        trash=[QS],
        phase=Phase.DRAW,
        initial_meld_done=[False, False],  # frozen for unopened side
    )
    assert not can_take_conditional_pile(state)
    assert DrawTrash() not in legal_actions(state)
    with pytest.raises(IllegalAction):
        apply_action(state, DrawTrash())


def test_opened_side_may_take_with_natural_plus_wild_when_unfrozen():
    state = make_state(
        cfg=CFG,
        hands=[[QC, JOKER, C9, C9], [C9], [C9], [C9]],
        stock=[QH],
        trash=[QS],
        phase=Phase.DRAW,
        initial_meld_done=[True, True],
    )
    assert can_take_conditional_pile(state)
    apply_action(state, DrawTrash())
    assert not state.pending_pile_pair_only
    acts = legal_actions(state)
    assert all(isinstance(a, CreateSet) and a.rank == Rank.QUEEN for a in acts)
    assert CreateSet(rank=Rank.QUEEN, wild=1) in acts  # joker allowed when unfrozen


def test_wild_top_or_frozen_pile_blocks_take():
    two_h = ct(Rank.TWO, Suit.HEARTS)
    state = make_state(
        cfg=CFG,
        hands=[[QC, QD, C9], [C9], [C9], [C9]],
        stock=[QH],
        trash=[two_h],  # wild on top
        phase=Phase.DRAW,
        initial_meld_done=[True, True],
    )
    assert not can_take_conditional_pile(state)

    # wild buried → frozen; natural top; only a pair unlocks it
    state2 = make_state(
        cfg=CFG,
        hands=[[QC, JOKER, C9], [C9], [C9], [C9]],
        stock=[QH],
        trash=[two_h, QS],
        phase=Phase.DRAW,
        initial_meld_done=[True, True],
        frozen=True,
    )
    assert not can_take_conditional_pile(state2)
    state2.hands[0][QD] += 1  # now a natural pair
    assert can_take_conditional_pile(state2)


def test_discarding_wild_freezes_pile():
    state = make_state(cfg=CFG, hands=[[JOKER, C9], [C9], [C9], [C9]])
    apply_action(state, Discard(JOKER))
    assert state.frozen


def test_black_three_blocks_next_player_only():
    state = make_state(
        cfg=CFG,
        hands=[[BLACK3_C, C9], [QC, QD, C9], [C9], [C9]],
        stock=[QH, QH, QH],
        trash=[QS],
        initial_meld_done=[True, True],
    )
    apply_action(state, Discard(BLACK3_C))
    assert state.pile_blocked_for_next
    # seat 1 cannot take the pile despite holding a queen pair
    assert not can_take_conditional_pile(state)
    apply_action(state, DrawDeck())  # seat 1 draws from stock instead
    assert not state.pile_blocked_for_next  # block expires after the draw


def test_frozen_take_with_existing_rank_set_joins_it():
    # Soak-found deadlock: frozen pile, natural pair in hand, but the side
    # already owns a set of that rank → the fresh set is barred by unique-rank,
    # so the forced use is joining the existing meld.
    from buraco.engine.actions import Add

    state = make_state(
        cfg=CFG,
        hands=[[QC, QD, C9, C9], [C9], [C9], [C9]],
        stock=[ct(Rank.EIGHT, Suit.HEARTS)] * 3,
        trash=[ct(Rank.TWO, Suit.HEARTS), QS],  # buried wild → frozen
        melds=[natural_set(Rank.QUEEN, 4)],
        phase=Phase.DRAW,
        initial_meld_done=[True, True],
        frozen=True,
    )
    assert can_take_conditional_pile(state)
    apply_action(state, DrawTrash())
    assert state.pending_pile_pair_only
    acts = legal_actions(state)
    assert acts and all(isinstance(a, Add) and Rank(a.ct % 13) == Rank.QUEEN for a in acts)
    apply_action(state, acts[0])
    assert state.pending_pile_card is None


# --- black-three melding -----------------------------------------------------------


def test_black_threes_meld_only_when_going_out():
    hand = [BLACK3_C, BLACK3_C, BLACK3_S, C9, C9]
    state = make_state(cfg=CFG, hands=[hand, [C9], [C9], [C9]],
                       melds=[natural_set(Rank.QUEEN, 7)])
    assert CreateSet(rank=Rank.THREE, wild=0) not in legal_actions(state)

    # going-out line: 3 threes + final discard, canasta on table
    state2 = make_state(cfg=CFG, hands=[[BLACK3_C, BLACK3_C, BLACK3_S, C9], [C9], [C9], [C9]],
                        melds=[natural_set(Rank.QUEEN, 7)])
    assert CreateSet(rank=Rank.THREE, wild=0) in legal_actions(state2)
    apply_action(state2, CreateSet(rank=Rank.THREE, wild=0))
    apply_action(state2, Discard(C9))
    assert state2.round_over and state2.went_out_side == 0


# --- initial-meld staging ------------------------------------------------------------


def test_staging_blocks_discard_until_threshold():
    # Aces are 20 each: one set of three = 60 ≥ 50 → opens in one move.
    aces = [card_id(Rank.ACE, s) for s in (Suit.CLUBS, Suit.DIAMONDS, Suit.HEARTS)]
    state = make_state(cfg=CFG, hands=[aces + [C9, C9], [C9], [C9], [C9]],
                       initial_meld_done=[False, True])
    apply_action(state, CreateSet(rank=Rank.ACE, wild=0))
    assert state.initial_meld_done[0]
    assert state.opened_on_turn[0] == state.turn_number
    assert any(isinstance(a, Discard) for a in legal_actions(state))


def test_staging_feasibility_masks_unreachable_openings():
    # Three 4s = 15 points, nothing else meldable → cannot reach 50; the
    # create is masked entirely (feasibility guard).
    fours = [card_id(Rank.FOUR, s) for s in (Suit.CLUBS, Suit.DIAMONDS, Suit.HEARTS)]
    state = make_state(cfg=CFG, hands=[fours + [C9], [C9], [C9], [C9]],
                       initial_meld_done=[False, True])
    assert CreateSet(rank=Rank.FOUR, wild=0) not in legal_actions(state)
    with pytest.raises(IllegalAction):
        apply_action(state, CreateSet(rank=Rank.FOUR, wild=0))
    # discarding without opening is always fine
    assert any(isinstance(a, Discard) for a in legal_actions(state))


def test_staging_across_multiple_melds():
    # 4s (15) + kings (30) = 45 < 50... add an ace pair + joker: use kings(30)
    # + aces with joker (20+20+50=90): total path exists; verify multi-meld
    # staging keeps discard masked until the threshold is crossed.
    kings = [card_id(Rank.KING, s) for s in (Suit.CLUBS, Suit.DIAMONDS, Suit.HEARTS)]
    aces = [card_id(Rank.ACE, Suit.CLUBS), card_id(Rank.ACE, Suit.DIAMONDS)]
    state = make_state(cfg=CFG, hands=[kings + aces + [JOKER, C9, C9], [C9], [C9], [C9]],
                       initial_meld_done=[False, True])
    apply_action(state, CreateSet(rank=Rank.KING, wild=0))  # 30 staged
    assert not state.initial_meld_done[0]
    assert not any(isinstance(a, Discard) for a in legal_actions(state))
    apply_action(state, CreateSet(rank=Rank.ACE, wild=1))  # +90 → opened
    assert state.initial_meld_done[0]
    assert any(isinstance(a, Discard) for a in legal_actions(state))


def test_max_stageable_points_bound():
    hand = Counter([QC, QD, JOKER, card_id(Rank.ACE, Suit.CLUBS)])
    # queens pair + joker = 10+10+50 = 70; lone ace unmeldable
    assert max_stageable_points(CFG, hand) == 70
    assert max_stageable_points(CFG, Counter([QC, C9])) == 0


def test_staging_order_cannot_strand():  # review blocker 1
    # Hand [4×FOUR, 4×FIVE, 3×ACE], threshold 50. Fours+fives total only 40,
    # so a four/five-first ordering melds the hand away below the threshold
    # and strands; the reserve-2 bound must mask everything except the
    # ace-first opening, and the mask must never empty.
    from buraco.engine.actions import Add

    fours = [card_id(Rank.FOUR, s) for s in Suit]
    fives = [card_id(Rank.FIVE, s) for s in Suit]
    aces = [card_id(Rank.ACE, s) for s in (Suit.CLUBS, Suit.DIAMONDS, Suit.HEARTS)]
    state = make_state(cfg=CFG, hands=[fours + fives + aces, [C9], [C9], [C9]],
                       initial_meld_done=[False, True])
    # Replay the reviewer's stranding prefix: fours melded away first is fine
    # (the ace continuation still fits the reserve-2 bound)...
    apply_action(state, CreateSet(rank=Rank.FOUR, wild=0))
    apply_action(state, Add(slot=0, ct=fours[3]))
    # ...but the step that would strand — melding the fives too — is masked,
    # and the ace opening remains available.
    acts = legal_actions(state)
    assert CreateSet(rank=Rank.FIVE, wild=0) not in acts
    assert CreateSet(rank=Rank.ACE, wild=0) in acts
    with pytest.raises(IllegalAction):
        apply_action(state, CreateSet(rank=Rank.FIVE, wild=0))
    apply_action(state, CreateSet(rank=Rank.ACE, wild=0))  # 60 pts → opened
    assert state.initial_meld_done[0]
    assert any(isinstance(a, Discard) for a in legal_actions(state))


def test_buried_red_three_take_rejected():  # review blocker 2
    # The buried red 3 goes to the tray on take, so the hand ends one card
    # short of the anti-stranding arithmetic unless it is subtracted.
    state = make_state(
        cfg=CFG,
        hands=[[QC, QD, C9], [C9], [C9], [C9]],
        stock=[QH] * 2,
        trash=[RED3_D, QS],  # red 3 buried under a takeable natural queen
        phase=Phase.DRAW,
        initial_meld_done=[True, True],
        frozen=True,
    )
    assert not can_take_conditional_pile(state)
    state.hands[0][ct(Rank.EIGHT, Suit.HEARTS)] += 1  # headroom restored
    assert can_take_conditional_pile(state)


def test_staging_fuzz_mask_never_empty():
    # Constructed unopened staging turns with random meld orderings — the
    # empirical backstop for the reserve-2 induction (review plan step 4).
    rng = random.Random(20260705)
    non_red3 = [ctype for ctype in range(53) if ctype not in RED_THREE_IDS]
    weights = [4 if ctype == 52 else 2 for ctype in non_red3]
    for trial in range(500):
        size = rng.randint(6, 16)
        hand = rng.choices(non_red3, weights=weights, k=size)
        state = make_state(
            cfg=CFG,
            hands=[hand, [C9], [C9], [C9]],
            stock=[C9] * 8,
            initial_meld_done=[False, True],
        )
        for _ in range(60):
            if state.round_over or state.phase is Phase.DRAW:
                break
            acts = legal_actions(state)
            assert acts, f"trial {trial}: mask emptied mid-staging"
            apply_action(state, rng.choice(acts))


# --- scoring --------------------------------------------------------------------------


def test_canasta_scoring_red_threes_and_bonuses():
    state = make_state(
        cfg=CFG,
        hands=[[], [C9], [], [C9]],
        melds=[natural_set(Rank.QUEEN, 7, owner=0)],  # natural canasta: 70 + 500
        initial_meld_done=[True, False],
    )
    state.red_threes[0] = [RED3_D]
    state.red_threes[1] = [RED_THREE_IDS[1]]
    state.round_over = True
    state.end_reason = EndReason.STOCK_EXHAUSTED
    scores = round_scores(state)
    assert scores[0] == 70 + 500 + 100  # melds + natural canasta + red 3
    assert scores[1] == -100 - 10 - 10  # red 3 negative (never melded) + two 9s in hand


def test_concealed_bonus_when_opening_on_the_going_out_turn():
    aces = [card_id(Rank.ACE, s) for s in Suit] + [card_id(Rank.ACE, Suit.CLUBS),
                                                   card_id(Rank.ACE, Suit.DIAMONDS),
                                                   card_id(Rank.ACE, Suit.HEARTS)]
    state = make_state(cfg=CFG, hands=[aces + [C9], [C9], [C9], [C9]],
                       initial_meld_done=[False, True])
    apply_action(state, CreateSet(rank=Rank.ACE, wild=0))  # 60 → opened this turn
    from buraco.engine.actions import Add

    for _ in range(4):  # grow to a 7-ace canasta
        acts = [a for a in legal_actions(state) if isinstance(a, Add)]
        apply_action(state, acts[0])
    apply_action(state, Discard(C9))  # goes out
    assert state.round_over and state.went_out_side == 0
    scores = round_scores(state)
    # 7 aces (140) + natural canasta 500 + go-out 100 + concealed 100,
    # minus the partner's remaining 9 (10) — the side's hands all count.
    assert scores[0] == 140 + 500 + 100 + 100 - 10


# --- soak -----------------------------------------------------------------------------


def test_soak_canasta_random():
    for cfg, n in ((canasta(4), 30), (canasta(2), 20)):
        for seed in range(n):
            play_engine_game(cfg, 88_000 + seed)


def test_canasta_env_smoke():
    from buraco.env.env import BuracoEnv

    env = BuracoEnv(canasta(4))
    obs, info = env.reset(seed=13)
    rng = random.Random(5)
    for _ in range(400):
        if not info["legal_actions"]:
            break
        obs, _, term, trunc, info = env.step(rng.choice(info["legal_actions"]))
        if term or trunc:
            break
    assert obs["red_threes"].shape == (2,)
    assert obs["staged"].shape == (2,)
