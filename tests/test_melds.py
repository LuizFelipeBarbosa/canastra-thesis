"""Meld creation/extension, wildcard resolution, canastra detection (M3).

Scenario numbers refer to docs/specs/test-scenarios-rules.md.
"""

from collections import Counter
from dataclasses import replace

import pytest

from buraco.cards import JOKER, Rank, Suit, card_id
from buraco.config import ACE_LOW_ONLY, WILD_TO_HAND
from buraco.engine.melds import (
    SEQ_WILD_JOKER,
    SEQ_WILD_NONE,
    SEQ_WILD_OFF_SUIT_TWO,
    SEQ_WILD_TWO_OF_SUIT,
    SET_WILD_JOKER,
    SET_WILD_NONE,
    SET_WILD_TWO,
    Meld,
    MeldError,
    MeldKind,
    Slot,
    SlotRole,
    apply_add,
    create_sequence,
    create_set,
    plan_add,
    plan_sequence,
    plan_set,
    validate_meld,
)
from buraco.profiles import buraco

CFG = buraco()


def ct(rank: Rank, suit: Suit) -> int:
    return card_id(rank, suit)


def hand_of(*cards: int) -> Counter:
    return Counter(cards)


# --- creation ------------------------------------------------------------------


def test_create_natural_sequence():
    hand = hand_of(ct(Rank.FOUR, Suit.HEARTS), ct(Rank.FIVE, Suit.HEARTS), ct(Rank.SIX, Suit.HEARTS))
    meld = create_sequence(CFG, hand, owner=0, meld_id=0, suit=Suit.HEARTS,
                           start_pos=4, wild_choice=SEQ_WILD_NONE)
    assert meld.size == 3 and meld.is_clean
    assert meld.start_pos == 4 and meld.end_pos == 6
    assert not hand  # all consumed


def test_create_sequence_with_joker_gap():
    hand = hand_of(ct(Rank.FOUR, Suit.DIAMONDS), ct(Rank.SIX, Suit.DIAMONDS), JOKER)
    meld = create_sequence(CFG, hand, 0, 0, Suit.DIAMONDS, 4, SEQ_WILD_JOKER)
    assert meld.wild_count == 1 and not meld.is_clean
    assert meld.slots[1].card == JOKER and meld.slots[1].role is SlotRole.WILD
    assert meld.represented_rank(1) == Rank.FIVE


def test_create_rejects_two_gaps():
    hand = hand_of(ct(Rank.FOUR, Suit.CLUBS), JOKER)
    assert plan_sequence(CFG, hand, Suit.CLUBS, 4, SEQ_WILD_JOKER) is None
    with pytest.raises(MeldError):
        create_sequence(CFG, hand, 0, 0, Suit.CLUBS, 4, SEQ_WILD_JOKER)


def test_off_suit_two_canonical_choice():
    hand = hand_of(
        ct(Rank.NINE, Suit.HEARTS), ct(Rank.JACK, Suit.HEARTS),
        ct(Rank.TWO, Suit.SPADES), ct(Rank.TWO, Suit.CLUBS),
    )
    meld = create_sequence(CFG, hand, 0, 0, Suit.HEARTS, 9, SEQ_WILD_OFF_SUIT_TWO)
    wild_slot = meld.slots[meld.wild_pos_index]
    assert wild_slot.card == ct(Rank.TWO, Suit.CLUBS)  # lowest suit index != HEARTS
    assert hand == hand_of(ct(Rank.TWO, Suit.SPADES))


def test_two_of_suit_natural_position_uses_natural_path():
    # Holding A♥ 2♥ 3♥: the 2♥ covers position 2 as a NATURAL, so the meld is
    # created wild-free; the wild-create encoding for the same span is illegal.
    hand = hand_of(ct(Rank.ACE, Suit.HEARTS), ct(Rank.TWO, Suit.HEARTS), ct(Rank.THREE, Suit.HEARTS))
    assert plan_sequence(CFG, hand, Suit.HEARTS, 1, SEQ_WILD_TWO_OF_SUIT) is None
    meld = create_sequence(CFG, hand, 0, 0, Suit.HEARTS, 1, SEQ_WILD_NONE)
    assert meld.is_clean and all(s.role is SlotRole.NATURAL for s in meld.slots)


def test_create_set_natural_duplicates_canonical():
    hand = hand_of(ct(Rank.KING, Suit.CLUBS), ct(Rank.KING, Suit.CLUBS), ct(Rank.KING, Suit.DIAMONDS))
    meld = create_set(CFG, hand, 0, 0, Rank.KING, SET_WILD_NONE)
    assert meld.kind is MeldKind.SET and meld.size == 3 and meld.is_clean
    assert not hand


def test_create_set_with_wild_two():
    hand = hand_of(ct(Rank.KING, Suit.HEARTS), ct(Rank.KING, Suit.SPADES), ct(Rank.TWO, Suit.DIAMONDS))
    meld = create_set(CFG, hand, 0, 0, Rank.KING, SET_WILD_TWO)
    assert meld.wild_count == 1 and not meld.is_clean


def test_scenario23_set_of_wilds_rejected():
    hand = hand_of(ct(Rank.TWO, Suit.HEARTS), ct(Rank.TWO, Suit.SPADES), JOKER)
    assert plan_set(CFG, hand, Rank.TWO, SET_WILD_NONE) is None
    assert plan_set(CFG, hand, Rank.TWO, SET_WILD_JOKER) is None


# --- wild limit (scenarios 17, 24) ----------------------------------------------


def test_scenario17_second_wild_rejected_on_sequence():
    hand = hand_of(ct(Rank.FOUR, Suit.SPADES), ct(Rank.SIX, Suit.SPADES), JOKER,
                   ct(Rank.TWO, Suit.HEARTS))
    meld = create_sequence(CFG, hand, 0, 0, Suit.SPADES, 4, SEQ_WILD_JOKER)
    assert plan_add(CFG, hand, meld, ct(Rank.TWO, Suit.HEARTS)) is None
    with pytest.raises(MeldError):
        apply_add(CFG, hand, meld, ct(Rank.TWO, Suit.HEARTS))


def test_scenario24_set_wild_limit_and_canasta_profile():
    hand = hand_of(ct(Rank.KING, Suit.HEARTS), ct(Rank.KING, Suit.SPADES),
                   ct(Rank.TWO, Suit.DIAMONDS), JOKER)
    meld = create_set(CFG, hand, 0, 0, Rank.KING, SET_WILD_TWO)
    assert plan_add(CFG, hand, meld, JOKER) is None  # Buraco: limit 1

    canasta_ish = replace(CFG, wildcard=replace(CFG.wildcard, wildcard_limit_per_meld=3))
    hand2 = hand_of(ct(Rank.KING, Suit.HEARTS), ct(Rank.KING, Suit.SPADES),
                    ct(Rank.TWO, Suit.DIAMONDS), JOKER)
    meld2 = create_set(canasta_ish, hand2, 0, 0, Rank.KING, SET_WILD_TWO)
    apply_add(canasta_ish, hand2, meld2, JOKER)
    assert meld2.wild_count == 2 and meld2.size == 4


# --- extension and swap-relocation (scenarios 16, 18, 19, 20, 21, 22) -----------


def build_hearts_16(cfg=CFG):
    """Scenario 16 meld: A♥-2♥-3♥-4♥-2♦(w@5)-6♥."""
    hand = hand_of(
        ct(Rank.ACE, Suit.HEARTS), ct(Rank.TWO, Suit.HEARTS), ct(Rank.THREE, Suit.HEARTS),
        ct(Rank.FOUR, Suit.HEARTS), ct(Rank.TWO, Suit.DIAMONDS), ct(Rank.SIX, Suit.HEARTS),
    )
    meld = create_sequence(cfg, hand, 0, 0, Suit.HEARTS, 1, SEQ_WILD_NONE)
    apply_add(cfg, hand, meld, ct(Rank.FOUR, Suit.HEARTS))
    apply_add(cfg, hand, meld, ct(Rank.TWO, Suit.DIAMONDS))  # wild onto high end (pos 5)
    apply_add(cfg, hand, meld, ct(Rank.SIX, Suit.HEARTS))
    return hand, meld


def test_scenario16_natural_two_and_wild_two_coexist():
    _, meld = build_hearts_16()
    assert meld.size == 6 and meld.wild_count == 1
    assert meld.slots[1].card == ct(Rank.TWO, Suit.HEARTS)
    assert meld.slots[1].role is SlotRole.NATURAL
    assert meld.slots[4].card == ct(Rank.TWO, Suit.DIAMONDS)
    assert meld.slots[4].role is SlotRole.WILD
    assert meld.represented_rank(4) == Rank.FIVE


def test_scenario18_swap_relocates_low_first():
    hand = hand_of(ct(Rank.FOUR, Suit.DIAMONDS), ct(Rank.FIVE, Suit.DIAMONDS), JOKER,
                   ct(Rank.SEVEN, Suit.DIAMONDS))
    meld = create_sequence(CFG, hand, 0, 0, Suit.DIAMONDS, 4, SEQ_WILD_JOKER)  # 4-5-J(6)
    apply_add(CFG, hand, meld, ct(Rank.SEVEN, Suit.DIAMONDS))  # 4-5-J(6)-7
    hand[ct(Rank.SIX, Suit.DIAMONDS)] += 1  # drawn later
    apply_add(CFG, hand, meld, ct(Rank.SIX, Suit.DIAMONDS))  # swap
    assert meld.start_pos == 3 and meld.end_pos == 7
    assert meld.slots[0].card == JOKER and meld.slots[0].role is SlotRole.WILD
    assert meld.represented_rank(0) == Rank.THREE
    assert all(s.role is SlotRole.NATURAL for s in meld.slots[1:])


def test_scenario19_swap_relocates_high_when_low_blocked():
    hand = hand_of(ct(Rank.ACE, Suit.DIAMONDS), ct(Rank.THREE, Suit.DIAMONDS), JOKER)
    meld = create_sequence(CFG, hand, 0, 0, Suit.DIAMONDS, 1, SEQ_WILD_JOKER)  # A-J(2)-3
    hand[ct(Rank.TWO, Suit.DIAMONDS)] += 1  # drawn later
    apply_add(CFG, hand, meld, ct(Rank.TWO, Suit.DIAMONDS))  # natural-2 swap
    assert meld.start_pos == 1 and meld.end_pos == 4
    assert meld.slots[1].role is SlotRole.NATURAL  # the 2♦ is natural at pos 2
    assert meld.slots[3].card == JOKER and meld.represented_rank(3) == Rank.FOUR


def test_scenario20_swap_blocked_on_full_span():
    slots = []
    for pos in range(1, 15):
        if pos == 6:
            slots.append(Slot(JOKER, SlotRole.WILD))
        else:
            suit_ct = ct(Rank.ACE if pos in (1, 14) else Rank(pos - 1), Suit.CLUBS)
            slots.append(Slot(suit_ct, SlotRole.NATURAL))
    meld = Meld(0, 0, MeldKind.SEQUENCE, suit=Suit.CLUBS, start_pos=1, slots=slots)
    validate_meld(CFG, meld)
    hand = hand_of(ct(Rank.SIX, Suit.CLUBS))
    assert plan_add(CFG, hand, meld, ct(Rank.SIX, Suit.CLUBS)) is None


def test_scenario21_ace_low_run_with_separate_wild():
    hand = hand_of(ct(Rank.ACE, Suit.HEARTS), ct(Rank.TWO, Suit.HEARTS),
                   ct(Rank.THREE, Suit.HEARTS), ct(Rank.FOUR, Suit.HEARTS),
                   JOKER, ct(Rank.SIX, Suit.HEARTS))
    meld = create_sequence(CFG, hand, 0, 0, Suit.HEARTS, 1, SEQ_WILD_NONE)
    apply_add(CFG, hand, meld, ct(Rank.FOUR, Suit.HEARTS))
    apply_add(CFG, hand, meld, JOKER)  # wild onto pos 5 (low end blocked)
    apply_add(CFG, hand, meld, ct(Rank.SIX, Suit.HEARTS))
    assert meld.size == 6 and meld.wild_count == 1
    assert meld.slots[4].card == JOKER and meld.represented_rank(4) == Rank.FIVE


def test_scenario22_swap_relocates_wild_to_ace_slot():
    hand = hand_of(ct(Rank.TWO, Suit.DIAMONDS), ct(Rank.THREE, Suit.DIAMONDS), JOKER,
                   ct(Rank.FIVE, Suit.DIAMONDS))
    meld = create_sequence(CFG, hand, 0, 0, Suit.DIAMONDS, 2, SEQ_WILD_JOKER)  # 2-3-J(4)
    apply_add(CFG, hand, meld, ct(Rank.FIVE, Suit.DIAMONDS))
    hand[ct(Rank.FOUR, Suit.DIAMONDS)] += 1  # drawn later
    apply_add(CFG, hand, meld, ct(Rank.FOUR, Suit.DIAMONDS))  # swap → joker to pos 1
    assert meld.start_pos == 1 and meld.slots[0].card == JOKER
    assert meld.represented_rank(0) == Rank.ACE


def test_wild_two_relocating_into_natural_position_becomes_natural():
    # 3♥-2♥(w@4)-5♥ + 4♥ → the freed 2♥ slides to position 2, its own natural
    # slot, and the meld turns clean.
    hand = hand_of(ct(Rank.THREE, Suit.HEARTS), ct(Rank.FIVE, Suit.HEARTS),
                   ct(Rank.TWO, Suit.HEARTS))
    meld = create_sequence(CFG, hand, 0, 0, Suit.HEARTS, 3, SEQ_WILD_TWO_OF_SUIT)
    assert meld.wild_count == 1
    hand[ct(Rank.FOUR, Suit.HEARTS)] += 1  # drawn later
    apply_add(CFG, hand, meld, ct(Rank.FOUR, Suit.HEARTS))
    assert meld.start_pos == 2
    assert meld.slots[0].card == ct(Rank.TWO, Suit.HEARTS)
    assert meld.slots[0].role is SlotRole.NATURAL
    assert meld.is_clean


def test_wild_to_hand_relocation_policy():
    cfg = replace(CFG, wildcard=replace(CFG.wildcard, wild_relocation=WILD_TO_HAND))
    hand = hand_of(ct(Rank.FOUR, Suit.DIAMONDS), ct(Rank.FIVE, Suit.DIAMONDS), JOKER)
    meld = create_sequence(cfg, hand, 0, 0, Suit.DIAMONDS, 4, SEQ_WILD_JOKER)
    hand[ct(Rank.SIX, Suit.DIAMONDS)] += 1  # drawn later
    apply_add(cfg, hand, meld, ct(Rank.SIX, Suit.DIAMONDS))
    assert meld.size == 3 and meld.is_clean
    assert hand[JOKER] == 1  # freed wild returned to hand


# --- ace policy ------------------------------------------------------------------


def test_ace_policy_low_only():
    cfg = replace(CFG, meld=replace(CFG.meld, ace_policy=ACE_LOW_ONLY))
    high = hand_of(ct(Rank.QUEEN, Suit.CLUBS), ct(Rank.KING, Suit.CLUBS), ct(Rank.ACE, Suit.CLUBS))
    assert plan_sequence(cfg, high, Suit.CLUBS, 12, SEQ_WILD_NONE) is None
    low = hand_of(ct(Rank.ACE, Suit.CLUBS), ct(Rank.TWO, Suit.CLUBS), ct(Rank.THREE, Suit.CLUBS))
    assert plan_sequence(cfg, low, Suit.CLUBS, 1, SEQ_WILD_NONE) is not None


# --- canastra detection (scenarios 31, 32, 35) ------------------------------------


def make_run(cfg, suit, start, length, hand_extra=()):
    cards = [ct(Rank(p - 1) if p not in (1, 14) else Rank.ACE, suit)
             for p in range(start, start + length)]
    hand = Counter(cards) + Counter(hand_extra)
    meld = create_sequence(cfg, hand, 0, 0, suit, start, SEQ_WILD_NONE)
    for c in cards[3:]:
        apply_add(cfg, hand, meld, c)
    return hand, meld


def test_scenario31_32_canastra_transitions():
    hand, meld = make_run(CFG, Suit.HEARTS, 3, 7, hand_extra=(ct(Rank.TWO, Suit.DIAMONDS),
                                                              ct(Rank.TEN, Suit.HEARTS)))
    assert meld.is_canastra(CFG.meld.canastra_min_size) and meld.is_clean  # limpa

    apply_add(CFG, hand, meld, ct(Rank.TEN, Suit.HEARTS))  # grow past 7, stays clean
    assert meld.size == 8 and meld.is_clean

    apply_add(CFG, hand, meld, ct(Rank.TWO, Suit.DIAMONDS))  # wild add → suja
    assert meld.size == 9 and not meld.is_clean
    assert meld.is_canastra(CFG.meld.canastra_min_size)


def test_scenario35_natural_two_canastra_is_clean():
    cards = [ct(Rank.TWO, Suit.HEARTS), ct(Rank.THREE, Suit.HEARTS), ct(Rank.FOUR, Suit.HEARTS),
             ct(Rank.FIVE, Suit.HEARTS), ct(Rank.SIX, Suit.HEARTS), ct(Rank.SEVEN, Suit.HEARTS),
             ct(Rank.EIGHT, Suit.HEARTS)]
    hand = Counter(cards)
    meld = create_sequence(CFG, hand, 0, 0, Suit.HEARTS, 2, SEQ_WILD_NONE)
    for c in cards[3:]:
        apply_add(CFG, hand, meld, c)
    assert meld.size == 7 and meld.is_clean
    assert meld.is_canastra(CFG.meld.canastra_min_size)


# --- structural validation ---------------------------------------------------------


def test_validate_rejects_bad_natural_claim():
    meld = Meld(0, 0, MeldKind.SEQUENCE, suit=Suit.CLUBS, start_pos=4, slots=[
        Slot(ct(Rank.FOUR, Suit.CLUBS), SlotRole.NATURAL),
        Slot(ct(Rank.NINE, Suit.CLUBS), SlotRole.NATURAL),  # wrong card for pos 5
        Slot(ct(Rank.SIX, Suit.CLUBS), SlotRole.NATURAL),
    ])
    with pytest.raises(MeldError):
        validate_meld(CFG, meld)


def test_validate_rejects_wild_role_on_natural_two():
    meld = Meld(0, 0, MeldKind.SEQUENCE, suit=Suit.HEARTS, start_pos=2, slots=[
        Slot(ct(Rank.TWO, Suit.HEARTS), SlotRole.WILD),  # must be NATURAL at pos 2
        Slot(ct(Rank.THREE, Suit.HEARTS), SlotRole.NATURAL),
        Slot(ct(Rank.FOUR, Suit.HEARTS), SlotRole.NATURAL),
    ])
    with pytest.raises(MeldError):
        validate_meld(CFG, meld)


def test_validate_rejects_wrong_rank_in_set():
    meld = Meld(0, 0, MeldKind.SET, rank=Rank.KING, slots=[
        Slot(ct(Rank.KING, Suit.CLUBS), SlotRole.NATURAL),
        Slot(ct(Rank.QUEEN, Suit.CLUBS), SlotRole.NATURAL),
        Slot(ct(Rank.KING, Suit.HEARTS), SlotRole.NATURAL),
    ])
    with pytest.raises(MeldError):
        validate_meld(CFG, meld)


def test_single_two_cannot_be_natural_and_wild_at_once():
    # Regression: one 2♠ cannot serve as both the natural at position 2 and
    # the wild for the gap; with a second copy (two decks) it can.
    hand = hand_of(ct(Rank.ACE, Suit.SPADES), ct(Rank.TWO, Suit.SPADES))
    assert plan_sequence(CFG, hand, Suit.SPADES, 1, SEQ_WILD_TWO_OF_SUIT) is None
    hand[ct(Rank.TWO, Suit.SPADES)] += 1  # second copy
    plan = plan_sequence(CFG, hand, Suit.SPADES, 1, SEQ_WILD_TWO_OF_SUIT)
    assert plan is not None
    meld = create_sequence(CFG, hand, 0, 0, Suit.SPADES, 1, SEQ_WILD_TWO_OF_SUIT)
    assert meld.slots[1].role is SlotRole.NATURAL  # 2♠ natural at position 2
    assert meld.slots[2].role is SlotRole.WILD  # second 2♠ wild at position 3
    assert not hand


def test_add_requires_card_in_hand():
    hand, meld = make_run(CFG, Suit.SPADES, 5, 3)
    assert plan_add(CFG, hand, meld, ct(Rank.EIGHT, Suit.SPADES)) is None
