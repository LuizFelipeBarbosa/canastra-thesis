"""Parity of the precomputed hot-path tables against the original formulas.

The RL data stream depends on exact action-mask equality, so every table-backed
function must reproduce the pre-table implementation bit-for-bit. The ``ref_*``
functions below are verbatim copies of the original (formula-based) code.
"""

from __future__ import annotations

import random
from collections import Counter

import pytest

from buraco.cards import (
    JOKER,
    POS_MAX,
    POS_MIN,
    RED_SUITS,
    Rank,
    Suit,
    card_id,
    id_rank,
    id_suit,
    is_red,
    nat,
    rank_at,
    rank_name,
)
from buraco.engine.melds import (
    SEQ_WILD_JOKER,
    SEQ_WILD_NONE,
    SEQ_WILD_OFF_SUIT_TWO,
    SEQ_WILD_TWO_OF_SUIT,
    SET_WILD_JOKER,
    SET_WILD_NONE,
    CreatePlan,
    SlotRole,
    is_natural_at,
    plan_sequence,
    plan_set,
    pos_allowed,
)
from buraco.profiles import load_profile

PROFILES = [
    ("buraco", 2),
    ("buraco", 4),
    ("canasta", 2),
    ("biriba", 4),
    ("rummy", 2),
]


def profile_cfgs():
    return [load_profile(name, num_players=n) for name, n in PROFILES]


# --- reference copies of the original implementations -------------------------


def ref_rank_at(pos):
    if not POS_MIN <= pos <= POS_MAX:
        raise ValueError(f"position out of range: {pos}")
    return Rank.ACE if pos in (POS_MIN, POS_MAX) else Rank(pos - 1)


def ref_nat(pos, suit):
    return suit * 13 + ref_rank_at(pos)


def ref_id_rank(ct):
    return None if ct == JOKER else Rank(ct % 13)


def ref_id_suit(ct):
    return None if ct == JOKER else Suit(ct // 13)


def ref_is_red(ct):
    return ct != JOKER and Suit(ct // 13) in RED_SUITS


def ref_is_wild_card(cfg, ct):
    if ct == JOKER:
        return cfg.wildcard.jokers_wild
    return ref_id_rank(ct) in cfg.wildcard.wild_ranks


def ref_card_value(cfg, ct):
    return cfg.scoring.card_points[rank_name(ct)]


def ref_is_natural_at(cfg, ct, pos, suit):
    if ct != ref_nat(pos, suit):
        return False
    if not ref_is_wild_card(cfg, ct):
        return True
    return cfg.wildcard.natural_two_in_suit


def ref_hand_covers(hand, consumed):
    needed = {}
    for c in consumed:
        needed[c] = needed.get(c, 0) + 1
    return all(hand.get(c, 0) >= n for c, n in needed.items())


def ref_seq_wild_card(cfg, hand, suit, gap_pos, wild_choice):
    if cfg.wildcard.wildcard_limit_per_meld < 1:
        return None
    if wild_choice == SEQ_WILD_JOKER:
        if cfg.wildcard.jokers_wild and hand.get(JOKER, 0) > 0:
            return JOKER
        return None
    if wild_choice == SEQ_WILD_TWO_OF_SUIT:
        ct = card_id(Rank.TWO, suit)
        if Rank.TWO in cfg.wildcard.wild_ranks and hand.get(ct, 0) > 0:
            if ref_is_natural_at(cfg, ct, gap_pos, suit):
                return None
            return ct
        return None
    if wild_choice == SEQ_WILD_OFF_SUIT_TWO:
        if Rank.TWO not in cfg.wildcard.wild_ranks:
            return None
        for other in Suit:
            if other == suit:
                continue
            ct = card_id(Rank.TWO, other)
            if hand.get(ct, 0) > 0:
                return ct
        return None
    return None


def ref_plan_sequence(cfg, hand, suit, start_pos, wild_choice):
    if not cfg.meld.allow_sequences or cfg.meld.min_meld_size > 3:
        return None
    positions = (start_pos, start_pos + 1, start_pos + 2)
    if start_pos < POS_MIN or positions[-1] > POS_MAX:
        return None
    if not all(pos_allowed(cfg, p) for p in positions):
        return None

    held = [p for p in positions if hand.get(ref_nat(p, suit), 0) > 0
            and ref_is_natural_at(cfg, ref_nat(p, suit), p, suit)]

    if wild_choice == SEQ_WILD_NONE:
        if len(held) != 3:
            return None
        slots = tuple((ref_nat(p, suit), SlotRole.NATURAL) for p in positions)
        return CreatePlan(consumed=tuple(c for c, _ in slots), slots=slots)

    if len(held) != 2:
        return None
    gap = next(p for p in positions if p not in held)
    wild_ct = ref_seq_wild_card(cfg, hand, suit, gap, wild_choice)
    if wild_ct is None:
        return None
    slots = tuple(
        (wild_ct, SlotRole.WILD) if p == gap else (ref_nat(p, suit), SlotRole.NATURAL)
        for p in positions
    )
    consumed = tuple(c for c, _ in slots)
    if not ref_hand_covers(hand, consumed):
        return None
    return CreatePlan(consumed=consumed, slots=slots)


def ref_plan_set(cfg, hand, rank, wild_choice, prefer=None):
    if not cfg.meld.allow_sets or cfg.meld.min_meld_size > 3:
        return None
    if rank in cfg.wildcard.wild_ranks:
        return None

    need_naturals = 3 if wild_choice == SET_WILD_NONE else 2
    if wild_choice != SET_WILD_NONE:
        if cfg.wildcard.wildcard_limit_per_meld < 1:
            return None
        if cfg.wildcard.min_naturals_per_meld > 2:
            return None

    naturals = []
    if prefer is not None:
        if ref_id_rank(prefer) != rank or hand.get(prefer, 0) < 1:
            return None
        naturals.append(prefer)
    for suit in Suit:
        ct = card_id(rank, suit)
        avail = hand.get(ct, 0) - (1 if ct == prefer else 0)
        take = min(avail, need_naturals - len(naturals))
        naturals.extend([ct] * take)
        if len(naturals) == need_naturals:
            break
    if len(naturals) < need_naturals:
        return None

    if wild_choice == SET_WILD_NONE:
        slots = tuple((c, SlotRole.NATURAL) for c in naturals)
        return CreatePlan(consumed=tuple(naturals), slots=slots)

    if wild_choice == SET_WILD_JOKER:
        if not (cfg.wildcard.jokers_wild and hand.get(JOKER, 0) > 0):
            return None
        wild_ct = JOKER
    else:
        if Rank.TWO not in cfg.wildcard.wild_ranks:
            return None
        wild_ct = next(
            (card_id(Rank.TWO, s) for s in Suit if hand.get(card_id(Rank.TWO, s), 0) > 0),
            -1,
        )
        if wild_ct < 0:
            return None

    slots = tuple([(c, SlotRole.NATURAL) for c in naturals] + [(wild_ct, SlotRole.WILD)])
    consumed = tuple(naturals) + (wild_ct,)
    if not ref_hand_covers(hand, consumed):
        return None
    return CreatePlan(consumed=consumed, slots=slots)


# --- card-level parity ---------------------------------------------------------


def test_rank_at_parity():
    for pos in range(POS_MIN, POS_MAX + 1):
        assert rank_at(pos) is ref_rank_at(pos)
    for pos in (0, POS_MAX + 1):
        with pytest.raises(ValueError):
            rank_at(pos)


def test_nat_parity_all_56():
    for suit in Suit:
        for pos in range(POS_MIN, POS_MAX + 1):
            assert nat(pos, suit) == ref_nat(pos, suit)
    for pos in (0, POS_MAX + 1):
        with pytest.raises(ValueError):
            nat(pos, Suit.CLUBS)


def test_id_rank_id_suit_is_red_parity():
    for ct in range(53):
        assert id_rank(ct) is ref_id_rank(ct)
        assert id_suit(ct) is ref_id_suit(ct)
        assert is_red(ct) == ref_is_red(ct)
    # id_rank's modulo quirk extends to PAD and is relied on by is_wild_card.
    assert id_rank(53) is ref_id_rank(53)


def test_is_wild_card_and_card_value_parity():
    for cfg in profile_cfgs():
        for ct in range(54):
            assert cfg.is_wild_card(ct) == ref_is_wild_card(cfg, ct), (cfg.name, ct)
            assert cfg.card_value(ct) == ref_card_value(cfg, ct), (cfg.name, ct)


def test_is_natural_at_parity_exhaustive():
    for cfg in profile_cfgs():
        for suit in Suit:
            for pos in range(POS_MIN, POS_MAX + 1):
                for ct in range(53):
                    assert is_natural_at(cfg, ct, pos, suit) == ref_is_natural_at(
                        cfg, ct, pos, suit
                    ), (cfg.name, ct, pos, suit)
    with pytest.raises(ValueError):
        is_natural_at(profile_cfgs()[0], 0, 0, Suit.CLUBS)


# --- planner parity on random hands --------------------------------------------


def random_hand(rng: random.Random) -> Counter:
    size = rng.randint(1, 15)
    cards = [rng.randrange(53) for _ in range(size)]
    return Counter(cards)


def test_plan_sequence_parity_random_hands():
    rng = random.Random(20260706)
    cfgs = profile_cfgs()
    for _ in range(300):
        hand = random_hand(rng)
        for cfg in cfgs:
            for suit in Suit:
                for start in range(1, 13):
                    for wild in range(4):
                        assert plan_sequence(cfg, hand, suit, start, wild) == (
                            ref_plan_sequence(cfg, hand, suit, start, wild)
                        ), (cfg.name, dict(hand), suit, start, wild)


def test_plan_set_parity_random_hands():
    rng = random.Random(20260707)
    cfgs = profile_cfgs()
    for _ in range(300):
        hand = random_hand(rng)
        prefer_pool = [None] + list(hand)
        for cfg in cfgs:
            for rank in Rank:
                for wild in range(3):
                    prefer = prefer_pool[rng.randrange(len(prefer_pool))]
                    assert plan_set(cfg, hand, rank, wild, prefer=prefer) == (
                        ref_plan_set(cfg, hand, rank, wild, prefer=prefer)
                    ), (cfg.name, dict(hand), rank, wild, prefer)
