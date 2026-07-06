"""Integer action codec: layout, round-trips, illegal-id guards (SPEC 02 §2.3–2.4)."""

import pytest

from buraco.cards import Rank, Suit
from buraco.engine.actions import (
    BASE_ADD,
    BASE_SEQ,
    BASE_SET,
    Add,
    CreateSeq,
    CreateSet,
    Discard,
    DrawDeck,
    DrawTrash,
    EndRound,
    GoOut,
    action_space_size,
    base_discard,
    decode,
    encode,
)

S = 24  # default max_meld_slots


def test_layout_constants_match_spec():
    assert BASE_SEQ == 2
    assert BASE_SET == 194
    assert BASE_ADD == 233
    assert base_discard(S) == 1529
    assert action_space_size(S) == 1585
    assert encode(GoOut(), S) == 1583
    assert encode(EndRound(), S) == 1584


def test_round_trip_all_ids():
    for a in range(action_space_size(S)):
        assert encode(decode(a, S), S) == a


def test_round_trip_structs():
    samples = [
        DrawDeck(), DrawTrash(), GoOut(), EndRound(),
        CreateSeq(suit=Suit.HEARTS, start=4, wild=1),
        CreateSeq(suit=Suit.SPADES, start=12, wild=3),
        CreateSet(rank=Rank.KING, wild=0),
        CreateSet(rank=Rank.TWO, wild=2),  # rank-2 ids exist (D14), masked in Buraco
        Add(slot=0, ct=0), Add(slot=23, ct=53),
        Discard(ct=52),
    ]
    for act in samples:
        assert decode(encode(act, S), S) == act


def test_encode_rejects_out_of_range_operands():
    for bad in [
        CreateSeq(suit=Suit.HEARTS, start=13, wild=0),  # start beyond shape range
        CreateSeq(suit=Suit.HEARTS, start=0, wild=0),
        CreateSeq(suit=Suit.HEARTS, start=4, wild=4),
        CreateSet(rank=Rank.KING, wild=3),
        Add(slot=S, ct=0),
        Add(slot=0, ct=54),
        Discard(ct=54),
    ]:
        with pytest.raises(AssertionError):
            encode(bad, S)


def test_decode_rejects_out_of_range_id():
    with pytest.raises(ValueError):
        decode(-1, S)
    with pytest.raises(ValueError):
        decode(action_space_size(S), S)


def test_slot_budget_shifts_only_tail_families():
    assert base_discard(30) == BASE_ADD + 30 * 54
    assert action_space_size(30) == base_discard(30) + 54 + 2
    a = encode(CreateSeq(suit=Suit.CLUBS, start=1, wild=0), 30)
    assert a == encode(CreateSeq(suit=Suit.CLUBS, start=1, wild=0), S)
