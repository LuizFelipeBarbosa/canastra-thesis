"""Dealing, seeded reproducibility, conservation, and state serialization."""

import random
from collections import Counter
from dataclasses import replace

import pytest

from buraco.cards import build_deck
from buraco.engine.serialize import round_from_dict, round_to_dict, state_hash
from buraco.engine.state import Phase, deal_round, dealt_multiset
from buraco.profiles import buraco


def test_deal_2p_zone_sizes():
    state = deal_round(buraco(2), random.Random(7))
    assert [state.hand_size(p) for p in range(2)] == [11, 11]
    assert len(state.morto) == 2 and all(len(m) == 11 for m in state.morto)
    assert state.morto_taken == [False, False]
    assert len(state.stock) == 104 - 22 - 22 == 60
    assert state.trash == []
    assert state.melds == []
    assert state.phase == Phase.DRAW
    assert state.current_player == 0


def test_deal_4p_zone_sizes():
    state = deal_round(buraco(4), random.Random(7))
    assert [state.hand_size(p) for p in range(4)] == [11, 11, 11, 11]
    assert len(state.morto) == 2
    assert len(state.stock) == 104 - 44 - 22 == 38


def test_deal_conservation():
    for players in (2, 4):
        state = deal_round(buraco(players), random.Random(123))
        assert dealt_multiset(state) == Counter(build_deck(2, 0))


def test_seeded_reproducibility():
    a = deal_round(buraco(2), random.Random(42))
    b = deal_round(buraco(2), random.Random(42))
    assert state_hash(a) == state_hash(b)
    assert a.hands == b.hands and a.stock == b.stock and a.morto == b.morto

    c = deal_round(buraco(2), random.Random(43))
    assert state_hash(c) != state_hash(a)


def test_first_player_offset():
    state = deal_round(buraco(2), random.Random(1), first_player=1)
    assert state.current_player == 1


def test_round_state_json_round_trip():
    cfg = buraco(4)
    state = deal_round(cfg, random.Random(99))
    restored = round_from_dict(cfg, round_to_dict(state))
    assert state_hash(restored) == state_hash(state)
    assert restored.hands == state.hands
    assert restored.morto == state.morto


def test_hash_sensitive_to_state():
    state = deal_round(buraco(2), random.Random(5))
    before = state_hash(state)
    state.trash.append(state.stock.pop())
    assert state_hash(state) != before


def test_morto_count_must_match_sides():
    cfg = buraco(2)
    bad = replace(cfg, morto=replace(cfg.morto, count=1))
    with pytest.raises(ValueError, match="morto.count"):
        deal_round(bad, random.Random(0))
