"""RulesConfig tree, Buraco profile defaults, and config serialization."""

import pytest

from buraco.cards import JOKER, Rank, Suit, card_id
from buraco.config import (
    DRAW_WHOLE_PILE,
    MODE_INDIVIDUAL,
    MODE_TEAMS,
    VISIBILITY_FULL_OPEN,
)
from buraco.engine.serialize import config_from_dict, config_to_dict
from buraco.profiles import buraco, load_profile


def test_buraco_defaults_match_spec():
    cfg = buraco()
    assert cfg.name == "buraco"
    assert cfg.deck.total_cards == 104 and cfg.deck.printed_jokers == 0
    assert cfg.table.cards_per_player == 11
    assert cfg.morto.count == 2 and cfg.morto.size == 11
    assert cfg.morto.untaken_penalty == 100
    assert cfg.meld.canastra_min_size == 7
    assert cfg.meld.canastra_bonus_clean == 200
    assert cfg.meld.canastra_bonus_dirty == 100
    assert cfg.wildcard.wild_ranks == frozenset({Rank.TWO})
    assert cfg.wildcard.wildcard_limit_per_meld == 1
    assert cfg.wildcard.natural_two_in_suit is True
    assert cfg.discard_pile.visibility == VISIBILITY_FULL_OPEN
    assert cfg.discard_pile.draw_rule == DRAW_WHOLE_PILE
    assert cfg.going_out.require_canastra is True
    assert cfg.going_out.require_morto_taken is True
    assert cfg.going_out.go_out_bonus == 100
    assert cfg.scoring.match_target == 3000
    # D1: Brazilian card points
    points = cfg.scoring.card_points
    assert points["A"] == 15 and points["2"] == 10 and points["JOKER"] == 20
    assert points["3"] == 5 and points["7"] == 5 and points["8"] == 10 and points["K"] == 10


def test_buraco_player_modes():
    two = buraco(2)
    assert two.table.mode == MODE_INDIVIDUAL and two.table.num_sides == 2
    assert [two.table.side(p) for p in range(2)] == [0, 1]

    four = buraco(4)
    assert four.table.mode == MODE_TEAMS and four.table.num_sides == 2
    # partners sit opposite: seats 0,2 = side 0; seats 1,3 = side 1
    assert [four.table.side(p) for p in range(4)] == [0, 1, 0, 1]

    with pytest.raises(ValueError):
        buraco(3)


def test_load_profile():
    assert load_profile("buraco", num_players=4).table.num_players == 4
    with pytest.raises(ValueError, match="unknown profile"):
        load_profile("skat")


def test_wildcard_predicate():
    cfg = buraco()
    assert cfg.is_wild_card(card_id(Rank.TWO, Suit.CLUBS))
    assert cfg.is_wild_card(card_id(Rank.TWO, Suit.SPADES))
    assert cfg.is_wild_card(JOKER)
    assert not cfg.is_wild_card(card_id(Rank.THREE, Suit.CLUBS))
    assert not cfg.is_wild_card(card_id(Rank.ACE, Suit.HEARTS))


def test_card_value():
    cfg = buraco()
    assert cfg.card_value(card_id(Rank.ACE, Suit.SPADES)) == 15
    assert cfg.card_value(card_id(Rank.TWO, Suit.HEARTS)) == 10
    assert cfg.card_value(card_id(Rank.SEVEN, Suit.CLUBS)) == 5
    assert cfg.card_value(card_id(Rank.QUEEN, Suit.DIAMONDS)) == 10
    assert cfg.card_value(JOKER) == 20


def test_config_json_round_trip():
    for players in (2, 4):
        cfg = buraco(players)
        d = config_to_dict(cfg)
        assert config_from_dict(d) == cfg
