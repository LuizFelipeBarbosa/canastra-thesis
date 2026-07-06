"""Card-id space and sequence-position model (SPEC 02 §2.1–2.2)."""

from collections import Counter

from buraco.cards import (
    JOKER,
    NUM_CARD_TYPES,
    PAD,
    Rank,
    Suit,
    build_deck,
    card_id,
    card_str,
    id_rank,
    id_suit,
    is_red,
    nat,
    positions_of,
    rank_at,
)


def test_id_round_trip_all_types():
    for suit in Suit:
        for rank in Rank:
            ct = card_id(rank, suit)
            assert 0 <= ct < 52
            assert id_rank(ct) == rank
            assert id_suit(ct) == suit
    assert id_rank(JOKER) is None
    assert id_suit(JOKER) is None


def test_suit_major_layout():
    assert card_id(Rank.ACE, Suit.CLUBS) == 0
    assert card_id(Rank.KING, Suit.CLUBS) == 12
    assert card_id(Rank.ACE, Suit.DIAMONDS) == 13
    assert card_id(Rank.TWO, Suit.HEARTS) == 27
    assert card_id(Rank.KING, Suit.SPADES) == 51
    assert JOKER == 52 and PAD == 53 and NUM_CARD_TYPES == 53


def test_sequence_positions():
    assert rank_at(1) == Rank.ACE
    assert rank_at(14) == Rank.ACE
    assert rank_at(2) == Rank.TWO
    assert rank_at(13) == Rank.KING
    assert nat(1, Suit.CLUBS) == 0
    assert nat(14, Suit.SPADES) == card_id(Rank.ACE, Suit.SPADES)
    assert nat(2, Suit.HEARTS) == card_id(Rank.TWO, Suit.HEARTS)
    assert positions_of(Rank.ACE) == (1, 14)
    assert positions_of(Rank.KING) == (13,)
    assert positions_of(Rank.TWO) == (2,)


def test_red_black():
    assert is_red(card_id(Rank.FIVE, Suit.HEARTS))
    assert is_red(card_id(Rank.THREE, Suit.DIAMONDS))
    assert not is_red(card_id(Rank.THREE, Suit.SPADES))
    assert not is_red(JOKER)


def test_build_deck_buraco_default():
    deck = build_deck(deck_count=2, printed_jokers=0)
    assert len(deck) == 104
    counts = Counter(deck)
    assert all(counts[ct] == 2 for ct in range(52))
    assert JOKER not in counts


def test_build_deck_with_jokers():
    deck = build_deck(deck_count=2, printed_jokers=4)
    assert len(deck) == 108
    assert Counter(deck)[JOKER] == 4


def test_card_str():
    assert card_str(0) == "A♣"
    assert card_str(card_id(Rank.TEN, Suit.HEARTS)) == "10♥"
    assert card_str(JOKER) == "Joker"
