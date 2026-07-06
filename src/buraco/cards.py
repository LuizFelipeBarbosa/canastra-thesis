"""Card, rank/suit enums, and card-type id space (SPEC 02 §2.1–2.2).

A card is identified only by type, never by physical instance. Ids are
suit-major: ``ct = suit * 13 + rank``. Id 52 is the joker (all printed jokers
collapse into one fungible type); id 53 is a pad sentinel used only in ordered
encodings and never held, melded, or discarded.
"""

from __future__ import annotations

from enum import IntEnum

CardId = int


class Suit(IntEnum):
    CLUBS = 0
    DIAMONDS = 1
    HEARTS = 2
    SPADES = 3


class Rank(IntEnum):
    ACE = 0
    TWO = 1
    THREE = 2
    FOUR = 3
    FIVE = 4
    SIX = 5
    SEVEN = 6
    EIGHT = 7
    NINE = 8
    TEN = 9
    JACK = 10
    QUEEN = 11
    KING = 12


JOKER: CardId = 52
PAD: CardId = 53
NUM_CARD_TYPES = 53  # 0..52; PAD is not a card type
CARD_SPACE = 54  # width of count vectors / per-card action dimensions

RED_SUITS = frozenset({Suit.DIAMONDS, Suit.HEARTS})

RANK_NAMES = ("A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K")
SUIT_SYMBOLS = ("♣", "♦", "♥", "♠")

# Sequence position model: 14 positions per suit, 1 = ace-low, 2..13 = ranks
# TWO..KING, 14 = ace-high. No-wrap is automatic (SPEC 02 §2.2).
POS_MIN = 1
POS_MAX = 14


def card_id(rank: Rank, suit: Suit) -> CardId:
    return suit * 13 + rank


def id_rank(ct: CardId) -> Rank | None:
    return None if ct == JOKER else Rank(ct % 13)


def id_suit(ct: CardId) -> Suit | None:
    return None if ct == JOKER else Suit(ct // 13)


def is_joker(ct: CardId) -> bool:
    return ct == JOKER


def is_red(ct: CardId) -> bool:
    return ct != JOKER and Suit(ct // 13) in RED_SUITS


def rank_at(pos: int) -> Rank:
    """Rank occupying sequence position ``pos`` (1..14)."""
    if not POS_MIN <= pos <= POS_MAX:
        raise ValueError(f"position out of range: {pos}")
    return Rank.ACE if pos in (POS_MIN, POS_MAX) else Rank(pos - 1)


def nat(pos: int, suit: Suit) -> CardId:
    """Natural card type at sequence position ``pos`` in ``suit``."""
    return card_id(rank_at(pos), suit)


def positions_of(rank: Rank) -> tuple[int, ...]:
    """Sequence positions a rank may occupy (aces occupy two)."""
    return (POS_MIN, POS_MAX) if rank == Rank.ACE else (rank + 1,)


def rank_name(ct: CardId) -> str:
    return "JOKER" if ct == JOKER else RANK_NAMES[ct % 13]


def card_str(ct: CardId) -> str:
    """Human-readable card name, e.g. ``A♣`` or ``Joker``."""
    if ct == JOKER:
        return "Joker"
    if ct == PAD:
        return "<pad>"
    return f"{RANK_NAMES[ct % 13]}{SUIT_SYMBOLS[ct // 13]}"


def build_deck(deck_count: int, printed_jokers: int) -> list[CardId]:
    """Canonically ordered (unshuffled) full deck as card-type ids."""
    deck = [ct for _ in range(deck_count) for ct in range(52)]
    deck.extend([JOKER] * printed_jokers)
    return deck
