"""Action types and integer-id encode/decode (SPEC 02 §2.3–2.4).

Fixed layout, parametrized only by ``S = max_meld_slots`` (frozen per training
run). With the default ``S = 24`` the space is ``A = 1585`` ids:

| family          | base        | size        |
|-----------------|-------------|-------------|
| DRAW_DECK       | 0           | 1           |
| DRAW_TRASH      | 1           | 1           |
| CREATE_SEQUENCE | 2           | 4·12·4 =192 |
| CREATE_SET      | 194         | 13·3 = 39   |
| ADD_TO_MELD     | 233         | S·54        |
| DISCARD         | 233 + S·54  | 54          |
| GO_OUT          | +54         | 1           |
| END_ROUND       | +55         | 1           |

The numpy action mask lives in the env layer (`env/encoding.py`); this module
is pure Python by design.
"""

from __future__ import annotations

from dataclasses import dataclass

from buraco.cards import CARD_SPACE, CardId, Rank, Suit

# Wild-source dimensions (SPEC 02 §2.3); values re-exported by melds.py.
NUM_SEQ_WILD = 4  # none / joker / two-of-suit / off-suit-two
NUM_SET_WILD = 3  # none / joker / two
NUM_SEQ_SHAPES = 12  # start positions 1..12 for length-3 creation

BASE_SEQ = 2
BASE_SET = BASE_SEQ + 4 * NUM_SEQ_SHAPES * NUM_SEQ_WILD  # 194
BASE_ADD = BASE_SET + 13 * NUM_SET_WILD  # 233


@dataclass(frozen=True)
class DrawDeck:
    pass


@dataclass(frozen=True)
class DrawTrash:
    pass


@dataclass(frozen=True)
class CreateSeq:
    suit: Suit
    start: int  # sequence position of the low card, 1..12
    wild: int  # SEQ_WILD_* constant


@dataclass(frozen=True)
class CreateSet:
    rank: Rank
    wild: int  # SET_WILD_* constant


@dataclass(frozen=True)
class Add:
    slot: int  # index into the acting side's melds, creation order
    ct: CardId


@dataclass(frozen=True)
class Discard:
    ct: CardId


@dataclass(frozen=True)
class GoOut:
    pass


@dataclass(frozen=True)
class EndRound:
    pass


Action = DrawDeck | DrawTrash | CreateSeq | CreateSet | Add | Discard | GoOut | EndRound


def base_discard(max_meld_slots: int) -> int:
    return BASE_ADD + max_meld_slots * CARD_SPACE


def action_space_size(max_meld_slots: int) -> int:
    return base_discard(max_meld_slots) + CARD_SPACE + 2


def encode(action: Action, max_meld_slots: int) -> int:
    """Structured action → stable integer id. Asserts operands in range, so no
    illegal move is constructible as an id (SPEC 02 §2.4)."""
    match action:
        case DrawDeck():
            return 0
        case DrawTrash():
            return 1
        case CreateSeq(suit=s, start=st, wild=w):
            assert 1 <= st <= NUM_SEQ_SHAPES and 0 <= w < NUM_SEQ_WILD
            return BASE_SEQ + (int(s) * NUM_SEQ_SHAPES + (st - 1)) * NUM_SEQ_WILD + w
        case CreateSet(rank=r, wild=w):
            assert 0 <= w < NUM_SET_WILD
            return BASE_SET + int(r) * NUM_SET_WILD + w
        case Add(slot=slot, ct=ct):
            assert 0 <= slot < max_meld_slots and 0 <= ct < CARD_SPACE
            return BASE_ADD + slot * CARD_SPACE + ct
        case Discard(ct=ct):
            assert 0 <= ct < CARD_SPACE
            return base_discard(max_meld_slots) + ct
        case GoOut():
            return base_discard(max_meld_slots) + CARD_SPACE
        case EndRound():
            return base_discard(max_meld_slots) + CARD_SPACE + 1
    raise ValueError(f"unknown action: {action!r}")


def decode(a: int, max_meld_slots: int) -> Action:
    """Integer id → structured action (total inverse of `encode`)."""
    if a < 0 or a >= action_space_size(max_meld_slots):
        raise ValueError(f"action id out of range: {a}")
    if a == 0:
        return DrawDeck()
    if a == 1:
        return DrawTrash()
    if a < BASE_SET:
        x = a - BASE_SEQ
        w = x % NUM_SEQ_WILD
        x //= NUM_SEQ_WILD
        return CreateSeq(suit=Suit(x // NUM_SEQ_SHAPES), start=x % NUM_SEQ_SHAPES + 1, wild=w)
    if a < BASE_ADD:
        x = a - BASE_SET
        return CreateSet(rank=Rank(x // NUM_SET_WILD), wild=x % NUM_SET_WILD)
    disc = base_discard(max_meld_slots)
    if a < disc:
        x = a - BASE_ADD
        return Add(slot=x // CARD_SPACE, ct=x % CARD_SPACE)
    if a < disc + CARD_SPACE:
        return Discard(ct=a - disc)
    if a == disc + CARD_SPACE:
        return GoOut()
    return EndRound()
