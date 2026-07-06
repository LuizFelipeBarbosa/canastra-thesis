"""Human-readable labels for structured actions (shared by CLI and web UI)."""

from __future__ import annotations

from buraco.cards import card_str
from buraco.engine.actions import (
    Action,
    Add,
    CreateSeq,
    CreateSet,
    Discard,
    DrawDeck,
    DrawTrash,
    EndRound,
    GoOut,
)


def describe(action: Action, slots: int) -> str:
    match action:
        case DrawDeck():
            return "draw from stock"
        case DrawTrash():
            return "take the trash pile"
        case CreateSeq(suit=s, start=st, wild=w):
            kinds = ["", " +joker", " +2-of-suit", " +off-suit 2"]
            return f"meld sequence suit={s.name} positions {st}-{st + 2}{kinds[w]}"
        case CreateSet(rank=r, wild=w):
            kinds = ["", " +joker", " +wild 2"]
            return f"meld set of {r.name}{kinds[w]}"
        case Add(slot=slot, ct=ct):
            return f"add {card_str(ct)} to meld #{slot}"
        case Discard(ct=ct):
            return f"discard {card_str(ct)}"
        case GoOut():
            return "go out (bater)"
        case EndRound():
            return "end the round (stock exhausted)"
    return str(action)
