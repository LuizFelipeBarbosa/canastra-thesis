"""Shared builders for engine tests: hand-crafted round states and melds."""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Sequence

from buraco.cards import CardId, Rank, Suit, card_id
from buraco.config import RulesConfig
from buraco.engine.melds import Meld, MeldKind, Slot, SlotRole, validate_meld
from buraco.engine.state import Phase, RoundState
from buraco.profiles import buraco


def ct(rank: Rank, suit: Suit) -> CardId:
    return card_id(rank, suit)


def make_state(
    cfg: RulesConfig | None = None,
    hands: Sequence[Iterable[CardId]] = ((), ()),
    stock: Iterable[CardId] = (),
    trash: Iterable[CardId] = (),
    morto: Sequence[tuple[CardId, ...] | None] | None = None,
    morto_taken: Sequence[bool] | None = None,
    melds: Sequence[Meld] = (),
    phase: Phase = Phase.PLAY,
    current_player: int = 0,
    initial_meld_done: Sequence[bool] | None = None,
    frozen: bool = False,
) -> RoundState:
    """Hand-crafted round state. Defaults: 2p Buraco, mortos already taken
    (no morto interaction unless a test asks for it), PLAY phase, initial
    melds satisfied (Canasta staging only when a test opts in)."""
    cfg = cfg or buraco(2)
    num_sides = cfg.table.num_sides
    if morto is None:
        morto = [None] * num_sides
    if morto_taken is None:
        morto_taken = [m is None for m in morto]
    if initial_meld_done is None:
        initial_meld_done = [True] * num_sides
    thresholds = cfg.initial_meld.thresholds
    minimum = max((pts for floor, pts in thresholds if floor <= 0), default=0)
    return RoundState(
        cfg=cfg,
        hands=[Counter(h) for h in hands],
        stock=list(stock),
        trash=list(trash),
        melds=list(melds),
        morto=list(morto),
        morto_taken=list(morto_taken),
        current_player=current_player,
        phase=phase,
        frozen=frozen,
        initial_meld_done=list(initial_meld_done),
        red_threes=[[] for _ in range(num_sides)],
        opened_on_turn=[None] * num_sides,
        initial_meld_min=[minimum] * num_sides,
    )


def natural_run(
    cfg: RulesConfig,
    suit: Suit,
    start: int,
    length: int,
    owner: int = 0,
    meld_id: int = 0,
) -> Meld:
    """A valid all-natural sequence meld built directly (no hand plumbing)."""
    slots = [
        Slot(ct(Rank.ACE if pos in (1, 14) else Rank(pos - 1), suit), SlotRole.NATURAL)
        for pos in range(start, start + length)
    ]
    meld = Meld(meld_id, owner, MeldKind.SEQUENCE, suit=suit, start_pos=start, slots=slots)
    validate_meld(cfg, meld)
    return meld
