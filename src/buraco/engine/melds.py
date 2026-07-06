"""Meld model, validation, wildcard resolution, canastra detection.

Implements SPEC 01 §4 (representation) and SPEC 02 §2.6–2.7 (creation and
extension semantics with canonical physical-card resolution).

A sequence meld stores its low sequence position (`start_pos`, 1 = ace-low,
14 = ace-high) and an ordered low→high slot list; slot ``i`` occupies position
``start_pos + i``, which makes the rank a WILD represents derivable rather
than stored.

The plan/apply split keeps legality checks pure: ``plan_*`` functions inspect
a hand and return a plan (or None when illegal) without mutating anything;
``create_*``/``apply_add`` execute a plan, mutating hand and meld. `legal.py`
enumerates actions by probing the planners.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

from buraco.cards import (
    JOKER,
    POS_MAX,
    POS_MIN,
    CardId,
    Rank,
    Suit,
    card_id,
    id_rank,
    nat,
    rank_at,
)
from buraco.config import (
    ACE_HIGH_ONLY,
    ACE_LOW_ONLY,
    WILD_TO_HAND,
    RulesConfig,
)

# Wild-source parameter values (SPEC 02 §2.3).
SEQ_WILD_NONE = 0
SEQ_WILD_JOKER = 1
SEQ_WILD_TWO_OF_SUIT = 2
SEQ_WILD_OFF_SUIT_TWO = 3

SET_WILD_NONE = 0
SET_WILD_JOKER = 1
SET_WILD_TWO = 2


class MeldError(Exception):
    """A meld operation violated the active rules."""


class MeldKind(IntEnum):
    SEQUENCE = 0
    SET = 1


class SlotRole(IntEnum):
    NATURAL = 0
    WILD = 1


@dataclass
class Slot:
    card: CardId
    role: SlotRole


@dataclass
class Meld:
    meld_id: int
    owner: int  # side id (team in 4p, player in 2p)
    kind: MeldKind
    suit: Suit | None = None  # SEQUENCE only
    rank: Rank | None = None  # SET only
    start_pos: int | None = None  # SEQUENCE only: position of slots[0]
    slots: list[Slot] = field(default_factory=list)

    # --- derived properties (never stored; SPEC 01 §4) ---

    @property
    def size(self) -> int:
        return len(self.slots)

    @property
    def wild_count(self) -> int:
        return sum(1 for s in self.slots if s.role is SlotRole.WILD)

    @property
    def is_clean(self) -> bool:
        """Limpa: contains no card acting as a wild."""
        return self.wild_count == 0

    @property
    def end_pos(self) -> int | None:
        """Sequence position of the last slot (None for sets)."""
        if self.start_pos is None:
            return None
        return self.start_pos + len(self.slots) - 1

    @property
    def wild_pos_index(self) -> int | None:
        """Slot index of the first wild, or None. Unique in Buraco
        (wildcard_limit_per_meld == 1); sets may hold more in Canasta."""
        for i, s in enumerate(self.slots):
            if s.role is SlotRole.WILD:
                return i
        return None

    def is_canastra(self, min_size: int) -> bool:
        return len(self.slots) >= min_size

    def represented_rank(self, slot_index: int) -> Rank:
        """Rank the slot stands for (its position rank in a sequence; the
        set's rank in a set)."""
        if self.kind is MeldKind.SEQUENCE:
            assert self.start_pos is not None
            return rank_at(self.start_pos + slot_index)
        assert self.rank is not None
        return self.rank

    def card_multiset(self) -> dict[CardId, int]:
        counts: dict[CardId, int] = {}
        for s in self.slots:
            counts[s.card] = counts.get(s.card, 0) + 1
        return counts


# --- position and role predicates --------------------------------------------


def pos_allowed(cfg: RulesConfig, pos: int) -> bool:
    """Ace policy: whether a sequence may occupy position ``pos``."""
    if pos < POS_MIN or pos > POS_MAX:
        return False
    policy = cfg.meld.ace_policy
    if pos == POS_MAX and policy == ACE_LOW_ONLY:
        return False
    if pos == POS_MIN and policy == ACE_HIGH_ONLY:
        return False
    return True


def is_natural_at(cfg: RulesConfig, ct: CardId, pos: int, suit: Suit) -> bool:
    """Whether ``ct`` fills sequence position ``pos`` in ``suit`` as a NATURAL.

    A wild-rank card is natural only in its own suit's own-rank position and
    only when the profile grants the natural-2 exception (SPEC 01 §4).
    """
    if ct != nat(pos, suit):
        return False
    if not cfg.is_wild_card(ct):
        return True
    return cfg.wildcard.natural_two_in_suit


# --- plans (pure legality + canonical resolution) -----------------------------


@dataclass(frozen=True)
class CreatePlan:
    consumed: tuple[CardId, ...]
    slots: tuple[tuple[CardId, SlotRole], ...]


def _hand_covers(hand: dict[CardId, int], consumed: tuple[CardId, ...]) -> bool:
    """Multiset check: the same card type may be needed more than once (e.g. a
    2 held once cannot serve as both the natural at position 2 and the wild)."""
    needed: dict[CardId, int] = {}
    for c in consumed:
        needed[c] = needed.get(c, 0) + 1
    return all(hand.get(c, 0) >= n for c, n in needed.items())


@dataclass(frozen=True)
class AddPlan:
    kind: str  # "extend" | "swap" | "set_append"
    role: SlotRole  # role of the added card's slot
    at_low: bool = False  # extend: which end; swap: where the freed wild goes
    wild_to_hand: bool = False  # swap under WILD_TO_HAND relocation policy


def _seq_wild_card(
    cfg: RulesConfig, hand: dict[CardId, int], suit: Suit, gap_pos: int, wild_choice: int
) -> CardId | None:
    """Canonical physical wild for a sequence gap, or None if unavailable."""
    if cfg.wildcard.wildcard_limit_per_meld < 1:
        return None
    if wild_choice == SEQ_WILD_JOKER:
        if cfg.wildcard.jokers_wild and hand.get(JOKER, 0) > 0:
            return JOKER
        return None
    if wild_choice == SEQ_WILD_TWO_OF_SUIT:
        ct = card_id(Rank.TWO, suit)
        if Rank.TWO in cfg.wildcard.wild_ranks and hand.get(ct, 0) > 0:
            # At its own natural position it would not act as a wild.
            if is_natural_at(cfg, ct, gap_pos, suit):
                return None
            return ct
        return None
    if wild_choice == SEQ_WILD_OFF_SUIT_TWO:
        if Rank.TWO not in cfg.wildcard.wild_ranks:
            return None
        for other in Suit:  # lowest suit index first (SPEC 02 §2.7)
            if other == suit:
                continue
            ct = card_id(Rank.TWO, other)
            if hand.get(ct, 0) > 0:
                return ct
        return None
    return None


def plan_sequence(
    cfg: RulesConfig,
    hand: dict[CardId, int],
    suit: Suit,
    start_pos: int,
    wild_choice: int,
) -> CreatePlan | None:
    """Minimum-size (3) sequence creation, SPEC 02 §2.6."""
    if not cfg.meld.allow_sequences or cfg.meld.min_meld_size > 3:
        return None
    positions = (start_pos, start_pos + 1, start_pos + 2)
    if start_pos < POS_MIN or positions[-1] > POS_MAX:
        return None
    if not all(pos_allowed(cfg, p) for p in positions):
        return None

    held = [p for p in positions if hand.get(nat(p, suit), 0) > 0
            and is_natural_at(cfg, nat(p, suit), p, suit)]

    if wild_choice == SEQ_WILD_NONE:
        if len(held) != 3:
            return None
        slots = tuple((nat(p, suit), SlotRole.NATURAL) for p in positions)
        return CreatePlan(consumed=tuple(c for c, _ in slots), slots=slots)

    if len(held) != 2:
        return None
    gap = next(p for p in positions if p not in held)
    wild_ct = _seq_wild_card(cfg, hand, suit, gap, wild_choice)
    if wild_ct is None:
        return None
    slots = tuple(
        (wild_ct, SlotRole.WILD) if p == gap else (nat(p, suit), SlotRole.NATURAL)
        for p in positions
    )
    consumed = tuple(c for c, _ in slots)
    if not _hand_covers(hand, consumed):
        return None
    return CreatePlan(consumed=consumed, slots=slots)


def plan_set(
    cfg: RulesConfig,
    hand: dict[CardId, int],
    rank: Rank,
    wild_choice: int,
    prefer: CardId | None = None,
) -> CreatePlan | None:
    """Minimum-size (3) same-rank set creation, SPEC 02 §2.6.

    ``prefer`` forces one specific card into the naturals (Canasta's forced
    pile-card meld must consume the taken top card itself, SPEC 06 G1 —
    canonical lowest-suit-first selection could otherwise leave it in hand).
    """
    if not cfg.meld.allow_sets or cfg.meld.min_meld_size > 3:
        return None
    if rank in cfg.wildcard.wild_ranks:
        return None  # a set of wilds is not a set (masked while the rank is wild)

    need_naturals = 3 if wild_choice == SET_WILD_NONE else 2
    if wild_choice != SET_WILD_NONE:
        if cfg.wildcard.wildcard_limit_per_meld < 1:
            return None
        if cfg.wildcard.min_naturals_per_meld > 2:
            return None

    naturals: list[CardId] = []
    if prefer is not None:
        if id_rank(prefer) != rank or hand.get(prefer, 0) < 1:
            return None
        naturals.append(prefer)
    for suit in Suit:  # canonical: lowest suit first, copies together
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
        wild_ct: CardId = JOKER
    else:  # SET_WILD_TWO
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
    if not _hand_covers(hand, consumed):
        return None
    return CreatePlan(consumed=consumed, slots=slots)


def plan_add(
    cfg: RulesConfig, hand: dict[CardId, int], meld: Meld, ct: CardId
) -> AddPlan | None:
    """Extension of an existing meld by one card, SPEC 02 §2.6–2.7."""
    if hand.get(ct, 0) <= 0:
        return None

    if meld.kind is MeldKind.SET:
        if id_rank(ct) == meld.rank and not cfg.is_wild_card(ct):
            return AddPlan(kind="set_append", role=SlotRole.NATURAL)
        if cfg.is_wild_card(ct) and meld.wild_count < cfg.wildcard.wildcard_limit_per_meld:
            return AddPlan(kind="set_append", role=SlotRole.WILD)
        return None

    assert meld.suit is not None and meld.start_pos is not None
    suit, st = meld.suit, meld.start_pos
    en = meld.end_pos
    assert en is not None
    low_open = pos_allowed(cfg, st - 1)
    high_open = pos_allowed(cfg, en + 1)

    # 1. Natural end extension (includes the natural-2 landing on position 2).
    if low_open and is_natural_at(cfg, ct, st - 1, suit):
        return AddPlan(kind="extend", role=SlotRole.NATURAL, at_low=True)
    if high_open and is_natural_at(cfg, ct, en + 1, suit):
        return AddPlan(kind="extend", role=SlotRole.NATURAL, at_low=False)

    # 2. Wild swap-and-relocate: ct is the natural at the wild's position.
    wi = meld.wild_pos_index
    if wi is not None and is_natural_at(cfg, ct, st + wi, suit):
        if cfg.wildcard.wild_relocation == WILD_TO_HAND:
            return AddPlan(kind="swap", role=SlotRole.NATURAL, wild_to_hand=True)
        if low_open:  # deterministic: low end first (D12)
            return AddPlan(kind="swap", role=SlotRole.NATURAL, at_low=True)
        if high_open:
            return AddPlan(kind="swap", role=SlotRole.NATURAL, at_low=False)
        return None  # meld spans the full run; no home for the freed wild

    # 3. Wild placement on an open end (low end first, D12).
    if cfg.is_wild_card(ct) and meld.wild_count < cfg.wildcard.wildcard_limit_per_meld:
        if low_open:
            return AddPlan(kind="extend", role=SlotRole.WILD, at_low=True)
        if high_open:
            return AddPlan(kind="extend", role=SlotRole.WILD, at_low=False)
    return None


# --- application (mutating) ---------------------------------------------------


def _consume(hand: dict[CardId, int], ct: CardId) -> None:
    n = hand.get(ct, 0)
    if n <= 0:
        raise MeldError(f"hand does not contain card type {ct}")
    if n == 1:
        del hand[ct]
    else:
        hand[ct] = n - 1


def create_sequence(
    cfg: RulesConfig,
    hand: dict[CardId, int],
    owner: int,
    meld_id: int,
    suit: Suit,
    start_pos: int,
    wild_choice: int,
) -> Meld:
    plan = plan_sequence(cfg, hand, suit, start_pos, wild_choice)
    if plan is None:
        raise MeldError(
            f"illegal sequence: suit={suit!r} start={start_pos} wild={wild_choice}"
        )
    for ct in plan.consumed:
        _consume(hand, ct)
    meld = Meld(
        meld_id=meld_id,
        owner=owner,
        kind=MeldKind.SEQUENCE,
        suit=suit,
        start_pos=start_pos,
        slots=[Slot(c, r) for c, r in plan.slots],
    )
    validate_meld(cfg, meld)
    return meld


def create_set(
    cfg: RulesConfig,
    hand: dict[CardId, int],
    owner: int,
    meld_id: int,
    rank: Rank,
    wild_choice: int,
    prefer: CardId | None = None,
) -> Meld:
    plan = plan_set(cfg, hand, rank, wild_choice, prefer=prefer)
    if plan is None:
        raise MeldError(f"illegal set: rank={rank!r} wild={wild_choice}")
    for ct in plan.consumed:
        _consume(hand, ct)
    meld = Meld(
        meld_id=meld_id,
        owner=owner,
        kind=MeldKind.SET,
        rank=rank,
        slots=[Slot(c, r) for c, r in plan.slots],
    )
    validate_meld(cfg, meld)
    return meld


def apply_add(cfg: RulesConfig, hand: dict[CardId, int], meld: Meld, ct: CardId) -> None:
    """Add one card from ``hand`` to ``meld`` per the canonical plan."""
    plan = plan_add(cfg, hand, meld, ct)
    if plan is None:
        raise MeldError(f"illegal add: card {ct} onto meld {meld.meld_id}")
    _consume(hand, ct)

    if plan.kind == "set_append":
        meld.slots.append(Slot(ct, plan.role))
    elif plan.kind == "extend":
        if plan.at_low:
            assert meld.start_pos is not None
            meld.slots.insert(0, Slot(ct, plan.role))
            meld.start_pos -= 1
        else:
            meld.slots.append(Slot(ct, plan.role))
    else:  # swap
        wi = meld.wild_pos_index
        assert wi is not None and meld.start_pos is not None and meld.suit is not None
        freed = meld.slots[wi].card
        meld.slots[wi] = Slot(ct, SlotRole.NATURAL)
        if plan.wild_to_hand:
            hand[freed] = hand.get(freed, 0) + 1
        elif plan.at_low:
            new_pos = meld.start_pos - 1
            role = (
                SlotRole.NATURAL
                if is_natural_at(cfg, freed, new_pos, meld.suit)
                else SlotRole.WILD
            )
            meld.slots.insert(0, Slot(freed, role))
            meld.start_pos = new_pos
        else:
            end = meld.end_pos
            assert end is not None
            role = (
                SlotRole.NATURAL
                if is_natural_at(cfg, freed, end + 1, meld.suit)
                else SlotRole.WILD
            )
            meld.slots.append(Slot(freed, role))
    validate_meld(cfg, meld)


# --- structural validation (invariant §8.5) -----------------------------------


def validate_meld(cfg: RulesConfig, meld: Meld) -> None:
    """Full structural check; raises MeldError on any violation."""
    if meld.size < cfg.meld.min_meld_size:
        raise MeldError(f"meld {meld.meld_id} below minimum size")
    if meld.wild_count > cfg.wildcard.wildcard_limit_per_meld:
        raise MeldError(f"meld {meld.meld_id} exceeds wildcard limit")

    if meld.kind is MeldKind.SEQUENCE:
        if not cfg.meld.allow_sequences:
            raise MeldError("sequences not allowed by profile")
        if meld.suit is None or meld.start_pos is None or meld.rank is not None:
            raise MeldError("malformed sequence meld")
        end = meld.end_pos
        assert end is not None
        if meld.start_pos < POS_MIN or end > POS_MAX:
            raise MeldError("sequence out of position range")
        if not (pos_allowed(cfg, meld.start_pos) and pos_allowed(cfg, end)):
            raise MeldError("sequence violates ace policy")
        for i, slot in enumerate(meld.slots):
            pos = meld.start_pos + i
            natural_here = is_natural_at(cfg, slot.card, pos, meld.suit)
            if slot.role is SlotRole.NATURAL and not natural_here:
                raise MeldError(f"slot {i} claims NATURAL but is not {nat(pos, meld.suit)}")
            if slot.role is SlotRole.WILD and (
                natural_here or not cfg.is_wild_card(slot.card)
            ):
                raise MeldError(f"slot {i} claims WILD illegitimately")
    else:
        if not cfg.meld.allow_sets:
            raise MeldError("sets not allowed by profile")
        if meld.rank is None or meld.suit is not None or meld.start_pos is not None:
            raise MeldError("malformed set meld")
        naturals = 0
        for i, slot in enumerate(meld.slots):
            if slot.role is SlotRole.NATURAL:
                if cfg.is_wild_card(slot.card) or id_rank(slot.card) != meld.rank:
                    raise MeldError(f"set slot {i} claims NATURAL illegitimately")
                naturals += 1
            elif not cfg.is_wild_card(slot.card):
                raise MeldError(f"set slot {i} claims WILD but card is not wild")
        if naturals < cfg.wildcard.min_naturals_per_meld:
            raise MeldError("set below minimum natural count")
