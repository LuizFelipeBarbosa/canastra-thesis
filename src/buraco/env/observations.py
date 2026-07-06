"""Per-player views and debug/perfect-info mode (SPEC 03).

`observe` builds a plain-Python, JSON-friendly view for one seat using ONLY
zones that seat is allowed to know: its own hand, all melds, the open trash,
public sizes/flags, and the public action history. Hidden information
(other hands' contents, stock order, morto contents) is unreachable by
construction — the function never reads those fields beyond their sizes.

Spec amendments (documented here, applied over SPEC 03 §3.1):
- `all_hand_sizes`: every seat's hand count is public at a real table and is
  essential signal (an opponent holding one card threatens bater).
- `round_score` uses public zones only (meld points + canastra bonuses +
  morto-untaken penalty). The terminal hand penalties depend on hidden hand
  values and would leak them mid-round.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from buraco.cards import JOKER, CardId
from buraco.engine.melds import Meld, MeldKind, SlotRole
from buraco.engine.scoring import canastra_bonus, meld_points
from buraco.engine.serialize import round_to_dict, state_hash
from buraco.engine.state import Phase, RoundState

WILD_SOURCE_NONE = 0
WILD_SOURCE_JOKER = 1
WILD_SOURCE_TWO = 2

HISTORY_FAMILIES = (
    "draw_deck",
    "draw_trash",
    "create_seq",
    "create_set",
    "add",
    "discard",
    "go_out",
)


@dataclass(frozen=True)
class HistoryItem:
    actor: int  # absolute seat
    family: str  # one of HISTORY_FAMILIES
    card: CardId | None  # add/discard only; None otherwise


def meld_view(meld: Meld) -> dict[str, Any]:
    """Public summary of one meld (melds are fully public)."""
    wild_index = meld.wild_pos_index
    if wild_index is None:
        wild_source = WILD_SOURCE_NONE
    elif meld.slots[wild_index].card == JOKER:
        wild_source = WILD_SOURCE_JOKER
    else:
        wild_source = WILD_SOURCE_TWO
    return {
        "kind": int(meld.kind),
        "suit": None if meld.suit is None else int(meld.suit),
        "rank": None if meld.rank is None else int(meld.rank),
        "start_pos": meld.start_pos,
        "size": meld.size,
        "naturals": sum(1 for s in meld.slots if s.role is SlotRole.NATURAL),
        "wild_count": meld.wild_count,
        "wild_source": wild_source,
        "wild_index": wild_index,
        "is_sequence": meld.kind is MeldKind.SEQUENCE,
    }


def public_side_scores(state: RoundState) -> list[int]:
    """Live per-side score from public zones only (no hidden hand values)."""
    cfg = state.cfg
    scores = [0] * cfg.table.num_sides
    for meld in state.melds:
        scores[meld.owner] += meld_points(cfg, meld) + canastra_bonus(cfg, meld)
    if cfg.morto.count:
        for side in range(cfg.table.num_sides):
            if side < len(state.morto_taken) and not state.morto_taken[side]:
                scores[side] -= cfg.morto.untaken_penalty
    if state.went_out_side is not None:
        scores[state.went_out_side] += cfg.going_out.go_out_bonus
    return scores


def observe(
    state: RoundState,
    seat: int,
    match_scores: list[int],
    history: list[HistoryItem],
    melds_this_turn: int,
) -> dict[str, Any]:
    """Hidden-information observation for ``seat`` (SPEC 03 §3.1, amended)."""
    cfg = state.cfg
    side = cfg.table.side(seat)
    opp = 1 - side
    num_players = cfg.table.num_players
    live = public_side_scores(state)

    return {
        "seat": seat,
        "side": side,
        "num_players": num_players,
        "partner_seat": (seat + 2) % 4 if num_players == 4 else None,
        "hand": dict(state.hands[seat]),
        "all_hand_sizes": [state.hand_size(p) for p in range(num_players)],
        "own_melds": [meld_view(m) for m in state.side_melds(side)],
        "opp_melds": [meld_view(m) for m in state.side_melds(opp)],
        "trash": list(state.trash),
        "deck_size": len(state.stock),
        "morto_taken": [state.morto_taken[side], state.morto_taken[opp]]
        if state.morto_taken
        else [True, True],
        "mortos_remaining": sum(1 for m in state.morto if m is not None),
        "round_score": [live[side], live[opp]],
        "match_score": [match_scores[side], match_scores[opp]],
        "phase": int(state.phase),
        "has_drawn": int(state.phase is Phase.PLAY),
        "turn_number": state.turn_number,
        "melds_this_turn": melds_this_turn,
        # Canasta-mode public state (zeros/None for other profiles)
        "red_threes": [
            len(state.red_threes[side]) if state.red_threes else 0,
            len(state.red_threes[opp]) if state.red_threes else 0,
        ],
        "pile_frozen": int(state.frozen),
        "pile_blocked": int(state.pile_blocked_for_next),
        "initial_meld_done": [
            int(state.initial_meld_done[side]),
            int(state.initial_meld_done[opp]),
        ],
        "pending_pile_card": state.pending_pile_card,
        "staged_points": state.staged_points,
        "initial_meld_min": state.initial_meld_min[side] if state.initial_meld_min else 0,
        "history": [
            {
                "actor_rel": (item.actor - seat) % num_players,
                "family": item.family,
                "card": item.card,
            }
            for item in history
        ],
    }


def debug_view(state: RoundState) -> dict[str, Any]:
    """Perfect-information ground truth. Only ever placed in ``info["debug"]``
    when the env runs with ``perfect_info=True`` (SPEC 03 §3.5)."""
    return {
        "all_hands": [dict(h) for h in state.hands],
        "deck_ordered": list(state.stock),
        "morto_contents": [None if m is None else list(m) for m in state.morto],
        "state": round_to_dict(state),
        "ground_truth_hash": state_hash(state),
    }
