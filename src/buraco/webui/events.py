"""Per-step event records and round summaries for the GUI.

The engine has no event system, and one `env.step` can trigger implicit
transitions (morto pickup, morto→stock conversion, red-3 auto-resolve,
pile freeze, round end). Each event is therefore built as a pre/post diff
of *public* zones around the step; the frontend animates from the event,
then hard-reconciles against the authoritative state view sent alongside.

Hidden-info discipline: card identities appear only for public zones
(trash, melds, red-three trays) and for the human's own hand deltas. Bot
deck draws are counts only.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from buraco.cards import CardId
from buraco.config import (
    HAND_PENALTY_OPPONENT_POSITIVE,
    HAND_PENALTY_SELF_NEGATIVE,
    RED3_BONUS_AUTOREPLACE,
    RulesConfig,
)
from buraco.describe import describe
from buraco.engine.actions import Action, Add, CreateSeq, CreateSet, Discard, DrawDeck, DrawTrash
from buraco.engine.scoring import hand_points, round_scores
from buraco.engine.state import EndReason, Phase, RoundState
from buraco.webui.views import family_of, meld_snapshot


def snapshot_public(state: RoundState) -> dict[str, Any]:
    """Copy of every public zone/flag; the diff of two of these is the event."""
    cfg = state.cfg
    return {
        "deck_size": len(state.stock),
        "trash": [int(c) for c in state.trash],
        "hand_sizes": [state.hand_size(p) for p in range(cfg.table.num_players)],
        "morto_taken": list(state.morto_taken),
        "mortos_remaining": sum(1 for m in state.morto if m is not None),
        "frozen": state.frozen,
        "pile_blocked": state.pile_blocked_for_next,
        "pending_pile_card": state.pending_pile_card,
        "staged_points": state.staged_points,
        "red_threes": [list(t) for t in state.red_threes],
        "melds_per_side": [len(state.side_melds(s)) for s in range(cfg.table.num_sides)],
        "phase": int(state.phase),
        "current_player": state.current_player,
        "turn_number": state.turn_number,
    }


def _diff(pre: dict[str, Any], post: dict[str, Any]) -> dict[str, Any]:
    remaining_delta = post["mortos_remaining"] - pre["mortos_remaining"]
    taken_flipped = [
        i for i, (a, b) in enumerate(zip(pre["morto_taken"], post["morto_taken"])) if a != b
    ]
    return {
        "deck_size": post["deck_size"] - pre["deck_size"],
        "hand_sizes": [b - a for a, b in zip(pre["hand_sizes"], post["hand_sizes"])],
        "trash_len": len(post["trash"]) - len(pre["trash"]),
        "morto_taken_by": taken_flipped,
        "morto_converted_to_stock": remaining_delta < 0 and not taken_flipped,
        "frozen": None if pre["frozen"] == post["frozen"] else post["frozen"],
        "pile_blocked": (
            None if pre["pile_blocked"] == post["pile_blocked"] else post["pile_blocked"]
        ),
        "red_threes_added": [
            [int(c) for c in b[len(a):]]
            for a, b in zip(pre["red_threes"], post["red_threes"])
        ],
        "pending_pile_card": post["pending_pile_card"],
        "pending_pile_changed": pre["pending_pile_card"] != post["pending_pile_card"],
        "staged_points": post["staged_points"] - pre["staged_points"],
    }


def build_event(
    *,
    seq: int,
    actor: int,
    action: Action,
    action_id: int,
    cfg: RulesConfig,
    human_seat: int,
    pre: dict[str, Any],
    post: dict[str, Any],
    pre_human_hand: Counter[CardId],
    post_human_hand: Counter[CardId],
    state_after: RoundState,
    terminated: bool,
    truncated: bool,
    new_round: bool,
    round_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    """One animatable event for the micro-action just applied.

    ``state_after`` is the round the action was applied to (still the terminal
    round when a match rolls into the next deal inside the same step).
    """
    slots = cfg.meld.max_meld_slots
    diff = _diff(pre, post)
    actor_side = cfg.table.side(actor)

    card: int | None = None
    slot: int | None = None
    cards_taken: list[int] | None = None
    meld: dict[str, Any] | None = None
    meld_index: int | None = None
    match action:
        case Add(slot=sl, ct=c):
            card, slot, meld_index = int(c), sl, sl
        case Discard(ct=c):
            card = int(c)
        case DrawTrash():
            # Draws take from the top, so what remains is a prefix of what was.
            cards_taken = pre["trash"][len(post["trash"]):]
        case CreateSeq() | CreateSet():
            meld_index = len(state_after.side_melds(actor_side)) - 1
    if meld_index is not None:
        side_melds = state_after.side_melds(actor_side)
        if 0 <= meld_index < len(side_melds):
            meld = meld_snapshot(side_melds[meld_index], cfg)

    hand_added = sorted(int(c) for c in (post_human_hand - pre_human_hand).elements())
    hand_removed = sorted(int(c) for c in (pre_human_hand - post_human_hand).elements())
    drawn_hidden = 0
    if actor != human_seat and isinstance(action, DrawDeck):
        drawn_hidden = -diff["deck_size"]

    return {
        "seq": seq,
        "actor": actor,
        "actor_side": actor_side,
        "actor_is_human": actor == human_seat,
        "family": family_of(action),
        "action_id": int(action_id),
        "label": describe(action, slots),
        "card": card,
        "slot": slot,
        "cards_taken": cards_taken,
        "drawn_hidden": drawn_hidden,
        "hand_added": hand_added,
        "hand_removed": hand_removed,
        "meld_side": actor_side if meld is not None else None,
        "meld_index": meld_index if meld is not None else None,
        "meld": meld,
        "diff": diff,
        "phase_after": Phase(state_after.phase).name,
        "to_play_after": state_after.current_player,
        "turn_number_after": state_after.turn_number,
        "round_ended": state_after.round_over,
        "new_round": new_round,
        "terminated": terminated,
        "truncated": truncated,
        "round_summary": round_summary,
    }


def build_round_summary(
    state: RoundState, match_scores_after: list[int], match_over: bool
) -> dict[str, Any]:
    """Itemized end-of-round breakdown from the terminal round state.

    Mirrors `round_scores` term by term; remaining hands are revealed here
    (the showdown is the one moment hidden cards become public). The itemized
    totals are asserted equal to the engine's authoritative scores.
    """
    cfg = state.cfg
    expected = round_scores(state)
    num_players = cfg.table.num_players
    per_side: list[dict[str, Any]] = []
    for side in range(cfg.table.num_sides):
        melds = [meld_snapshot(m, cfg) for m in state.side_melds(side)]
        meld_pts = sum(m["points"] for m in melds)
        can_bonus = sum(m["bonus"] for m in melds)

        go_out = concealed = 0
        if state.went_out_side == side:
            go_out = cfg.going_out.go_out_bonus
            if (
                cfg.going_out.concealed_bonus
                and state.opened_on_turn
                and state.opened_on_turn[side] == state.turn_number
            ):
                concealed = cfg.going_out.concealed_bonus

        red3 = 0
        if cfg.special_threes.red_three_mode == RED3_BONUS_AUTOREPLACE:
            tray = state.red_threes[side]
            if tray:
                bonus = len(tray) * cfg.special_threes.red_three_bonus
                if len(tray) == 4:
                    bonus += cfg.special_threes.red_three_all_bonus
                opened = any(m.owner == side for m in state.melds)
                negative = cfg.special_threes.red_three_negative_if_no_meld and not opened
                red3 = -bonus if negative else bonus

        morto_pen = 0
        if cfg.morto.count and side < len(state.morto_taken) and not state.morto_taken[side]:
            morto_pen = cfg.morto.untaken_penalty

        hands = [
            {
                "seat": p,
                "cards": sorted(int(c) for c in state.hands[p].elements()),
                "points": hand_points(cfg, state, p),
            }
            for p in range(num_players)
            if cfg.table.side(p) == side
        ]
        mode = cfg.scoring.hand_penalty_mode
        hand_pen = opp_gain = 0
        if mode == HAND_PENALTY_SELF_NEGATIVE:
            hand_pen = sum(h["points"] for h in hands)
        elif mode == HAND_PENALTY_OPPONENT_POSITIVE and state.went_out_side == side:
            opp_gain = sum(
                hand_points(cfg, state, p)
                for p in range(num_players)
                if cfg.table.side(p) != side
            )

        total = meld_pts + can_bonus + go_out + concealed + red3 - morto_pen - hand_pen + opp_gain
        assert total == expected[side], (
            f"round summary drifted from round_scores for side {side}: "
            f"{total} != {expected[side]}"
        )
        per_side.append(
            {
                "side": side,
                "melds": melds,
                "meld_points": meld_pts,
                "canastra_bonus": can_bonus,
                "go_out_bonus": go_out,
                "concealed_bonus": concealed,
                "red_three_bonus": red3,
                "morto_penalty": morto_pen,
                "hands": hands,
                "hand_penalty": hand_pen,
                "opponent_hand_gain": opp_gain,
                "total": total,
            }
        )

    return {
        "end_reason": (
            EndReason(state.end_reason).name if state.end_reason is not None else "TRUNCATED"
        ),
        "went_out_side": state.went_out_side,
        "turn_number": state.turn_number,
        "per_side": per_side,
        "totals": expected,
        "match_scores_after": list(match_scores_after),
        "match_over": match_over,
    }
