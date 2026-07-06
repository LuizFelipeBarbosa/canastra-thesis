"""Card points, canastra bonuses, round and match scoring (SPEC 04 scoring).

Every card scores its own face value regardless of what a wild represents
(SPEC 01 §4). Round scores are per side; match accumulation lives in the env.
"""

from __future__ import annotations

from buraco.config import (
    HAND_PENALTY_OPPONENT_POSITIVE,
    HAND_PENALTY_SELF_NEGATIVE,
    RED3_BONUS_AUTOREPLACE,
    RulesConfig,
)
from buraco.engine.melds import Meld
from buraco.engine.state import RoundState


def meld_points(cfg: RulesConfig, meld: Meld) -> int:
    return sum(cfg.card_value(slot.card) for slot in meld.slots)


def canastra_bonus(cfg: RulesConfig, meld: Meld) -> int:
    if not meld.is_canastra(cfg.meld.canastra_min_size):
        return 0
    return cfg.meld.canastra_bonus_clean if meld.is_clean else cfg.meld.canastra_bonus_dirty


def hand_points(cfg: RulesConfig, state: RoundState, player: int) -> int:
    return sum(cfg.card_value(ct) * n for ct, n in state.hands[player].items())


def round_scores(state: RoundState) -> list[int]:
    """Final (or would-be) round score per side. Meaningful at TERMINAL; on a
    truncated episode callers may still use it as the current differential."""
    cfg = state.cfg
    num_sides = cfg.table.num_sides
    scores = [0] * num_sides

    for meld in state.melds:
        scores[meld.owner] += meld_points(cfg, meld) + canastra_bonus(cfg, meld)

    if state.went_out_side is not None:
        scores[state.went_out_side] += cfg.going_out.go_out_bonus
        if (
            cfg.going_out.concealed_bonus
            and state.opened_on_turn
            and state.opened_on_turn[state.went_out_side] == state.turn_number
        ):
            scores[state.went_out_side] += cfg.going_out.concealed_bonus

    if cfg.special_threes.red_three_mode == RED3_BONUS_AUTOREPLACE:
        for side, tray in enumerate(state.red_threes):
            if not tray:
                continue
            bonus = len(tray) * cfg.special_threes.red_three_bonus
            if len(tray) == 4:
                bonus += cfg.special_threes.red_three_all_bonus
            opened = any(m.owner == side for m in state.melds)
            if cfg.special_threes.red_three_negative_if_no_meld and not opened:
                scores[side] -= bonus
            else:
                scores[side] += bonus

    if cfg.morto.count:
        for side in range(num_sides):
            if side < len(state.morto_taken) and not state.morto_taken[side]:
                scores[side] -= cfg.morto.untaken_penalty

    mode = cfg.scoring.hand_penalty_mode
    if mode == HAND_PENALTY_SELF_NEGATIVE:
        for player in range(cfg.table.num_players):
            scores[cfg.table.side(player)] -= hand_points(cfg, state, player)
    elif mode == HAND_PENALTY_OPPONENT_POSITIVE:
        if state.went_out_side is not None:
            gained = sum(
                hand_points(cfg, state, p)
                for p in range(cfg.table.num_players)
                if cfg.table.side(p) != state.went_out_side
            )
            scores[state.went_out_side] += gained
    else:  # pragma: no cover
        raise ValueError(f"unknown hand penalty mode: {mode}")

    return scores
