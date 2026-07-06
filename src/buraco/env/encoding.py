"""Numpy observation vectors and action masks (SPEC 03 §3.1–3.3).

The only module (with env.py) that imports numpy; the engine stays pure
Python. All shapes are fixed for a given config: hands/counts are 54-wide,
meld blocks are (max_meld_slots, 31), history is (history_len, 65).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from buraco.cards import CARD_SPACE, PAD, POS_MAX
from buraco.config import RulesConfig
from buraco.engine.actions import encode
from buraco.engine.legal import legal_actions
from buraco.engine.state import Phase, RoundState
from buraco.env.observations import HISTORY_FAMILIES

MELD_SLOT_WIDTH = 31
HISTORY_ITEM_WIDTH = 4 + len(HISTORY_FAMILIES) + CARD_SPACE  # 65


def counts_vector(counts: dict[int, int]) -> np.ndarray:
    vec = np.zeros(CARD_SPACE, dtype=np.int8)
    for card, n in counts.items():
        vec[card] = n
    return vec


def multiset_vector(cards: list[int]) -> np.ndarray:
    vec = np.zeros(CARD_SPACE, dtype=np.int8)
    for card in cards:
        vec[card] += 1
    return vec


def meld_slot_vector(view: dict[str, Any]) -> np.ndarray:
    """SPEC 03 §3.2, F=31: [occupied, type(2), suit(4), rank(13), start, end,
    length, naturals, wild_count, wild_source(3), wild_pos, canastra, limpa]."""
    vec = np.zeros(MELD_SLOT_WIDTH, dtype=np.float32)
    i = 0
    vec[i] = 1.0  # occupied
    i += 1
    vec[i if view["is_sequence"] else i + 1] = 1.0  # type one-hot
    i += 2
    if view["is_sequence"]:
        vec[i + view["suit"]] = 1.0  # suit one-hot (4)
    i += 4
    if not view["is_sequence"]:
        vec[i + view["rank"]] = 1.0  # rank one-hot (13)
    i += 13
    if view["is_sequence"]:
        start = view["start_pos"]
        vec[i] = start / POS_MAX
        vec[i + 1] = (start + view["size"] - 1) / POS_MAX
    i += 2
    vec[i] = view["size"] / POS_MAX
    vec[i + 1] = view["naturals"] / POS_MAX
    vec[i + 2] = float(view["wild_count"])
    i += 3
    vec[i + view["wild_source"]] = 1.0  # wild_source one-hot (3)
    i += 3
    if view["is_sequence"] and view["wild_index"] is not None:
        vec[i] = (view["start_pos"] + view["wild_index"]) / POS_MAX
    i += 1
    vec[i] = float(view["is_canastra"])
    vec[i + 1] = float(view["is_limpa"])
    assert i + 2 == MELD_SLOT_WIDTH
    return vec


def meld_block(views: list[dict[str, Any]], cfg: RulesConfig) -> np.ndarray:
    block = np.zeros((cfg.meld.max_meld_slots, MELD_SLOT_WIDTH), dtype=np.float32)
    min_size = cfg.meld.canastra_min_size
    for slot, view in enumerate(views[: cfg.meld.max_meld_slots]):
        view = dict(view)
        view["is_canastra"] = view["size"] >= min_size
        view["is_limpa"] = view["wild_count"] == 0
        block[slot] = meld_slot_vector(view)
    return block


def history_block(items: list[dict[str, Any]], history_len: int) -> np.ndarray:
    """(H, 65): [actor_rel(4), family(7), card(54)], newest-last, zero-padded."""
    block = np.zeros((history_len, HISTORY_ITEM_WIDTH), dtype=np.float32)
    recent = items[-history_len:]
    offset = history_len - len(recent)
    for row, item in enumerate(recent):
        vec = block[offset + row]
        vec[item["actor_rel"]] = 1.0
        vec[4 + HISTORY_FAMILIES.index(item["family"])] = 1.0
        if item["card"] is not None:
            vec[4 + len(HISTORY_FAMILIES) + item["card"]] = 1.0
    return block


def encode_observation(
    raw: dict[str, Any],
    cfg: RulesConfig,
    history_len: int = 8,
    trash_top_k: int = 8,
) -> dict[str, np.ndarray]:
    """Raw per-seat view → fixed-shape numpy dict (SPEC 03 §3.1, amended)."""
    trash = raw["trash"]
    top_k = np.full(trash_top_k, PAD, dtype=np.int16)
    for i, card in enumerate(reversed(trash[-trash_top_k:])):
        top_k[i] = card  # newest first

    hand_sizes = np.zeros(4, dtype=np.int16)
    hand_sizes[: raw["num_players"]] = raw["all_hand_sizes"]

    seat_onehot = np.zeros(4, dtype=np.int8)
    seat_onehot[raw["seat"]] = 1
    partner_onehot = np.zeros(4, dtype=np.int8)
    if raw["partner_seat"] is not None:
        partner_onehot[raw["partner_seat"]] = 1

    match_target = max(cfg.scoring.match_target, 1)
    return {
        "hand": counts_vector(raw["hand"]),
        "hand_size": np.array([sum(raw["hand"].values())], dtype=np.int16),
        "all_hand_sizes": hand_sizes,
        "own_melds": meld_block(raw["own_melds"], cfg),
        "opp_melds": meld_block(raw["opp_melds"], cfg),
        "trash_counts": multiset_vector(trash),
        "trash_top_k": top_k,
        "trash_size": np.array([len(trash)], dtype=np.int16),
        "deck_size": np.array([raw["deck_size"]], dtype=np.int16),
        "morto_taken": np.array(raw["morto_taken"], dtype=np.int8),
        "mortos_remaining": np.array([raw["mortos_remaining"]], dtype=np.int8),
        "round_score": np.array(raw["round_score"], dtype=np.float32) / 1000.0,
        "match_score": np.array(raw["match_score"], dtype=np.float32) / match_target,
        "phase": np.array(
            [raw["phase"] == int(Phase.DRAW), raw["phase"] == int(Phase.PLAY)],
            dtype=np.int8,
        ),
        "has_drawn": np.array([raw["has_drawn"]], dtype=np.int8),
        "turn_norm": np.array(
            [raw["turn_number"] / max(cfg.turn.truncation_cap, 1)], dtype=np.float32
        ),
        "melds_this_turn": np.array([raw["melds_this_turn"]], dtype=np.int8),
        "seat_rel": seat_onehot,
        "partner_rel": partner_onehot,
        "is_4p": np.array([int(raw["num_players"] == 4)], dtype=np.int8),
        "history": history_block(raw["history"], history_len),
        # Canasta-mode fields (SPEC 06; zero-filled in other profiles)
        "red_threes": np.array(raw["red_threes"], dtype=np.int8),
        "pile_flags": np.array([raw["pile_frozen"], raw["pile_blocked"]], dtype=np.int8),
        "initial_meld_done": np.array(raw["initial_meld_done"], dtype=np.int8),
        "pending_pile_card": np.array(
            [PAD if raw["pending_pile_card"] is None else raw["pending_pile_card"]],
            dtype=np.int16,
        ),
        "staged": np.array(
            [raw["staged_points"] / 120.0, raw["initial_meld_min"] / 120.0],
            dtype=np.float32,
        ),
    }


def action_mask(state: RoundState) -> np.ndarray:
    """(A,) int8 mask over integer action ids; all-zero iff terminal."""
    slots = state.cfg.meld.max_meld_slots
    from buraco.engine.actions import action_space_size

    mask = np.zeros(action_space_size(slots), dtype=np.int8)
    for action in legal_actions(state):
        mask[encode(action, slots)] = 1
    return mask
