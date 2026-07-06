"""JSON state snapshots and replay from seed plus action log (SPEC 01 §8.8).

Everything here is a pure mapping between the in-memory model and plain JSON
types. `state_hash` is the canonical fingerprint used by determinism and
replay tests. Replay itself (re-stepping an action log) lands with the env in
milestone M6; the snapshot format is stable from M2 on.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict
from typing import Any

from buraco.cards import Rank, Suit
from buraco.config import (
    DeckConfig,
    DiscardPileConfig,
    GoingOutConfig,
    InitialMeldConfig,
    MeldConfig,
    MortoConfig,
    RulesConfig,
    ScoringConfig,
    SpecialThreesConfig,
    TableConfig,
    TurnConfig,
    WildcardConfig,
)
from buraco.engine.melds import Meld, MeldKind, Slot, SlotRole
from buraco.engine.state import EndReason, MatchState, Phase, RoundState

# --- config ------------------------------------------------------------------


def config_to_dict(cfg: RulesConfig) -> dict[str, Any]:
    d = asdict(cfg)
    d["wildcard"]["wild_ranks"] = sorted(int(r) for r in cfg.wildcard.wild_ranks)
    d["initial_meld"]["thresholds"] = [list(t) for t in cfg.initial_meld.thresholds]
    d["scoring"]["card_points"] = dict(cfg.scoring.card_points)
    return d


def config_from_dict(d: dict[str, Any]) -> RulesConfig:
    wc = dict(d["wildcard"])
    wc["wild_ranks"] = frozenset(Rank(r) for r in wc["wild_ranks"])
    im = dict(d["initial_meld"])
    im["thresholds"] = tuple(tuple(t) for t in im["thresholds"])
    return RulesConfig(
        name=d["name"],
        table=TableConfig(**d["table"]),
        deck=DeckConfig(**d["deck"]),
        wildcard=WildcardConfig(**wc),
        meld=MeldConfig(**d["meld"]),
        morto=MortoConfig(**d["morto"]),
        discard_pile=DiscardPileConfig(**d["discard_pile"]),
        going_out=GoingOutConfig(**d["going_out"]),
        initial_meld=InitialMeldConfig(**im),
        special_threes=SpecialThreesConfig(**d["special_threes"]),
        scoring=ScoringConfig(**d["scoring"]),
        turn=TurnConfig(**d["turn"]),
    )


# --- melds -------------------------------------------------------------------


def meld_to_dict(meld: Meld) -> dict[str, Any]:
    return {
        "meld_id": meld.meld_id,
        "owner": meld.owner,
        "kind": int(meld.kind),
        "suit": None if meld.suit is None else int(meld.suit),
        "rank": None if meld.rank is None else int(meld.rank),
        "start_pos": meld.start_pos,
        "slots": [[s.card, int(s.role)] for s in meld.slots],
    }


def meld_from_dict(d: dict[str, Any]) -> Meld:
    return Meld(
        meld_id=d["meld_id"],
        owner=d["owner"],
        kind=MeldKind(d["kind"]),
        suit=None if d["suit"] is None else Suit(d["suit"]),
        rank=None if d["rank"] is None else Rank(d["rank"]),
        start_pos=d["start_pos"],
        slots=[Slot(card=c, role=SlotRole(r)) for c, r in d["slots"]],
    )


# --- round state -------------------------------------------------------------


def round_to_dict(state: RoundState) -> dict[str, Any]:
    return {
        "hands": [sorted(h.items()) for h in state.hands],
        "stock": list(state.stock),
        "trash": list(state.trash),
        "melds": [meld_to_dict(m) for m in state.melds],
        "morto": [None if p is None else list(p) for p in state.morto],
        "morto_taken": list(state.morto_taken),
        "current_player": state.current_player,
        "phase": int(state.phase),
        "turn_number": state.turn_number,
        "just_drawn_from_pile": state.just_drawn_from_pile,
        "pile_blocked_for_next": state.pile_blocked_for_next,
        "frozen": state.frozen,
        "initial_meld_done": list(state.initial_meld_done),
        "round_over": state.round_over,
        "went_out_side": state.went_out_side,
        "end_reason": None if state.end_reason is None else int(state.end_reason),
        "red_threes": [list(t) for t in state.red_threes],
        "pending_pile_card": state.pending_pile_card,
        "pending_pile_pair_only": state.pending_pile_pair_only,
        "staged_points": state.staged_points,
        "opened_on_turn": list(state.opened_on_turn),
        "initial_meld_min": list(state.initial_meld_min),
    }


def round_from_dict(cfg: RulesConfig, d: dict[str, Any]) -> RoundState:
    return RoundState(
        cfg=cfg,
        hands=[Counter(dict((ct, n) for ct, n in h)) for h in d["hands"]],
        stock=list(d["stock"]),
        trash=list(d["trash"]),
        melds=[meld_from_dict(m) for m in d["melds"]],
        morto=[None if p is None else tuple(p) for p in d["morto"]],
        morto_taken=list(d["morto_taken"]),
        current_player=d["current_player"],
        phase=Phase(d["phase"]),
        turn_number=d["turn_number"],
        just_drawn_from_pile=d["just_drawn_from_pile"],
        pile_blocked_for_next=d["pile_blocked_for_next"],
        frozen=d["frozen"],
        initial_meld_done=list(d["initial_meld_done"]),
        round_over=d["round_over"],
        went_out_side=d["went_out_side"],
        end_reason=None if d["end_reason"] is None else EndReason(d["end_reason"]),
        red_threes=[list(t) for t in d["red_threes"]],
        pending_pile_card=d["pending_pile_card"],
        pending_pile_pair_only=d["pending_pile_pair_only"],
        staged_points=d["staged_points"],
        opened_on_turn=list(d["opened_on_turn"]),
        initial_meld_min=list(d["initial_meld_min"]),
    )


# --- match state -------------------------------------------------------------


def match_to_dict(match: MatchState) -> dict[str, Any]:
    return {
        "config": config_to_dict(match.cfg),
        "seed": match.seed,
        "scores": list(match.scores),
        "round_index": match.round_index,
        "round": round_to_dict(match.round),
        "action_log": list(match.action_log),
        "match_over": match.match_over,
        "winner_side": match.winner_side,
    }


def match_from_dict(d: dict[str, Any]) -> MatchState:
    cfg = config_from_dict(d["config"])
    return MatchState(
        cfg=cfg,
        seed=d["seed"],
        round=round_from_dict(cfg, d["round"]),
        scores=list(d["scores"]),
        round_index=d["round_index"],
        action_log=list(d["action_log"]),
        match_over=d["match_over"],
        winner_side=d["winner_side"],
    )


# --- hashing -----------------------------------------------------------------


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def state_hash(state: RoundState) -> str:
    """Canonical fingerprint of a round state (determinism/replay tests)."""
    return hashlib.sha256(canonical_json(round_to_dict(state)).encode()).hexdigest()
