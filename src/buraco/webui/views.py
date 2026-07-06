"""Authoritative JSON state view for the browser (hidden-info discipline).

Everything the frontend renders comes from here. The view reads only zones
the human seat may know: its own hand, all melds (public), the open trash,
red-three trays, and public sizes/flags. Legal actions are sent decoded and
labeled so the client never re-implements action-id arithmetic; it only
POSTs back an id it was given.
"""

from __future__ import annotations

from typing import Any

from buraco.config import EPISODE_MATCH, RulesConfig
from buraco.describe import describe
from buraco.engine.actions import Add, CreateSeq, CreateSet, Discard, decode
from buraco.engine.legal import effective_frozen
from buraco.engine.melds import CreatePlan, Meld, MeldKind, SlotRole, plan_sequence, plan_set
from buraco.engine.scoring import canastra_bonus, meld_points
from buraco.engine.state import Phase, RoundState

FAMILY_NAMES = {
    "DrawDeck": "draw_deck",
    "DrawTrash": "draw_trash",
    "CreateSeq": "create_seq",
    "CreateSet": "create_set",
    "Add": "add",
    "Discard": "discard",
    "GoOut": "go_out",
    "EndRound": "end_round",
}


def family_of(action: Any) -> str:
    return FAMILY_NAMES[type(action).__name__]


def meld_snapshot(meld: Meld, cfg: RulesConfig) -> dict[str, Any]:
    """Full public picture of one meld, slots included (creation order)."""
    return {
        "kind": "seq" if meld.kind is MeldKind.SEQUENCE else "set",
        "suit": None if meld.suit is None else int(meld.suit),
        "rank": None if meld.rank is None else int(meld.rank),
        "start_pos": meld.start_pos,
        "size": meld.size,
        "slots": [
            {"card": int(s.card), "role": "wild" if s.role is SlotRole.WILD else "nat"}
            for s in meld.slots
        ],
        "wild_count": meld.wild_count,
        "is_clean": meld.is_clean,
        "is_canastra": meld.is_canastra(cfg.meld.canastra_min_size),
        "points": meld_points(cfg, meld),
        "bonus": canastra_bonus(cfg, meld),
    }


def _plan_slots(plan: CreatePlan | None) -> list[dict[str, Any]] | None:
    if plan is None:
        return None
    return [
        {"card": int(c), "role": "wild" if r is SlotRole.WILD else "nat"} for c, r in plan.slots
    ]


def legal_entries(legal_ids: list[int], cfg: RulesConfig, state: RoundState,
                  seat: int) -> list[dict[str, Any]]:
    """Decoded + labeled legal actions with `uses` previews for creates."""
    slots = cfg.meld.max_meld_slots
    hand = dict(state.hands[seat])
    entries: list[dict[str, Any]] = []
    for aid in legal_ids:
        action = decode(aid, slots)
        entry: dict[str, Any] = {
            "id": int(aid),
            "family": family_of(action),
            "label": describe(action, slots),
        }
        match action:
            case CreateSeq(suit=s, start=st, wild=w):
                plan = plan_sequence(cfg, hand, s, st, w)
                entry["suit"] = int(s)
                entry["start"] = st
                entry["wild"] = w
                entry["uses"] = sorted(int(c) for c in plan.consumed) if plan else []
                entry["slots_preview"] = _plan_slots(plan)
            case CreateSet(rank=r, wild=w):
                plan = plan_set(cfg, hand, r, w)
                entry["rank"] = int(r)
                entry["wild"] = w
                entry["uses"] = sorted(int(c) for c in plan.consumed) if plan else []
                entry["slots_preview"] = _plan_slots(plan)
            case Add(slot=sl, ct=c):
                entry["slot"] = sl
                entry["card"] = int(c)
            case Discard(ct=c):
                entry["card"] = int(c)
        entries.append(entry)
    return entries


def build_view(session: Any) -> dict[str, Any]:
    """The full state payload for the human seat. `session` is a GameSession."""
    env = session.env
    cfg: RulesConfig = session.cfg
    state: RoundState = env.state
    human: int = session.human_seat
    side = cfg.table.side(human)
    opp = 1 - side
    raw = env.observe_raw(human)
    done = session.terminated or session.truncated
    to_play = state.current_player
    human_to_play = to_play == human and not done

    winner_side = None
    if session.terminated:
        best = max(env.match_scores)
        leaders = [s for s, v in enumerate(env.match_scores) if v == best]
        winner_side = leaders[0] if len(leaders) == 1 else None

    return {
        "game_id": session.game_id,
        "cursor": len(env.action_log),
        "seed": env.seed,
        "profile": cfg.name,
        "episode": "match" if cfg.scoring.episode == EPISODE_MATCH else "round",
        "num_players": cfg.table.num_players,
        "human_seat": human,
        "human_side": side,
        "partner_seat": raw["partner_seat"],
        "bots": session.bot_names,
        "round_index": env.round_index,
        "turn_number": state.turn_number,
        "phase": Phase(state.phase).name,
        "to_play": to_play,
        "to_play_is_human": human_to_play,
        "done": done,
        "terminated": session.terminated,
        "truncated": session.truncated,
        "hand": sorted(int(c) for c in state.hands[human].elements()),
        "all_hand_sizes": raw["all_hand_sizes"],
        "deck_size": len(state.stock),
        "trash": [int(c) for c in state.trash],
        "melds": {
            "own": [meld_snapshot(m, cfg) for m in state.side_melds(side)],
            "opp": [meld_snapshot(m, cfg) for m in state.side_melds(opp)],
        },
        "morto": {
            "taken": raw["morto_taken"],
            "remaining": raw["mortos_remaining"],
            # present=False with taken=False means the packet was folded into
            # the stock (biriba exhaustion), not picked up.
            "present": [
                side < len(state.morto) and state.morto[side] is not None,
                opp < len(state.morto) and state.morto[opp] is not None,
            ],
        },
        "red_threes": {
            "own": [int(c) for c in state.red_threes[side]] if state.red_threes else [],
            "opp": [int(c) for c in state.red_threes[opp]] if state.red_threes else [],
        },
        "canasta": {
            "frozen": state.frozen,
            "pile_frozen_for_you": effective_frozen(state, side),
            "pile_blocked": state.pile_blocked_for_next,
            "pending_pile_card": state.pending_pile_card,
            "pending_pair_only": state.pending_pile_pair_only,
            "staged_points": state.staged_points,
            "initial_meld_enabled": cfg.initial_meld.enabled,
            "initial_meld_done": raw["initial_meld_done"],
            "initial_meld_min": raw["initial_meld_min"],
        },
        "scores": {
            "round_public": raw["round_score"],
            "match": raw["match_score"],
            "target": cfg.scoring.match_target,
        },
        "rules": {
            "canastra_min_size": cfg.meld.canastra_min_size,
            "cards_per_player": cfg.table.cards_per_player,
            "draw_rule": cfg.discard_pile.draw_rule,
            "morto_count": cfg.morto.count,
            "allow_sequences": cfg.meld.allow_sequences,
            "allow_sets": cfg.meld.allow_sets,
            "wild_ranks": sorted(int(r) for r in cfg.wildcard.wild_ranks),
            "jokers_wild": cfg.wildcard.jokers_wild,
        },
        "legal": (
            legal_entries(session.env.legal_actions(), cfg, state, human)
            if human_to_play else None
        ),
        "payoffs": env.get_payoffs() if done else None,
        "winner_side": winner_side,
        "last_round_summary": session.last_round_summary,
    }
