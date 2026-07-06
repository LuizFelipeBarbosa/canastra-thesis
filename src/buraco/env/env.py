"""RL environment, current-player perspective (SPEC 03 §3.7).

RLCard/OpenSpiel-style turn-based convention with a Gymnasium-shaped surface:
`reset(seed)` and `step(action_id)` return the observation of the seat now to
act; `info` carries the action mask and legal-action ids. Rewards are terminal
zero-sum team score differentials scaled by `cfg.scoring.reward_scale`;
`step`'s scalar reward is the acting seat's payoff and `get_payoffs()` /
`info["payoffs"]` expose the full per-seat vector at episode end.

Episode granularity follows `cfg.scoring.episode`: one round (default) or a
match to `match_target` with rotating dealer. A replay is
`(rules_config, seed, action_log)`; see `replay()`.
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np

from buraco.config import EPISODE_MATCH, RulesConfig
from buraco.engine.actions import (
    Action,
    Add,
    CreateSeq,
    CreateSet,
    Discard,
    action_space_size,
    decode,
    encode,
)
from buraco.engine.actions import DrawDeck as _DrawDeck
from buraco.engine.actions import DrawTrash as _DrawTrash
from buraco.engine.actions import GoOut as _GoOut
from buraco.engine.legal import legal_actions as engine_legal_actions
from buraco.engine.scoring import round_scores
from buraco.engine.state import Phase, RoundState, deal_round
from buraco.engine.turns import IllegalAction, apply_action
from buraco.env.encoding import action_mask as build_mask
from buraco.env.encoding import encode_observation
from buraco.env.observations import HistoryItem, debug_view, observe

__all__ = ["BuracoEnv", "IllegalAction", "replay"]


def _history_entry(action: Action, actor: int) -> HistoryItem:
    match action:
        case _DrawDeck():
            return HistoryItem(actor, "draw_deck", None)
        case _DrawTrash():
            return HistoryItem(actor, "draw_trash", None)
        case CreateSeq():
            return HistoryItem(actor, "create_seq", None)
        case CreateSet():
            return HistoryItem(actor, "create_set", None)
        case Add(ct=ct):
            return HistoryItem(actor, "add", ct)
        case Discard(ct=ct):
            return HistoryItem(actor, "discard", ct)
        case _GoOut():
            return HistoryItem(actor, "go_out", None)
    return HistoryItem(actor, "draw_deck", None)  # END_ROUND folds into draw slot


class BuracoEnv:
    """Turn-based hidden-information environment over the rules engine."""

    def __init__(
        self,
        cfg: RulesConfig | None = None,
        perfect_info: bool = False,
        history_len: int = 8,
        trash_top_k: int = 8,
        reward_per_round: bool = False,
        reward_on_truncation: bool = False,
    ) -> None:
        if cfg is None:
            from buraco.profiles import buraco

            cfg = buraco(2)
        if cfg.table.num_sides != 2:
            raise ValueError(
                "BuracoEnv models exactly two sides (2p head-to-head or 4p "
                f"teams); got num_sides={cfg.table.num_sides}"
            )
        self.cfg = cfg
        self.perfect_info = perfect_info
        self.history_len = history_len
        self.trash_top_k = trash_top_k
        self.reward_per_round = reward_per_round
        self.reward_on_truncation = reward_on_truncation
        self.num_actions = action_space_size(cfg.meld.max_meld_slots)
        self._rng = random.Random()
        self.state: RoundState | None = None
        self._done = True

    # --- episode lifecycle ---------------------------------------------------

    def reset(self, seed: int | None = None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        if seed is None:
            seed = random.SystemRandom().randrange(2**63)
        self.seed = seed
        self._rng = random.Random(seed)
        self.action_log: list[int] = []
        self.match_scores = [0] * self.cfg.table.num_sides
        self.round_index = 0
        self.history: list[HistoryItem] = []
        self._melds_this_turn = 0
        self._last_turn_number = 0
        self._payoffs = [0.0] * self.cfg.table.num_players
        self._done = False
        self._truncated = False
        self.state = deal_round(
            self.cfg, self._rng, first_player=0, match_scores=self.match_scores
        )
        return self._observation(self.state.current_player), self._info()

    def step(
        self, action_id: int
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        if self._done or self.state is None:
            raise IllegalAction("episode is over; call reset()")
        state = self.state
        actor = state.current_player
        action = decode(int(action_id), self.cfg.meld.max_meld_slots)
        apply_action(state, action)  # raises IllegalAction on masked ids
        self.action_log.append(int(action_id))
        self.history.append(_history_entry(action, actor))

        if state.turn_number != self._last_turn_number:
            self._last_turn_number = state.turn_number
            self._melds_this_turn = 0
        elif isinstance(action, (CreateSeq, CreateSet, Add)):
            self._melds_this_turn += 1

        reward = 0.0
        terminated = False
        truncated = False

        if state.round_over:
            scores = round_scores(state)
            for side in range(self.cfg.table.num_sides):
                self.match_scores[side] += scores[side]
            if self.cfg.scoring.episode == EPISODE_MATCH:
                if max(self.match_scores) >= self.cfg.scoring.match_target:
                    terminated = True
                    self._set_payoffs(self.match_scores)
                else:
                    if self.reward_per_round:
                        reward = self._seat_payoff(scores, actor)
                    self._next_round()
            else:
                terminated = True
                self._set_payoffs(scores)
        elif state.turn_number >= self.cfg.turn.truncation_cap:
            truncated = True
            if self.reward_on_truncation:
                self._set_payoffs(round_scores(state))

        if terminated or truncated:
            self._done = True
            reward = self._payoffs[actor]

        obs_seat = actor if self._done else self.state.current_player
        return self._observation(obs_seat), reward, terminated, truncated, self._info()

    def _next_round(self) -> None:
        self.round_index += 1
        self.history.clear()
        self._melds_this_turn = 0
        self._last_turn_number = 0
        self.state = deal_round(
            self.cfg,
            self._rng,
            first_player=self.round_index % self.cfg.table.num_players,
            match_scores=self.match_scores,
        )

    def _set_payoffs(self, side_scores: list[int]) -> None:
        self._payoffs = [
            self._seat_payoff(side_scores, seat)
            for seat in range(self.cfg.table.num_players)
        ]

    def _seat_payoff(self, side_scores: list[int], seat: int) -> float:
        side = self.cfg.table.side(seat)
        diff = side_scores[side] - side_scores[1 - side]
        return diff * self.cfg.scoring.reward_scale

    # --- queries ----------------------------------------------------------------

    def legal_actions(self) -> list[int]:
        if self._done or self.state is None:
            return []
        slots = self.cfg.meld.max_meld_slots
        return sorted(encode(a, slots) for a in engine_legal_actions(self.state))

    def action_mask(self) -> np.ndarray:
        if self._done or self.state is None:
            return np.zeros(self.num_actions, dtype=np.int8)
        return build_mask(self.state)

    def get_payoffs(self) -> list[float]:
        return list(self._payoffs)

    @property
    def current_player(self) -> int:
        assert self.state is not None
        return self.state.current_player

    # --- observation plumbing -----------------------------------------------------

    def observe_raw(self, seat: int) -> dict[str, Any]:
        assert self.state is not None
        return observe(self.state, seat, self.match_scores, self.history, self._melds_this_turn)

    def _observation(self, seat: int) -> dict[str, np.ndarray]:
        return encode_observation(
            self.observe_raw(seat), self.cfg, self.history_len, self.trash_top_k
        )

    def _info(self) -> dict[str, Any]:
        assert self.state is not None
        mask = self.action_mask()
        info: dict[str, Any] = {
            "action_mask": mask,
            "legal_actions": [int(a) for a in np.flatnonzero(mask)],
            "to_play": self.state.current_player,
            "team": self.cfg.table.side(self.state.current_player),
            "phase": Phase(self.state.phase).name,
            "round_index": self.round_index,
            "match_scores": list(self.match_scores),
        }
        if self._done:
            info["payoffs"] = self.get_payoffs()
        if self.perfect_info:
            info["debug"] = debug_view(self.state)
        return info


def replay(
    cfg: RulesConfig, seed: int, action_ids: list[int], **env_kwargs: Any
) -> BuracoEnv:
    """Reconstruct an episode from `(rules_config, seed, action_log)`.

    Returns the env at the resulting state; determinism means every state
    hash along the way matches the original run (SPEC 01 §8.8)."""
    env = BuracoEnv(cfg, **env_kwargs)
    env.reset(seed=seed)
    for action_id in action_ids:
        env.step(action_id)
    return env
