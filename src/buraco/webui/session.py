"""GameSession: one browser-driven game over BuracoEnv. HTTP-free.

Stepping is frontend-driven: the browser applies exactly one micro-action
per request (its own via `apply_human`, one bot micro-action via
`bot_step`) so the UI animates each move at its own pace. Every mutating
call carries the cursor (`len(env.action_log)`) the client last saw; a
mismatch raises StaleCursor so double-fired requests are harmless.
"""

from __future__ import annotations

import threading
import uuid
from collections import Counter
from dataclasses import replace
from typing import Any

from buraco.agents.heuristic_agent import HeuristicAgent
from buraco.agents.random_agent import RandomAgent
from buraco.config import EPISODE_MATCH, EPISODE_ROUND
from buraco.engine.actions import decode
from buraco.engine.turns import IllegalAction
from buraco.env.env import BuracoEnv
from buraco.profiles import PROFILES, load_profile
from buraco.webui.events import build_event, build_round_summary, snapshot_public
from buraco.webui.views import build_view

AGENT_FACTORIES = {"heuristic": HeuristicAgent, "random": RandomAgent}


class SessionError(Exception):
    """Request is well-formed but wrong right now (not your turn, game over…)."""


class StaleCursor(Exception):
    """Client acted on an outdated view; it should reconcile and retry."""


class GameSession:
    def __init__(
        self,
        profile: str = "buraco",
        num_players: int = 2,
        human_seat: int = 0,
        bots: list[str] | None = None,
        episode: str = "round",
        seed: int | None = None,
    ) -> None:
        if profile not in PROFILES:
            raise SessionError(f"unknown profile {profile!r}; available: {sorted(PROFILES)}")
        if episode not in ("round", "match"):
            raise SessionError(f"episode must be 'round' or 'match', got {episode!r}")
        cfg = load_profile(profile, num_players=num_players)
        wanted = EPISODE_MATCH if episode == "match" else EPISODE_ROUND
        if cfg.scoring.episode != wanted:
            cfg = replace(cfg, scoring=replace(cfg.scoring, episode=wanted))
        if not 0 <= human_seat < cfg.table.num_players:
            raise SessionError(f"human_seat {human_seat} out of range for {num_players} players")

        names = list(bots) if bots else ["heuristic"] * cfg.table.num_players
        if len(names) != cfg.table.num_players:
            raise SessionError(f"bots must list {cfg.table.num_players} entries, got {len(names)}")
        for seat, name in enumerate(names):
            if seat != human_seat and name not in AGENT_FACTORIES:
                raise SessionError(
                    f"unknown bot {name!r}; available: {sorted(AGENT_FACTORIES)}"
                )
        names[human_seat] = None

        self.cfg = cfg
        self.env = BuracoEnv(cfg)
        self.env.reset(seed=seed)
        self.human_seat = human_seat
        self.bot_names = names
        self.agents = {
            seat: AGENT_FACTORIES[name](self.env.seed + seat)
            for seat, name in enumerate(names)
            if seat != human_seat
        }
        self.game_id = uuid.uuid4().hex
        self.lock = threading.RLock()
        self.terminated = False
        self.truncated = False
        self.last_event: dict[str, Any] | None = None
        self.last_round_summary: dict[str, Any] | None = None

    # --- queries -----------------------------------------------------------

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated

    @property
    def cursor(self) -> int:
        return len(self.env.action_log)

    def view(self) -> dict[str, Any]:
        with self.lock:
            return build_view(self)

    # --- mutations (one micro-action each) -----------------------------------

    def apply_human(self, action_id: int, cursor: int) -> tuple[dict[str, Any], dict[str, Any]]:
        with self.lock:
            self._check_cursor(cursor)
            if self.done:
                raise SessionError("game is over")
            if self.env.current_player != self.human_seat:
                raise SessionError("not your turn")
            if int(action_id) not in self.env.legal_actions():
                raise IllegalAction(f"action {action_id} is not legal now")
            return self._step(int(action_id))

    def bot_step(self, cursor: int) -> tuple[dict[str, Any], dict[str, Any]]:
        with self.lock:
            self._check_cursor(cursor)
            if self.done:
                raise SessionError("game is over")
            seat = self.env.current_player
            if seat == self.human_seat:
                raise SessionError("human to play; POST the action instead")
            action_id = self.agents[seat].act(
                self.env.observe_raw(seat), self.env.legal_actions(), self.cfg
            )
            return self._step(int(action_id))

    # --- internals -----------------------------------------------------------

    def _check_cursor(self, cursor: int) -> None:
        if int(cursor) != self.cursor:
            raise StaleCursor(f"cursor {cursor} != {self.cursor}")

    def _step(self, action_id: int) -> tuple[dict[str, Any], dict[str, Any]]:
        env = self.env
        prev_state = env.state
        actor = prev_state.current_player
        action = decode(action_id, self.cfg.meld.max_meld_slots)
        pre = snapshot_public(prev_state)
        pre_hand = Counter(prev_state.hands[self.human_seat])

        _obs, _reward, terminated, truncated, _info = env.step(action_id)
        self.terminated, self.truncated = terminated, truncated

        # In EPISODE_MATCH, env.step deals the next round in place; prev_state
        # is then the intact terminal round the summary must be built from.
        new_round = env.state is not prev_state
        post = snapshot_public(prev_state)
        post_hand = Counter(prev_state.hands[self.human_seat])

        summary = None
        if prev_state.round_over:
            summary = build_round_summary(prev_state, env.match_scores, terminated)
            summary["round_index"] = env.round_index - (1 if new_round else 0)
            self.last_round_summary = summary

        event = build_event(
            seq=len(env.action_log),
            actor=actor,
            action=action,
            action_id=action_id,
            cfg=self.cfg,
            human_seat=self.human_seat,
            pre=pre,
            post=post,
            pre_human_hand=pre_hand,
            post_human_hand=post_hand,
            state_after=prev_state,
            terminated=terminated,
            truncated=truncated,
            new_round=new_round,
            round_summary=summary,
        )
        self.last_event = event
        return event, build_view(self)
