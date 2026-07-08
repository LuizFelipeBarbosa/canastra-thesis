"""Opponent pool: frozen past policies and scripted agents mixed into self-play.

Pure self-play never points a gradient at an external opponent, so strength
against scripted baselines plateaus (measured: flat vs-heuristic win rate over
12k updates). The mixture seats frozen opponents on the non-learner side for a
configurable fraction of episodes: `opp_heuristic` plays HeuristicAgent seats,
`opp_pool` plays a uniformly sampled past checkpoint of the learner itself.
Assignment is a pure function of the episode seed, so the episode stream stays
deterministic and replayable regardless of worker scheduling.

Pool members are bare state-dict files under `<run_dir>/pool/`; every net
dimension is recovered from tensor shapes, so files carry no config. Loading
is lazy and cached per manifest path — parallel workers each hold their own
cache and sync it from the manifest broadcast with every collect call.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from buraco.agents.heuristic_agent import HeuristicAgent
from buraco.config import RulesConfig
from buraco.rl.nets import PolicyValueNet

# Entropy word mixed into per-episode seeds so opponent assignment draws come
# from a stream disjoint from anything else derived from the episode seed.
_ASSIGN_STREAM = 0x0FF0


def save_pool_member(path: Path, net: torch.nn.Module) -> None:
    """Atomic write of a bare CPU state dict; dims are recovered from shapes."""
    state = {k: v.detach().cpu() for k, v in net.state_dict().items()}
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save({"model": state}, tmp)
    os.replace(tmp, path)


def load_pool_member(path: Path | str) -> PolicyValueNet:
    state = torch.load(path, map_location="cpu", weights_only=True)["model"]
    hidden, flat_dim = state["trunk.0.weight"].shape
    num_actions = state["policy_head.weight"].shape[0]
    layers = sum(
        1 for k, v in state.items()
        if k.startswith("trunk.") and k.endswith(".weight") and v.dim() == 2
    )
    net = PolicyValueNet(flat_dim, num_actions, hidden=hidden, layers=layers)
    net.load_state_dict(state)
    return net.eval()


@dataclass(frozen=True)
class EpisodeOpponents:
    """Seat assignment for one mixed episode."""

    net_seats: frozenset[int]  # seats played (and recorded) by the learner
    scripted: dict[int, Any]  # seat -> agent with act(raw_obs, legal_ids, cfg)
    frozen: dict[int, str]  # seat -> pool member manifest path
    learner_seat: int  # any learner seat; payoffs are shared per side


class OpponentMixture:
    """Per-episode opponent draw plus the lazy frozen-net cache."""

    def __init__(self, p_heuristic: float = 0.0, p_pool: float = 0.0):
        if p_heuristic < 0 or p_pool < 0 or p_heuristic + p_pool > 1:
            raise ValueError(
                f"opponent probabilities must be >= 0 and sum to <= 1, got "
                f"heuristic={p_heuristic} pool={p_pool}"
            )
        self.p_heuristic = p_heuristic
        self.p_pool = p_pool
        self.paths: list[str] = []
        self._cache: dict[str, PolicyValueNet] = {}

    @property
    def enabled(self) -> bool:
        return self.p_heuristic > 0 or self.p_pool > 0

    def set_manifest(self, paths: list[str]) -> None:
        self.paths = list(paths)
        for stale in set(self._cache) - set(self.paths):
            del self._cache[stale]

    def net_for(self, path: str, device: torch.device | None = None) -> PolicyValueNet:
        net = self._cache.get(path)
        if net is None:
            net = load_pool_member(path)
            self._cache[path] = net
        if device is not None and next(net.parameters()).device != device:
            net.to(device)  # serial collection runs on the trainer device
        return net

    def assign(
        self, episode_seed: int, episode_counter: int, cfg: RulesConfig
    ) -> EpisodeOpponents | None:
        """None = pure self-play. Deterministic in (episode_seed, manifest)."""
        rng = np.random.default_rng([_ASSIGN_STREAM, episode_seed])
        draw = rng.random()
        if draw >= self.p_heuristic + self.p_pool:
            return None
        pool_path: str | None = None
        if draw >= self.p_heuristic:
            if not self.paths:  # pool mass folds to self-play until seeded
                return None
            pool_path = self.paths[int(rng.integers(len(self.paths)))]

        num_players = cfg.table.num_players
        learner_side = episode_counter % 2  # side-swap kills first-mover bias
        net_seats = frozenset(
            s for s in range(num_players) if cfg.table.side(s) == learner_side
        )
        opp_seats = [s for s in range(num_players) if s not in net_seats]
        if pool_path is not None:
            return EpisodeOpponents(
                net_seats, {}, {s: pool_path for s in opp_seats}, min(net_seats)
            )
        return EpisodeOpponents(
            net_seats,
            {s: HeuristicAgent(episode_seed * num_players + s) for s in opp_seats},
            {},
            min(net_seats),
        )


class PoolManager:
    """Snapshots the learner into `<run_dir>/pool/`, capped at `size` members.

    The manifest is the ordered list of member file names; checkpoints persist
    the names and resolve them against the run dir on resume, so runs survive
    being moved wholesale (names, not absolute paths, are the durable record).
    """

    def __init__(self, pool_dir: Path, size: int, every: int):
        self.dir = pool_dir
        self.size = size
        self.every = every
        self.names: list[str] = []
        self._pending_unlink: list[str] = []

    @property
    def paths(self) -> list[str]:
        return [str(self.dir / name) for name in self.names]

    def restore(self, names: list[str]) -> None:
        missing = [n for n in names if not (self.dir / n).exists()]
        if missing:
            raise SystemExit(
                f"checkpoint references pool members missing from {self.dir}: "
                f"{', '.join(missing)}"
            )
        self.names = list(names)

    def maybe_snapshot(self, update: int, net: torch.nn.Module) -> bool:
        if self.every <= 0 or update % self.every:
            return False
        self.snapshot(update, net)
        return True

    def snapshot(self, update: int, net: torch.nn.Module) -> None:
        name = f"pool_{update:06d}.pt"
        self.dir.mkdir(parents=True, exist_ok=True)
        save_pool_member(self.dir / name, net)
        if name not in self.names:  # resume can revisit its snapshot update
            self.names.append(name)
        while len(self.names) > self.size:
            # Deferred: the file may still be referenced by the newest on-disk
            # checkpoint until the next one persists the shrunk manifest.
            self._pending_unlink.append(self.names.pop(0))

    def flush_evictions(self) -> None:
        """Unlink evicted members; call only after a checkpoint has been saved
        so a crash can never leave latest.pt referencing a deleted file."""
        for name in self._pending_unlink:
            if name not in self.names:  # paranoia: never delete a live member
                (self.dir / name).unlink(missing_ok=True)
        self._pending_unlink.clear()
