"""Process-pool self-play collection over W worker SelfPlayCollectors.

Each worker owns a slice of the envs and the stride-W episode-counter residue
class starting at its slot index, so every worker draws from a disjoint stream
of the same ``seed*1_000_000 + counter`` seed space and every episode stays
replayable from (rules_config, seed, action_log).

Tasks are stateless and keyed by slot: every update the parent broadcasts the
CPU net weights plus each slot's counter, and results are gathered in slot
order — determinism is independent of which OS process runs which task.
Workers always collect on CPU (the forward pass is negligible); the trainer's
--device only affects the parent's PPO update and evaluation.
"""

from __future__ import annotations

import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import torch

from buraco.config import RulesConfig
from buraco.engine.serialize import config_from_dict, config_to_dict
from buraco.rl.buffer import SeatTrajectory
from buraco.rl.nets import PolicyValueNet
from buraco.rl.obs import ObsSpec
from buraco.rl.rollout import RolloutStats, SelfPlayCollector

_WORKER: dict[str, Any] = {}


def _init_worker(
    rules_dict: dict,
    spec_dict: dict,
    net_dims: tuple[int, int, int, int],
    envs_per_worker: int,
    seed: int,
    stride: int,
    history_len: int,
    trash_top_k: int,
) -> None:
    torch.set_num_threads(1)  # W single-threaded workers; no oversubscription
    cfg = config_from_dict(rules_dict)
    spec = ObsSpec.from_dict(spec_dict)
    flat_dim, num_actions, hidden, layers = net_dims
    _WORKER["collector"] = SelfPlayCollector(
        cfg,
        spec,
        num_envs=envs_per_worker,
        seed=seed,
        history_len=history_len,
        trash_top_k=trash_top_k,
        counter_stride=stride,
    )
    _WORKER["net"] = PolicyValueNet(flat_dim, num_actions, hidden=hidden, layers=layers)
    _WORKER["seed"] = seed


def _collect_task(
    slot: int, state_dict: dict, counter_start: int, min_steps: int
) -> tuple[int, list[SeatTrajectory], RolloutStats, int]:
    collector: SelfPlayCollector = _WORKER["collector"]
    net: PolicyValueNet = _WORKER["net"]
    net.load_state_dict(state_dict)
    # Fixed formula of (seed, slot, counter_start): the sampling stream is
    # deterministic across reruns and resumes, independent of pool scheduling.
    torch.manual_seed(
        ((_WORKER["seed"] * 1_000_003 + slot) * 2_654_435_761 + counter_start) % (2**63)
    )
    collector.episode_counter = counter_start
    trajs, stats = collector.collect(net, torch.device("cpu"), min_steps)
    return slot, trajs, stats, collector.episode_counter


class ParallelCollector:
    """Drop-in for SelfPlayCollector.collect, fanned out over a spawn pool."""

    def __init__(
        self,
        cfg: RulesConfig,
        spec: ObsSpec,
        num_envs: int,
        seed: int,
        num_workers: int,
        num_actions: int,
        hidden: int,
        layers: int,
        history_len: int = 8,
        trash_top_k: int = 8,
    ) -> None:
        if num_workers < 1:
            raise ValueError("ParallelCollector needs num_workers >= 1")
        if num_envs < num_workers or num_envs % num_workers:
            # Silent floor-division would make the actual env topology diverge
            # from the run config (and config.json) — refuse instead.
            raise ValueError(
                f"num_envs ({num_envs}) must be a positive multiple of "
                f"num_workers ({num_workers})"
            )
        self.cfg = cfg
        self.num_players = cfg.table.num_players
        self.num_workers = num_workers
        self.num_actions = num_actions
        # Slot w owns residue class w (mod W) of the episode-seed counter.
        self.counters = list(range(num_workers))
        self._executor: ProcessPoolExecutor | None = ProcessPoolExecutor(
            max_workers=num_workers,
            mp_context=mp.get_context("spawn"),
            initializer=_init_worker,
            initargs=(
                config_to_dict(cfg),
                spec.to_dict(),
                (spec.flat_dim, num_actions, hidden, layers),
                num_envs // num_workers,
                seed,
                num_workers,
                history_len,
                trash_top_k,
            ),
        )

    @property
    def episode_counter(self) -> int:
        """Legacy scalar for checkpoints: the pool's high-water mark."""
        return max(self.counters)

    def collect(
        self, net: PolicyValueNet, device: torch.device, min_steps: int
    ) -> tuple[list[SeatTrajectory], RolloutStats]:
        assert self._executor is not None, "collector is closed"
        start = time.perf_counter()
        state_dict = {k: v.detach().cpu() for k, v in net.state_dict().items()}
        per_worker = max(1, min_steps // self.num_workers)
        futures = [
            self._executor.submit(
                _collect_task, slot, state_dict, self.counters[slot], per_worker
            )
            for slot in range(self.num_workers)
        ]
        results = sorted((f.result() for f in futures), key=lambda r: r[0])

        trajs: list[SeatTrajectory] = []
        episodes = env_steps = truncations = 0
        payoff_abs_sum = 0.0
        for slot, worker_trajs, stats, counter_end in results:
            self.counters[slot] = counter_end
            trajs.extend(worker_trajs)
            episodes += stats.episodes
            env_steps += stats.env_steps
            truncations += round(stats.truncation_rate * stats.episodes)
            payoff_abs_sum += stats.mean_abs_payoff * stats.episodes
        seconds = time.perf_counter() - start  # parent wall: steps_per_sec stays honest
        return trajs, RolloutStats(
            episodes=episodes,
            env_steps=env_steps,
            mean_ep_len=env_steps / max(episodes, 1),
            truncation_rate=truncations / max(episodes, 1),
            mean_abs_payoff=payoff_abs_sum / max(episodes, 1),
            seconds=seconds,
        )

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None
