"""Lockstep self-play rollout collection over N in-process envs.

One shared policy drives every seat: each iteration batches the seat-to-act
observation of all live envs into a single no_grad forward, then steps the
envs in a Python loop. Batches contain only complete episodes, so no
partial-episode bootstrapping is needed downstream. The `collect` interface is
deliberately process-agnostic so a multiprocessing pool of collectors can be
swapped in later without touching the trainer.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch

from buraco.config import RulesConfig
from buraco.env.env import BuracoEnv
from buraco.rl.buffer import SeatTrajectory
from buraco.rl.nets import PolicyValueNet, masked_dist
from buraco.rl.obs import ObsSpec


@dataclass
class RolloutStats:
    episodes: int
    env_steps: int
    mean_ep_len: float
    truncation_rate: float
    mean_abs_payoff: float
    seconds: float

    @property
    def steps_per_sec(self) -> float:
        return self.env_steps / max(self.seconds, 1e-9)


class SelfPlayCollector:
    def __init__(
        self,
        cfg: RulesConfig,
        spec: ObsSpec,
        num_envs: int,
        seed: int,
        history_len: int = 8,
        trash_top_k: int = 8,
        counter_stride: int = 1,
    ) -> None:
        self.cfg = cfg
        self.spec = spec
        self.envs = [
            BuracoEnv(cfg, history_len=history_len, trash_top_k=trash_top_k)
            for _ in range(num_envs)
        ]
        self.num_players = cfg.table.num_players
        # Monotone counter: every training episode is replayable from
        # (rules_config, seed, action_log). Persisted in checkpoints.
        # A parallel pool gives each worker slot w the stride-W residue class
        # starting at w, so seed streams stay disjoint (see rl/parallel.py).
        self.episode_counter = 0
        self.counter_stride = counter_stride
        self._seed_base = seed * 1_000_000

    @property
    def num_actions(self) -> int:
        return self.envs[0].num_actions

    def next_seed(self) -> int:
        seed = self._seed_base + self.episode_counter
        self.episode_counter += self.counter_stride
        return seed

    def collect(
        self, net: PolicyValueNet, device: torch.device, min_steps: int
    ) -> tuple[list[SeatTrajectory], RolloutStats]:
        start = time.perf_counter()
        done_trajs: list[SeatTrajectory] = []
        episodes = env_steps = truncations = 0
        payoff_abs_sum = 0.0

        obs_now: list[dict[str, np.ndarray]] = []
        info_now: list[dict] = []
        live_trajs: list[list[SeatTrajectory]] = []
        for env in self.envs:
            obs, info = env.reset(seed=self.next_seed())
            obs_now.append(obs)
            info_now.append(info)
            live_trajs.append([SeatTrajectory() for _ in range(self.num_players)])

        collecting = True
        # Stop once min_steps are collected AND every in-flight episode has
        # finished (complete-episodes-only); finished envs stop being stepped.
        active = list(range(len(self.envs)))
        while active:
            flat = np.stack([self.spec.flatten(obs_now[i]) for i in active])
            masks = np.stack([info_now[i]["action_mask"] for i in active])
            with torch.no_grad():
                logits, values = net(torch.from_numpy(flat).to(device))
                dist = masked_dist(logits, torch.from_numpy(masks).to(device))
                actions = dist.sample()
                logps = dist.log_prob(actions)
            actions_np = actions.cpu().numpy()
            logps_np = logps.cpu().numpy()
            values_np = values.cpu().numpy()

            still_active = []
            for row, i in enumerate(active):
                env = self.envs[i]
                seat = info_now[i]["to_play"]
                traj = live_trajs[i][seat]
                traj.obs.append(flat[row])
                traj.masks.append(masks[row])
                traj.actions.append(int(actions_np[row]))
                traj.logps.append(float(logps_np[row]))
                traj.values.append(float(values_np[row]))

                obs, _, terminated, truncated, info = env.step(int(actions_np[row]))
                env_steps += 1
                if terminated or truncated:
                    episodes += 1
                    truncations += int(truncated)
                    payoffs = env.get_payoffs()
                    for s, payoff in enumerate(payoffs):
                        live_trajs[i][s].payoff = payoff
                    payoff_abs_sum += abs(payoffs[0])
                    done_trajs.extend(t for t in live_trajs[i] if len(t))
                    if collecting and env_steps >= min_steps:
                        collecting = False
                    if collecting:
                        obs, info = env.reset(seed=self.next_seed())
                        live_trajs[i] = [SeatTrajectory() for _ in range(self.num_players)]
                        still_active.append(i)
                else:
                    still_active.append(i)
                obs_now[i], info_now[i] = obs, info
            if collecting and env_steps >= min_steps:
                collecting = False
            active = still_active

        seconds = time.perf_counter() - start
        stats = RolloutStats(
            episodes=episodes,
            env_steps=env_steps,
            mean_ep_len=env_steps / max(episodes, 1),
            truncation_rate=truncations / max(episodes, 1),
            mean_abs_payoff=payoff_abs_sum / max(episodes, 1),
            seconds=seconds,
        )
        return done_trajs, stats
