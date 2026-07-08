"""Lockstep self-play rollout collection over N in-process envs.

One shared policy drives every learner seat: each iteration batches the
seat-to-act observation of all live envs into a single no_grad forward, then
steps the envs in a Python loop. Batches contain only complete episodes, so no
partial-episode bootstrapping is needed downstream. The `collect` interface is
deliberately process-agnostic so a multiprocessing pool of collectors can be
swapped in later without touching the trainer.

An optional OpponentMixture seats frozen opponents (scripted heuristic or past
pool checkpoints) on the non-learner side for a per-episode-seeded fraction of
episodes; those seats act without being recorded, and `min_steps` counts only
recorded learner steps. With the mixture disabled every seat is a learner seat
and the episode/action/RNG stream is bit-identical to plain self-play.
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
from buraco.rl.pool import EpisodeOpponents, OpponentMixture


@dataclass
class RolloutStats:
    episodes: int
    env_steps: int
    mean_ep_len: float
    truncation_rate: float
    mean_abs_payoff: float
    seconds: float
    recorded_steps: int = 0  # learner-seat steps; == env_steps in pure self-play
    mixed_episodes: int = 0  # episodes that seated frozen opponents
    mixed_wins: int = 0  # of those, learner-side wins

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
        mixture: OpponentMixture | None = None,
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
        self.mixture = mixture if mixture is not None and mixture.enabled else None

    @property
    def num_actions(self) -> int:
        return self.envs[0].num_actions

    def next_seed(self) -> int:
        seed = self._seed_base + self.episode_counter
        self.episode_counter += self.counter_stride
        return seed

    def set_pool_manifest(self, paths: list[str]) -> None:
        if self.mixture is not None:
            self.mixture.set_manifest(paths)

    def _start_episode(self, i: int):
        counter = self.episode_counter
        seed = self.next_seed()
        obs, info = self.envs[i].reset(seed=seed)
        assign = (
            self.mixture.assign(seed, counter, self.cfg) if self.mixture else None
        )
        trajs = [SeatTrajectory() for _ in range(self.num_players)]
        return obs, info, trajs, assign

    def collect(
        self, net: PolicyValueNet, device: torch.device, min_steps: int
    ) -> tuple[list[SeatTrajectory], RolloutStats]:
        start = time.perf_counter()
        done_trajs: list[SeatTrajectory] = []
        episodes = env_steps = recorded = truncations = 0
        mixed_eps = mixed_wins = 0
        payoff_abs_sum = 0.0

        obs_now: list[dict[str, np.ndarray]] = []
        info_now: list[dict] = []
        live_trajs: list[list[SeatTrajectory]] = []
        assigns: list[EpisodeOpponents | None] = []
        for i in range(len(self.envs)):
            obs, info, trajs, assign = self._start_episode(i)
            obs_now.append(obs)
            info_now.append(info)
            live_trajs.append(trajs)
            assigns.append(assign)

        collecting = True
        # Stop once min_steps learner steps are recorded AND every in-flight
        # episode has finished (complete-episodes-only); finished envs stop
        # being stepped.
        active = list(range(len(self.envs)))
        while active:
            # Learner turns: one batched forward over all envs where the
            # current policy acts. With the mixture disabled this is every
            # active env, in active order — identical to plain self-play.
            learner_rows = [
                i for i in active
                if assigns[i] is None or info_now[i]["to_play"] in assigns[i].net_seats
            ]
            flat = masks = actions_np = logps_np = values_np = None
            if learner_rows:
                flat = np.stack([self.spec.flatten(obs_now[i]) for i in learner_rows])
                masks = np.stack([info_now[i]["action_mask"] for i in learner_rows])
                with torch.no_grad():
                    logits, values = net(torch.from_numpy(flat).to(device))
                    dist = masked_dist(logits, torch.from_numpy(masks).to(device))
                    actions = dist.sample()
                    logps = dist.log_prob(actions)
                actions_np = actions.cpu().numpy()
                logps_np = logps.cpu().numpy()
                values_np = values.cpu().numpy()
            row_of = {i: row for row, i in enumerate(learner_rows)}

            # Frozen opponent turns: batched per pool member, in sorted path
            # order so torch RNG consumption is deterministic.
            opp_actions: dict[int, int] = {}
            frozen_rows: dict[str, list[int]] = {}
            for i in active:
                if i in row_of:
                    continue
                seat = info_now[i]["to_play"]
                path = assigns[i].frozen.get(seat)
                if path is not None:
                    frozen_rows.setdefault(path, []).append(i)
            for path in sorted(frozen_rows):
                rows = frozen_rows[path]
                fnet = self.mixture.net_for(path, device)
                f_flat = np.stack([self.spec.flatten(obs_now[i]) for i in rows])
                f_masks = np.stack([info_now[i]["action_mask"] for i in rows])
                with torch.no_grad():
                    f_logits, _ = fnet(torch.from_numpy(f_flat).to(device))
                    f_actions = masked_dist(
                        f_logits, torch.from_numpy(f_masks).to(device)
                    ).sample()
                for row, i in enumerate(rows):
                    opp_actions[i] = int(f_actions[row])

            # Scripted opponent turns act one env at a time on the raw view.
            for i in active:
                if i in row_of or i in opp_actions:
                    continue
                seat = info_now[i]["to_play"]
                opp_actions[i] = assigns[i].scripted[seat].act(
                    self.envs[i].observe_raw(seat),
                    info_now[i]["legal_actions"],
                    self.cfg,
                )

            still_active = []
            for i in active:
                env = self.envs[i]
                seat = info_now[i]["to_play"]
                if i in row_of:
                    row = row_of[i]
                    traj = live_trajs[i][seat]
                    traj.obs.append(flat[row])
                    traj.masks.append(masks[row])
                    traj.actions.append(int(actions_np[row]))
                    traj.logps.append(float(logps_np[row]))
                    traj.values.append(float(values_np[row]))
                    recorded += 1
                    action = int(actions_np[row])
                else:
                    action = opp_actions[i]

                obs, _, terminated, truncated, info = env.step(action)
                env_steps += 1
                if terminated or truncated:
                    episodes += 1
                    truncations += int(truncated)
                    payoffs = env.get_payoffs()
                    for s, payoff in enumerate(payoffs):
                        live_trajs[i][s].payoff = payoff
                    payoff_abs_sum += abs(payoffs[0])
                    assign = assigns[i]
                    if assign is None:
                        done_trajs.extend(t for t in live_trajs[i] if len(t))
                    else:
                        mixed_eps += 1
                        mixed_wins += payoffs[assign.learner_seat] > 0
                        done_trajs.extend(
                            live_trajs[i][s]
                            for s in sorted(assign.net_seats)
                            if len(live_trajs[i][s])
                        )
                    if collecting and recorded >= min_steps:
                        collecting = False
                    if collecting:
                        obs, info, live_trajs[i], assigns[i] = self._start_episode(i)
                        still_active.append(i)
                else:
                    still_active.append(i)
                obs_now[i], info_now[i] = obs, info
            if collecting and recorded >= min_steps:
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
            recorded_steps=recorded,
            mixed_episodes=mixed_eps,
            mixed_wins=mixed_wins,
        )
        return done_trajs, stats
