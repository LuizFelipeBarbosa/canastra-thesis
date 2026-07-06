"""Per-seat trajectories, GAE, and the flat update batch (numpy-only; no torch).

Each seat's decision points form an independent episode in the turn-based
reduction: rewards are all zero except the seat's last step, which carries that
seat's terminal payoff from `env.get_payoffs()` — never the scalar reward that
`env.step` returns at the terminal step (that belongs to the acting seat only).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SeatTrajectory:
    """One seat's experience within one episode."""

    obs: list[np.ndarray] = field(default_factory=list)
    masks: list[np.ndarray] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    logps: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    payoff: float = 0.0  # terminal zero-sum payoff for this seat

    def __len__(self) -> int:
        return len(self.actions)


def compute_gae(
    rewards: np.ndarray, values: np.ndarray, gamma: float, lam: float
) -> tuple[np.ndarray, np.ndarray]:
    """GAE over one complete trajectory with terminal bootstrap V_T = 0.

    Returns (advantages, returns) with returns = advantages + values.
    """
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(T)):
        next_value = values[t + 1] if t + 1 < T else 0.0
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
    return advantages, advantages + values.astype(np.float32)


@dataclass
class RolloutBatch:
    """Flat batch across all seats and episodes of one update."""

    obs: np.ndarray  # (N, D) float32
    masks: np.ndarray  # (N, A) int8
    actions: np.ndarray  # (N,) int64
    logps: np.ndarray  # (N,) float32
    values: np.ndarray  # (N,) float32
    advantages: np.ndarray  # (N,) float32
    returns: np.ndarray  # (N,) float32

    def __len__(self) -> int:
        return len(self.actions)


def build_batch(trajs: list[SeatTrajectory], gamma: float, lam: float) -> RolloutBatch:
    obs, masks, actions, logps, values, advantages, returns = [], [], [], [], [], [], []
    for traj in trajs:
        if not traj.actions:
            continue
        rewards = np.zeros(len(traj), dtype=np.float32)
        rewards[-1] = traj.payoff
        vals = np.asarray(traj.values, dtype=np.float32)
        adv, ret = compute_gae(rewards, vals, gamma, lam)
        obs.extend(traj.obs)
        masks.extend(traj.masks)
        actions.extend(traj.actions)
        logps.extend(traj.logps)
        values.append(vals)
        advantages.append(adv)
        returns.append(ret)
    return RolloutBatch(
        obs=np.stack(obs).astype(np.float32, copy=False),
        masks=np.stack(masks),
        actions=np.asarray(actions, dtype=np.int64),
        logps=np.asarray(logps, dtype=np.float32),
        values=np.concatenate(values),
        advantages=np.concatenate(advantages),
        returns=np.concatenate(returns),
    )
