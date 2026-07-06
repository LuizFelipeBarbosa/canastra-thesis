"""Self-play training skeleton: the exact loop a DQN/PPO implementation plugs
into. No deep-learning dependency — the "policy" here is masked-random.

What your network replaces:
  * `policy(obs, mask)`  -> action id (argmax/sample over masked logits)
  * `update(trajectories)` -> gradient step

Everything else — perspective handling, masking, per-seat trajectory
bookkeeping, terminal payoffs — is shown working below.

Run: uv run python examples/selfplay_skeleton.py --episodes 5 --seed 3
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from buraco.env.env import BuracoEnv
from buraco.profiles import load_profile


@dataclass
class Trajectory:
    """One seat's experience within an episode."""

    observations: list[dict[str, np.ndarray]] = field(default_factory=list)
    masks: list[np.ndarray] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    reward: float = 0.0  # terminal zero-sum payoff for this seat


def masked_random_policy(
    obs: dict[str, np.ndarray], mask: np.ndarray, rng: random.Random
) -> int:
    """Stand-in for a network: sample uniformly over legal ids.

    A network would produce logits of shape (env.num_actions,), apply
    `logits[mask == 0] = -inf`, then sample/argmax.
    """
    legal = np.flatnonzero(mask)
    return int(rng.choice(legal))


def play_episode(env: BuracoEnv, seed: int, rng: random.Random) -> list[Trajectory]:
    trajectories = [Trajectory() for _ in range(env.cfg.table.num_players)]
    obs, info = env.reset(seed=seed)
    terminated = truncated = False
    while not (terminated or truncated):
        seat = info["to_play"]
        mask = info["action_mask"]
        action = masked_random_policy(obs, mask, rng)

        traj = trajectories[seat]
        traj.observations.append(obs)
        traj.masks.append(mask)
        traj.actions.append(action)

        obs, _, terminated, truncated, info = env.step(action)

    if terminated:
        for seat, payoff in enumerate(env.get_payoffs()):
            trajectories[seat].reward = payoff
    return trajectories


def update(trajectories: list[Trajectory]) -> dict[str, Any]:
    """Placeholder for the learner. Replace with your DQN/PPO update.

    All seats share one policy in self-play; teammates already share reward,
    and the episode is zero-sum across sides."""
    steps = sum(len(t.actions) for t in trajectories)
    return {"steps": steps, "rewards": [t.reward for t in trajectories]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="buraco")
    parser.add_argument("--players", type=int, default=2)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = load_profile(args.profile, num_players=args.players)
    env = BuracoEnv(cfg)
    rng = random.Random(args.seed)
    print(f"action space: {env.num_actions} ids | "
          f"obs fields: {len(env.reset(seed=0)[0])}")

    for episode in range(args.episodes):
        trajectories = play_episode(env, seed=args.seed + episode, rng=rng)
        stats = update(trajectories)
        print(f"episode {episode}: {stats['steps']} steps, "
              f"payoffs {['%.3f' % r for r in stats['rewards']]}")


if __name__ == "__main__":
    main()
