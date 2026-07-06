"""SelfPlayCollector: complete episodes only, zero-sum payoffs, step accounting."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import numpy as np  # noqa: E402

from buraco.profiles import load_profile  # noqa: E402
from buraco.rl.buffer import build_batch  # noqa: E402
from buraco.rl.nets import PolicyValueNet  # noqa: E402
from buraco.rl.obs import ObsSpec  # noqa: E402
from buraco.rl.rollout import SelfPlayCollector  # noqa: E402


@pytest.mark.parametrize("players", [2, 4])
def test_collect_complete_episodes_zero_sum_and_accounting(players):
    torch.manual_seed(0)
    cfg = load_profile("buraco", num_players=players)
    spec = ObsSpec.from_cfg(cfg)
    net = PolicyValueNet(spec.flat_dim, 1585, hidden=32, layers=1)
    collector = SelfPlayCollector(cfg, spec, num_envs=2, seed=0)
    device = torch.device("cpu")

    trajs, stats = collector.collect(net, device, min_steps=300)

    assert stats.env_steps >= 300
    assert stats.episodes >= 1
    # Complete episodes only: trajectories arrive in groups of num_players
    # (every seat acted at least once in a full round) and step counts add up.
    assert len(trajs) == stats.episodes * players
    assert sum(len(t) for t in trajs) == stats.env_steps
    # Zero-sum per episode: seat payoffs of each episode group sum to ~0.
    for e in range(stats.episodes):
        group = trajs[e * players : (e + 1) * players]
        assert abs(sum(t.payoff for t in group)) < 1e-9
    # The batch built from these trajectories is internally consistent.
    batch = build_batch(trajs, gamma=1.0, lam=0.95)
    assert len(batch) == stats.env_steps
    assert np.all(np.isfinite(batch.advantages))


def test_episode_seeds_advance_monotonically():
    cfg = load_profile("buraco", num_players=2)
    spec = ObsSpec.from_cfg(cfg)
    net = PolicyValueNet(spec.flat_dim, 1585, hidden=32, layers=1)
    collector = SelfPlayCollector(cfg, spec, num_envs=2, seed=3)
    before = collector.episode_counter
    collector.collect(net, torch.device("cpu"), min_steps=100)
    assert collector.episode_counter > before
