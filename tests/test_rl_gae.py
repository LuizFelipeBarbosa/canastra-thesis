"""GAE on hand-crafted trajectories with terminal-only reward (numpy-only)."""

from __future__ import annotations

import numpy as np

from buraco.rl.buffer import SeatTrajectory, build_batch, compute_gae


def test_gamma1_lambda1_gives_return_minus_value():
    rewards = np.array([0.0, 0.0, 0.5], dtype=np.float32)
    values = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    adv, ret = compute_gae(rewards, values, gamma=1.0, lam=1.0)
    assert np.allclose(adv, 0.5 - values)
    assert np.allclose(ret, [0.5, 0.5, 0.5])


def test_lambda0_gives_one_step_td():
    rewards = np.array([0.0, 0.0, 0.5], dtype=np.float32)
    values = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    adv, _ = compute_gae(rewards, values, gamma=1.0, lam=0.0)
    expected = np.array(
        [0.0 + values[1] - values[0], 0.0 + values[2] - values[1], 0.5 + 0.0 - values[2]]
    )
    assert np.allclose(adv, expected)


def test_three_step_hand_computed_case():
    rewards = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    values = np.array([0.2, 0.4, 0.6], dtype=np.float32)
    gamma, lam = 0.9, 0.5
    d2 = 1.0 + 0.0 - 0.6
    d1 = 0.0 + 0.9 * 0.6 - 0.4
    d0 = 0.0 + 0.9 * 0.4 - 0.2
    a2 = d2
    a1 = d1 + gamma * lam * a2
    a0 = d0 + gamma * lam * a1
    adv, ret = compute_gae(rewards, values, gamma, lam)
    assert np.allclose(adv, [a0, a1, a2])
    assert np.allclose(ret, adv + values)


def test_build_batch_places_payoff_on_last_step_only():
    def traj(n: int, payoff: float) -> SeatTrajectory:
        return SeatTrajectory(
            obs=[np.zeros(4, dtype=np.float32)] * n,
            masks=[np.ones(3, dtype=np.int8)] * n,
            actions=[0] * n,
            logps=[0.0] * n,
            values=[0.0] * n,
            payoff=payoff,
        )

    batch = build_batch([traj(2, 1.0), traj(3, -1.0)], gamma=1.0, lam=1.0)
    assert len(batch) == 5
    # gamma=1, lam=1, values=0: every step's return equals the seat payoff.
    assert np.allclose(batch.returns, [1.0, 1.0, -1.0, -1.0, -1.0])
    assert np.allclose(batch.advantages, batch.returns)
    assert batch.obs.shape == (5, 4) and batch.masks.shape == (5, 3)
    assert batch.actions.dtype == np.int64


def test_build_batch_skips_empty_trajectories():
    empty = SeatTrajectory()
    full = SeatTrajectory(
        obs=[np.zeros(2, dtype=np.float32)],
        masks=[np.ones(2, dtype=np.int8)],
        actions=[1],
        logps=[-0.5],
        values=[0.2],
        payoff=0.7,
    )
    batch = build_batch([empty, full], gamma=1.0, lam=0.95)
    assert len(batch) == 1
    assert np.allclose(batch.returns, [0.7])
