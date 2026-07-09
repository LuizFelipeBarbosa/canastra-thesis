"""Opponent mixture and pool: assignment, learner-only recording, persistence."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import numpy as np  # noqa: E402

from buraco.profiles import load_profile  # noqa: E402
from buraco.rl.buffer import build_batch  # noqa: E402
from buraco.rl.checkpoint import load_checkpoint, save_checkpoint  # noqa: E402
from buraco.rl.config import TrainConfig  # noqa: E402
from buraco.rl.nets import PolicyValueNet, net_config  # noqa: E402
from buraco.rl.obs import ObsSpec  # noqa: E402
from buraco.rl.parallel import ParallelCollector  # noqa: E402
from buraco.rl.pool import (  # noqa: E402
    OpponentMixture,
    PoolManager,
    load_pool_member,
    save_pool_member,
)
from buraco.rl.rollout import SelfPlayCollector  # noqa: E402

NUM_ACTIONS = 1585


def _setup(players: int):
    cfg = load_profile("buraco", num_players=players)
    spec = ObsSpec.from_cfg(cfg)
    net = PolicyValueNet(spec.flat_dim, NUM_ACTIONS, hidden=32, layers=1)
    return cfg, spec, net


def test_mixture_validates_probabilities():
    with pytest.raises(ValueError, match="sum to <= 1"):
        OpponentMixture(0.7, 0.7)
    with pytest.raises(ValueError, match="sum to <= 1"):
        OpponentMixture(-0.1, 0.0)
    assert not OpponentMixture(0.0, 0.0).enabled
    assert OpponentMixture(0.5, 0.0).enabled


def test_assignment_is_deterministic_and_side_swaps():
    cfg = load_profile("buraco", num_players=4)
    mix = OpponentMixture(p_heuristic=1.0)

    a1 = mix.assign(123, 0, cfg)
    a2 = mix.assign(123, 0, cfg)
    assert a1.net_seats == a2.net_seats
    assert set(a1.scripted) == set(a2.scripted)
    assert not a1.frozen

    # Learner side follows episode-counter parity; each side is one team.
    even = mix.assign(500, 0, cfg)
    odd = mix.assign(501, 1, cfg)
    assert even.net_seats != odd.net_seats
    assert even.net_seats | odd.net_seats == frozenset(range(4))
    for assign in (even, odd):
        assert set(assign.scripted) == set(range(4)) - assign.net_seats
        assert assign.learner_seat in assign.net_seats


def test_disabled_mixture_matches_plain_self_play():
    cfg, spec, net = _setup(2)

    torch.manual_seed(11)
    plain = SelfPlayCollector(cfg, spec, num_envs=2, seed=0)
    trajs_a, stats_a = plain.collect(net, torch.device("cpu"), 120)

    torch.manual_seed(11)
    zeroed = SelfPlayCollector(
        cfg, spec, num_envs=2, seed=0, mixture=OpponentMixture(0.0, 0.0)
    )
    trajs_b, stats_b = zeroed.collect(net, torch.device("cpu"), 120)

    assert zeroed.mixture is None  # disabled mixture is normalized away
    assert stats_a.env_steps == stats_b.env_steps
    assert stats_b.recorded_steps == stats_b.env_steps
    assert stats_b.mixed_episodes == 0
    batch_a = build_batch(trajs_a, gamma=1.0, lam=0.95)
    batch_b = build_batch(trajs_b, gamma=1.0, lam=0.95)
    assert np.array_equal(batch_a.obs, batch_b.obs)
    assert np.array_equal(batch_a.actions, batch_b.actions)
    assert np.array_equal(batch_a.logps, batch_b.logps)


def test_heuristic_mixture_records_learner_side_only():
    cfg, spec, net = _setup(4)
    torch.manual_seed(3)
    collector = SelfPlayCollector(
        cfg, spec, num_envs=2, seed=1, mixture=OpponentMixture(p_heuristic=1.0)
    )
    trajs, stats = collector.collect(net, torch.device("cpu"), 150)

    assert stats.mixed_episodes == stats.episodes
    assert 0 <= stats.mixed_wins <= stats.mixed_episodes
    # Only the learner team (2 of 4 seats) is recorded.
    assert len(trajs) <= stats.episodes * 2
    assert stats.recorded_steps < stats.env_steps
    assert sum(len(t) for t in trajs) == stats.recorded_steps
    batch = build_batch(trajs, gamma=1.0, lam=0.95)
    assert len(batch) == stats.recorded_steps
    assert np.all(np.isfinite(batch.advantages))


def test_pool_member_roundtrip(tmp_path):
    net = PolicyValueNet(37, 11, hidden=16, layers=2)
    path = tmp_path / "member.pt"
    save_pool_member(path, net)
    loaded = load_pool_member(path)

    assert not loaded.training  # frozen opponents stay in eval mode
    for key, tensor in net.state_dict().items():
        assert torch.equal(loaded.state_dict()[key], tensor)


def test_pool_mixture_plays_frozen_members(tmp_path):
    cfg, spec, net = _setup(2)
    member = tmp_path / "pool_000000.pt"
    save_pool_member(member, PolicyValueNet(spec.flat_dim, NUM_ACTIONS, hidden=32, layers=1))

    mix = OpponentMixture(p_pool=1.0)
    mix.set_manifest([str(member)])
    torch.manual_seed(4)
    collector = SelfPlayCollector(cfg, spec, num_envs=2, seed=2, mixture=mix)
    trajs, stats = collector.collect(net, torch.device("cpu"), 100)

    assert stats.mixed_episodes == stats.episodes
    assert len(trajs) <= stats.episodes  # 2p: learner side is a single seat
    assert stats.recorded_steps < stats.env_steps
    assert sum(len(t) for t in trajs) == stats.recorded_steps


def test_empty_pool_mass_folds_to_self_play():
    cfg, spec, net = _setup(2)
    torch.manual_seed(5)
    collector = SelfPlayCollector(
        cfg, spec, num_envs=2, seed=3, mixture=OpponentMixture(p_pool=1.0)
    )
    _, stats = collector.collect(net, torch.device("cpu"), 100)

    assert stats.mixed_episodes == 0
    assert stats.recorded_steps == stats.env_steps


def test_pool_manager_cadence_and_eviction(tmp_path):
    net = PolicyValueNet(8, 5, hidden=8, layers=1)
    pm = PoolManager(tmp_path / "pool", size=2, every=10)

    assert pm.maybe_snapshot(0, net)
    assert not pm.maybe_snapshot(5, net)
    assert pm.maybe_snapshot(10, net)
    assert pm.maybe_snapshot(20, net)

    assert pm.names == ["pool_000010.pt", "pool_000020.pt"]
    # Eviction is deferred: the newest on-disk checkpoint may still reference
    # the member until the next checkpoint persists the shrunk manifest.
    assert (tmp_path / "pool" / "pool_000000.pt").exists()
    pm.flush_evictions()
    assert not (tmp_path / "pool" / "pool_000000.pt").exists()
    assert all((tmp_path / "pool" / n).exists() for n in pm.names)

    # Re-snapshotting the same update (resume) does not duplicate the member.
    pm.snapshot(20, net)
    assert pm.names == ["pool_000010.pt", "pool_000020.pt"]

    restored = PoolManager(tmp_path / "pool", size=2, every=10)
    restored.restore(pm.names)
    assert restored.paths == pm.paths
    with pytest.raises(SystemExit, match="missing"):
        restored.restore(["pool_999999.pt"])


def test_checkpoint_roundtrips_pool_manifest(tmp_path):
    cfg = load_profile("buraco", num_players=2)
    spec = ObsSpec.from_cfg(cfg)
    net = PolicyValueNet(spec.flat_dim, NUM_ACTIONS, hidden=8, layers=1)
    optimizer = torch.optim.Adam(net.parameters())
    args = (net, optimizer, 7, 100, 10, TrainConfig(), cfg, spec, 42)

    with_pool = tmp_path / "with_pool.pt"
    save_checkpoint(with_pool, *args, pool_manifest=["pool_000000.pt"])
    assert load_checkpoint(with_pool).pool_manifest == ["pool_000000.pt"]

    without = tmp_path / "without.pt"
    save_checkpoint(without, *args)
    assert load_checkpoint(without).pool_manifest is None


def test_parallel_collector_with_mixture(tmp_path):
    cfg, spec, net = _setup(2)
    member = tmp_path / "pool_000000.pt"
    save_pool_member(member, PolicyValueNet(spec.flat_dim, NUM_ACTIONS, hidden=32, layers=1))

    collector = ParallelCollector(
        cfg, spec, num_envs=2, seed=0, num_workers=1,
        net_config=net_config("mlp", spec, NUM_ACTIONS, hidden=32, layers=1),
        p_heuristic=0.5, p_pool=0.5,
    )
    try:
        collector.set_pool_manifest([str(member)])
        trajs, stats = collector.collect(net, torch.device("cpu"), min_steps=120)
    finally:
        collector.close()

    assert stats.recorded_steps >= 120
    assert stats.mixed_episodes > 0  # p=1.0 combined: every episode is mixed
    assert stats.mixed_episodes == stats.episodes
    assert sum(len(t) for t in trajs) == stats.recorded_steps
    assert stats.recorded_steps < stats.env_steps
