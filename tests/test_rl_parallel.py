"""ParallelCollector: accounting, determinism, counter residues, migration."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import numpy as np  # noqa: E402

from buraco.profiles import load_profile  # noqa: E402
from buraco.rl.buffer import build_batch  # noqa: E402
from buraco.rl.checkpoint import migrate_counters  # noqa: E402
from buraco.rl.nets import PolicyValueNet, net_config  # noqa: E402
from buraco.rl.obs import ObsSpec  # noqa: E402
from buraco.rl.parallel import ParallelCollector  # noqa: E402

WORKERS = 2


def _net_config(spec):
    return net_config("mlp", spec, 1585, hidden=32, layers=1)


@pytest.fixture(scope="module")
def pool():
    """One spawn pool for the module: startup dominates these tests' cost."""
    cfg = load_profile("buraco", num_players=2)
    spec = ObsSpec.from_cfg(cfg)
    collector = ParallelCollector(
        cfg, spec, num_envs=4, seed=0, num_workers=WORKERS,
        net_config=_net_config(spec),
    )
    net = PolicyValueNet(spec.flat_dim, 1585, hidden=32, layers=1)
    yield collector, net
    collector.close()


def test_collect_accounting_and_zero_sum(pool):
    collector, net = pool
    players = 2
    trajs, stats = collector.collect(net, torch.device("cpu"), min_steps=200)

    assert stats.env_steps >= 200
    assert stats.episodes >= WORKERS  # every worker finishes its in-flight episodes
    assert len(trajs) == stats.episodes * players
    assert sum(len(t) for t in trajs) == stats.env_steps
    # Slot-ordered concatenation preserves per-episode grouping.
    for e in range(stats.episodes):
        group = trajs[e * players : (e + 1) * players]
        assert abs(sum(t.payoff for t in group)) < 1e-9
    batch = build_batch(trajs, gamma=1.0, lam=0.95)
    assert len(batch) == stats.env_steps
    assert np.all(np.isfinite(batch.advantages))


def test_counters_stay_in_slot_residue_classes(pool):
    collector, net = pool
    collector.collect(net, torch.device("cpu"), min_steps=150)
    assert len(collector.counters) == WORKERS
    for slot, counter in enumerate(collector.counters):
        assert counter % WORKERS == slot
    assert collector.episode_counter == max(collector.counters)


def test_collect_is_deterministic_for_fixed_counters(pool):
    collector, net = pool
    saved = list(collector.counters)

    collector.counters = list(range(WORKERS))
    trajs_a, stats_a = collector.collect(net, torch.device("cpu"), min_steps=150)
    collector.counters = list(range(WORKERS))
    trajs_b, stats_b = collector.collect(net, torch.device("cpu"), min_steps=150)
    collector.counters = saved

    assert stats_a.env_steps == stats_b.env_steps
    assert stats_a.episodes == stats_b.episodes
    batch_a = build_batch(trajs_a, gamma=1.0, lam=0.95)
    batch_b = build_batch(trajs_b, gamma=1.0, lam=0.95)
    assert np.array_equal(batch_a.obs, batch_b.obs)
    assert np.array_equal(batch_a.masks, batch_b.masks)
    assert np.array_equal(batch_a.actions, batch_b.actions)
    assert np.array_equal(batch_a.logps, batch_b.logps)


def test_double_close_is_safe():
    cfg = load_profile("buraco", num_players=2)
    spec = ObsSpec.from_cfg(cfg)
    collector = ParallelCollector(
        cfg, spec, num_envs=2, seed=1, num_workers=1,
        net_config=_net_config(spec),
    )
    collector.close()
    collector.close()
    with pytest.raises(AssertionError):
        collector.collect(
            PolicyValueNet(spec.flat_dim, 1585, hidden=32, layers=1),
            torch.device("cpu"),
            10,
        )


def test_rejects_env_topology_that_would_be_silently_rounded():
    """num_envs must split exactly across workers (Codex adversarial review)."""
    cfg = load_profile("buraco", num_players=2)
    spec = ObsSpec.from_cfg(cfg)
    # Non-divisible: 17 envs over 4 workers would silently run 16.
    with pytest.raises(ValueError, match="multiple of"):
        ParallelCollector(cfg, spec, num_envs=17, seed=0, num_workers=4,
                          net_config=_net_config(spec))
    # More workers than envs: 2 envs over 8 workers would silently run 8.
    with pytest.raises(ValueError, match="multiple of"):
        ParallelCollector(cfg, spec, num_envs=2, seed=0, num_workers=8,
                          net_config=_net_config(spec))


def test_migrate_counters_rules():
    # serial -> serial: pass-through of the scalar.
    assert migrate_counters({"episode_counter": 7}, 0) == 7
    # serial -> W slots: smallest value >= C in each residue class.
    assert migrate_counters({"episode_counter": 5}, 4) == [8, 5, 6, 7]
    assert migrate_counters({"episode_counter": 0}, 2) == [0, 1]
    # parallel -> same W: exact pass-through.
    rng = {"episode_counter": 8, "episode_counters": [8, 5, 6, 7], "num_workers": 4}
    assert migrate_counters(rng, 4) == [8, 5, 6, 7]
    # parallel -> serial: resume at the high-water mark.
    assert migrate_counters(rng, 0) == 8
    # parallel -> different W: rebuild from the high-water mark.
    assert migrate_counters(rng, 3) == [9, 10, 8]
    # every migration target is >= the high-water mark of consumed seeds
    for w in (1, 2, 3, 5, 8):
        counters = migrate_counters(rng, w)
        assert all(c >= 8 for c in counters)
        assert sorted(c % w for c in counters) == list(range(w))
