"""Structured encoder arch: factory dispatch, training, persistence, retention."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import csv  # noqa: E402
import os  # noqa: E402

from buraco.env.env import BuracoEnv  # noqa: E402
from buraco.profiles import load_profile  # noqa: E402
from buraco.rl.agent import TorchAgent  # noqa: E402
from buraco.rl.config import TrainConfig  # noqa: E402
from buraco.rl.nets import build_net, net_config  # noqa: E402
from buraco.rl.obs import ObsSpec  # noqa: E402
from buraco.rl.pool import PoolManager, load_pool_member, save_pool_member  # noqa: E402
from buraco.rl.train import Trainer  # noqa: E402

TINY_STRUCTURED = TrainConfig(
    players=2,
    num_envs=2,
    min_steps_per_update=128,
    updates=2,
    minibatch=64,
    hidden=32,
    layers=1,
    arch="structured",
    embed_dim=16,
    eval_every=0,
    checkpoint_every=1,
    device="cpu",
    seed=0,
)


def test_build_net_dispatch_shapes_and_errors():
    cfg = load_profile("buraco", num_players=4)
    spec = ObsSpec.from_cfg(cfg)
    x = torch.randn(5, spec.flat_dim)
    for arch in ("mlp", "structured"):
        net = build_net(net_config(arch, spec, 1585, hidden=32, layers=1, embed_dim=16))
        logits, value = net(x)
        assert logits.shape == (5, 1585) and value.shape == (5,)
        assert torch.isfinite(logits).all() and torch.isfinite(value).all()
    with pytest.raises(ValueError, match="unknown arch"):
        net_config("transformer", spec, 1585, 32, 1)


def test_structured_smoke_train_resume_and_agent(tmp_path):
    run_dir = tmp_path / "run-structured"
    Trainer(TINY_STRUCTURED, run_dir).run()

    with open(run_dir / "metrics.csv") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 2
    for row in rows:
        for key in ("loss_pi", "loss_v", "entropy", "approx_kl"):
            assert float(row[key]) == float(row[key])  # finite, parseable

    latest = run_dir / "checkpoints" / "latest.pt"
    resumed = Trainer(
        TrainConfig(updates=3, device="cpu"), run_dir, resume=latest
    )
    assert resumed.cfg.arch == "structured"  # checkpoint defines the arch
    assert resumed.cfg.embed_dim == 16
    resumed.run()

    # The eval path rebuilds the structured net from the checkpoint alone.
    agent = TorchAgent.from_checkpoint(latest, greedy=True)
    cfg = load_profile("buraco", num_players=2)
    env = BuracoEnv(cfg)
    _, info = env.reset(seed=123)
    action = agent.act(env.observe_raw(info["to_play"]), info["legal_actions"], cfg)
    assert action in info["legal_actions"]


def test_structured_pool_member_roundtrip(tmp_path):
    cfg = load_profile("buraco", num_players=2)
    spec = ObsSpec.from_cfg(cfg)
    conf = net_config("structured", spec, 1585, hidden=32, layers=1, embed_dim=16)
    net = build_net(conf)
    save_pool_member(tmp_path / "m.pt", net, conf)
    loaded = load_pool_member(tmp_path / "m.pt")
    x = torch.randn(3, spec.flat_dim)
    assert torch.equal(net.eval()(x)[0], loaded(x)[0])


def test_legacy_pool_member_still_loads(tmp_path):
    """V2a files carry no config; MLP dims are recovered from shapes."""
    cfg = load_profile("buraco", num_players=2)
    spec = ObsSpec.from_cfg(cfg)
    net = build_net(net_config("mlp", spec, 1585, hidden=32, layers=1))
    state = {k: v.detach().cpu() for k, v in net.state_dict().items()}
    torch.save({"model": state}, tmp_path / "legacy.pt")
    loaded = load_pool_member(tmp_path / "legacy.pt")
    x = torch.randn(3, spec.flat_dim)
    assert torch.equal(net.eval()(x)[0], loaded(x)[0])


def test_spaced_retention_keeps_anchor_and_spread(tmp_path):
    net = build_net(net_config("mlp",
                    ObsSpec.from_cfg(load_profile("buraco", num_players=2)),
                    11, hidden=8, layers=1))
    pm = PoolManager(tmp_path / "pool", size=4, every=1, retention="spaced")
    for update in range(8):
        pm.snapshot(update, net)
    kept = [int(n.removeprefix("pool_").removesuffix(".pt")) for n in pm.names]
    assert kept[0] == 0  # oldest anchor survives
    assert kept[-1] == 7  # newest always survives
    assert kept == sorted(kept) and len(kept) == 4
    assert kept == [0, 2, 4, 7]
    pm.flush_evictions()
    assert sorted(os.listdir(tmp_path / "pool")) == sorted(pm.names)

    with pytest.raises(ValueError, match="retention"):
        PoolManager(tmp_path / "pool", size=4, every=1, retention="fifo")
