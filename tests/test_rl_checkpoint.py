"""Checkpoint save/load round-trip and agent restoration."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import numpy as np  # noqa: E402

from buraco.env.env import BuracoEnv  # noqa: E402
from buraco.profiles import load_profile  # noqa: E402
from buraco.rl.agent import TorchAgent  # noqa: E402
from buraco.rl.checkpoint import load_checkpoint, save_checkpoint  # noqa: E402
from buraco.rl.config import TrainConfig  # noqa: E402
from buraco.rl.nets import PolicyValueNet  # noqa: E402
from buraco.rl.obs import ObsSpec  # noqa: E402


def _greedy_action_log(agent: TorchAgent, cfg, seed: int = 5, steps: int = 40) -> list[int]:
    env = BuracoEnv(cfg)
    _, info = env.reset(seed=seed)
    for _ in range(steps):
        seat = info["to_play"]
        action = agent.act(env.observe_raw(seat), info["legal_actions"], cfg)
        _, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    return list(env.action_log)


def test_checkpoint_round_trip(tmp_path):
    torch.manual_seed(0)
    cfg = load_profile("buraco", num_players=2)
    train_cfg = TrainConfig(hidden=32, layers=1)
    spec = ObsSpec.from_cfg(cfg)
    net = PolicyValueNet(spec.flat_dim, 1585, hidden=32, layers=1)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    # One optimizer step so its state is non-trivial.
    logits, value = net(torch.randn(4, spec.flat_dim))
    (logits.sum() + value.sum()).backward()
    optimizer.step()

    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, net, optimizer, 7, 1234, 56, train_cfg, cfg, spec, 99)
    ckpt = load_checkpoint(path)

    assert ckpt.update == 7
    assert ckpt.global_env_steps == 1234
    assert ckpt.global_episodes == 56
    assert ckpt.train_config == train_cfg
    assert ckpt.obs_spec == spec
    assert ckpt.rng["episode_counter"] == 99
    for name, tensor in net.state_dict().items():
        assert torch.equal(ckpt.model[name], tensor)
    reloaded = PolicyValueNet(spec.flat_dim, 1585, hidden=32, layers=1)
    reloaded.load_state_dict(ckpt.model)
    opt2 = torch.optim.Adam(reloaded.parameters(), lr=1e-3)
    opt2.load_state_dict(ckpt.optimizer)
    assert opt2.state_dict()["param_groups"] == optimizer.state_dict()["param_groups"]


def test_from_checkpoint_reproduces_greedy_actions(tmp_path):
    torch.manual_seed(1)
    cfg = load_profile("buraco", num_players=2)
    train_cfg = TrainConfig(hidden=32, layers=1)
    spec = ObsSpec.from_cfg(cfg)
    net = PolicyValueNet(spec.flat_dim, 1585, hidden=32, layers=1)
    optimizer = torch.optim.Adam(net.parameters())
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, net, optimizer, 0, 0, 0, train_cfg, cfg, spec, 0)

    before = _greedy_action_log(TorchAgent(net, spec), cfg)
    after = _greedy_action_log(TorchAgent.from_checkpoint(path), cfg)
    assert before == after and len(before) > 0


def test_checkpoint_write_is_atomic(tmp_path):
    cfg = load_profile("buraco", num_players=2)
    spec = ObsSpec.from_cfg(cfg)
    net = PolicyValueNet(spec.flat_dim, 1585, hidden=32, layers=1)
    optimizer = torch.optim.Adam(net.parameters())
    path = tmp_path / "latest.pt"
    for update in range(2):  # second save overwrites via os.replace
        save_checkpoint(path, net, optimizer, update, 0, 0, TrainConfig(), cfg, spec, 0)
    assert load_checkpoint(path).update == 1
    assert not path.with_suffix(".pt.tmp").exists()


def test_flatten_batch_matches_single(tmp_path):
    cfg = load_profile("buraco", num_players=2)
    spec = ObsSpec.from_cfg(cfg)
    env = BuracoEnv(cfg)
    obs, _ = env.reset(seed=2)
    single = spec.flatten(obs)
    batch = spec.flatten_batch([obs, obs])
    assert np.array_equal(batch[0], single) and np.array_equal(batch[1], single)
