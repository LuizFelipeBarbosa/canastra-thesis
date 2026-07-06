"""TorchAgent conforms to the run_random_games agent protocol on 2p and 4p."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from buraco.env.env import BuracoEnv  # noqa: E402
from buraco.profiles import load_profile  # noqa: E402
from buraco.rl.agent import TorchAgent  # noqa: E402
from buraco.rl.nets import PolicyValueNet  # noqa: E402
from buraco.rl.obs import ObsSpec  # noqa: E402


def _play_episode(agent: TorchAgent, cfg, seed: int) -> BuracoEnv:
    env = BuracoEnv(cfg)
    _, info = env.reset(seed=seed)
    terminated = truncated = False
    while not (terminated or truncated):
        seat = info["to_play"]
        legal = info["legal_actions"]
        action = agent.act(env.observe_raw(seat), legal, cfg)
        assert action in legal
        _, _, terminated, truncated, info = env.step(action)
    return env


@pytest.mark.parametrize("players", [2, 4])
def test_full_episode_all_actions_legal(players):
    torch.manual_seed(0)
    cfg = load_profile("buraco", num_players=players)
    spec = ObsSpec.from_cfg(cfg)
    net = PolicyValueNet(spec.flat_dim, 1585, hidden=32, layers=1)
    agent = TorchAgent(net, spec, greedy=False, seed=0)
    env = _play_episode(agent, cfg, seed=11)
    assert len(env.action_log) > 0


def test_greedy_agent_is_deterministic():
    torch.manual_seed(0)
    cfg = load_profile("buraco", num_players=2)
    spec = ObsSpec.from_cfg(cfg)
    net = PolicyValueNet(spec.flat_dim, 1585, hidden=32, layers=1)
    agent = TorchAgent(net, spec, greedy=True)
    log_a = _play_episode(agent, cfg, seed=13).action_log
    log_b = _play_episode(agent, cfg, seed=13).action_log
    assert log_a == log_b
