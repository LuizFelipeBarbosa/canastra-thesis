"""PolicyValueNet: shapes, finiteness, legal sampling on real observations."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import numpy as np  # noqa: E402

from buraco.env.env import BuracoEnv  # noqa: E402
from buraco.profiles import load_profile  # noqa: E402
from buraco.rl.nets import PolicyValueNet, masked_dist  # noqa: E402
from buraco.rl.obs import ObsSpec  # noqa: E402


def _batch_from_env(players: int, steps: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
    cfg = load_profile("buraco", num_players=players)
    spec = ObsSpec.from_cfg(cfg)
    env = BuracoEnv(cfg)
    rng = np.random.default_rng(0)
    obs, info = env.reset(seed=0)
    xs, masks = [], []
    for _ in range(steps):
        xs.append(spec.flatten(obs))
        masks.append(info["action_mask"].copy())
        action = int(rng.choice(info["legal_actions"]))
        obs, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset(seed=int(rng.integers(2**31)))
    return torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(masks))


@pytest.mark.parametrize("players", [2, 4])
def test_forward_shapes_and_finiteness(players):
    x, masks = _batch_from_env(players)
    net = PolicyValueNet(x.shape[1], masks.shape[1], hidden=64, layers=2)
    logits, value = net(x)
    assert logits.shape == (x.shape[0], masks.shape[1])
    assert value.shape == (x.shape[0],)
    assert torch.all(torch.isfinite(logits)) and torch.all(torch.isfinite(value))


def test_sampled_actions_always_legal():
    torch.manual_seed(0)
    x, masks = _batch_from_env(2, steps=100)
    net = PolicyValueNet(x.shape[1], masks.shape[1], hidden=32, layers=1)
    logits, _ = net(x)
    actions = masked_dist(logits, masks).sample()
    assert torch.all(masks.gather(1, actions.unsqueeze(1)).squeeze(1) == 1)


def test_near_uniform_initial_policy():
    torch.manual_seed(0)
    x, masks = _batch_from_env(2, steps=8)
    net = PolicyValueNet(x.shape[1], masks.shape[1])
    logits, _ = net(x)
    dist = masked_dist(logits, masks)
    n_legal = masks.sum(dim=1).float()
    # Entropy of the masked policy should be close to uniform-over-legal
    # (rows with a single legal action have entropy 0 by definition).
    multi = n_legal >= 2
    assert torch.all(dist.entropy()[multi] > 0.9 * torch.log(n_legal[multi]))
