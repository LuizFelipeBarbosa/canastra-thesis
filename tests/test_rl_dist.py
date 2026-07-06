"""Masked categorical distribution: legal-only probabilities, log-probs, entropy."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from buraco.rl.nets import masked_dist  # noqa: E402


def test_illegal_actions_get_zero_probability():
    logits = torch.randn(5, 10)
    mask = torch.zeros(5, 10, dtype=torch.int8)
    mask[:, [1, 4, 7]] = 1
    dist = masked_dist(logits, mask)
    probs = dist.probs
    assert torch.all(probs[:, [0, 2, 3, 5, 6, 8, 9]] == 0.0)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(5))


def test_log_prob_matches_manual_legal_renormalization():
    logits = torch.randn(10)
    legal = [2, 5, 6]
    mask = torch.zeros(10, dtype=torch.int8)
    mask[legal] = 1
    dist = masked_dist(logits, mask)
    manual = torch.log_softmax(logits[legal], dim=-1)
    for i, a in enumerate(legal):
        assert torch.allclose(dist.log_prob(torch.tensor(a)), manual[i], atol=1e-5)


def test_entropy_matches_legal_only_distribution():
    logits = torch.randn(10)
    legal = [0, 3, 9]
    mask = torch.zeros(10, dtype=torch.int8)
    mask[legal] = 1
    dist = masked_dist(logits, mask)
    legal_dist = torch.distributions.Categorical(logits=logits[legal])
    assert torch.allclose(dist.entropy(), legal_dist.entropy(), atol=1e-5)


def test_single_legal_action_prob_one_entropy_zero_finite_grads():
    logits = torch.randn(6, requires_grad=True)
    mask = torch.zeros(6, dtype=torch.int8)
    mask[3] = 1
    dist = masked_dist(logits, mask)
    assert torch.allclose(dist.probs[3], torch.tensor(1.0))
    assert torch.allclose(dist.entropy(), torch.tensor(0.0), atol=1e-6)
    logp = dist.log_prob(torch.tensor(3))
    logp.backward()
    assert torch.all(torch.isfinite(logits.grad))


def test_sampling_always_legal():
    gen = torch.Generator().manual_seed(0)
    for _ in range(50):
        logits = torch.randn(20, generator=gen)
        mask = torch.zeros(20, dtype=torch.int8)
        legal = torch.randperm(20, generator=gen)[: 1 + int(torch.randint(19, (1,), generator=gen))]
        mask[legal] = 1
        a = masked_dist(logits, mask).sample()
        assert mask[a] == 1
