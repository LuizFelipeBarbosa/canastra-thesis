"""Clipped-surrogate PPO update over one RolloutBatch."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from buraco.rl.buffer import RolloutBatch
from buraco.rl.config import TrainConfig
from buraco.rl.nets import PolicyValueNet, masked_dist


@dataclass
class PpoStats:
    loss_pi: float
    loss_v: float
    entropy: float
    approx_kl: float
    clip_frac: float
    grad_norm: float
    epochs_run: int


def ppo_update(
    net: PolicyValueNet,
    optimizer: torch.optim.Optimizer,
    batch: RolloutBatch,
    cfg: TrainConfig,
    device: torch.device,
) -> PpoStats:
    obs = torch.from_numpy(batch.obs).to(device)
    masks = torch.from_numpy(batch.masks).to(device)
    actions = torch.from_numpy(batch.actions).to(device)
    old_logps = torch.from_numpy(batch.logps).to(device)
    returns = torch.from_numpy(batch.returns).to(device)

    # Standardize advantages once per update batch (lower variance than
    # per-minibatch at these batch sizes).
    adv = batch.advantages
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    advantages = torch.from_numpy(adv).to(device)

    n = len(batch)
    stats: list[tuple[float, ...]] = []
    grad_norm = 0.0
    epochs_run = 0
    stop = False
    for _ in range(cfg.epochs):
        if stop:
            break
        epochs_run += 1
        # torch CPU RNG: seeded once per run and restored on resume.
        perm = torch.randperm(n)
        for start in range(0, n, cfg.minibatch):
            idx = perm[start : start + cfg.minibatch].to(device)
            logits, values = net(obs[idx])
            dist = masked_dist(logits, masks[idx])
            logps = dist.log_prob(actions[idx])
            ratio = torch.exp(logps - old_logps[idx])

            mb_adv = advantages[idx]
            surrogate = torch.min(
                ratio * mb_adv,
                torch.clamp(ratio, 1.0 - cfg.clip, 1.0 + cfg.clip) * mb_adv,
            )
            loss_pi = -surrogate.mean()
            loss_v = 0.5 * (values - returns[idx]).pow(2).mean()
            entropy = dist.entropy().mean()
            loss = loss_pi + cfg.value_coef * loss_v - cfg.entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
            )
            optimizer.step()

            with torch.no_grad():
                # http://joschu.net/blog/kl-approx.html (k3 estimator)
                log_ratio = logps - old_logps[idx]
                approx_kl = float(((torch.exp(log_ratio) - 1) - log_ratio).mean())
                clip_frac = float(((ratio - 1.0).abs() > cfg.clip).float().mean())
            stats.append(
                (loss_pi.item(), loss_v.item(), entropy.item(), approx_kl, clip_frac)
            )
            if cfg.target_kl is not None and approx_kl > cfg.target_kl:
                stop = True
                break

    means = np.mean(np.asarray(stats), axis=0)
    return PpoStats(
        loss_pi=float(means[0]),
        loss_v=float(means[1]),
        entropy=float(means[2]),
        approx_kl=float(means[3]),
        clip_frac=float(means[4]),
        grad_norm=grad_norm,
        epochs_run=epochs_run,
    )
