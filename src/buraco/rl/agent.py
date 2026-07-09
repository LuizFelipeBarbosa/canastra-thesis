"""Learned agent conforming to the `act(raw_obs, legal_ids, cfg)` protocol used
by examples/run_random_games.py, so checkpoints slot into the existing harness."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from buraco.config import RulesConfig
from buraco.env.encoding import encode_observation
from buraco.rl.nets import masked_dist
from buraco.rl.obs import ObsSpec


class TorchAgent:
    def __init__(
        self,
        net: torch.nn.Module,
        spec: ObsSpec,
        device: str = "cpu",
        greedy: bool = True,
        seed: int | None = None,
        history_len: int = 8,
        trash_top_k: int = 8,
    ) -> None:
        self.net = net.to(device).eval()
        self.spec = spec
        self.device = torch.device(device)
        self.greedy = greedy
        self.history_len = history_len
        self.trash_top_k = trash_top_k
        self._gen = torch.Generator()
        if seed is not None:
            self._gen.manual_seed(seed)

    def act(self, raw_obs: dict[str, Any], legal_ids: list[int], cfg: RulesConfig) -> int:
        obs = encode_observation(raw_obs, cfg, self.history_len, self.trash_top_k)
        flat = torch.from_numpy(self.spec.flatten(obs)).unsqueeze(0).to(self.device)
        num_actions = self.net.policy_head.out_features
        mask = np.zeros(num_actions, dtype=np.int8)
        mask[legal_ids] = 1
        mask_t = torch.from_numpy(mask).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, _ = self.net(flat)
            dist = masked_dist(logits, mask_t)
            if self.greedy:
                action = int(dist.probs.argmax(dim=-1))
            else:
                probs = dist.probs.squeeze(0).cpu()
                action = int(torch.multinomial(probs, 1, generator=self._gen))
        return action

    @classmethod
    def from_checkpoint(
        cls, path: Path | str, device: str = "cpu", greedy: bool = True,
        seed: int | None = None,
    ) -> TorchAgent:
        from buraco.rl.checkpoint import load_checkpoint
        from buraco.rl.nets import build_net, net_config

        ckpt = load_checkpoint(Path(path), map_location=device)
        train_cfg = ckpt.train_config  # from_dict defaults arch="mlp" for old files
        net = build_net(
            net_config(
                train_cfg.arch,
                ckpt.obs_spec,
                ckpt.model["policy_head.weight"].shape[0],
                train_cfg.hidden,
                train_cfg.layers,
                embed_dim=train_cfg.embed_dim,
            )
        )
        net.load_state_dict(ckpt.model)
        return cls(
            net,
            ckpt.obs_spec,
            device=device,
            greedy=greedy,
            seed=seed,
            history_len=train_cfg.history_len,
            trash_top_k=train_cfg.trash_top_k,
        )
