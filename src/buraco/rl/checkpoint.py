"""Atomic checkpoints carrying everything a resume needs: weights, optimizer,
counters, both configs, the ObsSpec (feature-order guard), and RNG states."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from buraco.config import RulesConfig
from buraco.engine.serialize import config_to_dict
from buraco.rl.config import TrainConfig
from buraco.rl.obs import ObsSpec


@dataclass
class Checkpoint:
    model: dict[str, Any]
    optimizer: dict[str, Any]
    update: int
    global_env_steps: int
    global_episodes: int
    train_config: TrainConfig
    rules_config: dict[str, Any]
    obs_spec: ObsSpec
    rng: dict[str, Any]


def save_checkpoint(
    path: Path,
    net: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    update: int,
    global_env_steps: int,
    global_episodes: int,
    train_cfg: TrainConfig,
    rules_cfg: RulesConfig,
    obs_spec: ObsSpec,
    episode_counter: int,
) -> None:
    payload = {
        "model": net.state_dict(),
        "optimizer": optimizer.state_dict(),
        "update": update,
        "global_env_steps": global_env_steps,
        "global_episodes": global_episodes,
        "train_config": train_cfg.to_dict(),
        "rules_config": config_to_dict(rules_cfg),
        "obs_spec": obs_spec.to_dict(),
        "rng": {
            "python": random.getstate(),
            "torch_cpu": torch.get_rng_state(),
            "episode_counter": episode_counter,
        },
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: Path, map_location: str = "cpu") -> Checkpoint:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    return Checkpoint(
        model=payload["model"],
        optimizer=payload["optimizer"],
        update=payload["update"],
        global_env_steps=payload["global_env_steps"],
        global_episodes=payload["global_episodes"],
        train_config=TrainConfig.from_dict(payload["train_config"]),
        rules_config=payload["rules_config"],
        obs_spec=ObsSpec.from_dict(payload["obs_spec"]),
        rng=payload["rng"],
    )


def restore_rng(rng: dict[str, Any]) -> int:
    """Restore RNG states; returns the persisted episode-seed counter."""
    random.setstate(rng["python"])
    torch.set_rng_state(rng["torch_cpu"])
    return int(rng["episode_counter"])
