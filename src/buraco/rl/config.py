"""Training hyperparameters, serializable for run snapshots and checkpoints."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields


@dataclass(frozen=True)
class TrainConfig:
    profile: str = "buraco"
    players: int = 2
    num_envs: int = 16
    min_steps_per_update: int = 4096
    updates: int = 1000
    lr: float = 3e-4
    clip: float = 0.2
    epochs: int = 4
    minibatch: int = 512
    gamma: float = 1.0
    gae_lambda: float = 0.95
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    grad_clip: float = 0.5
    target_kl: float | None = 0.03
    hidden: int = 512
    layers: int = 2
    eval_every: int = 20
    eval_games: int = 100
    checkpoint_every: int = 20
    device: str = "auto"
    seed: int = 0
    history_len: int = 8
    trash_top_k: int = 8
    num_workers: int = 0  # 0 = in-process collection; N = process pool of N collectors
    # Opponent mixture: per-episode probability that the non-learner side is
    # played by frozen opponents instead of the current policy. Remaining mass
    # stays plain self-play. Both zero = exact v1 self-play behavior.
    opp_heuristic: float = 0.0  # scripted HeuristicAgent opponents
    opp_pool: float = 0.0  # frozen past-checkpoint opponents
    pool_every: int = 500  # snapshot cadence (updates) for the opponent pool
    pool_size: int = 10  # max pool members kept; oldest evicted

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> TrainConfig:
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})
