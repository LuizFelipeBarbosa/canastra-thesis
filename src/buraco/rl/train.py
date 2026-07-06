"""PPO self-play trainer CLI.

    uv run --group rl python -m buraco.rl.train --profile buraco --players 2 \
        --updates 200 --run-dir runs/ppo-2p --seed 0 --device cpu

Resume: --resume runs/ppo-2p/checkpoints/latest.pt (appends to the same CSVs).
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import replace
from pathlib import Path

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        f"missing RL dependencies ({exc}); install with: uv sync --group rl"
    ) from exc

import numpy as np

from buraco.engine.serialize import config_to_dict
from buraco.profiles import load_profile
from buraco.rl.agent import TorchAgent
from buraco.rl.buffer import build_batch
from buraco.rl.checkpoint import load_checkpoint, restore_rng, save_checkpoint
from buraco.rl.config import TrainConfig
from buraco.rl.evaluate import evaluate_vs
from buraco.rl.metrics import CsvLogger
from buraco.rl.nets import PolicyValueNet
from buraco.rl.obs import ObsSpec
from buraco.rl.ppo import ppo_update
from buraco.rl.rollout import SelfPlayCollector

METRIC_FIELDS = [
    "update", "env_steps", "global_env_steps", "episodes", "global_episodes",
    "ep_len_mean", "truncation_rate", "mean_abs_payoff", "loss_pi", "loss_v",
    "entropy", "approx_kl", "clip_frac", "grad_norm", "epochs_run",
    "steps_per_sec", "wall_s",
]
EVAL_FIELDS = [
    "update", "opponent", "games", "win_rate", "draw_rate", "mean_payoff",
    "mean_steps", "truncation_rate",
]


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    return torch.device(name)


class Trainer:
    def __init__(self, cfg: TrainConfig, run_dir: Path, resume: Path | None = None):
        ckpt = load_checkpoint(resume) if resume else None
        if ckpt is not None:
            # The checkpoint defines the run (profile/players/net/seed/...);
            # only per-invocation knobs come from the CLI.
            cfg = replace(ckpt.train_config, updates=cfg.updates, device=cfg.device)
        self.cfg = cfg
        self.run_dir = run_dir
        self.device = resolve_device(cfg.device)
        self.rules_cfg = load_profile(cfg.profile, num_players=cfg.players)
        if ckpt is not None and config_to_dict(self.rules_cfg) != ckpt.rules_config:
            raise SystemExit(
                "resume refused: rules config drifted from the checkpoint"
            )

        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

        self.spec = (
            ckpt.obs_spec
            if ckpt is not None
            else ObsSpec.from_cfg(self.rules_cfg, cfg.history_len, cfg.trash_top_k)
        )
        self.collector = SelfPlayCollector(
            self.rules_cfg,
            self.spec,
            num_envs=cfg.num_envs,
            seed=cfg.seed,
            history_len=cfg.history_len,
            trash_top_k=cfg.trash_top_k,
        )
        self.net = PolicyValueNet(
            self.spec.flat_dim,
            self.collector.envs[0].num_actions,
            hidden=cfg.hidden,
            layers=cfg.layers,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=cfg.lr, eps=1e-5)

        self.start_update = 0
        self.global_env_steps = 0
        self.global_episodes = 0
        if ckpt is not None:
            self.net.load_state_dict(ckpt.model)
            self.optimizer.load_state_dict(ckpt.optimizer)
            self.start_update = ckpt.update + 1
            self.global_env_steps = ckpt.global_env_steps
            self.global_episodes = ckpt.global_episodes
            self.collector.episode_counter = restore_rng(ckpt.rng)

        (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (run_dir / "config.json").write_text(
            json.dumps(
                {"train": cfg.to_dict(), "rules": config_to_dict(self.rules_cfg)},
                indent=2,
            )
        )
        self.metrics = CsvLogger(run_dir / "metrics.csv", METRIC_FIELDS)
        self.evals = CsvLogger(run_dir / "eval.csv", EVAL_FIELDS)

    def run(self) -> None:
        cfg = self.cfg
        for update in range(self.start_update, cfg.updates):
            start = time.perf_counter()
            trajs, roll = self.collector.collect(
                self.net, self.device, cfg.min_steps_per_update
            )
            batch = build_batch(trajs, cfg.gamma, cfg.gae_lambda)
            ppo = ppo_update(self.net, self.optimizer, batch, cfg, self.device)
            wall = time.perf_counter() - start
            self.global_env_steps += roll.env_steps
            self.global_episodes += roll.episodes

            self.metrics.log(
                {
                    "update": update,
                    "env_steps": roll.env_steps,
                    "global_env_steps": self.global_env_steps,
                    "episodes": roll.episodes,
                    "global_episodes": self.global_episodes,
                    "ep_len_mean": round(roll.mean_ep_len, 2),
                    "truncation_rate": round(roll.truncation_rate, 4),
                    "mean_abs_payoff": round(roll.mean_abs_payoff, 4),
                    "loss_pi": round(ppo.loss_pi, 6),
                    "loss_v": round(ppo.loss_v, 6),
                    "entropy": round(ppo.entropy, 4),
                    "approx_kl": round(ppo.approx_kl, 6),
                    "clip_frac": round(ppo.clip_frac, 4),
                    "grad_norm": round(ppo.grad_norm, 4),
                    "epochs_run": ppo.epochs_run,
                    "steps_per_sec": round(roll.steps_per_sec, 1),
                    "wall_s": round(wall, 2),
                }
            )
            print(
                f"update {update}: {roll.env_steps} steps, {roll.episodes} eps, "
                f"loss_pi {ppo.loss_pi:+.4f}, loss_v {ppo.loss_v:.4f}, "
                f"entropy {ppo.entropy:.3f}, kl {ppo.approx_kl:.4f}, "
                f"{roll.steps_per_sec:.0f} steps/s"
            )

            last = update == cfg.updates - 1
            if cfg.eval_every and (update % cfg.eval_every == 0 or last):
                self._evaluate(update)
            if cfg.checkpoint_every and (update % cfg.checkpoint_every == 0 or last):
                self._checkpoint(update)

    def _evaluate(self, update: int) -> None:
        agent = TorchAgent(
            self.net,
            self.spec,
            device=str(self.device),
            greedy=True,
            history_len=self.cfg.history_len,
            trash_top_k=self.cfg.trash_top_k,
        )
        for opponent in ("random", "heuristic"):
            result = evaluate_vs(
                agent, self.rules_cfg, opponent, self.cfg.eval_games, seed=self.cfg.seed
            )
            self.evals.log(
                {
                    "update": update,
                    "opponent": opponent,
                    "games": result.games,
                    "win_rate": round(result.win_rate, 4),
                    "draw_rate": round(result.draw_rate, 4),
                    "mean_payoff": round(result.mean_payoff, 4),
                    "mean_steps": round(result.mean_steps, 1),
                    "truncation_rate": round(result.truncation_rate, 4),
                }
            )
            print(f"  eval @ {update} {result}")
        self.net.train()

    def _checkpoint(self, update: int) -> None:
        args = (
            self.net,
            self.optimizer,
            update,
            self.global_env_steps,
            self.global_episodes,
            self.cfg,
            self.rules_cfg,
            self.spec,
            self.collector.episode_counter,
        )
        save_checkpoint(self.run_dir / "checkpoints" / f"ckpt_{update:06d}.pt", *args)
        save_checkpoint(self.run_dir / "checkpoints" / "latest.pt", *args)


def resolve_run_dir(run_dir: str | None, resume: Path | None, cfg: TrainConfig) -> Path:
    if run_dir:
        return Path(run_dir)
    if resume is not None:
        # Checkpoints live at <run_dir>/checkpoints/<name>.pt; append to that run.
        parent = resume.resolve().parent
        return parent.parent if parent.name == "checkpoints" else parent
    return Path(f"runs/{cfg.profile}{cfg.players}p-{time.strftime('%Y%m%d-%H%M%S')}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = TrainConfig()
    parser.add_argument("--profile", default=defaults.profile)
    parser.add_argument("--players", type=int, default=defaults.players)
    parser.add_argument("--updates", type=int, default=defaults.updates)
    parser.add_argument("--num-envs", type=int, default=defaults.num_envs)
    parser.add_argument("--min-steps", type=int, default=defaults.min_steps_per_update)
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument("--hidden", type=int, default=defaults.hidden)
    parser.add_argument("--layers", type=int, default=defaults.layers)
    parser.add_argument("--eval-every", type=int, default=defaults.eval_every)
    parser.add_argument("--eval-games", type=int, default=defaults.eval_games)
    parser.add_argument("--checkpoint-every", type=int, default=defaults.checkpoint_every)
    parser.add_argument("--device", default=defaults.device)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    cfg = TrainConfig(
        profile=args.profile,
        players=args.players,
        num_envs=args.num_envs,
        min_steps_per_update=args.min_steps,
        updates=args.updates,
        lr=args.lr,
        hidden=args.hidden,
        layers=args.layers,
        eval_every=args.eval_every,
        eval_games=args.eval_games,
        checkpoint_every=args.checkpoint_every,
        device=args.device,
        seed=args.seed,
    )
    resume = Path(args.resume) if args.resume else None
    run_dir = resolve_run_dir(args.run_dir, resume, cfg)
    Trainer(cfg, run_dir, resume=resume).run()


if __name__ == "__main__":
    main()
