"""PPO self-play trainer CLI.

    uv run --group rl python -m buraco.rl.train --profile buraco --players 2 \
        --updates 200 --run-dir runs/ppo-2p --seed 0 --device cpu

Resume: --resume runs/ppo-2p/checkpoints/latest.pt (appends to the same CSVs).

Opponent mixture (V2a): --opp-heuristic / --opp-pool seat frozen opponents on
the non-learner side for that fraction of episodes (see rl/pool.py). On resume
these four pool flags override the checkpoint config when passed explicitly,
so a plain self-play checkpoint can be fine-tuned against a mixture:

    ... --resume runs/ppo-4p/checkpoints/latest.pt --run-dir runs/ppo-4p-mix \
        --opp-heuristic 0.25 --opp-pool 0.25

Prefer a fresh --run-dir when enabling the mixture on an old run: the mixture
adds metrics.csv columns, and appending wider rows to a v1 CSV misaligns it.
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

from buraco.engine.actions import action_space_size
from buraco.engine.serialize import config_to_dict
from buraco.profiles import load_profile
from buraco.rl.agent import TorchAgent
from buraco.rl.buffer import build_batch
from buraco.rl.checkpoint import (
    load_checkpoint,
    migrate_counters,
    restore_rng,
    save_checkpoint,
)
from buraco.rl.config import TrainConfig
from buraco.rl.evaluate import evaluate_vs
from buraco.rl.metrics import CsvLogger
from buraco.rl.nets import build_net, net_config
from buraco.rl.obs import ObsSpec
from buraco.rl.parallel import ParallelCollector
from buraco.rl.pool import OpponentMixture, PoolManager
from buraco.rl.ppo import ppo_update
from buraco.rl.rollout import SelfPlayCollector

METRIC_FIELDS = [
    "update", "env_steps", "global_env_steps", "episodes", "global_episodes",
    "ep_len_mean", "truncation_rate", "mean_abs_payoff", "loss_pi", "loss_v",
    "entropy", "approx_kl", "clip_frac", "grad_norm", "epochs_run",
    "steps_per_sec", "wall_s",
]
MIXTURE_FIELDS = ["recorded_steps", "mixed_episodes", "mixed_win_rate"]
EVAL_FIELDS = [
    "update", "opponent", "games", "win_rate", "draw_rate", "mean_payoff",
    "mean_steps", "truncation_rate",
]


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    return torch.device(name)


def _existing_run_artifacts(run_dir: Path) -> list[str]:
    found = [name for name in ("metrics.csv", "eval.csv") if (run_dir / name).exists()]
    for sub in ("checkpoints", "pool"):
        if (run_dir / sub).is_dir() and any((run_dir / sub).iterdir()):
            found.append(f"{sub}/")
    return found


class Trainer:
    def __init__(
        self,
        cfg: TrainConfig,
        run_dir: Path,
        resume: Path | None = None,
        resume_overrides: dict | None = None,
    ):
        if resume is None and (existing := _existing_run_artifacts(run_dir)):
            # A fresh run would append to the old CSVs and overwrite checkpoints.
            raise SystemExit(
                f"refusing fresh run: {run_dir} already contains "
                f"{', '.join(existing)}; continue it with "
                f"--resume {run_dir / 'checkpoints' / 'latest.pt'} "
                "or pick a new --run-dir"
            )
        ckpt = load_checkpoint(resume) if resume else None
        if ckpt is not None:
            # The checkpoint defines the run (profile/players/net/seed/...);
            # only per-invocation knobs come from the CLI. Explicitly passed
            # pool flags and --num-envs also override, so a self-play
            # checkpoint can be fine-tuned against an opponent mixture on a
            # different env topology.
            cfg = replace(
                ckpt.train_config,
                updates=cfg.updates,
                device=cfg.device,
                num_workers=cfg.num_workers,
                **(resume_overrides or {}),
            )
        self.cfg = cfg
        self.run_dir = run_dir
        self.device = resolve_device(cfg.device)
        try:
            self.mixture = OpponentMixture(cfg.opp_heuristic, cfg.opp_pool)
        except ValueError as exc:
            raise SystemExit(str(exc)) from None
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
        if cfg.num_workers > 0 and (
            cfg.num_envs < cfg.num_workers or cfg.num_envs % cfg.num_workers
        ):
            raise SystemExit(
                f"num_envs ({cfg.num_envs}) must be a positive multiple of "
                f"--num-workers ({cfg.num_workers}); pick a compatible worker count"
            )
        try:
            self.net_config = net_config(
                cfg.arch,
                self.spec,
                action_space_size(self.rules_cfg.meld.max_meld_slots),
                cfg.hidden,
                cfg.layers,
                embed_dim=cfg.embed_dim,
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from None
        if cfg.num_workers > 0:
            self.collector: SelfPlayCollector | ParallelCollector = ParallelCollector(
                self.rules_cfg,
                self.spec,
                num_envs=cfg.num_envs,
                seed=cfg.seed,
                num_workers=cfg.num_workers,
                net_config=self.net_config,
                history_len=cfg.history_len,
                trash_top_k=cfg.trash_top_k,
                p_heuristic=cfg.opp_heuristic,
                p_pool=cfg.opp_pool,
            )
        else:
            self.collector = SelfPlayCollector(
                self.rules_cfg,
                self.spec,
                num_envs=cfg.num_envs,
                seed=cfg.seed,
                history_len=cfg.history_len,
                trash_top_k=cfg.trash_top_k,
                mixture=self.mixture,
            )
        self.net = build_net(self.net_config).to(self.device)
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
            restore_rng(ckpt.rng)
            counters = migrate_counters(ckpt.rng, cfg.num_workers)
            if isinstance(self.collector, ParallelCollector):
                assert isinstance(counters, list)
                self.collector.counters = counters
            else:
                assert isinstance(counters, int)
                self.collector.episode_counter = counters

        try:
            self.pool = (
                PoolManager(
                    run_dir / "pool",
                    cfg.pool_size,
                    cfg.pool_every,
                    net_config=self.net_config,
                    retention=cfg.pool_retention,
                )
                if cfg.opp_pool > 0
                else None
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from None
        if self.pool is not None:
            if ckpt is not None and ckpt.pool_manifest:
                self.pool.restore(ckpt.pool_manifest)
            if not self.pool.names:
                # Seed with the starting policy so pool episodes exist from the
                # first update (fresh runs and v1-checkpoint fine-tunes alike).
                self.pool.snapshot(self.start_update, self.net)
            self.collector.set_pool_manifest(self.pool.paths)

        (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (run_dir / "config.json").write_text(
            json.dumps(
                {"train": cfg.to_dict(), "rules": config_to_dict(self.rules_cfg)},
                indent=2,
            )
        )
        metric_fields = METRIC_FIELDS + (MIXTURE_FIELDS if self.mixture.enabled else [])
        self.metrics = CsvLogger(run_dir / "metrics.csv", metric_fields)
        self.evals = CsvLogger(run_dir / "eval.csv", EVAL_FIELDS)

    def run(self) -> None:
        cfg = self.cfg
        for update in range(self.start_update, cfg.updates):
            start = time.perf_counter()
            if self.pool is not None and self.pool.maybe_snapshot(update, self.net):
                self.collector.set_pool_manifest(self.pool.paths)
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
                    "recorded_steps": roll.recorded_steps,
                    "mixed_episodes": roll.mixed_episodes,
                    "mixed_win_rate": round(roll.mixed_wins / roll.mixed_episodes, 4)
                    if roll.mixed_episodes
                    else "",
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
        counters = getattr(self.collector, "counters", None)
        manifest = self.pool.names if self.pool is not None else None
        save_checkpoint(
            self.run_dir / "checkpoints" / f"ckpt_{update:06d}.pt",
            *args,
            episode_counters=counters,
            pool_manifest=manifest,
        )
        save_checkpoint(
            self.run_dir / "checkpoints" / "latest.pt",
            *args,
            episode_counters=counters,
            pool_manifest=manifest,
        )
        if self.pool is not None:
            self.pool.flush_evictions()

    def close(self) -> None:
        close = getattr(self.collector, "close", None)
        if close is not None:
            close()


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
    # default=None marks "not passed" so resume can fall back to the
    # checkpoint's env topology (see the pool flags below for the pattern).
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--min-steps", type=int, default=defaults.min_steps_per_update)
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument("--hidden", type=int, default=defaults.hidden)
    parser.add_argument("--layers", type=int, default=defaults.layers)
    parser.add_argument("--arch", choices=("mlp", "structured"), default=defaults.arch)
    parser.add_argument("--embed-dim", type=int, default=defaults.embed_dim)
    parser.add_argument("--eval-every", type=int, default=defaults.eval_every)
    parser.add_argument("--eval-games", type=int, default=defaults.eval_games)
    parser.add_argument("--checkpoint-every", type=int, default=defaults.checkpoint_every)
    parser.add_argument("--device", default=defaults.device)
    parser.add_argument("--num-workers", type=int, default=defaults.num_workers)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--resume", default=None)
    # default=None marks "not passed": on resume these fall back to the
    # checkpoint config instead of silently disabling a pool run's mixture.
    parser.add_argument("--opp-heuristic", type=float, default=None)
    parser.add_argument("--opp-pool", type=float, default=None)
    parser.add_argument("--pool-every", type=int, default=None)
    parser.add_argument("--pool-size", type=int, default=None)
    parser.add_argument(
        "--pool-retention", choices=("recent", "spaced"), default=None
    )
    args = parser.parse_args()

    resume_overrides = {
        name: value
        for name, value in (
            ("num_envs", args.num_envs),
            ("opp_heuristic", args.opp_heuristic),
            ("opp_pool", args.opp_pool),
            ("pool_every", args.pool_every),
            ("pool_size", args.pool_size),
            ("pool_retention", args.pool_retention),
        )
        if value is not None
    }
    cfg = TrainConfig(
        profile=args.profile,
        players=args.players,
        num_envs=args.num_envs if args.num_envs is not None else defaults.num_envs,
        min_steps_per_update=args.min_steps,
        updates=args.updates,
        lr=args.lr,
        hidden=args.hidden,
        layers=args.layers,
        arch=args.arch,
        embed_dim=args.embed_dim,
        eval_every=args.eval_every,
        eval_games=args.eval_games,
        checkpoint_every=args.checkpoint_every,
        device=args.device,
        num_workers=args.num_workers,
        seed=args.seed,
        **{k: v for k, v in resume_overrides.items() if k != "num_envs"},
    )
    resume = Path(args.resume) if args.resume else None
    run_dir = resolve_run_dir(args.run_dir, resume, cfg)
    trainer = Trainer(cfg, run_dir, resume=resume, resume_overrides=resume_overrides)
    try:
        trainer.run()
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
