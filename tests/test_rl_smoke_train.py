"""End-to-end tiny training run: finite losses, CSV rows, checkpoint, resume."""

from __future__ import annotations

import csv

import pytest

torch = pytest.importorskip("torch")

from pathlib import Path  # noqa: E402

from buraco.rl.config import TrainConfig  # noqa: E402
from buraco.rl.train import Trainer, resolve_run_dir  # noqa: E402

TINY = TrainConfig(
    players=2,
    num_envs=2,
    min_steps_per_update=128,
    updates=2,
    minibatch=64,
    hidden=32,
    layers=1,
    eval_every=0,  # eval covered by test_rl_agent; keep smoke fast
    eval_games=2,
    checkpoint_every=1,
    device="cpu",
    seed=0,
)


def _rows(path) -> list[dict]:
    with open(path) as fh:
        return list(csv.DictReader(fh))


def test_smoke_train_and_resume(tmp_path):
    run_dir = tmp_path / "run"
    Trainer(TINY, run_dir).run()

    rows = _rows(run_dir / "metrics.csv")
    assert len(rows) == 2
    for row in rows:
        for key in ("loss_pi", "loss_v", "entropy", "approx_kl"):
            assert row[key] not in ("", "nan", "inf", "-inf")
            float(row[key])
    latest = run_dir / "checkpoints" / "latest.pt"
    assert latest.exists()
    assert (run_dir / "config.json").exists()

    # Resume for one more update appends a third row and advances the counter.
    resumed = Trainer(
        TrainConfig(**{**TINY.to_dict(), "updates": 3}), run_dir, resume=latest
    )
    resumed.run()
    rows = _rows(run_dir / "metrics.csv")
    assert len(rows) == 3
    assert [int(r["update"]) for r in rows] == [0, 1, 2]
    assert int(rows[2]["global_env_steps"]) > int(rows[1]["global_env_steps"])


def test_resume_infers_run_config_from_checkpoint(tmp_path):
    """--resume must not require repeating --profile/--players (Codex review P2)."""
    run_dir = tmp_path / "run4p"
    Trainer(TrainConfig(**{**TINY.to_dict(), "players": 4}), run_dir).run()
    # Default CLI-style config (2p) + resume from the 4p checkpoint: the
    # checkpoint defines the run; only updates/device come from the CLI.
    resumed = Trainer(
        TrainConfig(updates=3, device="cpu"),
        run_dir,
        resume=run_dir / "checkpoints" / "latest.pt",
    )
    assert resumed.cfg.players == 4
    assert resumed.cfg.hidden == TINY.hidden
    resumed.run()
    assert [int(r["update"]) for r in _rows(run_dir / "metrics.csv")] == [0, 1, 2]


def test_resolve_run_dir_reuses_checkpoint_run(tmp_path):
    """--resume without --run-dir must append to the original run (Codex review P2)."""
    ckpt = tmp_path / "myrun" / "checkpoints" / "latest.pt"
    assert resolve_run_dir(None, ckpt, TINY) == tmp_path / "myrun"
    assert resolve_run_dir("explicit", ckpt, TINY) == Path("explicit")
    bare = tmp_path / "elsewhere" / "ckpt.pt"
    assert resolve_run_dir(None, bare, TINY) == tmp_path / "elsewhere"
    fresh = resolve_run_dir(None, None, TINY)
    assert str(fresh).startswith("runs/buraco2p-")
