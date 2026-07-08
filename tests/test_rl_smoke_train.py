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


def test_smoke_train_parallel_and_resume(tmp_path):
    """num_workers=2: trains, checkpoints per-slot counters, resumes them."""
    run_dir = tmp_path / "run-mp"
    cfg = TrainConfig(**{**TINY.to_dict(), "num_workers": 2})
    trainer = Trainer(cfg, run_dir)
    try:
        trainer.run()
    finally:
        trainer.close()
    rows = _rows(run_dir / "metrics.csv")
    assert len(rows) == 2
    for row in rows:
        float(row["loss_pi"])

    # Same worker count: per-slot counters pass through the checkpoint intact.
    resumed = Trainer(
        TrainConfig(**{**cfg.to_dict(), "updates": 3}),
        run_dir,
        resume=run_dir / "checkpoints" / "latest.pt",
    )
    try:
        assert len(resumed.collector.counters) == 2
        assert all(c % 2 == slot for slot, c in enumerate(resumed.collector.counters))
        resumed.run()
    finally:
        resumed.close()
    assert [int(r["update"]) for r in _rows(run_dir / "metrics.csv")] == [0, 1, 2]


def test_trainer_refuses_indivisible_worker_env_split(tmp_path):
    """The refusal happens before any pool spawns or files are written."""
    bad = TrainConfig(**{**TINY.to_dict(), "num_envs": 3, "num_workers": 2})
    with pytest.raises(SystemExit, match="multiple of"):
        Trainer(bad, tmp_path / "run-bad")
    assert not (tmp_path / "run-bad").exists()


def test_resume_across_worker_counts(tmp_path):
    """Serial checkpoints resume into a pool and back without seed reuse."""
    run_dir = tmp_path / "run-serial"
    Trainer(TINY, run_dir).run()  # legacy-style checkpoint: scalar counter only
    latest = run_dir / "checkpoints" / "latest.pt"

    to_parallel = Trainer(
        TrainConfig(**{**TINY.to_dict(), "updates": 3, "num_workers": 2}),
        run_dir,
        resume=latest,
    )
    try:
        high = max(to_parallel.collector.counters)
        assert all(c % 2 == slot for slot, c in enumerate(to_parallel.collector.counters))
        to_parallel.run()
        assert max(to_parallel.collector.counters) > high
    finally:
        to_parallel.close()

    # Parallel checkpoint back to serial: resume at the high-water mark.
    back = Trainer(
        TrainConfig(**{**TINY.to_dict(), "updates": 4}),
        run_dir,
        resume=latest,
    )
    assert isinstance(back.collector.episode_counter, int)
    back.run()
    assert [int(r["update"]) for r in _rows(run_dir / "metrics.csv")] == [0, 1, 2, 3]


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


def test_fresh_run_refused_in_used_directory(tmp_path):
    """A non-resume run must not append to or overwrite an existing run."""
    run_dir = tmp_path / "run"
    Trainer(TINY, run_dir).run()
    metrics_before = (run_dir / "metrics.csv").read_bytes()
    latest = run_dir / "checkpoints" / "latest.pt"
    ckpt_before = latest.read_bytes()

    with pytest.raises(SystemExit, match="refusing fresh run"):
        Trainer(TINY, run_dir)

    # Nothing was written or modified by the refused attempt.
    assert (run_dir / "metrics.csv").read_bytes() == metrics_before
    assert latest.read_bytes() == ckpt_before
    # Resume into the same directory still works.
    Trainer(
        TrainConfig(**{**TINY.to_dict(), "updates": 3}), run_dir, resume=latest
    ).run()
    assert [int(r["update"]) for r in _rows(run_dir / "metrics.csv")] == [0, 1, 2]


def test_fresh_run_allowed_in_new_or_empty_directory(tmp_path):
    Trainer(TINY, tmp_path / "brand-new").run()  # dir does not exist yet
    empty = tmp_path / "empty"
    empty.mkdir()
    Trainer(TINY, empty).run()  # dir exists but holds no run artifacts
    assert (empty / "metrics.csv").exists()


def test_resolve_run_dir_reuses_checkpoint_run(tmp_path):
    """--resume without --run-dir must append to the original run (Codex review P2)."""
    ckpt = tmp_path / "myrun" / "checkpoints" / "latest.pt"
    assert resolve_run_dir(None, ckpt, TINY) == tmp_path / "myrun"
    assert resolve_run_dir("explicit", ckpt, TINY) == Path("explicit")
    bare = tmp_path / "elsewhere" / "ckpt.pt"
    assert resolve_run_dir(None, bare, TINY) == tmp_path / "elsewhere"
    fresh = resolve_run_dir(None, None, TINY)
    assert str(fresh).startswith("runs/buraco2p-")
