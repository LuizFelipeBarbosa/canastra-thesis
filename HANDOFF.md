# Session handoff — V2b training in flight

Written 2026-07-10 when moving development from the old Mac to this machine.
Read this first; it replaces the session memory that did not transfer.

## Current state

**A V2b training run is paused at update 24,180 of 40,000** and must be
resumed. Its run directory lives on the T7 external drive:

    /Volumes/T7/canastra-thesis-runs/ppo-4p-v2b/

Resume it with (tune `--num-workers` to this machine's cores; keep
`num_envs`/everything else from the checkpoint):

    PYTHONUNBUFFERED=1 nohup uv run --group rl python -m buraco.rl.train \
      --updates 40000 --num-workers <N> \
      --resume /Volumes/T7/canastra-thesis-runs/ppo-4p-v2b/checkpoints/latest.pt \
      >> /Volumes/T7/canastra-thesis-runs/ppo-4p-v2b.log 2>&1 &

Rules of thumb from measurement: workers scale rollout (pure-Python engine,
CPU-bound, ~90% of wall time; GPU is irrelevant at this model size); give
each worker ≥4 envs (`num_envs` must be a multiple of `num_workers`;
`--num-envs` may be passed on resume to widen the topology). The old Mac ran
4 workers × 16 envs at ~7k steps/s when quiet; a 20-core box hit ~14.8k with
16 × 64.

## What the runs established (all evals: 4p buraco, greedy play)

| Agent | Training | Held-out test (seed 9000, 1000 deals) |
|---|---|---|
| v1  (flat MLP, pure self-play) | 40k updates | 50.6% win, payoff −0.008 |
| V2a (flat MLP + opponent pool) | 40k + 20k mix | **52.7% win, payoff +0.023** |
| V2b (structured encoder + pool from scratch) | at 24k of 40k | pending — tracking +4–5pp over v1 at matched ages; hit 51% on the fixed set at 18k, above v1's 40k endpoint |

Key findings so far:
- Pure self-play plateaus vs the scripted heuristic (transfer gap); mixing
  opponents into training (25% heuristic / 25% frozen pool) fixes it.
- The structured encoder (shared card embedding, per-meld-slot MLP; see
  `src/buraco/rl/nets.py`) beats the flat baseline at equal ages with fewer
  params.
- Checkpoints oscillate ±4pp → NEVER report the final checkpoint. Protocol:
  scan late checkpoints on the seed-7000 validation set (500–1000 games via
  `buraco.rl.evaluate.evaluate_vs`, `seed=7000`), confirm the winner on the
  untouched seed-9000 test set. v1/V2a numbers above used exactly this.
- V2a's pool decayed late with newest-K retention; V2b runs
  `--pool-retention spaced`. Whether the 30k–40k stretch stays stable is the
  main open question of the run.

## When the run finishes (~16k updates to go)

1. Checkpoint-selection scan on seed-7000, confirm on seed-9000 (see above).
2. Fill the V2b row of the table; that completes the three-way ablation
   (v1 / v1+pool / structured+pool) for the thesis.
3. V3 shortlist (evidence-based, in priority order): upgrade the yardstick
   (checkpoint Elo ladder, match-to-3000 play, human games via the web GUI);
   belief modeling / card-counting auxiliary head; training-stability package
   (EMA policy, LR schedule, validation early-stop); decision-time search as
   a stretch goal.

## Operational notes

- metrics.csv / eval.csv have duplicate-update rows at every crash/resume
  seam — dedup keep-LAST per update before plotting. v1's files additionally
  have a two-writer interleaved band (updates ~9.5k–13.4k) — treat as
  ambiguous.
- Thesis-table checkpoints preserved on T7: ppo-4p/checkpoints/ckpt_039999.pt
  (v1) and ppo-4p-mix/checkpoints/ckpt_052200.pt (V2a).
- molab sandboxes recycle within hours — fine for probes/evals, never for
  multi-day runs without frequent checkpoint sync-out.
- The pure-Python rollout is very sensitive to machine load (3–5× slowdown
  under contention); throughput recovers on its own.
