# buraco — configurable Buraco-family game engine for AI training

A rules-configurable card-game engine for the Buraco family — Buraco, classic
Canasta, Rummy, and Biriba — with a reinforcement-learning environment: fixed
integer action space with legal-action masking, hidden-information
observations, seeded determinism, and replayable logs.

Built for deep-learning self-play research. Pure-Python engine; numpy only in
the observation-encoding layer.

## Quickstart

```bash
uv sync                                   # Python 3.12 env with numpy/pytest
uv run pytest                             # full test suite incl. soak games
uv run buraco-gui                         # play in the browser (GUI)
uv run python examples/play_cli.py --seed 42 -v          # watch a 2p game
uv run python examples/play_cli.py --human 0             # play seat 0 yourself
uv run python examples/play_cli.py --players 4           # two teams of two
uv run python examples/run_random_games.py --games 1000  # batch simulations
uv run python examples/selfplay_skeleton.py              # RL loop shape
uv sync --group rl                                       # + PyTorch (PPO trainer)
uv run buraco-train --updates 200 --run-dir runs/ppo-2p  # PPO self-play training
```

## Playing: terminal or browser

Two human frontends drive the same engine; use whichever you like.

**Terminal** — `examples/play_cli.py --human 0`: numbered menu of the exact
legal actions each turn.

**Browser GUI** — `uv run buraco-gui` (or `uv run python -m buraco.webui`, or
`examples/play_gui.py`): serves a local single-page table at
`http://127.0.0.1:8377` and opens it. No extra dependencies — stdlib HTTP
server, self-contained HTML/CSS/JS, loopback only. Setup screen in the
browser, or preconfigure from the CLI:

```bash
uv run buraco-gui --profile canasta --players 4 --seed 42 --episode match
uv run buraco-gui --port 9000 --no-open
```

Click a hand card to see its legal targets glow (discard to the lixo, adds to
your melds, new-meld tray); the stock and pile glow on draw turns; a
collapsible “All moves” list always shows the raw legal-action menu. Card
movement is animated (FLIP + ghost flights, `prefers-reduced-motion`
respected, ×½–×4 speed control, skip-to-my-turn), with signals for canastra
limpa/suja, morto states, frozen/blocked piles, red-3 trays, initial-meld
thresholds, and itemized round/match score sheets. The engine stays
authoritative: the browser only ever picks among server-sent legal action
ids, one micro-action per request (`src/buraco/webui/`).

## The environment API

```python
from buraco.env.env import BuracoEnv
from buraco.profiles import load_profile

env = BuracoEnv(load_profile("buraco", num_players=2))
obs, info = env.reset(seed=123)           # obs = seat-to-act's view (numpy dict)
mask = info["action_mask"]                # (1585,) int8; also env.action_mask()
legal = info["legal_actions"]             # sorted list[int]; also env.legal_actions()
obs, reward, terminated, truncated, info = env.step(legal[0])
payoffs = env.get_payoffs()               # per-seat zero-sum payoffs at episode end
```

- Turn-based, current-player perspective (RLCard/OpenSpiel convention): each
  `step` applies one micro-action (a draw, one meld operation, or a discard)
  and returns the observation of the seat now to act.
- Rewards are terminal team score differentials × `scoring.reward_scale`
  (zero-sum; teammates identical). `info["payoffs"]` has the full vector.
- Illegal action ids raise `IllegalAction`; anything the mask allows applies.
- Hidden information never appears in observations (opponent and partner
  hands, stock order, morto contents). `BuracoEnv(cfg, perfect_info=True)`
  adds ground truth to `info["debug"]` only — for debugging, never training.
- Episodes are one round by default; set `scoring.episode = "MATCH"` for full
  matches to `scoring.match_target` (3000 for Buraco).
- Replays: an episode is fully determined by `(config, seed, action_log)` —
  `buraco.env.env.replay(cfg, seed, log)` rebuilds it bit-for-bit. Export/verify
  logs with `run_random_games.py --save-logs / --replay`.

## Training: PPO self-play (`buraco.rl`)

PyTorch lives only in the optional `rl` dependency group — the engine and env
stay numpy-only. Install with `uv sync --group rl`.

```bash
uv run buraco-train --profile buraco --players 2 --updates 200 \
    --run-dir runs/ppo-2p --seed 0 --device cpu     # or: python -m buraco.rl.train
uv run buraco-eval --checkpoint runs/ppo-2p/checkpoints/latest.pt \
    --opponent heuristic --games 200                # or: python -m buraco.rl.evaluate
uv run buraco-train --resume runs/ppo-2p/checkpoints/latest.pt --updates 400 \
    --run-dir runs/ppo-2p                           # continue a run
```

One shared policy/value MLP drives every seat (observations are
perspective-relative, so 2p and 4p share the same network and checkpoints).
Rollouts run N envs in lockstep with one batched forward per decision;
per-seat trajectories get the terminal zero-sum payoff from `get_payoffs()`
and per-trajectory GAE (`gamma=1.0` — reward is terminal-only within a round).
Each run directory holds `config.json` (exact train + rules config),
`metrics.csv` / `eval.csv` (thesis-figure-ready), and atomic checkpoints that
carry the optimizer, RNG states, and the observation layout, so `--resume` is
deterministic on CPU and a checkpoint can never be applied to a permuted
feature order. Fixed-seed evals side-swap the learned agent every other game
against the random and heuristic baselines. `--device cpu` outruns MPS at
this batch size (~2.3k vs ~1.4k steps/s on an M-series laptop).

## Default Buraco rules (the house profile)

- 2 players head-to-head or 4 players in two teams of two (partners opposite).
- Two 52-card decks, no printed jokers; 2s are the wildcards (max 1 per meld;
  a 2 of the run's own suit in the A-2-3 position counts natural).
- Melds: same-suit sequences ≥3 (ace high or low) and same-rank sets ≥3.
- Open trash pile: fully visible, and drawing it takes the entire pile.
- Morto: two 11-card dead hands; first hand-emptying picks one up. Going out
  (bater, +100) requires your morto taken and ≥1 canastra (any meld of 7+;
  limpa 200 / suja 100).
- Standard Brazilian card points (A 15, 2 10, 3–7 5, 8–K 10, joker 20); cards
  left in hand count against you; −100 for a side that never took its morto.

## Changing the rules

Every rule is a field in the frozen `RulesConfig` dataclass tree
(`src/buraco/config.py`, documented in `docs/specs/04-rules-config-schema.md`).
Profiles are plain constructors; variants are overrides, not new engines:

```python
from dataclasses import replace
from buraco.profiles import buraco

cfg = buraco(2)
cfg = replace(cfg, going_out=replace(cfg.going_out, require_clean_canastra=True))
cfg = replace(cfg, wildcard=replace(cfg.wildcard, wildcard_limit_per_meld=2))
```

Configs serialize to JSON (`buraco.engine.serialize.config_to_dict/from_dict`).
Bundled profiles: `buraco` (default), `canasta` (classic US rules: sets-only,
frozen/conditional pile with forced top-card meld, red-3 trays, black-3 pile
blocking, initial-meld thresholds — see `docs/specs/06-canasta-mechanics.md`),
`rummy` (heads-up only: single deck, no wilds, top-card draw), `biriba` (jokers, biribakia
converting to fresh stock). The action-id layout is identical across
profiles — variants only change the legal mask.

## Action space (fixed, 1585 ids)

Turns decompose into micro-actions so the space stays small and maskable:
draw (stock / whole pile), create minimum meld (parametric by suit/start/wild
source or rank/wild source), add one card to an own meld slot, discard, go
out, end round. Full id arithmetic: `docs/specs/02-action-space.md`;
observation layout: `docs/specs/03-observation-space.md`.

## Project layout

```
src/buraco/
  cards.py, config.py     # card-id space; RulesConfig tree (single rule source)
  profiles/               # buraco, rummy, biriba constructors
  engine/                 # pure-Python rules engine
    state.py melds.py legal.py turns.py scoring.py actions.py serialize.py
  env/                    # RL layer (numpy lives here only)
    env.py observations.py encoding.py
  agents/                 # random + heuristic baselines
  rl/                     # PPO self-play trainer (torch; optional dep group)
    obs.py buffer.py      #   numpy-only: canonical flattening, GAE
    nets.py ppo.py rollout.py agent.py evaluate.py train.py checkpoint.py
  describe.py             # human-readable action labels (CLI + GUI)
  webui/                  # browser GUI: session/views/events + stdlib server
    session.py views.py events.py server.py static/
docs/specs/               # design specs 00-05 + test-scenario catalogues
examples/                 # play_cli, play_gui, run_random_games, selfplay_skeleton
tests/                    # unit, property-based (hypothesis), soak suites
```

## Design documents

`docs/specs/00-decisions.md` is the decisions register (locked house rules,
defaults, open confirmations). Specs 01–05 define the game-state model, action
space, observation space, config schema, and variant diffs; the two
`test-scenarios-*.md` files are the edge-case catalogues the test suite tracks.
