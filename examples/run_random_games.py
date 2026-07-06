"""Batch simulation runner with multiprocessing and exportable game logs.

Examples:
    uv run python examples/run_random_games.py --games 5000 --players 2
    uv run python examples/run_random_games.py --games 500 --players 4 --agents heuristic
    uv run python examples/run_random_games.py --games 100 --save-logs logs.jsonl
    uv run python examples/run_random_games.py --replay logs.jsonl   # verify determinism
"""

from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool

from buraco.engine.serialize import state_hash
from buraco.engine.state import EndReason
from buraco.env.env import BuracoEnv, replay
from buraco.profiles import load_profile

_WORKER = {}


def _init_worker(profile: str, players: int, agent_kind: str) -> None:
    from buraco.agents.heuristic_agent import HeuristicAgent
    from buraco.agents.random_agent import RandomAgent

    cfg = load_profile(profile, num_players=players)
    _WORKER["cfg"] = cfg
    _WORKER["env"] = BuracoEnv(cfg)
    _WORKER["make_agents"] = lambda seed: [
        HeuristicAgent(seed + p) if agent_kind == "heuristic" else RandomAgent(seed + p)
        for p in range(cfg.table.num_players)
    ]


def _play(seed: int) -> dict:
    env: BuracoEnv = _WORKER["env"]
    agents = _WORKER["make_agents"](seed)
    obs, info = env.reset(seed=seed)
    terminated = truncated = False
    steps = 0
    while not (terminated or truncated):
        seat = info["to_play"]
        action = agents[seat].act(env.observe_raw(seat), info["legal_actions"], env.cfg)
        obs, _, terminated, truncated, info = env.step(action)
        steps += 1
    state = env.state
    return {
        "seed": seed,
        "steps": steps,
        "terminated": terminated,
        "end_reason": EndReason(state.end_reason).name if state.end_reason is not None else None,
        "went_out_side": state.went_out_side,
        "match_scores": env.match_scores,
        "payoffs": env.get_payoffs(),
        "state_hash": state_hash(state),
        "action_log": env.action_log,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--players", type=int, default=2)
    parser.add_argument("--profile", default="buraco")
    parser.add_argument("--agents", default="random", choices=["random", "heuristic"])
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--save-logs", default=None, help="write JSONL replay logs here")
    parser.add_argument("--replay", default=None, help="verify a JSONL log file re-plays")
    args = parser.parse_args()

    if args.replay:
        verify_logs(args.replay)
        return

    start = time.perf_counter()
    with Pool(args.workers, initializer=_init_worker,
              initargs=(args.profile, args.players, args.agents)) as pool:
        results = pool.map(_play, range(args.games), chunksize=16)
    elapsed = time.perf_counter() - start

    total_steps = sum(r["steps"] for r in results)
    reasons: dict[str, int] = {}
    wins = [0, 0]
    for r in results:
        reasons[r["end_reason"] or "TRUNCATED"] = reasons.get(r["end_reason"] or "TRUNCATED", 0) + 1
        if r["went_out_side"] is not None:
            wins[r["went_out_side"]] += 1
    print(f"{args.games} games ({args.profile}, {args.players}p, {args.agents}) "
          f"in {elapsed:.1f}s — {total_steps / elapsed:,.0f} steps/s")
    print(f"end reasons: {reasons}")
    print(f"went-out by side: {wins}")
    mean0 = sum(r["match_scores"][0] for r in results) / len(results)
    mean1 = sum(r["match_scores"][1] for r in results) / len(results)
    print(f"mean side scores: [{mean0:.1f}, {mean1:.1f}]")

    if args.save_logs:
        with open(args.save_logs, "w") as fh:
            for r in results:
                fh.write(json.dumps({
                    "profile": args.profile, "players": args.players,
                    "seed": r["seed"], "action_log": r["action_log"],
                    "state_hash": r["state_hash"],
                }) + "\n")
        print(f"wrote {len(results)} replayable logs to {args.save_logs}")


def verify_logs(path: str) -> None:
    """Re-step every log and compare terminal state hashes (SPEC scenario 14)."""
    ok = bad = 0
    with open(path) as fh:
        for line in fh:
            rec = json.loads(line)
            cfg = load_profile(rec["profile"], num_players=rec["players"])
            env = replay(cfg, rec["seed"], rec["action_log"])
            if state_hash(env.state) == rec["state_hash"]:
                ok += 1
            else:
                bad += 1
                print(f"MISMATCH seed={rec['seed']}")
    print(f"replay verification: {ok} ok, {bad} mismatched")
    if bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
