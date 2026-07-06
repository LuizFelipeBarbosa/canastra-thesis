"""Play a game in the terminal: agents or a human seat, any profile.

Examples:
    uv run python examples/play_cli.py --seed 42
    uv run python examples/play_cli.py --players 4 --agents heuristic,random,random,random -v
    uv run python examples/play_cli.py --human 0            # you play seat 0
    uv run python examples/play_cli.py --profile rummy --seed 7
"""

from __future__ import annotations

import argparse
import random

from buraco.agents.heuristic_agent import HeuristicAgent
from buraco.agents.random_agent import RandomAgent
from buraco.cards import card_str
from buraco.describe import describe
from buraco.engine.actions import decode
from buraco.engine.scoring import round_scores
from buraco.engine.state import EndReason
from buraco.env.env import BuracoEnv
from buraco.profiles import load_profile


def show_seat(env: BuracoEnv, seat: int) -> None:
    raw = env.observe_raw(seat)
    hand = " ".join(card_str(c) for c, n in sorted(raw["hand"].items()) for _ in range(n))
    print(f"\n== seat {seat} (side {raw['side']}) | turn {raw['turn_number']} ==")
    print(f"hand ({sum(raw['hand'].values())}): {hand}")
    print(f"deck: {raw['deck_size']}  trash({len(raw['trash'])}):"
          f" {' '.join(card_str(c) for c in raw['trash'][-8:])}")
    print(f"hand sizes: {raw['all_hand_sizes']}  morto taken [own, opp]: {raw['morto_taken']}")
    for label, melds in (("own", raw["own_melds"]), ("opp", raw["opp_melds"])):
        for i, m in enumerate(melds):
            kind = "seq" if m["is_sequence"] else "set"
            tag = " CANASTRA" if m["size"] >= env.cfg.meld.canastra_min_size else ""
            print(f"  {label} meld #{i}: {kind} size={m['size']} wilds={m['wild_count']}{tag}")


def human_pick(env: BuracoEnv, legal: list[int]) -> int:
    slots = env.cfg.meld.max_meld_slots
    show_seat(env, env.current_player)
    for i, a in enumerate(legal):
        print(f"  [{i}] {describe(decode(a, slots), slots)}")
    while True:
        try:
            choice = int(input("choose action> "))
            if 0 <= choice < len(legal):
                return legal[choice]
        except (ValueError, EOFError):
            pass
        print("invalid choice")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="buraco", help="buraco | canasta | rummy | biriba")
    parser.add_argument("--players", type=int, default=2)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--agents", default=None,
                        help="comma list per seat: random|heuristic (default heuristic)")
    parser.add_argument("--human", type=int, default=None, help="seat you play yourself")
    parser.add_argument("-v", "--verbose", action="store_true", help="print every action")
    args = parser.parse_args()

    cfg = load_profile(args.profile, num_players=args.players)
    env = BuracoEnv(cfg)
    seed = args.seed if args.seed is not None else random.randrange(10**6)

    names = (args.agents.split(",") if args.agents
             else ["heuristic"] * cfg.table.num_players)
    agents = [RandomAgent(seed + i) if n.strip() == "random" else HeuristicAgent(seed + i)
              for i, n in enumerate(names)]

    obs, info = env.reset(seed=seed)
    print(f"{cfg.name} | {cfg.table.num_players} players | seed {seed}")
    terminated = truncated = False
    steps = 0
    while not (terminated or truncated):
        seat = info["to_play"]
        if args.human == seat:
            action = human_pick(env, info["legal_actions"])
        else:
            action = agents[seat].act(env.observe_raw(seat), info["legal_actions"], cfg)
        if args.verbose or args.human == seat:
            print(f"seat {seat}: {describe(decode(action, cfg.meld.max_meld_slots), 0)}")
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1

    state = env.state
    reason = EndReason(state.end_reason).name if state.end_reason is not None else "TRUNCATED"
    print(f"\nround over after {steps} steps: {reason}"
          + (f", side {state.went_out_side} went out" if state.went_out_side is not None else ""))
    print(f"round scores by side: {round_scores(state)}")
    print(f"match scores by side: {env.match_scores}")
    print(f"payoffs by seat: {env.get_payoffs()}")


if __name__ == "__main__":
    main()
