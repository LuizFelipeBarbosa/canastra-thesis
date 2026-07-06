"""Fixed-seed evaluation of a learned policy against scripted opponents.

Drives games with the exact run_random_games.py loop. The learned side plays
side 0 for half the games and side 1 for the other half (reset always makes
seat 0 the first player, so side-swapping removes first-mover bias). The seed
set is fixed and disjoint from training seeds, making evals across a run a
paired comparison on the same deals.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Callable

from buraco.agents.heuristic_agent import HeuristicAgent
from buraco.agents.random_agent import RandomAgent
from buraco.config import RulesConfig
from buraco.env.env import BuracoEnv

EVAL_SEED_BASE = 10_000_000


@dataclass
class EvalResult:
    opponent: str
    games: int
    win_rate: float
    draw_rate: float
    mean_payoff: float
    mean_steps: float
    truncation_rate: float

    def __str__(self) -> str:
        return (
            f"vs {self.opponent}: {self.games} games | "
            f"win {self.win_rate:.1%} draw {self.draw_rate:.1%} | "
            f"mean payoff {self.mean_payoff:+.3f} | "
            f"mean steps {self.mean_steps:.0f} | trunc {self.truncation_rate:.1%}"
        )


OPPONENTS: dict[str, Callable[[int], Any]] = {
    "random": lambda seed: RandomAgent(seed),
    "heuristic": lambda seed: HeuristicAgent(seed),
}


def evaluate_vs(
    agent: Any,
    cfg: RulesConfig,
    opponent: str,
    n_games: int,
    seed: int = 0,
) -> EvalResult:
    """`agent` is anything with the act(raw_obs, legal_ids, cfg) protocol."""
    env = BuracoEnv(cfg)
    num_players = cfg.table.num_players
    wins = draws = truncations = 0
    payoff_sum = steps_sum = 0.0

    for game in range(n_games):
        learned_side = game % 2  # side-swap half the games
        seats = {
            s: agent
            if cfg.table.side(s) == learned_side
            else OPPONENTS[opponent](seed + game * num_players + s)
            for s in range(num_players)
        }
        _, info = env.reset(seed=EVAL_SEED_BASE + seed + game)
        terminated = truncated = False
        steps = 0
        while not (terminated or truncated):
            seat = info["to_play"]
            action = seats[seat].act(env.observe_raw(seat), info["legal_actions"], cfg)
            _, _, terminated, truncated, info = env.step(action)
            steps += 1
        learned_seat = learned_side  # side(seat) == seat % 2, so seat==side works
        payoff = env.get_payoffs()[learned_seat]
        wins += payoff > 0
        draws += payoff == 0
        truncations += int(truncated)
        payoff_sum += payoff
        steps_sum += steps

    return EvalResult(
        opponent=opponent,
        games=n_games,
        win_rate=wins / n_games,
        draw_rate=draws / n_games,
        mean_payoff=payoff_sum / n_games,
        mean_steps=steps_sum / n_games,
        truncation_rate=truncations / n_games,
    )


def main() -> None:
    try:
        from buraco.rl.agent import TorchAgent
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            f"missing RL dependencies ({exc}); install with: uv sync --group rl"
        ) from exc

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--opponent", default="random", choices=sorted(OPPONENTS))
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sample", action="store_true", help="sample instead of argmax")
    args = parser.parse_args()

    agent = TorchAgent.from_checkpoint(
        args.checkpoint, device=args.device, greedy=not args.sample, seed=args.seed
    )
    from buraco.engine.serialize import config_from_dict
    from buraco.rl.checkpoint import load_checkpoint

    cfg = config_from_dict(load_checkpoint(args.checkpoint).rules_config)
    print(evaluate_vs(agent, cfg, args.opponent, args.games, seed=args.seed))


if __name__ == "__main__":
    main()
