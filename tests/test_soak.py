"""Soak suite (M7): many seeded games, invariants checked throughout.

Engine-level games run without observation encoding for speed; a smaller
env-level batch exercises the full obs/mask/reward path plus replay equality.
"""

import random

import pytest

from buraco.agents.heuristic_agent import HeuristicAgent
from buraco.agents.random_agent import RandomAgent
from buraco.engine.legal import legal_actions
from buraco.engine.melds import validate_meld
from buraco.engine.scoring import round_scores
from buraco.engine.serialize import state_hash
from buraco.engine.state import EndReason, deal_round, dealt_multiset
from buraco.engine.turns import apply_action
from buraco.env.env import BuracoEnv, replay
from buraco.profiles import buraco

MAX_TURNS = 2000


def play_engine_game(cfg, seed):
    rng = random.Random(seed)
    state = deal_round(cfg, rng)
    full = dealt_multiset(state)
    steps = 0
    while not state.round_over and state.turn_number < MAX_TURNS:
        acts = legal_actions(state)
        assert acts, f"seed {seed}: empty legal set at step {steps}"
        apply_action(state, rng.choice(acts))
        steps += 1
        assert steps < 60_000, f"seed {seed}: runaway game"
    assert dealt_multiset(state) == full, f"seed {seed}: card conservation broken"
    for meld in state.melds:
        validate_meld(cfg, meld)
    for side, taken in enumerate(state.morto_taken):
        assert (state.morto[side] is None) or not taken
    round_scores(state)  # must always be computable
    return state


def test_soak_engine_2p_random():
    cfg = buraco(2)
    ended, melds_made = 0, 0
    for seed in range(150):
        state = play_engine_game(cfg, seed)
        ended += state.round_over
        melds_made += len(state.melds)
    assert ended >= 140  # random games overwhelmingly finish within the cap
    assert melds_made > 0


def test_soak_engine_4p_random():
    cfg = buraco(4)
    ended = 0
    baters = 0
    for seed in range(100):
        state = play_engine_game(cfg, 10_000 + seed)
        ended += state.round_over
        baters += state.end_reason is EndReason.BATER
    assert ended >= 90


def run_env_game(env, agents, seed, max_steps=20_000):
    obs, info = env.reset(seed=seed)
    terminated = truncated = False
    steps = 0
    while not (terminated or truncated) and steps < max_steps:
        seat = info["to_play"]
        raw = env.observe_raw(seat)
        action = agents[seat].act(raw, info["legal_actions"], env.cfg)
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1
    return terminated, truncated, info


def test_soak_env_random_with_replay():
    cfg = buraco(2)
    env = BuracoEnv(cfg)
    replayed = 0
    for seed in range(20):
        agents = [RandomAgent(seed), RandomAgent(seed + 1)]
        terminated, truncated, info = run_env_game(env, agents, seed)
        assert terminated or truncated
        if terminated:
            assert abs(sum(info["payoffs"])) < 1e-9
            if replayed < 5:
                twin = replay(cfg, seed, list(env.action_log))
                assert state_hash(twin.state) == state_hash(env.state)
                assert twin.get_payoffs() == env.get_payoffs()
                replayed += 1
    assert replayed >= 1


def test_soak_env_heuristic_vs_random_2p_and_4p():
    for players, n_games in ((2, 15), (4, 10)):
        cfg = buraco(players)
        env = BuracoEnv(cfg)
        finished = 0
        for seed in range(n_games):
            agents = [
                HeuristicAgent(seed) if p % 2 == 0 else RandomAgent(seed + p)
                for p in range(players)
            ]
            terminated, truncated, info = run_env_game(env, agents, 500 + seed)
            finished += terminated or truncated
        assert finished == n_games


@pytest.mark.parametrize("players", [2, 4])
def test_soak_determinism_same_seed_same_hash(players):
    cfg = buraco(players)

    def run(seed):
        rng = random.Random(999)
        state = deal_round(cfg, random.Random(seed))
        while not state.round_over and state.turn_number < MAX_TURNS:
            apply_action(state, rng.choice(legal_actions(state)))
        return state_hash(state)

    assert run(7) == run(7)
