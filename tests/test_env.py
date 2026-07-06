"""RL environment: masks, hidden info, determinism, rewards (M6).

Scenario numbers refer to docs/specs/test-scenarios-rl.md.
"""

import random
from dataclasses import replace

import numpy as np
import pytest

from buraco.cards import Rank, Suit, card_id
from buraco.config import EPISODE_MATCH
from buraco.engine.serialize import state_hash
from buraco.engine.turns import IllegalAction
from buraco.env.env import BuracoEnv, replay
from buraco.profiles import buraco


def run_random_episode(env: BuracoEnv, seed: int, max_steps: int = 5000):
    """Drive an episode with seeded uniform-random legal actions."""
    rng = random.Random(seed * 7 + 1)
    obs, info = env.reset(seed=seed)
    steps = 0
    terminated = truncated = False
    while not (terminated or truncated) and steps < max_steps:
        action = rng.choice(info["legal_actions"])
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1
    return obs, reward, terminated, truncated, info


def test_obs_shapes_invariant_across_steps():  # scenario 31
    env = BuracoEnv(buraco(2))
    obs, info = env.reset(seed=11)
    shapes = {k: (v.shape, v.dtype) for k, v in obs.items()}
    rng = random.Random(0)
    for _ in range(200):
        if not info["legal_actions"]:
            break
        obs, _, term, trunc, info = env.step(rng.choice(info["legal_actions"]))
        assert {k: (v.shape, v.dtype) for k, v in obs.items()} == shapes
        if term or trunc:
            break


def test_mask_matches_legal_actions_and_rejects_unmasked():  # scenarios 5, 6
    env = BuracoEnv(buraco(2))
    _, info = env.reset(seed=13)
    rng = random.Random(1)
    for _ in range(60):
        mask = info["action_mask"]
        legal = info["legal_actions"]
        assert list(np.flatnonzero(mask)) == legal
        assert legal == env.legal_actions()
        illegal = next(a for a in range(env.num_actions) if mask[a] == 0)
        with pytest.raises(IllegalAction):
            env.step(illegal)
        _, _, term, trunc, info = env.step(rng.choice(legal))
        if term or trunc:
            break


def test_mask_nonempty_until_terminal_then_zero():  # scenario 7
    env = BuracoEnv(buraco(2))
    _, info = env.reset(seed=17)
    rng = random.Random(2)
    terminated = truncated = False
    while not (terminated or truncated):
        assert info["legal_actions"], "empty mask mid-episode"
        _, _, terminated, truncated, info = env.step(rng.choice(info["legal_actions"]))
    assert env.action_mask().sum() == 0
    with pytest.raises(IllegalAction):
        env.step(0)


def _assert_obs_equal(a, b):
    assert a.keys() == b.keys()
    for key in a:
        assert np.array_equal(a[key], b[key]), f"obs field {key} differs"


def test_hidden_hand_and_deck_and_morto_invariance():  # scenarios 8, 10, 11
    env = BuracoEnv(buraco(2))
    env.reset(seed=19)
    state = env.state
    seat = state.current_player
    opp = 1 - seat
    before = env._observation(seat)
    legal_before = env.legal_actions()

    # (10) shuffle the undrawn stock
    random.Random(99).shuffle(state.stock)
    # (11) swap morto contents between the sides
    state.morto[0], state.morto[1] = state.morto[1], state.morto[0]
    # (8) exchange a card between the opponent hand and the stock (sizes fixed)
    opp_card = next(iter(state.hands[opp]))
    stock_card = next(c for c in state.stock if state.hands[opp].get(c, 0) == 0)
    state.hands[opp][opp_card] -= 1
    if state.hands[opp][opp_card] == 0:
        del state.hands[opp][opp_card]
    state.hands[opp][stock_card] += 1
    state.stock.remove(stock_card)
    state.stock.append(opp_card)

    _assert_obs_equal(before, env._observation(seat))
    assert env.legal_actions() == legal_before


def test_partner_hand_hidden_in_4p():  # scenario 9
    env = BuracoEnv(buraco(4))
    env.reset(seed=23)
    state = env.state
    seat = state.current_player
    partner = (seat + 2) % 4
    before = env._observation(seat)

    partner_card = next(iter(state.hands[partner]))
    stock_card = next(c for c in state.stock if state.hands[partner].get(c, 0) == 0)
    state.hands[partner][partner_card] -= 1
    if state.hands[partner][partner_card] == 0:
        del state.hands[partner][partner_card]
    state.hands[partner][stock_card] += 1
    state.stock.remove(stock_card)
    state.stock.append(partner_card)

    _assert_obs_equal(before, env._observation(seat))


def test_perfect_info_isolation():  # scenario 12
    plain = BuracoEnv(buraco(2))
    debug = BuracoEnv(buraco(2), perfect_info=True)
    obs_a, info_a = plain.reset(seed=29)
    obs_b, info_b = debug.reset(seed=29)
    _assert_obs_equal(obs_a, obs_b)
    assert "debug" not in info_a and "debug" in info_b
    assert info_b["debug"]["ground_truth_hash"] == state_hash(debug.state)
    assert len(info_b["debug"]["all_hands"]) == 2


def test_determinism_and_replay():  # scenarios 13, 14
    cfg = buraco(2)
    env = BuracoEnv(cfg)
    run_random_episode(env, seed=31)
    original_hash = state_hash(env.state)
    log = list(env.action_log)

    replayed = replay(cfg, 31, log)
    assert state_hash(replayed.state) == original_hash
    assert replayed.match_scores == env.match_scores


def test_zero_sum_payoffs_2p():  # scenario 17
    env = BuracoEnv(buraco(2))
    for seed in (37, 38, 39):
        _, reward, terminated, truncated, info = run_random_episode(env, seed)
        if terminated:
            payoffs = info["payoffs"]
            assert abs(sum(payoffs)) < 1e-9
            assert payoffs[env.state.current_player] == pytest.approx(reward)
            return
    pytest.fail("no seed terminated")


def test_zero_sum_and_teammate_equality_4p():  # scenarios 18, 19
    env = BuracoEnv(buraco(4))
    for seed in (41, 42, 43, 44):
        _, _, terminated, _, info = run_random_episode(env, seed)
        if terminated:
            payoffs = info["payoffs"]
            assert abs(sum(payoffs)) < 1e-9
            assert payoffs[0] == pytest.approx(payoffs[2])
            assert payoffs[1] == pytest.approx(payoffs[3])
            assert payoffs[0] == pytest.approx(-payoffs[1])
            return
    pytest.fail("no seed terminated")


def test_truncation_semantics():  # scenario 16
    cfg = buraco(2)
    cfg = replace(cfg, turn=replace(cfg.turn, truncation_cap=3))
    env = BuracoEnv(cfg)
    _, reward, terminated, truncated, info = run_random_episode(env, seed=47)
    assert truncated and not terminated
    assert reward == 0.0
    assert env.action_mask().sum() == 0


def test_match_mode_accumulates_rounds():  # scenario 20
    cfg = buraco(2)
    cfg = replace(
        cfg,
        scoring=replace(cfg.scoring, episode=EPISODE_MATCH, match_target=200),
        turn=replace(cfg.turn, truncation_cap=100_000),
    )
    env = BuracoEnv(cfg)
    _, _, terminated, truncated, info = run_random_episode(env, seed=53, max_steps=100_000)
    assert terminated and not truncated
    assert max(info["match_scores"]) >= 200 or min(info["match_scores"]) <= -200


def test_rank2_create_set_masked_in_buraco():  # scenario 33
    env = BuracoEnv(buraco(2))
    env.reset(seed=59)
    state = env.state
    seat = state.current_player
    from collections import Counter

    state.hands[seat] = Counter(
        [card_id(Rank.TWO, Suit.HEARTS), card_id(Rank.TWO, Suit.SPADES),
         card_id(Rank.TWO, Suit.DIAMONDS), card_id(Rank.NINE, Suit.CLUBS)]
    )
    from buraco.engine.actions import CreateSet, encode
    from buraco.engine.state import Phase

    state.phase = Phase.PLAY
    mask = env.action_mask()
    for wild in range(3):
        assert mask[encode(CreateSet(rank=Rank.TWO, wild=wild), cfg_slots(env))] == 0


def cfg_slots(env: BuracoEnv) -> int:
    return env.cfg.meld.max_meld_slots


def test_info_contract_and_seat_perspective():  # scenario 32
    env = BuracoEnv(buraco(4))
    obs, info = env.reset(seed=61)
    for key in ("action_mask", "legal_actions", "to_play", "team", "phase"):
        assert key in info
    assert info["team"] == env.cfg.table.side(info["to_play"])
    assert obs["seat_rel"][info["to_play"]] == 1
    assert obs["partner_rel"][(info["to_play"] + 2) % 4] == 1


def test_history_records_public_actions():
    env = BuracoEnv(buraco(2))
    obs, info = env.reset(seed=67)
    rng = random.Random(3)
    for _ in range(6):
        _, _, term, trunc, info = env.step(rng.choice(info["legal_actions"]))
        if term or trunc:
            break
    raw = env.observe_raw(env.state.current_player)
    assert raw["history"]
    assert all(item["family"] in {
        "draw_deck", "draw_trash", "create_seq", "create_set", "add", "discard", "go_out"
    } for item in raw["history"])
