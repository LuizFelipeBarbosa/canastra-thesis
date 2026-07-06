"""Variant profiles: Rummy and Biriba behave per SPEC 05; ids stay stable.

The Canastra profile is deferred pending decision D9.
"""

import random
from collections import Counter

from buraco.cards import JOKER, Rank, Suit, card_id
from buraco.engine.actions import CreateSeq, CreateSet, Discard, DrawDeck
from buraco.engine.melds import SET_WILD_NONE, plan_sequence, plan_set
from buraco.engine.state import EndReason, Phase, deal_round, dealt_multiset
from buraco.engine.turns import apply_action
from buraco.env.env import BuracoEnv
from buraco.profiles import biriba, load_profile, rummy
from tests.helpers import ct, make_state
from tests.test_soak import play_engine_game


def test_rummy_deal_and_upcard():
    cfg = rummy(2)
    state = deal_round(cfg, random.Random(3))
    assert [state.hand_size(p) for p in range(2)] == [10, 10]
    assert state.morto == [] and state.morto_taken == []
    assert len(state.trash) == 1  # initial upcard
    assert len(state.stock) == 52 - 20 - 1
    assert dealt_multiset(state) == Counter(range(52))


def test_rummy_has_no_wild_moves_but_allows_two_sets():
    cfg = rummy(2)
    hand = Counter([card_id(Rank.TWO, Suit.HEARTS), card_id(Rank.TWO, Suit.SPADES),
                    card_id(Rank.TWO, Suit.DIAMONDS), card_id(Rank.NINE, Suit.CLUBS),
                    JOKER])
    # scenario 33: rank-2 sets are real melds when 2 is not wild
    assert plan_set(cfg, hand, Rank.TWO, SET_WILD_NONE) is not None
    # and no wild-flavoured plan is ever legal
    for wild in (1, 2, 3):
        for suit in Suit:
            for start in range(1, 13):
                assert plan_sequence(cfg, hand, suit, start, wild) is None
    for wild in (1, 2):
        for rank in Rank:
            assert plan_set(cfg, hand, rank, wild) is None


def test_rummy_ace_high_forbidden():
    cfg = rummy(2)
    hand = Counter([card_id(Rank.QUEEN, Suit.CLUBS), card_id(Rank.KING, Suit.CLUBS),
                    card_id(Rank.ACE, Suit.CLUBS)])
    assert plan_sequence(cfg, hand, Suit.CLUBS, 12, 0) is None  # Q-K-A(high)
    assert plan_sequence(cfg, hand, Suit.CLUBS, 1, 0) is None  # A-2-3 needs 2,3
    hand2 = Counter([card_id(Rank.ACE, Suit.CLUBS), card_id(Rank.TWO, Suit.CLUBS),
                     card_id(Rank.THREE, Suit.CLUBS)])
    assert plan_sequence(cfg, hand2, Suit.CLUBS, 1, 0) is not None


def test_rummy_goes_out_without_canastra():
    cfg = rummy(2)
    state = make_state(cfg=cfg, hands=[[ct(Rank.NINE, Suit.CLUBS)], [ct(Rank.TEN, Suit.CLUBS)]],
                       morto=[], morto_taken=[])
    apply_action(state, Discard(ct(Rank.NINE, Suit.CLUBS)))
    assert state.round_over and state.end_reason is EndReason.BATER
    assert state.went_out_side == 0


def test_biriba_deal_with_jokers_and_upcard():
    cfg = biriba(4)
    state = deal_round(cfg, random.Random(5))
    assert [state.hand_size(p) for p in range(4)] == [11, 11, 11, 11]
    assert len(state.morto) == 2 and all(len(m) == 11 for m in state.morto)
    assert len(state.trash) == 1
    assert len(state.stock) == 108 - 44 - 22 - 1
    assert dealt_multiset(state)[JOKER] == 4


def test_biriba_convert_morto_on_stock_exhaustion():
    cfg = biriba(2)
    nine = ct(Rank.NINE, Suit.CLUBS)
    state = make_state(
        cfg=cfg,
        hands=[[nine], [nine]],
        stock=[ct(Rank.FOUR, Suit.HEARTS)],
        morto=[(ct(Rank.FIVE, Suit.SPADES),) * 1, (ct(Rank.SIX, Suit.SPADES),)],
        morto_taken=[False, False],
        phase=Phase.DRAW,
    )
    apply_action(state, DrawDeck())
    # stock emptied -> side 0's biribaki became the new stock, NOT taken
    assert state.stock == [ct(Rank.FIVE, Suit.SPADES)]
    assert state.morto[0] is None and state.morto_taken[0] is False
    assert state.morto[1] is not None


def test_profiles_registry_and_id_stability():
    sizes = set()
    for name in ("buraco", "rummy", "biriba"):
        cfg = load_profile(name)
        env = BuracoEnv(cfg)
        sizes.add(env.num_actions)
    assert sizes == {1585}  # scenario 15: same layout across profiles


def test_soak_rummy_and_biriba():
    for cfg, n in ((rummy(2), 40), (biriba(4), 25), (biriba(2), 15)):
        ended = 0
        for seed in range(n):
            state = play_engine_game(cfg, 77_000 + seed)
            ended += state.round_over
        assert ended >= n - 2, f"{cfg.name}: too many capped games"


def test_variant_env_smoke():
    for cfg in (rummy(2), biriba(4)):
        env = BuracoEnv(cfg)
        obs, info = env.reset(seed=9)
        rng = random.Random(1)
        for _ in range(300):
            if not info["legal_actions"]:
                break
            obs, _, term, trunc, info = env.step(rng.choice(info["legal_actions"]))
            if term or trunc:
                break
        assert obs["hand"].shape == (54,)


def test_rummy_wild_create_ids_never_masked_on():
    env = BuracoEnv(rummy(2))
    _, info = env.reset(seed=21)
    rng = random.Random(2)
    from buraco.engine.actions import decode

    for _ in range(150):
        for a in info["legal_actions"]:
            act = decode(a, 24)
            if isinstance(act, CreateSeq):
                assert act.wild == 0
            if isinstance(act, CreateSet):
                assert act.wild == 0
        _, _, term, trunc, info = env.step(rng.choice(info["legal_actions"]))
        if term or trunc:
            break
