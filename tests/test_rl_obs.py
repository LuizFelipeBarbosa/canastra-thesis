"""ObsSpec: flat_dim invariance, canonical order, one-hot and scaling correctness."""

from __future__ import annotations

import numpy as np

from buraco.cards import CARD_SPACE, PAD
from buraco.env.env import BuracoEnv
from buraco.profiles import load_profile
from buraco.rl.obs import ObsSpec


def test_flat_dim_invariant_across_players_and_profiles():
    specs = [
        ObsSpec.from_cfg(load_profile("buraco", num_players=2)),
        ObsSpec.from_cfg(load_profile("buraco", num_players=4)),
        ObsSpec.from_cfg(load_profile("canasta", num_players=2)),
    ]
    assert len({s.flat_dim for s in specs}) == 1
    # Same profile, different player count: identical spec (shared checkpoints).
    # Canasta differs only in normalization constants (printed jokers change
    # deck.total_cards), never in layout.
    assert specs[0] == specs[1]
    assert [f.offset for f in specs[0].fields] == [f.offset for f in specs[2].fields]


def test_field_order_sorted_and_offsets_contiguous():
    spec = ObsSpec.from_cfg(load_profile("buraco", num_players=2))
    names = [f.name for f in spec.fields]
    assert names == sorted(names)
    offset = 0
    for f in spec.fields:
        assert f.offset == offset
        offset += f.flat_size
    assert offset == spec.flat_dim


def test_flatten_dtype_and_determinism():
    cfg = load_profile("buraco", num_players=2)
    spec = ObsSpec.from_cfg(cfg)
    env = BuracoEnv(cfg)
    obs, _ = env.reset(seed=7)
    a, b = spec.flatten(obs), spec.flatten(obs)
    assert a.dtype == np.float32 and a.shape == (spec.flat_dim,)
    assert np.array_equal(a, b)


def test_card_id_onehot_pad_placement():
    cfg = load_profile("buraco", num_players=2)
    spec = ObsSpec.from_cfg(cfg)
    env = BuracoEnv(cfg)
    obs, _ = env.reset(seed=0)
    flat = spec.flatten(obs)
    top_k = next(f for f in spec.fields if f.name == "trash_top_k")
    # Fresh deal: trash empty, so every slot holds PAD and one-hots at PAD fire.
    assert np.all(obs["trash_top_k"] == PAD)
    for i in range(obs["trash_top_k"].size):
        row = flat[top_k.offset + i * CARD_SPACE : top_k.offset + (i + 1) * CARD_SPACE]
        assert row[PAD] == 1.0 and row.sum() == 1.0


def test_count_fields_normalized():
    cfg = load_profile("buraco", num_players=2)
    spec = ObsSpec.from_cfg(cfg)
    env = BuracoEnv(cfg)
    obs, _ = env.reset(seed=3)
    flat = spec.flatten(obs)
    deck = next(f for f in spec.fields if f.name == "deck_size")
    assert 0.0 <= flat[deck.offset] <= 1.0
    hand = next(f for f in spec.fields if f.name == "hand_size")
    assert 0.0 < flat[hand.offset] <= 1.0


def test_spec_dict_round_trip():
    spec = ObsSpec.from_cfg(load_profile("buraco", num_players=4))
    assert ObsSpec.from_dict(spec.to_dict()) == spec
