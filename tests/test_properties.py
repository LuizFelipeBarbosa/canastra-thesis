"""Property-based tests (hypothesis): planners never lie, codec is total."""

from collections import Counter

from hypothesis import given
from hypothesis import strategies as st

from buraco.cards import NUM_CARD_TYPES, Rank, Suit
from buraco.engine.actions import action_space_size, decode, encode
from buraco.engine.melds import (
    apply_add,
    create_sequence,
    create_set,
    plan_add,
    plan_sequence,
    plan_set,
    validate_meld,
)
from buraco.profiles import buraco

CFG = buraco(2)

hands = st.lists(
    st.integers(min_value=0, max_value=NUM_CARD_TYPES - 1), min_size=0, max_size=18
).map(Counter)


@given(hand=hands, suit=st.sampled_from(list(Suit)),
       start=st.integers(min_value=1, max_value=12),
       wild=st.integers(min_value=0, max_value=3))
def test_sequence_plan_always_applies(hand, suit, start, wild):
    plan = plan_sequence(CFG, hand, suit, start, wild)
    if plan is None:
        return
    assert Counter(plan.consumed) <= hand  # never plans cards it doesn't have
    before = Counter(hand)
    meld = create_sequence(CFG, hand, 0, 0, suit, start, wild)
    validate_meld(CFG, meld)
    assert hand + Counter(plan.consumed) == before  # exact consumption


@given(hand=hands, rank=st.sampled_from(list(Rank)),
       wild=st.integers(min_value=0, max_value=2))
def test_set_plan_always_applies(hand, rank, wild):
    plan = plan_set(CFG, hand, rank, wild)
    if plan is None:
        return
    assert Counter(plan.consumed) <= hand
    meld = create_set(CFG, hand, 0, 0, rank, wild)
    validate_meld(CFG, meld)


@given(hand=hands, suit=st.sampled_from(list(Suit)),
       start=st.integers(min_value=1, max_value=12),
       extra=st.integers(min_value=0, max_value=NUM_CARD_TYPES - 1))
def test_add_plan_always_applies(hand, suit, start, extra):
    if plan_sequence(CFG, hand, suit, start, 0) is None:
        return
    meld = create_sequence(CFG, hand, 0, 0, suit, start, 0)
    hand[extra] += 1
    if plan_add(CFG, hand, meld, extra) is None:
        return
    apply_add(CFG, hand, meld, extra)
    validate_meld(CFG, meld)


@given(slots=st.integers(min_value=1, max_value=40),
       data=st.data())
def test_codec_total_inverse_for_any_slot_budget(slots, data):
    a = data.draw(st.integers(min_value=0, max_value=action_space_size(slots) - 1))
    assert encode(decode(a, slots), slots) == a
