"""GameSession / view / event layer tests (SPEC-free: pins the GUI contract).

The frontend is dumb by design; these tests pin everything it relies on:
JSON-serializable payloads, hidden-info discipline, legal-action fidelity,
cursor idempotence, event diffs, and round-summary arithmetic.
"""

from __future__ import annotations

import json

import pytest

from buraco.agents.random_agent import RandomAgent
from buraco.engine.actions import Add, CreateSeq, CreateSet, Discard, decode
from buraco.engine.scoring import round_scores
from buraco.engine.serialize import state_hash
from buraco.engine.turns import IllegalAction
from buraco.profiles import PROFILES, load_profile
from buraco.webui.session import GameSession, SessionError, StaleCursor

VIEW_KEYS = {
    "game_id", "cursor", "seed", "profile", "episode", "num_players", "human_seat",
    "human_side", "partner_seat", "bots", "round_index", "turn_number", "phase",
    "to_play", "to_play_is_human", "done", "terminated", "truncated", "hand",
    "all_hand_sizes", "deck_size", "trash", "melds", "morto", "red_threes",
    "canasta", "scores", "rules", "legal", "payoffs", "winner_side",
    "last_round_summary",
}


def make_session(profile="buraco", players=2, seed=0, episode="round", human_seat=0):
    return GameSession(
        profile=profile, num_players=players, human_seat=human_seat,
        episode=episode, seed=seed,
    )


def drive(session, max_steps=60_000, on_step=None):
    """Play to completion; the human seat is driven by a seeded random policy
    through apply_human so the legality path is exercised."""
    human_policy = RandomAgent(seed=123)
    events = []
    while not session.done and len(events) < max_steps:
        view = session.view()
        if view["to_play_is_human"]:
            aid = human_policy.act(
                session.env.observe_raw(session.human_seat),
                session.env.legal_actions(), session.cfg,
            )
            event, view_after = session.apply_human(aid, view["cursor"])
        else:
            event, view_after = session.bot_step(view["cursor"])
        events.append(event)
        if on_step is not None:
            on_step(event, view_after)
    assert session.done, "game did not finish within the step budget"
    return events


def _combos() -> list[tuple[str, int]]:
    out = []
    for name in sorted(PROFILES):
        for players in (2, 4):
            try:
                load_profile(name, num_players=players)
            except ValueError:
                continue  # e.g. rummy is 2-player only
            out.append((name, players))
    return out


@pytest.mark.parametrize("profile,players", _combos())
def test_soak_full_game_json_clean(profile, players):
    session = make_session(profile=profile, players=players, seed=7)
    raw = session.env.observe_raw

    def on_step(event, view):
        json.dumps(event)
        if event["seq"] % 10 == 0:
            json.dumps(view)
        # Add.slot indexes own melds in creation order: view must mirror
        # observe_raw's own_melds order exactly, never re-sorted.
        own = raw(session.human_seat)["own_melds"]
        assert [m["size"] for m in view["melds"]["own"]] == [m["size"] for m in own]

    events = drive(session, on_step=on_step)
    json.dumps(session.view())
    assert events[-1]["round_ended"] or session.truncated


def test_view_schema_and_hidden_info():
    session = make_session(seed=11)
    view = session.view()
    assert set(view) == VIEW_KEYS
    assert view["hand"] == sorted(session.env.state.hands[0].elements())
    assert view["cursor"] == 0

    def on_step(event, view):
        if not event["actor_is_human"]:
            # A bot action never touches the human hand, and bot deck draws
            # are counts only — no hidden card ids in the event.
            assert event["hand_added"] == [] and event["hand_removed"] == []
            if event["family"] == "draw_deck":
                assert event["drawn_hidden"] >= 1 and event["card"] is None

    drive(session, on_step=on_step)


def test_legal_entries_match_engine():
    session = make_session(seed=3)
    view = session.view()
    assert view["to_play_is_human"]
    legal = view["legal"]
    slots = session.cfg.meld.max_meld_slots
    hand = dict(session.env.state.hands[0])
    assert [e["id"] for e in legal] == session.env.legal_actions()
    for entry in legal:
        action = decode(entry["id"], slots)
        match action:
            case CreateSeq(suit=s, start=st, wild=w):
                assert (entry["suit"], entry["start"], entry["wild"]) == (int(s), st, w)
                assert entry["uses"], "legal create must have a uses preview"
            case CreateSet(rank=r, wild=w):
                assert (entry["rank"], entry["wild"]) == (int(r), w)
                assert entry["uses"]
            case Add(slot=sl, ct=c):
                assert (entry["slot"], entry["card"]) == (sl, int(c))
            case Discard(ct=c):
                assert entry["card"] == int(c)
        for c in entry.get("uses", []):
            assert hand.get(c, 0) >= entry["uses"].count(c)


def test_cursor_guard_and_turn_guards():
    session = make_session(seed=5)
    before = state_hash(session.env.state)
    with pytest.raises(StaleCursor):
        session.apply_human(0, cursor=99)
    with pytest.raises(StaleCursor):
        session.bot_step(cursor=99)
    # Human to play: bot_step must refuse rather than act for the human.
    with pytest.raises(SessionError):
        session.bot_step(cursor=0)
    assert state_hash(session.env.state) == before
    assert session.cursor == 0

    session.apply_human(session.env.legal_actions()[0], cursor=0)
    with pytest.raises(StaleCursor):  # replayed request is a harmless no-op
        session.apply_human(0, cursor=0)


def test_illegal_action_rejected_cleanly():
    session = make_session(seed=5)
    legal = set(session.env.legal_actions())
    illegal = next(a for a in range(session.env.num_actions) if a not in legal)
    before = state_hash(session.env.state)
    with pytest.raises(IllegalAction):
        session.apply_human(illegal, cursor=0)
    assert state_hash(session.env.state) == before


def test_draw_and_discard_events():
    session = make_session(seed=9)
    event, view = session.apply_human(0, cursor=0)  # id 0 = draw from stock
    assert event["family"] == "draw_deck" and event["actor_is_human"]
    assert event["diff"]["deck_size"] == -1
    assert len(event["hand_added"]) == 1
    assert event["diff"]["hand_sizes"][0] == 1
    assert view["phase"] == "PLAY"

    discard = next(e for e in view["legal"] if e["family"] == "discard")
    event, view = session.apply_human(discard["id"], cursor=view["cursor"])
    assert event["family"] == "discard" and event["card"] == discard["card"]
    assert event["diff"]["trash_len"] == 1
    assert event["hand_removed"] == [discard["card"]]
    assert event["to_play_after"] == 1 and event["phase_after"] == "DRAW"


def test_pile_take_event_reveals_public_cards_only():
    for seed in range(20):
        session = make_session(seed=seed)
        taken = []
        drive(session, on_step=lambda e, v: taken.append(e) if e["family"] == "draw_trash" else None)
        if taken:
            event = taken[0]
            assert event["cards_taken"], "whole-pile take must list the public pile"
            assert event["diff"]["trash_len"] == -len(event["cards_taken"])
            return
    pytest.fail("no pile take occurred in 20 seeded games")


def test_round_summary_matches_engine_scores():
    for seed in range(20):
        session = make_session(seed=seed)
        events = drive(session)
        if session.truncated:
            continue
        summary = events[-1]["round_summary"]
        assert summary is not None
        assert summary["totals"] == round_scores(session.env.state)
        assert summary["match_scores_after"] == session.env.match_scores
        assert summary["match_over"] is True
        for per_side, total in zip(summary["per_side"], summary["totals"]):
            assert per_side["total"] == total
            assert all(h["cards"] is not None for h in per_side["hands"])
        return
    pytest.fail("no untruncated round in 20 seeded games")


def test_match_mode_round_transitions():
    session = make_session(seed=2, episode="match")
    transitions = []

    def on_step(event, view):
        if event["round_ended"]:
            transitions.append((event, view))

    drive(session, max_steps=200_000, on_step=on_step)
    assert transitions, "match produced no completed rounds"
    for event, view in transitions:
        summary = event["round_summary"]
        assert summary is not None
        if event["new_round"]:
            # Env dealt the next round inside the same step: the event's
            # summary describes the finished round, the view the fresh one.
            assert not event["terminated"]
            assert view["round_index"] == summary["round_index"] + 1
            assert view["turn_number"] == 0
        else:
            assert event["terminated"]
            assert summary["match_over"]
    final = session.view()
    assert final["done"] and max(final["scores"]["match"]) >= 0
    assert session.env.match_scores == transitions[-1][0]["round_summary"]["match_scores_after"]
