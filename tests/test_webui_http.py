"""HTTP smoke tests: the stdlib server end-to-end over urllib."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import pytest

from buraco.webui.server import make_server


@pytest.fixture()
def server():
    httpd, app = make_server(port=0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    yield base, app
    httpd.shutdown()
    httpd.server_close()


def get(base: str, path: str):
    try:
        with urllib.request.urlopen(base + path) as resp:
            return resp.status, resp.headers.get_content_type(), resp.read()
    except urllib.error.HTTPError as err:
        return err.code, err.headers.get_content_type(), err.read()


def post(base: str, path: str, payload: dict):
    req = urllib.request.Request(
        base + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as err:
        return err.code, json.loads(err.read())


def test_static_meta_and_traversal(server):
    base, _app = server
    status, ctype, body = get(base, "/")
    assert status == 200 and ctype == "text/html" and body
    status, _, body = get(base, "/api/meta")
    meta = json.loads(body)
    assert status == 200 and set(meta["profiles"]) == {"biriba", "buraco", "canasta", "rummy"}
    assert meta["profiles"]["rummy"]["players"] == [2]  # 3-4p rummy would break 2-side model
    assert meta["profiles"]["buraco"]["players"] == [2, 4]
    assert not meta["has_game"]
    for bad in ("/static/../pyproject.toml", "/static/..%2f..%2fpyproject.toml",
                "/static/nope.js", "/nope"):
        status, _, _ = get(base, bad)
        assert status == 404, bad


def test_game_flow_and_guards(server):
    base, _app = server
    status, _, _ = get(base, "/api/game/state")
    assert status == 404

    status, state = post(base, "/api/game", {"profile": "buraco", "seed": 5})
    assert status == 200 and state["to_play_is_human"]
    gid, cursor = state["game_id"], state["cursor"]

    status, err = post(base, "/api/game", {"profile": "nope"})
    assert status == 400 and "profile" in err["error"]

    status, err = post(base, "/api/game/bot-step", {"game_id": gid, "cursor": cursor})
    assert status == 400  # human to play

    status, err = post(base, "/api/game/action",
                       {"game_id": "wrong", "cursor": cursor, "action_id": 0})
    assert status == 409 and "state" in err

    draw = state["legal"][0]
    status, resp = post(base, "/api/game/action",
                        {"game_id": gid, "cursor": cursor, "action_id": draw["id"]})
    assert status == 200 and resp["event"]["family"] == "draw_deck"
    assert resp["state"]["cursor"] == cursor + 1

    status, err = post(base, "/api/game/action",  # replay of the same request
                       {"game_id": gid, "cursor": cursor, "action_id": draw["id"]})
    assert status == 409 and err["state"]["cursor"] == cursor + 1


def test_full_game_over_http(server):
    base, _app = server
    _, state = post(base, "/api/game", {"profile": "buraco", "seed": 42})
    gid = state["game_id"]
    for _ in range(20_000):
        if state["done"]:
            break
        if state["to_play_is_human"]:
            status, resp = post(base, "/api/game/action", {
                "game_id": gid, "cursor": state["cursor"],
                "action_id": state["legal"][0]["id"],
            })
        else:
            status, resp = post(base, "/api/game/bot-step",
                                {"game_id": gid, "cursor": state["cursor"]})
        assert status == 200, resp
        state = resp["state"]
    assert state["done"]
    assert state["truncated"] or state["last_round_summary"] is not None
