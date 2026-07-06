"""Stdlib HTTP layer: JSON routes over one GameSession + static assets.

Local single-human app: every state change is a response to a client POST
(frontend-driven stepping), so plain request/response suffices — no
websockets, no polling, no dependencies. Binds loopback only.
"""

from __future__ import annotations

import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from typing import Any
from urllib.parse import urlparse

from buraco.engine.turns import IllegalAction
from buraco.profiles import PROFILES, load_profile
from buraco.webui.session import AGENT_FACTORIES, GameSession, SessionError, StaleCursor

STATIC_NAME = re.compile(r"[A-Za-z0-9_-]+\.[A-Za-z0-9]+")
MIME = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
}
GAME_PARAMS = ("profile", "num_players", "human_seat", "bots", "episode", "seed")


def _allowed_players(name: str) -> list[int]:
    out = []
    for n in (2, 3, 4):
        try:
            load_profile(name, num_players=n)
            out.append(n)
        except Exception:
            continue
    return out


class WebApp:
    """Server-held state: the single active session + launch defaults."""

    def __init__(self, defaults: dict[str, Any] | None = None) -> None:
        self.defaults = defaults or {}
        self.session: GameSession | None = None
        self.lock = threading.Lock()

    def new_game(self, params: dict[str, Any]) -> dict[str, Any]:
        merged = {**self.defaults, **{k: v for k, v in params.items() if k in GAME_PARAMS}}
        session = GameSession(**merged)
        with self.lock:
            self.session = session
        return session.view()

    def meta(self) -> dict[str, Any]:
        return {
            "profiles": {name: {"players": _allowed_players(name)} for name in sorted(PROFILES)},
            "bots": sorted(AGENT_FACTORIES),
            "episodes": ["round", "match"],
            "defaults": self.defaults,
            "has_game": self.session is not None,
        }


class Handler(BaseHTTPRequestHandler):
    app: WebApp  # bound per-server via make_server

    def log_message(self, fmt: str, *args: Any) -> None:  # keep the terminal quiet
        pass

    # --- plumbing ----------------------------------------------------------

    def _send_json(self, code: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_static(self, name: str) -> None:
        if not STATIC_NAME.fullmatch(name):
            return self._send_json(404, {"error": "not found"})
        entry = resources.files("buraco.webui") / "static" / name
        if not entry.is_file():
            return self._send_json(404, {"error": "not found"})
        body = entry.read_bytes()
        ext = name[name.rfind("."):]
        self.send_response(200)
        self.send_header("Content-Type", MIME.get(ext, "application/octet-stream"))
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        data = json.loads(self.rfile.read(length))
        if not isinstance(data, dict):
            raise ValueError("body must be a JSON object")
        return data

    # --- routes ----------------------------------------------------------------

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            return self._send_static("index.html")
        if path.startswith("/static/"):
            return self._send_static(path[len("/static/"):])
        if path == "/api/meta":
            return self._send_json(200, self.app.meta())
        if path == "/api/game/state":
            session = self.app.session
            if session is None:
                return self._send_json(404, {"error": "no game yet; POST /api/game"})
            return self._send_json(200, session.view())
        self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            body = self._read_body()
        except (json.JSONDecodeError, ValueError) as exc:
            return self._send_json(400, {"error": f"invalid JSON body: {exc}"})
        if path == "/api/game":
            try:
                return self._send_json(200, self.app.new_game(body))
            except (SessionError, ValueError, TypeError) as exc:
                return self._send_json(400, {"error": str(exc)})
        if path == "/api/game/action":
            return self._mutate(body, human=True)
        if path == "/api/game/bot-step":
            return self._mutate(body, human=False)
        self._send_json(404, {"error": "not found"})

    def _mutate(self, body: dict[str, Any], human: bool) -> None:
        session = self.app.session
        if session is None:
            return self._send_json(404, {"error": "no game yet; POST /api/game"})
        if body.get("game_id") != session.game_id:
            return self._send_json(
                409, {"error": "game_id mismatch", "state": session.view()}
            )
        try:
            cursor = int(body["cursor"])
            if human:
                event, state = session.apply_human(int(body["action_id"]), cursor)
            else:
                event, state = session.bot_step(cursor)
        except StaleCursor as exc:
            return self._send_json(409, {"error": str(exc), "state": session.view()})
        except (SessionError, IllegalAction) as exc:
            return self._send_json(400, {"error": str(exc), "state": session.view()})
        except (KeyError, ValueError, TypeError) as exc:
            return self._send_json(400, {"error": f"bad request: {exc}"})
        self._send_json(200, {"event": event, "state": state})


def make_server(
    host: str = "127.0.0.1", port: int = 8377, defaults: dict[str, Any] | None = None
) -> tuple[ThreadingHTTPServer, WebApp]:
    app = WebApp(defaults)
    handler = type("BoundHandler", (Handler,), {"app": app})
    return ThreadingHTTPServer((host, port), handler), app
