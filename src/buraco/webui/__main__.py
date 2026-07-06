"""Launch the web GUI: `uv run buraco-gui` or `uv run python -m buraco.webui`.

Examples:
    uv run buraco-gui                                # setup screen in browser
    uv run buraco-gui --profile canasta --players 4 --seed 42 --episode match
    uv run buraco-gui --port 9000 --no-open
"""

from __future__ import annotations

import argparse
import threading
import webbrowser
from typing import Any

from buraco.webui.server import make_server


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default=None, help="buraco | canasta | rummy | biriba")
    parser.add_argument("--players", type=int, default=None)
    parser.add_argument("--human", type=int, default=None, help="seat you play (default 0)")
    parser.add_argument("--bots", default=None,
                        help="comma list per seat: random|heuristic (default heuristic)")
    parser.add_argument("--episode", default=None, choices=("round", "match"))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--port", type=int, default=8377)
    parser.add_argument("--no-open", action="store_true", help="don't open the browser")
    args = parser.parse_args(argv)

    defaults: dict[str, Any] = {}
    if args.profile is not None:
        defaults["profile"] = args.profile
    if args.players is not None:
        defaults["num_players"] = args.players
    if args.human is not None:
        defaults["human_seat"] = args.human
    if args.bots is not None:
        defaults["bots"] = [b.strip() for b in args.bots.split(",")]
    if args.episode is not None:
        defaults["episode"] = args.episode
    if args.seed is not None:
        defaults["seed"] = args.seed

    httpd, app = make_server(port=args.port, defaults=defaults)
    if defaults:  # game fully described on the CLI: start it before the browser opens
        app.new_game({})

    url = f"http://127.0.0.1:{httpd.server_address[1]}/"
    print(f"buraco-gui serving at {url}  (Ctrl-C to stop)")
    if not args.no_open:
        threading.Timer(0.3, webbrowser.open, [url]).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
