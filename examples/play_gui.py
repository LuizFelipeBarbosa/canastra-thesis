"""Play in the browser: thin shim over `python -m buraco.webui`.

    uv run python examples/play_gui.py --profile buraco --players 2
"""

from buraco.webui.__main__ import main

if __name__ == "__main__":
    main()
