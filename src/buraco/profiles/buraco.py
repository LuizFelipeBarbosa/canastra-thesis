"""Default Buraco profile: the user's house rules (docs/specs/00-decisions.md).

Open trash taken whole, 2s wild (no printed jokers), sequences and sets, morto
per side, any canastra (7+) unlocks bater, standard Brazilian scoring to 3000.
"""

from __future__ import annotations

from buraco.config import (
    MODE_INDIVIDUAL,
    MODE_TEAMS,
    RulesConfig,
    TableConfig,
)


def buraco(num_players: int = 2) -> RulesConfig:
    """Buraco rules for 2 players (individual) or 4 players (two teams of two).

    Everything except the table layout is the SPEC 04 default.
    """
    if num_players == 2:
        table = TableConfig(num_players=2, mode=MODE_INDIVIDUAL, team_of=1)
    elif num_players == 4:
        table = TableConfig(num_players=4, mode=MODE_TEAMS, team_of=2)
    else:
        raise ValueError(f"buraco supports 2 or 4 players, got {num_players}")
    return RulesConfig(name="buraco", table=table)
