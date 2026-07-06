"""Greek Biriba profile (SPEC 05): two decks plus four jokers, biribakia (dead
hands) that convert into fresh stock on exhaustion, biriba = 7+ meld.

Canastra scoring uses the simplified flat clean/dirty bonuses; the full pagat
tiered/bonus-suit (κόζι) table is engine gap G4 and intentionally deferred.
"""

from __future__ import annotations

from buraco.config import (
    BURACO_CARD_POINTS,
    EXHAUSTION_CONVERT_MORTO,
    MODE_INDIVIDUAL,
    MODE_TEAMS,
    DeckConfig,
    DiscardPileConfig,
    RulesConfig,
    ScoringConfig,
    TableConfig,
    TurnConfig,
)


def biriba(num_players: int = 4) -> RulesConfig:
    if num_players == 2:
        table = TableConfig(num_players=2, mode=MODE_INDIVIDUAL, team_of=1)
    elif num_players == 4:
        table = TableConfig(num_players=4, mode=MODE_TEAMS, team_of=2)
    else:
        raise ValueError(f"biriba supports 2 or 4 players, got {num_players}")
    return RulesConfig(
        name="biriba",
        table=table,
        deck=DeckConfig(deck_count=2, printed_jokers=4),
        discard_pile=DiscardPileConfig(
            initial_upcard=True, no_immediate_redraw_discard=True
        ),
        scoring=ScoringConfig(card_points=dict(BURACO_CARD_POINTS), match_target=5000),
        turn=TurnConfig(deck_exhaustion_policy=EXHAUSTION_CONVERT_MORTO),
    )
