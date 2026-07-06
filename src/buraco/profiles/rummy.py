"""Basic Rummy profile (SPEC 05): single 52-card deck, no wildcards, top-card
draw, no morto, no canastra requirement, winner collects opponents' pips."""

from __future__ import annotations

from buraco.config import (
    ACE_LOW_ONLY,
    DISCARD_OUT_OPTIONAL,
    DRAW_TOP_CARD,
    HAND_PENALTY_OPPONENT_POSITIVE,
    MODE_INDIVIDUAL,
    DeckConfig,
    DiscardPileConfig,
    GoingOutConfig,
    MeldConfig,
    MortoConfig,
    RulesConfig,
    ScoringConfig,
    TableConfig,
    WildcardConfig,
)

RUMMY_CARD_POINTS = {
    "A": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7,
    "8": 8, "9": 9, "10": 10, "J": 10, "Q": 10, "K": 10, "JOKER": 0,
}


def rummy(num_players: int = 2) -> RulesConfig:
    # MODE_INDIVIDUAL with 3-4 players yields 3-4 sides, but the env payoffs,
    # observations, and GUI all model exactly two sides (own vs opponent).
    if num_players != 2:
        raise ValueError(f"rummy profile is 2-player only, got {num_players}")
    cards = 10
    return RulesConfig(
        name="rummy",
        table=TableConfig(
            num_players=num_players, mode=MODE_INDIVIDUAL, team_of=1, cards_per_player=cards
        ),
        deck=DeckConfig(deck_count=1, printed_jokers=0),
        wildcard=WildcardConfig(
            wild_ranks=frozenset(), jokers_wild=False, natural_two_in_suit=False,
            wildcard_limit_per_meld=0, min_naturals_per_meld=3,
        ),
        meld=MeldConfig(
            ace_policy=ACE_LOW_ONLY,
            canastra_bonus_clean=0,
            canastra_bonus_dirty=0,
        ),
        morto=MortoConfig(count=0),
        discard_pile=DiscardPileConfig(
            draw_rule=DRAW_TOP_CARD, initial_upcard=True, no_immediate_redraw_discard=True
        ),
        going_out=GoingOutConfig(
            require_canastra=False,
            require_morto_taken=False,
            discard_to_go_out=DISCARD_OUT_OPTIONAL,
            go_out_bonus=0,
        ),
        scoring=ScoringConfig(
            card_points=dict(RUMMY_CARD_POINTS),
            hand_penalty_mode=HAND_PENALTY_OPPONENT_POSITIVE,
            match_target=100,
        ),
    )
