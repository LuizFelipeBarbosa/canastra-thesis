"""Classic (US) Canasta profile (D9, SPEC 06): sets-only melds, conditional/
frozen discard pile with a forced top-card meld, red-3 bonus trays, black-3
pile blocking, initial-meld thresholds, canastas 500/300, match to 5000.

Deliberate simplifications (documented in SPEC 06): after taking the pile all
in-hand cards count toward the initial-meld threshold (D19); the pile's full
composition stays in the observation (perfect-memory convention — every card
in it arrived via a public discard).
"""

from __future__ import annotations

from buraco.config import (
    DISCARD_OUT_OPTIONAL,
    DRAW_CONDITIONAL_MELD_TOP,
    MODE_INDIVIDUAL,
    MODE_TEAMS,
    RED3_BONUS_AUTOREPLACE,
    DeckConfig,
    DiscardPileConfig,
    GoingOutConfig,
    InitialMeldConfig,
    MeldConfig,
    MortoConfig,
    RulesConfig,
    ScoringConfig,
    SpecialThreesConfig,
    TableConfig,
    TurnConfig,
    WildcardConfig,
)

CANASTA_CARD_POINTS = {
    "A": 20, "2": 20, "3": 5, "4": 5, "5": 5, "6": 5, "7": 5,
    "8": 10, "9": 10, "10": 10, "J": 10, "Q": 10, "K": 10, "JOKER": 50,
}

CANASTA_THRESHOLDS = ((-(10**9), 15), (0, 50), (1500, 90), (3000, 120))


def canasta(num_players: int = 4) -> RulesConfig:
    if num_players == 4:
        table = TableConfig(num_players=4, mode=MODE_TEAMS, team_of=2, cards_per_player=11)
        draw_count, min_canastras = 1, 1
    elif num_players == 2:
        table = TableConfig(
            num_players=2, mode=MODE_INDIVIDUAL, team_of=1, cards_per_player=15
        )
        draw_count, min_canastras = 2, 2
    else:
        raise ValueError(f"canasta supports 2 or 4 players, got {num_players}")
    return RulesConfig(
        name="canasta",
        table=table,
        deck=DeckConfig(deck_count=2, printed_jokers=4),
        wildcard=WildcardConfig(
            natural_two_in_suit=False,  # sets-only; a 2 is always wild
            wildcard_limit_per_meld=3,
            min_naturals_per_meld=2,
        ),
        meld=MeldConfig(
            allow_sequences=False,
            canastra_bonus_clean=500,
            canastra_bonus_dirty=300,
        ),
        morto=MortoConfig(count=0),
        discard_pile=DiscardPileConfig(
            draw_rule=DRAW_CONDITIONAL_MELD_TOP,
            initial_upcard=True,
            freeze_enabled=True,
            frozen_needs_two_naturals=True,
        ),
        going_out=GoingOutConfig(
            require_canastra=True,
            require_morto_taken=False,
            discard_to_go_out=DISCARD_OUT_OPTIONAL,
            go_out_bonus=100,
            concealed_bonus=100,
            go_out_min_canastras=min_canastras,
        ),
        initial_meld=InitialMeldConfig(enabled=True, thresholds=CANASTA_THRESHOLDS),
        special_threes=SpecialThreesConfig(
            red_three_mode=RED3_BONUS_AUTOREPLACE,
            red_three_bonus=100,
            red_three_all_bonus=400,
            red_three_negative_if_no_meld=True,
            black_three_blocks_pile=True,
            black_three_meld_only_going_out=True,
        ),
        scoring=ScoringConfig(card_points=dict(CANASTA_CARD_POINTS), match_target=5000),
        turn=TurnConfig(draw_count=draw_count),
    )
