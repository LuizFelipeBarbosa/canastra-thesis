"""RulesConfig dataclass tree: the single source of rule truth (SPEC 04).

All classes are frozen; a profile constructs one tree and the engine only ever
reads it. String-valued policy fields use module-level constants rather than
enums so configs serialize to plain JSON verbatim.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from buraco.cards import Rank

# --- policy constants -------------------------------------------------------

MODE_INDIVIDUAL = "INDIVIDUAL"
MODE_TEAMS = "TEAMS"

ACE_HIGH_OR_LOW = "HIGH_OR_LOW"
ACE_LOW_ONLY = "LOW_ONLY"
ACE_HIGH_ONLY = "HIGH_ONLY"

WILD_RELOCATE_EXTEND = "RELOCATE_EXTEND"
WILD_TO_HAND = "TO_HAND"

VISIBILITY_FULL_OPEN = "FULL_OPEN"
VISIBILITY_TOP_ONLY = "TOP_ONLY"

DRAW_WHOLE_PILE = "WHOLE_PILE_UNCONDITIONAL"
DRAW_TOP_CARD = "TOP_CARD"
DRAW_CONDITIONAL_MELD_TOP = "CONDITIONAL_MELD_TOP"

DISCARD_OUT_REQUIRED = "REQUIRED"
DISCARD_OUT_OPTIONAL = "OPTIONAL"
DISCARD_OUT_FORBIDDEN = "FORBIDDEN"

MORTO_PICKUP_ON_EMPTY = "ON_EMPTY_FORCED"

HAND_PENALTY_SELF_NEGATIVE = "SELF_NEGATIVE"
HAND_PENALTY_OPPONENT_POSITIVE = "OPPONENT_POSITIVE"

EPISODE_ROUND = "ROUND"
EPISODE_MATCH = "MATCH"

RED3_NONE = "NONE"
RED3_BONUS_AUTOREPLACE = "BONUS_AUTOREPLACE"

EXHAUSTION_END_ROUND = "END_ROUND"
EXHAUSTION_CONVERT_MORTO = "CONVERT_MORTO"

BURACO_CARD_POINTS: Mapping[str, int] = {
    "A": 15, "2": 10, "3": 5, "4": 5, "5": 5, "6": 5, "7": 5,
    "8": 10, "9": 10, "10": 10, "J": 10, "Q": 10, "K": 10, "JOKER": 20,
}

# --- config tree -------------------------------------------------------------


@dataclass(frozen=True)
class TableConfig:
    num_players: int = 2
    mode: str = MODE_INDIVIDUAL
    team_of: int = 1
    cards_per_player: int = 11

    def __post_init__(self) -> None:
        if self.num_players not in (2, 3, 4):
            raise ValueError(f"unsupported num_players: {self.num_players}")
        if self.num_players % self.team_of:
            raise ValueError("num_players must be divisible by team_of")

    @property
    def num_sides(self) -> int:
        return self.num_players // self.team_of

    def side(self, player: int) -> int:
        """Side (team in 4p, player in 2p) owning seat ``player``."""
        return player % self.num_sides


@dataclass(frozen=True)
class DeckConfig:
    deck_count: int = 2
    printed_jokers: int = 0

    @property
    def total_cards(self) -> int:
        return self.deck_count * 52 + self.printed_jokers


@dataclass(frozen=True)
class WildcardConfig:
    wild_ranks: frozenset[Rank] = frozenset({Rank.TWO})
    jokers_wild: bool = True
    natural_two_in_suit: bool = True
    wildcard_limit_per_meld: int = 1
    min_naturals_per_meld: int = 2
    wild_relocation: str = WILD_RELOCATE_EXTEND


@dataclass(frozen=True)
class MeldConfig:
    allow_sequences: bool = True
    allow_sets: bool = True
    min_meld_size: int = 3
    ace_policy: str = ACE_HIGH_OR_LOW
    allow_wrap: bool = False
    unique_set_rank_per_side: bool = True
    canastra_min_size: int = 7
    canastra_bonus_clean: int = 200
    canastra_bonus_dirty: int = 100
    max_meld_slots: int = 24


@dataclass(frozen=True)
class MortoConfig:
    count: int = 2
    size: int = 11
    pickup: str = MORTO_PICKUP_ON_EMPTY
    untaken_penalty: int = 100


@dataclass(frozen=True)
class DiscardPileConfig:
    visibility: str = VISIBILITY_FULL_OPEN
    draw_rule: str = DRAW_WHOLE_PILE
    initial_upcard: bool = False
    freeze_enabled: bool = False
    frozen_needs_two_naturals: bool = False
    no_immediate_redraw_discard: bool = False


@dataclass(frozen=True)
class GoingOutConfig:
    require_canastra: bool = True
    require_clean_canastra: bool = False
    require_morto_taken: bool = True
    discard_to_go_out: str = DISCARD_OUT_REQUIRED
    go_out_bonus: int = 100
    concealed_bonus: int = 0
    go_out_min_canastras: int = 1


@dataclass(frozen=True)
class InitialMeldConfig:
    enabled: bool = False
    thresholds: tuple[tuple[int, int], ...] = ()


@dataclass(frozen=True)
class SpecialThreesConfig:
    red_three_mode: str = RED3_NONE
    red_three_bonus: int = 0
    red_three_all_bonus: int = 0
    red_three_negative_if_no_meld: bool = False
    black_three_blocks_pile: bool = False
    black_three_meld_only_going_out: bool = False


@dataclass(frozen=True)
class ScoringConfig:
    card_points: Mapping[str, int] = field(default_factory=lambda: dict(BURACO_CARD_POINTS))
    hand_penalty_mode: str = HAND_PENALTY_SELF_NEGATIVE
    match_target: int = 3000
    episode: str = EPISODE_ROUND
    reward_scale: float = 0.001


@dataclass(frozen=True)
class TurnConfig:
    draw_count: int = 1
    no_op_available: bool = False
    deck_exhaustion_policy: str = EXHAUSTION_END_ROUND
    truncation_cap: int = 400


@dataclass(frozen=True)
class RulesConfig:
    name: str = "buraco"
    table: TableConfig = field(default_factory=TableConfig)
    deck: DeckConfig = field(default_factory=DeckConfig)
    wildcard: WildcardConfig = field(default_factory=WildcardConfig)
    meld: MeldConfig = field(default_factory=MeldConfig)
    morto: MortoConfig = field(default_factory=MortoConfig)
    discard_pile: DiscardPileConfig = field(default_factory=DiscardPileConfig)
    going_out: GoingOutConfig = field(default_factory=GoingOutConfig)
    initial_meld: InitialMeldConfig = field(default_factory=InitialMeldConfig)
    special_threes: SpecialThreesConfig = field(default_factory=SpecialThreesConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    turn: TurnConfig = field(default_factory=TurnConfig)

    def card_value(self, ct: int) -> int:
        """Point value of a card type under this profile's scoring table."""
        from buraco.cards import rank_name

        return self.scoring.card_points[rank_name(ct)]

    def is_wild_card(self, ct: int) -> bool:
        """Whether ``ct`` is a wildcard type (context-free; the natural-2
        positional exception is applied by meld validation, not here)."""
        from buraco.cards import JOKER, id_rank

        if ct == JOKER:
            return self.wildcard.jokers_wild
        return id_rank(ct) in self.wildcard.wild_ranks
