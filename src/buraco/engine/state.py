"""GameState: hands, melds, deck, trash, mortos, scores, phase (SPEC 01 §5).

`RoundState` is the mutable per-round ground truth; `MatchState` wraps it with
cumulative scores and the replay log. Dealing is deterministic given the
caller-supplied `random.Random`.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field
from enum import IntEnum

from buraco.cards import CardId, Rank, Suit, build_deck
from buraco.config import RED3_BONUS_AUTOREPLACE, RulesConfig
from buraco.engine.melds import Meld


class Phase(IntEnum):
    DRAW = 0
    PLAY = 1  # meld/discard window (SPEC 02 §2.5)
    TERMINAL = 2


class EndReason(IntEnum):
    BATER = 0
    STOCK_EXHAUSTED = 1


@dataclass
class RoundState:
    cfg: RulesConfig
    hands: list[Counter[CardId]]
    stock: list[CardId]  # draw from stock[-1]
    trash: list[CardId] = field(default_factory=list)  # top is trash[-1]
    melds: list[Meld] = field(default_factory=list)  # all sides; filter by .owner
    morto: list[tuple[CardId, ...] | None] = field(default_factory=list)  # by side
    morto_taken: list[bool] = field(default_factory=list)  # by side
    # --- Canasta mechanics (SPEC 06; inert in Buraco/Rummy/Biriba) ---
    red_threes: list[list[CardId]] = field(default_factory=list)  # face-up trays, by side
    pending_pile_card: CardId | None = None  # taken pile top that MUST be melded next
    pending_pile_pair_only: bool = False  # pile was frozen: forced meld = fresh natural set
    staged_points: int = 0  # meld points laid this turn by a not-yet-opened side
    opened_on_turn: list[int | None] = field(default_factory=list)  # by side
    initial_meld_min: list[int] = field(default_factory=list)  # threshold per side this round
    # --- turn machine ---
    current_player: int = 0
    phase: Phase = Phase.DRAW
    turn_number: int = 0
    just_drawn_from_pile: CardId | None = None  # top-card draw rules only
    pile_blocked_for_next: bool = False  # Canasta black-3; always False in Buraco
    frozen: bool = False  # Canasta frozen pile; always False in Buraco
    initial_meld_done: list[bool] = field(default_factory=list)  # by side
    # --- terminal bookkeeping ---
    round_over: bool = False
    went_out_side: int | None = None
    end_reason: EndReason | None = None

    def side_melds(self, side: int) -> list[Meld]:
        return [m for m in self.melds if m.owner == side]

    def hand_size(self, player: int) -> int:
        return sum(self.hands[player].values())

    def side_has_canastra(self, side: int, clean_required: bool = False) -> bool:
        min_size = self.cfg.meld.canastra_min_size
        return any(
            m.is_canastra(min_size) and (m.is_clean or not clean_required)
            for m in self.side_melds(side)
        )

    def side_canastra_count(self, side: int) -> int:
        min_size = self.cfg.meld.canastra_min_size
        return sum(1 for m in self.side_melds(side) if m.is_canastra(min_size))


@dataclass
class MatchState:
    cfg: RulesConfig
    seed: int
    round: RoundState
    scores: list[int] = field(default_factory=list)  # cumulative, by side
    round_index: int = 0
    action_log: list[int] = field(default_factory=list)  # encoded action ids
    match_over: bool = False
    winner_side: int | None = None


def deal_round(
    cfg: RulesConfig,
    rng: random.Random,
    first_player: int = 0,
    match_scores: list[int] | None = None,
) -> RoundState:
    """Shuffle and deal a fresh round. Deterministic given ``rng`` state.

    Deal order (fixed for reproducibility): shuffle; deal hands one card at a
    time round-robin from the top of the stock starting with ``first_player``;
    then each morto as a block; then the optional upcard. The remainder is the
    stock. ``match_scores`` picks each side's initial-meld threshold (Canasta).
    """
    table, deck_cfg, morto_cfg = cfg.table, cfg.deck, cfg.morto
    stock = build_deck(deck_cfg.deck_count, deck_cfg.printed_jokers)
    rng.shuffle(stock)

    hands: list[Counter[CardId]] = [Counter() for _ in range(table.num_players)]
    for _ in range(table.cards_per_player):
        for offset in range(table.num_players):
            player = (first_player + offset) % table.num_players
            hands[player][stock.pop()] += 1

    num_sides = table.num_sides
    if morto_cfg.count and morto_cfg.count != num_sides:
        raise ValueError(
            f"morto.count={morto_cfg.count} must be 0 or match num_sides={num_sides}"
        )
    morto: list[tuple[CardId, ...] | None] = []
    for _ in range(morto_cfg.count):
        morto.append(tuple(stock.pop() for _ in range(morto_cfg.size)))

    trash: list[CardId] = []
    frozen = False
    if cfg.discard_pile.initial_upcard:
        upcard = stock.pop()
        trash.append(upcard)
        if cfg.discard_pile.freeze_enabled and (
            cfg.is_wild_card(upcard) or _is_red_three(upcard)
        ):
            frozen = True

    thresholds = cfg.initial_meld.thresholds
    scores = match_scores or [0] * num_sides
    initial_meld_min = [
        max((pts for floor, pts in thresholds if scores[s] >= floor), default=0)
        for s in range(num_sides)
    ]

    state = RoundState(
        cfg=cfg,
        hands=hands,
        stock=stock,
        trash=trash,
        morto=morto,
        morto_taken=[False] * len(morto),
        current_player=first_player,
        frozen=frozen,
        initial_meld_done=[not cfg.initial_meld.enabled] * num_sides,
        red_threes=[[] for _ in range(num_sides)],
        opened_on_turn=[None] * num_sides,
        initial_meld_min=initial_meld_min,
    )
    if cfg.special_threes.red_three_mode == RED3_BONUS_AUTOREPLACE:
        for player in range(cfg.table.num_players):
            resolve_red_threes(state, player)
    return state


RED_THREE_IDS = (
    Rank.THREE + 13 * Suit.DIAMONDS,
    Rank.THREE + 13 * Suit.HEARTS,
)


def _is_red_three(ct: CardId) -> bool:
    return ct in RED_THREE_IDS


def resolve_red_threes(state: RoundState, player: int, replace: bool = True) -> bool:
    """Move red 3s from ``player``'s hand to the side tray, drawing
    replacements from the stock (SPEC 06 G2). No-op outside Canasta mode.

    Returns True when a red 3 could not be replaced (stock empty) — per pagat
    a red 3 drawn as the last stock card ends the round immediately; the
    caller must finish the round."""
    if state.cfg.special_threes.red_three_mode != RED3_BONUS_AUTOREPLACE:
        return False
    side = state.cfg.table.side(player)
    hand = state.hands[player]
    unreplaced = False
    while True:
        red = next((ct for ct in RED_THREE_IDS if hand.get(ct, 0) > 0), None)
        if red is None:
            return unreplaced
        if hand[red] == 1:
            del hand[red]
        else:
            hand[red] -= 1
        state.red_threes[side].append(red)
        if replace:
            if state.stock:
                hand[state.stock.pop()] += 1
            else:
                unreplaced = True


def dealt_multiset(state: RoundState) -> Counter[CardId]:
    """Full card multiset across all zones — the conservation invariant
    (SPEC 01 §8.1). Constant for the lifetime of a round."""
    total: Counter[CardId] = Counter()
    for hand in state.hands:
        total.update(hand)
    for meld in state.melds:
        total.update(meld.card_multiset())
    total.update(state.stock)
    total.update(state.trash)
    for packet in state.morto:
        if packet is not None:
            total.update(packet)
    for tray in state.red_threes:
        total.update(tray)
    return total
