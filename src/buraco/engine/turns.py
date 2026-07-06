"""Turn state machine: draw phase then meld/discard phase (SPEC 01 §7).

`apply_action` is the single mutation entry point for a round. It re-checks
legality through the same predicates `legal.py` enumerates with, so an action
outside `legal_actions(state)` always raises `IllegalAction` and never
corrupts state.
"""

from __future__ import annotations

from collections import Counter

from buraco.cards import id_rank
from buraco.config import (
    DISCARD_OUT_REQUIRED,
    DRAW_CONDITIONAL_MELD_TOP,
    DRAW_TOP_CARD,
    DRAW_WHOLE_PILE,
    EXHAUSTION_CONVERT_MORTO,
)
from buraco.engine.actions import (
    Action,
    Add,
    CreateSeq,
    CreateSet,
    Discard,
    DrawDeck,
    DrawTrash,
    EndRound,
    GoOut,
)
from buraco.engine.legal import (
    add_result_allowed,
    bater_ready,
    can_take_conditional_pile,
    discard_allowed,
    effective_frozen,
    is_black_three,
    meld_result_allowed,
    morto_available,
    pile_draw_allowed,
    play_action_extra_rejection,
)
from buraco.engine.melds import MeldError, MeldKind, apply_add, create_sequence, create_set
from buraco.engine.scoring import meld_points
from buraco.engine.state import (
    RED_THREE_IDS,
    EndReason,
    Phase,
    RoundState,
    resolve_red_threes,
)


class IllegalAction(Exception):
    """The submitted action is not legal in the current state."""


def apply_action(state: RoundState, action: Action) -> None:
    if state.round_over:
        raise IllegalAction("round is over; reset required")
    if state.phase is Phase.DRAW:
        _apply_draw(state, action)
    elif state.phase is Phase.PLAY:
        _apply_play(state, action)
    else:  # pragma: no cover - TERMINAL guarded by round_over
        raise IllegalAction("terminal state")


# --- draw phase ---------------------------------------------------------------


def _apply_draw(state: RoundState, action: Action) -> None:
    player = state.current_player
    if isinstance(action, DrawDeck):
        if not state.stock:
            raise IllegalAction("stock is empty")
        for _ in range(min(state.cfg.turn.draw_count, len(state.stock))):
            state.hands[player][state.stock.pop()] += 1
        if resolve_red_threes(state, player):
            # A red 3 drawn as the last stock card ends the round (pagat).
            _finish(state, EndReason.STOCK_EXHAUSTED, None)
            return
        _maybe_convert_morto(state)
        state.pile_blocked_for_next = False
        state.phase = Phase.PLAY
    elif isinstance(action, DrawTrash):
        if not pile_draw_allowed(state):
            raise IllegalAction("cannot draw from the trash pile")
        rule = state.cfg.discard_pile.draw_rule
        if rule == DRAW_WHOLE_PILE:
            state.hands[player].update(state.trash)
            state.trash.clear()
        elif rule == DRAW_TOP_CARD:
            ct = state.trash.pop()
            state.hands[player][ct] += 1
            state.just_drawn_from_pile = ct
        elif rule == DRAW_CONDITIONAL_MELD_TOP:  # SPEC 06 G1
            if not can_take_conditional_pile(state):
                raise IllegalAction("pile conditions not met")
            side = state.cfg.table.side(player)
            top = state.trash[-1]
            pair_only = effective_frozen(state, side)
            state.hands[player].update(state.trash)
            state.trash.clear()
            state.frozen = False
            # A red 3 buried in the pile (initial upcard) goes to the tray
            # without replacement.
            for red in RED_THREE_IDS:
                while state.hands[player].get(red, 0) > 0:
                    if state.hands[player][red] == 1:
                        del state.hands[player][red]
                    else:
                        state.hands[player][red] -= 1
                    state.red_threes[side].append(red)
            state.pending_pile_card = top
            state.pending_pile_pair_only = pair_only
        else:
            raise IllegalAction(f"unknown draw rule {rule}")
        state.pile_blocked_for_next = False
        state.phase = Phase.PLAY
    elif isinstance(action, EndRound):
        if state.stock:
            raise IllegalAction("END_ROUND only when the stock is empty")
        _finish(state, EndReason.STOCK_EXHAUSTED, None)
    else:
        raise IllegalAction(f"expected a draw-phase action, got {action!r}")


def _maybe_convert_morto(state: RoundState) -> None:
    """CONVERT_MORTO exhaustion policy: an untaken morto becomes the new stock
    the moment the stock empties (D5 alternative; Buraco default never hits
    this). The side does not count as having taken its morto."""
    if state.stock or state.cfg.turn.deck_exhaustion_policy != EXHAUSTION_CONVERT_MORTO:
        return
    for side, packet in enumerate(state.morto):
        if packet is not None and not state.morto_taken[side]:
            state.stock = list(packet)
            state.morto[side] = None
            return


# --- meld/discard phase --------------------------------------------------------


def _apply_play(state: RoundState, action: Action) -> None:
    cfg = state.cfg
    player = state.current_player
    side = cfg.table.side(player)
    hand = state.hands[player]
    hand_size = state.hand_size(player)

    rejection = play_action_extra_rejection(state, action)
    if rejection is not None:
        raise IllegalAction(rejection)

    if isinstance(action, (CreateSeq, CreateSet)):
        if hand_size == 0:
            raise IllegalAction("hand is empty; only GO_OUT is legal")
        side_melds = state.side_melds(side)
        if len(side_melds) >= cfg.meld.max_meld_slots:
            raise IllegalAction("meld slot cap reached")
        if not meld_result_allowed(state, side, hand_size - 3):
            raise IllegalAction("meld would strand the hand")
        if isinstance(action, CreateSet) and cfg.meld.unique_set_rank_per_side:
            if any(m.kind is MeldKind.SET and m.rank == action.rank for m in side_melds):
                raise IllegalAction(f"side already owns a set of {action.rank!r}")
        try:
            if isinstance(action, CreateSeq):
                meld = create_sequence(
                    cfg, hand, side, len(state.melds), action.suit, action.start, action.wild
                )
            else:
                # A pending pile-card obligation must consume the taken top
                # card itself, not a lower-suit copy of its rank (SPEC 06 G1).
                meld = create_set(
                    cfg, hand, side, len(state.melds), action.rank, action.wild,
                    prefer=state.pending_pile_card,
                )
        except MeldError as exc:
            raise IllegalAction(str(exc)) from exc
        state.melds.append(meld)
        _after_meld_action(state, side, action, meld_points(cfg, meld))
        _resolve_empty_hand(state, side, via_discard=False)

    elif isinstance(action, Add):
        if hand_size == 0:
            raise IllegalAction("hand is empty; only GO_OUT is legal")
        side_melds = state.side_melds(side)
        if not 0 <= action.slot < len(side_melds):
            raise IllegalAction(f"no meld in slot {action.slot}")
        if not add_result_allowed(state, side, side_melds[action.slot], action.ct):
            raise IllegalAction("add would strand the hand")
        try:
            apply_add(cfg, hand, side_melds[action.slot], action.ct)
        except MeldError as exc:
            raise IllegalAction(str(exc)) from exc
        _after_meld_action(state, side, action, cfg.card_value(action.ct))
        _resolve_empty_hand(state, side, via_discard=False)

    elif isinstance(action, Discard):
        if not discard_allowed(state, action.ct):
            raise IllegalAction(f"cannot discard card type {action.ct}")
        n = hand[action.ct]
        if n == 1:
            del hand[action.ct]
        else:
            hand[action.ct] = n - 1
        state.trash.append(action.ct)
        if cfg.discard_pile.freeze_enabled and cfg.is_wild_card(action.ct):
            state.frozen = True
        if cfg.special_threes.black_three_blocks_pile and is_black_three(action.ct):
            state.pile_blocked_for_next = True
        if state.hand_size(player) == 0:
            _resolve_empty_hand(state, side, via_discard=True)
        else:
            _end_turn(state)

    elif isinstance(action, GoOut):
        if hand_size != 0:
            raise IllegalAction("GO_OUT requires an empty hand")
        if cfg.going_out.discard_to_go_out == DISCARD_OUT_REQUIRED:
            raise IllegalAction("profile requires going out with a discard")
        if not bater_ready(state, side):
            raise IllegalAction("going-out requirements not met")
        _finish(state, EndReason.BATER, side)

    else:
        raise IllegalAction(f"expected a play-phase action, got {action!r}")


def _after_meld_action(state: RoundState, side: int, action: Action, points: int) -> None:
    """Post-meld bookkeeping: clear a satisfied pending-pile obligation and
    accumulate initial-meld staging (SPEC 06)."""
    if state.pending_pile_card is not None:
        pending_rank = id_rank(state.pending_pile_card)
        # A create of the pending rank consumed the card itself (create_set
        # runs with prefer=pending); an add satisfies only with that card.
        satisfied = (isinstance(action, CreateSet) and action.rank == pending_rank) or (
            isinstance(action, Add) and action.ct == state.pending_pile_card
        )
        if satisfied:
            state.pending_pile_card = None
            state.pending_pile_pair_only = False
    cfg = state.cfg
    if cfg.initial_meld.enabled and not state.initial_meld_done[side]:
        state.staged_points += points
        if state.staged_points >= state.initial_meld_min[side]:
            state.initial_meld_done[side] = True
            state.opened_on_turn[side] = state.turn_number


def _resolve_empty_hand(state: RoundState, side: int, via_discard: bool) -> None:
    """SPEC 01 §7 emptying resolver. Only runs in the PLAY phase."""
    player = state.current_player
    if state.hand_size(player) != 0:
        if via_discard:
            _end_turn(state)
        return

    if morto_available(state, side):
        packet = state.morto[side]
        assert packet is not None
        state.hands[player] = Counter(packet)
        state.morto[side] = None
        state.morto_taken[side] = True
        if via_discard:
            _end_turn(state)  # batida indireta: new hand waits for next turn
        # batida direta: same turn continues with the morto in hand
        return

    if via_discard:
        # Legality (`discard_allowed`) already guaranteed bater readiness.
        _finish(state, EndReason.BATER, side)
    # else: hand stays empty; the only legal follow-up is GO_OUT.


def _end_turn(state: RoundState) -> None:
    state.current_player = (state.current_player + 1) % state.cfg.table.num_players
    state.phase = Phase.DRAW
    state.turn_number += 1
    state.just_drawn_from_pile = None
    state.staged_points = 0  # staging is per-turn; a turn cannot end mid-staging


def _finish(state: RoundState, reason: EndReason, went_out_side: int | None) -> None:
    state.round_over = True
    state.phase = Phase.TERMINAL
    state.end_reason = reason
    state.went_out_side = went_out_side
