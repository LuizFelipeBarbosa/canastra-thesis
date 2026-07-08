"""Legal-action enumeration and shared legality predicates (SPEC 02 §2.5–2.6).

`legal_actions` returns structured actions in stable id order. The predicates
here are also used by `turns.apply_action`, so enumeration and application can
never disagree.

The anti-stranding guard (SPEC 01 §8.3): no legal action may leave the player
unable to finish their turn legally. Melding down to 1 card is legal only if
the forced final discard will itself be legal (morto pickup or bater); melding
to 0 is legal only if it triggers a morto pickup or a permitted meld-out.
"""

from __future__ import annotations

import copy
from collections import Counter

from buraco.cards import PAD, CardId, Rank, Suit, card_id, id_rank, id_suit
from buraco.config import (
    DISCARD_OUT_FORBIDDEN,
    DISCARD_OUT_REQUIRED,
    DRAW_CONDITIONAL_MELD_TOP,
    DRAW_TOP_CARD,
    DRAW_WHOLE_PILE,
    RulesConfig,
)
from buraco.engine.actions import (
    NUM_SEQ_SHAPES,
    NUM_SEQ_WILD,
    NUM_SET_WILD,
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
from buraco.engine.melds import (
    SET_WILD_NONE,
    Meld,
    MeldKind,
    apply_add,
    plan_add,
    plan_sequence,
    plan_set,
)
from buraco.engine.state import RED_THREE_IDS, Phase, RoundState


def morto_available(state: RoundState, side: int) -> bool:
    return (
        side < len(state.morto)
        and not state.morto_taken[side]
        and state.morto[side] is not None
    )


def bater_ready(state: RoundState, side: int, melds: list[Meld] | None = None) -> bool:
    """Whether the side meets the going-out requirements (canastra/morto).

    ``melds`` overrides the side's meld list — used to evaluate readiness on a
    hypothetical post-add state (scenario 15).
    """
    g = state.cfg.going_out
    if g.require_morto_taken and len(state.morto_taken) > side and not state.morto_taken[side]:
        return False
    if g.require_canastra:
        min_size = state.cfg.meld.canastra_min_size
        source = state.side_melds(side) if melds is None else melds
        qualifying = sum(
            1
            for m in source
            if m.is_canastra(min_size) and (m.is_clean or not g.require_clean_canastra)
        )
        if qualifying < g.go_out_min_canastras:
            return False
    return True


def meld_result_allowed(state: RoundState, side: int, resulting_hand: int) -> bool:
    """Anti-stranding guard for CREATE leaving ``resulting_hand`` cards.

    Pre-state evaluation is correct for CREATE: a fresh minimum-size meld can
    never itself be a canastra. ADDs use `add_result_allowed` instead."""
    if resulting_hand >= 2:
        return True
    if morto_available(state, side):
        return True
    if not bater_ready(state, side):
        return False
    policy = state.cfg.going_out.discard_to_go_out
    if resulting_hand == 0:
        return policy != DISCARD_OUT_REQUIRED  # meld-out ends via GO_OUT
    return policy != DISCARD_OUT_FORBIDDEN  # the forced last discard goes out


def add_result_allowed(state: RoundState, side: int, meld: Meld, ct: int) -> bool:
    """Anti-stranding guard for a specific ADD, evaluated on the POST-add
    state: the added card may itself complete the qualifying canastra
    (scenario 15), and a WILD_TO_HAND swap returns the freed wild so the net
    hand change is zero."""
    cfg = state.cfg
    player = state.current_player
    hand = state.hands[player]
    plan = plan_add(cfg, hand, meld, ct)
    if plan is None:
        return True  # not this guard's concern; application rejects it properly
    resulting = state.hand_size(player) - 1 + (1 if plan.wild_to_hand else 0)
    if resulting >= 2:
        return True
    if morto_available(state, side):
        return True
    # Simulate the add on copies to judge bater readiness afterwards.
    meld_copy = copy.deepcopy(meld)
    hand_copy = Counter(hand)
    apply_add(cfg, hand_copy, meld_copy, ct)
    melds_after = [meld_copy if m is meld else m for m in state.side_melds(side)]
    if not bater_ready(state, side, melds=melds_after):
        return False
    policy = cfg.going_out.discard_to_go_out
    if resulting == 0:
        return policy != DISCARD_OUT_REQUIRED
    return policy != DISCARD_OUT_FORBIDDEN


def discard_allowed(state: RoundState, ct: int) -> bool:
    player = state.current_player
    side = state.cfg.table.side(player)
    if state.hands[player].get(ct, 0) <= 0 or ct >= PAD:
        return False
    if (
        state.cfg.discard_pile.no_immediate_redraw_discard
        and ct == state.just_drawn_from_pile
    ):
        return False
    if state.hand_size(player) == 1:  # emptying discard
        if morto_available(state, side):
            return True  # batida indireta
        return (
            bater_ready(state, side)
            and state.cfg.going_out.discard_to_go_out != DISCARD_OUT_FORBIDDEN
        )
    return True


def pile_draw_allowed(state: RoundState) -> bool:
    if not state.trash or state.pile_blocked_for_next:
        return False
    rule = state.cfg.discard_pile.draw_rule
    if rule in (DRAW_WHOLE_PILE, DRAW_TOP_CARD):
        return True
    if rule == DRAW_CONDITIONAL_MELD_TOP:
        return can_take_conditional_pile(state)
    return False


# --- Canasta mechanics (SPEC 06) -----------------------------------------------


def is_black_three(ct: CardId) -> bool:
    return id_rank(ct) == Rank.THREE and ct not in RED_THREE_IDS and id_suit(ct) is not None


def effective_frozen(state: RoundState, side: int) -> bool:
    """The pile is frozen for a side that hasn't opened, or for everyone while
    it contains a wild / red-3 upcard (SPEC 06 G1)."""
    if state.frozen:
        return True
    return state.cfg.initial_meld.enabled and not state.initial_meld_done[side]


def max_stageable_points(
    cfg: RulesConfig,
    hand: dict[CardId, int],
    staged_set_ranks: frozenset[Rank] | set[Rank] = frozenset(),
    staged_wild_capacity: int = 0,
    reserve: int = 0,
) -> int:
    """Constructive lower bound of meld points layable from ``hand`` (sets
    only) while leaving at least ``reserve`` cards unmelded.

    The bound is a strict-prefix construction over atoms in dependency order
    (staged-rank adds, fresh-set cores then extensions, pair-openings, wild
    placements), so every counted point corresponds to a playable action whose
    intermediate states all keep ≥ ``reserve`` cards. The staging guard's
    mask-nonempty induction rests on this: with ``reserve=2`` every permitted
    line ends with ≥2 cards, so the closing discard is always legal. The bound
    may under-count (safe); it never counts an unrealizable line (review
    blocker: melding below 2 cards without bater is illegal, so a bound that
    assumed it could strand the mask).

    ``staged_set_ranks``/``staged_wild_capacity`` describe sets the side has
    already laid this turn: any natural of a staged rank is addable, and their
    remaining wild slots extend the wild-placement capacity."""
    if not cfg.meld.allow_sets:
        return 0
    budget = sum(hand.values()) - reserve
    if budget <= 0:
        return 0
    limit = cfg.wildcard.wildcard_limit_per_meld
    wild_values = sorted(
        (cfg.card_value(ct) for ct, n in hand.items() if cfg.is_wild_card(ct) for _ in range(n)),
        reverse=True,
    )
    atoms: list[tuple[int, int]] = []  # (cards, points), construction order
    capacity = staged_wild_capacity
    pairs: list[int] = []  # value of each openable 2-natural rank
    for rank in Rank:
        if rank in cfg.wildcard.wild_ranks:
            continue
        if cfg.special_threes.black_three_meld_only_going_out and rank == Rank.THREE:
            continue
        n = sum(hand.get(card_id(rank, s), 0) for s in Suit)
        value = cfg.card_value(card_id(rank, Suit.CLUBS))
        if rank in staged_set_ranks:
            atoms.extend([(1, value)] * n)  # every copy is addable
        elif n >= 3:
            atoms.append((3, 3 * value))  # fresh natural core
            atoms.extend([(1, value)] * (n - 3))
            capacity += limit
        elif n == 2 and cfg.wildcard.min_naturals_per_meld <= 2:
            pairs.append(value)
    pairs.sort(reverse=True)
    wilds_used = 0
    if limit >= 1:
        for value in pairs:
            if wilds_used >= len(wild_values):
                break
            atoms.append((3, 2 * value + wild_values[wilds_used]))
            wilds_used += 1
            capacity += limit - 1
    for wild_value in wild_values[wilds_used:]:
        if capacity <= 0:
            break
        atoms.append((1, wild_value))
        capacity -= 1

    total = used = 0
    for cards, points in atoms:
        if used + cards > budget:
            break  # strict prefix keeps dependency order valid
        used += cards
        total += points
    return total


def _remove_naturals(hand: dict[CardId, int], rank: Rank, count: int) -> dict[CardId, int]:
    remaining = dict(hand)
    for suit in Suit:
        ct = card_id(rank, suit)
        take = min(remaining.get(ct, 0), count)
        if take:
            remaining[ct] -= take
            count -= take
        if count == 0:
            break
    return remaining


def can_take_conditional_pile(state: RoundState) -> bool:
    """CONDITIONAL_MELD_TOP take-pile legality (SPEC 06 G1), including the
    anti-stranding and initial-meld feasibility guards."""
    cfg = state.cfg
    if not state.trash or state.pile_blocked_for_next:
        return False
    player = state.current_player
    side = cfg.table.side(player)
    top = state.trash[-1]
    if cfg.is_wild_card(top) or is_black_three(top) or top in RED_THREE_IDS:
        return False
    hand = state.hands[player]
    rank = id_rank(top)
    assert rank is not None
    naturals = sum(hand.get(card_id(rank, s), 0) for s in Suit)
    frozen_for = effective_frozen(state, side)
    open_set = any(
        m.kind is MeldKind.SET and m.rank == rank for m in state.side_melds(side)
    )
    has_wild = any(cfg.is_wild_card(ct) and n > 0 for ct, n in hand.items())
    if frozen_for:
        if naturals < 2:
            return False
        # forced use: join the existing rank set (1 card) or a fresh natural
        # set (top + two hand naturals)
        consumptions = [1] if open_set else [3]
    else:
        if not (naturals >= 2 or (naturals >= 1 and has_wild) or open_set):
            return False
        consumptions = []
        if open_set:
            consumptions.append(1)  # add the top card to the open set
        if naturals >= 2 or (naturals >= 1 and has_wild):
            consumptions.append(3)

    # Unopened side: taking the pile must leave the threshold reachable (D19:
    # every in-hand card counts). Evaluated after the forced pair meld.
    if cfg.initial_meld.enabled and not state.initial_meld_done[side]:
        combined = dict(hand)
        for ct in state.trash:
            combined[ct] = combined.get(ct, 0) + 1
        forced_points = 3 * cfg.card_value(top)
        remaining = _remove_naturals(combined, rank, 3)
        bound = max_stageable_points(
            cfg, remaining, {rank}, cfg.wildcard.wildcard_limit_per_meld, reserve=2
        )
        if forced_points + bound < state.initial_meld_min[side]:
            return False

    # Anti-stranding: at least one forced-meld option must leave a finishable
    # turn (>=2 cards for a normal discard, or a permitted going-out line).
    # Buried red 3s go to the tray on take, not the hand (review blocker 2).
    buried_red_threes = sum(1 for c in state.trash if c in RED_THREE_IDS)
    total_after_take = state.hand_size(player) + len(state.trash) - buried_red_threes
    policy = cfg.going_out.discard_to_go_out
    for consumed in consumptions:
        left = total_after_take - consumed
        if left >= 2:
            return True
        if left == 1 and bater_ready(state, side) and policy != DISCARD_OUT_FORBIDDEN:
            return True
        if left == 0 and bater_ready(state, side) and policy != DISCARD_OUT_REQUIRED:
            return True
    return False


def _action_points(state: RoundState, action: Action) -> int | None:
    """Card-point value a meld action would stage; None if it cannot apply."""
    cfg = state.cfg
    hand = state.hands[state.current_player]
    if isinstance(action, CreateSeq):
        plan = plan_sequence(cfg, hand, action.suit, action.start, action.wild)
    elif isinstance(action, CreateSet):
        plan = plan_set(cfg, hand, action.rank, action.wild)
    else:
        assert isinstance(action, Add)
        return cfg.card_value(action.ct)
    if plan is None:
        return None
    return sum(cfg.card_value(ct) for ct in plan.consumed)


def play_action_extra_rejection(state: RoundState, action: Action) -> str | None:
    """Canasta-mode restrictions layered over base legality: the forced
    pending-pile meld, black-three limits, and initial-meld staging. Returns a
    rejection reason or None. No-ops for Buraco-family configs."""
    cfg = state.cfg
    player = state.current_player
    side = cfg.table.side(player)

    if state.pending_pile_card is not None:
        pending = state.pending_pile_card
        rank = id_rank(pending)
        if isinstance(action, Add):
            # Only the taken top card itself discharges the obligation — a
            # same-rank card from hand would leave the top card stranded.
            ok = action.ct == pending
        elif isinstance(action, CreateSet) and action.rank == rank:
            # Frozen take: a fresh natural set (the natural pair was verified
            # at take time). Either way the created set must consume the
            # pending card itself — canonical lowest-suit-first selection
            # could otherwise meld a different copy of its rank.
            if state.pending_pile_pair_only and action.wild != SET_WILD_NONE:
                ok = False
            else:
                ok = (
                    plan_set(
                        cfg, state.hands[player], action.rank, action.wild,
                        prefer=pending,
                    )
                    is not None
                )
        else:
            ok = False
        if not ok:
            return "must meld the taken pile's top card itself"
        return None  # the forced meld is exempt from the staging guard

    if cfg.special_threes.black_three_meld_only_going_out:
        if isinstance(action, CreateSet) and action.rank == Rank.THREE:
            if action.wild != SET_WILD_NONE:
                return "black-three sets take no wilds"
            resulting = state.hand_size(player) - 3
            if resulting > 1 or not meld_result_allowed(state, side, resulting):
                return "black threes meld only when going out"
        if isinstance(action, Add):
            side_melds = state.side_melds(side)
            if 0 <= action.slot < len(side_melds):
                target = side_melds[action.slot]
                if target.kind is MeldKind.SET and target.rank == Rank.THREE:
                    if cfg.is_wild_card(action.ct):
                        return "black-three sets take no wilds"
                    if state.hand_size(player) - 1 > 1:
                        return "black threes meld only when going out"

    if cfg.initial_meld.enabled and not state.initial_meld_done[side]:
        minimum = state.initial_meld_min[side]
        if isinstance(action, (Discard, GoOut)):
            if 0 < state.staged_points < minimum:
                return "initial meld is below the threshold"
            return None
        if isinstance(action, (CreateSeq, CreateSet, Add)):
            points = _action_points(state, action)
            if points is None:
                return None  # base legality will reject it with a better error
            hand = state.hands[player]
            remaining = dict(hand)
            if isinstance(action, Add):
                remaining[action.ct] = remaining.get(action.ct, 0) - 1
            else:
                if isinstance(action, CreateSeq):
                    plan = plan_sequence(cfg, hand, action.suit, action.start, action.wild)
                else:
                    plan = plan_set(cfg, hand, action.rank, action.wild)
                assert plan is not None
                for ct in plan.consumed:
                    remaining[ct] = remaining.get(ct, 0) - 1
            # Post-action staged sets: their ranks stay addable and their free
            # wild slots extend placement capacity (soak-found deadlock fix).
            limit = cfg.wildcard.wildcard_limit_per_meld
            staged_sets = [
                m for m in state.side_melds(side) if m.kind is MeldKind.SET
            ]
            staged_ranks = {m.rank for m in staged_sets if m.rank is not None}
            capacity = sum(max(0, limit - m.wild_count) for m in staged_sets)
            if isinstance(action, CreateSet):
                staged_ranks.add(action.rank)
                capacity += limit - (1 if action.wild != SET_WILD_NONE else 0)
            elif isinstance(action, Add) and cfg.is_wild_card(action.ct):
                capacity = max(0, capacity - 1)
            if (
                state.staged_points
                + points
                + max_stageable_points(cfg, remaining, staged_ranks, capacity, reserve=2)
                < minimum
            ):
                return "cannot reach the initial-meld threshold"
    return None


def legal_actions(state: RoundState) -> list[Action]:
    """All legal structured actions for the player to act, in stable id order."""
    if state.round_over:
        return []
    cfg = state.cfg
    player = state.current_player
    side = cfg.table.side(player)
    hand = state.hands[player]

    if state.phase is Phase.DRAW:
        actions: list[Action] = []
        if state.stock:
            actions.append(DrawDeck())
        if pile_draw_allowed(state):
            actions.append(DrawTrash())
        if not state.stock:
            actions.append(EndRound())
        return actions

    assert state.phase is Phase.PLAY
    hand_size = state.hand_size(player)
    if hand_size == 0:
        # Reachable only via a permitted meld-out; the confirm is forced.
        return [GoOut()]

    actions = []
    side_melds = state.side_melds(side)
    can_create = len(side_melds) < cfg.meld.max_meld_slots

    if can_create and meld_result_allowed(state, side, hand_size - 3):
        if cfg.meld.allow_sequences:
            for suit in Suit:
                for start in range(1, NUM_SEQ_SHAPES + 1):
                    for wild in range(NUM_SEQ_WILD):
                        if plan_sequence(cfg, hand, suit, start, wild) is not None:
                            actions.append(CreateSeq(suit=suit, start=start, wild=wild))
        if cfg.meld.allow_sets:
            taken_ranks = {
                m.rank for m in side_melds if m.kind is MeldKind.SET
            } if cfg.meld.unique_set_rank_per_side else set()
            for rank in Rank:
                if rank in taken_ranks:
                    continue
                for wild in range(NUM_SET_WILD):
                    if plan_set(cfg, hand, rank, wild) is not None:
                        actions.append(CreateSet(rank=rank, wild=wild))

    sorted_hand = sorted(hand)  # enumeration never mutates the hand
    for slot, meld in enumerate(side_melds):
        for ct in sorted_hand:
            if plan_add(cfg, hand, meld, ct) is not None and add_result_allowed(
                state, side, meld, ct
            ):
                actions.append(Add(slot=slot, ct=ct))

    for ct in sorted_hand:
        if discard_allowed(state, ct):
            actions.append(Discard(ct=ct))

    return [a for a in actions if play_action_extra_rejection(state, a) is None]
