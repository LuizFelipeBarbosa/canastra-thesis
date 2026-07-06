# SPEC 06 — classic Canasta mechanics (engine gaps G1–G3)

Implements decision D9 (classic US Canasta). Rules source: pagat.com/rummy/canasta.html.
No new action ids: TAKE_PILE reuses DRAW_TRASH (id 1); all conditional behavior is masking.

## New round-state fields (all inert outside Canasta-style configs)

| Field | Type | Meaning |
|---|---|---|
| `red_threes` | `list[list[CardId]]` per side | face-up red-3 trays (public) |
| `pending_pile_card` | `CardId \| None` | top card of a just-taken pile that MUST be melded before anything else |
| `pending_pile_pair_only` | `bool` | pile was frozen at take time → the forced meld must be a fresh natural set (top + 2 hand naturals) |
| `staged_points` | `int` | meld points laid this turn by a side that had not yet opened |
| `opened_on_turn` | `list[int \| None]` per side | turn each side completed its initial meld (concealed-bonus detection) |

`frozen`, `pile_blocked_for_next`, `initial_meld_done` already exist.

## G2 — red threes

With `special_threes.red_three_mode == "BONUS_AUTOREPLACE"`:
- After the deal and after every stock draw, red 3s in the drawing player's hand are moved to
  the side's tray and replaced from the stock (loop). A red 3 that cannot be replaced (drawn
  as the last stock card) ends the round immediately with STOCK_EXHAUSTED — pagat rule, and
  it prevents a hand entering PLAY one card short (soak-found deadlock).
- A red 3 acquired by taking the pile (possible only via the initial upcard) goes to the tray
  WITHOUT replacement.
- Card conservation (invariant §8.1) counts tray cards.
- Scoring: `red_three_bonus` (+100) each, `red_three_all_bonus` (+400 extra) for all four; the
  sign flips negative if the side never opened (`red_three_negative_if_no_meld`).
- Red 3s never sit in a hand at decision time, so no legality changes.

## G3 — black threes

With `special_threes.black_three_blocks_pile == True`: discarding a black 3 sets
`pile_blocked_for_next = True`; the flag clears when the next player completes their draw.
(The black 3 does NOT freeze the pile; it protects it for one turn only.)

With `black_three_meld_only_going_out == True`: `CreateSet(THREE)` (naturals only — red 3s
never reach hands) is masked unless the create is a going-out line, i.e. the anti-stranding
guard's post-action evaluation confirms the player can finish this turn (hand after create
≤ 1 with bater conditions met). ADD to a black-3 set follows the same rule. Wilds may never
join a black-3 set.

## G1 — conditional / frozen pile

With `discard_pile.draw_rule == "CONDITIONAL_MELD_TOP"`:

**Freezing.** The pile is frozen when it contains a wild (or a red-3 initial upcard):
discarding a wild sets `frozen = True`; taking the pile clears it. Independently, the pile is
*frozen against* a side that has not opened. `effective_frozen(side) = frozen or not
initial_meld_done[side]`.

**TAKE_PILE legality.** Illegal if the pile is empty, `pile_blocked_for_next`, or the top card
is a wild, a black 3, or a red 3. Let `r = rank(top)`, naturals of `r` in hand `n`:
- frozen for the side: legal iff `n ≥ 2` (fresh natural set required),
- else: legal iff `n ≥ 2`, or (`n ≥ 1` and a wild in hand), or the side has an open set of `r`
  that can take the top card.
- Unopened side additionally needs threshold reachability (below) counting the top card.

**Forced top-card meld.** Applying TAKE_PILE moves the whole pile to hand and sets
`pending_pile_card = top`, `pending_pile_pair_only = effective_frozen(side)` (evaluated before
taking). While pending, the ONLY legal actions are melds that consume that card type:
`CreateSet(r, ·)` or `Add(slot_of_open_r_set, top)`. Pair-only restricts to
`CreateSet(r, wild=0)` — except when the side already owns a set of `r`
(unique-rank melds make a fresh set illegal): the top card then joins the
existing meld, mirroring the physical rule where the pair and top merge into
it; the natural-pair requirement was verified at take time. The flag clears
when such an action applies. Legality of TAKE_PILE guarantees at least one
such action exists → mask never empties.

**Simplification (D19).** After the pile is taken, all in-hand cards (including former pile
cards) count toward the initial-meld threshold; pagat counts only the top card plus original
hand cards. Tracking card provenance inside the hand multiset is not worth the state cost for
RL; documented deviation.

## Initial meld thresholds (staging)

With `initial_meld.enabled`, `thresholds = ((floor, min_points), …)` ascending by floor; the
active requirement is the last entry whose floor ≤ the side's cumulative match score.

Micro-action staging for an unopened side:
- Meld actions accumulate `staged_points` (card point values of everything laid this turn,
  including wilds' own values).
- While `0 < staged_points < threshold`: DISCARD and GO_OUT are masked.
- `staged_points ≥ threshold` ⇒ `initial_meld_done[side] = True`,
  `opened_on_turn[side] = turn_number`, staging ends.
- **Feasibility guard (mask-nonempty invariant):** any meld action by an unopened side is
  legal only if, after it, `staged_points + max_remaining(hand) ≥ threshold`, where
  `max_remaining` is a constructive lower bound of further meldable points **that leaves at
  least 2 cards unmelded** (`reserve=2`). The reserve is load-bearing (post-implementation
  review, blocker 1): the anti-stranding guard forbids melding below 2 cards without bater,
  so a bound that assumed cards could meld to zero counted lines that are not actually legal
  and could strand the mask mid-staging. Implementation: a strict-prefix construction over
  atoms in dependency order (staged-rank adds; fresh-set cores then extensions; pair-openings
  consuming wilds; leftover wild placements up to capacity), cut at the card budget
  `|hand| − 2`. Because the bound is an explicit prefix construction whose every intermediate
  state keeps ≥2 cards, an allowed action always leaves a completable-and-closable line
  (the final discard from ≥2 cards is always legal) and the mask cannot empty. The
  `staged + action_points ≥ threshold` escape still permits single-meld openings that go out.
  The bound may reject some exotic-but-completable lines: over-restrictive, never stranding.
  Verified empirically: 60k constructed staging turns (4p and 2p) with random meld orderings,
  zero empty masks.
- A turn with `staged_points == 0` needs no guard (nothing staged, discard open).

## Going out & bonuses

- 4p classic: `go_out_min_canastras = 1`; 2p variant: 2 canastas, `draw_count = 2`,
  15-card hands.
- `discard_to_go_out = OPTIONAL` (meld-out via GO_OUT allowed).
- Concealed bonus: `went_out_side` opened on the going-out turn itself
  (`opened_on_turn[side] == turn_number` at the moment of going out) → `concealed_bonus`
  (+100) on top of `go_out_bonus`. Simplified vs pagat (no distinction for pile-taking);
  documented.

## Observation additions (all profiles; zero-filled where inert)

`red_threes (2,)` int8 counts [own, opp]; `pile_frozen (1,)`; `pile_blocked (1,)`;
`initial_meld_done (2,)`; `pending_pile_card (1,)` int16 (PAD=53 when none);
`staged_points_norm (1,)` and `initial_meld_threshold_norm (1,)` (÷120).

## Profile

`canasta(num_players=4)`: 2 decks + 4 jokers, deal 11 (4p) / 15 (2p), sets-only, 2s+jokers
wild (limit 3, min 2 naturals), no morto, CONDITIONAL_MELD_TOP + initial upcard + freeze,
thresholds ((−10⁹, 15), (0, 50), (1500, 90), (3000, 120)), red/black-3 rules on, canasta
bonuses 500/300, go-out 100 + concealed 100, Canasta card points (joker 50, 2/A 20, 8–K 10,
4–7 5, black 3 5), match 5000, hand penalty SELF_NEGATIVE.
