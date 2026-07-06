# SPEC 01 — game-state model

## 1. Card identity and the id space (`cards.py`)

A card is identified **only by type**, never by physical instance. Two identical cards are
game-theoretically interchangeable in hand; nothing in Buraco requires distinguishing them.
Physical order matters only in three ordered structures (deck, trash, melds), which are *not*
multisets.

```python
class Suit(IntEnum):   CLUBS=0; DIAMONDS=1; HEARTS=2; SPADES=3     # red = {DIAMONDS, HEARTS}
class Rank(IntEnum):   A=0; R2=1; R3=2; R4=3; R5=4; R6=5; R7=6; R8=7; R9=8; R10=9; J=10; Q=11; K=12

CardId = int           # 0..51 standard, 52 = JOKER
JOKER: CardId = 52
NUM_CARD_TYPES = 53    # 52 when printed_jokers == 0

def card_id(rank: Rank, suit: Suit) -> CardId:   return suit * 13 + rank
def id_rank(c: CardId) -> Rank | None:           return None if c == JOKER else Rank(c % 13)
def id_suit(c: CardId) -> Suit | None:           return None if c == JOKER else Suit(c // 13)
def is_red(c: CardId) -> bool:                   return id_suit(c) in (Suit.DIAMONDS, Suit.HEARTS)
```

- **Sequence ordering value**: `A` = 1 (low) or 14 (high); `R2`..`K` = 2..13. The natural `2`
  sits at value 2, i.e. between `A`(low, 1) and `R3`(3).
- **Jokers** are one indistinguishable type (`52`), multiplicity 0 or `printed_jokers`.
- **Red/black 3** distinction (Canasta) is derived from suit; no extra id needed.

## 2. Hand representation — multiset

**Choice: `collections.Counter[CardId]` in the mutable engine state; a fixed-length
`np.ndarray[int8]` count vector (length `NUM_CARD_TYPES`) for observations/encoding.**

Justification:
1. **Interchangeability** — the two physical copies of `A♠` confer identical legal moves and
   value; a list of instances would add meaningless permutation entropy an RL policy must
   learn to ignore.
2. **Conservation & legality as arithmetic** — "do I hold the cards this meld needs?" is a
   multiset `<=` check; card conservation is a Counter sum. O(1) per card.
3. **Canonical & deterministic** — one representation per hand, so serialization/replay
   hashing is stable regardless of draw order.
4. **Compact fixed-width encoding** — the length-53 count vector is the natural network input.
5. **Nothing needs per-instance identity in hand** — the only identity/position tracking
   Buraco needs is *inside melds* (which slot a wild occupies) and *in the trash* (order),
   both modeled separately as ordered structures.

Trade-off accepted: you cannot ask "where did this specific physical card come from"; Buraco
never asks.

## 3. Ordered structures

| Structure | Type | Order semantics |
|-----------|------|-----------------|
| `stock` (deck) | `list[CardId]` | LIFO stack; draw from `stock[-1]` (top). |
| `trash` (discard pile) | `list[CardId]` | Append discards; `trash[-1]` = top. `TAKE_PILE` moves the entire list into hand and clears it. Fully open (all visible). |
| `morto[side]` | `tuple[CardId, ...]` | Dealt packet; order irrelevant (converts to a hand = multiset), stored as tuple for deterministic dealing/serialization. |

## 4. Meld model (`melds.py`) — composition, wild identity/position, ownership, derivations

```python
class MeldKind(IntEnum):  SEQUENCE=0; SET=1
class SlotRole(IntEnum):  NATURAL=0; WILD=1

@dataclass
class Slot:
    card: CardId
    role: SlotRole                 # WILD iff the card is acting as a substitute
    represents: Rank | None = None # WILD-in-SEQUENCE only: rank it stands in for (derived from position)

@dataclass
class Meld:
    meld_id: int
    owner: int                     # side id (team in 4p, player in 2p)
    kind: MeldKind
    suit: Suit | None              # SEQUENCE: the suit;  SET: None
    rank: Rank | None              # SET: the rank;       SEQUENCE: None
    slots: list[Slot]              # SEQUENCE: ordered low→high;  SET: insertion order
```

Wild-identity rules encoded in `role`:
- A **joker** slot → always `WILD`.
- A **`2`** slot → `WILD`, **except** a `2` of the sequence's own suit sitting in its natural
  value-2 position (between `A`-low and `3`) → `NATURAL` (`natural_two_in_suit=True`). This is
  the natural-2 exception; it keeps the meld clean.
- In a `SET`, every `2`/joker is `WILD` (no natural-2 exception for sets); a set of only wilds
  is therefore impossible.

Derived (never stored; computed):

| Property | Definition |
|----------|-----------|
| `size` | `len(slots)` |
| `wild_count` | `sum(s.role == WILD)` |
| `is_canastra` | `size >= cfg.meld.canastra_min_size` (7) |
| `is_clean` (limpa) | `wild_count == 0` |
| `canastra_bonus` | `is_canastra ? (is_clean ? 200 : 100) : 0` |
| `card_points` | `sum(card_value(s.card) for s in slots)` — each card scores its **own** face value regardless of what a wild represents |
| `meld_score` | `card_points + canastra_bonus` |

Sequence invariants: contiguous run in one suit, no rank repeated, `A` not at both ends,
`represents` of a WILD = the run rank at its slot index.
Set invariants: `size >= 3`, `wild_count <= wildcard_limit_per_meld`,
`naturals >= min_naturals_per_meld`, at most one set per rank per side if
`unique_set_rank_per_side`.

## 5. Round- and match-level state (`state.py`)

```python
@dataclass
class RoundState:
    cfg: RulesConfig
    rng_state: tuple                       # random.Random.getstate() snapshot (determinism)
    hands: list[Counter[CardId]]           # index by player
    melds: list[Meld]                      # all melds, filter by .owner
    stock: list[CardId]
    trash: list[CardId]
    morto: list[tuple[CardId, ...] | None] # index by side; None once taken
    morto_taken: list[bool]                # index by side
    # --- turn machine ---
    current_player: int
    phase: Phase                           # DRAW | MELD | DISCARD | TERMINAL
    turn_number: int
    pile_blocked_for_next: bool            # Canasta black-3; always False in Buraco
    frozen: bool                           # Canasta frozen pile; always False in Buraco
    initial_meld_done: list[bool]          # index by side (Canasta min-meld gate; True in Buraco)
    # --- terminal bookkeeping ---
    round_over: bool
    went_out_side: int | None              # side that batered, else None (exhaustion)
    end_reason: EndReason | None           # BATER | STOCK_EXHAUSTED

@dataclass
class MatchState:
    cfg: RulesConfig
    seed: int
    scores: list[int]                      # cumulative, index by side
    round: RoundState
    round_index: int
    action_log: list[Action]               # replay = (cfg, seed, action_log)
    match_over: bool
    winner_side: int | None
```

`num_sides = 2` always (2 teams, or 2 players in 2p individual). Player→side map:
2p → `side(p) = p`; 4p → `side(p) = p % 2` (players 0,2 = side 0; players 1,3 = side 1).
Each side owns exactly **one** morto (`morto` has length 2 in both modes).

## 6. Action taxonomy (consumed by `actions.py` / `legal.py`)

| Action | Phase | Effect |
|--------|-------|--------|
| `DRAW_STOCK` | DRAW→MELD | Pop `stock[-1]` into hand. Illegal if stock empty. |
| `TAKE_PILE` | DRAW→MELD | Move all of `trash` into hand, clear `trash`. Illegal if `trash` empty. (Canasta: gated by `CONDITIONAL_MELD_TOP` + freeze.) |
| `MELD_NEW(slots)` | MELD | Create a meld for current side from hand cards. Validated by melds.py. |
| `LAYOFF(meld_id, slots)` | MELD | Add hand cards to an owned meld. |
| `SWAP_WILD(meld_id, natural_card, wild_to_end)` | MELD | Add the natural a WILD represents; relocate that WILD to `end ∈ {LOW, HIGH}`. |
| `DISCARD(card)` | DISCARD→(next) | Move one hand card to `trash[-1]`; end turn (unless it triggers batida indireta / bater — see §7). |
| `END_ROUND` | DRAW | Only legal when no draw is possible (stock empty and pile empty/declined); sets `STOCK_EXHAUSTED`. |

`no_op_available=False` for Buraco: there is no pass. A turn is exactly
`draw → (0+ meld/layoff/swap) → discard`.

Note: MELD phase and DISCARD phase are one continuous "act" window — `DISCARD` is legal any
time after the draw; the `Phase` enum tracks whether the player has drawn yet.

## 7. Turn state machine (the complete morto flows)

States: `DRAW → MELD → DISCARD → {next player DRAW | TERMINAL}`. The hand can only reach **0**
in MELD (via meld/layoff) or DISCARD; **never in DRAW** (drawing only adds cards) — so morto
pickup and bater can never occur on the draw.

**Emptying resolver** (invoked whenever `hands[current_player]` hits 0):

```
side = side(current_player)
if not morto_taken[side]:                      # ── FIRST emptying → MORTO PICKUP (never a bater) ──
    hand ← morto[side]  (11 cards);  morto[side] ← None;  morto_taken[side] ← True
    if emptied in MELD phase (no discard yet):     # (b) BATIDA DIRETA
        stay in MELD; player continues same turn; must still DISCARD to end
    else emptied by the DISCARD action:            # (a) BATIDA INDIRETA
        turn ENDS immediately; new 11-card hand is used next turn
else:                                          # ── SECOND emptying → BATER (going out) ──
    require: side has ≥1 canastra (require_clean_canastra? clean) AND morto_taken[side]
    round_over ← True; went_out_side ← side; end_reason ← BATER; phase ← TERMINAL
```

Legality guard that prevents stranding (enforced in `legal.py`): **an action that would reduce
the hand to 0 is legal only if the resolver above produces a forced morto pickup or a
completed bater.** Concretely, when `morto_taken[side]` is already True:
- a hand-emptying `DISCARD` is legal **only if** bater conditions hold (else the player would
  be left with 0 cards and no way to end legally);
- a hand-emptying `MELD/LAYOFF` is legal **only if** `going_out.discard_to_go_out != REQUIRED`
  (meld-out permitted). With the Buraco default `REQUIRED`, you may **not** meld away your last
  card once your morto is taken — you must retain a card to discard.

Flow summary:

| Flow | Precondition | How hand hits 0 | Result | Turn ends? |
|------|--------------|-----------------|--------|-----------|
| (a) Batida indireta | side's morto untaken | final DISCARD | take morto (→11), **not** going out | **Yes**, immediately |
| (b) Batida direta | side's morto untaken | last card MELDed | take morto (→11), **not** going out | **No** — continue in MELD, then discard |
| (c) Bater by discard | side's morto taken, ≥1 canastra | final DISCARD | round ends, +100 bater | Yes → TERMINAL |
| (c') Bater by meld-out | as (c) AND `discard_to_go_out=OPTIONAL/FORBIDDEN` | last card MELDed | round ends, +100 bater | Yes → TERMINAL |
| (d) Stock exhaustion | at DRAW, no legal draw | (no draw) | round ends, `STOCK_EXHAUSTED` | Yes → TERMINAL |

2p vs 4p: identical logic; only `side(p)` differs. In 2p each player owns one independent
morto, so the opponent's pickup status has no effect on yours. In 4p either partner emptying
their hand takes the *team* morto; the partner never takes a second (one morto per side).

Deck exhaustion (d), `END_ROUND` policy: at DRAW, legal draws =
`{DRAW_STOCK if stock} ∪ {TAKE_PILE if trash}`. If both empty → auto-`END_ROUND`. If only the
pile remains, the player may `TAKE_PILE` (continue) or `END_ROUND` (stop). Under
`CONVERT_MORTO` policy, when stock empties and any morto is untaken, that morto is dealt to
`stock` instead and play continues.

## 8. Invariants (must hold after every applied action)

1. **Card conservation**: `Σhands ⊎ Σmelds.slots ⊎ multiset(stock) ⊎ multiset(trash) ⊎
   Σ(untaken morto)` == the full dealt multiset, constant all round.
   (Cardinality = `104 + printed_jokers`.)
2. **Phase well-formed**: exactly one of `{DRAW, MELD, DISCARD, TERMINAL}`; `current_player`
   defined unless TERMINAL.
3. **No stranded hand**: at every turn boundary, every non-terminal state has
   `hand_count >= 1` for the player about to act — a hand can only pass through 0 via morto
   pickup (→11) or bater (→TERMINAL).
4. **Morto monotonicity**: `morto_taken[side]` never resets within a round;
   `morto[side] is None ⇔ morto_taken[side]`.
5. **Meld legality**: every meld satisfies its kind invariants (§4);
   `wild_count <= wildcard_limit_per_meld`; ≤1 set per rank per side if configured.
6. **Trash integrity**: `TAKE_PILE` leaves `trash == []`; `DISCARD` appends exactly one card;
   otherwise `trash` order is preserved.
7. **Ownership**: `LAYOFF`/`SWAP_WILD` target a meld with `owner == side(current_player)`.
8. **Determinism**: given `(cfg, seed, action_log)`, replay reproduces every field
   bit-for-bit (`rng_state` snapshot makes shuffles reproducible).
