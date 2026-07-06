# SPEC 02 — action space

> Reconciliation notes (orchestrator, applied over the designer's draft):
> 1. `END_ROUND` added as an explicit id (SPEC 01 §6/D6: when the stock is empty a player may
>    decline the pile; without this id the mask would force TAKE_PILE, changing the game).
> 2. `CREATE_SET` covers all 13 ranks (39 ids, not 36/12 ranks): rank-2 sets are unrepresentable
>    otherwise, breaking the Rummy profile where 2 is not wild. Rank-2 set ids are masked
>    whenever 2 ∈ `wild_ranks`.
> 3. Wild swap-and-relocate on ADD is deterministic (low end first) rather than a player
>    choice (SPEC 01 §6 `SWAP_WILD(end)` is dropped); config flag can switch to freed-wild-to-hand.
> These change the id bases relative to the designer's draft; the tables below are final.

## 2.1 Card-type space (54 slots)

| Index range | Meaning | Encoding |
|---|---|---|
| `0..51` | rank×suit | `ct = suit*13 + rank_ord` |
| `52` | JOKER (wild) | all printed jokers collapse here (strategically interchangeable) |
| `53` | NONE / PAD sentinel | never held, melded, discarded, or counted |

- **Suit order (suit-major):** `C=0, D=1, H=2, S=3` — each suit's 13 ranks contiguous.
- **Rank ordinal:** `A=0, 2=1, 3=2, …, 10=9, J=10, Q=11, K=12`.
- **Copies:** 2 decks ⇒ up to 2 physical copies per `ct` in `0..51`; up to 4 jokers at 52.
  Count-vectors use `int8`. `NUM_CARD_TYPES = 53` for counts; slot 53 is pad-only in ordered
  encodings (`trash_top_k`, history).
- **Instance id (canonical resolution):** `(ct, copy_index)`; deterministic tie-break: always
  consume the lowest available `copy_index`.
- **No printed jokers (Buraco default):** `hand[52]` stays 0 and all joker-consuming ids are
  masked — no id-layout change.

## 2.2 Sequence position model

14 positions per suit, `p ∈ 1..14`:

| p | 1 | 2 | 3 | … | 10 | 11 | 12 | 13 | 14 |
|---|---|---|---|---|---|----|----|----|----|
| card | A(low) | 2 | 3 | … | 10 | J | Q | K | A(high) |
| rank_ord | 0 | 1 | 2 | … | 9 | 10 | 11 | 12 | 0 |

`rank_ord(p) = 0 if p ∈ {1,14} else p−1`. Natural card-type at position `p` in suit `s`:
`nat(p,s) = s*13 + rank_ord(p)`.

**No-wrap is automatic**: a run is a contiguous `[start..end]` within `1..14`; K→A→2 wrap is
unrepresentable. A meld may contain both aces of its suit only as a full `1..14` run.

**CREATE is minimum-size only** (length 3; `start ∈ 1..12`): everything longer is built with
ADD in the same meld/discard phase. This is the key space-saver (length-capped CREATE at 7
would cost ~800 sequence ids alone). Cost: a 7-card canastra from hand = 1 CREATE + 4 ADDs.

## 2.3 Fixed id layout (`A = 1585`)

| Family | Base | Size | Range | Operand arithmetic |
|---|---|---|---|---|
| DRAW_DECK | 0 | 1 | `{0}` | — |
| DRAW_TRASH | 1 | 1 | `{1}` | — |
| CREATE_SEQUENCE | `BASE_SEQ = 2` | `4·12·4 = 192` | `[2, 194)` | `((suit*12)+shape)*4 + w` |
| CREATE_SET | `BASE_SET = 194` | `13·3 = 39` | `[194, 233)` | `rank_ord*3 + w` |
| ADD_TO_MELD | `BASE_ADD = 233` | `S·54 = 1296` | `[233, 1529)` | `slot*54 + ct` |
| DISCARD | `BASE_DISCARD = 1529` | `54` | `[1529, 1583)` | `ct` (53 permanently masked) |
| GO_OUT | `1583` | 1 | `{1583}` | — |
| END_ROUND | `1584` | 1 | `{1584}` | — |

- `shape = start_pos − 1 ∈ 0..11`, `end_pos = start_pos + 2`.
- Sequence wild `w`: `0=none, 1=joker, 2=two-of-suit, 3=off-suit-two`.
- Set wild `w`: `0=none, 1=joker, 2=two` (a set has no suit; the specific 2 is chosen by
  canonical resolution).
- `slot ∈ 0..S−1` indexes the **acting player's own side's** melds in creation order
  (opponent melds are never addable). `S = cfg.max_meld_slots = 24` per side
  (deck hard cap `floor(108/3) = 36`; realistic worst case ≈ 20; 24 is the budgeted middle).
  When a side occupies all `S` slots both CREATE families are masked for it (ADD remains).
- Only `S` shifts later bases; joker config shifts nothing. Freeze `S` per training run ⇒
  ids stable.

## 2.4 Encode / decode (pure functions in `engine/actions.py`)

```python
def decode(a: int) -> Action:
    if a == 0: return DrawDeck()
    if a == 1: return DrawTrash()
    if a < 194:                        # CREATE_SEQUENCE
        x = a - 2; w = x % 4; x //= 4
        shape = x % 12; suit = x // 12
        return CreateSeq(suit, start=shape + 1, end=shape + 3, wild=w)
    if a < 233:                        # CREATE_SET
        x = a - 194; w = x % 3; rank = Rank(x // 3)
        return CreateSet(rank, wild=w)
    if a < 1529:                       # ADD_TO_MELD
        x = a - 233; ct = x % 54; slot = x // 54
        return Add(slot, ct)
    if a < 1583:                       # DISCARD
        return Discard(ct=a - 1529)
    if a == 1583: return GoOut()
    return EndRound()

def encode(act: Action) -> int:        # inverse; asserts operands in range
    match act:
        case CreateSeq(s, st, en, w): assert en == st + 2 and 1 <= st <= 12 and s < 4 and w < 4
                                      return 2 + ((s * 12) + (st - 1)) * 4 + w
        case CreateSet(r, w):         assert w < 3; return 194 + int(r) * 3 + w
        case Add(slot, ct):           assert slot < S and ct < 54; return 233 + slot * 54 + ct
        case Discard(ct):             assert ct < 54; return 1529 + ct
        ...
```

Round-trip properties (tested): `encode(decode(a)) == a` for all `a ∈ [0, A)` and
`decode(encode(act)) == act` for all well-formed structs. Out-of-range operands assert in
`encode` ⇒ no illegal move is constructible as an id; residual illegality is handled by the
mask, never by missing ids.

## 2.5 Turn state machine

```
TURN(player):
  ── DRAW phase ──  legal ⊆ { DRAW_DECK if stock>0,
                              DRAW_TRASH if trash>0,
                              END_ROUND if stock==0 }
      exactly one of DRAW_DECK/DRAW_TRASH, or END_ROUND (terminates the round,
      STOCK_EXHAUSTED). When stock==0 and trash==0, END_ROUND is the only legal
      action — the mask is never empty mid-episode.
      DRAW_TRASH moves the ENTIRE pile → hand.
  ── PLAY phase (meld/discard) ──  loop, legal ⊆ { CREATE_SEQUENCE*, CREATE_SET*,
                                    ADD_TO_MELD*, DISCARD*, GO_OUT? }
      • CREATE_*/ADD_* do not end the turn.
      • Engine-automatic: hand hits 0 with side's morto untaken → auto-pickup
        (11 new cards), continue same turn (batida direta). Not an action id.
      • DISCARD(ct): removes one card, ends the turn → next player.
          - empties hand + morto untaken → auto-pickup, turn still ends
            (batida indireta).
          - empties hand + morto taken + ≥1 canastra → bater, round ends.
          - empties hand otherwise → ILLEGAL (masked).
      • GO_OUT: only when profile allows finishing without a discard AND hand
        would empty via melds AND morto taken AND ≥1 canastra → round ends.
```

**Mask-nonempty invariant:** at any mid-episode decision point ≥1 id is legal (DRAW always
offers a draw or END_ROUND; PLAY always permits discarding some held card).
All-zero mask ⇔ terminal ⇔ requires `reset()`.

## 2.6 Legality per family (`engine/legal.py`)

**CREATE_SEQUENCE(suit s, start st, end en=st+2, wild w):**
let `held = #{p ∈ [st..en] : hand has nat(p,s)}`.
- `w=0`: legal iff `held == 3`.
- `w ∈ {1,2,3}`: legal iff `held == 2` (single gap `g`) AND the wild is held AND the wild is
  not natural at `g`:
  - `w=2` (two-of-suit, `ct = s*13+1`): requires `g ≠ 2` (at position 2 that card is natural
    ⇒ only `w=0` applies).
  - `w=3` (off-suit two): requires a 2 of some suit `≠ s`; always wild.
  - `w=1` (joker): requires a joker held.
- `held < 2` ⇒ illegal (would need > 1 wild).
- Known, accepted canonicalization: with `held == 3`, wild-create (`w>0`) is masked — you
  cannot burn a wild while holding all three naturals. Duplicate copies (2 decks) keep the
  residual strategic loss negligible.

**CREATE_SET(rank R, wild w):** `R ∈ wild_ranks` ⇒ all ids for R masked. `nat = hand count of
rank R` (naturals only). `w=0`: `nat ≥ 3`. `w=1`: `nat ≥ 2` and joker held. `w=2`: `nat ≥ 2`
and some wild-2 held. Also masked if `unique_set_rank_per_side` and the side already has a
set of R (use ADD instead).

**ADD_TO_MELD(slot j, ct c):** `j < own_side.meld_count`; player holds `c`; and one of:
- **sequence** target: `c` is the natural at `nat(st−1, s)` or `nat(en+1, s)` (end extension);
  **or** `c` is the natural at the current wild's position (wild swap-and-relocate, §2.7);
  **or** `c` is a wild, the meld has no wild yet, and it lands on an open end position.
- **set** target: `c` is a rank-R natural; **or** `c` is a wild and
  `wild_count < wildcard_limit_per_meld`.

**DISCARD(ct c):** PLAY phase; player holds `c`; `c ≤ 52`; hand-emptying rules per §2.5.

**GO_OUT / DRAW_* / END_ROUND:** per §2.5.

## 2.7 Canonical physical-card resolution (deterministic)

- **Naturals first:** natural positions are filled by natural cards; the single wild fills
  the one remaining gap.
- **Duplicate card types:** consume `copy_index = 0` first.
- **Off-suit-two selection:** lowest suit index `≠ s` holding a 2 (`copy_index = 0` first).
- **Wild swap-and-relocate** (ADD of the natural at a wild-held sequence position): the
  natural takes the wild's slot; the freed wild relocates to **extend the low end (`st−1`)
  first, else the high end (`en+1`)**; if neither position exists (meld spans full `1..14`)
  the add is **illegal (masked)**. Result: length +1, wild stays in the meld, hand loses only
  the natural. Config `wild_relocation ∈ {RELOCATE_EXTEND (default), TO_HAND}`.

## 2.8 Profile masking

- **No printed jokers:** mask all `w=joker` creates, `ct=52` ADD/DISCARD ids.
- **2 not wild (Rummy):** unmask rank-2 CREATE_SET ids; mask `w∈{2,3}` sequence creates and
  `w=2` set creates; the natural 2 participates in sequences at position 2 via `w=0`/ADD.
- **Variant toggles** (sets disallowed, top-card draw, no morto, …) flip masks only; id
  positions never change across profiles.
