# Design decisions register

Decisions made where standard Buraco has table variation or where the user's locked rules
diverge from pagat's "standard." Each is a config field with the chosen default.

Status legend: **LOCKED** = user-confirmed; **DEFAULT** = standard pick, safe to override in
config; **CONFIRM** = needs explicit user sign-off before the affected milestone ships.

| # | Decision | Default chosen | Rationale / conflict | Config field | Status |
|---|----------|----------------|----------------------|--------------|--------|
| D1 | Value of the wild `2` | **10 points** | Brazilian Buraco standard (pagat: two = 10, joker = 20, ace = 15). | `scoring.card_points["2"]` | LOCKED (confirmed 2026-07-05) |
| D2 | Going-out canastra requirement | **any canastra (clean or dirty)** | User's locked rule. pagat "Buraco Aberto" requires a *limpa*; both exposed. | `going_out.require_clean_canastra=False` | LOCKED |
| D3 | Meld types allowed | **sequences AND sets** | User's locked rule. pagat Brazilian Buraco is sequences-only; this is a house override. | `meld.allow_sets=True` | LOCKED |
| D4 | Printed jokers | **0** (2s only wild) | User's house rule. `printed_jokers=4` restores the pagat 108-card deck. | `deck.printed_jokers=0` | LOCKED |
| D5 | Deck-exhaustion behavior | **END_ROUND** | User's locked rule. pagat standard converts an untaken morto into new stock; exposed as `CONVERT_MORTO`. | `turn.deck_exhaustion_policy` | LOCKED |
| D6 | Stock-empty draw semantics | Player may still TAKE_PILE; round ends only when no legal draw exists or player declines via END_ROUND. | Keeps determinism and preserves the real decision for the RL agent. | derived from D5 | DEFAULT |
| D7 | Hand-at-round-end scoring sign | **SELF_NEGATIVE** | Buraco/Canasta/Biriba convention. Rummy uses OPPONENT_POSITIVE. | `scoring.hand_penalty_mode` | DEFAULT |
| D8 | Morto-untaken penalty | **−100** per side that never took its morto | pagat-confirmed standard. | `morto.untaken_penalty=100` | DEFAULT |
| D9 | "Canastra" variant identity | **Classic (US) Canasta**: sets-only, frozen pile, red-3 bonus, min initial meld, 500/300 canastas, 5000 match | Name is BR/US-ambiguous (in Brazil "Canastra" often = Buraco). | new profile | LOCKED: classic US Canasta (confirmed 2026-07-05) |
| D10 | Match target | Buraco **3000**, Canasta **5000**, Biriba **5000**, Rummy = deal-count | Per-profile; Buraco lock = 3000. | `scoring.match_target` | DEFAULT |
| D11 | Episode granularity | **one round = one RL episode** | User's lock; match-length episodes available. | `scoring.episode="ROUND"` | LOCKED |
| D12 | Wild relocation on ADD | **deterministic, low-end-first** (freed wild extends low end, else high; masked if meld spans full 14) | Removes a marginal player choice; keeps ADD space at S×54 instead of doubling. Config alt: freed wild to hand. | `wildcard.wild_relocation` | DEFAULT |
| D13 | END_ROUND is an explicit action id | Legal only when stock is empty; sole legal action when stock and trash are both empty | Preserves the real "decline the pile" decision (D6) and the mask-nonempty invariant; no silent auto-termination mid-episode. | id 1584 | DEFAULT |
| D14 | CREATE_SET covers all 13 ranks | rank-2 set ids exist but are **masked while 2 is wild** | Rummy (2 not wild) needs 2-2-2 sets; id positions must not differ across profiles. Total action space `A = 1585`. | SPEC 02 §2.3 | DEFAULT |
| D15 | Reward | **zero-sum team score differential** at round end, × 1/1000 | Clean for self-play; teammates identical. Non-zero-sum absolute-score reward available via config if match-play magnitude matters to the thesis. | `scoring.reward_scale` | LOCKED (confirmed 2026-07-05) |
| D16 | CREATE melds at minimum size only | length-3 sequences / 3-card sets; longer melds built via ADD micro-actions | Biggest action-space saver (length-capped CREATE would triple sequence ids). Cost: a from-hand canastra = 1 CREATE + 4 ADDs. | SPEC 02 §2.2 | DEFAULT |
| D17 | Observation amendments over SPEC 03 | Added `all_hand_sizes` (every seat's card count — public at a real table, critical bater signal); the live `round_score` field uses public zones only (melds + bonuses + morto penalty), never hidden hand values | SPEC 03 §3.1 omitted public hand counts and its round_score would have leaked opponent hand values mid-round. | observations.py | DEFAULT |
| D18 | CONVERT_MORTO + untaken penalty | A morto converted into stock leaves `morto_taken=False`, so the −100 penalty currently applies to that side at round end | Review flagged: "converted" vs "abandoned" are conflated. Variant-specific (Biriba); Buraco default (END_ROUND) unaffected. Decide when finalizing Biriba/Canasta scoring. | turns.py `_maybe_convert_morto` | LOCKED: keep the −100 (confirmed 2026-07-05) |
| D19 | Batida seca (meld-out bater) in Buraco | **Allowed** — buraco profile sets `discard_to_go_out=OPTIONAL`; SPEC 04 schema default stays REQUIRED (Biriba keeps it) | User hit the block in play (hand was exactly a 9-2-9 trio and could not win) and chose to allow going out by melding the whole hand. Scenario-5 tests now construct REQUIRED explicitly. | profiles/buraco.py | LOCKED (confirmed 2026-07-05) |

## Engine gaps (not expressible as pure config; base-engine features required for M8 variants)

- **G1 — Canasta conditional/frozen discard pile.** ✅ Implemented in M10 (SPEC 06): frozen flag,
  conditional TAKE_PILE with forced top-card meld via `pending_pile_card`, feasibility-guarded
  initial-meld staging.
- **G2 — Canasta red-3 auto-replace.** ✅ Implemented in M10: per-side trays, deal/draw resolution,
  round ends when a red 3 is drawn as the last stock card.
- **G3 — Canasta black-3.** ✅ Implemented in M10: `pile_blocked_for_next` + meld-only-when-going-out.
- **D19 (new, M10)** — after taking the pile, all in-hand cards count toward the initial-meld
  threshold (pagat counts only top card + original hand). Documented simplification.
- **G4 — Biriba tiered / bonus-suit (κόζι) canastra scoring.** Flat clean/dirty bonus pair can't express
  it; needs a pluggable `score_canastra(meld, cfg)`. Default collapses Biriba to flat 200/100; optional.
- **Expressible with config alone:** Canasta 2-player (draw_count=2, 2 canastas to go out, 15-card hand),
  Rummy (no morto, OPPONENT_POSITIVE, top-card draw), Biriba dead hands (morto machinery + CONVERT_MORTO).
