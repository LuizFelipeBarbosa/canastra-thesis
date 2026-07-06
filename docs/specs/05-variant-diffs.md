# SPEC 05 — variant diffs

Overrides relative to the Buraco default (SPEC 04). Blank/— = inherits Buraco.
Verified against pagat.com (Buraco, Burraco, Canasta, Biriba, Rummy — links at bottom).

**D9 (needs user confirmation before M8):** "Canastra" here = classic (US) Canasta. In Brazil
"Canastra" often just means Buraco; if that's the intent, the profile collapses into Buraco.

| Field | Buraco (default) | Canastra = classic Canasta | Rummy (basic) | Biriba (Greek) |
|-------|------------------|----------------------------|---------------|----------------|
| `table.num_players` | 2 (or 4) | 4 (2p variant differs) | 2 | 4 |
| `table.mode` | INDIVIDUAL/TEAMS | TEAMS | INDIVIDUAL | TEAMS |
| `table.cards_per_player` | 11 | 11 (4p) / **15 (2p)** | **10 (2p)** / 7 (3–4p) | 11 |
| `deck.deck_count` | 2 | 2 | **1** | 2 |
| `deck.printed_jokers` | 0 | **4** | **0** | **4** |
| `wildcard.wild_ranks` | {2} | {2} | **{} (none)** | {2} |
| `wildcard.jokers_wild` | True | True | **False** | True |
| `wildcard.wildcard_limit_per_meld` | 1 | **3** | — (no wilds) | 1 |
| `wildcard.min_naturals_per_meld` | 2 | **2 (binding)** | — | 2 |
| `wildcard.natural_two_in_suit` | True | **False** (sets-only; 2 always wild) | — | True |
| `meld.allow_sequences` | True | **False** | True | True |
| `meld.allow_sets` | True | True | True | True |
| `meld.ace_policy` | HIGH_OR_LOW | n/a (sets) | **LOW_ONLY** | HIGH_OR_LOW |
| `meld.canastra_min_size` | 7 | 7 | — (no canasta) | 7 |
| `meld.canastra_bonus_clean` | 200 | **500** | 0 | 200 (simplified; see G4) |
| `meld.canastra_bonus_dirty` | 100 | **300** | 0 | 100 |
| `morto.count` | 2 | **0** | **0** | 2 ("biribakia") |
| `morto.size` | 11 | — | — | 11 |
| `discard_pile.draw_rule` | WHOLE_PILE_UNCONDITIONAL | **CONDITIONAL_MELD_TOP** (G1) | **TOP_CARD** | WHOLE_PILE_UNCONDITIONAL |
| `discard_pile.initial_upcard` | False | **True** | **True** | **True** |
| `discard_pile.freeze_enabled` | False | **True** (G1) | False | False |
| `discard_pile.frozen_needs_two_naturals` | False | **True** | False | False |
| `discard_pile.no_immediate_redraw_discard` | False | — | **True** | **True** |
| `going_out.require_canastra` | True | True | **False** | True |
| `going_out.require_clean_canastra` | False | False | — | False |
| `going_out.require_morto_taken` | True | **False** | **False** | True (biribaki) |
| `going_out.discard_to_go_out` | REQUIRED | **OPTIONAL** | **OPTIONAL** | REQUIRED |
| `going_out.go_out_bonus` | 100 | 100 | **0** | 100 |
| `going_out.concealed_bonus` | 0 | **100** (200 total) | 0 | 0 |
| `going_out.go_out_min_canastas` | 1 | 1 (4p) / **2 (2p)** | 0 | 1 |
| `initial_meld.enabled` | False | **True** | False | False* |
| `initial_meld.thresholds` | () | **((−∞,15),(0,50),(1500,90),(3000,120))** | () | () |
| `special_threes.red_three_mode` | NONE | **BONUS_AUTOREPLACE** (G2) | NONE | NONE |
| `special_threes.red_three_bonus` | 0 | **100** | 0 | 0 |
| `special_threes.red_three_all_bonus` | 0 | **400** | 0 | 0 |
| `special_threes.red_three_negative_if_no_meld` | False | **True** | False | False |
| `special_threes.black_three_blocks_pile` | False | **True** (G3) | False | False |
| `special_threes.black_three_meld_only_going_out` | False | **True** | False | False |
| `scoring.card_points` | Buraco table | **Canasta table** ↓ | **Rummy table** ↓ | Buraco table |
| `scoring.hand_penalty_mode` | SELF_NEGATIVE | SELF_NEGATIVE | **OPPONENT_POSITIVE** | SELF_NEGATIVE |
| `scoring.match_target` | 3000 | **5000** | deal-count | **5000** |
| `turn.draw_count` | 1 | 1 (4p) / **2 (2p)** | 1 | 1 |
| `turn.deck_exhaustion_policy` | END_ROUND | END_ROUND | END_ROUND | **CONVERT_MORTO** (biribaki → stock) |

\* Biriba minimum-meld: some tables require an opening threshold; default off. Confirm if wanted.

Canasta `card_points`: Joker 50, 2 = 20, A = 20, K–8 = 10, 7–4 = 5, black 3 = 5, red 3 = bonus-only.
Rummy `card_points`: A = 1, 2–10 = face value, J/Q/K = 10.

## Engine gaps (base-engine features required for M8; not pure config)

- **G1 — Canasta conditional/frozen discard pile.** Take-pile legality requires immediately
  melding the top card with two hand cards; pile freezes when it contains a wild, was started
  by a red 3, or the side hasn't opened. Needs a frozen flag + wild-buried marker on `trash`
  and a conditional TAKE_PILE legality check. Substantial.
- **G2 — Canasta red-3 auto-replace.** Post-draw resolution hook + per-side red-3 tray.
- **G3 — Canasta black-3.** Discard side-effect (`pile_blocked_for_next`) + meldable only when
  going out.
- **G4 — Biriba tiered / bonus-suit (κόζι) scoring.** pagat Biriba scores canastras by length
  band and per-deal bonus suit (7–12 clean 200 / bonus-suit 400; 13-card 1000/2000; sets
  300/150). Flat clean/dirty pair can't express it; needs a pluggable
  `score_canastra(meld, cfg)`. Default collapses Biriba to flat 200/100; full table optional.
- **Expressible with config alone:** Canasta 2p (`draw_count=2`, `go_out_min_canastas=2`,
  15-card hand), Rummy (no morto, OPPONENT_POSITIVE, top-card draw), Biriba dead hands
  (morto machinery + CONVERT_MORTO).

## Sources
- https://www.pagat.com/rummy/buraco.html
- https://www.pagat.com/rummy/burraco.html
- https://www.pagat.com/rummy/canasta.html
- https://www.pagat.com/rummy/biriba.html
- https://www.pagat.com/rummy/rummy.html
- https://en.wikipedia.org/wiki/Buraco
