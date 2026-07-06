# SPEC 04 — RulesConfig schema

All dataclasses `@dataclass(frozen=True)`; the tree is the single source of rule truth.
"Module" = primary consumer. Defaults shown are the Buraco profile.

## `TableConfig`
| Field | Type | Buraco default | Module | Notes |
|-------|------|----------------|--------|-------|
| `num_players` | `int` | `2` | state, env | 2 or 4 |
| `mode` | `str` | `"INDIVIDUAL"` | state | `INDIVIDUAL` \| `TEAMS` |
| `team_of` | `int` | `1` | state | 1 in 2p, 2 in 4p; `num_sides = num_players // team_of` |
| `cards_per_player` | `int` | `11` | state | initial deal |

## `DeckConfig`
| Field | Type | Default | Module | Notes |
|-------|------|---------|--------|-------|
| `deck_count` | `int` | `2` | cards, state | copies of the 52-card pack |
| `printed_jokers` | `int` | `0` | cards | adds N joker cards (`4` = pagat deck) |
| `num_ranks` | `int` | `13` | cards | fixed |
| `num_suits` | `int` | `4` | cards | fixed |

Derived deck size = `deck_count*52 + printed_jokers` = 104 (Buraco default).

## `WildcardConfig`
| Field | Type | Default | Module | Notes |
|-------|------|---------|--------|-------|
| `wild_ranks` | `frozenset[Rank]` | `{R2}` | melds, legal | ranks that act wild |
| `jokers_wild` | `bool` | `True` | melds | joker always wild if present |
| `natural_two_in_suit` | `bool` | `True` | melds | 2 of own suit in value-2 slot = NATURAL |
| `wildcard_limit_per_meld` | `int` | `1` | melds, legal | Canasta = 3 |
| `min_naturals_per_meld` | `int` | `2` | melds | not binding in Buraco (limit 1); binding in Canasta |
| `wild_relocation` | `str` | `"RELOCATE_EXTEND"` | melds | `RELOCATE_EXTEND` \| `TO_HAND` (SPEC 02 §2.7) |

## `MeldConfig`
| Field | Type | Default | Module | Notes |
|-------|------|---------|--------|-------|
| `allow_sequences` | `bool` | `True` | melds, legal | |
| `allow_sets` | `bool` | `True` | melds, legal | D3 (house rule) |
| `min_meld_size` | `int` | `3` | melds | |
| `ace_policy` | `str` | `"HIGH_OR_LOW"` | melds | `HIGH_OR_LOW` \| `LOW_ONLY` \| `HIGH_ONLY` |
| `allow_wrap` | `bool` | `False` | melds | Q-K-A-2 wrap disallowed |
| `unique_set_rank_per_side` | `bool` | `True` | legal | ≤1 set per rank per side |
| `canastra_min_size` | `int` | `7` | melds, scoring | |
| `canastra_bonus_clean` | `int` | `200` | scoring | limpa |
| `canastra_bonus_dirty` | `int` | `100` | scoring | suja |
| `max_meld_slots` | `int` | `24` | actions, legal | per side; frozen per training run (SPEC 02 §2.3) |

## `MortoConfig`
| Field | Type | Default | Module | Notes |
|-------|------|---------|--------|-------|
| `count` | `int` | `2` | state | one per side; `0` disables (Rummy, Canasta) |
| `size` | `int` | `11` | state | |
| `per` | `str` | `"SIDE"` | state | `SIDE` (team in 4p, player in 2p) |
| `pickup` | `str` | `"ON_EMPTY_FORCED"` | turns | forced when hand first empties |
| `untaken_penalty` | `int` | `100` | scoring | subtracted from a side that never took its morto (D8) |

## `DiscardPileConfig`
| Field | Type | Default | Module | Notes |
|-------|------|---------|--------|-------|
| `visibility` | `str` | `"FULL_OPEN"` | env, obs | `FULL_OPEN` \| `TOP_ONLY` |
| `draw_rule` | `str` | `"WHOLE_PILE_UNCONDITIONAL"` | legal, actions | `WHOLE_PILE_UNCONDITIONAL` \| `TOP_CARD` \| `CONDITIONAL_MELD_TOP` |
| `initial_upcard` | `bool` | `False` | state | flip one card to start the pile (Canasta/Rummy/Biriba = True) |
| `freeze_enabled` | `bool` | `False` | legal | Canasta frozen pile (G1) |
| `frozen_needs_two_naturals` | `bool` | `False` | legal | Canasta take-when-frozen rule |
| `no_immediate_redraw_discard` | `bool` | `False` | legal | can't discard the card just drawn *from the pile* (Biriba/Rummy top-card) |

## `GoingOutConfig`
| Field | Type | Default | Module | Notes |
|-------|------|---------|--------|-------|
| `require_canastra` | `bool` | `True` | turns, legal | |
| `require_clean_canastra` | `bool` | `False` | turns, legal | D2 |
| `require_morto_taken` | `bool` | `True` | turns, legal | |
| `discard_to_go_out` | `str` | `"REQUIRED"` | legal | `REQUIRED` \| `OPTIONAL` \| `FORBIDDEN` |
| `go_out_bonus` | `int` | `100` | scoring | bater |
| `concealed_bonus` | `int` | `0` | scoring | Canasta extra +100 |
| `go_out_min_canastas` | `int` | `1` | turns | Canasta-2p = 2 |

## `InitialMeldConfig`
| Field | Type | Default | Module | Notes |
|-------|------|---------|--------|-------|
| `enabled` | `bool` | `False` | legal | Buraco has none |
| `thresholds` | `tuple[tuple[int,int], ...]` | `()` | legal | `(cumulative_score_floor, min_points)` ascending; a side's first opening meld batch must total ≥ min_points |

## `SpecialThreesConfig` (all inert in Buraco)
| Field | Type | Default | Module | Notes |
|-------|------|---------|--------|-------|
| `red_three_mode` | `str` | `"NONE"` | turns | `NONE` \| `BONUS_AUTOREPLACE` (G2) |
| `red_three_bonus` | `int` | `0` | scoring | Canasta 100 each |
| `red_three_all_bonus` | `int` | `0` | scoring | Canasta +400 for all four (800 total) |
| `red_three_negative_if_no_meld` | `bool` | `False` | scoring | |
| `black_three_blocks_pile` | `bool` | `False` | legal | (G3) |
| `black_three_meld_only_going_out` | `bool` | `False` | legal | |

## `ScoringConfig`
| Field | Type | Buraco default | Module | Notes |
|-------|------|----------------|--------|-------|
| `card_points` | `Mapping[str, int]` | table below | scoring | key = rank name or `"JOKER"` |
| `hand_penalty_mode` | `str` | `"SELF_NEGATIVE"` | scoring | vs `OPPONENT_POSITIVE` (Rummy) |
| `match_target` | `int` | `3000` | env | |
| `episode` | `str` | `"ROUND"` | env | `ROUND` \| `MATCH` (D11) |
| `reward_scale` | `float` | `0.001` | env | SPEC 03 §3.7 |

Buraco `card_points` (D1: wild 2 = 10, Brazilian standard):

| A | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | J | Q | K | JOKER |
|---|---|---|---|---|---|---|---|---|----|---|---|---|-------|
| 15 | 10 | 5 | 5 | 5 | 5 | 5 | 10 | 10 | 10 | 10 | 10 | 10 | 20 |

## `TurnConfig`
| Field | Type | Default | Module | Notes |
|-------|------|---------|--------|-------|
| `draw_count` | `int` | `1` | turns | Canasta-2p = 2 |
| `no_op_available` | `bool` | `False` | legal | no pass in Buraco |
| `deck_exhaustion_policy` | `str` | `"END_ROUND"` | turns | `END_ROUND` \| `CONVERT_MORTO` (D5) |
| `truncation_cap` | `int` | `400` | env | RL truncation turn cap (env-level, not a rule) |

## `RulesConfig` (root; consumed by `profiles/`, `serialize.py`, `env/`)
| Field | Type |
|-------|------|
| `name` | `str` |
| `table` | `TableConfig` |
| `deck` | `DeckConfig` |
| `wildcard` | `WildcardConfig` |
| `meld` | `MeldConfig` |
| `morto` | `MortoConfig` |
| `discard_pile` | `DiscardPileConfig` |
| `going_out` | `GoingOutConfig` |
| `initial_meld` | `InitialMeldConfig` |
| `special_threes` | `SpecialThreesConfig` |
| `scoring` | `ScoringConfig` |
| `turn` | `TurnConfig` |

`profiles/buraco.py` returns this fully defaulted tree; other profiles are override deltas
(SPEC 05). JSON round-trip via `serialize.py` is a thin layer over the dataclass tree.
