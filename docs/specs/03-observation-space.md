# SPEC 03 — observation space

Per-player, perspective-relative (own side / opposing side; self-relative seats) so the same
network serves every seat. Dict-shaped; every field has a fixed shape for a given config.
Constants (default 4p Buraco): `C=54`, `S=24` meld slots per side, `P=4` seats, `k=8`
(trash top-k), `H=8` (history length), per-slot meld width `F=31`.

## 3.1 Fields

| Field | Shape | dtype | Contents |
|---|---|---|---|
| `hand` | `(54,)` | int8 | own hand count-vector (`0..2`; joker slot `0..4`; slot 53 always 0) |
| `hand_size` | `(1,)` | int16 | `sum(hand)` |
| `own_melds` | `(24, 31)` | float32 | own side's meld slots, per-slot vector (§3.2) |
| `opp_melds` | `(24, 31)` | float32 | opposing side's meld slots |
| `trash_counts` | `(54,)` | int8 | full open pile multiset (mechanically complete — whole-pile draw makes order irrelevant to legality) |
| `trash_top_k` | `(8,)` | int16 | newest-first card-type ids, pad `53` (opponent-modeling signal) |
| `trash_size` | `(1,)` | int16 | pile size |
| `deck_size` | `(1,)` | int16 | stock count (public; identities hidden) |
| `morto_taken` | `(2,)` | int8 | `[own, opp]` has-taken flag |
| `mortos_remaining` | `(1,)` | int8 | `2/1/0` |
| `round_score` | `(2,)` | float32 | `[own, opp] / 1000` |
| `match_score` | `(2,)` | float32 | `[own, opp] / 3000` |
| `phase` | `(2,)` | int8 | one-hot `{draw, play}` |
| `has_drawn` | `(1,)` | int8 | |
| `turn_norm` | `(1,)` | float32 | `turn_count / truncation_cap` |
| `melds_this_turn` | `(1,)` | int8 | micro-actions since the draw |
| `seat_rel` | `(4,)` | int8 | one-hot self-relative seat (unused slots 0 in 2p) |
| `partner_rel` | `(4,)` | int8 | one-hot partner seat (4p; all-0 in 2p) |
| `is_4p` | `(1,)` | int8 | |
| `history` | `(8, 65)` | float32 | last `H` public actions (§3.3) |

## 3.2 Per-slot meld vector (`F = 31`, lossless given max-1-wild)

`[ occupied(1), meld_type one-hot{seq,set}(2), suit one-hot(4), rank one-hot(13),
start_norm(1), end_norm(1), length_norm(1), natural_count_norm(1), wild_count(1),
wild_source one-hot{none,joker,two}(3), wild_pos_norm(1), is_canastra(1), is_limpa(1) ]`

`suit/start/end/wild_pos` are zero for sets; `rank` zero for sequences; all zero when
`occupied = 0`. Positions/lengths normalized by 14, counts by an upper bound. A sequence is
fully recoverable from `(suit, start, end, wild_pos, wild_source)`; a set from
`(rank, natural_count, wild_source)` ⇒ no information loss vs a per-slot count-vector at
about half the width. Slots `≥ meld_count` are all-zero (matches the ADD truncation mask).

## 3.3 History item (65 wide) 

`[ actor_rel one-hot(4), family one-hot{draw_deck, draw_trash, create_seq, create_set, add,
discard, go_out}(7), card one-hot(54) ]` — card = discarded/added type; all-0 for draws.
Newest-last, zero-padded. Partly redundant with public melds+trash; kept modest (`H = 8`,
config `history_len`) as a signal for recurrence-free policies.

## 3.4 Hidden-information guarantee (tested by invariance)

The seat-`i` observation contains, and permits derivation of, **nothing** about:
(a) any other seat's hand — opponents **and partner** in 4p;
(b) stock card identities or order (only `deck_size`);
(c) morto contents (only taken flags and remaining count).
Only aggregates/sizes appear for these. The trash pile is a locked "fully open" object: its
full multiset and order are public and allowed.

**Strong test:** permuting any hidden hand, shuffling the undrawn stock tail, or swapping
morto contents — holding public state fixed — must yield a byte-identical observation.

## 3.5 Debug / perfect-info mode (flag `perfect_info`)

Adds `all_hands (P, 54)`, `deck_ordered (deck_size,)`, `morto_contents (2, 11)`,
`ground_truth_hash` — **only** inside `info["debug"]`, never in the policy observation
returned by `reset`/`step`.

## 3.6 Flattened size estimate (default 4p)

`54 + 1 + 744 + 744 + 54 + 8 + 1 + 1 + 2 + 1 + 2 + 2 + 2 + 1 + 1 + 1 + 4 + 4 + 1 + 520
= 2148` scalars (`own/opp_melds = 24·31 = 744` each; `history = 8·65 = 520`) ≈ **2.1k**
floats. `action_mask (1585,)` and `legal_actions` travel in `info`, not the obs vector.
2p configs have the same shapes (seat one-hots partially zero).

## 3.7 Reward design and env contract

- **Reward:** terminal-only. `r_seat = (own_side_round_score − opp_side_round_score) / 1000`
  at round end; `0.0` otherwise. Zero-sum by construction in 2p and 4p; teammates identical.
  Match mode (`episode = "MATCH"`): accumulate per-round differentials; episode terminates
  when a side reaches `match_target`; `reward_per_round=True` optionally emits per-round
  rewards. Config `reward_scale = 1/1000`. Potential-based shaping hooks exist but are
  **off by default**.
- **`info` contract** (every `reset`/`step`):
  `info = { "action_mask": np.ndarray((1585,), int8), "legal_actions": sorted list[int],
  "to_play": int seat, "team": int side, "phase": str }` plus `"debug": {…}` only when
  `perfect_info`. `step` returns `(obs, reward, terminated, truncated, info)`; `obs` is the
  observation of the seat now to act (RLCard-style current-player perspective).
- **Truncation:** hitting `truncation_cap` ⇒ `truncated=True, terminated=False`, reward 0.0
  (or current differential if `reward_on_truncation=True`).
