# RL / env test scenarios

Numbers reflect the reconciled id layout (`A = 1585`, SPEC 02 §2.3). Each scenario is a
pytest or hypothesis property.

1. **Round-trip ids:** `∀ a ∈ [0, 1585): encode(decode(a)) == a`.
2. **Round-trip actions:** `∀` well-formed action structs, `decode(encode(act)) == act`.
3. **Range partition:** the eight ranges are contiguous, non-overlapping, cover `[0, 1585)`, and match SPEC 02 §2.3 bases exactly (regression guard on constants).
4. **No illegal id constructible:** `encode` asserts on out-of-range operands (span ≠ 3, `slot ≥ S`, `ct = 53` targets, `w` out of range) — property test that such structs cannot yield an id.
5. **Mask soundness:** over N seeded games, every `a` with `mask[a] == 1` steps successfully and produces a rules-legal transition.
6. **Mask completeness:** enumerate every rules-applicable action from ground-truth state; each maps to an id with `mask[a] == 1`; forcing any `mask[a] == 0` action raises `IllegalAction`.
7. **Mask-nonempty invariant:** at every mid-episode decision point `mask.sum() ≥ 1`; at terminal `mask.sum() == 0` and `step` raises until `reset`.
8. **Hidden-hand invariance (leak test):** permute opponent hands with public state fixed ⇒ byte-identical obs for the acting seat (all seats, full games).
9. **Partner-hidden (4p):** scenario 8 for the partner's hand specifically.
10. **Deck-order invariance:** shuffle the undrawn stock tail ⇒ identical obs and identical legal set.
11. **Morto-content invariance:** swap morto contents (pre-pickup) ⇒ identical obs.
12. **Perfect-info isolation:** with `perfect_info=True`, the policy obs is unchanged; ground truth appears only in `info["debug"]`.
13. **Determinism:** same `(rules_config, seed, action_log)` ⇒ identical sequence of state hashes and identical obs at each step.
14. **Replay reconstruction:** `serialize.replay(cfg, seed, action_log)` reproduces the original terminal state hash.
15. **Cross-run id stability:** re-import with same config ⇒ identical bases/`A`; profile masking flips availability but never id positions.
16. **Truncation semantics:** hitting `truncation_cap` ⇒ `truncated=True, terminated=False`, reward 0.0 (or current differential if `reward_on_truncation=True`), obs shape-valid, further `step` requires `reset`.
17. **Terminal reward zero-sum (2p):** `Σ_seats r == 0` at round end.
18. **Terminal reward zero-sum (4p):** `Σ_4seats r == 0`.
19. **Teammate equality:** both seats of a team get identical terminal reward.
20. **Match accumulation:** in match mode, summed per-round differentials equal the accumulated match reward; episode ends exactly when a side crosses `match_target`.
21. **DRAW_TRASH takes entire pile:** afterwards `trash_size == 0`, hand gains exactly the old pile; multiset conservation holds.
22. **DISCARD terminates turn:** advances `to_play` to next seat and resets phase to DRAW; no other action ends a turn (except round-enders).
23. **Bater-with-discard vs illegal empty:** a hand-emptying DISCARD is legal iff (morto taken ∧ ≥1 canastra) → round ends; with morto available it triggers auto-pickup instead; otherwise masked.
24. **GO_OUT legality:** masked unless profile allows no-discard finish ∧ hand empties via melds ∧ morto taken ∧ ≥1 canastra; when legal it ends the round.
25. **Auto-morto pickup:** emptying the hand while the side's morto remains auto-assigns it and (batida direta) continues the turn — no action id consumed; contents revealed only into the taker's hand/obs.
26. **END_ROUND semantics:** stock empty + trash nonempty ⇒ mask = {DRAW_TRASH, END_ROUND} (player may decline the pile); stock and trash both empty ⇒ END_ROUND is the only legal id; stepping it terminates with STOCK_EXHAUSTED scoring. END_ROUND masked whenever stock > 0.
27. **Wild-source fidelity:** CREATE_SEQUENCE with each `w ∈ {joker, two-of-suit, off-suit-two}` consumes exactly the intended physical wild; a `w` that would be natural at the gap (two-of-suit at position 2) is masked; off-suit-two picks the lowest-suit 2 deterministically.
28. **Wild swap-and-relocate:** ADD of the natural at a wild-held position relocates the wild low-end-first and grows length by 1; masked when the meld spans the full 14.
29. **Canonical duplicate resolution:** with two copies of a needed `ct`, `copy_index = 0` is consumed first; repeated identical actions are deterministic.
30. **CREATE-cap truncation:** a side holding `S = 24` melds has both CREATE families masked; ADD remains; obs meld blocks fully occupied and consistent.
31. **Obs shape invariance:** every obs field keeps its declared shape/dtype across all steps of all games for a fixed config.
32. **`info` contract:** `action_mask` shape `(1585,)` int8 matches the engine mask; `legal_actions == np.flatnonzero(mask)`; `to_play` matches the seat whose perspective the obs encodes.
33. **Rank-2 set masking:** rank-2 CREATE_SET ids masked in Buraco (2 wild); unmasked and functional in Rummy (2 natural); id positions identical in both profiles.
