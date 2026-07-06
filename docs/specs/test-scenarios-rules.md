# Rule edge-case test scenarios

Default Buraco profile unless a variant is named. Scenarios 18/19/22 reflect the reconciled
deterministic wild-relocation design (SPEC 02 §2.7): relocation is low-end-first, not a
player choice.

### Morto flows
1. **Batida direta.** Side's morto untaken; player has 3 cards, all meld/extend, plays all → hand 0 in PLAY. **Expect:** take morto (hand→11), `morto_taken[side]=True`, **turn continues**, no bater, must still discard to end.
2. **Batida indireta.** Side's morto untaken; player melds down to 1 card, discards it → hand 0 via DISCARD. **Expect:** take morto (hand→11), **turn ends immediately**; not a bater.
3. **Bater by discard.** Morto already taken, side has a canastra; player melds to 1 card and discards last. **Expect:** round ends, `end_reason=BATER`, +100 to side.
4. **Bater same turn morto was taken.** Batida direta (scenario 1) gives 11 new cards; player then melds all 11 (side has a canastra) and discards last. **Expect:** legal — morto taken earlier this turn satisfies `require_morto_taken`; round ends, +100.
5. **Meld-out attempt with morto taken, REQUIRED policy.** Morto taken, canastra present; player tries to MELD/ADD away their last card (no discard left). **Expect:** action **illegal** (`discard_to_go_out=REQUIRED` — schema default; the buraco profile is OPTIONAL since D19, so tests set REQUIRED explicitly); must keep a card to discard.
6. **Meld-out bater, OPTIONAL policy (buraco profile default since D19).** Same as 5 but `discard_to_go_out=OPTIONAL`. **Expect:** GO_OUT legal; round ends, +100.
7. **Deck exhaustion, one morto untaken, END_ROUND.** Stock empty, pile empty at a player's DRAW; side X never took its morto. **Expect:** END_ROUND is the only legal action; side X gets −100 (untaken penalty); every side subtracts remaining hand value.
8. **Deck exhaustion, CONVERT_MORTO profile.** Stock empties at DRAW, side X's morto untaken. **Expect:** morto X becomes the new stock, `morto_taken[X]` stays False, play continues; round does not end here.
9. **Cannot empty by drawing.** Player holds 1 card, DRAW phase, takes a 20-card pile → hand 21. **Expect:** no morto pickup, no bater; drawing never triggers the emptying resolver.
10. **2p independent mortos.** 2p; opponent already took their morto. Current player empties hand for the first time. **Expect:** current player takes *their own* morto normally.
11. **4p partner already took team morto.** 4p; side 0's morto taken by player 0 earlier. Player 2 (same side) now empties hand. **Expect:** no morto to take (one per side); this emptying is evaluated as a **bater** attempt, requires canastra.
12. **First emptying needs no canastra.** Morto untaken, side has zero canastras; player discards last card. **Expect:** legal — morto pickup does not require a canastra (only bater does).
13. **Bater rejected — no canastra.** Morto taken, side has no canastra; hand-emptying discard. **Expect:** **illegal/masked** (would strand the player); must retain ≥1 card.
14. **Bater rejected — clean-canastra profile with only a dirty canastra.** `require_clean_canastra=True`; side has one suja only. **Expect:** bater illegal.
15. **Going out with the canastra just completed.** A meld reaches 7 cards this turn, morto taken; player melds to 1 and discards last. **Expect:** legal bater.

### Wild edge cases
16. **Natural-2 vs wild-2 in one sequence.** Hearts run `A♥-2♥-3♥-4♥-2♦(wild@5)-6♥`. **Expect:** `2♥` role NATURAL (own suit, value-2 slot), `2♦` role WILD representing 5♥. `wild_count=1` ≤ limit. Legal; if length ≥7, **suja**.
17. **Wild-limit rejection.** `4♠-2♥(wild)-Joker-7♠` (two wilds). **Expect:** rejected — `wildcard_limit_per_meld=1` exceeded.
18. **Wild swap relocates low-first.** Meld `4♦-5♦-Joker(@6)-7♦`; ADD `6♦`. **Expect:** `6♦` fills slot 6 as NATURAL; joker relocates to the low end representing `3♦`; result spans 3–7, still one wild. Turn continues.
19. **Wild swap relocates high when low end blocked.** Meld `A♦-Joker(@2)-3♦` (start position 1); ADD `2♦` (the natural at the wild's position — natural-2 of own suit). **Expect:** `2♦` fills as NATURAL, low end (position 0) doesn't exist → joker relocates high, representing `4♦`; result `A-2-3-J(4)`, meld stays one-wild.
20. **Wild swap with both ends blocked.** Full `A..K..A` 14-span with an internal wild; ADD of the natural at the wild's position. **Expect:** **illegal (masked)** — no legal end for the freed wild, no wrap.
21. **Ace-low run with natural 2 plus a separate wild.** `A♥-2♥-3♥-4♥-Joker(@5)-6♥`. **Expect:** legal; `A` low, `2♥` natural, one wild → suja if ≥7.
22. **Wild relocation onto the ace slot.** Meld `2♦(nat)-3♦-Joker(@4)-5♦`; ADD `4♦`. **Expect:** joker relocates low-first to position 1 representing `A♦`; result spans 1–5 (`J(A)-2-3-4-5`). Deterministic, no player choice.
23. **Set of only wilds rejected.** `2♥-2♠-Joker` as a set. **Expect:** rejected — sets need ≥`min_naturals_per_meld` naturals; no natural-2 exception for sets; rank-2 CREATE_SET ids masked while 2 is wild.
24. **Set at wild limit.** Buraco set `K♥-K♠-2♦(wild)`: OK. ADD a joker → 2 wilds. **Expect:** rejected (limit 1). Canasta profile (`limit=3`, `min_naturals=2`): `K-K-2-Joker` legal.
25. **Joker scores its own value.** Set `Q♥-Q♠-Q♦-Joker`. **Expect:** card points `10+10+10+20=50` (Buraco values); wild present → suja if it reaches 7 cards.

### Trash edge cases
26. **Take 30-card pile with 1 card in hand.** **Expect:** hand → 31; unconditional, no meld required. Legal.
27. **Discard the card just drawn from stock.** DRAW_STOCK gets `9♣`; DISCARD(`9♣`) immediately. **Expect:** legal in Buraco. (`no_immediate_redraw_discard` applies only to pile draws in Biriba/Rummy — stock redraws stay legal there too.)
28. **Empty pile → TAKE_PILE illegal.** First turn, `trash=[]`. **Expect:** only DRAW_STOCK legal (Buraco `initial_upcard=False`).
29. **No freeze in Buraco.** Pile top is a wild `2♠`. **Expect:** still takeable unconditionally. (Canasta profile: same pile is frozen — needs two naturals matching the top.)
30. **Rummy top-card draw.** Rummy profile. **Expect:** exactly one card (`trash[-1]`) enters hand; `no_immediate_redraw_discard` forbids discarding it back this turn.

### Canastra transitions
31. **Limpa → suja on wild add.** Clean 7-card canastra (200); ADD a wild `2♦` as 8th card. **Expect:** `wild_count=1` → suja; bonus 200→100 at scoring.
32. **Grow past 7, stays clean.** 7-card limpa; ADD natural 8th. **Expect:** 8-card canastra, still limpa, one 200 bonus (never two).
33. **One meld = one canastra.** A 14-card sequence. **Expect:** exactly one canastra bonus regardless of length.
34. **Canastra threshold enables bater.** Side had none; an ADD pushes a meld to 7. **Expect:** `is_canastra=True`; bater now satisfiable.
35. **Natural-2 doesn't dirty.** `2♥-3♥-4♥-5♥-6♥-7♥-8♥-9♥` with the `2♥` natural at value-2. **Expect:** `wild_count=0` → **limpa**, 200.

### Scoring
36. **Negative round score.** Side never melded, holds `A♠,K♥,K♦,10♣` (45 pts) at exhaustion, never took morto. **Expect:** `0 − 45 − 100 = −145`.
37. **Exhaustion scoring for both sides.** STOCK_EXHAUSTED, no bater. **Expect:** each side sums meld points + canastra bonuses, subtracts each of its players' remaining hand values; no bater bonus for anyone.
38. **Untaken-morto penalty.** Side melds +540 but never emptied a hand. **Expect:** −100 → +440.
39. **Bater bonus additive.** Winner: melds +680, bonuses +300 (limpa 200 + suja 100), bater +100, empty hand. **Expect:** +1080. Loser subtracts its hand.
40. **Rummy scoring flips sign.** Rummy `OPPONENT_POSITIVE`: winner goes out; opponent holds `K♥,7♠,A♦` = 10+7+1 = 18. **Expect:** winner +18, opponent 0.

### Going-out guards (recap)
41. **Emptying while morto untaken is never a bater.** Player empties hand believing they went out. **Expect:** engine performs morto pickup (scenarios 1–2); round continues.
42. **Legal set never strands.** Morto taken, no canastra: every hand-emptying action is filtered from the legal set. **Expect:** legal actions always leave `hand ≥ 1` until a valid bater exists (SPEC 01 invariant §8.3).
