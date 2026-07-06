// Pure state → DOM. Rebuilds each zone from the authoritative view; every
// card node carries data-fkey (stable identity for FLIP) and data-card.
// Own melds render in server order — Add.slot indexes that order.

export const RANKS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"];
export const SUITS = ["♣", "♦", "♥", "♠"];
export const JOKER_ID = 52;

export function cardLabel(id) {
  return id === JOKER_ID ? "Joker" : RANKS[id % 13] + SUITS[Math.floor(id / 13)];
}

function isRed(id) {
  const suit = Math.floor(id / 13);
  return id !== JOKER_ID && (suit === 1 || suit === 2);
}

export function cardNode(id, { key, faceDown = false, wild = false, mini = false } = {}) {
  const el = document.createElement("div");
  let cls = "card";
  if (mini) cls += " mini";
  if (faceDown) cls += " down";
  else if (id === JOKER_ID) cls += " joker";
  else if (isRed(id)) cls += " red";
  if (wild) cls += " wild-acting";
  el.className = cls;
  if (key) el.dataset.fkey = key;
  if (!faceDown && id != null) el.dataset.card = String(id);
  if (faceDown) {
    el.innerHTML = '<div class="back"></div>';
  } else {
    const rank = id === JOKER_ID ? "JOKER" : RANKS[id % 13];
    const suit = id === JOKER_ID ? "★" : SUITS[Math.floor(id / 13)];
    el.innerHTML =
      `<div class="face"><div class="idx">${rank}<small>${id === JOKER_ID ? "" : suit}</small></div>` +
      `<div class="pip">${suit}</div></div>`;
  }
  return el;
}

// ---------- per-zone renderers ------------------------------------------------

const $ = (id) => document.getElementById(id);

function seatName(state, seat) {
  if (seat === state.human_seat) return "You";
  const kind = state.bots[seat] || "bot";
  return (state.num_players === 4 && seat === state.partner_seat)
    ? `Partner (${kind})` : `Bot ${seat} (${kind})`;
}

function plaque(state, seat) {
  const el = document.createElement("div");
  const active = !state.done && state.to_play === seat;
  el.className = "seat-plaque" + (active ? " active" : "") + (seat === state.human_seat ? "" : " bot");
  el.dataset.seat = String(seat);
  el.innerHTML =
    `<div class="nome">${seatName(state, seat)}</div>` +
    `<div class="contagem">${state.all_hand_sizes[seat]} card${state.all_hand_sizes[seat] === 1 ? "" : "s"}</div>` +
    '<div class="thinking">thinking…</div>';
  return el;
}

function backRow(state, seat) {
  const row = document.createElement("div");
  row.className = "back-row";
  const count = Math.min(state.all_hand_sizes[seat], 12);
  for (let j = 0; j < count; j++) {
    row.appendChild(cardNode(null, { key: `b-${seat}-${j}`, faceDown: true }));
  }
  return row;
}

function renderSeats(state) {
  const top = $("seat-top"), left = $("seat-left"), right = $("seat-right");
  top.replaceChildren();
  left.replaceChildren();
  right.replaceChildren();
  const n = state.num_players;
  const rel = (seat) => (seat - state.human_seat + n) % n;
  const zones = n === 2 ? { 1: top } : { 1: right, 2: top, 3: left };
  left.hidden = right.hidden = n === 2;
  for (let seat = 0; seat < n; seat++) {
    if (seat === state.human_seat) continue;
    const zone = zones[rel(seat)];
    zone.appendChild(plaque(state, seat));
    zone.appendChild(backRow(state, seat));
  }
}

function meldPanel(state, meld, side, index, ui) {
  const el = document.createElement("div");
  const min = state.rules.canastra_min_size;
  el.className = "meld" + (meld.is_canastra ? (meld.is_clean ? " canastra" : " canastra suja") : "");
  el.dataset.side = side;
  el.dataset.index = String(index);
  if (side === "own" && ui.addTargets && ui.addTargets.has(index)) el.classList.add("legal-target");

  const cards = document.createElement("div");
  cards.className = "meld-cards";
  const occ = {};
  for (const slot of meld.slots) {
    occ[slot.card] = (occ[slot.card] || 0) + 1;
    cards.appendChild(cardNode(slot.card, {
      key: `m-${side}-${index}-${slot.card}-${occ[slot.card]}`,
      wild: slot.role === "wild",
    }));
  }
  const tag = document.createElement("div");
  tag.className = "meld-tag";
  const kind = meld.kind === "seq" ? `seq ${SUITS[meld.suit]}` : `set ${RANKS[meld.rank]}`;
  let badge = "";
  if (meld.is_canastra) {
    badge = `<span class="badge">${meld.is_clean ? "LIMPA" : "SUJA"}</span>`;
  } else if (min > 0 && min <= 12) {
    const pips = Array.from({ length: min }, (_, i) =>
      `<span class="${i < meld.size ? "on" : ""}">●</span>`).join("");
    badge = `<span class="pips" title="${meld.size}/${min} to canastra">${pips}</span>`;
  }
  tag.innerHTML = `<span>${kind}</span><span class="pts">${meld.points + meld.bonus} pts</span>${badge}`;
  el.appendChild(cards);
  el.appendChild(tag);
  return el;
}

function renderShelf(state, ui, side) {
  const shelf = $(side === "own" ? "own-melds" : "opp-melds");
  shelf.replaceChildren();
  const label = document.createElement("div");
  label.className = "shelf-label";
  label.textContent = side === "own"
    ? (state.num_players === 4 ? "our melds" : "your melds")
    : "their melds";
  shelf.appendChild(label);
  const melds = state.melds[side];
  if (!melds.length) {
    const ph = document.createElement("div");
    ph.className = "placeholder";
    ph.textContent = side === "own" ? "no melds yet — meld from your hand" : "no melds yet";
    shelf.appendChild(ph);
    return;
  }
  melds.forEach((meld, i) => shelf.appendChild(meldPanel(state, meld, side, i, ui)));
}

function pileNode(id, label, opts = {}) {
  const el = document.createElement("div");
  el.className = "pile" + (opts.empty ? " empty" : "") + (opts.taken ? " taken" : "");
  el.id = id;
  if (opts.key) el.dataset.fkey = opts.key;
  const depth = opts.empty ? 0 : Math.min(opts.depth || 1, 3);
  for (let i = 0; i < depth; i++) el.appendChild(cardNode(null, { faceDown: true }));
  if (opts.count != null && !opts.empty) {
    const c = document.createElement("div");
    c.className = "pile-count";
    c.textContent = opts.count;
    el.appendChild(c);
  }
  const l = document.createElement("div");
  l.className = "pile-label";
  l.textContent = label;
  el.appendChild(l);
  return el;
}

function renderCenter(state, ui) {
  const deckZone = $("deck-zone");
  deckZone.replaceChildren();
  const deck = pileNode("deck-pile", "stock", {
    key: "deck", count: state.deck_size, empty: state.deck_size === 0,
    depth: state.deck_size > 20 ? 3 : state.deck_size > 5 ? 2 : 1,
  });
  if (ui.canDrawDeck) deck.classList.add("legal-target");
  deckZone.appendChild(deck);

  const mortoZone = $("morto-zone");
  mortoZone.replaceChildren();
  if (state.rules.morto_count > 0) {
    const taken = state.morto.taken; // [own, opp]
    const present = state.morto.present;
    ["own", "opp"].forEach((who, i) => {
      const intoStock = !present[i] && !taken[i]; // folded into the stock (biriba)
      const stack = pileNode(`morto-${who}`, i === 0 ? "morto·you" : "morto·them", {
        key: `morto-${who}`, taken: taken[i], empty: intoStock, depth: 2,
      });
      if (!taken[i] && present[i] && i === 0) {
        stack.title = "Taken when your hand empties — untaken costs 100";
      }
      mortoZone.appendChild(stack);
    });
  }

  const trashZone = $("trash-zone");
  trashZone.replaceChildren();
  trashZone.className = "";
  const label = document.createElement("div");
  label.className = "zone-label";
  label.textContent = `lixo · ${state.trash.length}`;
  trashZone.appendChild(label);
  const fan = document.createElement("div");
  fan.id = "trash-fan";
  if (state.trash.length > 14) fan.classList.add("tight");
  state.trash.forEach((card, i) => {
    fan.appendChild(cardNode(card, { key: `t-${i}` }));
  });
  if (!state.trash.length) {
    const hint = document.createElement("div");
    hint.className = "empty-hint";
    hint.textContent = "empty — discards land here, face up";
    fan.appendChild(hint);
  }
  trashZone.appendChild(fan);
  if (state.canasta.pile_frozen_for_you || state.canasta.pile_blocked) {
    trashZone.classList.add("frozen");
    if (state.canasta.pile_blocked) trashZone.classList.add("blocked");
    const tag = document.createElement("div");
    tag.className = "frost-tag";
    tag.textContent = state.canasta.pile_blocked ? "⛔ blocked (black 3)"
      : state.canasta.frozen ? "❄ frozen" : "❄ frozen for you";
    trashZone.appendChild(tag);
  }
  if (ui.canDrawTrash || ui.canDiscard) trashZone.classList.add("legal-target");

  renderCanastaSignals(state);
}

function renderCanastaSignals(state) {
  const box = $("canasta-signals");
  box.replaceChildren();
  const sigs = [];
  const c = state.canasta;
  if (c.pending_pile_card != null) {
    sigs.push({ card: c.pending_pile_card, html: "<b>must meld</b> this pile card now" });
  }
  if (c.initial_meld_enabled && !c.initial_meld_done[0] && !state.done) {
    sigs.push({ html: `open at ≥ <b>${c.initial_meld_min}</b> pts` +
      (c.staged_points ? ` · staged <b>${c.staged_points}</b>` : "") });
  }
  const r3 = state.red_threes;
  if (r3.own.length || r3.opp.length) {
    sigs.push({ html: `red 3s — you <b>${r3.own.length}</b> · them <b>${r3.opp.length}</b>`,
                cards: [...r3.own] });
  }
  for (const sig of sigs) {
    const row = document.createElement("div");
    row.className = "sig";
    if (sig.card != null) row.appendChild(cardNode(sig.card, { mini: true, key: `pin-${sig.card}` }));
    if (sig.cards) sig.cards.forEach((cd, i) =>
      row.appendChild(cardNode(cd, { mini: true, key: `r3-own-${i}` })));
    const span = document.createElement("span");
    span.innerHTML = sig.html;
    row.appendChild(span);
    box.appendChild(row);
  }
}

function renderHand(state, ui) {
  const zone = $("hand-zone");
  zone.replaceChildren();
  const fan = document.createElement("div");
  fan.id = "hand-fan";
  const occ = {};
  for (const card of state.hand) {
    occ[card] = (occ[card] || 0) + 1;
    const el = cardNode(card, { key: `h-${card}-${occ[card]}` });
    el.dataset.hand = "1";
    const usable = ui.usableCards && ui.usableCards.has(card);
    el.classList.add(usable ? "playable" : "inert");
    if (ui.selected === card) el.classList.add("selected");
    fan.appendChild(el);
  }
  zone.appendChild(fan);
}

function renderBanner(state, ui) {
  const banner = $("phase-banner");
  banner.className = "";
  let lead = "", hint = "";
  if (state.done) {
    if (state.truncated) { lead = "Adjourned"; hint = "turn cap reached — no result"; }
    else {
      const won = state.winner_side === state.human_side;
      lead = won ? "Victory" : state.winner_side == null ? "Draw" : "Defeat";
      hint = "see the score sheet";
    }
  } else if (state.to_play_is_human) {
    banner.classList.add("yours");
    if (state.phase === "DRAW") {
      lead = "Your turn — draw";
      hint = "click the stock, or take the whole lixo";
    } else {
      lead = "Your turn — meld or discard";
      hint = ui.selected != null
        ? `${cardLabel(ui.selected)} selected — click a glowing target, or click it again to unselect`
        : "click a card in your hand to see its moves";
    }
  } else {
    lead = `${seatName(state, state.to_play)} is playing…`;
    hint = state.phase === "DRAW" ? "drawing" : "melding / discarding";
  }
  banner.innerHTML = `<span class="lead">${lead}</span><span class="hint">${hint}</span>`;
}

function renderButtons(state, ui) {
  const box = $("action-buttons");
  box.replaceChildren();
  if (!state.to_play_is_human) return;
  const mk = (label, cls, act) => {
    const b = document.createElement("button");
    b.className = cls;
    b.textContent = label;
    b.dataset.act = act;
    box.appendChild(b);
  };
  if (ui.creates && ui.creates.length) mk(`New meld (${ui.creates.length})`, "big-btn quiet", "tray");
  if (ui.goOutId != null) mk("Bater — go out", "big-btn", "goout");
  if (ui.endRoundId != null) mk("End round", "big-btn", "endround");
}

function renderPlacar(state) {
  const placar = $("placar");
  const [usM, themM] = state.scores.match;
  const [usR, themR] = state.scores.round_public;
  const row = (who, m, r, lead) =>
    `<div class="side-score${lead ? " leading" : ""}"><span class="who">${who}</span>` +
    `<span class="pts">${m}</span><span class="delta">${r >= 0 ? "+" : ""}${r}</span></div>`;
  placar.innerHTML =
    row(state.num_players === 4 ? "us" : "you", usM, usR, usM >= themM) +
    row("them", themM, themR, themM > usM) +
    (state.episode === "match" ? `<span class="target">to ${state.scores.target}</span>` : "");
  $("profile-tag").textContent =
    `${state.profile} · ${state.num_players}p · ${state.episode === "match" ? "match" : "round " } · seed ${state.seed}`;
}

function renderTicker(log) {
  const el = $("ticker");
  el.replaceChildren();
  for (const row of log.slice(-6)) {
    const div = document.createElement("div");
    div.className = "row" + (row.you ? " you" : "");
    div.innerHTML = `<b>${row.who}</b> ${row.text}`;
    el.appendChild(div);
  }
}

// ---------- top-level ----------------------------------------------------------

export function render(state, ui, log) {
  renderPlacar(state);
  renderSeats(state);
  renderShelf(state, ui, "opp");
  renderCenter(state, ui);
  renderShelf(state, ui, "own");
  renderBanner(state, ui);
  renderButtons(state, ui);
  renderHand(state, ui);
  renderTicker(log);
}
