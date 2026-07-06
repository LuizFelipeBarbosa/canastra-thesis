// Orchestrator: interaction state machine + frontend-driven bot loop.
// The server is authoritative; this file only picks among server-sent
// action ids, animates each returned event, then reconciles to the state.

import { api } from "./api.js";
import { Animator } from "./anim.js";
import { cardLabel, cardNode, render } from "./render.js";

const $ = (id) => document.getElementById(id);

let state = null;        // latest authoritative view
let prevState = null;    // previous view (for "became canastra" checks)
let meta = null;
let selected = null;     // selected hand card id
let busy = false;        // one in-flight mutation at a time
let botLoopRunning = false;
let awaitingModal = false;
let skipMode = false;
const log = [];

const animator = new Animator($("fx-layer"), document.body);
const SPEEDS = [
  { mult: 2, label: "½×" },
  { mult: 1, label: "1×" },
  { mult: 0.5, label: "2×" },
  { mult: 0.25, label: "4×" },
];

// ---------- ui derivation -----------------------------------------------------

function deriveUi() {
  const ui = {
    selected, usableCards: new Set(), addTargets: new Set(), creates: [],
    canDrawDeck: false, canDrawTrash: false, canDiscard: false,
    drawDeckId: null, drawTrashId: null, goOutId: null, endRoundId: null,
    discards: new Map(), adds: [],
  };
  if (!state || !state.legal) return ui;
  for (const entry of state.legal) {
    switch (entry.family) {
      case "draw_deck": ui.canDrawDeck = true; ui.drawDeckId = entry.id; break;
      case "draw_trash": ui.canDrawTrash = true; ui.drawTrashId = entry.id; break;
      case "discard": ui.discards.set(entry.card, entry.id); ui.usableCards.add(entry.card); break;
      case "add": ui.adds.push(entry); ui.usableCards.add(entry.card); break;
      case "create_seq":
      case "create_set":
        ui.creates.push(entry);
        for (const c of entry.uses) ui.usableCards.add(c);
        break;
      case "go_out": ui.goOutId = entry.id; break;
      case "end_round": ui.endRoundId = entry.id; break;
    }
  }
  if (selected != null) {
    ui.canDiscard = ui.discards.has(selected);
    for (const a of ui.adds) if (a.card === selected) ui.addTargets.add(a.slot);
  }
  return ui;
}

let lastUi = deriveUi();

function rerender() {
  lastUi = deriveUi();
  render(state, lastUi, log);
  $("moves-btn").hidden = !(state.to_play_is_human && state.legal);
  $("skip-btn").hidden = state.done || state.to_play_is_human;
}

// ---------- animation hints ---------------------------------------------------

function rectOf(sel) {
  const el = document.querySelector(sel);
  return el ? el.getBoundingClientRect() : null;
}

function plaqueRect(seat) {
  return rectOf(`.seat-plaque[data-seat="${seat}"]`) || rectOf("#deck-pile");
}

function enterSourceFor(event) {
  return (key) => {
    const deck = () => ({ rect: rectOf("#deck-pile") || centerRect(), flip: true });
    const mortoRect = event.actor_side === state.human_side ? "#morto-own" : "#morto-opp";
    const tookMorto = event.diff.morto_taken_by.includes(event.actor_side);
    if (key.startsWith("h-")) {
      if (tookMorto && event.actor_is_human) return { rect: rectOf(mortoRect) || centerRect(), flip: true };
      return deck();
    }
    if (key.startsWith("b-")) {
      if (event.family === "draw_trash") return { rect: rectOf("#trash-zone") };
      if (tookMorto) return { rect: rectOf(mortoRect) || centerRect() };
      return { rect: rectOf("#deck-pile") || centerRect() };
    }
    if (key.startsWith("m-")) {
      return { rect: plaqueRect(event.actor) || centerRect(), flip: !event.actor_is_human };
    }
    if (key.startsWith("t-")) {
      return event.actor_is_human ? null : { rect: plaqueRect(event.actor), flip: true };
    }
    if (key.startsWith("r3-") || key.startsWith("pin-")) return deck();
    return null;
  };
}

function centerRect() {
  return { left: innerWidth / 2, top: innerHeight / 2, width: 60, height: 84 };
}

function exitTargetFor(event) {
  return (key) => {
    if (key.startsWith("t-") && !event.actor_is_human) return plaqueRect(event.actor);
    if (key.startsWith("b-")) return "drop";
    return null;
  };
}

function orderOf(key) {
  const m = key.match(/(\d+)$/);
  return m ? Number(m[1]) : 0;
}

// ---------- event application -------------------------------------------------

function pushLog(event) {
  const who = event.actor_is_human ? "you" : `bot ${event.actor}`;
  log.push({ who, text: event.label, you: event.actor_is_human });
  if (log.length > 40) log.shift();
}

async function animateEvent(event, newState) {
  pushLog(event);
  prevState = state;
  state = newState;
  selected = null;
  closePopovers();

  // When the match rolls into the next round, newState is already the fresh
  // deal — keep showing the finished round until the score sheet is closed.
  const deferRender = event.new_round;
  if (!deferRender) {
    await animator.transition(() => rerender(), {
      enterSource: enterSourceFor(event),
      exitTarget: exitTargetFor(event),
      orderOf,
    });
  }

  // flourishes, driven by the event/diff — signals, not state
  const jobs = [];
  if (event.meld && event.meld.is_canastra && prevState) {
    const side = event.meld_side === state.human_side ? "own" : "opp";
    const was = prevState.melds[side]?.[event.meld_index];
    if (!was || !was.is_canastra) {
      const panel = document.querySelector(`.meld[data-side="${side}"][data-index="${event.meld_index}"]`);
      animator.pulse(panel);
      jobs.push(animator.splash(event.meld.is_clean ? "CANASTRA LIMPA" : "CANASTRA SUJA", { small: true }));
    }
  }
  if (event.diff.morto_taken_by.length) jobs.push(animator.splash("MORTO!", { small: true }));
  if (event.diff.morto_converted_to_stock) jobs.push(animator.splash("morto → stock", { small: true }));
  if (event.diff.frozen === true) jobs.push(animator.splash("❄ pile frozen", { small: true }));
  if (event.family === "go_out") jobs.push(animator.splash("BATER!"));
  if (jobs.length) await Promise.race([Promise.all(jobs), animator.sleep(1100)]);

  if (event.round_ended && event.round_summary) {
    awaitingModal = true;
    await animator.sleep(350);
    showRoundModal(event.round_summary, event);
  }
}

// ---------- server calls -------------------------------------------------------

function toast(msg) {
  const el = $("toast");
  el.textContent = msg;
  el.classList.add("show");
  clearTimeout(el._t);
  el._t = setTimeout(() => el.classList.remove("show"), 2600);
}

async function act(actionId) {
  if (busy || !state || awaitingModal) return;
  busy = true;
  try {
    const { status, data } = await api.action(state.game_id, state.cursor, actionId);
    if (status === 200) {
      await animateEvent(data.event, data.state);
    } else if (status === 409 && data.state) {
      state = data.state; rerender(); toast("Out of sync — table reloaded");
    } else {
      if (data.state) { state = data.state; rerender(); }
      toast(data.error || "That move was refused");
    }
  } finally {
    busy = false;
  }
  botLoop();
}

async function botLoop() {
  if (botLoopRunning || !state) return;
  botLoopRunning = true;
  const baseSpeed = animator.mult;
  try {
    while (state && !state.done && !state.to_play_is_human && !awaitingModal) {
      if (skipMode) animator.setSpeed(0);
      const { status, data } = await api.botStep(state.game_id, state.cursor);
      if (status === 200) {
        await animateEvent(data.event, data.state);
        await animator.sleep(170);
      } else if (status === 409 && data.state) {
        state = data.state; rerender();
      } else {
        toast(data.error || "Bot step failed"); break;
      }
    }
  } finally {
    if (skipMode) { skipMode = false; animator.setSpeed(baseSpeed); rerender(); }
    botLoopRunning = false;
  }
}

// ---------- popovers ------------------------------------------------------------

function closePopovers() {
  $("meld-tray").hidden = true;
  $("all-moves").hidden = true;
}

function openTray() {
  const tray = $("meld-tray");
  const creates = selected != null
    ? lastUi.creates.filter((c) => c.uses.includes(selected))
    : lastUi.creates;
  tray.replaceChildren();
  const h = document.createElement("h3");
  h.textContent = selected != null ? `new melds with ${cardLabel(selected)}` : "new melds";
  tray.appendChild(h);
  for (const entry of creates) {
    const btn = document.createElement("button");
    btn.className = "tray-option";
    btn.dataset.actionId = entry.id;
    const cards = document.createElement("div");
    cards.className = "cards";
    for (const slot of entry.slots_preview || []) {
      cards.appendChild(miniCard(slot.card, slot.role === "wild"));
    }
    const what = document.createElement("div");
    what.className = "what";
    what.innerHTML = `${entry.family === "create_seq" ? "Sequence" : "Set"}<small>${entry.label}</small>`;
    btn.appendChild(cards);
    btn.appendChild(what);
    tray.appendChild(btn);
  }
  tray.hidden = creates.length === 0;
}

function miniCard(id, wild) {
  return cardNode(id, { mini: true, wild });
}

function openAllMoves() {
  const box = $("all-moves");
  box.replaceChildren();
  const h = document.createElement("h3");
  h.textContent = "all legal moves";
  box.appendChild(h);
  for (const entry of state.legal || []) {
    const btn = document.createElement("button");
    btn.className = "move-row";
    btn.dataset.actionId = entry.id;
    btn.innerHTML = `<span class="fam">${entry.family.replace("_", " ")}</span>${entry.label}`;
    box.appendChild(btn);
  }
  box.hidden = false;
}

// ---------- modals ---------------------------------------------------------------

function sideName(side) {
  return side === state.human_side ? (state.num_players === 4 ? "Us" : "You") : "Them";
}

function showRoundModal(summary, event) {
  const root = $("modal-root");
  const you = state.human_side, them = 1 - you;
  const bySide = {};
  for (const per of summary.per_side) bySide[per.side] = per;
  const Y = bySide[you], T = bySide[them];
  const fmt = (v) => v === 0 ? "·" : String(v);
  const cls = (v) => v > 0 ? ' class="pos"' : v < 0 ? ' class="neg"' : "";
  const row = (label, a, b) =>
    `<tr><td>${label}</td><td${cls(a)}>${fmt(a)}</td><td${cls(b)}>${fmt(b)}</td></tr>`;

  const why = summary.end_reason === "BATER"
    ? `${sideName(summary.went_out_side)} went out (bater), turn ${summary.turn_number}`
    : "stock exhausted";
  let html = `<div class="modal"><h2>Round ${summary.round_index + 1} — score sheet</h2>` +
    `<div class="why">${why}</div><table class="score-table">` +
    `<tr><th></th><th>${sideName(you)}</th><th>${sideName(them)}</th></tr>` +
    row("Meld cards", Y.meld_points, T.meld_points) +
    row("Canastra bonuses", Y.canastra_bonus, T.canastra_bonus) +
    row("Going out", Y.go_out_bonus + Y.concealed_bonus, T.go_out_bonus + T.concealed_bonus) +
    (Y.red_three_bonus || T.red_three_bonus ? row("Red threes", Y.red_three_bonus, T.red_three_bonus) : "") +
    (Y.morto_penalty || T.morto_penalty ? row("Morto untaken", -Y.morto_penalty, -T.morto_penalty) : "") +
    row("Cards left in hand", -Y.hand_penalty + Y.opponent_hand_gain, -T.hand_penalty + T.opponent_hand_gain) +
    `<tr class="total"><td>Round total</td><td>${Y.total}</td><td>${T.total}</td></tr>` +
    `</table>`;

  const revealed = summary.per_side.flatMap((p) => p.hands).filter((h) => h.cards.length);
  if (revealed.length) {
    html += '<div class="revealed">';
    for (const h of revealed) {
      html += `<div class="who">${h.seat === state.human_seat ? "your hand" : `seat ${h.seat}`} — ${h.points} pts left</div><div class="cards" data-hand-of="${h.seat}"></div>`;
    }
    html += "</div>";
  }
  html += `<div class="match-line"><span>Match</span>` +
    `<span class="big">${summary.match_scores_after[you]} × ${summary.match_scores_after[them]}</span>` +
    (state.episode === "match" ? `<span>to ${state.scores.target}</span>` : "<span></span>") +
    `</div><div class="actions">` +
    (summary.match_over
      ? '<button class="big-btn" data-modal="final">Final result</button>'
      : '<button class="big-btn" data-modal="next">Next round</button>') +
    "</div></div>";

  root.innerHTML = html;
  for (const per of summary.per_side) {
    for (const h of per.hands) {
      const box = root.querySelector(`[data-hand-of="${h.seat}"]`);
      if (box) for (const c of h.cards) box.appendChild(miniCard(c, false));
    }
  }
  root.hidden = false;
  root.querySelector("[data-modal]").onclick = () => {
    root.hidden = true;
    awaitingModal = false;
    if (summary.match_over || state.done) showFinalModal(summary);
    else { dealAnimation(); }
  };
}

function showFinalModal(summary) {
  const root = $("modal-root");
  const you = state.human_side;
  const scores = summary ? summary.match_scores_after : state.scores.match.slice();
  const yourScore = summary ? scores[you] : state.scores.match[0];
  const theirScore = summary ? scores[1 - you] : state.scores.match[1];
  const won = state.winner_side === you;
  const draw = state.winner_side == null;
  root.innerHTML =
    `<div class="modal final"><h2>${state.truncated ? "Adjourned" : won ? "Vitória!" : draw ? "Empate" : "Derrota"}</h2>` +
    `<div class="verdict ${won ? "win" : draw ? "" : "loss"}">` +
    (state.truncated ? "Turn cap reached — no result." :
      `${sideName(state.winner_side ?? you)} ${draw ? "— dead even" : "take the match"}, ${Math.max(yourScore, theirScore)} × ${Math.min(yourScore, theirScore)}`) +
    `</div><div class="match-line"><span>Final</span><span class="big">${yourScore} × ${theirScore}</span><span></span></div>` +
    `<div class="actions"><button class="big-btn quiet" data-modal="close">Back to the table</button>` +
    `<button class="big-btn" data-modal="new">New game</button></div></div>`;
  root.hidden = false;
  root.querySelector('[data-modal="close"]').onclick = () => { root.hidden = true; };
  root.querySelector('[data-modal="new"]').onclick = () => { root.hidden = true; showSetup(); };
}

async function dealAnimation() {
  awaitingModal = false;
  await animator.transition(() => rerender(), {
    enterSource: (key) => ({
      rect: rectOf("#deck-pile") || centerRect(),
      flip: key.startsWith("h-") || key.startsWith("t-"),
    }),
    exitTarget: () => "drop",
    orderOf,
    pair: false,
  });
  botLoop();
}

// ---------- setup screen ----------------------------------------------------------

const setupState = { profile: "buraco", players: 2, human: 0, episode: "round", bots: {}, seed: "" };

function showSetup() {
  $("table").hidden = true;
  $("setup").hidden = false;
  buildSetup();
}

function buildSetup() {
  const box = $("setup");
  const profiles = Object.keys(meta.profiles);
  const allowed = meta.profiles[setupState.profile].players;
  if (!allowed.includes(setupState.players)) setupState.players = allowed[0];
  if (setupState.human >= setupState.players) setupState.human = 0;

  const seg = (opts, cur, name) => opts.map((o) =>
    `<button data-set="${name}" data-val="${o}" class="${String(o) === String(cur) ? "on" : ""}">${o}</button>`).join("");

  let botRows = "";
  for (let seat = 0; seat < setupState.players; seat++) {
    if (seat === setupState.human) continue;
    const cur = setupState.bots[seat] || "heuristic";
    botRows += `<div class="field"><label>seat ${seat} bot</label><div class="seg">` +
      meta.bots.map((b) =>
        `<button data-set="bot-${seat}" data-val="${b}" class="${b === cur ? "on" : ""}">${b}</button>`).join("") +
      "</div></div>";
  }

  box.innerHTML = `<div class="setup-card">
    <h1>BURACO</h1>
    <div class="sub">the whole lixo is open — take it all or draw blind</div>
    <div class="field"><label>rules</label><div class="seg">${seg(profiles, setupState.profile, "profile")}</div></div>
    <div class="field"><label>players</label><div class="seg">${seg(allowed, setupState.players, "players")}</div></div>
    <div class="field"><label>your seat</label><div class="seg">${seg([...Array(setupState.players).keys()], setupState.human, "human")}</div></div>
    ${botRows}
    <div class="field"><label>length</label><div class="seg">
      <button data-set="episode" data-val="round" class="${setupState.episode === "round" ? "on" : ""}">single round</button>
      <button data-set="episode" data-val="match" class="${setupState.episode === "match" ? "on" : ""}">full match</button>
    </div></div>
    <div class="field"><label>seed (optional)</label><input type="number" id="seed-input" value="${setupState.seed}" placeholder="random"></div>
    <button class="deal-btn" id="deal-btn">Deal the cards</button>
    <div class="error" id="setup-error"></div>
  </div>`;

  box.querySelectorAll("[data-set]").forEach((btn) => {
    btn.onclick = () => {
      const k = btn.dataset.set, v = btn.dataset.val;
      if (k === "profile") setupState.profile = v;
      else if (k === "players") setupState.players = Number(v);
      else if (k === "human") setupState.human = Number(v);
      else if (k === "episode") setupState.episode = v;
      else if (k.startsWith("bot-")) setupState.bots[Number(k.slice(4))] = v;
      setupState.seed = $("seed-input").value;
      buildSetup();
    };
  });
  $("deal-btn").onclick = startGame;
}

async function startGame() {
  setupState.seed = $("seed-input").value;
  const bots = [];
  for (let seat = 0; seat < setupState.players; seat++) {
    bots.push(seat === setupState.human ? "heuristic" : (setupState.bots[seat] || "heuristic"));
  }
  const params = {
    profile: setupState.profile,
    num_players: setupState.players,
    human_seat: setupState.human,
    episode: setupState.episode,
    bots,
  };
  if (setupState.seed !== "") params.seed = Number(setupState.seed);
  const { status, data } = await api.newGame(params);
  if (status !== 200) {
    $("setup-error").textContent = data.error || "could not start the game";
    return;
  }
  log.length = 0;
  state = data;
  prevState = null;
  selected = null;
  $("setup").hidden = true;
  $("table").hidden = false;
  dealAnimation();
}

// ---------- input wiring -----------------------------------------------------------

function wire() {
  $("table").addEventListener("click", (e) => {
    if (busy || awaitingModal || !state) return;
    const card = e.target.closest(".card[data-hand]");
    if (card && state.to_play_is_human && state.phase === "PLAY") {
      const id = Number(card.dataset.card);
      selected = selected === id ? null : id;
      rerender();
      if (selected != null) {
        const withCard = lastUi.creates.filter((c) => c.uses.includes(selected));
        if (withCard.length && lastUi.addTargets.size === 0 && !lastUi.canDiscard) openTray();
        else closePopovers();
      } else closePopovers();
      return;
    }
    const meld = e.target.closest(".meld.legal-target");
    if (meld && selected != null) {
      const slot = Number(meld.dataset.index);
      const add = lastUi.adds.find((a) => a.card === selected && a.slot === slot);
      if (add) act(add.id);
      return;
    }
    const deck = e.target.closest("#deck-pile.legal-target");
    if (deck) { act(lastUi.drawDeckId); return; }
    const trash = e.target.closest("#trash-zone.legal-target");
    if (trash) {
      if (state.phase === "DRAW" && lastUi.canDrawTrash) act(lastUi.drawTrashId);
      else if (selected != null && lastUi.canDiscard) act(lastUi.discards.get(selected));
      return;
    }
    const btn = e.target.closest("button[data-act]");
    if (btn) {
      if (btn.dataset.act === "tray") { $("meld-tray").hidden ? openTray() : closePopovers(); }
      else if (btn.dataset.act === "goout") act(lastUi.goOutId);
      else if (btn.dataset.act === "endround") act(lastUi.endRoundId);
    }
  });

  document.addEventListener("click", (e) => {
    const opt = e.target.closest("[data-action-id]");
    if (opt && !busy && !awaitingModal) {
      closePopovers();
      act(Number(opt.dataset.actionId));
    }
  });

  $("moves-btn").onclick = () => {
    $("all-moves").hidden ? openAllMoves() : closePopovers();
  };
  $("skip-btn").onclick = () => { skipMode = true; };
  $("new-game-btn").onclick = () => showSetup();

  const slider = $("speed");
  slider.oninput = () => {
    const s = SPEEDS[Number(slider.value)];
    animator.setSpeed(s.mult);
    $("speed-label").textContent = s.label;
  };

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") { selected = null; closePopovers(); if (state) rerender(); }
  });
}

// ---------- boot -----------------------------------------------------------------

async function boot() {
  wire();
  if (location.search.includes("layout-debug")) {
    const probe = document.createElement("div");
    probe.style.cssText = "position:fixed;bottom:0;left:0;background:#000;color:#0f0;" +
      "font:11px monospace;z-index:99;padding:2px 5px;pointer-events:none";
    setInterval(() => {
      const tr = document.querySelector(".topbar-right");
      probe.textContent =
        `doc ${document.documentElement.scrollWidth}w inner ${innerWidth}w ` +
        `topbar ${$("topbar").offsetWidth} right ${tr.offsetWidth}@${tr.offsetLeft} ` +
        `ng ${$("new-game-btn").offsetWidth}@${$("new-game-btn").offsetLeft}`;
    }, 400);
    document.body.appendChild(probe);
  }
  const m = await api.meta();
  meta = m.data;
  if (meta.defaults) {
    if (meta.defaults.profile) setupState.profile = meta.defaults.profile;
    if (meta.defaults.num_players) setupState.players = meta.defaults.num_players;
    if (meta.defaults.human_seat != null) setupState.human = meta.defaults.human_seat;
    if (meta.defaults.episode) setupState.episode = meta.defaults.episode;
  }
  const st = await api.state();
  if (st.status === 200) {
    state = st.data;
    $("table").hidden = false;
    rerender();
    if (state.done && state.last_round_summary) showFinalModal(state.last_round_summary);
    else botLoop();
  } else {
    showSetup();
  }
}

boot();
