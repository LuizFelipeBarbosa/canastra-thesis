// FLIP engine over data-fkey nodes.
//
// transition(mutate, hints): snapshot keyed rects → re-render → snapshot
// again → same key moved: slide it; key entered: pair it with an exited key
// carrying the same card id (hand→lixo, lixo→hand, hand→meld…) and fly a
// ghost between the rects, else fly from a hint source (stock, a seat
// plaque, the morto stack) with a face-flip when the card was hidden; key
// exited unpaired: fly a ghost to the hint target or fade. Animation is
// decoration — the DOM is already authoritative when it runs.

const REDUCED = matchMedia("(prefers-reduced-motion: reduce)").matches;

export class Animator {
  constructor(fxLayer, root) {
    this.fx = fxLayer;
    this.root = root;
    this.mult = 1; // 2 = slow, 1, 0.5, 0.25 = fast, 0 = instant
  }

  setSpeed(mult) { this.mult = mult; }

  dur(base) { return REDUCED || this.mult === 0 ? 0 : base * this.mult; }

  snapshot() {
    const map = new Map();
    for (const el of this.root.querySelectorAll("[data-fkey]")) {
      map.set(el.dataset.fkey, {
        rect: el.getBoundingClientRect(),
        card: el.dataset.card ?? null,
        down: el.querySelector(":scope > .back") != null,
      });
    }
    return map;
  }

  async transition(mutate, { enterSource, exitTarget, orderOf, pair = true } = {}) {
    if (this.dur(1) === 0) { mutate(); return; }
    const before = this.snapshot();
    mutate();

    const after = new Map();
    for (const el of this.root.querySelectorAll("[data-fkey]")) {
      after.set(el.dataset.fkey, el);
    }

    const moves = [], enters = [];
    for (const [key, el] of after) {
      const prev = before.get(key);
      if (prev) moves.push({ el, from: prev.rect });
      else enters.push({ key, el });
    }
    const exits = [];
    for (const [key, info] of before) {
      if (!after.has(key)) exits.push({ key, ...info });
    }

    const flights = [];
    for (const enter of enters) {
      const card = enter.el.dataset.card ?? null;
      let src = null, flip = false;
      if (pair && card != null) {
        const i = exits.findIndex((x) => x.card === card);
        if (i >= 0) src = exits.splice(i, 1)[0].rect;
      }
      if (!src && enterSource) {
        const hint = enterSource(enter.key, enter.el);
        if (hint) { src = hint.rect; flip = !!hint.flip; }
      }
      if (src) flights.push({ el: enter.el, from: src, flip, key: enter.key });
      else this.fadeIn(enter.el);
    }

    const jobs = [];
    for (const move of moves) jobs.push(this.slide(move.el, move.from));

    flights.sort((a, b) => (orderOf ? orderOf(a.key) - orderOf(b.key) : 0));
    const stagger = Math.min(55, 300 / Math.max(flights.length, 1));
    flights.forEach((f, i) => jobs.push(this.fly(f.el, f.from, { flip: f.flip, delay: i * stagger })));

    const exitStagger = Math.min(45, 260 / Math.max(exits.length, 1));
    exits.forEach((x, i) => {
      const target = exitTarget ? exitTarget(x.key) : null;
      if (target === "drop") return;
      jobs.push(this.flyOut(x, target, i * exitStagger));
    });

    await Promise.allSettled(jobs);
  }

  // --- primitives -----------------------------------------------------------

  slide(el, from) {
    const to = el.getBoundingClientRect();
    const dx = from.left - to.left, dy = from.top - to.top;
    if (Math.abs(dx) < 1 && Math.abs(dy) < 1) return Promise.resolve();
    const sx = from.width / to.width, sy = from.height / to.height;
    el.style.transformOrigin = "0 0";
    return el.animate(
      [{ transform: `translate(${dx}px, ${dy}px) scale(${sx}, ${sy})` }, { transform: "none" }],
      { duration: this.dur(300), easing: "cubic-bezier(0.25, 0.9, 0.3, 1)" },
    ).finished.catch(() => {});
  }

  fadeIn(el) {
    return el.animate(
      [{ opacity: 0, transform: "translateY(8px)" }, { opacity: 1, transform: "none" }],
      { duration: this.dur(240), easing: "ease-out" },
    ).finished.catch(() => {});
  }

  async fly(el, from, { flip = false, delay = 0 } = {}) {
    const to = el.getBoundingClientRect();
    if (!to.width) return;
    const ghost = this.makeGhost(el, to);
    const dx = from.left - to.left, dy = from.top - to.top;
    const sx = from.width / to.width, sy = from.height / to.height;
    el.style.visibility = "hidden";
    this.fx.appendChild(ghost);
    const D = this.dur(380);
    // Safety net: never strand a ghost (or a hidden card) if the animation's
    // finished promise fails to settle.
    setTimeout(() => { el.style.visibility = ""; ghost.remove(); }, D + delay + 1500);
    try {
      const lift = ghost.animate(
        [
          { transform: `translate(${dx}px, ${dy}px) scale(${sx}, ${sy})` },
          { transform: "none" },
        ],
        { duration: D, delay, easing: "cubic-bezier(0.3, 0.8, 0.3, 1)", fill: "backwards" },
      );
      if (flip) {
        const backSide = document.createElement("div");
        backSide.className = "back";
        ghost.firstChild.appendChild(backSide);
        backSide.animate(
          [
            { transform: "scaleX(1)", offset: 0 },
            { transform: "scaleX(1)", offset: 0.45 },
            { transform: "scaleX(0)", offset: 0.7 },
            { transform: "scaleX(0)", offset: 1 },
          ],
          { duration: D, delay, fill: "forwards" },
        );
      }
      await lift.finished;
    } catch { /* interrupted */ }
    el.style.visibility = "";
    ghost.remove();
  }

  async flyOut(exit, targetRect, delay = 0) {
    const ghost = this.makeGhostFromInfo(exit);
    this.fx.appendChild(ghost);
    setTimeout(() => ghost.remove(), this.dur(400) + delay + 1500);
    const from = exit.rect;
    let frames;
    if (targetRect) {
      const dx = targetRect.left + targetRect.width / 2 - (from.left + from.width / 2);
      const dy = targetRect.top + targetRect.height / 2 - (from.top + from.height / 2);
      frames = [
        { transform: "none", opacity: 1 },
        { transform: `translate(${dx}px, ${dy}px) scale(0.4)`, opacity: 0.15 },
      ];
    } else {
      frames = [{ opacity: 1 }, { opacity: 0, transform: "scale(0.9)" }];
    }
    try {
      await ghost.animate(frames, {
        duration: this.dur(targetRect ? 380 : 200),
        delay, easing: "cubic-bezier(0.4, 0.2, 0.6, 1)", fill: "forwards",
      }).finished;
    } catch { /* interrupted */ }
    ghost.remove();
  }

  makeGhost(el, rect) {
    const ghost = document.createElement("div");
    ghost.className = "ghost";
    ghost.style.cssText =
      `left:${rect.left}px; top:${rect.top}px; width:${rect.width}px; height:${rect.height}px;`;
    const clone = el.cloneNode(true);
    clone.removeAttribute("data-fkey");
    clone.style.cssText = "position:absolute; inset:0; margin:0; visibility:visible;";
    ghost.appendChild(clone);
    return ghost;
  }

  makeGhostFromInfo(exit) {
    const rect = exit.rect;
    const ghost = document.createElement("div");
    ghost.className = "ghost";
    ghost.style.cssText =
      `left:${rect.left}px; top:${rect.top}px; width:${rect.width}px; height:${rect.height}px;`;
    const card = document.createElement("div");
    card.className = "card";
    card.style.cssText = "position:absolute; inset:0; width:100%; height:100%;";
    if (exit.down || exit.card == null) {
      card.innerHTML = '<div class="back"></div>';
    } else {
      const id = Number(exit.card);
      const RANKS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"];
      const SUITS = ["♣", "♦", "♥", "♠"];
      const rank = id === 52 ? "JOKER" : RANKS[id % 13];
      const suit = id === 52 ? "★" : SUITS[Math.floor(id / 13)];
      const red = id !== 52 && (Math.floor(id / 13) === 1 || Math.floor(id / 13) === 2);
      card.className += red ? " red" : id === 52 ? " joker" : "";
      card.innerHTML =
        `<div class="face"><div class="idx">${rank}<small>${id === 52 ? "" : suit}</small></div>` +
        `<div class="pip">${suit}</div></div>`;
    }
    ghost.appendChild(card);
    return ghost;
  }

  // --- flourishes ------------------------------------------------------------

  splash(text, { small = false } = {}) {
    if (this.dur(1) === 0) return Promise.resolve();
    const el = document.createElement("div");
    el.className = "splash" + (small ? " small" : "");
    el.textContent = text;
    el.style.setProperty("--anim", `${this.dur(320)}ms`);
    this.fx.appendChild(el);
    el.classList.add("show");
    return new Promise((resolve) => {
      el.addEventListener("animationend", () => { el.remove(); resolve(); }, { once: true });
      setTimeout(() => { el.remove(); resolve(); }, this.dur(320) * 5 + 400);
    });
  }

  pulse(el) {
    if (!el || this.dur(1) === 0) return;
    el.animate(
      [
        { boxShadow: "0 0 0 0 rgba(223, 166, 62, 0)" },
        { boxShadow: "0 0 34px 8px rgba(223, 166, 62, 0.55)" },
        { boxShadow: "0 0 16px 0 rgba(223, 166, 62, 0.22)" },
      ],
      { duration: this.dur(900), easing: "ease-out" },
    );
  }

  sleep(ms) { return new Promise((r) => setTimeout(r, this.dur(ms))); }
}
