// Thin fetch wrappers. Every mutating call carries {game_id, cursor}; the
// server 409s stale cursors with the authoritative state attached.

async function call(method, path, body) {
  const resp = await fetch(path, {
    method,
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  let data = {};
  try { data = await resp.json(); } catch { /* non-JSON error body */ }
  return { status: resp.status, data };
}

export const api = {
  meta: () => call("GET", "/api/meta"),
  state: () => call("GET", "/api/game/state"),
  newGame: (params) => call("POST", "/api/game", params),
  action: (gameId, cursor, actionId) =>
    call("POST", "/api/game/action", { game_id: gameId, cursor, action_id: actionId }),
  botStep: (gameId, cursor) =>
    call("POST", "/api/game/bot-step", { game_id: gameId, cursor }),
};
