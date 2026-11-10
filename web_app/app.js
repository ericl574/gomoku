"use strict";

const board = document.getElementById("board");
const stage = document.getElementById("board-stage");
const newGameButton = document.getElementById("new-game");
const errorBox = document.getElementById("error");
const buttons = [];
let state = null;
let busy = true;
let phase = "opening";
let activeCell = 9 * 19 + 9;

function svgElement(name, attributes) {
  const element = document.createElementNS("http://www.w3.org/2000/svg", name);
  for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, value);
  return element;
}

function buildBoard() {
  const lines = document.getElementById("board-lines");
  for (let index = 0; index < 19; index += 1) {
    const point = 38 + index * 38;
    const className = index === 0 || index === 18 ? "grid-line grid-border" : "grid-line";
    lines.append(svgElement("line", { x1: 38, y1: point, x2: 722, y2: point, class: className }));
    lines.append(svgElement("line", { x1: point, y1: 38, x2: point, y2: 722, class: className }));
    const column = svgElement("text", { x: point, y: 16, class: "coordinate" });
    column.textContent = String.fromCharCode(65 + index);
    const row = svgElement("text", { x: 16, y: point, class: "coordinate" });
    row.textContent = index + 1;
    lines.append(column, row);
  }
  for (const [row, col] of [[3, 3], [3, 15], [9, 9], [15, 3], [15, 15]]) {
    lines.append(svgElement("circle", { cx: 38 + col * 38, cy: 38 + row * 38, r: 3.5, class: "star-point" }));
  }
  for (let row = 0; row < 19; row += 1) {
    for (let col = 0; col < 19; col += 1) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "intersection";
      button.dataset.row = row;
      button.dataset.col = col;
      button.addEventListener("click", () => playMove(row, col));
      button.addEventListener("focus", () => {
        activeCell = row * 19 + col;
        updateTabStops();
      });
      buttons.push(button);
      board.append(button);
    }
  }
}

function updateTabStops() {
  if (buttons[activeCell].disabled) {
    const available = buttons.map((button, index) => ({ button, index })).filter(({ button }) => !button.disabled);
    available.sort((a, b) => Math.abs(a.index - activeCell) - Math.abs(b.index - activeCell));
    if (available.length) activeCell = available[0].index;
  }
  buttons.forEach((button, index) => { button.tabIndex = index === activeCell && !button.disabled ? 0 : -1; });
}

board.addEventListener("keydown", (event) => {
  const directions = { ArrowUp: [-1, 0], ArrowDown: [1, 0], ArrowLeft: [0, -1], ArrowRight: [0, 1] };
  if (!directions[event.key] || busy || !state || state.game_over) return;
  event.preventDefault();
  const [dr, dc] = directions[event.key];
  let row = Math.floor(activeCell / 19) + dr;
  let col = activeCell % 19 + dc;
  while (row >= 0 && row < 19 && col >= 0 && col < 19) {
    const index = row * 19 + col;
    if (!buttons[index].disabled) {
      activeCell = index;
      buttons[index].focus();
      return;
    }
    row += dr;
    col += dc;
  }
});

function render() {
  const winning = new Set((state?.winning_line || []).map(([row, col]) => row * 19 + col));
  const last = state?.last_move;
  buttons.forEach((button, index) => {
    const row = Math.floor(index / 19);
    const col = index % 19;
    const value = state?.board[row][col] || 0;
    const color = value === 1 ? "black" : value === -1 ? "white" : "empty";
    if (button.dataset.value !== color) {
      button.replaceChildren();
      if (value) {
        const stone = document.createElement("span");
        stone.className = `stone ${color}`;
        button.append(stone);
      }
      button.dataset.value = color;
    }
    button.disabled = busy || !state || state.game_over || state.turn !== "white" || value !== 0;
    button.classList.toggle("last-move", Boolean(last && last.row === row && last.col === col));
    button.classList.toggle("winning", winning.has(index));
    button.setAttribute("aria-label", `${String.fromCharCode(65 + col)}${row + 1}, ${color}`);
  });
  updateTabStops();
  stage.setAttribute("aria-busy", String(busy));
  newGameButton.disabled = busy;
  document.getElementById("move-count").textContent = `MOVE ${String(state?.move_count || 0).padStart(2, "0")}`;
  const aiMove = state?.last_ai_move;
  document.getElementById("last-ai-move").textContent = aiMove ? `${String.fromCharCode(65 + aiMove.col)}${aiMove.row + 1}` : "—";
  const model = state?.model;
  const modelInfo = document.getElementById("model-info");
  modelInfo.textContent = model ? `${(model.run || "").replace("alphazero_", "").slice(0, 6)} · iter ${model.iteration ?? "?"} · ${model.simulations} sims` : "—";
  modelInfo.title = model ? `${model.run}/${model.file}, ${model.updates ?? "?"} updates, ${model.architecture}. A newer checkpoint loads when you start a new game.` : "";

  let title = "Your turn.";
  let description = "Find an empty intersection and place your white stone.";
  let label = "WHITE TO PLAY";
  if (busy) {
    title = phase === "opening" ? "Setting the board…" : "Your AI is thinking…";
    description = phase === "opening" ? "Black makes the first move. Then it’s over to you." : "Your stone is down. Black is finding its next move.";
    label = "BLACK TO PLAY";
  } else if (!state) {
    title = "Let’s try again.";
    description = "Start a new game to reconnect to your local game.";
    label = "NOT CONNECTED";
  } else if (state.game_over) {
    title = state.winner === "white" ? "You win!" : state.winner === "black" ? "Your AI wins." : "A draw.";
    description = state.winner === "draw" ? "Every intersection is filled. Ready for another round?" : "Five in a row. A good game deserves another.";
    label = "GAME OVER";
  }
  document.getElementById("status-title").textContent = title;
  document.getElementById("status-description").textContent = description;
  document.getElementById("status-label").textContent = label;
  document.getElementById("status-dot").classList.toggle("thinking", busy);
  document.getElementById("player-black").classList.toggle("active", busy || state?.winner === "black");
  document.getElementById("player-white").classList.toggle("active", !busy && Boolean(state) && (state.turn === "white" || state.winner === "white"));
  document.getElementById("black-tag").textContent = state?.winner === "black" ? "WINNER" : busy ? "THINKING" : "AI";
  document.getElementById("white-tag").textContent = state?.winner === "white" ? "WINNER" : !busy && state?.turn === "white" ? "YOUR TURN" : "YOU";
}

async function request(path, payload) {
  const controller = new AbortController();
  // MCTS with hundreds of simulations (plus a checkpoint load on New game) can take a while.
  const timeout = setTimeout(() => controller.abort(), 120000);
  try {
    const options = { cache: "no-store", signal: controller.signal };
    if (payload !== undefined) {
      options.method = "POST";
      options.headers = { "Content-Type": "application/json" };
      options.body = JSON.stringify(payload);
    }
    const response = await fetch(path, options);
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "The move could not be completed.");
    return data;
  } finally {
    clearTimeout(timeout);
  }
}

function showError(error) {
  errorBox.textContent = error.name === "TypeError" || error.name === "AbortError" ? "Couldn’t reach the game. Check that your local server is running, then try again." : error.message;
  errorBox.hidden = false;
}

async function playMove(row, col) {
  if (busy || !state || state.game_over || state.turn !== "white" || state.board[row][col]) return;
  const previous = state;
  const restoreFocus = board.contains(document.activeElement);
  busy = true;
  phase = "thinking";
  errorBox.hidden = true;
  // Show white immediately; the server still validates and owns the board.
  state = JSON.parse(JSON.stringify(previous));
  state.board[row][col] = -1;
  state.move_count += 1;
  state.last_move = { row, col, player: "white" };
  state.turn = "black";
  render();
  try {
    state = await request("/api/move", { row, col, revision: previous.revision });
  } catch (error) {
    state = previous;
    try { state = await request("/api/state"); } catch (_) { /* Keep the last known board. */ }
    showError(error);
  } finally {
    busy = false;
    render();
    if (restoreFocus && !state?.game_over) buttons[activeCell].focus();
  }
}

newGameButton.addEventListener("click", async () => {
  if (busy) return;
  busy = true;
  phase = "opening";
  errorBox.hidden = true;
  render();
  try {
    state = await request("/api/new", {});
    activeCell = 9 * 19 + 9;
  } catch (error) {
    try { state = await request("/api/state"); } catch (_) { /* Keep the last known board. */ }
    showError(error);
  } finally {
    busy = false;
    render();
  }
});

async function start() {
  buildBoard();
  render();
  try { state = await request("/api/state"); } catch (error) { showError(error); }
  busy = false;
  render();
}

start();
