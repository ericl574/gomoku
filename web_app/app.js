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
const AUTO_NEXT_SECONDS = 0; // learning mode: start the next game right away
let gamesAtStart = null; // learning.games when the current game began (null = unknown)
let nextGameAt = null;   // learning mode: when the next game auto-starts
let countdownTimer = null;
let lastGameNote = "";   // learning result of the previous game, shown at the start of the next

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
  // Colors this game (the AI alternates colors in learning mode).
  const human = state?.human_color || "white";
  const ai = state?.ai_color || "black";
  const cap = (color) => color[0].toUpperCase() + color.slice(1);
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
    button.disabled = busy || !state || state.game_over || state.turn !== human || value !== 0;
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
  modelInfo.textContent = model ? `R${model.iteration ?? "?"}${model.games != null ? ` · ${model.games} games` : ""} · ${model.simulations} sims` : "—";
  modelInfo.title = model ? `${model.run}/${model.file}, ${model.updates ?? "?"} updates, ${model.architecture}. A newer checkpoint loads when you start a new game.` : "";
  const learning = state?.learning;
  const learningInfo = document.getElementById("learning-info");
  learningInfo.textContent = !learning ? "—" : learning.enabled ? `on${learning.imitate_weight ? ` · copies you ×${learning.imitate_weight}` : ""} · ${learning.games} games` : "off";
  const lastGame = learning?.last;
  learningInfo.title = lastGame ? `Last game: ${lastGame.result}, ${lastGame.positions} positions. Loss: policy ${lastGame.policy_loss.toFixed(3)}, value ${lastGame.value_loss.toFixed(3)}. Replay: ${lastGame.buffer} positions.` : "";

  let title = "Your turn.";
  let description = `Find an empty intersection and place your ${human} stone.`;
  let label = `${human.toUpperCase()} TO PLAY`;
  if (busy) {
    title = phase === "opening" ? "Setting the board…" : "Your AI is thinking…";
    description = phase === "opening" ? "Starting a new game." : `Your stone is down. ${cap(ai)} is finding its next move.`;
    label = phase === "opening" ? "NEW GAME" : `${ai.toUpperCase()} TO PLAY`;
  } else if (!state) {
    title = "Let’s try again.";
    description = "Start a new game to reconnect to your local game.";
    label = "NOT CONNECTED";
  } else if (!state.game_over && lastGameNote) {
    description = `${lastGameNote} ${description}`;
  } else if (state.game_over) {
    title = state.winner === human ? "You win!" : state.winner === ai ? "Your AI wins." : "A draw.";
    description = state.winner === "draw" ? "Every intersection is filled. Ready for another round?" : "Five in a row. A good game deserves another.";
    if (learning?.enabled) {
      // gamesAtStart is the server's learned-game count when this game began.
      const learned = gamesAtStart != null && learning.games > gamesAtStart && lastGame;
      if (learned) {
        const moved = lastGame.weight_change != null ? ` Weights moved by ${lastGame.weight_change.toFixed(4)}.` : "";
        description = `Weights updated from this game (learned game #${learning.games}).${moved} Loss: policy ${lastGame.policy_loss.toFixed(3)}, value ${lastGame.value_loss.toFixed(3)}.`;
        lastGameNote = `Last game: ${lastGame.result}; weights updated${lastGame.weight_change != null ? ` (moved ${lastGame.weight_change.toFixed(3)})` : ""}.`;
      } else if (gamesAtStart != null) {
        description = "This game was NOT learned. Check the server terminal for the error.";
      }
    }
    label = "GAME OVER";
  }
  document.getElementById("status-title").textContent = title;
  document.getElementById("status-description").textContent = description;
  document.getElementById("status-label").textContent = label;
  document.getElementById("status-dot").classList.toggle("thinking", busy);
  for (const color of ["black", "white"]) {
    const isAI = color === ai;
    document.getElementById(`${color}-name`).textContent = isAI ? "Your AI" : "You";
    document.getElementById(`player-${color}`).classList.toggle("active", Boolean(state) && (state.winner === color || (isAI ? busy : !busy && state.turn === color)));
    document.getElementById(`${color}-tag`).textContent = state?.winner === color ? "WINNER" : isAI ? (busy ? "THINKING" : "AI") : (!busy && state?.turn === color ? "YOUR TURN" : "YOU");
  }
  document.getElementById("intro-note").textContent = `Your trained AI plays ${ai} this game. You play ${human}.`;
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
  if (busy || !state || state.game_over || state.turn !== state.human_color || state.board[row][col]) return;
  lastGameNote = "";
  const previous = state;
  const restoreFocus = board.contains(document.activeElement);
  busy = true;
  phase = "thinking";
  errorBox.hidden = true;
  // Show your stone immediately; the server still validates and owns the board.
  state = JSON.parse(JSON.stringify(previous));
  state.board[row][col] = previous.human_color === "black" ? 1 : -1;
  state.move_count += 1;
  state.last_move = { row, col, player: previous.human_color };
  state.turn = previous.ai_color;
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
    scheduleNextGame();
  }
}

// Learning mode: after a game ends, show the result briefly, then start the next one.
function scheduleNextGame() {
  if (!state?.game_over || !state.learning?.enabled || nextGameAt) return;
  nextGameAt = Date.now() + AUTO_NEXT_SECONDS * 1000;
  countdownTimer = setInterval(() => {
    if (Date.now() >= nextGameAt) startNewGame();
    else render();
  }, 250);
}

function cancelNextGame() {
  clearInterval(countdownTimer);
  countdownTimer = null;
  nextGameAt = null;
}

async function startNewGame() {
  cancelNextGame();
  if (busy) return;
  busy = true;
  phase = "opening";
  errorBox.hidden = true;
  render();
  try {
    state = await request("/api/new", {});
    gamesAtStart = state.learning?.games ?? null;
    activeCell = 9 * 19 + 9;
  } catch (error) {
    try { state = await request("/api/state"); } catch (_) { /* Keep the last known board. */ }
    showError(error);
  } finally {
    busy = false;
    render();
  }
}

newGameButton.addEventListener("click", startNewGame);

async function start() {
  buildBoard();
  render();
  try { state = await request("/api/state"); } catch (error) { showError(error); }
  busy = false;
  // Opened mid-game: this game's learning is still ahead. Opened after it ended: unknown.
  if (state && !state.game_over) gamesAtStart = state.learning?.games ?? null;
  render();
  scheduleNextGame();
}

start();
