"use strict";
// Gomoku rules + AlphaZero MCTS for the browser: a line-by-line port of state.py and mcts.py
// (same constants, same child order, same tie-breaking), so with the same network the search
// visits the same moves. The network is injected as `evaluate(planes) -> {logits, value}`.
// Checked against the Python search by tools/js/parity.mjs (python check_site_engine.py).

(function (root) {
  const SIZE = 19;
  const CELLS = SIZE * SIZE;
  // Must equal state.py COLOR_PLANE: plane 2 ("black to move") is all zeros while it is false.
  const COLOR_PLANE = false;

  // Flat indices of every line of 5 cells (rows, columns, both diagonals): 1,020 windows.
  const FIVE_WINDOWS = (() => {
    const windows = [];
    for (let r = 0; r < SIZE; r += 1) {
      for (let c = 0; c < SIZE; c += 1) {
        for (const [dr, dc] of [[0, 1], [1, 0], [1, 1], [1, -1]]) {
          const endR = r + 4 * dr;
          const endC = c + 4 * dc;
          if (endR >= 0 && endR < SIZE && endC >= 0 && endC < SIZE) {
            const window = [];
            for (let k = 0; k < 5; k += 1) window.push((r + k * dr) * SIZE + (c + k * dc));
            windows.push(window);
          }
        }
      }
    }
    return windows;
  })();

  class GomokuState {
    constructor() {
      this.size = SIZE;
      this.board = new Int8Array(CELLS);
      this.currentPlayer = 1;
      this.done = false;
      this.winner = 0;
    }

    clone() {
      const other = new GomokuState();
      other.board = this.board.slice();
      other.currentPlayer = this.currentPlayer;
      other.done = this.done;
      other.winner = this.winner;
      return other;
    }

    legalActions() {
      const actions = [];
      if (this.done) return actions;
      for (let i = 0; i < CELLS; i += 1) if (this.board[i] === 0) actions.push(i);
      return actions;
    }

    step(action) {
      if (this.done) throw new Error("Game is already over");
      if (!Number.isInteger(action) || action < 0 || action >= CELLS) throw new Error("Action outside board");
      if (this.board[action] !== 0) throw new Error("Position already occupied");
      const player = this.currentPlayer;
      this.board[action] = player;
      const row = Math.floor(action / SIZE);
      const col = action % SIZE;
      if (this._isWinner(row, col, player)) {
        this.done = true;
        this.winner = player;
      } else if (!this.board.includes(0)) {
        this.done = true;
        this.winner = 0;
      }
      this.currentPlayer = -player;
    }

    _isWinner(row, col, player) {
      for (const [dr, dc] of [[1, 0], [0, 1], [1, 1], [1, -1]]) {
        let count = 1;
        for (const direction of [-1, 1]) {
          let r = row + dr * direction;
          let c = col + dc * direction;
          while (r >= 0 && r < SIZE && c >= 0 && c < SIZE && this.board[r * SIZE + c] === player) {
            count += 1;
            r += dr * direction;
            c += dc * direction;
          }
        }
        if (count >= 5) return true;
      }
      return false;
    }

    // For each window holding 4 stones of `player` and 1 empty cell, call visit(emptyCell).
    _fours(player, visit) {
      for (const window of FIVE_WINDOWS) {
        let own = 0;
        let empty = -1;
        let empties = 0;
        for (const cell of window) {
          const value = this.board[cell];
          if (value === player) own += 1;
          else if (value === 0) { empties += 1; empty = cell; }
        }
        if (own === 4 && empties === 1 && visit(empty) === true) return true;
      }
      return false;
    }

    hasImmediateWin() {
      if (this.done) return false;
      return this._fours(this.currentPlayer, () => true);
    }

    winningMoves() {
      if (this.done) return [];
      const cells = new Set();
      this._fours(this.currentPlayer, (cell) => { cells.add(cell); });
      return [...cells].sort((a, b) => a - b);
    }

    // Unable to make five now, while the opponent already has 2+ different five-making cells.
    losesInTwo() {
      if (this.done || this.hasImmediateWin()) return false;
      const threats = new Set();
      return this._fours(-this.currentPlayer, (cell) => { threats.add(cell); return threats.size >= 2; });
    }

    terminalValue() {
      if (!this.done) throw new Error("Game is not over");
      return this.winner * this.currentPlayer;
    }

    // Network input (3, 19, 19): own stones, opponent stones, black-to-move plane.
    encode() {
      const planes = new Float32Array(3 * CELLS);
      const colorValue = COLOR_PLANE && this.currentPlayer === 1 ? 1 : 0;
      for (let i = 0; i < CELLS; i += 1) {
        const value = this.board[i];
        if (value === this.currentPlayer) planes[i] = 1;
        else if (value === -this.currentPlayer) planes[CELLS + i] = 1;
        planes[2 * CELLS + i] = colorValue;
      }
      return planes;
    }

    key() {
      return `${this.board.join("")}|${this.currentPlayer}|${this.done}|${this.winner}`;
    }
  }

  class TreeNode {
    constructor(prior = 1.0) {
      this.basePrior = prior;
      this.prior = prior;
      this.visitCount = 0;
      this.valueSum = 0;
      this.children = new Map(); // action -> TreeNode, in ascending action order
    }

    get q() {
      return this.visitCount === 0 ? 0 : this.valueSum / this.visitCount;
    }
  }

  // Softmax over legal moves only (network.masked_policy).
  function maskedPolicy(logits, legal) {
    let max = -Infinity;
    for (const action of legal) max = Math.max(max, logits[action]);
    const probabilities = new Map();
    let total = 0;
    for (const action of legal) {
      const p = Math.exp(logits[action] - max);
      probabilities.set(action, p);
      total += p;
    }
    for (const action of legal) probabilities.set(action, probabilities.get(action) / total);
    return probabilities;
  }

  class MCTS {
    constructor(evaluate, { simulations = 512, cPuct = 1.5, fpuReduction = 0.2, winCheck = true, yieldEvery = 16 } = {}) {
      this.evaluateNetwork = evaluate;
      this.simulations = simulations;
      this.cPuct = cPuct;
      this.fpuReduction = fpuReduction;
      this.winCheck = winCheck;
      this.yieldEvery = yieldEvery; // let the page repaint every N simulations (0 = never)
      this.reset();
    }

    reset() {
      this.root = null;
      this.rootKey = null;
      this.rootState = null;
      this.lastSearchStats = {};
    }

    // Keep the chosen subtree after a real move, if it really is the position we searched from.
    advance(action, state) {
      let child = null;
      if (this.root !== null && this.rootState !== null) {
        const expected = this.rootState.clone();
        try {
          expected.step(action);
          if (expected.key() === state.key()) child = this.root.children.get(action) || null;
        } catch (_) { /* illegal from the old root: drop the tree */ }
      }
      this.root = child;
      this.rootKey = child !== null ? state.key() : null;
      this.rootState = child !== null ? state.clone() : null;
    }

    // Search without root noise (play mode). Returns {policy: Float32Array(361) of visit shares, root}.
    async search(state, onProgress) {
      if (state.done) throw new Error("Cannot search a terminal state");
      const key = state.key();
      if (this.root === null || this.rootKey !== key) {
        this.root = new TreeNode();
        this.rootKey = key;
        this.rootState = state.clone();
      }
      const root = this.root;
      this.lastSearchStats = { reusedVisits: root.visitCount, networkCalls: 0, winInOne: 0, lossInTwo: 0 };
      if (root.children.size === 0) await this._evaluate(state, root);
      if (this.winCheck) {
        // A move that keeps an open four also scores +1, so make five now.
        const wins = new Set(state.winningMoves());
        if (wins.size) {
          for (const action of [...root.children.keys()]) if (!wins.has(action)) root.children.delete(action);
        }
      }
      for (const child of root.children.values()) child.prior = child.basePrior;

      for (let simulation = 0; simulation < this.simulations; simulation += 1) {
        const simulationState = state.clone();
        let node = root;
        const path = [node];
        while (node.children.size && !simulationState.done) {
          const [action, child] = this._select(node);
          simulationState.step(action);
          node = child;
          path.push(node);
        }
        let value;
        if (this.winCheck && simulationState.hasImmediateWin()) {
          value = 1.0; // left unexpanded on purpose: revisits re-check and get +1 again
          this.lastSearchStats.winInOne += 1;
        } else if (this.winCheck && simulationState.losesInTwo()) {
          value = -1.0;
          this.lastSearchStats.lossInTwo += 1;
        } else {
          value = await this._evaluate(simulationState, node);
        }
        this._backup(path, value);
        if (this.yieldEvery && (simulation + 1) % this.yieldEvery === 0) {
          if (onProgress) onProgress(simulation + 1, this.simulations);
          await new Promise((resolve) => setTimeout(resolve, 0));
        }
      }

      const policy = new Float32Array(CELLS);
      let total = 0;
      for (const [action, child] of root.children) { policy[action] = child.visitCount; total += child.visitCount; }
      for (let i = 0; i < CELLS; i += 1) policy[i] /= total;
      return { policy, root };
    }

    // PUCT with first-play-urgency reduction: an unvisited child is valued at the parent's value
    // minus fpuReduction * sqrt(prior mass already visited). First maximum wins.
    _select(node) {
      const exploration = this.cPuct * Math.sqrt(node.visitCount + 1);
      let visitedPrior = 0;
      for (const child of node.children.values()) if (child.visitCount) visitedPrior += child.prior;
      const fpuValue = Math.max(-1.0, node.q - this.fpuReduction * Math.sqrt(visitedPrior));
      let best = null;
      let bestScore = -Infinity;
      for (const item of node.children) {
        const child = item[1];
        const score = child.visitCount
          ? -child.q + exploration * child.prior / (1 + child.visitCount)
          : fpuValue + exploration * child.prior;
        if (best === null || score > bestScore) {
          best = item;
          bestScore = score;
        }
      }
      return best;
    }

    async _evaluate(state, node) {
      if (state.done) return state.terminalValue();
      const { logits, value } = await this.evaluateNetwork(state.encode());
      this.lastSearchStats.networkCalls += 1;
      const legal = state.legalActions();
      const probabilities = maskedPolicy(logits, legal);
      for (const action of legal) node.children.set(action, new TreeNode(probabilities.get(action)));
      return value;
    }

    _backup(path, value) {
      for (let i = path.length - 1; i >= 0; i -= 1) {
        path[i].visitCount += 1;
        path[i].valueSum += value;
        value = -value;
      }
    }
  }

  // Most-visited move; ties broken at random (self_play.select_action with sample=False).
  function mostVisited(policy, random = Math.random) {
    let max = -Infinity;
    for (const p of policy) max = Math.max(max, p);
    const best = [];
    policy.forEach((p, action) => { if (p === max) best.push(action); });
    return best[Math.floor(random() * best.length)];
  }

  const api = { SIZE, CELLS, COLOR_PLANE, FIVE_WINDOWS, GomokuState, TreeNode, MCTS, maskedPolicy, mostVisited };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.GomokuEngine = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
