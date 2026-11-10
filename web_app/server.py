"""Start with: python -m web_app.server (requires the existing ml environment)."""
import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import sys
import threading
from urllib.parse import urlsplit

import numpy as np
import torch

from mcts import MCTS
from network import GomokuNetwork
from self_play import select_action
from state import GomokuState


APP_DIR = Path(__file__).resolve().parent
CHECKPOINT_ROOT = APP_DIR.parent / "checkpoints"
DEFAULT_SIMULATIONS = 512


def find_latest_checkpoint(root=CHECKPOINT_ROOT):
    """Newest iteration_NNN.pt by write time across every training run.

    train.save_checkpoint writes `*.pt.tmp` first and renames it, so a file
    matching this pattern is always complete.
    """
    candidates = []
    for path in Path(root).glob("*/iteration_*.pt"):
        try:
            candidates.append((path.stat().st_mtime_ns, str(path), path))
        except OSError:
            continue
    if not candidates:
        raise FileNotFoundError(f"No iteration_*.pt checkpoints under {root}")
    return max(candidates)[2]


def load_model(path):
    """Strictly load one AlphaZero checkpoint on CPU for inference only."""
    path = Path(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = GomokuNetwork(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.cpu().eval().requires_grad_(False)
    info = {
        "run": path.parent.name,
        "file": path.name,
        "iteration": checkpoint.get("iteration"),
        "updates": checkpoint.get("updates_completed"),
        "architecture": checkpoint["model_config"].get("architecture", "cnn"),
    }
    return model, info


class CheckpointWatcher:
    """Decides which checkpoint should play: the newest one, or a pinned file."""

    def __init__(self, root=CHECKPOINT_ROOT, pinned=None):
        self.root = Path(root)
        self.pinned = Path(pinned) if pinned is not None else None

    def current(self):
        path = self.pinned if self.pinned is not None else find_latest_checkpoint(self.root)
        return path, path.stat().st_mtime_ns


class GameConflict(ValueError):
    """A stale request, a wrong turn, or a move after game over."""


class GomokuGame:
    """One shared game. The AI is black and moves with MCTS, no root noise."""

    def __init__(self, model=None, info=None, simulations=DEFAULT_SIMULATIONS,
                 watcher=None, seed=None):
        if model is None and watcher is None:
            raise ValueError("Give a model or a checkpoint watcher")
        self.simulations = simulations
        self.watcher = watcher
        self.rng = np.random.default_rng(seed)
        self.lock = threading.Lock()
        self.revision = 0
        self.model_key = None
        self.model = None
        self.info = None
        if model is not None:
            self._use_model(model, info or {})
        self.new_game()

    # ---- model management -------------------------------------------------

    def _use_model(self, model, info):
        self.model = model.cpu().eval()
        self.info = dict(info)
        self.searcher = MCTS(self.model, simulations=self.simulations)

    def _refresh_model(self):
        """Swap in a newer checkpoint. Only called between games."""
        if self.watcher is None:
            return
        try:
            key = self.watcher.current()
            if key == self.model_key:
                return
            model, info = load_model(key[0])
        except Exception as error:  # noqa: BLE001 - keep playing with the old model
            if self.model is None:
                raise
            print(f"Checkpoint reload failed, keeping {self.info.get('file')}: {error}",
                  file=sys.stderr, flush=True)
            return
        self._use_model(model, info)
        self.model_key = key
        print(f"Loaded {info['run']}/{info['file']}", flush=True)

    # ---- game state -------------------------------------------------------

    def _snapshot(self):
        state = self.game_state
        winner = None
        if state.done:
            winner = {1: "black", -1: "white", 0: "draw"}[int(state.winner)]
        return {
            "size": state.size,
            "board": state.board.tolist(),
            "turn": None if state.done else ("black" if state.current_player == 1 else "white"),
            "game_over": bool(state.done),
            "winner": winner,
            "winning_line": self.winning_line,
            "last_move": self.moves[-1] if self.moves else None,
            "last_ai_move": next((m for m in reversed(self.moves) if m["player"] == "black"), None),
            "move_count": len(self.moves),
            "revision": self.revision,
            "model": {**self.info, "simulations": self.simulations},
        }

    def state(self):
        with self.lock:
            return self._snapshot()

    def _winning_stones(self, row, col, player):
        # Display only. GomokuState.step decides whether someone won.
        board, size = self.game_state.board, self.game_state.size
        for dr, dc in ((1, 0), (0, 1), (1, 1), (1, -1)):
            line = [[row, col]]
            for direction in (-1, 1):
                r, c = row + dr * direction, col + dc * direction
                while 0 <= r < size and 0 <= c < size and board[r, c] == player:
                    line.append([r, c])
                    r += dr * direction
                    c += dc * direction
            if len(line) >= 5:
                return line
        return []

    def _place(self, action):
        state = self.game_state
        player = state.current_player
        state.step(action)
        # Keep the searched subtree for the next AI move (checked inside advance).
        self.searcher.advance(action, state)
        row, col = divmod(int(action), state.size)
        self.moves.append({"row": row, "col": col, "player": "black" if player == 1 else "white"})
        if state.done and state.winner != 0:
            self.winning_line = self._winning_stones(row, col, player)

    def _ai_move(self):
        policy, root = self.searcher.search(self.game_state, add_noise=False)
        priors_ok = all(math.isfinite(child.prior) for child in root.children.values())
        if not (priors_ok and math.isfinite(root.value_sum) and np.isfinite(policy).all()):
            raise FloatingPointError("The network returned a nonfinite policy or value")
        # Most-visited move; ties broken at random (same rule as evaluate.py).
        self._place(select_action(policy, self.rng, sample=False))

    def _save(self):
        return (self.game_state.clone(), list(self.moves), list(self.winning_line))

    def _restore(self, saved):
        self.game_state, self.moves, self.winning_line = saved
        self.searcher.reset()

    def new_game(self):
        with self.lock:
            previous = self._save() if hasattr(self, "game_state") else None
            old_model = (self.model, self.info, self.model_key,
                         getattr(self, "searcher", None))
            try:
                self._refresh_model()
                self.game_state = GomokuState()
                self.moves = []
                self.winning_line = []
                self.searcher.reset()
                self._ai_move()
            except Exception:
                if previous is not None:
                    self.model, self.info, self.model_key, self.searcher = old_model
                    self._restore(previous)
                raise
            self.revision += 1
            return self._snapshot()

    def play_white(self, row, col, revision):
        with self.lock:
            state = self.game_state
            if any(type(value) is not int for value in (row, col, revision)):
                raise ValueError("Row, column, and revision must be integers")
            if revision != self.revision:
                raise GameConflict("The board changed. Please try your move again.")
            if state.done:
                raise GameConflict("This game is over. Start a new game.")
            if state.current_player != -1:
                raise GameConflict("Please wait for your turn.")
            if not (0 <= row < state.size and 0 <= col < state.size):
                raise ValueError("Choose an intersection inside the board")
            if state.board[row, col] != 0:
                raise ValueError("That intersection is already occupied")

            # Roll back white's move if the search fails, so it can be retried.
            saved = self._save()
            try:
                self._place(row * state.size + col)
                if not self.game_state.done:
                    self._ai_move()
            except Exception:
                self._restore(saved)
                raise
            self.revision += 1
            return self._snapshot()


def make_server(game, port=8000):
    # Serve only this small allowlist; never expose project files/checkpoints.
    assets = {
        "/": ("index.html", "text/html; charset=utf-8"),
        "/style.css": ("style.css", "text/css; charset=utf-8"),
        "/app.js": ("app.js", "text/javascript; charset=utf-8"),
        "/favicon.svg": ("favicon.svg", "image/svg+xml"),
    }

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass  # keep the terminal quiet; model reloads are still printed

        def _send(self, status, data, content_type="application/json; charset=utf-8"):
            body = json.dumps(data).encode("utf-8") if not isinstance(data, bytes) else data
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = urlsplit(self.path).path
            if path == "/api/state":
                self._send(HTTPStatus.OK, game.state())
            elif path in assets:
                filename, content_type = assets[path]
                self._send(HTTPStatus.OK, (APP_DIR / filename).read_bytes(), content_type)
            else:
                self._send(HTTPStatus.NOT_FOUND, {"error": "Not found"})

        def do_POST(self):
            path = urlsplit(self.path).path
            if path not in ("/api/new", "/api/move"):
                self._send(HTTPStatus.NOT_FOUND, {"error": "Not found"})
                return
            if self.headers.get_content_type() != "application/json":
                self._send(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, {"error": "Send application/json"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 4096:
                    raise ValueError("Invalid request size")
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError("Expected a JSON object")
                if path == "/api/new":
                    state = game.new_game()
                else:
                    state = game.play_white(payload.get("row"), payload.get("col"), payload.get("revision"))
                self._send(HTTPStatus.OK, state)
            except GameConflict as error:
                self._send(HTTPStatus.CONFLICT, {"error": str(error)})
            except (ValueError, UnicodeDecodeError) as error:
                self._send(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            except (RuntimeError, FloatingPointError):
                self._send(HTTPStatus.SERVICE_UNAVAILABLE, {"error": "AI inference failed. Please try again."})

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main():
    parser = argparse.ArgumentParser(description="Play your AlphaZero Gomoku AI in a local browser.")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--simulations", type=int, default=DEFAULT_SIMULATIONS,
                        help="MCTS simulations per AI move (default %(default)s)")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Pin one .pt file. Default: newest checkpoints/*/iteration_*.pt, "
                             "re-checked at every new game")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    if args.simulations < 1:
        parser.error("simulations must be positive")
    # Batch-1 inference is no faster with more threads (measured); leave CPU to training.
    torch.set_num_threads(1)
    try:
        watcher = CheckpointWatcher(pinned=args.checkpoint)
        game = GomokuGame(simulations=args.simulations, watcher=watcher)
        server = make_server(game, args.port)
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        parser.error(str(error))
    mode = "pinned" if args.checkpoint else "auto-updates to the newest checkpoint at each new game"
    print(f"Gomoku is ready: http://127.0.0.1:{server.server_port}", flush=True)
    print(f"AI: {game.info['run']}/{game.info['file']} ({mode}), "
          f"{args.simulations} MCTS simulations per move, CPU.", flush=True)
    print("AI is black; you are white. Press Ctrl+C to stop.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nGood game!")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
