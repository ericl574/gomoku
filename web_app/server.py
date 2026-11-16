"""Start with: python -m web_app.server (requires the existing ml environment)."""
import argparse
from datetime import datetime
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
import state as game_rules
from train import ReplayBuffer, find_latest_checkpoint, save_checkpoint, train_step
from human_games import CUTOFF_FILE, load_human_games, normalized_web_game, read_cutoff  # noqa: F401
from promote import read_best


APP_DIR = Path(__file__).resolve().parent
CHECKPOINT_ROOT = APP_DIR.parent / "checkpoints"
GAME_LOG_DIR = APP_DIR.parent / "game_logs"
DEFAULT_SIMULATIONS = 512


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
        "games": checkpoint.get("games_completed"),  # absent in old checkpoints
        "updates": checkpoint.get("updates_completed"),
        "architecture": checkpoint["model_config"].get("architecture", "cnn"),
    }
    return model, info


def best_checkpoint(root=CHECKPOINT_ROOT):
    """The model named in <root>/best.json (set only by promote.py), or None."""
    record = read_best(root)
    return Path(record["path"]) if record else None


class CheckpointWatcher:
    """Decides which checkpoint should play.

    pinned file > best (best.json, if it exists) > latest (newest .pt file).
    use_best=False skips best and always follows the latest file.
    """

    def __init__(self, root=CHECKPOINT_ROOT, pinned=None, use_best=True):
        self.root = Path(root)
        self.pinned = Path(pinned) if pinned is not None else None
        self.use_best = use_best

    def current(self):
        path = self.pinned
        if path is None and self.use_best:
            path = best_checkpoint(self.root)
        if path is None:
            path = find_latest_checkpoint(self.root)
        return path, path.stat().st_mtime_ns

    def describe(self):
        if self.pinned is not None:
            return "pinned with --checkpoint"
        record = read_best(self.root) if self.use_best else None
        if record:
            return (f"BEST ({record['status']}; checked again at each new game; "
                    f"use --latest for the newest checkpoint instead)")
        return "LATEST checkpoint (no best.json yet), checked again at each new game"


class WebLearner:
    """Learning mode: the AI trains on every finished game it plays against you.

    Exactly the self-play learning rule, only the opponent is you:
      input  = the board before each AI move
      policy = the AI's own MCTS visit counts for that move
      value  = the final result from the AI's side (+1 win, -1 loss, 0 draw)
    By default your moves are never used as targets, so it does not copy you.

    With imitate_weight > 0 (--imitate), each of YOUR moves is also an example:
      input  = the board before your move, seen from your side
      policy = your move, scaled to imitate_weight (a target summing to w gives
               exactly w times the gradient of a normal example)
      value  = the final result from your side
    Only the "which move" part copies you; the value part is still win/loss.
    """

    def __init__(self, model, source_path, simulations, root=CHECKPOINT_ROOT,
                 updates=10, batch_size=32, learning_rate=1e-3,
                 buffer_capacity=10000, seed=None, imitate_weight=0.0, fresh=False):
        if not 0.0 <= imitate_weight <= 1.0:
            raise ValueError("Imitation weight must be between 0 and 1")
        self.imitate_weight = float(imitate_weight)
        source_path = Path(source_path)
        checkpoint = torch.load(source_path, map_location="cpu", weights_only=True)
        self.model = model.requires_grad_(True)
        self.model_config = checkpoint["model_config"]
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        if "optimizer_state_dict" in checkpoint:
            # Continue the same Adam moments the checkpoint was trained with.
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate
        self.updates_completed = int(checkpoint.get("updates_completed") or 0)
        self.source = f"{source_path.parent.name}/{source_path.name}"
        self.simulations = simulations
        self.root = Path(root)
        self.updates = updates
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)
        self.augmentation_rng = np.random.default_rng(None if seed is None else seed + 1)
        self.buffer = ReplayBuffer(buffer_capacity)
        # Games from earlier learning sessions stay in the replay buffer, except
        # those before the cutoff that --fresh records (run names start with
        # their start time, so they sort in time order).
        if fresh:
            self.root.mkdir(parents=True, exist_ok=True)
            (self.root / CUTOFF_FILE).write_text(f"{datetime.now():%Y-%m-%d_%H%M}\n")
        self.cutoff = read_cutoff(self.root)
        samples, _, self.skipped_games = load_human_games(self.root)
        self.buffer.add_game(samples)
        self.run_dir = None  # created when the first game finishes
        self.games = 0
        self.last = None

    def _make_run_dir(self):
        name = f"{datetime.now():%Y-%m-%d_%H%M}_web_sims{self.simulations}"
        run_dir, suffix = self.root / name, 2
        while run_dir.exists():
            run_dir, suffix = self.root / f"{name}-{suffix}", suffix + 1
        run_dir.mkdir(parents=True)
        return run_dir

    def learn(self, records, winner, ai_player=1):
        """records: (board encoding, policy target, player who moved) per example.

        ai_player is the AI's color in this game (1 black, -1 white).
        """
        samples = [(obs, policy, float(winner * player)) for obs, policy, player in records]
        if not samples:
            return None
        if self.run_dir is None:
            self.run_dir = self._make_run_dir()
        self.games += 1
        np.savez_compressed(
            self.run_dir / f"game_{self.games:04d}.npz",
            observations=np.stack([s[0] for s in samples]),
            policies=np.stack([s[1] for s in samples]),
            outcomes=np.array([s[2] for s in samples], dtype=np.float32),
        )
        self.buffer.add_game(samples)

        batch_size = min(self.batch_size, len(self.buffer))
        before = [p.detach().clone() for p in self.model.parameters()]
        metrics = [
            train_step(self.model, self.optimizer,
                       self.buffer.sample(batch_size, self.rng, self.augmentation_rng))
            for _ in range(self.updates)
        ]
        self.model.eval()
        # Proof the weights really changed: L2 distance between old and new weights.
        weight_change = math.sqrt(sum(
            float(((p.detach() - b) ** 2).sum()) for p, b in zip(self.model.parameters(), before)))
        self.updates_completed += self.updates
        averages = {k: float(np.mean([m[k] for m in metrics])) for k in metrics[0]}

        path = self.run_dir / f"round_{self.games:03d}_games{self.games:04d}.pt"
        save_checkpoint({
            "iteration": self.games,
            "games_completed": self.games,
            "updates_completed": self.updates_completed,
            "model_config": self.model_config,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": averages,
            "evaluation": None,
            "simulations": self.simulations,
            "runtime_config": {"mode": "web_learning", "started_from": self.source,
                               "color_plane": game_rules.COLOR_PLANE},
        }, path)
        result = {1: "AI won", -1: "AI lost", 0: "draw"}[int(winner * ai_player)]
        your_moves = sum(player != ai_player for _, _, player in records)
        result += " as " + ("black" if ai_player == 1 else "white")
        self.last = {"game": self.games, "result": result, "positions": len(samples),
                     "your_moves": your_moves,
                     "buffer": len(self.buffer), "weight_change": weight_change, **averages}
        with (self.run_dir / "history.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps({"game": self.games, **self.last}) + "\n")
        print(f"Learned from game {self.games} ({result}, {len(samples)} positions): "
              f"policy {averages['policy_loss']:.3f}, value {averages['value_loss']:.3f}, "
              f"weights moved {weight_change:.4f} "
              f"-> {path.parent.name}/{path.name}", flush=True)
        return path


class GameConflict(ValueError):
    """A stale request, a wrong turn, or a move after game over."""


class GomokuGame:
    """One shared game against an AI that moves with MCTS.

    Both modes: no root noise (the human opponent supplies the variety).
    Normal mode: the AI is always black.
    Learning mode: every finished game is
    handed to the WebLearner, and the AI's color alternates each game
    (black, white, black, ...) so both colors win and lose. Otherwise the
    value head can just learn "the side to move loses" from the turn plane.
    """

    def __init__(self, model=None, info=None, simulations=DEFAULT_SIMULATIONS,
                 watcher=None, seed=None, learner=None, log_dir=None):
        if model is None and watcher is None:
            raise ValueError("Give a model or a checkpoint watcher")
        if learner is not None and watcher is not None:
            raise ValueError("Learning mode keeps its own model; don't auto-swap checkpoints")
        # Game log: every finished game, every move, and what the AI's search saw.
        self.log_path = (Path(log_dir) / f"{datetime.now():%Y-%m-%d_%H%M%S}.txt"
                         if log_dir is not None else None)
        self.logged_games = 0
        self.ai_details = {}  # move index -> what the search saw for that AI move
        self.simulations = simulations
        self.watcher = watcher
        self.learner = learner
        self.alternate_colors = learner is not None
        self.ai_player = -1 if self.alternate_colors else 1  # first new_game flips to black
        self.records = []
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
        ai_color = "black" if self.ai_player == 1 else "white"
        return {
            "ai_color": ai_color,
            "human_color": "white" if ai_color == "black" else "black",
            "size": state.size,
            "board": state.board.tolist(),
            "turn": None if state.done else ("black" if state.current_player == 1 else "white"),
            "game_over": bool(state.done),
            "winner": winner,
            "winning_line": self.winning_line,
            "last_move": self.moves[-1] if self.moves else None,
            "last_ai_move": next((m for m in reversed(self.moves) if m["player"] == ai_color), None),
            "move_count": len(self.moves),
            "revision": self.revision,
            "model": {**self.info, "simulations": self.simulations},
            "learning": ({"enabled": True, "games": self.learner.games,
                          "imitate_weight": self.learner.imitate_weight,
                          "positions": len(self.learner.buffer), "last": self.learner.last}
                         if self.learner is not None else {"enabled": False}),
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
        learning = self.learner is not None
        # No Dirichlet noise, even when learning: self-play needs it to vary its
        # games, but against you the variety comes from your play. With a flat
        # prior, the noise alone was choosing moves (e.g. random edge openings).
        policy, root = self.searcher.search(self.game_state, add_noise=False)
        priors_ok = all(math.isfinite(child.prior) for child in root.children.values())
        if not (priors_ok and math.isfinite(root.value_sum) and np.isfinite(policy).all()):
            raise FloatingPointError("The network returned a nonfinite policy or value")
        if learning:
            # Same training example self-play records: board, search policy, who moved.
            state = self.game_state
            self.records.append((state.encode(), policy.copy(), state.current_player))
        # Most-visited move; ties broken at random (same rule as evaluate.py).
        self._place(select_action(policy, self.rng, sample=False))
        # For the game log. q is from the AI's side: -child.q, as in MCTS._select.
        top = sorted(root.children.items(), key=lambda kv: kv[1].visit_count, reverse=True)[:5]
        self.ai_details[len(self.moves) - 1] = {
            "value": root.q,
            "visited": sum(child.visit_count > 0 for child in root.children.values()),
            "candidates": len(root.children),
            "top": [(action, child.visit_count, child.base_prior, -child.q)
                    for action, child in top],
        }
        self._print_move(len(self.moves) - 1)

    @staticmethod
    def _coord(row, col):
        return f"{chr(65 + col)}{row + 1}"  # same labels as the web board (A-S, 1-19)

    def _move_line(self, index):
        move = self.moves[index]
        who = "AI " if move["player"] == ("black" if self.ai_player == 1 else "white") else "You"
        line = f"{index + 1:3d}. {move['player'][0].upper()} {who} {self._coord(move['row'], move['col']):>4}"
        details = self.ai_details.get(index)
        if details:
            top = ", ".join(
                f"{self._coord(*divmod(a, 19))} n={n} p={p:.3f} q={q:+.2f}"
                for a, n, p, q in details["top"])
            line += (f"   value {details['value']:+.2f}, searched {details['visited']}/"
                     f"{details['candidates']} moves | top: {top}")
        return line

    def _print_move(self, index):
        if self.log_path is not None:
            print(self._move_line(index), flush=True)

    def _write_game_log(self):
        """Append the finished game to the session log file. Never breaks the game."""
        if self.log_path is None:
            return
        try:
            state = self.game_state
            self.logged_games += 1
            ai = "black" if self.ai_player == 1 else "white"
            winner = {1: "black", -1: "white", 0: "draw"}[int(state.winner)]
            outcome = "draw" if winner == "draw" else ("AI won" if winner == ai else "You won")
            lines = [
                f"=== Game {self.logged_games} | {datetime.now():%Y-%m-%d %H:%M:%S} | AI {ai} | "
                f"{outcome} ({winner}) in {len(self.moves)} moves",
                f"model {self.info.get('run')}/{self.info.get('file')}, "
                f"{self.simulations} simulations"
                + (f", learning, imitate {self.learner.imitate_weight}" if self.learner else ""),
                "AI lines: value = AI's search estimate (+1 AI winning, -1 losing); "
                "top = most-visited moves: n visits, p network prior, q AI's estimate",
            ]
            lines += [self._move_line(i) for i in range(len(self.moves))]
            if self.learner is not None and self.learner.last is not None:
                last = self.learner.last
                lines.append(f"learned: {last['result']}, {last['positions']} positions, "
                             f"policy {last['policy_loss']:.3f}, value {last['value_loss']:.3f}, "
                             f"weights moved {last['weight_change']:.4f}")
            lines.append("final board (X black, O white, lowercase = last move):")
            lines.append("    " + " ".join(chr(65 + c) for c in range(state.size)))
            last_move = self.moves[-1] if self.moves else None
            for r in range(state.size):
                cells = []
                for c in range(state.size):
                    mark = {1: "X", -1: "O", 0: "."}[int(state.board[r, c])]
                    if last_move and (last_move["row"], last_move["col"]) == (r, c):
                        mark = mark.lower()
                    cells.append(mark)
                lines.append(f"{r + 1:3d} " + " ".join(cells))
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as file:
                file.write("\n".join(lines) + "\n\n")
            print(f"--- {outcome} ({winner}) in {len(self.moves)} moves. "
                  f"Logged as game {self.logged_games} in {self.log_path}", flush=True)
        except Exception as error:  # noqa: BLE001
            print(f"Writing the game log failed: {error}", file=sys.stderr, flush=True)

    def _learn_from_finished_game(self):
        """Train on the game that just ended. A failure never undoes the game."""
        try:
            path = self.learner.learn(self.records, int(self.game_state.winner), self.ai_player)
        except Exception as error:  # noqa: BLE001
            print(f"Learning from this game failed: {error}", file=sys.stderr, flush=True)
            return
        self.records = []
        self.searcher.reset()  # the weights changed; old search statistics are stale
        if path is not None:
            self.info = {**self.info, "run": path.parent.name, "file": path.name,
                         "iteration": self.learner.games, "games": self.learner.games,
                         "updates": self.learner.updates_completed}

    def _save(self):
        return (self.game_state.clone(), list(self.moves), list(self.winning_line),
                list(self.records), dict(self.ai_details))

    def _restore(self, saved):
        (self.game_state, self.moves, self.winning_line,
         self.records, self.ai_details) = saved
        self.searcher.reset()

    def new_game(self):
        with self.lock:
            previous = self._save() if hasattr(self, "game_state") else None
            old_model = (self.model, self.info, self.model_key,
                         getattr(self, "searcher", None))
            old_ai_player = self.ai_player
            try:
                self._refresh_model()
                if self.alternate_colors:
                    self.ai_player = -self.ai_player
                self.game_state = GomokuState()
                self.moves = []
                self.winning_line = []
                self.records = []  # an unfinished game has no result, so it is not learned
                self.ai_details = {}
                self.searcher.reset()
                if self.log_path is not None:
                    print(f"=== New game: AI plays {'black' if self.ai_player == 1 else 'white'}",
                          flush=True)
                if self.ai_player == 1:  # black always moves first
                    self._ai_move()
            except Exception:
                self.ai_player = old_ai_player
                if previous is not None:
                    self.model, self.info, self.model_key, self.searcher = old_model
                    self._restore(previous)
                raise
            self.revision += 1
            return self._snapshot()

    def play_human(self, row, col, revision):
        """Your move (white, or black in learning-mode games where the AI is white)."""
        with self.lock:
            state = self.game_state
            if any(type(value) is not int for value in (row, col, revision)):
                raise ValueError("Row, column, and revision must be integers")
            if revision != self.revision:
                raise GameConflict("The board changed. Please try your move again.")
            if state.done:
                raise GameConflict("This game is over. Start a new game.")
            if state.current_player != -self.ai_player:
                raise GameConflict("Please wait for your turn.")
            if not (0 <= row < state.size and 0 <= col < state.size):
                raise ValueError("Choose an intersection inside the board")
            if state.board[row, col] != 0:
                raise ValueError("That intersection is already occupied")

            # Roll back your move if the search fails, so it can be retried.
            saved = self._save()
            try:
                action = row * state.size + col
                if self.learner is not None and self.learner.imitate_weight > 0:
                    # Your move as an example, from your side of the board.
                    target = np.zeros(state.size * state.size, dtype=np.float32)
                    target[action] = self.learner.imitate_weight
                    self.records.append((state.encode(), target, state.current_player))
                self._place(action)
                self._print_move(len(self.moves) - 1)
                if not self.game_state.done:
                    self._ai_move()
            except Exception:
                self._restore(saved)
                raise
            if self.game_state.done and self.learner is not None:
                self._learn_from_finished_game()
            if self.game_state.done:
                self._write_game_log()
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
                    state = game.play_human(payload.get("row"), payload.get("col"), payload.get("revision"))
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
                        help="Pin one .pt file. Default: the best model (checkpoints/best.json), "
                             "or the newest checkpoint if no best exists yet")
    parser.add_argument("--latest", action="store_true",
                        help="Use the newest checkpoint instead of the best model")
    parser.add_argument("--learn", action="store_true",
                        help="Learning mode: continue from the newest checkpoint and train "
                             "on every finished game against you")
    parser.add_argument("--learn-updates", type=int, default=10,
                        help="Gradient updates after each finished game (default %(default)s)")
    parser.add_argument("--imitate", type=float, nargs="?", const=0.5, default=0.0,
                        metavar="WEIGHT",
                        help="With --learn: also learn from YOUR moves, weighted "
                             "(default 0.5 when given; 0 = off)")
    parser.add_argument("--fresh", action="store_true",
                        help="With --learn: keep the weights, but forget games learned before "
                             "now (later restarts skip them too)")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    if args.simulations < 1:
        parser.error("simulations must be positive")
    # Batch-1 inference is no faster with more threads (measured); leave CPU to training.
    torch.set_num_threads(1)
    if args.learn_updates < 1:
        parser.error("learn-updates must be positive")
    if args.imitate and not args.learn:
        parser.error("--imitate only works together with --learn")
    if args.fresh and not args.learn:
        parser.error("--fresh only works together with --learn")
    if not 0.0 <= args.imitate <= 1.0:
        parser.error("--imitate weight must be between 0 and 1")
    try:
        if args.learn:
            # Start from one checkpoint, then keep training that same model.
            # Learning continues the newest training (self-play or web) so the two
            # modes can alternate: --checkpoint > latest. Never starts from best.
            path, _ = CheckpointWatcher(pinned=args.checkpoint, use_best=False).current()
            model, info = load_model(path)
            learner = WebLearner(model, path, args.simulations, updates=args.learn_updates,
                                 imitate_weight=args.imitate, fresh=args.fresh)
            game = GomokuGame(model, info, simulations=args.simulations, learner=learner,
                              log_dir=GAME_LOG_DIR)
        else:
            watcher = CheckpointWatcher(pinned=args.checkpoint, use_best=not args.latest)
            game = GomokuGame(simulations=args.simulations, watcher=watcher,
                              log_dir=GAME_LOG_DIR)
        server = make_server(game, args.port)
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        parser.error(str(error))
    if args.learn:
        copies = (f"also learns from YOUR moves (weight {args.imitate})"
                  if args.imitate else "learns from its own moves only")
        mode = (f"LEARNING: {copies}; trains after every finished game, "
                f"{args.learn_updates} updates each; "
                f"{len(learner.buffer)} positions from earlier sessions loaded"
                + (f", {learner.skipped_games} games from before {learner.cutoff} skipped"
                   if learner.cutoff else "")
                + ". Its saves count as LATEST only; only promote.py can change BEST")
    else:
        mode = watcher.describe()
    print(f"Gomoku is ready: http://127.0.0.1:{server.server_port}", flush=True)
    print(f"AI: {game.info['run']}/{game.info['file']} ({mode}), "
          f"{args.simulations} MCTS simulations per move, CPU.", flush=True)
    print(f"Every finished game is logged to {game.log_path}", flush=True)
    print("Press Ctrl+C to stop.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nGood game!")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
