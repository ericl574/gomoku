"""Web game and HTTP checks for the AlphaZero web app. No training; real checkpoints are only read."""
import hashlib
from http.client import HTTPConnection
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest

import numpy as np
import torch
from torch import nn

from network import GomokuNetwork
from state import GomokuState
from web_app.server import (
    CHECKPOINT_ROOT, CheckpointWatcher, GameConflict, GomokuGame,
    find_latest_checkpoint, load_model, make_server,
)

SIMS = 8


class FixedPrior(nn.Module):
    """Policy strongly prefers low action indices; value is always 0."""

    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(-2.0 * torch.arange(361, dtype=torch.float32))
        self.value = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        if torch.is_grad_enabled() or self.training:
            raise AssertionError("Game inference must disable gradients and use eval mode")
        batch = x.shape[0]
        return self.logits.expand(batch, -1), torch.tanh(self.value).expand(batch, 1)


def no_five_board():
    """A full 19x19 board with no five in a row (181 black, 180 white)."""
    return np.array([[1 if ((c // 2) + r) % 2 == 0 else -1 for c in range(19)]
                     for r in range(19)], dtype=np.int8)


def save_fake_checkpoint(path, seed, iteration):
    torch.manual_seed(seed)
    config = {"channels": 8, "value_hidden": 8, "policy_kernel": 3,
              "architecture": "resnet_global", "residual_blocks": 1}
    model = GomokuNetwork(**config)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"iteration": iteration, "updates_completed": iteration * 10,
                "model_config": config, "model_state_dict": model.state_dict()}, path)
    return model


class GameTests(unittest.TestCase):
    def setUp(self):
        self.model = FixedPrior()
        self.game = GomokuGame(self.model, {"run": "fake", "file": "x.pt"}, simulations=SIMS, seed=0)

    def test_ai_opens_as_black_then_white_gets_the_turn(self):
        state = self.game.state()
        self.assertEqual(state["board"][0][0], 1)
        self.assertEqual(state["move_count"], 1)
        self.assertEqual(state["turn"], "white")
        self.assertFalse(state["game_over"])
        self.assertEqual(state["model"]["simulations"], SIMS)
        self.assertIsNone(self.model.logits.grad)

    def test_white_move_gets_a_legal_ai_reply_and_weights_do_not_change(self):
        before = self.model.logits.detach().clone()
        state = self.game.play_white(0, 1, self.game.revision)
        self.assertEqual(state["board"][0][:3], [1, -1, 1])
        self.assertEqual(state["turn"], "white")
        self.assertEqual(state["move_count"], 3)
        torch.testing.assert_close(self.model.logits, before)

    def test_invalid_and_occupied_moves_do_not_change_the_game(self):
        before = self.game.state()
        for row, col in ((0, 0), (-1, 0), (19, 0), (0, 19), (True, 1), (0.5, 1), ("0", 1)):
            with self.subTest(row=row, col=col), self.assertRaises(ValueError):
                self.game.play_white(row, col, before["revision"])
            self.assertEqual(self.game.state(), before)

    def test_stale_revision_rejects_duplicate_requests(self):
        old_revision = self.game.revision
        self.game.play_white(0, 1, old_revision)
        before = self.game.state()
        with self.assertRaises(GameConflict):
            self.game.play_white(0, 3, old_revision)
        self.assertEqual(self.game.state(), before)

    def test_wrong_turn_is_rejected(self):
        self.game.game_state.current_player = 1
        with self.assertRaises(GameConflict):
            self.game.play_white(0, 1, self.game.revision)

    def test_white_wins_in_each_direction_and_no_ai_reply_follows(self):
        for dr, dc in ((1, 0), (0, 1), (1, 1), (1, -1)):
            with self.subTest(direction=(dr, dc)):
                self.game.new_game()
                self.game.game_state.board.fill(0)
                self.game.game_state.current_player = -1
                for offset in range(4):
                    self.game.game_state.board[5 + dr * offset, 9 + dc * offset] = -1
                state = self.game.play_white(5 + dr * 4, 9 + dc * 4, self.game.revision)
                self.assertEqual(state["winner"], "white")
                self.assertTrue(state["game_over"])
                self.assertIsNone(state["turn"])
                self.assertEqual(len(state["winning_line"]), 5)
                self.assertEqual(state["last_move"]["player"], "white")
                with self.assertRaises(GameConflict):
                    self.game.play_white(0, 0, self.game.revision)

    def test_ai_win_is_reported_as_black(self):
        board = self.game.game_state.board
        board.fill(0)
        board[0, 1:5] = 1
        state = self.game.play_white(2, 2, self.game.revision)
        self.assertEqual(state["winner"], "black")
        self.assertTrue(state["game_over"])
        self.assertEqual(len(state["winning_line"]), 5)
        self.assertEqual(state["last_move"], {"row": 0, "col": 0, "player": "black"})

    def test_full_board_draw_after_white(self):
        board = no_five_board()
        board[0, 2] = 0  # a white square
        self.assertEqual(no_five_board()[0, 2], -1)
        self.game.game_state.board[:] = board
        self.game.game_state.current_player = -1
        state = self.game.play_white(0, 2, self.game.revision)
        self.assertEqual(state["winner"], "draw")
        self.assertTrue(state["game_over"])
        self.assertEqual(state["winning_line"], [])

    def test_full_board_draw_after_black(self):
        board = no_five_board()
        board[0, 0] = 0  # black square, AI fills it last
        board[0, 2] = 0  # white square
        self.game.game_state.board[:] = board
        self.game.game_state.current_player = -1
        state = self.game.play_white(0, 2, self.game.revision)
        self.assertEqual(state["winner"], "draw")
        self.assertEqual(state["last_move"], {"row": 0, "col": 0, "player": "black"})

    def test_new_game_resets_and_ai_opens_again(self):
        self.game.play_white(0, 1, self.game.revision)
        old_revision = self.game.revision
        state = self.game.new_game()
        self.assertEqual(state["move_count"], 1)
        self.assertEqual(state["turn"], "white")
        self.assertEqual(state["board"][0][:3], [1, 0, 0])
        self.assertGreater(state["revision"], old_revision)

    def test_inference_error_rolls_back_white_move(self):
        before = self.game.state()
        with torch.no_grad():
            self.model.value.fill_(float("nan"))
        with self.assertRaises(FloatingPointError):
            self.game.play_white(0, 1, self.game.revision)
        self.assertEqual(self.game.state(), before)
        with torch.no_grad():
            self.model.value.fill_(0.0)
        self.assertEqual(self.game.play_white(0, 1, self.game.revision)["move_count"], 3)

    def test_new_game_inference_error_preserves_previous_game(self):
        self.game.play_white(0, 1, self.game.revision)
        before = self.game.state()
        with torch.no_grad():
            self.model.logits[0] = float("nan")
        with self.assertRaises(FloatingPointError):
            self.game.new_game()
        self.assertEqual(self.game.state(), before)


class CheckpointWatchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.first = self.root / "alphazero_aaaa" / "iteration_001.pt"
        save_fake_checkpoint(self.first, seed=1, iteration=1)
        os.utime(self.first, ns=(1_000_000_000, 1_000_000_000))

    def tearDown(self):
        self.tmp.cleanup()

    def add_newer(self, run, iteration, seconds):
        path = self.root / run / f"iteration_{iteration:03d}.pt"
        save_fake_checkpoint(path, seed=iteration + 100, iteration=iteration)
        os.utime(path, ns=(seconds * 1_000_000_000,) * 2)
        return path

    def test_latest_is_newest_file_across_runs_and_ignores_tmp(self):
        newer = self.add_newer("alphazero_bbbb", 1, 2)
        (self.root / "alphazero_aaaa" / "iteration_002.pt.tmp").write_bytes(b"partial")
        self.assertEqual(find_latest_checkpoint(self.root), newer)

    def test_new_checkpoint_switches_only_at_new_game(self):
        game = GomokuGame(simulations=SIMS, watcher=CheckpointWatcher(self.root), seed=0)
        self.assertEqual((game.info["run"], game.info["iteration"]), ("alphazero_aaaa", 1))
        game.play_white(*self.free_cell(game), game.revision)
        self.add_newer("alphazero_aaaa", 2, 5)
        game.play_white(*self.free_cell(game), game.revision)
        self.assertEqual(game.state()["model"]["iteration"], 1)  # never mid-game
        state = game.new_game()
        self.assertEqual((state["model"]["iteration"], state["model"]["updates"]), (2, 20))
        self.assertEqual(state["move_count"], 1)

    def test_broken_newer_checkpoint_keeps_current_model(self):
        game = GomokuGame(simulations=SIMS, watcher=CheckpointWatcher(self.root), seed=0)
        broken = self.root / "alphazero_cccc" / "iteration_001.pt"
        broken.parent.mkdir()
        broken.write_bytes(b"not a checkpoint")
        os.utime(broken, ns=(9_000_000_000,) * 2)
        state = game.new_game()
        self.assertEqual(state["model"]["run"], "alphazero_aaaa")
        self.assertEqual(state["move_count"], 1)

    def test_pinned_checkpoint_never_switches(self):
        game = GomokuGame(simulations=SIMS, seed=0,
                          watcher=CheckpointWatcher(self.root, pinned=self.first))
        self.add_newer("alphazero_bbbb", 3, 5)
        self.assertEqual(game.new_game()["model"]["run"], "alphazero_aaaa")

    @staticmethod
    def free_cell(game):
        board = game.game_state.board
        row, col = map(int, np.argwhere(board == 0)[-1])
        return row, col


class RealCheckpointTests(unittest.TestCase):
    def test_newest_real_checkpoint_loads_on_cpu_plays_and_stays_unchanged(self):
        try:
            path = find_latest_checkpoint(CHECKPOINT_ROOT)
        except FileNotFoundError:
            self.skipTest("no checkpoints on disk")
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        model, info = load_model(path)
        self.assertFalse(model.training)
        self.assertTrue(all(p.device.type == "cpu" and not p.requires_grad for p in model.parameters()))
        self.assertEqual(info["file"], path.name)
        game = GomokuGame(model, info, simulations=SIMS, seed=0)
        state = game.play_white(*map(int, np.argwhere(game.game_state.board == 0)[0]), game.revision)
        self.assertEqual(state["move_count"], 3)
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)


class HTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = make_server(GomokuGame(FixedPrior(), {"run": "fake"}, simulations=SIMS, seed=0), port=0)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def request(self, method, path, payload=None, raw=None, content_type="application/json"):
        connection = HTTPConnection("127.0.0.1", self.server.server_port, timeout=10)
        body = json.dumps(payload) if payload is not None else raw
        connection.request(method, path, body=body, headers={"Content-Type": content_type})
        response = connection.getresponse()
        status, headers, data = response.status, dict(response.getheaders()), response.read()
        connection.close()
        return status, headers, data

    def test_server_binds_to_loopback_and_serves_assets(self):
        self.assertEqual(self.server.server_address[0], "127.0.0.1")
        for path in ("/", "/app.js", "/style.css", "/favicon.svg"):
            with self.subTest(path=path):
                status, _, data = self.request("GET", path)
                self.assertEqual(status, 200)
                self.assertTrue(data)

    def test_http_game_flow_and_stale_move(self):
        status, _, body = self.request("POST", "/api/new", {})
        self.assertEqual(status, 200)
        state = json.loads(body)
        self.assertEqual(state["model"]["run"], "fake")
        payload = {"row": 0, "col": 1, "revision": state["revision"]}
        status, _, body = self.request("POST", "/api/move", payload)
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["move_count"], 3)
        status, _, _ = self.request("POST", "/api/move", payload)
        self.assertEqual(status, 409)
        status, _, body = self.request("GET", "/api/state")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["move_count"], 3)

    def test_bad_requests_and_project_file_access_are_rejected(self):
        status, _, _ = self.request("POST", "/api/move", raw="{broken")
        self.assertEqual(status, 400)
        status, _, _ = self.request("POST", "/api/move", [])
        self.assertEqual(status, 400)
        status, _, _ = self.request("POST", "/api/new", {}, content_type="text/plain")
        self.assertEqual(status, 415)
        for path in ("/checkpoints/", "/../mcts.py", "/server.py"):
            status, _, _ = self.request("GET", path)
            self.assertEqual(status, 404)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
