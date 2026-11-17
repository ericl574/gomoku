"""Web game and HTTP checks for the AlphaZero web app. No training; real checkpoints are only read."""
import hashlib
from http.client import HTTPConnection
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import unittest

import numpy as np
import torch
from torch import nn
from unittest.mock import patch

from network import GomokuNetwork
from state import GomokuState
from web_app.server import (
    CHECKPOINT_ROOT, CheckpointWatcher, GameConflict, GomokuGame, WebLearner,
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
                "games_completed": iteration * 12,
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
        state = self.game.play_human(0, 1, self.game.revision)
        self.assertEqual(state["board"][0][:3], [1, -1, 1])
        self.assertEqual(state["turn"], "white")
        self.assertEqual(state["move_count"], 3)
        torch.testing.assert_close(self.model.logits, before)

    def test_invalid_and_occupied_moves_do_not_change_the_game(self):
        before = self.game.state()
        for row, col in ((0, 0), (-1, 0), (19, 0), (0, 19), (True, 1), (0.5, 1), ("0", 1)):
            with self.subTest(row=row, col=col), self.assertRaises(ValueError):
                self.game.play_human(row, col, before["revision"])
            self.assertEqual(self.game.state(), before)

    def test_stale_revision_rejects_duplicate_requests(self):
        old_revision = self.game.revision
        self.game.play_human(0, 1, old_revision)
        before = self.game.state()
        with self.assertRaises(GameConflict):
            self.game.play_human(0, 3, old_revision)
        self.assertEqual(self.game.state(), before)

    def test_wrong_turn_is_rejected(self):
        self.game.game_state.current_player = 1
        with self.assertRaises(GameConflict):
            self.game.play_human(0, 1, self.game.revision)

    def test_white_wins_in_each_direction_and_no_ai_reply_follows(self):
        for dr, dc in ((1, 0), (0, 1), (1, 1), (1, -1)):
            with self.subTest(direction=(dr, dc)):
                self.game.new_game()
                self.game.game_state.board.fill(0)
                self.game.game_state.current_player = -1
                for offset in range(4):
                    self.game.game_state.board[5 + dr * offset, 9 + dc * offset] = -1
                state = self.game.play_human(5 + dr * 4, 9 + dc * 4, self.game.revision)
                self.assertEqual(state["winner"], "white")
                self.assertTrue(state["game_over"])
                self.assertIsNone(state["turn"])
                self.assertEqual(len(state["winning_line"]), 5)
                self.assertEqual(state["last_move"]["player"], "white")
                with self.assertRaises(GameConflict):
                    self.game.play_human(0, 0, self.game.revision)

    def test_ai_win_is_reported_as_black(self):
        board = self.game.game_state.board
        board.fill(0)
        board[0, 1:5] = 1
        state = self.game.play_human(2, 2, self.game.revision)
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
        state = self.game.play_human(0, 2, self.game.revision)
        self.assertEqual(state["winner"], "draw")
        self.assertTrue(state["game_over"])
        self.assertEqual(state["winning_line"], [])

    def test_full_board_draw_after_black(self):
        board = no_five_board()
        board[0, 0] = 0  # black square, AI fills it last
        board[0, 2] = 0  # white square
        self.game.game_state.board[:] = board
        self.game.game_state.current_player = -1
        state = self.game.play_human(0, 2, self.game.revision)
        self.assertEqual(state["winner"], "draw")
        self.assertEqual(state["last_move"], {"row": 0, "col": 0, "player": "black"})

    def test_new_game_resets_and_ai_opens_again(self):
        self.game.play_human(0, 1, self.game.revision)
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
            self.game.play_human(0, 1, self.game.revision)
        self.assertEqual(self.game.state(), before)
        with torch.no_grad():
            self.model.value.fill_(0.0)
        self.assertEqual(self.game.play_human(0, 1, self.game.revision)["move_count"], 3)

    def test_new_game_inference_error_preserves_previous_game(self):
        self.game.play_human(0, 1, self.game.revision)
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
        # New-style name; self.first keeps the old name, so both styles are covered.
        path = self.root / run / f"round_{iteration:03d}_games{iteration * 12:04d}.pt"
        save_fake_checkpoint(path, seed=iteration + 100, iteration=iteration)
        os.utime(path, ns=(seconds * 1_000_000_000,) * 2)
        return path

    def test_latest_is_newest_file_across_runs_and_ignores_tmp(self):
        newer = self.add_newer("alphazero_bbbb", 1, 2)
        (self.root / "alphazero_aaaa" / "round_002_games0024.pt.tmp").write_bytes(b"partial")
        self.assertEqual(find_latest_checkpoint(self.root), newer)

    def test_new_checkpoint_switches_only_at_new_game(self):
        game = GomokuGame(simulations=SIMS, watcher=CheckpointWatcher(self.root), seed=0)
        self.assertEqual((game.info["run"], game.info["iteration"]), ("alphazero_aaaa", 1))
        game.play_human(*self.free_cell(game), game.revision)
        self.add_newer("alphazero_aaaa", 2, 5)
        game.play_human(*self.free_cell(game), game.revision)
        self.assertEqual(game.state()["model"]["iteration"], 1)  # never mid-game
        state = game.new_game()
        self.assertEqual((state["model"]["iteration"], state["model"]["games"],
                          state["model"]["updates"]), (2, 24, 20))
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

    def test_best_json_wins_over_newer_files_and_latest_flag_ignores_it(self):
        from promote import write_best
        self.add_newer("alphazero_bbbb", 2, 5)  # newer than self.first
        write_best({"path": str(self.first), "status": "promoted"}, root=self.root)
        self.assertEqual(CheckpointWatcher(self.root).current()[0], self.first)
        self.assertIn("BEST (promoted", CheckpointWatcher(self.root).describe())
        latest = CheckpointWatcher(self.root, use_best=False).current()[0]
        self.assertEqual(latest.parent.name, "alphazero_bbbb")

    def test_without_best_json_falls_back_to_latest(self):
        self.add_newer("alphazero_bbbb", 2, 5)
        self.assertEqual(CheckpointWatcher(self.root).current()[0].parent.name, "alphazero_bbbb")
        self.assertIn("no best.json yet", CheckpointWatcher(self.root).describe())

    @staticmethod
    def free_cell(game):
        board = game.game_state.board
        row, col = map(int, np.argwhere(board == 0)[-1])
        return row, col


class LearningModeTests(unittest.TestCase):
    def setUp(self):
        # These tests check who moved via plane 2, so show the color plane here.
        color_plane = patch("state.COLOR_PLANE", True)
        color_plane.start()
        self.addCleanup(color_plane.stop)
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.source = self.root / "2026-01-01_0000_sims8" / "round_001_games0012.pt"
        save_fake_checkpoint(self.source, seed=1, iteration=1)

    def tearDown(self):
        self.tmp.cleanup()

    def make_game(self, imitate_weight=0.0):
        model, info = load_model(self.source)
        learner = WebLearner(model, self.source, SIMS, root=self.root, updates=3, seed=0,
                             imitate_weight=imitate_weight)
        return GomokuGame(model, info, simulations=SIMS, learner=learner, seed=0), model

    @staticmethod
    def white_wins_next(game):
        board = game.game_state.board
        board[board == -1] = 0
        board[15, 2:6] = -1
        board[15, 6] = 0
        return game.play_human(15, 6, game.revision)

    def test_finished_game_trains_saves_and_never_imitates_white(self):
        game, model = self.make_game()
        before = [p.detach().clone() for p in model.parameters()]
        game.play_human(*map(int, np.argwhere(game.game_state.board == 0)[-1]), game.revision)
        ai_moves = sum(m["player"] == "black" for m in game.moves)
        state = self.white_wins_next(game)

        self.assertEqual(state["winner"], "white")
        self.assertEqual(state["learning"]["games"], 1)
        self.assertEqual(state["learning"]["last"]["result"], "AI lost as black")
        self.assertGreater(state["learning"]["last"]["weight_change"], 0.0)
        self.assertFalse(model.training)
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, model.parameters())))

        run_dir = next(self.root.glob("*_web_sims8"))
        saved = np.load(run_dir / "game_0001.npz")
        # One example per AI move only; value = loss from the AI's side.
        self.assertEqual(len(saved["outcomes"]), ai_moves)
        self.assertTrue(np.all(saved["outcomes"] == -1.0))
        np.testing.assert_allclose(saved["policies"].sum(axis=1), 1.0, rtol=1e-5)
        self.assertTrue(np.all(saved["observations"][:, 2] == 1.0))  # all black-to-move

        model2, info2 = load_model(run_dir / "round_001_games0001.pt")
        self.assertEqual((info2["games"], info2["updates"]), (1, 13))
        self.assertEqual(state["model"]["file"], "round_001_games0001.pt")
        for a, b in zip(model.parameters(), model2.parameters()):
            torch.testing.assert_close(a, b)

    def test_imitate_adds_your_moves_at_reduced_weight(self):
        game, _ = self.make_game(imitate_weight=0.5)
        first = tuple(map(int, np.argwhere(game.game_state.board == 0)[-1]))
        game.play_human(*first, game.revision)
        state = self.white_wins_next(game)
        self.assertEqual(state["learning"]["imitate_weight"], 0.5)
        self.assertEqual(state["learning"]["last"]["your_moves"], 2)

        saved = np.load(next(self.root.glob("*_web_sims8")) / "game_0001.npz")
        yours = saved["observations"][:, 2, 0, 0] == 0.0  # white-to-move examples
        self.assertEqual(int(yours.sum()), 2)
        # Your examples: target is your move at weight 0.5, value = you won (+1).
        np.testing.assert_allclose(saved["policies"][yours].sum(axis=1), 0.5)
        self.assertEqual(int(saved["policies"][yours][0].argmax()), first[0] * 19 + first[1])
        self.assertEqual(int(saved["policies"][yours][1].argmax()), 15 * 19 + 6)
        self.assertTrue(np.all(saved["outcomes"][yours] == 1.0))
        # The AI's own examples are unchanged: full-weight search policy, value -1.
        np.testing.assert_allclose(saved["policies"][~yours].sum(axis=1), 1.0, rtol=1e-5)
        self.assertTrue(np.all(saved["outcomes"][~yours] == -1.0))

    def test_colors_alternate_and_ai_learns_as_white(self):
        game, _ = self.make_game()
        self.assertEqual(game.state()["ai_color"], "black")
        self.white_wins_next(game)  # game 1: AI black, learned

        state = game.new_game()  # game 2: AI white, you (black) move first
        self.assertEqual((state["ai_color"], state["human_color"]), ("white", "black"))
        self.assertEqual((state["move_count"], state["turn"]), (0, "black"))
        with self.assertRaises(GameConflict):  # you are black now, so a move on white's turn is refused
            game.game_state.current_player = -1
            game.play_human(0, 0, game.revision)
        game.game_state.current_player = 1

        state = game.play_human(18, 18, game.revision)
        self.assertEqual(state["last_ai_move"]["player"], "white")
        board = game.game_state.board
        board[board == 1] = 0
        board[3, 2:6] = 1
        board[3, 6] = 0
        state = game.play_human(3, 6, game.revision)
        self.assertEqual(state["winner"], "black")
        self.assertEqual(state["learning"]["last"]["result"], "AI lost as white")

        saved = np.load(next(self.root.glob("*_web_sims8")) / "game_0002.npz")
        self.assertTrue(np.all(saved["observations"][:, 2, 0, 0] == 0.0))  # white to move
        self.assertTrue(np.all(saved["outcomes"] == -1.0))
        self.assertEqual(game.new_game()["ai_color"], "black")  # game 3: back to black

    def test_finished_game_is_logged_with_search_details(self):
        model, info = load_model(self.source)
        learner = WebLearner(model, self.source, SIMS, root=self.root, updates=3, seed=0)
        game = GomokuGame(model, info, simulations=SIMS, learner=learner, seed=0,
                          log_dir=self.root / "logs")
        self.white_wins_next(game)
        text = game.log_path.read_text()
        self.assertIn("=== Game 1 |", text)
        self.assertIn("AI black | You won (white) in", text)
        self.assertIn("top: ", text)          # AI moves show what the search looked at
        self.assertIn("learned: AI lost as black", text)
        self.assertIn("  1. B AI ", text)
        self.assertIn("  2. W You  G16", text)  # board label: column G, row 16
        self.assertEqual(text.count("\n 19 "), 1)  # one 19-row board diagram

    def test_fresh_forgets_older_games_now_and_on_later_restarts(self):
        game, _ = self.make_game()
        self.white_wins_next(game)
        old_run = next(self.root.glob("*_web_sims8"))
        old_run.rename(self.root / "2020-01-01_0000_web_sims8")  # clearly older than now

        model, info = load_model(self.source)
        fresh = WebLearner(model, self.source, SIMS, root=self.root, updates=3, seed=0, fresh=True)
        self.assertEqual((len(fresh.buffer), fresh.skipped_games), (0, 1))
        game = GomokuGame(model, info, simulations=SIMS, learner=fresh, seed=0)
        self.white_wins_next(game)  # a new, post-cutoff game
        new_positions = game.state()["learning"]["positions"]

        model, _ = load_model(self.source)
        later = WebLearner(model, self.source, SIMS, root=self.root, seed=0)  # no --fresh
        self.assertEqual((len(later.buffer), later.skipped_games), (new_positions, 1))

    def test_normal_mode_ai_stays_black(self):
        game = GomokuGame(FixedPrior(), {"run": "fake"}, simulations=SIMS, seed=0)
        for _ in range(3):
            self.assertEqual(game.new_game()["ai_color"], "black")

    def test_play_only_ai_color_white_and_alternate(self):
        white = GomokuGame(FixedPrior(), {"run": "fake"}, simulations=SIMS, seed=0, ai_color="white")
        for _ in range(2):
            snapshot = white.state()
            self.assertEqual((snapshot["ai_color"], snapshot["turn"]), ("white", "black"))
            self.assertEqual(len(white.moves), 0)  # the human (black) moves first
            white.play_human(9, 9, snapshot["revision"])
            self.assertEqual(len(white.moves), 2)  # the AI answered as white
            white.new_game()
        alternate = GomokuGame(FixedPrior(), {"run": "fake"}, simulations=SIMS, seed=0,
                               ai_color="alternate")
        colors = [alternate.state()["ai_color"]] + [alternate.new_game()["ai_color"] for _ in range(3)]
        self.assertEqual(colors, ["black", "white", "black", "white"])

    def test_unfinished_game_is_not_learned(self):
        game, model = self.make_game()
        game.play_human(*map(int, np.argwhere(game.game_state.board == 0)[-1]), game.revision)
        before = [p.detach().clone() for p in model.parameters()]
        game.new_game()
        self.assertEqual(game.state()["learning"]["games"], 0)
        self.assertFalse(list(self.root.glob("*_web_*")))
        for a, b in zip(before, model.parameters()):
            torch.testing.assert_close(a, b)

    def test_next_session_reloads_earlier_games(self):
        game, _ = self.make_game()
        self.white_wins_next(game)
        positions = game.state()["learning"]["positions"]
        model, _ = load_model(self.source)
        learner = WebLearner(model, self.source, SIMS, root=self.root, seed=0)
        self.assertEqual(len(learner.buffer), positions)

    def test_learning_mode_refuses_auto_swap(self):
        model, info = load_model(self.source)
        learner = WebLearner(model, self.source, SIMS, root=self.root)
        with self.assertRaises(ValueError):
            GomokuGame(model, info, simulations=SIMS, learner=learner,
                       watcher=CheckpointWatcher(self.root))


class PromotionTests(unittest.TestCase):
    def test_decision_needs_enough_games_and_a_clear_score(self):
        from promote import decide
        self.assertTrue(decide([1] * 26 + [-1] * 14, 40, 0.6)["promoted"])     # 65% of 40
        self.assertFalse(decide([1] * 22 + [-1] * 18, 40, 0.6)["promoted"])    # 55%: not clear
        self.assertFalse(decide([1] * 10, 40, 0.6)["promoted"])                # too few games
        verdict = decide([1, 0, -1, 1], 4, 0.6)
        self.assertEqual((verdict["wins"], verdict["draws"], verdict["losses"]), (2, 1, 1))
        self.assertAlmostEqual(verdict["score"], 0.625)

    def test_match_swaps_colors_on_the_same_opening_and_never_writes_best(self):
        import promote
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a = root / "run" / "a.pt"
            b = root / "run" / "b.pt"
            save_fake_checkpoint(a, seed=1, iteration=1)
            save_fake_checkpoint(b, seed=2, iteration=1)
            calls = []
            real = promote.play_from_opening

            def spy(candidate, opponent, color, opening, simulations, seed):
                calls.append((color, tuple(opening)))
                return real(candidate, opponent, color, opening, simulations, seed)

            with patch.object(promote, "play_from_opening", side_effect=spy):
                results, lengths = promote.run_match(str(a), str(b), games=4, simulations=2,
                                                     workers=1, seed=0)
            self.assertEqual(len(results), 4)
            self.assertTrue(all(r in (-1, 0, 1) for r in results))
            self.assertTrue(all(n >= 9 for n in lengths))  # 2 opening stones + a five at least
            self.assertEqual([c for c, _ in calls], [1, -1, 1, -1])
            self.assertEqual(calls[0][1], calls[1][1])  # same opening, colors swapped
            self.assertFalse((root / "best.json").exists())

    def test_best_record_round_trip(self):
        from promote import read_best, write_best
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertIsNone(read_best(root))
            write_best({"path": str(root / "x.pt"), "status": "baseline"}, root=root)
            self.assertEqual(read_best(root)["status"], "baseline")
            self.assertFalse(list(root.glob("*.tmp")))


class ColorBlindTests(unittest.TestCase):
    def test_default_encoding_hides_whose_turn_it_is(self):
        state = GomokuState()
        black_view = state.encode()
        state.step(0)
        white_view = state.encode()
        self.assertTrue(np.all(black_view[2] == 0) and np.all(white_view[2] == 0))
        # Own/opponent planes still flip with the mover, and stone counts still differ.
        self.assertEqual((white_view[0].sum(), white_view[1].sum()), (0, 1))


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
        state = game.play_human(*map(int, np.argwhere(game.game_state.board == 0)[0]), game.revision)
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

    def test_app_js_parses(self):
        # A JS syntax error blanks the whole page while every API test still passes.
        node = shutil.which("node")
        if node is None:
            self.skipTest("node is not installed")
        app_js = Path(__file__).resolve().parent / "web_app" / "app.js"
        result = subprocess.run([node, "--check", str(app_js)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

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
