"""Saving games against you, reviewing lost ones, and self-play from their positions."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

import train as training
from human_games import (ai_did_not_win, ai_positions, load_human_games, lost_game_positions,
                         review_path, save_web_game, state_from_observation)
from mcts import MCTS
from network import GomokuNetwork
from review import games_to_review, review_games
from self_play import play_game
from state import GomokuState
from test_web_app import FixedPrior
from web_app.server import GameArchive, GomokuGame

CONFIG = dict(channels=8, value_hidden=8, policy_kernel=3,
              architecture="resnet_global", residual_blocks=1)


def save_model(path, seed=0):
    torch.manual_seed(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    training.save_checkpoint({"model_config": CONFIG,
                              "model_state_dict": GomokuNetwork(**CONFIG).state_dict()}, path)


def lost_game(path, ai_player=-1):
    """A short game that black wins; the AI (white by default) loses.

    Every position is saved: the AI's with a (fake) search policy, yours with
    your move at imitation weight 0.5. Returns the actions played.
    """
    moves = [180, 0, 181, 1, 182, 2, 183, 3, 184]  # black: row 9, cols 9-13
    state, records = GomokuState(), []
    for action in moves:
        if state.current_player == ai_player:
            policy = np.full(361, 1 / 361, np.float32)
        else:
            policy = np.zeros(361, np.float32)
            policy[action] = 0.5
        records.append((state.encode(), policy, state.current_player))
        state.step(action)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_web_game(path, records, int(state.winner), ai_player, moves)
    return moves


class SavedGameTests(unittest.TestCase):
    def test_position_rebuilds_from_encoding_for_both_sides(self):
        rng = np.random.default_rng(0)
        state = GomokuState()
        for _ in range(30):
            encoded = state.encode()
            rebuilt = state_from_observation(encoded)
            np.testing.assert_array_equal(rebuilt.board, state.board)
            self.assertEqual(rebuilt.current_player, state.current_player)
            np.testing.assert_array_equal(rebuilt.encode(), encoded)
            state.step(int(rng.choice(state.legal_actions())))
            if state.done:
                break

    def test_ai_moves_and_result_in_new_and_old_files(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "x_web_play" / "game_0001.npz"
            lost_game(path)
            with np.load(path) as game:
                new = dict(game)
            self.assertEqual(ai_positions(new).tolist(), [False, True] * 4 + [False])
            self.assertTrue(ai_did_not_win(new))
            # Older files: no players/winner; AI search policies sum to 1.
            old = {k: new[k] for k in ("observations", "policies", "outcomes")}
            self.assertEqual(ai_positions(old).tolist(), ai_positions(new).tolist())
            self.assertTrue(ai_did_not_win(old))
            old["outcomes"] = -old["outcomes"]  # same game, but the AI won
            self.assertFalse(ai_did_not_win(old))

    def test_play_only_mode_saves_finished_games(self):
        with tempfile.TemporaryDirectory() as root:
            game = GomokuGame(FixedPrior(), {"run": "fake"}, simulations=8, seed=0,
                              archive=GameArchive(root), imitate_weight=0.5)
            patcher = patch("builtins.print")
            patcher.start()
            self.addCleanup(patcher.stop)
            game.play_human(18, 18, game.revision)
            board = game.game_state.board  # set up a win for you (white)
            board[board == -1] = 0
            board[15, 2:6] = -1
            game.play_human(15, 6, game.revision)
            self.assertEqual(game.state()["learning"], {"enabled": False, "saved_games": 1})
            files = list(Path(root).glob("*_web_play/game_0001.npz"))
            self.assertEqual(len(files), 1)
            with np.load(files[0]) as saved:
                self.assertEqual((int(saved["ai_player"]), int(saved["winner"])), (1, -1))
                self.assertEqual(saved["moves"].tolist()[-1], 15 * 19 + 6)
                self.assertEqual(int((saved["players"] == -1).sum()), 2)  # your 2 moves
                self.assertTrue(np.all(saved["outcomes"][saved["players"] == 1] == -1))
            self.assertEqual(load_human_games(root)[1], 1)
            # Without an archive (--no-save), nothing is saved.
            unsaved = GomokuGame(FixedPrior(), {"run": "fake"}, simulations=8, seed=0)
            unsaved.play_human(18, 18, unsaved.revision)
            self.assertEqual(unsaved.records, [])
            self.assertEqual(len(list(Path(root).iterdir())), 1)


class ReviewTests(unittest.TestCase):
    def test_lost_game_ai_moves_get_deeper_targets_once(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            model_path = root / "run" / "model.pt"
            save_model(model_path)
            lost = root / "2026-01-01_0000_web_play" / "game_0001.npz"
            lost_game(lost)
            won = root / "2026-01-01_0000_web_play" / "game_0002.npz"
            lost_game(won, ai_player=1)  # black (the AI here) wins: nothing to review
            self.assertEqual(games_to_review(root), [lost])

            self.assertEqual(review_games(root, model_path, workers=1, simulations=16,
                                          log=lambda *_: None), 1)
            self.assertFalse(review_path(won).exists())
            with np.load(lost) as original, np.load(review_path(lost)) as reviewed:
                ai = ai_positions(original)
                np.testing.assert_array_equal(reviewed["outcomes"], original["outcomes"])
                np.testing.assert_array_equal(reviewed["policies"][~ai], original["policies"][~ai])
                # Each AI move's target is now the review search's policy.
                model = GomokuNetwork(**CONFIG)
                model.load_state_dict(torch.load(model_path, weights_only=True)["model_state_dict"])
                model.eval()
                for i in np.flatnonzero(ai):
                    expected, _ = MCTS(model, simulations=16).search(
                        state_from_observation(original["observations"][i]), add_noise=False)
                    np.testing.assert_allclose(reviewed["policies"][i], expected, atol=1e-6)
                reviewed_policies = reviewed["policies"].copy()
            # Training uses the reviewed file; the original is kept unchanged.
            samples, loaded, _ = load_human_games(root)
            self.assertEqual(loaded, 2)
            np.testing.assert_allclose(samples[1][1], reviewed_policies[1])
            self.assertEqual(games_to_review(root), [])  # reviewed only once
            # Only the lost game, without its first 4 (nearly empty) positions.
            self.assertEqual(len(lost_game_positions(root)), 5)


class StartPositionTests(unittest.TestCase):
    def test_play_game_continues_from_a_position(self):
        start = GomokuState()
        for action in (180, 0, 181, 1):
            start.step(action)
        samples, final = play_game(GomokuNetwork(**CONFIG).eval(), simulations=4, seed=0,
                                   exploration_moves=2, start_state=start)
        np.testing.assert_array_equal(samples[0][0], start.encode())
        self.assertEqual(int((final.board != 0).sum()), 4 + len(samples))
        self.assertEqual(int(start.board.sum()), 0)  # the start position is not changed

    def test_some_selfplay_games_start_from_lost_games_and_skip_color_counts(self):
        with tempfile.TemporaryDirectory() as root, \
                patch.object(training, "__file__", str(Path(root) / "train.py")), \
                patch.object(training, "Live"), patch("builtins.print"):
            checkpoints = Path(root) / "checkpoints"
            lost_game(checkpoints / "2026-01-01_0000_web_play" / "game_0001.npz")
            _, _, run_dir = training.train(
                iterations=3, games_per_iteration=4, simulations=2, updates_per_iteration=1,
                batch_size=8, eval_every=1000, eval_games=2, seed=3, start_fraction=0.5)
            history = [json.loads(line) for line in
                       (run_dir / "history.jsonl").read_text().splitlines()]
            from_positions = sum(h["position_games"] for h in history)
            counted = sum(history[-1]["self_play"].values())
            self.assertGreater(from_positions, 0)
            self.assertEqual(counted + from_positions, 12)
            log = (Path(root) / "game_logs" / f"selfplay_{run_dir.name}.txt").read_text()
            self.assertEqual(log.count("from your game"), from_positions)

    def test_changed_settings_start_a_new_run_from_the_newest_model(self):
        small = dict(games_per_iteration=1, simulations=2, updates_per_iteration=1,
                     batch_size=8, eval_games=2, eval_every=1000, human_fraction=0.1)
        with tempfile.TemporaryDirectory() as root, \
                patch.object(training, "__file__", str(Path(root) / "train.py")), \
                patch.dict(training.SELF_PLAY_SETTINGS, small), \
                patch("review.REVIEW_SIMULATIONS", 16), \
                patch.object(training, "Live"), patch("builtins.print"):
            checkpoints = Path(root) / "checkpoints"
            save_model(checkpoints / "start" / "model.pt")
            with patch.dict(training.SELF_PLAY_SETTINGS, {"start_fraction": 0.0}):
                _, _, old_run = training.continue_selfplay(
                    1, workers=1, start_from=checkpoints / "start" / "model.pt")
            # Code now has start_fraction 0.2, the old run 0.0: no exact resume.
            lost_game(checkpoints / "2026-01-01_0000_web_play" / "game_0001.npz")
            _, _, new_run = training.continue_selfplay(1, workers=1)
            self.assertNotEqual(new_run, old_run)
            self.assertTrue(review_path(
                checkpoints / "2026-01-01_0000_web_play" / "game_0001.npz").exists())
            # The same settings again: exact resume of the new run.
            _, _, again = training.continue_selfplay(1, workers=1)
            self.assertEqual(again, new_run)


if __name__ == "__main__":
    unittest.main()
