"""Small targeted regression tests for the maintenance changes."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from diagnose_defense import scenarios, check_scenario
from train import ReplayBuffer, pack_resume_state, load_resume_state, save_checkpoint
from web_app.server import normalized_web_game


class MaintenanceTests(unittest.TestCase):
    def test_legacy_color_channel_only_normalized_in_memory(self):
        observations = np.zeros((2, 3, 19, 19), dtype=np.float32)
        observations[0, 2] = 1
        observations[0, 0, 9, 9] = 1
        policies = np.zeros((2, 361), dtype=np.float32)
        policies[:, 180] = 1
        outcomes = np.array([1, -1], dtype=np.float32)
        original = observations.copy()
        samples = normalized_web_game(observations, policies, outcomes)
        self.assertTrue(all(np.all(s[0][2] == 0) for s in samples))
        np.testing.assert_array_equal(observations, original)
        np.testing.assert_array_equal(samples[0][0][:2], original[0, :2])

    def test_bad_web_shapes_are_rejected(self):
        with self.assertRaises(ValueError):
            normalized_web_game(np.zeros((1, 2, 19, 19)), np.zeros((1, 361)), np.array([1]))

    def test_diagnostic_positions_validated(self):
        for name, state, answer in scenarios():
            self.assertTrue(check_scenario(name, state, answer))

    def test_resume_state_roundtrip(self):
        model = nn.Linear(2, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        replay = ReplayBuffer(4)
        replay.add_game([(np.zeros((3,19,19),np.float32), np.ones(361,np.float32)/361, -1.0)])
        rng = np.random.default_rng(22)
        aug = np.random.default_rng(23)
        erng = np.random.default_rng(24)
        config = {'model_config': {'channels':32}, 'games_per_iteration':1}
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/'resume_state.bin'
            save_checkpoint(pack_resume_state(model, optimizer, replay, rng, aug, erng,
                model, 3, 2, 5, {1:2,-1:0,0:0}, 0, config, root,
                np.array([4,5],dtype=np.int64), None), path)
            data = load_resume_state(path)
            self.assertEqual(data['iteration'], 3)
            self.assertEqual(data['games_completed'], 2)
            self.assertEqual(len(data['replay'][0]), 1)
            self.assertEqual(data['rng_state'], rng.bit_generator.state)
            self.assertEqual(data['color_results'][1], 2)
            np.testing.assert_array_equal(data['eval_seeds'].numpy(), [4,5])

    def test_human_samples_mix_without_changing_plain_sampling(self):
        def sample(data):
            return (np.full((3, 19, 19), data, np.float32), np.ones(361, np.float32) / 361, 0.0)
        plain, mixed = ReplayBuffer(50), ReplayBuffer(50)
        for buffer in (plain, mixed):
            buffer.add_game([sample(0) for _ in range(50)])
        mixed.add_fixed([sample(1) for _ in range(20)])
        mixed.add_game([sample(0) for _ in range(50)])  # self-play churn never evicts human
        self.assertEqual((len(mixed), len(mixed.fixed)), (50, 20))
        # fraction 0: identical batches and rng use as a buffer without human games
        a = plain.sample(16, np.random.default_rng(3))
        b = mixed.sample(16, np.random.default_rng(3), fixed_fraction=0.0)
        for x, y in zip(a, b):
            self.assertTrue(torch.equal(x, y))
        # fraction 0.1: about 10% of samples are human positions
        rng = np.random.default_rng(4)
        human = sum(int((mixed.sample(32, rng, fixed_fraction=0.1)[0][:, 0, 0, 0] == 1).sum())
                    for _ in range(200))
        self.assertAlmostEqual(human / (200 * 32), 0.1, delta=0.02)

    def test_selfplay_and_web_learning_alternate(self):
        """python train.py: new run -> exact resume -> (web save) -> new run with carried replay."""
        import json
        import train as training
        from network import GomokuNetwork
        from state import GomokuState

        def history(run_dir):
            return [json.loads(l) for l in (run_dir / "history.jsonl").read_text().splitlines()]

        small = dict(games_per_iteration=1, simulations=2, updates_per_iteration=2,
                     batch_size=8, eval_games=2, eval_every=1000, human_fraction=0.1)
        with tempfile.TemporaryDirectory() as root, \
                patch.object(training, "__file__", str(Path(root) / "train.py")), \
                patch.dict(training.SELF_PLAY_SETTINGS, small), \
                patch.object(training, "Live"), patch("builtins.print"):
            checkpoints = Path(root) / "checkpoints"
            config = dict(channels=32, value_hidden=64, policy_kernel=3,
                          architecture="resnet_global", residual_blocks=3)
            (checkpoints / "start").mkdir(parents=True)
            start = checkpoints / "start" / "model.pt"
            save_checkpoint({"model_config": config,
                             "model_state_dict": GomokuNetwork(**config).state_dict()}, start)
            web = checkpoints / "2026-01-01_0000_web_sims8"
            web.mkdir()
            policy = np.zeros((5, 361), np.float32)
            policy[:, 180] = 1
            np.savez_compressed(web / "game_0001.npz", policies=policy,
                                observations=np.stack([GomokuState().encode()] * 5),
                                outcomes=np.ones(5, np.float32))

            _, _, run_a = training.continue_selfplay(2, workers=1, start_from=start)
            self.assertEqual(history(run_a)[-1]["human_positions"], 5)
            _, _, resumed = training.continue_selfplay(1, workers=1)   # exact resume
            self.assertEqual(resumed, run_a)
            self.assertEqual([r["iteration"] for r in history(run_a)], [1, 2, 3])

            # Web learning saves a newer model; an exact resume would discard it.
            web_model = web / "round_001_games0001.pt"
            web_model.write_bytes(sorted(run_a.glob("round_003_*.pt"))[0].read_bytes())
            with self.assertRaises(ValueError):
                training.train(resume_from=str(run_a / "resume_state.bin"),
                               seed=training.load_resume_state(run_a / "resume_state.bin")["seed"],
                               **small)
            _, replay, run_b = training.continue_selfplay(1, workers=1)
            self.assertNotEqual(run_b, run_a)
            checkpoint = torch.load(sorted(run_b.glob("round_001_*.pt"))[0], weights_only=True)
            self.assertEqual(Path(checkpoint["init_checkpoint"]).resolve(), web_model.resolve())
            carried = history(run_a)[-1]["buffer_size"]
            self.assertGreater(history(run_b)[0]["buffer_size"], carried)  # old memory + new games

    def test_resume_matches_uninterrupted_run(self):
        """2 rounds + resume 2 rounds == 4 rounds straight, and the last round is evaluated."""
        import json
        import train as training
        kw = dict(games_per_iteration=1, simulations=2, updates_per_iteration=2,
                  batch_size=8, eval_every=1000, eval_games=2, seed=7)

        def history(run_dir):
            lines = (run_dir / "history.jsonl").read_text().splitlines()
            return [(r["iteration"], r["evaluation"] is not None) for r in map(json.loads, lines)]

        with tempfile.TemporaryDirectory() as root, \
                patch.object(training, "__file__", str(Path(root) / "train.py")), \
                patch.object(training, "Live"), patch("builtins.print"):
            straight, _, straight_dir = training.train(iterations=4, **kw)
            _, _, original_dir = training.train(iterations=2, **kw)

            # Move the run folder (e.g. project copied to another machine).
            run_dir = Path(root) / "moved" / original_dir.name
            run_dir.parent.mkdir()
            original_dir.rename(run_dir)
            # Simulate a crash after history was written but before resume_state:
            # a re-run round 3 line plus a half-written line.
            with (run_dir / "history.jsonl").open("a") as file:
                file.write('{"iteration": 3, "evaluation": null}\n{"iterat')

            # A resume file inside a differently named folder is refused.
            renamed = Path(root) / "renamed"
            renamed.mkdir()
            (renamed / "resume_state.bin").write_bytes((run_dir / "resume_state.bin").read_bytes())
            with self.assertRaises(ValueError):
                training.train(iterations=2, resume_from=str(renamed / "resume_state.bin"), **kw)

            resumed, _, resumed_dir = training.train(
                iterations=2, resume_from=str(run_dir / "resume_state.bin"), **kw)

            self.assertEqual(resumed_dir, run_dir.resolve())
            for a, b in zip(straight.state_dict().values(), resumed.state_dict().values()):
                self.assertTrue(torch.equal(a, b))
            self.assertEqual(history(straight_dir), [(1, False), (2, False), (3, False), (4, True)])
            # Round 2 ended the first call; round 4 ends the resumed call.
            self.assertEqual(history(run_dir), [(1, False), (2, True), (3, False), (4, True)])


if __name__ == '__main__':
    unittest.main()
