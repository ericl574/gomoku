import tempfile
import unittest
from pathlib import Path
from unittest.mock import mock_open, patch

import numpy as np
import torch
from torch import nn

import train as training
from evaluate import play_match
from mcts import MCTS, TreeNode
from self_play import play_game
from state import GomokuState


class CountingNetwork(nn.Module):
    """A deterministic diagnostic network; never used as training data."""

    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0))
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        logits = torch.zeros(len(x), 361, device=x.device)
        logits[:, 180] = 10.0
        values = x[:, 2, 0, 0].unsqueeze(1) * 0.2 - 0.3 + self.anchor
        return logits, values


class SearchTests(unittest.TestCase):
    def test_optimized_selection_matches_reference_formula(self):
        rng = np.random.default_rng(19)
        searcher = MCTS(CountingNetwork())
        for _ in range(20):
            root = TreeNode()
            root.visit_count = 512
            priors = rng.dirichlet(np.ones(50))
            for action, prior in enumerate(priors):
                child = TreeNode(prior)
                child.visit_count = int(rng.integers(0, 20))
                child.value_sum = float(rng.uniform(-1, 1) * child.visit_count)
                root.children[action] = child
            expected = max(
                root.children.items(),
                key=lambda item: (
                    -item[1].q
                    + searcher.c_puct * item[1].prior
                    * np.sqrt(root.visit_count + 1)
                    / (1 + item[1].visit_count)
                ),
            )
            self.assertEqual(searcher._select(root)[0], expected[0])

    def test_reuse_retains_statistics_and_adds_exact_budget(self):
        state = GomokuState()
        searcher = MCTS(CountingNetwork(), simulations=8)
        _, root = searcher.search(state)
        child = root.children[180]
        prior_visits = child.visit_count
        prior_child_visits = sum(c.visit_count for c in child.children.values())
        board = state.board.copy()
        state.step(180)
        searcher.advance(180, state)
        policy, reused = searcher.search(state)
        self.assertIs(reused, child)
        self.assertEqual(reused.visit_count, prior_visits + 8)
        self.assertEqual(
            sum(c.visit_count for c in reused.children.values()),
            prior_child_visits + 8,
        )
        self.assertEqual(searcher.last_search_stats["reused_visits"], prior_visits)
        self.assertAlmostEqual(float(policy.sum()), 1.0)
        self.assertEqual(policy[180], 0)
        board[9, 9] = 1
        np.testing.assert_array_equal(state.board, board)

    def test_mismatched_advance_discards_tree(self):
        state = GomokuState()
        searcher = MCTS(CountingNetwork(), simulations=4)
        _, old = searcher.search(state)
        state.step(1)
        searcher.advance(180, state)
        _, root = searcher.search(state)
        self.assertIsNot(root, old.children[180])
        self.assertEqual(searcher.last_search_stats["reused_visits"], 0)
        self.assertEqual(root.visit_count, 4)

    def test_unannounced_position_change_discards_tree(self):
        state = GomokuState()
        searcher = MCTS(CountingNetwork(), simulations=4)
        searcher.search(state)
        state.step(0)
        _, root = searcher.search(state)
        self.assertEqual(searcher.last_search_stats["reused_visits"], 0)
        self.assertEqual(root.visit_count, 4)

    def test_root_noise_is_fresh_and_can_be_disabled(self):
        state = GomokuState()
        searcher = MCTS(CountingNetwork(), simulations=1, seed=17)
        rng = np.random.default_rng(17)
        for _ in range(2):
            _, root = searcher.search(state, add_noise=True)
            noise = rng.dirichlet(np.full(361, 0.03))
            actual = [child.prior for child in root.children.values()]
            expected = [
                0.75 * child.base_prior + 0.25 * n
                for child, n in zip(root.children.values(), noise)
            ]
            np.testing.assert_allclose(actual, expected)
        _, root = searcher.search(state, add_noise=False)
        self.assertTrue(all(c.prior == c.base_prior for c in root.children.values()))

    @torch.no_grad()
    @patch("state.COLOR_PLANE", True)  # the fake network tells players apart by plane 2
    def test_cache_keys_include_player_and_reset_invalidates_cache(self):
        model = CountingNetwork().eval()
        searcher = MCTS(model, simulations=1, cache_size=2)
        state = GomokuState()
        first = searcher._evaluate(state, TreeNode())
        self.assertEqual(searcher._evaluate(state, TreeNode()), first)
        self.assertEqual(model.calls, 1)
        state.current_player = -1
        other = searcher._evaluate(state, TreeNode())
        self.assertNotEqual(first, other)
        self.assertEqual(model.calls, 2)
        state.step(0)
        searcher._evaluate(state, TreeNode())
        self.assertEqual(len(searcher._cache), 2)
        model.anchor.add_(1)
        searcher.reset()
        self.assertEqual(len(searcher._cache), 0)
        self.assertNotEqual(searcher._evaluate(state, TreeNode()), other)

    def test_cache_does_not_change_search_results(self):
        state = GomokuState()
        cached = MCTS(CountingNetwork(), simulations=8, seed=5, cache_size=64)
        uncached = MCTS(CountingNetwork(), simulations=8, seed=5, cache_size=0)
        for _ in range(4):
            a, ra = cached.search(state, add_noise=True)
            b, rb = uncached.search(state, add_noise=True)
            np.testing.assert_array_equal(a, b)
            self.assertEqual(ra.visit_count, rb.visit_count)
            self.assertEqual(ra.value_sum, rb.value_sum)
            action = int(np.argmax(a))
            state.step(action)
            cached.advance(action, state)
            uncached.advance(action, state)

    def test_callbacks_preserve_search_and_report_endpoints(self):
        events = []
        state = GomokuState()
        plain = MCTS(CountingNetwork(), simulations=8, seed=3)
        observed = MCTS(CountingNetwork(), simulations=8, seed=3)
        a, _ = plain.search(state, add_noise=True)
        b, _ = observed.search(
            state, add_noise=True,
            progress_callback=lambda done, total: events.append((done, total)),
        )
        np.testing.assert_array_equal(a, b)
        self.assertEqual(events[0], (0, 8))
        self.assertEqual(events[-1], (8, 8))
        self.assertEqual(events, sorted(events))

    def test_terminal_sign_and_network_bypass_for_both_colors(self):
        for color in (1, -1):
            state = GomokuState()
            state.current_player = color
            state.board[9, 5:9] = color
            model = CountingNetwork()
            policy, root = MCTS(model, simulations=8).search(state)
            self.assertEqual(policy[180], 1)
            self.assertEqual(root.q, 1)
            self.assertEqual(root.children[180].q, -1)
            self.assertEqual(model.calls, 1)
            self.assertTrue(model.training)


class IntegrationTests(unittest.TestCase):
    def test_legacy_evaluation_can_disable_tree_reuse(self):
        import evaluate as evaluation
        retained = []

        class ObservedMCTS(MCTS):
            def search(self, *args, **kwargs):
                result = super().search(*args, **kwargs)
                retained.append(self.last_search_stats["reused_visits"])
                return result

        with patch.object(evaluation, "MCTS", ObservedMCTS):
            play_match(
                CountingNetwork().eval(), CountingNetwork().eval(),
                1, "mcts", 2, 0, reuse_tree=False,
            )
        self.assertTrue(retained)
        self.assertTrue(all(count == 0 for count in retained))

    def test_complete_self_play_and_evaluation_callbacks(self):
        model = CountingNetwork().eval()
        events = []
        samples, final = play_game(
            model, simulations=2, seed=0, progress_callback=events.append,
        )
        self.assertTrue(final.done)
        self.assertEqual(len(samples), np.count_nonzero(final.board))
        for i, (obs, policy, value) in enumerate(samples):
            self.assertEqual(obs[:2].sum(), i)
            self.assertAlmostEqual(float(policy.sum()), 1)
            self.assertFalse(policy[(obs[0] + obs[1]).ravel() != 0].any())
            self.assertEqual(value, final.winner * (1 if i % 2 == 0 else -1))
        self.assertEqual(events[0]["move"], 1)
        self.assertEqual(events[-1]["move"], len(samples))
        self.assertEqual(events[-1]["simulations_done"], 2)
        self.assertTrue(any(e["reused_visits"] > 0 for e in events))
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        events.clear()
        result, moves = play_match(
            model, CountingNetwork().eval(), 1, "mcts", 2, 7,
            progress_callback=events.append,
        )
        self.assertIn(result, (-1, 0, 1))
        self.assertEqual(events[-1]["move"], moves)

    def test_default_evaluation_frequency_and_save_before_eval(self):
        event_order = []
        state = GomokuState()
        state.done = True
        state.winner = 1
        policy = np.zeros(361, dtype=np.float32)
        policy[180] = 1
        sample = (GomokuState().encode(), policy, 1.0)

        def notify(callback):
            callback({
                "move": 1, "player": 1, "simulations_done": 2,
                "simulations_total": 2, "elapsed_seconds": 0.1,
                "reused_visits": 0, "network_calls": 2, "cache_hits": 0,
            })

        def fake_game(*args, **kwargs):
            notify(kwargs["progress_callback"])
            return [sample], state

        def fake_match(*args, **kwargs):
            event_order.append("eval")
            notify(kwargs["progress_callback"])
            return 1, 9

        def capture(checkpoint, path):
            event_order.append(("save", checkpoint["iteration"], checkpoint["evaluation"] is None))

        with (
            patch.object(training.Path, "mkdir"),
            patch.object(training.Path, "open", mock_open()),
            patch.object(training, "Live"),
            patch.object(training, "play_game", side_effect=fake_game),
            patch.object(training, "play_match", side_effect=fake_match) as matches,
            patch.object(training, "train_step", return_value={"loss": 2., "policy_loss": 1., "value_loss": 1.}),
            patch.object(training, "save_checkpoint", side_effect=capture),
            patch.object(training.torch.optim, "Adam"),
            patch("builtins.print"),
        ):
            training.train(iterations=2, simulations=2, updates_per_iteration=1, eval_games=2)
        self.assertEqual(matches.call_count, 2)
        self.assertEqual(event_order, [
            ("save", 1, True), ("save", 2, True),
            "eval", "eval", ("save", 2, False),
        ])

    def test_failed_checkpoint_write_keeps_previous_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            training.save_checkpoint({"value": 1}, path)
            before = path.read_bytes()

            def interrupted(checkpoint, temporary):
                temporary.write_bytes(b"incomplete")
                raise RuntimeError("simulated failed write")

            with patch.object(training.torch, "save", side_effect=interrupted):
                with self.assertRaises(RuntimeError):
                    training.save_checkpoint({"value": 2}, path)
            self.assertEqual(path.read_bytes(), before)
            self.assertFalse(path.with_suffix(".pt.tmp").exists())


if __name__ == "__main__":
    unittest.main()
