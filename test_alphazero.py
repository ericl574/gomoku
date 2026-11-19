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
            root.value_sum = float(rng.uniform(-1, 1) * root.visit_count)
            visited = sum(c.prior for c in root.children.values() if c.visit_count)
            fpu = root.q - searcher.fpu_reduction * np.sqrt(visited)
            expected = max(
                root.children.items(),
                key=lambda item: (
                    (-item[1].q if item[1].visit_count else fpu)
                    + searcher.c_puct * item[1].prior
                    * np.sqrt(root.visit_count + 1)
                    / (1 + item[1].visit_count)
                ),
            )
            self.assertEqual(searcher._select(root)[0], expected[0])

    def test_losing_side_does_not_spread_search_to_unvisited_moves(self):
        # Mover thinks it is losing: every visited reply scores about -0.6.
        # With unvisited moves valued 0 (old behaviour), the next visit went to
        # an unvisited low-prior move; with FPU it stays on a real candidate.
        searcher = MCTS(CountingNetwork())
        root = TreeNode()
        root.visit_count, root.value_sum = 100, -60.0
        for action in range(300):
            root.children[action] = TreeNode(0.5 if action < 2 else 0.5 / 1000)
        for action in (0, 1):
            root.children[action].visit_count = 50
            root.children[action].value_sum = 0.6 * 50  # child side: +0.6
        self.assertIn(searcher._select(root)[0], (0, 1))
        old = max(root.children.items(), key=lambda item: (
            -item[1].q + searcher.c_puct * item[1].prior
            * np.sqrt(root.visit_count + 1) / (1 + item[1].visit_count)))
        self.assertNotIn(old[0], (0, 1))  # the old rule did spread

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


class FlatNetwork(nn.Module):
    """Knows nothing: equal priors everywhere, value 0. Isolates what the search finds."""

    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        return torch.zeros(len(x), 361) + self.anchor, torch.zeros(len(x), 1) + self.anchor


def brute_force_immediate_win(state):
    for action in state.legal_actions():
        trial = state.clone()
        trial.step(int(action))
        if trial.winner == state.current_player:
            return True
    return False


class WinInOneTests(unittest.TestCase):
    def test_fast_check_matches_brute_force_on_random_boards(self):
        rng = np.random.default_rng(5)
        checked = wins = 0
        for _ in range(300):
            state = GomokuState()
            # Dense, clustered random games produce many near-five shapes.
            for _ in range(int(rng.integers(8, 60))):
                legal = [int(a) for a in state.legal_actions()
                         if abs(a // 19 - 9) <= 4 and abs(a % 19 - 9) <= 4]
                if state.done or not legal:
                    break
                state.step(int(rng.choice(legal)))
            if state.done:
                continue
            expected = brute_force_immediate_win(state)
            self.assertEqual(state.has_immediate_win(), expected)
            checked += 1
            wins += expected
        self.assertGreater(checked, 200)
        self.assertGreater(wins, 20)  # the test really exercised winning positions

    def test_five_or_more_and_board_edges(self):
        state = GomokuState()
        state.board[0, 0:4] = 1          # four at the edge: win at (0, 4)
        self.assertTrue(state.has_immediate_win())
        state.board[0, 4] = -1           # blocked: no win
        self.assertFalse(state.has_immediate_win())
        state.board[5, [2, 3, 5, 6]] = 1  # split four X X . X X: filling the gap wins
        self.assertTrue(state.has_immediate_win())
        state.current_player = -1        # but not for white
        self.assertFalse(state.has_immediate_win())

    def _closed_four_position(self):
        state = GomokuState()
        state.board[9, 5:9] = 1      # black four, left end blocked by white
        state.board[9, 4] = -1
        state.board[3, 3] = -1
        state.board[15, 15] = -1
        state.current_player = -1    # white must block at (9, 9)
        return state

    def test_search_blocks_a_four_it_cannot_see_without_the_check(self):
        block = 9 * 19 + 9
        with_check = MCTS(FlatNetwork(), simulations=400, win_check=True)
        policy, _ = with_check.search(self._closed_four_position())
        self.assertEqual(int(np.argmax(policy)), block)
        self.assertGreater(policy[block], 0.05)
        self.assertGreater(with_check.last_search_stats["win_in_one"], 0)

        without = MCTS(FlatNetwork(), simulations=400, win_check=False)
        policy, _ = without.search(self._closed_four_position())
        self.assertLess(policy[block], 0.05)  # the old search spreads ~1 visit per move

    def _own_win_position(self):
        state = GomokuState()
        state.board[9, 5:9] = 1
        state.board[3, 3:6] = -1
        return state

    def test_own_win_is_played_now(self):
        wins = [9 * 19 + 4, 9 * 19 + 9]
        # Keeping the open four would also be a certain win (loss-in-two), so
        # the root keeps only the moves that make five now, even for a
        # know-nothing network, and the search policy is only on them.
        for fpu in (0.0, 0.2):
            policy, _ = MCTS(FlatNetwork(), simulations=400, fpu_reduction=fpu).search(
                self._own_win_position())
            self.assertEqual(np.flatnonzero(policy).tolist(), wins)
        self.assertEqual(self._own_win_position().winning_moves().tolist(), wins)

    def test_winning_moves_match_brute_force(self):
        rng = np.random.default_rng(7)
        for _ in range(200):
            state = GomokuState()
            for _ in range(int(rng.integers(8, 60))):
                legal = [int(a) for a in state.legal_actions()
                         if abs(a // 19 - 9) <= 4 and abs(a % 19 - 9) <= 4]
                if state.done or not legal:
                    break
                state.step(int(rng.choice(legal)))
            if state.done:
                continue
            expected = []
            for action in state.legal_actions():
                trial = state.clone()
                trial.step(int(action))
                if trial.winner == state.current_player:
                    expected.append(int(action))
            self.assertEqual(state.winning_moves().tolist(), expected)


def brute_force_loses_in_two(state):
    """Every move of the player to move lets the opponent make five next."""
    if brute_force_immediate_win(state):
        return False
    for action in state.legal_actions():
        trial = state.clone()
        trial.step(int(action))
        if not trial.done and not trial.has_immediate_win():
            return False
    return True


class LossInTwoTests(unittest.TestCase):
    def test_fast_check_matches_brute_force_on_random_boards(self):
        rng = np.random.default_rng(11)
        checked = losses = 0
        for _ in range(300):
            state = GomokuState()
            for _ in range(int(rng.integers(8, 60))):
                legal = [int(a) for a in state.legal_actions()
                         if abs(a // 19 - 9) <= 4 and abs(a % 19 - 9) <= 4]
                if state.done or not legal:
                    break
                state.step(int(rng.choice(legal)))
            if state.done:
                continue
            expected = brute_force_loses_in_two(state)
            self.assertEqual(state.loses_in_two(), expected)
            checked += 1
            losses += expected
        self.assertGreater(checked, 200)
        self.assertGreater(losses, 5)  # the test really exercised lost positions

    def test_open_four_split_fours_and_a_single_four(self):
        state = GomokuState()
        state.board[9, 6:10] = 1           # black open four, white to move
        state.current_player = -1
        self.assertTrue(state.loses_in_two())
        state.board[9, 5] = -1             # one end blocked: only (9, 10) left
        self.assertFalse(state.loses_in_two())
        state.board[3, [2, 3, 5, 6]] = 1   # plus a split four X X . X X: two cells
        self.assertTrue(state.loses_in_two())
        state.board[12, 2:6] = -1          # but white can make five first
        self.assertFalse(state.loses_in_two())

    def test_search_makes_the_open_four(self):
        state = GomokuState()
        state.board[9, 6:9] = 1            # black open three, black to move
        state.board[[2, 4, 15], [2, 15, 4]] = -1
        open_fours = {9 * 19 + 5, 9 * 19 + 9}
        search = MCTS(FlatNetwork(), simulations=400, fpu_reduction=0.0)
        policy, _ = search.search(state)
        self.assertIn(int(np.argmax(policy)), open_fours)
        self.assertGreater(search.last_search_stats["loss_in_two"], 0)


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
            # resume_state has no "evaluation" key; it counts as an unevaluated save.
            event_order.append(("save", checkpoint["iteration"], checkpoint.get("evaluation") is None))

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
            ("save", 1, True), ("save", 1, True),
            ("save", 2, True), "eval", "eval", ("save", 2, False),
            ("save", 2, True),
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
