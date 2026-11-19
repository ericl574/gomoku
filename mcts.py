import math
import time
from collections import OrderedDict

import numpy as np
import torch

from network import masked_policy


class TreeNode:
    __slots__ = ("base_prior", "prior", "visit_count", "value_sum", "children")

    def __init__(self, prior=1.0):
        self.base_prior = float(prior)
        self.prior = self.base_prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.children = {}

    @property
    def q(self):
        if self.visit_count == 0:
            return 0.0

        return self.value_sum / self.visit_count


class MCTS:
    def __init__(
        self,
        model,
        simulations=32,
        c_puct=1.5,
        dirichlet_alpha=0.03,
        noise_fraction=0.25,
        seed=None,
        cache_size=0,
        win_check=True,
        fpu_reduction=0.2,
    ):
        if (
            isinstance(simulations, bool)
            or not isinstance(simulations, (int, np.integer))
            or simulations < 1
        ):
            raise ValueError("Simulations must be a positive integer")

        if not math.isfinite(c_puct) or c_puct <= 0:
            raise ValueError("c_puct must be positive and finite")

        if not math.isfinite(dirichlet_alpha) or dirichlet_alpha <= 0:
            raise ValueError("Dirichlet alpha must be positive and finite")

        if not 0 <= noise_fraction <= 1:
            raise ValueError("Noise fraction must be between 0 and 1")
        if not math.isfinite(fpu_reduction) or fpu_reduction < 0:
            raise ValueError("FPU reduction must be nonnegative and finite")
        # 0.2 is KataGo's value (Wu 2019, "Accelerating Self-Play Learning in Go").
        self.fpu_reduction = float(fpu_reduction)
        if (
            isinstance(cache_size, bool)
            or not isinstance(cache_size, int)
            or cache_size < 0
        ):
            raise ValueError("Cache size must be a nonnegative integer")

        self.model = model
        self.simulations = int(simulations)
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.noise_fraction = noise_fraction
        self.rng = np.random.default_rng(seed)
        self.cache_size = cache_size
        # Win-in-one: a leaf whose player to move can make five right now is a
        # certain win (+1), decided by the rules alone, without the network.
        # Loss-in-two: a leaf whose player to move cannot make five but faces
        # two or more different five-making cells (open four, two fours) is a
        # certain loss (-1), so an open four is never mistaken for a fair position.
        self.win_check = bool(win_check)
        self._cache = OrderedDict()
        self._root = None
        self._root_key = None
        self._root_state = None
        self.last_search_stats = {}

    @staticmethod
    def _state_key(state):
        return (
            state.size,
            state.board.tobytes(),
            state.current_player,
            state.done,
            state.winner,
        )

    def reset(self):
        """Discard cached results; required if this searcher's model changes."""
        self._root = None
        self._root_key = None
        self._root_state = None
        self._cache.clear()
        self.last_search_stats = {}

    def advance(self, action, state):
        """Retain the chosen subtree after a real move, checking the position."""
        child = None
        if self._root is not None and self._root_state is not None:
            expected = self._root_state.clone()
            try:
                expected.step(action)
            except ValueError:
                pass
            else:
                if self._state_key(expected) == self._state_key(state):
                    child = self._root.children.get(int(action))

        self._root = child
        self._root_key = self._state_key(state) if child is not None else None
        self._root_state = state.clone() if child is not None else None

    @torch.no_grad()
    def search(self, state, add_noise=False, progress_callback=None):
        if state.done:
            raise ValueError("Cannot search a terminal state")

        was_training = self.model.training
        self.model.eval()

        try:
            key = self._state_key(state)
            if self._root is None or self._root_key != key:
                self._root = TreeNode()
                self._root_key = key
                self._root_state = state.clone()
            root = self._root
            self.last_search_stats = {
                "reused_visits": root.visit_count,
                "network_calls": 0,
                "cache_hits": 0,
                "win_in_one": 0,
                "loss_in_two": 0,
            }
            if progress_callback is not None:
                progress_callback(0, self.simulations)
            if not root.children:
                self._evaluate(state, root)
            if self.win_check:
                # With loss-in-two, a move that keeps an open four also scores
                # +1, so make five now: only five-making moves stay at the root.
                wins = set(state.winning_moves().tolist())
                if wins:
                    root.children = {a: c for a, c in root.children.items() if a in wins}

            # Never mix fresh root noise with noise left over from an earlier search.
            for child in root.children.values():
                child.prior = child.base_prior

            if add_noise:
                children = list(root.children.values())
                noise = self.rng.dirichlet(
                    np.full(len(children), self.dirichlet_alpha)
                )

                for child, noise_prior in zip(children, noise):
                    child.prior = (
                        (1 - self.noise_fraction) * child.prior
                        + self.noise_fraction * noise_prior
                    )

            next_report = time.perf_counter() + 0.2
            for simulation in range(self.simulations):
                simulation_state = state.clone()
                node = root
                path = [node]

                while node.children and not simulation_state.done:
                    # KataGo: no FPU reduction at a noised root, so the
                    # Dirichlet noise can still steer exploration there.
                    action, node = self._select(
                        node, 0.0 if (add_noise and node is root) else None)
                    simulation_state.step(action)
                    path.append(node)

                if self.win_check and simulation_state.has_immediate_win():
                    # Left unexpanded on purpose: revisits re-check and get +1 again.
                    value = 1.0
                    self.last_search_stats["win_in_one"] += 1
                elif self.win_check and simulation_state.loses_in_two():
                    # Opponent has an open four (or two fours): certain loss.
                    value = -1.0
                    self.last_search_stats["loss_in_two"] += 1
                else:
                    value = self._evaluate(simulation_state, node)
                self._backup(path, value)

                completed = simulation + 1
                if progress_callback is not None and (
                    completed == self.simulations
                    or time.perf_counter() >= next_report
                ):
                    progress_callback(completed, self.simulations)
                    next_report = time.perf_counter() + 0.2

            policy = np.zeros(state.size * state.size, dtype=np.float32)

            for action, child in root.children.items():
                policy[action] = child.visit_count

            # Retained visits remain part of the search policy; the loop above
            # always adds exactly self.simulations NEW simulations.
            policy /= policy.sum()

            return policy, root
        finally:
            self.model.train(was_training)

    def _select(self, node, fpu_reduction=None):
        """PUCT with first-play urgency (FPU) reduction, as in Leela Zero / KataGo.

        An unvisited child is valued at the parent's value minus
        fpu_reduction * sqrt(prior mass of already-visited children), i.e.
        "probably a bit worse than what we have", not 0 (a draw). With 0, a
        side that thinks it is losing (visited q < 0) rates every unvisited
        move above its real candidates and spreads its search over the whole
        board, one visit each.
        """
        if fpu_reduction is None:
            fpu_reduction = self.fpu_reduction
        exploration_scale = self.c_puct * math.sqrt(node.visit_count + 1)

        visited_prior = 0.0
        for child in node.children.values():
            if child.visit_count:
                visited_prior += child.prior
        # node.q is from the mover's side at `node`, the same side as -child.q.
        # Values live in [-1, 1]: an unexplored move is never rated below a
        # certain loss, so when every tried move loses, untried ones get looked at.
        fpu_value = max(-1.0, node.q - fpu_reduction * math.sqrt(visited_prior))

        best_item = None
        best_score = -math.inf
        for item in node.children.items():
            child = item[1]
            if child.visit_count:
                score = -child.q + exploration_scale * child.prior / (1 + child.visit_count)
            else:
                score = fpu_value + exploration_scale * child.prior
            # First maximum wins, like max(); with NaN scores (a broken network)
            # the first child is returned, as before, and callers detect the NaN.
            if best_item is None or score > best_score:
                best_item = item
                best_score = score
        return best_item

    def _evaluate(self, state, node):
        if state.done:
            return state.terminal_value()

        key = self._state_key(state) if self.cache_size else None
        cached = self._cache.get(key) if self.cache_size else None
        if cached is not None:
            self._cache.move_to_end(key)
            probabilities, value = cached
            self.last_search_stats["cache_hits"] = (
                self.last_search_stats.get("cache_hits", 0) + 1
            )
        else:
            device = next(self.model.parameters()).device
            observation = torch.from_numpy(state.encode())
            observation = observation.unsqueeze(0).to(device)
            logits, predicted_value = self.model(observation)
            legal_indices = torch.as_tensor(
                state.legal_actions(),
                dtype=torch.long,
                device=device,
            )
            legal_mask = torch.zeros_like(logits, dtype=torch.bool)
            legal_mask[0, legal_indices] = True
            probabilities = masked_policy(logits, legal_mask)
            probabilities = probabilities[0].cpu().numpy()
            value = predicted_value.item()
            self.last_search_stats["network_calls"] = (
                self.last_search_stats.get("network_calls", 0) + 1
            )
            if self.cache_size:
                probabilities = probabilities.copy()
                probabilities.setflags(write=False)
                self._cache[key] = (probabilities, value)
                if len(self._cache) > self.cache_size:
                    self._cache.popitem(last=False)

        for action in state.legal_actions():
            action = int(action)
            node.children[action] = TreeNode(probabilities[action])

        return value

    def _backup(self, path, value):
        for node in reversed(path):
            node.visit_count += 1
            node.value_sum += value
            value = -value
