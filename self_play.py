import time

import numpy as np
import torch

from mcts import MCTS
from network import GomokuNetwork
from state import GomokuState


def select_action(policy, rng, sample=True):
    probabilities = np.asarray(policy, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()

    if sample:
        return int(
            rng.choice(len(probabilities), p=probabilities)
        )

    best_actions = np.flatnonzero(
        probabilities == probabilities.max()
    )
    return int(rng.choice(best_actions))


def play_game(
    model,
    simulations=8,
    exploration_moves=30,
    seed=None,
    progress_callback=None,
    move_log=None,
):
    """move_log: optional list; one dict per move is appended (for game logs)."""
    if (
        isinstance(exploration_moves, bool)
        or not isinstance(exploration_moves, (int, np.integer))
        or exploration_moves < 0
    ):
        raise ValueError("Exploration moves must be a nonnegative integer")

    rng = np.random.default_rng(seed)

    searcher = MCTS(
        model,
        simulations=simulations,
        seed=int(rng.integers(0, 2**32)),
    )

    state = GomokuState()
    history = []
    started = time.perf_counter()

    while not state.done:
        def report_search(completed, total):
            if progress_callback is not None:
                progress_callback({
                    "move": len(history) + 1,
                    "player": state.current_player,
                    "simulations_done": completed,
                    "simulations_total": total,
                    "elapsed_seconds": time.perf_counter() - started,
                    **searcher.last_search_stats,
                })

        policy, root = searcher.search(
            state,
            add_noise=True,
            progress_callback=report_search if progress_callback else None,
        )

        history.append(
            (
                state.encode(),
                policy.copy(),
                state.current_player,
            )
        )

        action = select_action(
            policy,
            rng,
            sample=len(history) <= exploration_moves,
        )
        if move_log is not None:
            # What the search saw; q is from the mover's side (-child.q, as in _select).
            top = sorted(root.children.items(), key=lambda kv: kv[1].visit_count,
                         reverse=True)[:5]
            move_log.append({
                "player": int(state.current_player),
                "action": int(action),
                "value": root.q,
                "visited": sum(c.visit_count > 0 for c in root.children.values()),
                "candidates": len(root.children),
                "win_in_one": searcher.last_search_stats.get("win_in_one", 0),
                "top": [(int(a), c.visit_count, c.base_prior, -c.q) for a, c in top],
            })
        state.step(action)
        searcher.advance(action, state)

    samples = []

    for observation, policy, player in history:
        outcome = float(state.winner * player)
        samples.append((observation, policy, outcome))

    return samples, state


def main():
    torch.manual_seed(0)

    model = GomokuNetwork(
        policy_kernel=3,
        architecture="resnet_global",
        residual_blocks=3,
    )
    model.eval()

    started = time.perf_counter()

    samples, final_state = play_game(
        model,
        simulations=8,
        exploration_moves=30,
        seed=0,
    )

    elapsed = time.perf_counter() - started

    result = {
        1: "Black wins",
        -1: "White wins",
        0: "Draw",
    }

    print("Result:", result[final_state.winner])
    print("Moves:", len(samples))
    print(f"Elapsed: {elapsed:.2f} seconds")

    observation, policy, outcome = samples[0]

    print("First observation shape:", observation.shape)
    print("First policy shape:", policy.shape)
    print("First policy sum:", float(policy.sum()))
    print("First outcome:", outcome)


if __name__ == "__main__":
    main()
