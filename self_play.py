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
):
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

        policy, _ = searcher.search(
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
