"""Read-only 19x19 tactical diagnosis. Never used as training samples.

Tests are intentionally narrow: a unique single-square block and a forced win.
A double-ended four cannot be saved by a single block; do not mark such cases solved.
"""
import argparse
import numpy as np
import torch
from evaluate import load_model, policy_for_state
from mcts import MCTS
from state import GomokuState


def scenarios():
    # Position built by directly specifying a valid alternating-move board;
    # the stone counts and nonterminal nature are checked below.
    a = GomokuState()
    a.board[9, 5:9] = 1
    a.board[9, 4] = -1
    a.board[3, 3] = -1
    a.board[15, 15] = -1
    a.current_player = -1
    # Black's only immediate winning cell is 9,9. White has no immediate win.
    yield "white must block black", a, 9 * 19 + 9

    b = GomokuState()
    b.board[9, 5:9] = 1
    b.board[3, 3] = -1
    b.board[15, 15] = -1
    b.board[3, 15] = -1
    b.board[15, 3] = -1
    b.current_player = 1
    # Two winning ends exist; either is valid.
    yield "black wins now", b, {9 * 19 + 4, 9 * 19 + 9}

    c = GomokuState()
    c.board[9, 5:9] = 1
    c.board[12, 4] = 1  # blocks the left end of white's four
    c.board[12, 5:9] = -1
    c.current_player = -1
    # Both sides threaten to win. White should take its own immediate win.
    yield "white win despite black threat", c, 12 * 19 + 9


def check_scenario(name, state, accepted):
    own = int(np.count_nonzero(state.board == state.current_player))
    opponent = int(np.count_nonzero(state.board == -state.current_player))
    if abs(own - opponent) > 1 or state.done:
        raise ValueError(f"{name}: invalid turn/board")
    accepted = {accepted} if isinstance(accepted, int) else set(accepted)
    for a in accepted:
        if a not in state.legal_actions():
            raise ValueError(f"{name}: answer is illegal")
    if name.startswith("white must"):
        # Verify black has exactly one winning cell and that blocking removes it.
        trial = state.clone()
        trial.current_player = 1
        threats = []
        for action in trial.legal_actions():
            next_state = trial.clone()
            next_state.step(int(action))
            if next_state.winner == 1:
                threats.append(int(action))
        if set(threats) != accepted:
            raise ValueError("Forced block answer is not uniquely validated")
    else:
        wins = set()
        for action in state.legal_actions():
            next_state = state.clone()
            next_state.step(int(action))
            if next_state.winner == state.current_player:
                wins.add(int(action))
        if wins != accepted:
            raise ValueError("Immediate win answers do not match rules")
    return accepted


@torch.no_grad()
def diagnose(model, simulations):
    for name, state, answer in scenarios():
        accepted = check_scenario(name, state, answer)
        policy_only = policy_for_state(model, state)
        search = MCTS(model, simulations=simulations, seed=0)
        visits, root = search.search(state, add_noise=False)
        choice = int(np.argmax(visits))
        print(f"{name} | {simulations} sims | chosen {divmod(choice, 19)} | "
              f"PASS={choice in accepted} | raw policy argmax {divmod(int(policy_only.argmax()), 19)}")
        for action in sorted(accepted):
            child = root.children[action]
            print(f"  required {divmod(action,19)}: NN prior {policy_only[action]:.4f}, "
                  f"MCTS visits {child.visit_count}, root-policy {visits[action]:.4f}, "
                  f"Q for mover {-child.q:+.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--simulations', type=int, nargs='+', default=[128, 512])
    args = parser.parse_args()
    model, _ = load_model(args.checkpoint)
    for simulations in args.simulations:
        if simulations < 1:
            parser.error('simulations must be positive')
        diagnose(model, simulations)


if __name__ == '__main__':
    main()
