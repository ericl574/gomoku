import argparse
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from mcts import MCTS
from network import GomokuNetwork, masked_policy
from self_play import select_action
from state import GomokuState


def load_model(path):
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    model = GomokuNetwork(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint


def policy_for_state(model, state):
    device = next(model.parameters()).device

    observation = torch.from_numpy(state.encode())
    observation = observation.unsqueeze(0).to(device)

    logits, _ = model(observation)

    legal_indices = torch.as_tensor(
        state.legal_actions(),
        dtype=torch.long,
        device=device,
    )

    legal_mask = torch.zeros_like(logits, dtype=torch.bool)
    legal_mask[0, legal_indices] = True

    return masked_policy(logits, legal_mask)[0].cpu().numpy()


@torch.no_grad()
def play_match(
    candidate,
    opponent,
    candidate_color,
    mode,
    simulations,
    seed,
    progress_callback=None,
    reuse_tree=True,
):
    state = GomokuState()
    rng = np.random.default_rng(seed)

    models = {
        candidate_color: candidate,
        -candidate_color: opponent,
    }

    searchers = {}

    if mode == "mcts":
        searchers = {
            color: MCTS(model, simulations=simulations)
            for color, model in models.items()
        }

    moves = 0
    started = time.perf_counter()

    while not state.done:
        player = state.current_player

        if mode == "mcts":
            searcher = searchers[player]

            def report_search(completed, total):
                if progress_callback is not None:
                    progress_callback({
                        "move": moves + 1,
                        "player": player,
                        "simulations_done": completed,
                        "simulations_total": total,
                        "elapsed_seconds": time.perf_counter() - started,
                        **searcher.last_search_stats,
                    })

            policy, _ = searchers[player].search(
                state,
                add_noise=False,
                progress_callback=report_search if progress_callback else None,
            )
        elif mode == "policy":
            policy = policy_for_state(models[player], state)
        else:
            raise ValueError("Unknown evaluation mode")

        action = select_action(policy, rng, sample=False)
        state.step(action)
        for searcher in searchers.values():
            if reuse_tree:
                searcher.advance(action, state)
            else:
                searcher.reset()
        moves += 1

    result = state.winner * candidate_color

    return int(result), moves


def evaluate(
    candidate,
    opponent,
    mode,
    games=4,
    simulations=8,
    seed=12345,
    reuse_tree=True,
):
    if games < 2 or games % 2 != 0:
        raise ValueError("Games must be a positive even number")

    candidate.eval()
    opponent.eval()

    rng = np.random.default_rng(seed)
    records = []

    for game in tqdm(
        range(games),
        desc=f"Evaluation: {mode}",
        unit="game",
    ):
        candidate_color = 1 if game % 2 == 0 else -1
        game_seed = int(rng.integers(0, 2**32))

        result, moves = play_match(
            candidate,
            opponent,
            candidate_color,
            mode,
            simulations,
            game_seed,
            reuse_tree=reuse_tree,
        )

        records.append(
            {
                "color": candidate_color,
                "result": result,
                "moves": moves,
                "seed": game_seed,
            }
        )

        color_name = "black" if candidate_color == 1 else "white"
        result_name = {
            1: "win",
            0: "draw",
            -1: "loss",
        }[result]

        tqdm.write(
            f"Game {game + 1}/{games}: "
            f"candidate={color_name}, result={result_name}, "
            f"moves={moves}, seed={game_seed}"
        )

    print(f"\nMode: {mode}")

    for label, color in (
        ("All", None),
        ("As black", 1),
        ("As white", -1),
    ):
        subset = [
            record
            for record in records
            if color is None or record["color"] == color
        ]

        wins = sum(record["result"] == 1 for record in subset)
        draws = sum(record["result"] == 0 for record in subset)
        losses = sum(record["result"] == -1 for record in subset)
        count = len(subset)

        print(
            f"{label}: W/D/L={wins}/{draws}/{losses}, "
            f"win_rate={wins / count:.1%}, "
            f"score={(wins + 0.5 * draws) / count:.1%}"
        )

    return records


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--opponent", type=Path, default=None)
    parser.add_argument("--games", type=int, default=4)
    parser.add_argument("--simulations", type=int, default=8)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--no-tree-reuse",
        action="store_true",
        help="Rebuild the search tree each move for comparison with older runs",
    )
    parser.add_argument(
        "--mode",
        choices=("mcts", "policy", "both"),
        default="both",
    )

    args = parser.parse_args()

    if args.games < 2 or args.games % 2 != 0:
        parser.error("--games must be an even number >= 2")

    if args.simulations < 1:
        parser.error("--simulations must be >= 1")

    candidate, checkpoint = load_model(args.checkpoint)

    if args.opponent is None:
        if checkpoint.get("init_checkpoint") is not None:
            parser.error("This run started from a checkpoint; specify --opponent explicitly")
        if checkpoint.get("runtime_config", {}).get("mode") == "web_learning":
            parser.error("Web-learning checkpoint: specify --opponent explicitly")
        if "seed" not in checkpoint:
            parser.error("Checkpoint has no initial-model seed; specify --opponent explicitly")
        # train() draws the initial model after torch.manual_seed(seed).
        # No random torch operations may be inserted before that initialization.
        torch.manual_seed(checkpoint["seed"])
        opponent = GomokuNetwork(**checkpoint["model_config"])
        opponent.eval()
        opponent_name = "Reconstructed random initial network (same seed and architecture)"
    else:
        opponent, _ = load_model(args.opponent)
        opponent_name = str(args.opponent)

    print("Candidate:", args.checkpoint)
    print("Opponent:", opponent_name)
    print("Games per mode:", args.games)
    print("MCTS simulations per move:", args.simulations)
    print("Evaluation seed:", args.seed)
    print("Tree reuse:", not args.no_tree_reuse)

    modes = ("mcts", "policy") if args.mode == "both" else (args.mode,)

    for mode in modes:
        evaluate(
            candidate,
            opponent,
            mode,
            games=args.games,
            simulations=args.simulations,
            seed=args.seed,
            reuse_tree=not args.no_tree_reuse,
        )


if __name__ == "__main__":
    main()
