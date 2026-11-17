"""Review saved games the AI did not win, with a much deeper search.

In a lost game the AI's own policy targets come from the same shallow search
that made the mistake (e.g. 512 simulations chose J7 instead of blocking).
Training on them would repeat the mistake. Here every AI move of such a game
is searched again with REVIEW_SIMULATIONS (default 4096) and that result
becomes its policy target. Outcomes stay the real result. The reviewed game is
written next to the original as game_NNNN.review.npz (the original is never
changed) and load_human_games() uses it from then on.

Each game is reviewed once, by the model training starts from.
`python train.py` runs this automatically; standalone:
    python review.py [--checkpoint FILE] [--simulations 4096] [--workers 6]
"""
import argparse
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from human_games import ai_did_not_win, ai_positions, human_game_files, review_path, \
    state_from_observation
from mcts import MCTS
from network import GomokuNetwork

REVIEW_SIMULATIONS = 4096

_worker_model = None


def _init_worker(model_config, state_dict):
    global _worker_model
    torch.set_num_threads(1)
    _worker_model = GomokuNetwork(**model_config)
    _worker_model.load_state_dict(state_dict)
    _worker_model.eval()


def _search(observation, simulations):
    policy, _ = MCTS(_worker_model, simulations=simulations).search(
        state_from_observation(observation), add_noise=False)
    return policy.astype(np.float32)


def games_to_review(root):
    """Saved games (after the --fresh cutoff) the AI did not win and not reviewed yet."""
    pending = []
    for path in human_game_files(root)[0]:
        if review_path(path).exists():
            continue
        with np.load(path) as game:
            if ai_did_not_win(game) and ai_positions(game).any():
                pending.append(path)
    return pending


def review_games(root, checkpoint_path, workers=6, simulations=None, log=None):
    """Review every pending game. Returns the number of games reviewed."""
    simulations = REVIEW_SIMULATIONS if simulations is None else simulations
    log = log or (lambda message: print(message, flush=True))
    pending = games_to_review(root)
    if not pending:
        return 0
    games = []
    for path in pending:
        with np.load(path) as game:
            games.append((path, {key: game[key] for key in game.files}))
    jobs = [(g, int(i)) for g, (_, data) in enumerate(games)
            for i in np.flatnonzero(ai_positions(data))]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    log(f"Reviewing {len(games)} lost games ({len(jobs)} AI moves, {simulations} simulations "
        f"each, model {Path(checkpoint_path).parent.name}/{Path(checkpoint_path).name})...")
    initargs = (checkpoint["model_config"], checkpoint["model_state_dict"])
    if workers > 1:
        pool = ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"),
                                   initializer=_init_worker, initargs=initargs)
        results = pool.map(_search, (games[g][1]["observations"][i] for g, i in jobs),
                           [simulations] * len(jobs))
    else:
        pool = None
        _init_worker(*initargs)
        results = (_search(games[g][1]["observations"][i], simulations) for g, i in jobs)
    try:
        policies = {}
        for done, ((g, i), policy) in enumerate(zip(jobs, results), 1):
            policies[g, i] = policy
            if done % 25 == 0 or done == len(jobs):
                log(f"  reviewed {done}/{len(jobs)} moves")
    finally:
        if pool is not None:
            pool.shutdown()
    changed = 0
    for g, (path, data) in enumerate(games):
        reviewed = dict(data)
        reviewed["policies"] = data["policies"].copy()
        for i in np.flatnonzero(ai_positions(data)):
            old, new = int(np.argmax(reviewed["policies"][i])), int(np.argmax(policies[g, int(i)]))
            changed += old != new
            reviewed["policies"][i] = policies[g, int(i)]
        reviewed["review_simulations"] = np.int32(simulations)
        temporary = review_path(path).with_suffix(".tmp.npz")
        np.savez_compressed(temporary, **reviewed)
        temporary.replace(review_path(path))
    log(f"Review done: the deeper search prefers a different move in {changed} of "
        f"{len(jobs)} AI moves; those are now its training targets.")
    return len(games)


def main():
    from train import checkpoint_root, find_latest_checkpoint
    parser = argparse.ArgumentParser(description="Review saved games the AI did not win.")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Model used for the review (default: newest checkpoint)")
    parser.add_argument("--simulations", type=int, default=REVIEW_SIMULATIONS)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if args.simulations < 1 or args.workers < 1:
        parser.error("simulations and workers must be positive")
    checkpoint = args.checkpoint or find_latest_checkpoint()
    if not review_games(checkpoint_root(), checkpoint, args.workers, args.simulations):
        print("No lost games waiting for review.")


if __name__ == "__main__":
    main()
