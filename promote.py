"""Keep a "best" model separate from the "latest" one.

latest = the newest checkpoint file (training resumes from it).
best   = checkpoints/best.json, changed ONLY here, after a head-to-head match.

  python promote.py --set-baseline <file.pt>   # one-time: name the starting reference
  python promote.py                            # latest vs best; promote only if it wins clearly
  python promote.py --candidate <file.pt>      # a specific candidate vs best

Fair match: each game starts from a random 2-stone opening near the center
(otherwise two deterministic searchers would replay the same game), and every
opening is played twice with colors swapped. No search noise, win check on.
Every match is appended to checkpoints/promotions.jsonl, promoted or not.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import json
import math
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np
import torch

from evaluate import load_model
from mcts import MCTS
from self_play import select_action
from state import GomokuState

PROJECT = Path(__file__).resolve().parent
CHECKPOINT_ROOT = PROJECT / "checkpoints"


def best_file(root=CHECKPOINT_ROOT):
    return Path(root) / "best.json"


def read_best(root=CHECKPOINT_ROOT):
    """The best.json record, or None if no best has been set yet."""
    path = best_file(root)
    if not path.exists():
        return None
    record = json.loads(path.read_text())
    record["path"] = str((PROJECT / record["path"]).resolve()
                         if not Path(record["path"]).is_absolute() else Path(record["path"]))
    return record


def write_best(record, root=CHECKPOINT_ROOT):
    """Atomic replace, so best.json is never half-written."""
    path = best_file(root)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    os.replace(temporary, path)


def relative(path):
    path = Path(path).resolve()
    try:
        return str(path.relative_to(PROJECT))
    except ValueError:
        return str(path)


def random_opening(rng, stones=2, size=19, radius=4):
    """Distinct random cells in the central (2*radius+1)^2 square, as actions."""
    center = size // 2
    cells = [r * size + c
             for r in range(center - radius, center + radius + 1)
             for c in range(center - radius, center + radius + 1)]
    return [int(a) for a in rng.choice(cells, size=stones, replace=False)]


def play_from_opening(candidate, opponent, candidate_color, opening, simulations, seed):
    """+1 candidate win, 0 draw, -1 loss; and the number of moves."""
    state = GomokuState()
    for action in opening:
        state.step(action)
    rng = np.random.default_rng(seed)
    searchers = {candidate_color: MCTS(candidate, simulations=simulations),
                 -candidate_color: MCTS(opponent, simulations=simulations)}
    with torch.no_grad():
        while not state.done:
            policy, _ = searchers[state.current_player].search(state, add_noise=False)
            action = select_action(policy, rng, sample=False)
            state.step(action)
            for searcher in searchers.values():
                searcher.advance(action, state)
    return int(state.winner * candidate_color), int(np.count_nonzero(state.board))


def match_worker(candidate_path, opponent_path, candidate_color, opening, simulations, seed):
    torch.set_num_threads(1)
    candidate, _ = load_model(candidate_path)
    opponent, _ = load_model(opponent_path)
    return play_from_opening(candidate, opponent, candidate_color, opening, simulations, seed)


def decide(results, min_games, threshold):
    """Score = (wins + draws/2) / games; promote only with enough games AND a clear win."""
    games = len(results)
    wins, draws = results.count(1), results.count(0)
    score = (wins + 0.5 * draws) / games
    margin = 1.96 * math.sqrt(max(score * (1 - score), 1e-9) / games)
    return {
        "games": games, "wins": wins, "draws": draws, "losses": results.count(-1),
        "score": score, "ci95": [max(0.0, score - margin), min(1.0, score + margin)],
        "promoted": games >= min_games and score >= threshold,
    }


def run_match(candidate_path, opponent_path, games, simulations, workers, seed):
    rng = np.random.default_rng(seed)
    jobs = []
    for pair in range(games // 2):
        opening = random_opening(rng)
        for color in (1, -1):  # same opening, colors swapped
            jobs.append((candidate_path, opponent_path, color, opening, simulations,
                         int(rng.integers(0, 2**32))))
    if workers > 1:
        with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as pool:
            outcomes = list(pool.map(match_worker, *zip(*jobs)))
    else:
        outcomes = [match_worker(*job) for job in jobs]
    return [result for result, _ in outcomes], [moves for _, moves in outcomes]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set-baseline", type=Path, metavar="FILE",
                        help="Name FILE as the starting reference (marked: not evaluated)")
    parser.add_argument("--force", action="store_true",
                        help="Allow --set-baseline to replace an existing best")
    parser.add_argument("--candidate", type=Path, default=None,
                        help="Checkpoint to test (default: the latest checkpoint)")
    parser.add_argument("--games", type=int, default=40, help="Even number (default 40)")
    parser.add_argument("--simulations", type=int, default=512)
    parser.add_argument("--threshold", type=float, default=0.60,
                        help="Minimum score to promote (default 0.60)")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.set_baseline:
        if not args.set_baseline.exists():
            parser.error(f"{args.set_baseline} does not exist")
        load_model(args.set_baseline)  # must be a loadable checkpoint
        if read_best() and not args.force:
            parser.error("best.json already exists; add --force to replace it")
        write_best({"path": relative(args.set_baseline), "status": "baseline",
                    "note": "chosen by the user as the starting reference; not evaluated",
                    "time": f"{datetime.now():%Y-%m-%d %H:%M:%S}", "evaluation": None})
        print(f"best = {relative(args.set_baseline)} (baseline, not evaluated)")
        return

    if args.games < 2 or args.games % 2:
        parser.error("--games must be an even number >= 2")
    best = read_best()
    if best is None:
        parser.error("No best yet. First run: python promote.py --set-baseline <file.pt>")
    if args.candidate is None:
        from web_app.server import find_latest_checkpoint
        args.candidate = find_latest_checkpoint()
    candidate, opponent = args.candidate.resolve(), Path(best["path"]).resolve()
    if candidate == opponent:
        print("The candidate already is the best model; nothing to do.")
        return

    print(f"candidate: {relative(candidate)}")
    print(f"best:      {relative(opponent)} ({best['status']})")
    print(f"{args.games} games, {args.simulations} simulations, {args.workers} workers...", flush=True)
    results, lengths = run_match(str(candidate), str(opponent), args.games,
                                 args.simulations, args.workers, args.seed)
    verdict = decide(results, min_games=40, threshold=args.threshold)
    record = {"time": f"{datetime.now():%Y-%m-%d %H:%M:%S}", "candidate": relative(candidate),
              "best": relative(opponent), "simulations": args.simulations,
              "threshold": args.threshold, "average_moves": float(np.mean(lengths)), **verdict}
    with (CHECKPOINT_ROOT / "promotions.jsonl").open("a", encoding="utf-8") as file:
        file.write(json.dumps(record) + "\n")

    low, high = verdict["ci95"]
    print(f"W/D/L {verdict['wins']}/{verdict['draws']}/{verdict['losses']}, "
          f"score {verdict['score']:.1%} (95% range {low:.0%}-{high:.0%}), "
          f"average {record['average_moves']:.0f} moves per game")
    if verdict["promoted"]:
        write_best({"path": relative(candidate), "status": "promoted",
                    "previous": relative(opponent), "time": record["time"],
                    "evaluation": verdict})
        print(f"PROMOTED: best = {relative(candidate)}")
    else:
        need = "at least 40 games" if verdict["games"] < 40 else f"score >= {args.threshold:.0%}"
        print(f"Not promoted (needs {need}). best stays {relative(opponent)}")


if __name__ == "__main__":
    main()
