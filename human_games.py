"""Games played against a human in web learning mode, saved as
checkpoints/<time>_web_*/game_NNNN.npz. Shared by web learning and self-play.

Each sample is (board encoding, policy target, outcome for the side that moved):
the AI's own MCTS visits for its moves, and (with --imitate) the human's move
scaled to the imitation weight for the human's moves.

Games the AI did not win are reviewed once by review.py (a much deeper search
re-labels the AI's moves) into game_NNNN.review.npz, which is used instead.
Their positions are also starting points for some self-play games (train.py).
"""
from pathlib import Path

import numpy as np

import state as game_rules


CUTOFF_FILE = "web_buffer_start.txt"  # written by `--learn --fresh`


def normalized_web_game(observations, policies, outcomes):
    """Return validated, current-format samples; leave the saved NPZ unchanged.

    Old web games used a color plane. The current network uses COLOR_PLANE=False.
    Never normalize the two stone planes or rewrite the original game archive.
    """
    observations = np.asarray(observations, dtype=np.float32)
    policies = np.asarray(policies, dtype=np.float32)
    outcomes = np.asarray(outcomes, dtype=np.float32)
    n = len(observations)
    if observations.shape != (n, 3, 19, 19) or policies.shape != (n, 361) or outcomes.shape != (n,):
        raise ValueError("Invalid saved web game sample shapes")
    if not (np.isfinite(observations).all() and np.isfinite(policies).all() and np.isfinite(outcomes).all()):
        raise ValueError("Non-finite saved web game samples")
    if not (np.isin(observations, (0.0, 1.0)).all()
            and np.all((observations[:, 0] + observations[:, 1]) <= 1)):
        raise ValueError("Invalid saved board encoding")
    if (np.any(policies < 0) or np.any(policies.sum(axis=1) > 1 + 1e-3) or np.any(policies.sum(axis=1) <= 0)
            or np.any(np.abs(outcomes) > 1)):
        # Human imitation targets may have a policy sum below 1, so check those separately.
        raise ValueError("Invalid saved policy or outcome")
    normalized = observations.copy()
    if not game_rules.COLOR_PLANE:
        normalized[:, 2] = 0.0
    return list(zip(normalized, policies.copy(), outcomes.tolist()))


def read_cutoff(root):
    """Run-name prefix before which games are ignored ("" = keep all)."""
    path = Path(root) / CUTOFF_FILE
    return path.read_text().strip() if path.exists() else ""


def save_web_game(path, records, winner, ai_player, moves):
    """Save one finished game against a human.

    records: (board encoding, policy target, player who moved) per example.
    Returns the training samples (outcome from each mover's side).
    """
    samples = [(obs, policy, float(winner * player)) for obs, policy, player in records]
    np.savez_compressed(
        path,
        observations=np.stack([s[0] for s in samples]),
        policies=np.stack([s[1] for s in samples]),
        outcomes=np.array([s[2] for s in samples], dtype=np.float32),
        players=np.array([player for _, _, player in records], dtype=np.int8),
        ai_player=np.int8(ai_player), winner=np.int8(winner),
        moves=np.array(moves, dtype=np.int16),
    )
    return samples


def ai_positions(game):
    """Which saved positions were AI moves (their policy is the AI's search).

    Newer files store who moved; in older files the AI's search policy sums to
    1 and an imitated human move sums to the imitation weight (< 1).
    """
    if "players" in game:
        return game["players"] == int(game["ai_player"])
    return game["policies"].sum(axis=1) > 0.99


def ai_did_not_win(game):
    if "winner" in game:
        return int(game["winner"]) != int(game["ai_player"])
    ai = ai_positions(game)
    return bool(ai.any()) and float(game["outcomes"][ai][0]) <= 0


def review_path(path):
    """game_0007.npz -> game_0007.review.npz"""
    return Path(path).with_name(Path(path).stem + ".review.npz")


def human_game_files(root):
    """(kept, skipped) saved game files, oldest first, honouring the --fresh cutoff.

    Run names start with their start time, so comparing name prefixes compares times.
    """
    cutoff = read_cutoff(root)
    kept, skipped = [], 0
    for path in sorted(Path(root).glob("*_web_*/game_*.npz")):
        if ".review" in path.name:  # review.py output (or its temporary file)
            continue
        if path.parent.name[:len(cutoff)] < cutoff:
            skipped += 1
        else:
            kept.append(path)
    return kept, skipped


def load_human_games(root):
    """All saved human games under root, oldest first, honouring the --fresh cutoff.

    A reviewed game (review.py) is loaded from its .review.npz instead.
    Returns (samples, games_loaded, games_skipped).
    """
    files, skipped = human_game_files(root)
    samples = []
    for path in files:
        reviewed = review_path(path)
        with np.load(reviewed if reviewed.exists() else path) as game:
            samples.extend(normalized_web_game(
                game["observations"], game["policies"], game["outcomes"]))
    return samples, len(files), skipped


def state_from_observation(observation):
    """Rebuild the game position from a saved board encoding.

    Plane 0 holds the mover's stones, plane 1 the opponent's. Black moves
    first, so equal stone counts mean black is to move.
    """
    own = np.asarray(observation[0]) > 0.5
    other = np.asarray(observation[1]) > 0.5
    black_to_move = own.sum() == other.sum()
    if not (black_to_move or own.sum() + 1 == other.sum()):
        raise ValueError("Saved position has impossible stone counts")
    state = game_rules.GomokuState()
    player = 1 if black_to_move else -1
    state.board = (own.astype(np.int8) * player - other.astype(np.int8) * player)
    state.current_player = player
    return state


def lost_game_positions(root, min_stones=4):
    """Board encodings from saved games the AI did not win (self-play start points).

    Openings with fewer than min_stones stones are left out: those are ordinary
    self-play starts, not the places where it went wrong.
    """
    positions = []
    for path in human_game_files(root)[0]:
        with np.load(path) as game:
            if ai_did_not_win(game):
                positions.extend(normalized_web_game(
                    game["observations"], game["policies"], game["outcomes"]))
    return [observation for observation, _, _ in positions
            if observation[:2].sum() >= min_stones]
