"""Games played against a human in web learning mode, saved as
checkpoints/<time>_web_*/game_NNNN.npz. Shared by web learning and self-play.

Each sample is (board encoding, policy target, outcome for the side that moved):
the AI's own MCTS visits for its moves, and (with --imitate) the human's move
scaled to the imitation weight for the human's moves.
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


def load_human_games(root):
    """All saved human games under root, oldest first, honouring the --fresh cutoff.

    Returns (samples, games_loaded, games_skipped). Run names start with their
    start time, so comparing name prefixes compares times.
    """
    cutoff = read_cutoff(root)
    samples, loaded, skipped = [], 0, 0
    for path in sorted(Path(root).glob("*_web_*/game_*.npz")):
        if path.parent.name[:len(cutoff)] < cutoff:
            skipped += 1
            continue
        with np.load(path) as game:
            samples.extend(normalized_web_game(
                game["observations"], game["policies"], game["outcomes"]))
        loaded += 1
    return samples, loaded, skipped
