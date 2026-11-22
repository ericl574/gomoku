"""Check that the browser engine (site/engine.js) searches exactly like mcts.py.

    python check_site_engine.py            # needs: cd tools/js && npm install (once)

Same positions, same network, both searches; compare the visit counts.
  flat:  a network that knows nothing (equal priors, value 0). Pure rules + search logic,
         no float noise from the network: the visit counts must be identical.
  onnx:  the real model (site/model.onnx) through ONNX Runtime on both sides. Softmax is float32
         in Python and float64 in JS, so a near-tie may rarely split differently; the chosen
         (most-visited) move must still agree everywhere.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnxruntime
import torch
from torch import nn

from mcts import MCTS
from state import GomokuState

ROOT = Path(__file__).resolve().parent
ONNX_PATH = ROOT / "site" / "model.onnx"


class FlatNetwork(nn.Module):
    """Equal priors everywhere, value 0 (as in test_alphazero)."""

    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        return torch.zeros(len(x), 361) + self.anchor, torch.zeros(len(x), 1) + self.anchor


class OnnxNetwork(nn.Module):
    """The exported model through ONNX Runtime, shaped like GomokuNetwork for mcts.py."""

    def __init__(self, path):
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0))  # mcts.py reads the device from a parameter
        self.session = onnxruntime.InferenceSession(str(path))

    def forward(self, x):
        policy, value = self.session.run(None, {"board": x.cpu().numpy()})
        return torch.from_numpy(policy), torch.from_numpy(value)


def positions(count=24, seed=1):
    """Random-game positions: openings, middle games, and some with fours on the board."""
    rng = np.random.default_rng(seed)
    found = [[]]
    while len(found) < count:
        state, moves = GomokuState(), []
        # Random play near the centre so lines (and fours) actually form.
        for _ in range(int(rng.integers(1, 70))):
            legal = [a for a in state.legal_actions() if abs(a // 19 - 9) <= 4 and abs(a % 19 - 9) <= 4]
            if state.done or not legal:
                break
            action = int(rng.choice(legal))
            state.step(action)
            moves.append(action)
        if not state.done:
            found.append(moves)
    return found


def python_visits(model, moves, simulations):
    state = GomokuState()
    for action in moves:
        state.step(action)
    _, root = MCTS(model, simulations=simulations).search(state, add_noise=False)
    return {action: child.visit_count for action, child in root.children.items() if child.visit_count}


def js_visits(model_name, games, simulations):
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        json.dump({"model": model_name, "simulations": simulations, "positions": games}, handle)
    output = subprocess.run(["node", str(ROOT / "tools/js/parity.mjs"), handle.name],
                            capture_output=True, text=True, check=True, cwd=ROOT)
    return [{int(a): n for a, n in result["visits"].items()} for result in json.loads(output.stdout)]


def compare(label, model, model_name, games, simulations, exact):
    js = js_visits(model_name, games, simulations)
    identical = same_move = 0
    for moves, js_result in zip(games, js):
        py_result = python_visits(model, moves, simulations)
        identical += py_result == js_result
        top = lambda visits: {a for a, n in visits.items() if n == max(visits.values())}  # noqa: E731
        same_move += bool(top(py_result) & top(js_result))
    print(f"{label}: {len(games)} positions, {simulations} simulations | "
          f"identical visit counts {identical}/{len(games)}, same chosen move {same_move}/{len(games)}")
    return identical == len(games) if exact else same_move == len(games)


def main():
    games = positions()
    fours = sum(1 for g in games if _has_four(g))
    print(f"{fours}/{len(games)} test positions have a four on the board (exercises the win/loss rules)")
    ok = compare("flat", FlatNetwork(), "flat", games, 200, exact=True)
    ok &= compare("onnx", OnnxNetwork(ONNX_PATH), str(ONNX_PATH), games, 200, exact=False)
    if not ok:
        sys.exit("MISMATCH: site/engine.js does not search like mcts.py")
    print("OK: the browser engine matches the Python search")


def _has_four(moves):
    state = GomokuState()
    for action in moves:
        state.step(action)
    flipped = state.clone()
    flipped.current_player = -flipped.current_player
    return state.has_immediate_win() or flipped.has_immediate_win()


if __name__ == "__main__":
    main()
