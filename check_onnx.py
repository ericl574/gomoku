"""Check that the exported ONNX model gives the same answers as the PyTorch original.

    python check_onnx.py

Both models get the same boards (an empty one, plus positions from random games); we compare
policy logits and value. Tiny differences (~1e-6) are normal float rounding; big ones mean a
broken export.
"""

import json
from pathlib import Path

import numpy as np
import onnxruntime
import torch

from state import GomokuState
from web_app.server import load_model

ONNX_PATH = "site/model.onnx"
TOLERANCE = 1e-4  # far above float rounding, far below anything that could change a move


def test_boards(count=50, seed=0):
    """An empty board plus `count` positions from random games (1 to 60 moves in)."""
    rng = np.random.default_rng(seed)
    boards = [GomokuState().encode()]
    while len(boards) < count + 1:
        state = GomokuState()
        for _ in range(rng.integers(1, 61)):
            if state.done:
                break
            state.step(int(rng.choice(state.legal_actions())))
        if not state.done:
            boards.append(state.encode())
    return boards


def main():
    # 1. Load both versions of the AI's network.
    checkpoint = json.loads(Path("checkpoints/best.json").read_text())["path"]
    original, _ = load_model(checkpoint)                     # PyTorch
    copy = onnxruntime.InferenceSession(ONNX_PATH)           # ONNX

    worst_policy = worst_value = 0.0
    boards = test_boards()
    for board in boards:
        x = board[np.newaxis]                                # (3, 19, 19) -> (1, 3, 19, 19)

        # 2. Same board into both.
        with torch.no_grad():
            torch_policy, torch_value = original(torch.from_numpy(x))
        onnx_policy, onnx_value = copy.run(None, {"board": x})

        # 3. How far apart are the answers?
        worst_policy = max(worst_policy, float(np.abs(torch_policy.numpy() - onnx_policy).max()))
        worst_value = max(worst_value, float(np.abs(torch_value.numpy() - onnx_value).max()))

    # 4. Verdict.
    print(f"{len(boards)} boards | largest difference: policy {worst_policy:.2e}, value {worst_value:.2e}")
    if max(worst_policy, worst_value) < TOLERANCE:
        print("OK: the ONNX copy matches the original")
    else:
        raise SystemExit("MISMATCH: the export is broken, do not deploy it")


if __name__ == "__main__":
    main()
