"""Export the best Gomoku network to ONNX, so it can run without PyTorch (e.g. in a browser).

    python export_onnx.py                          # best model -> site/model.onnx
    python export_onnx.py --checkpoint <file.pt>   # a specific checkpoint

Only the network goes into the file: board in, (policy_logits, value) out.
Rules (illegal-move masking) and the MCTS search stay outside, in the caller.
"""

import argparse
import json
from pathlib import Path

import torch

from web_app.server import load_model  # same loader the web app uses: weights only, eval mode, CPU


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None, help="default: the path in checkpoints/best.json")
    parser.add_argument("--out", default="site/model.onnx")
    args = parser.parse_args()

    # 1. Which model: the promoted "best" unless one is given.
    checkpoint = args.checkpoint or json.loads(Path("checkpoints/best.json").read_text())["path"]
    model, info = load_model(checkpoint)  # the optimizer state in the .pt file is simply not used
    print(f"loaded {checkpoint} (round {info['iteration']}, {info['games']} games)")

    # 2. A fake input with the real shape: (batch=1, planes=3, 19, 19).
    #    The exporter runs the model once on it and records every operation it performs.
    dummy_board = torch.zeros(1, 3, 19, 19)

    # 3. Export: the recorded operations + the weights -> one .onnx file.
    torch.onnx.export(
        model,
        (dummy_board,),                               # inputs, as a tuple
        args.out,
        input_names=["board"],                        # names the browser uses to feed and read
        output_names=["policy_logits", "value"],
        external_data=False,
    )

    # 4. Was the size prediction right?
    size_kb = Path(args.out).stat().st_size / 1024
    print(f"wrote {args.out}: {size_kb:.0f} KB (the .pt checkpoint is {Path(checkpoint).stat().st_size / 1024:.0f} KB)")

    # 5. Which model this is, for the web page's "AI MODEL" line.
    meta = {key: info[key] for key in ("run", "file", "iteration", "games", "updates", "architecture")}
    meta_path = Path(args.out).with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=1) + "\n")
    print(f"wrote {meta_path}")


if __name__ == "__main__":
    main()
