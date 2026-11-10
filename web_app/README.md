# Play your AlphaZero Gomoku AI

From the `wuziqi` project directory, with the `ml` environment active:

```bash
python -m web_app.server                      # newest checkpoint, 512 simulations/move
python -m web_app.server --simulations 128    # faster, weaker
python -m web_app.server --checkpoint checkpoints/<run>/iteration_002.pt   # pin one model
```

Open **http://127.0.0.1:8000** (`--port 8001` if 8000 is busy). Ctrl+C stops it.

## Which AI you are playing

- By default the server plays the **newest `checkpoints/*/iteration_*.pt` by file
  write time**, across all training runs. "Newest" is not necessarily "strongest".
- It re-checks at every **New game**. If training has written a newer checkpoint,
  the next game uses it. It never swaps models in the middle of a game.
  You can leave the server running while `train()` runs.
- The panel's **AI MODEL** line shows `run · iter N · sims`; hover it for the full
  run name and update count. The terminal prints `Loaded <run>/<file>` on every swap.
- If a newer file fails to load, the server prints why and keeps the current model.
- `--checkpoint` pins one file and turns auto-update off.

## How the AI moves

AI is black and opens; you are white. Each AI move runs the same `MCTS` as
training and evaluation (`mcts.py`), with no Dirichlet noise, and plays the
most-visited move (ties are broken at random, as in `evaluate.py`). The search
tree is reused across moves. Rules come from `state.py`, the same rules training
uses: five or more in a row wins; a full board is a draw.

Inference is CPU-only with gradients off and one PyTorch thread, so it leaves
cores for a training run. Checkpoints are only read, never written.
One shared game lives in memory; refreshing a tab reconnects to it.

## Tests

```bash
python -B -m unittest -v test_web_app
```
