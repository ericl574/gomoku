# Play your AlphaZero Gomoku AI

From the `gomoku` project directory, with the `ml` environment active:

```bash
python -m web_app.server                      # newest checkpoint, 512 simulations/move
python -m web_app.server --simulations 128    # faster, weaker
python -m web_app.server --checkpoint checkpoints/<run>/round_005_games0060.pt   # pin one model
```

Training writes `checkpoints/<start time>_sims<N>/round_<R>_games<G>.pt`, e.g.
`checkpoints/2026-11-10_1230_sims512/round_005_games0060.pt`: round 5, 60 self-play
games so far. Older runs use `alphazero_<id>/iteration_<R>.pt`; both are found.

Open **http://127.0.0.1:8000** (`--port 8001` if 8000 is busy). Ctrl+C stops it.

## Which AI you are playing

- By default the server plays the **newest `checkpoints/<run>/*.pt` by file
  write time**, across all training runs. "Newest" is not necessarily "strongest".
- It re-checks at every **New game**. If training has written a newer checkpoint,
  the next game uses it. It never swaps models in the middle of a game.
  You can leave the server running while `train()` runs.
- The panel's **AI MODEL** line shows `R<round> · <games> games · <sims> sims`; hover
  it for the run folder and update count. The terminal prints `Loaded <run>/<file>` on every swap.
- If a newer file fails to load, the server prints why and keeps the current model.
- `--checkpoint` pins one file and turns auto-update off.

## Learning mode (`--learn`)

```bash
python -m web_app.server --learn                       # start from the newest checkpoint
python -m web_app.server --learn --checkpoint <file>   # start from a chosen one
```

The AI trains on every game you **finish** against it (an abandoned game has no
result and is skipped). It uses the same rule as self-play: for each AI move, the
board, the AI's own MCTS visit counts, and the final result from the AI's side.
Your moves are never used as targets (unless `--imitate`). Unlike self-play, there is no root noise: your play already varies the games.
After each game it runs `--learn-updates` (default 10) gradient steps, saves
`checkpoints/<time>_web_sims<N>/round_<G>_games<G>.pt` and the game itself
(`game_<G>.npz`), and the next game uses the new weights. Games from earlier
learning sessions are reloaded into the replay buffer at startup.

`--imitate [WEIGHT]` (default 0.5) also turns each of **your** moves into an
example: the board from your side, your move as the policy target scaled to
WEIGHT (so it pulls exactly WEIGHT times as hard as one of the AI's own examples),
and the result from your side. It is a head start (cf. "kickstarting"); self-play
later uses no human moves, so the copying fades out there.

To hand the taught model to self-play later:
`train(..., init_checkpoint="checkpoints/<web run>/round_XXX_gamesXXXX.pt")`.

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
