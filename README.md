# Gomoku — 19×19 AlphaZero-style reinforcement learning

Policy/value residual network, PUCT MCTS, rule-based terminal detection, self-play,
experience replay, and optional learning from games against a human.

## Usage

Activate your local PyTorch environment, then run from the repository root.

- `python train.py --rounds N` — self-play training; continues automatically from the newest checkpoint.
- `python -m unittest test_alphazero test_maintenance test_web_app test_review -v` — regression tests.
- `python diagnose_defense.py --checkpoint checkpoints/<run>/<model>.pt` — read-only tactical diagnosis.
- `python -m web_app.server` — browser play using best (if configured) or latest checkpoint; finished games are saved for training (`--no-save` to skip).
- `python -m web_app.server --learn` — continue from the newest checkpoint and learn after completed human games.
- `python promote.py` — separately evaluate latest against best; promotion is not automatic.

## Public web version (`site/`)

A static copy of the play page in which the AI runs entirely in the visitor's browser: no
server, no PyTorch. Rules and MCTS are ported to JavaScript (`site/engine.js`, a line-by-line
port of `state.py` and `mcts.py`), and the network runs through ONNX Runtime Web.

1. `python export_onnx.py` — best model → `site/model.onnx` (+ `site/model.json`). Re-run after a promotion.
2. `python check_onnx.py` — ONNX vs PyTorch outputs on 51 boards (max difference ~1e-5).
3. `python check_site_engine.py` — JS vs Python search on 24 positions: visit counts must match
   (first time: `cd tools/js && npm install`).
4. Preview: `python -m http.server 8765 -d site`, open http://127.0.0.1:8765.
5. Deploy the `site/` folder as a static site, e.g. `cd site && npx vercel` (then `npx vercel --prod`).

512 simulations per move (about 0.3–3.5 s on an M-series Mac); `?sims=128` in the address gives a
faster, weaker AI. The page plays only; it never learns or saves games.

See `command.md` for the two training modes and how they alternate. `init_checkpoint`
is a *warm start* (a new run; `init_replay_from` can carry over a replay buffer).
`resume_from` continues an existing run using its `resume_state.bin`, restoring
optimizer, replay and RNG. Self-play mixes saved human games into ~10% of each batch
(`human_fraction`). Before training, games the AI did not win are re-labelled once by
a 4096-simulation search (`review.py`, written to `game_*.review.npz`), and ~20% of
self-play games start from their positions (`start_fraction`). No hand-written
tactical examples enter training.
