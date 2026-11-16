# Gomoku — 19×19 AlphaZero-style reinforcement learning

Policy/value residual network, PUCT MCTS, rule-based terminal detection, self-play,
experience replay, and optional learning from games against a human.

## Usage

Activate your local PyTorch environment, then run from the repository root.

- `python train.py --rounds N` — self-play training; continues automatically from the newest checkpoint.
- `python -m unittest test_alphazero test_maintenance test_web_app -v` — regression tests.
- `python diagnose_defense.py --checkpoint checkpoints/<run>/<model>.pt` — read-only tactical diagnosis.
- `python -m web_app.server` — browser play using best (if configured) or latest checkpoint.
- `python -m web_app.server --learn` — continue from the newest checkpoint and learn after completed human games.
- `python promote.py` — separately evaluate latest against best; promotion is not automatic.

See `command.md` for the two training modes and how they alternate. `init_checkpoint`
is a *warm start* (a new run; `init_replay_from` can carry over a replay buffer).
`resume_from` continues an existing run using its `resume_state.bin`, restoring
optimizer, replay and RNG. Self-play mixes saved human games into ~10% of each batch
(`human_fraction`); no hand-written tactical examples enter training.
