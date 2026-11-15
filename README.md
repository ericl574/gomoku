# Gomoku — 19×19 AlphaZero-style reinforcement learning

Policy/value residual network, PUCT MCTS, rule-based terminal detection, self-play,
experience replay, and optional learning from games against a human.

## Usage

Activate your local PyTorch environment, then run from the repository root.

- `python -m unittest test_alphazero test_maintenance -v` — regression tests.
- `python diagnose_defense.py --checkpoint checkpoints/<run>/<model>.pt` — read-only tactical diagnosis.
- `python -m web_app.server` — browser play using best (if configured) or latest checkpoint.
- `python -m web_app.server --learn` — learn after completed human games.
- `python promote.py` — separately evaluate latest against best; promotion is not automatic.

See `command.md` for training and resume examples. `init_checkpoint` is a *warm start*:
it starts a new experiment with an empty replay buffer. `resume_from` continues an
existing run using its `resume_state.bin`, restoring optimizer, replay and RNG.
The original training algorithm is unchanged; no tactical examples enter training.
