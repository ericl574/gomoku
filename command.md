# Gomoku AlphaZero: commands

Every terminal first:

```bash
conda activate ml
cd /Users/ericl/Desktop/projects/gomoku
```

## Two model references

| Name | What it is | Who changes it | Used by |
|---|---|---|---|
| **latest** | the newest `.pt` file in `checkpoints/` (self-play *or* web learning) | every training save | resuming training |
| **best** | the model named in `checkpoints/best.json` | **only** `promote.py`, after a 40-game match | the web app (by default) |

The newest model is never assumed to be the best. Nothing is ever overwritten: every run gets a
new folder `checkpoints/<start time>_.../` and every save is a new `round_NNN_gamesNNNN.pt`.

---

## 0. One time only: set the starting "best"

```bash
python promote.py --set-baseline checkpoints/2026-11-10_2307_web_sims512/round_013_games0013.pt
```

This writes `checkpoints/best.json` with status **baseline** (your choice, not evaluated).
From then on, only a won match changes best.

## 1. Overnight self-play, first start (from the stronger checkpoint)

```bash
caffeinate -i python -c "from train import train; train(iterations=1000, games_per_iteration=6, simulations=512, updates_per_iteration=50, batch_size=64, eval_games=2, eval_every=1000, self_play_workers=6, init_checkpoint='checkpoints/2026-11-10_2307_web_sims512/round_013_games0013.pt')"
```

- Check that the dashboard says **"first 6 moves sampled"** (that confirms the fixed code).
- It saves after **every finished round** (6 games + 50 updates). **Ctrl+C anytime**: you lose at
  most the round in progress. The worker tracebacks after Ctrl+C are harmless.
- Games are logged to `game_logs/selfplay_<run>.txt`; round stats to `checkpoints/<run>/history.jsonl`.

## 2. Resume after an interruption (from latest)

```bash
caffeinate -i python -c "from train import train; from web_app.server import find_latest_checkpoint as latest; train(iterations=1000, games_per_iteration=6, simulations=512, updates_per_iteration=50, batch_size=64, eval_games=2, eval_every=1000, self_play_workers=6, init_checkpoint=latest())"
```

- Continues from the newest `.pt` file (weights + optimizer). The replay memory starts empty again.
- "latest" includes web learning saves. If you ran `--learn` in between, it resumes from that.

## 3. In the morning: test latest against best

```bash
python promote.py
```

- Plays latest vs best: 40 games (20 random openings x both colors), 512 simulations, about 1.5-3 minutes.
- Promoted only if latest scores **>= 60% over >= 40 games**. Otherwise best stays the same.
- Every match is recorded in `checkpoints/promotions.jsonl`, whether it was promoted or not.
- To test a specific model: `python promote.py --candidate checkpoints/<run>/round_NNN_gamesNNNN.pt`

## 4. Play against it in the browser (http://127.0.0.1:8000)

| Command | Model | Learns from your games? |
|---|---|---|
| `python -m web_app.server` | **best** (re-checked at every New game) | no; safe while self-play runs |
| `python -m web_app.server --latest` | **latest** (newest checkpoint) | no; safe while self-play runs |
| `python -m web_app.server --learn --imitate 0.7` | starts from **best** | **yes**; its saves become latest |

- **Never run `--learn` and self-play at the same time.** Each trains its own copy of the model.
- Add `--checkpoint <file.pt>` to any of these to use one specific model.

## 5. Is self-play healthy? (dashboard / history.jsonl)

| Healthy | Collapsing (stop and investigate) |
|---|---|
| black/white wins roughly 40-60% each | one color wins > 90% |
| games well above 10 moves | games ~9-10 moves (a pure race) |
| value loss well above 0 | value loss near 0 |
