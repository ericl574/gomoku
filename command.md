caffeinate -i python -c "from train import train; from web_app.server import find_latest_checkpoint as latest; train(iterations=1000, games_per_iteration=6, simulations=512, updates_per_iteration=50, batch_size=64, eval_games=2, eval_every=1000, self_play_workers=6, init_checkpoint=latest())"


python -m web_app.server --learn --imitate 0.7
Ctrl+C when you're done. Every finished game is saved, and step 1 continues from it next time.

python -m web_app.server