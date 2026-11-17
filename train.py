import json
import secrets
import multiprocessing as mp
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from itertools import repeat
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from torch import nn
from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
)
from rich.table import Table

from evaluate import play_match
from network import GomokuNetwork
from self_play import play_game
from human_games import load_human_games, lost_game_positions, state_from_observation
from review import review_games
import state as game_rules


class ReplayBuffer:
    def __init__(self, capacity=10000):
        if capacity < 1:
            raise ValueError("Buffer capacity must be positive")

        self.data = deque(maxlen=capacity)
        # Human games (human_games.py): kept for good, never evicted, and not
        # counted in len(); sample() mixes them in at a fixed fraction.
        self.fixed = []

    def __len__(self):
        return len(self.data)

    def add_game(self, samples):
        self.data.extend(samples)

    def add_fixed(self, samples):
        self.fixed.extend(samples)

    def sample(self, batch_size, rng, augmentation_rng=None, fixed_fraction=0.0):
        if not 1 <= batch_size <= len(self):
            raise ValueError("Invalid batch size")

        if augmentation_rng is None:
            augmentation_rng = rng

        # On average fixed_fraction of the batch comes from the fixed (human)
        # samples. With none, rng use is exactly as before (no extra draws).
        fixed_count = 0
        if fixed_fraction > 0 and self.fixed:
            fixed_count = min(int(rng.binomial(batch_size, fixed_fraction)), len(self.fixed))

        indices = rng.choice(
            len(self),
            size=batch_size - fixed_count,
            replace=False,
        )
        items = [self.data[int(index)] for index in indices]
        if fixed_count:
            items += [self.fixed[int(index)] for index in
                      rng.choice(len(self.fixed), size=fixed_count, replace=False)]

        observations = []
        policies = []
        outcomes = []

        for observation, policy, outcome in items:

            turns = int(augmentation_rng.integers(0, 4))
            mirror = bool(augmentation_rng.integers(0, 2))

            transformed_observation = np.rot90(
                observation,
                k=turns,
                axes=(1, 2),
            )
            transformed_policy = np.rot90(
                policy.reshape(19, 19),
                k=turns,
            )

            if mirror:
                transformed_observation = np.flip(
                    transformed_observation,
                    axis=2,
                )
                transformed_policy = np.flip(
                    transformed_policy,
                    axis=1,
                )

            observations.append(transformed_observation.copy())
            policies.append(
                transformed_policy.reshape(361).copy()
            )
            outcomes.append(outcome)

        return (
            torch.from_numpy(np.stack(observations)),
            torch.from_numpy(np.stack(policies)),
            torch.tensor(
                outcomes,
                dtype=torch.float32,
            ).unsqueeze(1),
        )

def self_play_worker(model_config, state_dict, simulations, seed, exploration_moves=6,
                     start=None):
    """One self-play game in a worker process; same game as the serial path.

    start: board encoding to play on from (None = empty board).
    """
    torch.set_num_threads(1)
    model = GomokuNetwork(**model_config)
    model.load_state_dict(state_dict)
    model.eval()
    move_log = []
    samples, final_state = play_game(
        model, simulations=simulations, seed=seed, move_log=move_log,
        exploration_moves=exploration_moves,
        start_state=None if start is None else state_from_observation(start),
    )
    return samples, final_state, move_log


def format_selfplay_game(iteration, game, move_log, final_state, detailed, start=None):
    """One self-play game as text. detailed=True adds what the search saw per move.

    start: the board encoding a from-your-game game started at (None = empty board).
    """
    def coord(action):
        return f"{chr(65 + action % 19)}{action // 19 + 1}"  # web board labels

    def color(player):
        return "B" if player == 1 else "W"

    winner = {1: "black won", -1: "white won", 0: "draw"}[int(final_state.winner)]
    if start is None:
        lines = [f"=== Round {iteration} game {game} | {winner} in {len(move_log)} moves"]
    else:  # worded so the "in N moves" length statistics skip these games
        stones = int(start[0].sum() + start[1].sum())
        lines = [f"=== Round {iteration} game {game} | from your game, {stones} stones on the "
                 f"board | {winner} after {len(move_log)} more moves"]
    if detailed:
        lines.append("value = mover's search estimate (+1 winning); top: n visits, "
                     "p network prior, q mover's estimate; w1 = win-in-one hits")
        for number, move in enumerate(move_log, 1):
            top = ", ".join(f"{coord(a)} n={n} p={p:.3f} q={q:+.2f}"
                            for a, n, p, q in move["top"])
            lines.append(
                f"{number:3d}. {color(move['player'])} {coord(move['action']):>4}   "
                f"value {move['value']:+.2f}, searched {move['visited']}/"
                f"{move['candidates']}, w1 {move['win_in_one']} | top: {top}")
        lines.append("final board (X black, O white):")
        lines.append("    " + " ".join(chr(65 + c) for c in range(19)))
        for r in range(19):
            marks = " ".join({1: "X", -1: "O", 0: "."}[int(v)] for v in final_state.board[r])
            lines.append(f"{r + 1:3d} {marks}")
    else:
        moves = [f"{color(m['player'])} {coord(m['action'])}" for m in move_log]
        lines += ["    " + "  ".join(moves[k:k + 12]) for k in range(0, len(moves), 12)]
    return "\n".join(lines) + "\n\n"


def train_step(model, optimizer, batch):
    model.train()
    device = next(model.parameters()).device

    observations, target_policies, target_values = (
        tensor.to(device) for tensor in batch
    )

    optimizer.zero_grad(set_to_none=True)

    logits, predicted_values = model(observations)
    log_probabilities = torch.log_softmax(logits, dim=1)

    policy_loss = -(
        target_policies * log_probabilities
    ).sum(dim=1).mean()

    value_loss = nn.functional.mse_loss(
        predicted_values,
        target_values,
    )

    loss = policy_loss + value_loss
    loss.backward()
    optimizer.step()

    return {
        "loss": loss.item(),
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
    }


def save_checkpoint(checkpoint, path):
    """Replace a checkpoint only after the new file is fully written."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(checkpoint, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def pack_resume_state(model, optimizer, replay, rng, augmentation_rng,
                      eval_rng, initial_opponent, iteration, games, updates,
                      color_results, seed, config, run_dir, eval_seeds, initial_source):
    """Tensor-only replay for safe torch.load(weights_only=True)."""
    records = list(replay.data)
    if records:
        observations = torch.from_numpy(np.stack([x[0] for x in records]).astype(np.float32))
        policies = torch.from_numpy(np.stack([x[1] for x in records]).astype(np.float32))
        outcomes = torch.tensor([x[2] for x in records], dtype=torch.float32)
    else:
        observations = torch.empty((0, 3, 19, 19), dtype=torch.float32)
        policies = torch.empty((0, 361), dtype=torch.float32)
        outcomes = torch.empty((0,), dtype=torch.float32)
    return {
        "resume_version": 1,
        "model_config": config["model_config"],
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "initial_opponent_state_dict": initial_opponent.state_dict(),
        "replay": (observations, policies, outcomes),
        "rng_state": rng.bit_generator.state,
        "augmentation_rng_state": augmentation_rng.bit_generator.state,
        "eval_rng_state": eval_rng.bit_generator.state,
        "torch_rng_state": torch.get_rng_state(),
        "iteration": iteration, "games_completed": games, "updates_completed": updates,
        "color_results": {int(k): int(v) for k, v in color_results.items()},
        "seed": seed, "config": config, "run_dir": str(run_dir),
        "eval_seeds": torch.from_numpy(eval_seeds.copy()),
        "init_checkpoint": initial_source,
    }


def load_resume_state(path):
    """Read a trusted local resume file, rejecting mismatched checkpoint formats."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("resume_version") != 1:
        raise ValueError("Unsupported or missing resume format")
    return state


def checkpoint_root():
    """<project>/checkpoints (read at call time, so tests can redirect it)."""
    return Path(__file__).resolve().parent / "checkpoints"


def find_latest_checkpoint(root=None):
    """Newest .pt by write time across every training run folder.

    Matches new round_NNN_gamesNNNN.pt and old iteration_NNN.pt names alike.
    save_checkpoint writes `*.pt.tmp` first and renames it, so a file
    matching this pattern is always complete.
    """
    root = checkpoint_root() if root is None else Path(root)
    candidates = []
    for path in root.glob("*/*.pt"):
        try:
            candidates.append((path.stat().st_mtime_ns, str(path), path))
        except OSError:
            continue
    if not candidates:
        raise FileNotFoundError(f"No .pt checkpoints under {root}/<run>/")
    return max(candidates)[2]


def replay_samples(replay_tensors, capacity):
    """(observations, policies, outcomes) tensors from a resume state -> samples."""
    obs, pol, val = replay_tensors
    if obs.shape[1:] != (3, 19, 19) or pol.shape != (len(obs), 361) or val.shape != (len(obs),):
        raise ValueError("Invalid replay data in resume state")
    if len(obs) > capacity:
        raise ValueError("Resume replay larger than configured capacity")
    return [(o.numpy().copy(), p.numpy().copy(), float(v)) for o, p, v in zip(obs, pol, val)]


def trim_history(path, last_iteration):
    """Keep only history lines up to the resume point.

    history.jsonl is written before resume_state.bin, so after a crash between
    the two it can hold rounds (or a half-written line) that will be re-run.
    Drop those so every round appears exactly once. Atomic replace.
    """
    if not path.exists():
        return
    kept = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue  # half-written line from an interrupted save
        if record.get("iteration", 0) <= last_iteration:
            kept.append(line)
    temporary = path.with_suffix(".jsonl.tmp")
    temporary.write_text("".join(f"{line}\n" for line in kept), encoding="utf-8")
    temporary.replace(path)


def render_dashboard(progress, status):
    table = Table.grid(padding=(0, 2))
    table.add_column(style="cyan", no_wrap=True)
    table.add_column()

    for label, value in status.items():
        table.add_row(label, str(value))

    return Panel(
        Group(progress, "", table),
        title="Gomoku AlphaZero",
        border_style="cyan",
    )


def train(
    iterations=1,
    games_per_iteration=1,
    simulations=8,
    updates_per_iteration=5,
    batch_size=32,
    buffer_capacity=10000,
    learning_rate=1e-3,
    seed=0,
    eval_every=10,
    eval_games=20,
    cpu_threads=None,
    self_play_workers=1,
    init_checkpoint=None,
    exploration_moves=6,
    resume_from=None,
    human_fraction=0.0,
    init_replay_from=None,
    start_fraction=0.0,
):
    """One self-play training run.

    init_checkpoint: warm start a NEW run from a model. init_replay_from: with
    it, fill the new run's replay from a resume_state.bin (e.g. the previous
    self-play run), instead of starting empty.
    resume_from: continue the SAME run exactly (see command.md).
    human_fraction: average share of each batch drawn from the saved human
    games (human_games.py); 0 = self-play data only.
    start_fraction: average share of self-play games that start from a position
    of a saved game the AI did not win (in a random one of the 8 symmetries),
    instead of an empty board. These are left out of the black/white counts.
    """
    if init_checkpoint is not None and resume_from is not None:
        raise ValueError("Choose init_checkpoint (new run) OR resume_from (same run)")
    if init_replay_from is not None and init_checkpoint is None:
        raise ValueError("init_replay_from only works with init_checkpoint")
    if not 0.0 <= human_fraction < 1.0:
        raise ValueError("human_fraction must be in [0, 1)")
    if not 0.0 <= start_fraction < 1.0:
        raise ValueError("start_fraction must be in [0, 1)")
    # exploration_moves: the first N moves of each self-play game are sampled in
    # proportion to visits (variety); later moves play the most-visited move.
    # AlphaZero used 30 for Go (200+ move games); Gomoku games are ~10-40 moves,
    # so 30 would make nearly every move random.
    counts = (
        self_play_workers,
        iterations,
        games_per_iteration,
        simulations,
        updates_per_iteration,
        batch_size,
        buffer_capacity,
        eval_every,
        eval_games,
    )

    if any(
        isinstance(n, bool)
        or not isinstance(n, (int, np.integer))
        or n < 1
        for n in counts
    ):
        raise ValueError("Counts must be positive integers")

    if eval_games % 2 != 0:
        raise ValueError("Evaluation games must be even")

    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Learning rate must be positive and finite")
    if cpu_threads is not None:
        if (
            isinstance(cpu_threads, bool)
            or not isinstance(cpu_threads, int)
            or cpu_threads < 1
        ):
            raise ValueError("CPU threads must be a positive integer")
        torch.set_num_threads(cpu_threads)

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    augmentation_rng = np.random.default_rng(seed + 200000)

    # Separate evaluation randomness from training randomness.
    eval_rng = np.random.default_rng(seed + 100000)
    eval_seeds = eval_rng.integers(
        0,
        2**32,
        size=eval_games,
    )

    model_config = {
        "channels": 32,
        "value_hidden": 64,
        "policy_kernel": 3,
        "architecture": "resnet_global",
        "residual_blocks": 3,
    }

    resume = load_resume_state(resume_from) if resume_from is not None else None
    start = None
    if init_checkpoint is not None:
        # Continue from a saved model (e.g. one taught in web learning mode)
        # instead of random weights. The replay buffer still starts empty.
        start = torch.load(init_checkpoint, map_location="cpu", weights_only=True)
        model_config = start["model_config"]

    config = {
        "model_config": model_config,
        "games_per_iteration": games_per_iteration,
        "simulations": simulations,
        "updates_per_iteration": updates_per_iteration,
        "batch_size": batch_size,
        "buffer_capacity": buffer_capacity,
        "learning_rate": learning_rate,
        "exploration_moves": exploration_moves,
        "eval_every": eval_every,
        "eval_games": eval_games,
        "human_fraction": human_fraction,
        "start_fraction": start_fraction,
    }
    if resume is not None:
        model_config = resume["model_config"]
        config["model_config"] = model_config
        stored = {"start_fraction": 0.0, **resume["config"]}  # runs saved before it existed
        if stored != config or resume["seed"] != seed:
            raise ValueError("Resume configuration or seed differs; use the original parameters")
    model = GomokuNetwork(**model_config)
    if start is not None:
        model.load_state_dict(start["model_state_dict"])
    if resume is not None:
        model.load_state_dict(resume["model_state_dict"])

    # This opponent stays unchanged throughout the run
    # (the random start, or the loaded model when init_checkpoint is given).
    initial_opponent = deepcopy(model)
    if resume is not None:
        initial_opponent.load_state_dict(resume["initial_opponent_state_dict"])
    initial_opponent.eval()
    initial_opponent.requires_grad_(False)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )
    if start is not None and "optimizer_state_dict" in start:
        optimizer.load_state_dict(start["optimizer_state_dict"])
        for group in optimizer.param_groups:
            group["lr"] = learning_rate  # this run's learning rate wins
    if resume is not None:
        eval_seeds = resume["eval_seeds"].numpy().copy()
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
    replay = ReplayBuffer(buffer_capacity)
    if init_replay_from is not None:
        # Carried-over self-play memory; the deque keeps the newest samples.
        replay.add_game(replay_samples(load_resume_state(init_replay_from)["replay"],
                                       float("inf")))
    human_games = 0
    if human_fraction > 0:
        human, human_games, _ = load_human_games(checkpoint_root())
        replay.add_fixed(human)
    # Positions from games you won against it: some self-play games start there.
    start_positions = lost_game_positions(checkpoint_root()) if start_fraction > 0 else []
    if resume is not None:
        replay.add_game(replay_samples(resume["replay"], buffer_capacity))
        rng.bit_generator.state = resume["rng_state"]
        augmentation_rng.bit_generator.state = resume["augmentation_rng_state"]
        eval_rng.bit_generator.state = resume["eval_rng_state"]
        torch.set_rng_state(resume["torch_rng_state"])

    initial_source = (resume.get('init_checkpoint') if resume is not None
                      else (str(init_checkpoint) if init_checkpoint is not None else None))

    # Readable run name: start time + search budget, e.g. 2026-11-10_1230_sims512.
    run_name = f"{datetime.now():%Y-%m-%d_%H%M}_sims{simulations}"
    if resume is None:
        run_dir = checkpoint_root() / run_name
        suffix = 2
        while run_dir.exists():  # two runs started in the same minute
            run_dir = run_dir.with_name(f"{run_name}-{suffix}")
            suffix += 1
        run_dir.mkdir(parents=True, exist_ok=False)
    else:
        # Use the folder the resume file is in now, so the project can be moved
        # (e.g. to another machine). The folder name must still match, so a
        # resume file copied into a different run cannot write into it.
        resume_path = Path(resume_from).resolve()
        run_dir = resume_path.parent
        if resume_path.name != "resume_state.bin" or run_dir.name != Path(resume["run_dir"]).name:
            raise ValueError("Resume file must be resume_state.bin in its original run directory")
        # A newer checkpoint elsewhere (e.g. web learning) would be silently
        # discarded by an exact resume; `python train.py` handles that case.
        latest = find_latest_checkpoint(run_dir.parent)
        if latest.parent.resolve() != run_dir.resolve():
            raise ValueError(
                f"Newer checkpoint {latest.parent.name}/{latest.name} exists; resuming would "
                "discard it. Run `python train.py --rounds N` to continue from it instead.")
    history_path = run_dir / "history.jsonl"
    if resume is not None:
        trim_history(history_path, int(resume["iteration"]))
    # Every self-play game; the first game of each round with full search details.
    game_log_path = (
        Path(__file__).resolve().parent / "game_logs" / f"selfplay_{run_dir.name}.txt"
    )

    total_games = iterations * games_per_iteration
    total_updates = iterations * updates_per_iteration

    color_results = dict(resume["color_results"]) if resume is not None else {1: 0, -1: 0, 0: 0}
    completed_games = int(resume["games_completed"]) if resume is not None else 0
    completed_updates = int(resume["updates_completed"]) if resume is not None else 0
    start_iteration = int(resume["iteration"]) if resume is not None else 0
    # `iterations` counts rounds in THIS call; round numbers are absolute.
    final_iteration = start_iteration + iterations

    status = {
        "Stage": "Starting",
        "Search / batch": (f"{simulations} simulations, first {exploration_moves} moves "
                           f"sampled / batch up to {batch_size}"),
        "CPU threads": str(torch.get_num_threads()),
        "Current move": "--",
        "Game elapsed": "--",
        "Last search": "--",
        "Loss: last batch": "--",
        "Loss: round mean": "--",
        "Replay buffer": f"{len(replay)} / {buffer_capacity}",
        "Human games": (f"{human_games} games, {len(replay.fixed)} positions, "
                        f"~{human_fraction:.0%} of each batch" if human_fraction else "off"),
        "From your games": (f"{len(start_positions)} positions from games it lost; "
                            f"~{start_fraction:.0%} of self-play games start there"
                            if start_positions else "off"),
        "Self-play B/W/D": "No completed games",
        "Eval vs initial": "Not evaluated yet",
        "Last checkpoint": "--",
    }

    progress = Progress(
        TextColumn("{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("{task.completed:.0f}/{task.total:.0f}"),
        auto_refresh=False,
    )

    round_task = progress.add_task("Round", total=iterations)
    game_task = progress.add_task("Self-play", total=total_games)
    update_task = progress.add_task("Updates", total=total_updates)
    search_task = progress.add_task("MCTS", total=simulations, visible=False)
    eval_task = progress.add_task(
        "Evaluation",
        total=eval_games,
        visible=False,
    )

    # Workers persist for the whole run; "spawn" is the safe start method with torch.
    pool = (
        ProcessPoolExecutor(self_play_workers, mp_context=mp.get_context("spawn"))
        if self_play_workers > 1
        else None
    )

    print("Run directory:", run_dir, flush=True)
    print("Self-play games are logged to", game_log_path, flush=True)

    with Live(
        render_dashboard(progress, status),
        refresh_per_second=4,
    ) as live:
        def report_search(event):
            player = "Black" if event["player"] == 1 else "White"
            status["Current move"] = f"#{event['move']} {player}"
            status["Game elapsed"] = f"{event['elapsed_seconds']:.1f}s"
            status["Last search"] = (
                f"retained={event['reused_visits']} "
                f"NN={event['network_calls']} cache={event['cache_hits']}"
            )
            progress.update(
                search_task,
                completed=event["simulations_done"],
                total=event["simulations_total"],
                visible=True,
            )
            live.update(render_dashboard(progress, status))

        def hide_search():
            progress.update(search_task, visible=False)
            status["Current move"] = "--"
            status["Game elapsed"] = "--"

        for iteration in range(start_iteration + 1, start_iteration + iterations + 1):
            snapshot = deepcopy(model)
            snapshot.eval()
            snapshot.requires_grad_(False)

            # Same seeds, drawn in the same order, whether serial or parallel.
            seeds = [int(rng.integers(0, 2**32)) for _ in range(games_per_iteration)]
            # Some games start from a position of a game it lost to you. Drawn only
            # when there are positions, so other runs use the rng exactly as before.
            starts = [None] * games_per_iteration
            if start_fraction > 0 and start_positions:
                for index in range(games_per_iteration):
                    if rng.random() < start_fraction:
                        board = start_positions[int(rng.integers(len(start_positions)))]
                        board = np.rot90(board, k=int(rng.integers(0, 4)), axes=(1, 2))
                        if rng.integers(0, 2):
                            board = np.flip(board, axis=2)
                        starts[index] = board.copy()
            position_games = 0
            if pool is not None:
                status["Stage"] = (
                    f"Round {iteration}/{final_iteration}: "
                    f"self-play on {self_play_workers} workers"
                )
                live.update(render_dashboard(progress, status))
                parallel_games = pool.map(
                    self_play_worker,
                    repeat(model_config),
                    repeat(snapshot.state_dict()),
                    repeat(simulations),
                    seeds,
                    repeat(exploration_moves),
                    starts,
                )

            for game in range(1, games_per_iteration + 1):
                if pool is not None:
                    samples, final_state, move_log = next(parallel_games)
                else:
                    status["Stage"] = (
                        f"Round {iteration}/{final_iteration}: "
                        f"self-play game {game}/{games_per_iteration}"
                    )
                    live.update(render_dashboard(progress, status))

                    move_log = []
                    samples, final_state = play_game(
                        snapshot,
                        simulations=simulations,
                        seed=seeds[game - 1],
                        progress_callback=report_search,
                        move_log=move_log,
                        exploration_moves=exploration_moves,
                        start_state=(None if starts[game - 1] is None
                                     else state_from_observation(starts[game - 1])),
                    )

                start = starts[game - 1]
                game_log_path.parent.mkdir(parents=True, exist_ok=True)
                with game_log_path.open("a", encoding="utf-8") as file:
                    file.write(format_selfplay_game(
                        iteration, game, move_log, final_state,
                        detailed=game == 1 or start is not None, start=start))

                replay.add_game(samples)
                completed_games += 1
                if start is None:  # mid-game starts would skew the black/white balance
                    color_results[final_state.winner] += 1
                else:
                    position_games += 1

                black = color_results[1]
                white = color_results[-1]
                draws = color_results[0]
                counted = max(1, black + white + draws)

                status["Replay buffer"] = (
                    f"{len(replay)} / {buffer_capacity}"
                )
                status["Self-play B/W/D"] = (
                    f"{black}/{white}/{draws}; "
                    f"black {black / counted:.1%}, "
                    f"white {white / counted:.1%}, "
                    f"draw {draws / counted:.1%}"
                )

                progress.update(game_task, advance=1)
                live.update(render_dashboard(progress, status))

            actual_batch_size = min(batch_size, len(replay))
            metrics = []
            hide_search()

            for update in range(1, updates_per_iteration + 1):
                status["Stage"] = (
                    f"Round {iteration}/{final_iteration}: "
                    f"update {update}/{updates_per_iteration}"
                )

                batch = replay.sample(
                    actual_batch_size,
                    rng,
                    augmentation_rng=augmentation_rng,
                    fixed_fraction=human_fraction,
                )
                result = train_step(model, optimizer, batch)
                metrics.append(result)
                completed_updates += 1

                status["Loss: last batch"] = (
                    f"total {result['loss']:.4f} | "
                    f"policy {result['policy_loss']:.4f} | "
                    f"value {result['value_loss']:.4f}"
                )

                progress.update(update_task, advance=1)
                live.update(render_dashboard(progress, status))

            averages = {
                name: float(np.mean([m[name] for m in metrics]))
                for name in metrics[0]
            }

            status["Loss: round mean"] = (
                f"R{iteration}: total {averages['loss']:.4f} | "
                f"policy {averages['policy_loss']:.4f} | "
                f"value {averages['value_loss']:.4f}"
            )

            evaluation = None

            checkpoint_path = (
                run_dir / f"round_{iteration:03d}_games{completed_games:04d}.pt"
            )
            checkpoint = {
                "iteration": iteration,
                "games_completed": completed_games,
                "updates_completed": completed_updates,
                "model_config": model_config,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "metrics": averages,
                "evaluation": None,
                "seed": seed,
                "init_checkpoint": initial_source,
                "simulations": simulations,
                "runtime_config": {
                    "cpu_threads": torch.get_num_threads(),
                    "tree_reuse": True,
                    "color_plane": game_rules.COLOR_PLANE,
                    "exploration_moves": exploration_moves,
                    "win_check": True,
                    "fpu_reduction": 0.2,
                    "human_fraction": human_fraction,
                    "human_positions": len(replay.fixed),
                    "start_fraction": start_fraction,
                    "start_positions": len(start_positions),
                    "cache_size": 0,
                    "eval_every": eval_every,
                    "eval_games": eval_games,
                },
            }
            # Preserve completed updates before potentially lengthy evaluation.
            status["Stage"] = f"Round {iteration}: saving"
            live.update(render_dashboard(progress, status))
            save_checkpoint(checkpoint, checkpoint_path)
            status["Last checkpoint"] = checkpoint_path.name

            if iteration % eval_every == 0 or iteration == final_iteration:
                model.eval()
                results = []

                progress.update(
                    eval_task,
                    completed=0,
                    visible=True,
                    description=f"Eval R{iteration}",
                )

                for game in range(eval_games):
                    status["Stage"] = (
                        f"Round {iteration}/{final_iteration}: "
                        f"evaluation game {game + 1}/{eval_games}"
                    )
                    live.update(render_dashboard(progress, status))

                    result, _ = play_match(
                        model,
                        initial_opponent,
                        candidate_color=1 if game % 2 == 0 else -1,
                        mode="mcts",
                        simulations=simulations,
                        seed=int(eval_seeds[game]),
                        progress_callback=report_search,
                    )
                    results.append(result)

                    progress.update(eval_task, advance=1)
                    live.update(render_dashboard(progress, status))

                wins = results.count(1)
                draws = results.count(0)
                losses = results.count(-1)

                evaluation = {
                    "iteration": iteration,
                    "games": eval_games,
                    "simulations": simulations,
                    "wins": wins,
                    "draws": draws,
                    "losses": losses,
                    "win_rate": wins / eval_games,
                    "score": (wins + 0.5 * draws) / eval_games,
                }

                status["Eval vs initial"] = (
                    f"R{iteration}, n={eval_games}: "
                    f"win {evaluation['win_rate']:.1%}, "
                    f"score {evaluation['score']:.1%}, "
                    f"W/D/L={wins}/{draws}/{losses}"
                )
                model.train()
                hide_search()
                status["Stage"] = f"Round {iteration}: saving evaluation"
                live.update(render_dashboard(progress, status))
                checkpoint["evaluation"] = evaluation
                save_checkpoint(checkpoint, checkpoint_path)

            record = {
                "iteration": iteration,
                "games_completed": completed_games,
                "updates_completed": completed_updates,
                "buffer_size": len(replay),
                "human_positions": len(replay.fixed),
                "position_games": position_games,
                "metrics": averages,
                "self_play": {
                    "black_wins": color_results[1],
                    "white_wins": color_results[-1],
                    "draws": color_results[0],
                },
                "evaluation": evaluation,
            }

            # History first, then the resume point. A crash in between leaves an
            # extra history line, which trim_history() removes on resume.
            with history_path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(record) + "\n")
            # Atomic replace is the durable resume point. It includes the exact
            # replay, RNG streams, opponent, optimizer, and model for this round.
            save_checkpoint(pack_resume_state(
                model, optimizer, replay, rng, augmentation_rng, eval_rng,
                initial_opponent, iteration, completed_games, completed_updates,
                color_results, seed, config, run_dir, eval_seeds, initial_source,
            ), run_dir / "resume_state.bin")

            status["Last checkpoint"] = checkpoint_path.name
            progress.update(round_task, advance=1)
            live.update(render_dashboard(progress, status))

        status["Stage"] = "Completed"
        live.update(render_dashboard(progress, status))

    if pool is not None:
        pool.shutdown()

    return model, replay, run_dir


# Settings for `python train.py`. Only used when a NEW run starts; an exact
# resume always reuses the settings stored in its resume_state.bin.
SELF_PLAY_SETTINGS = {
    "games_per_iteration": 6,
    "simulations": 512,
    "updates_per_iteration": 50,
    "batch_size": 64,
    "eval_games": 2,
    "eval_every": 1000,
    # ~10% of each batch from your saved games. With several hundred human
    # positions vs a 10,000-sample replay, each human position is reused at
    # roughly the rate of a self-play sample (~0.3-0.5 times per round).
    "human_fraction": 0.1,
    # ~20% of self-play games (about 1 of 6 per round) start from a position of
    # a game it lost to you, so it practises exactly where it went wrong.
    "start_fraction": 0.2,
}


def continue_selfplay(rounds, workers=6, start_from=None):
    """Continue training from wherever the newest checkpoint is.

    - Newest checkpoint is in a self-play run with resume_state.bin: resume that
      run exactly (same replay, optimizer, random streams).
    - Otherwise (e.g. newest is a web-learning save, or SELF_PLAY_SETTINGS changed
      since that run started), or with start_from: start a new run from that
      model, carrying over the newest self-play replay memory.
    Before training, saved games the AI did not win are reviewed (review.py).
    """
    if start_from is None:
        latest = find_latest_checkpoint()
        state_path = latest.parent / "resume_state.bin"
        if state_path.exists():
            state = load_resume_state(state_path)
            settings = {k: v for k, v in state["config"].items() if k != "model_config"}
            if all(settings.get(k) == v for k, v in SELF_PLAY_SETTINGS.items()):
                review_games(checkpoint_root(), latest, workers)
                print(f"Resuming run {latest.parent.name} exactly", flush=True)
                return train(iterations=rounds, self_play_workers=workers, seed=state["seed"],
                             resume_from=state_path, **settings)
            print(f"Training settings changed since run {latest.parent.name}; "
                  "starting a new run from its newest model", flush=True)
        start_from = latest
    review_games(checkpoint_root(), start_from, workers)
    states = sorted(checkpoint_root().glob("*/resume_state.bin"), key=lambda p: p.stat().st_mtime_ns)
    carry = states[-1] if states else None
    print(f"New run from {Path(start_from).parent.name}/{Path(start_from).name}; replay "
          + (f"carried over from {carry.parent.name}" if carry else "starts empty"), flush=True)
    return train(iterations=rounds, self_play_workers=workers, seed=secrets.randbelow(2**31),
                 init_checkpoint=start_from, init_replay_from=carry, **SELF_PLAY_SETTINGS)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Self-play training that continues from the newest checkpoint "
                    "(self-play or web learning). Ctrl+C anytime; run again to continue.")
    parser.add_argument("--rounds", type=int, required=True,
                        help="Rounds to run now (each: 6 games + 50 updates)")
    parser.add_argument("--workers", type=int, default=6, help="Parallel self-play processes")
    parser.add_argument("--from", dest="start_from", type=Path, default=None,
                        help="Start a new run from this .pt instead of the newest checkpoint")
    args = parser.parse_args()
    if args.rounds < 1 or args.workers < 1:
        parser.error("rounds and workers must be positive")
    continue_selfplay(args.rounds, args.workers, args.start_from)
