import json
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
import state as game_rules


class ReplayBuffer:
    def __init__(self, capacity=10000):
        if capacity < 1:
            raise ValueError("Buffer capacity must be positive")

        self.data = deque(maxlen=capacity)

    def __len__(self):
        return len(self.data)

    def add_game(self, samples):
        self.data.extend(samples)

    def sample(self, batch_size, rng, augmentation_rng=None):
        if not 1 <= batch_size <= len(self):
            raise ValueError("Invalid batch size")

        if augmentation_rng is None:
            augmentation_rng = rng

        indices = rng.choice(
            len(self),
            size=batch_size,
            replace=False,
        )

        observations = []
        policies = []
        outcomes = []

        for index in indices:
            observation, policy, outcome = self.data[int(index)]

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

def self_play_worker(model_config, state_dict, simulations, seed):
    """One self-play game in a worker process; same game as the serial path."""
    torch.set_num_threads(1)
    model = GomokuNetwork(**model_config)
    model.load_state_dict(state_dict)
    model.eval()
    return play_game(model, simulations=simulations, seed=seed)


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
):
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

    start = None
    if init_checkpoint is not None:
        # Continue from a saved model (e.g. one taught in web learning mode)
        # instead of random weights. The replay buffer still starts empty.
        start = torch.load(init_checkpoint, map_location="cpu", weights_only=True)
        model_config = start["model_config"]

    model = GomokuNetwork(**model_config)
    if start is not None:
        model.load_state_dict(start["model_state_dict"])

    # This opponent stays unchanged throughout the run
    # (the random start, or the loaded model when init_checkpoint is given).
    initial_opponent = deepcopy(model)
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
    replay = ReplayBuffer(buffer_capacity)

    # Readable run name: start time + search budget, e.g. 2026-11-10_1230_sims512.
    run_name = f"{datetime.now():%Y-%m-%d_%H%M}_sims{simulations}"
    run_dir = Path(__file__).resolve().parent / "checkpoints" / run_name
    suffix = 2
    while run_dir.exists():  # two runs started in the same minute
        run_dir = run_dir.with_name(f"{run_name}-{suffix}")
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    history_path = run_dir / "history.jsonl"

    total_games = iterations * games_per_iteration
    total_updates = iterations * updates_per_iteration

    color_results = {1: 0, -1: 0, 0: 0}
    completed_games = 0
    completed_updates = 0

    status = {
        "Stage": "Starting",
        "Search / batch": f"{simulations} simulations / up to {batch_size}",
        "CPU threads": str(torch.get_num_threads()),
        "Current move": "--",
        "Game elapsed": "--",
        "Last search": "--",
        "Loss: last batch": "--",
        "Loss: round mean": "--",
        "Replay buffer": f"0 / {buffer_capacity}",
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

        for iteration in range(1, iterations + 1):
            snapshot = deepcopy(model)
            snapshot.eval()
            snapshot.requires_grad_(False)

            # Same seeds, drawn in the same order, whether serial or parallel.
            seeds = [int(rng.integers(0, 2**32)) for _ in range(games_per_iteration)]
            if pool is not None:
                status["Stage"] = (
                    f"Round {iteration}/{iterations}: "
                    f"self-play on {self_play_workers} workers"
                )
                live.update(render_dashboard(progress, status))
                parallel_games = pool.map(
                    self_play_worker,
                    repeat(model_config),
                    repeat(snapshot.state_dict()),
                    repeat(simulations),
                    seeds,
                )

            for game in range(1, games_per_iteration + 1):
                if pool is not None:
                    samples, final_state = next(parallel_games)
                else:
                    status["Stage"] = (
                        f"Round {iteration}/{iterations}: "
                        f"self-play game {game}/{games_per_iteration}"
                    )
                    live.update(render_dashboard(progress, status))

                    samples, final_state = play_game(
                        snapshot,
                        simulations=simulations,
                        seed=seeds[game - 1],
                        progress_callback=report_search,
                    )

                replay.add_game(samples)
                completed_games += 1
                color_results[final_state.winner] += 1

                black = color_results[1]
                white = color_results[-1]
                draws = color_results[0]

                status["Replay buffer"] = (
                    f"{len(replay)} / {buffer_capacity}"
                )
                status["Self-play B/W/D"] = (
                    f"{black}/{white}/{draws}; "
                    f"black {black / completed_games:.1%}, "
                    f"white {white / completed_games:.1%}, "
                    f"draw {draws / completed_games:.1%}"
                )

                progress.update(game_task, advance=1)
                live.update(render_dashboard(progress, status))

            actual_batch_size = min(batch_size, len(replay))
            metrics = []
            hide_search()

            for update in range(1, updates_per_iteration + 1):
                status["Stage"] = (
                    f"Round {iteration}/{iterations}: "
                    f"update {update}/{updates_per_iteration}"
                )

                batch = replay.sample(
                    actual_batch_size,
                    rng,
                    augmentation_rng=augmentation_rng,
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
                "init_checkpoint": None if init_checkpoint is None else str(init_checkpoint),
                "simulations": simulations,
                "runtime_config": {
                    "cpu_threads": torch.get_num_threads(),
                    "tree_reuse": True,
                    "color_plane": game_rules.COLOR_PLANE,
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

            if iteration % eval_every == 0 or iteration == iterations:
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
                        f"Round {iteration}/{iterations}: "
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
                "metrics": averages,
                "self_play": {
                    "black_wins": color_results[1],
                    "white_wins": color_results[-1],
                    "draws": color_results[0],
                },
                "evaluation": evaluation,
            }

            with history_path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(record) + "\n")

            status["Last checkpoint"] = checkpoint_path.name
            progress.update(round_task, advance=1)
            live.update(render_dashboard(progress, status))

        status["Stage"] = "Completed"
        live.update(render_dashboard(progress, status))

    if pool is not None:
        pool.shutdown()

    return model, replay, run_dir


if __name__ == "__main__":
    train()
