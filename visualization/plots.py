"""Publication-ready plots for CRL-Atari experiments."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

STYLE_DEFAULTS = {
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
}

METHOD_COLORS = {
    "Individual": "#2196F3",
    "Sequential": "#F44336",
    "EWC": "#FF9800",
    "HTCL": "#4CAF50",
}

METHOD_HATCHES = {
    "Individual": "",
    "Sequential": "//",
    "EWC": "\\\\",
    "HTCL": "",
}


def _save(fig: plt.Figure, png_path: str, svg_path: str):
    """Save figure in both PNG and SVG formats."""
    fig.savefig(png_path, format="png")
    fig.savefig(svg_path, format="svg")
    plt.close(fig)


def plot_per_game_rewards(
    method_results: Dict[str, Dict[str, Dict[str, float]]],
    game_labels: Dict[str, str],
    png_path: str,
    svg_path: str,
):
    """
    Grouped bar chart: mean reward per game per method, with error bars.

    method_results[method_name][game_id] = {"mean_reward": ..., "std_reward": ...}
    """
    plt.rcParams.update(STYLE_DEFAULTS)

    methods = list(method_results.keys())
    games = list(game_labels.keys())
    labels = [game_labels[g] for g in games]
    n_methods = len(methods)
    n_games = len(games)

    width = 0.8 / n_methods
    x = np.arange(n_games)

    fig, ax = plt.subplots(figsize=(8, 4.5))

    for i, method in enumerate(methods):
        means = [method_results[method].get(g, {}).get("mean_reward", 0) for g in games]
        stds = [method_results[method].get(g, {}).get("std_reward", 0) for g in games]
        offset = (i - n_methods / 2 + 0.5) * width
        bars = ax.bar(
            x + offset, means, width * 0.9,
            yerr=stds, capsize=3,
            color=METHOD_COLORS.get(method, "#999"),
            hatch=METHOD_HATCHES.get(method, ""),
            edgecolor="black", linewidth=0.5,
            label=method, alpha=0.85,
        )

    ax.set_xlabel("Game")
    ax.set_ylabel("Mean Episode Reward")
    ax.set_title("Per-Game Performance Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(frameon=True, fancybox=False, edgecolor="black")
    ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    _save(fig, png_path, svg_path)


def plot_forgetting(
    forgetting_per_method: Dict[str, Dict[str, float]],
    game_labels: Dict[str, str],
    png_path: str,
    svg_path: str,
):
    """Bar chart of forgetting per game per method."""
    plt.rcParams.update(STYLE_DEFAULTS)

    methods = list(forgetting_per_method.keys())
    games = list(game_labels.keys())
    labels = [game_labels[g] for g in games]
    n_methods = len(methods)
    n_games = len(games)

    width = 0.8 / n_methods
    x = np.arange(n_games)

    fig, ax = plt.subplots(figsize=(8, 4.5))

    for i, method in enumerate(methods):
        vals = [forgetting_per_method[method].get(g, 0) for g in games]
        offset = (i - n_methods / 2 + 0.5) * width
        ax.bar(
            x + offset, vals, width * 0.9,
            color=METHOD_COLORS.get(method, "#999"),
            hatch=METHOD_HATCHES.get(method, ""),
            edgecolor="black", linewidth=0.5,
            label=method, alpha=0.85,
        )

    ax.set_xlabel("Game")
    ax.set_ylabel("Forgetting (Reward Drop)")
    ax.set_title("Catastrophic Forgetting per Game")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(frameon=True, fancybox=False, edgecolor="black")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    _save(fig, png_path, svg_path)


def plot_performance_heatmap(
    method_results: Dict[str, Dict[str, Dict[str, float]]],
    game_labels: Dict[str, str],
    png_path: str,
    svg_path: str,
):
    """
    Heatmap: rows = methods, columns = games, cell = mean reward.
    """
    plt.rcParams.update(STYLE_DEFAULTS)

    methods = list(method_results.keys())
    games = list(game_labels.keys())
    labels = [game_labels[g] for g in games]

    data = np.array([
        [method_results[m].get(g, {}).get("mean_reward", 0) for g in games]
        for m in methods
    ])

    fig, ax = plt.subplots(figsize=(7, 3.5))
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods)

    for i in range(len(methods)):
        for j in range(len(games)):
            val = data[i, j]
            color = "white" if abs(val) > 0.7 * np.max(np.abs(data)) else "black"
            ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                    fontsize=10, color=color, fontweight="bold")

    ax.set_title("Performance Matrix (Mean Reward)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Reward")

    fig.tight_layout()
    _save(fig, png_path, svg_path)


def plot_training_curves(
    training_logs: Dict[str, List[Dict]],
    game_labels: Dict[str, str],
    png_path: str,
    svg_path: str,
):
    """
    Training reward curves for local agents (one subplot per game).
    """
    plt.rcParams.update(STYLE_DEFAULTS)

    n_games = len(training_logs)
    fig, axes = plt.subplots(1, n_games, figsize=(5 * n_games, 4), sharey=False)
    if n_games == 1:
        axes = [axes]

    colors = ["#2196F3", "#4CAF50", "#F44336"]

    for idx, (game_id, log) in enumerate(training_logs.items()):
        ax = axes[idx]
        steps = [entry["step"] for entry in log]
        rewards = [entry["mean_reward_20"] for entry in log]

        ax.plot(steps, rewards, color=colors[idx % len(colors)], linewidth=1.5)
        ax.set_xlabel("Environment Steps")
        ax.set_ylabel("Mean Reward (20 ep)")
        ax.set_title(game_labels.get(game_id, game_id))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}k"))

    fig.suptitle("Local Agent Training Curves", fontsize=14, y=1.02)
    fig.tight_layout()
    _save(fig, png_path, svg_path)


def plot_forgetting_trajectory(
    trajectory: Dict[str, Dict[str, List[float]]],
    game_labels: Dict[str, str],
    game_ids: List[str],
    png_path: str,
    svg_path: str,
):
    """
    Line plot showing reward on each game after training on each
    successive game, for each method.

    trajectory[method][game_id] = [reward_after_game1, reward_after_game2, ...]
    """
    plt.rcParams.update(STYLE_DEFAULTS)

    methods = list(trajectory.keys())
    n_games = len(game_ids)

    fig, axes = plt.subplots(1, n_games, figsize=(5 * n_games, 4), sharey=False)
    if n_games == 1:
        axes = [axes]

    x_labels = [f"After {game_labels.get(g, g)}" for g in game_ids]

    for g_idx, gid in enumerate(game_ids):
        ax = axes[g_idx]
        for method in methods:
            if gid in trajectory.get(method, {}):
                values = trajectory[method][gid]
                ax.plot(
                    range(len(values)), values,
                    marker="o", markersize=5, linewidth=1.5,
                    color=METHOD_COLORS.get(method, "#999"),
                    label=method,
                )
        ax.set_xlabel("Training Stage")
        ax.set_ylabel("Mean Reward")
        ax.set_title(f"Performance on {game_labels.get(gid, gid)}")
        ax.set_xticks(range(n_games))
        ax.set_xticklabels([f"G{i+1}" for i in range(n_games)], rotation=0)
        ax.legend(frameon=True, fancybox=False, edgecolor="black", fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Reward Trajectory Across Training Stages", fontsize=14, y=1.02)
    fig.tight_layout()
    _save(fig, png_path, svg_path)


def plot_aggregate_summary(
    method_aggregates: Dict[str, Dict[str, float]],
    png_path: str,
    svg_path: str,
):
    """
    Summary bar chart: aggregate mean reward ± std across all games.

    method_aggregates[method] = {"mean_reward": ..., "std_reward": ...}
    """
    plt.rcParams.update(STYLE_DEFAULTS)

    methods = list(method_aggregates.keys())
    means = [method_aggregates[m]["mean_reward"] for m in methods]
    stds = [method_aggregates[m]["std_reward"] for m in methods]
    colors = [METHOD_COLORS.get(m, "#999") for m in methods]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(
        methods, means, yerr=stds, capsize=5,
        color=colors, edgecolor="black", linewidth=0.5, alpha=0.85,
    )

    for bar, mean in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            f"{mean:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    ax.set_ylabel("Mean Reward (All Games)")
    ax.set_title("Aggregate Performance Comparison")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")

    fig.tight_layout()
    _save(fig, png_path, svg_path)


def plot_radar_comparison(
    method_results: Dict[str, Dict[str, Dict[str, float]]],
    game_labels: Dict[str, str],
    png_path: str,
    svg_path: str,
):
    """Radar (spider) chart comparing methods across games."""
    plt.rcParams.update(STYLE_DEFAULTS)

    methods = list(method_results.keys())
    games = list(game_labels.keys())
    labels = [game_labels[g] for g in games]
    n_games = len(games)

    angles = np.linspace(0, 2 * np.pi, n_games, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"polar": True})

    for method in methods:
        values = [method_results[method].get(g, {}).get("mean_reward", 0) for g in games]
        values += values[:1]
        ax.plot(angles, values, linewidth=1.5,
                color=METHOD_COLORS.get(method, "#999"),
                label=method, marker="o", markersize=4)
        ax.fill(angles, values, alpha=0.1,
                color=METHOD_COLORS.get(method, "#999"))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_title("Multi-Game Performance Comparison", y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1),
              frameon=True, fancybox=False, edgecolor="black")

    fig.tight_layout()
    _save(fig, png_path, svg_path)
