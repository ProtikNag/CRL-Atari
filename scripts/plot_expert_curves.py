"""
Parse expert training logs and generate reward curves for each game.

Reads the training log to extract per-step evaluation rewards, then
generates publication-quality reward curves in both PNG and SVG formats.

Usage:
    python scripts/plot_expert_curves.py
    python scripts/plot_expert_curves.py --log results/logs/experts_default/training.log
    python scripts/plot_expert_curves.py --tag default
"""

import argparse
import os
import re
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Palette & Style (from visualization.instructions.md) ─────────────────────

PALETTE = {
    "pastel_blue":   "#A8D8EA",
    "pastel_pink":   "#F4B6C2",
    "pastel_green":  "#B5EAD7",
    "pastel_yellow": "#FFEEAD",
    "pastel_purple": "#C3B1E1",
    "pastel_orange": "#FFD8B1",
    "pastel_red":    "#F5A6A6",
    "pastel_teal":   "#A0D2DB",
}
COLORS = list(PALETTE.values())
EDGE_COLOR = "#1a1a1a"

# Assign a unique color + marker per game
GAME_STYLE = {
    "Pong":           {"color": PALETTE["pastel_blue"],   "marker": "o"},
    "Breakout":       {"color": PALETTE["pastel_orange"], "marker": "s"},
    "SpaceInvaders":  {"color": PALETTE["pastel_green"],  "marker": "^"},
}


def setup_mpl_style():
    """Apply the standard publication style from visualization rules."""
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        # Font
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        # Axes
        "axes.edgecolor": EDGE_COLOR,
        "axes.linewidth": 1.2,
        "axes.facecolor": "#FAFAFA",
        "axes.grid": True,
        "axes.axisbelow": True,
        # Grid
        "grid.color": "#E0E0E0",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.7,
        # Figure
        "figure.facecolor": "white",
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        # Legend
        "legend.frameon": True,
        "legend.edgecolor": EDGE_COLOR,
        "legend.fancybox": False,
        "legend.framealpha": 0.9,
        # Ticks
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
    })


def parse_log(log_path: str) -> Dict[str, List[Tuple[int, float]]]:
    """Parse training log and extract eval rewards per game.

    Extracts data only from the LAST training run in the log
    (handles multiple appended runs).

    Args:
        log_path: Path to training.log file.

    Returns:
        Dict mapping game name -> list of (step, reward) tuples.
    """
    # Pattern: [Pong] Eval at step 10000: mean reward = -21.00
    eval_pattern = re.compile(
        r"\[(\w+)\] Eval at step (\d+): mean reward = ([-\d.]+)"
    )
    # Pattern to detect a new run starting
    run_start_pattern = re.compile(r"Model parameters: [\d,]+")

    # Read all lines, find line index of last run start
    with open(log_path, "r") as f:
        lines = f.readlines()

    last_run_line = 0
    for i, line in enumerate(lines):
        if run_start_pattern.search(line):
            last_run_line = i

    # Parse eval entries from last run only
    data: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
    for line in lines[last_run_line:]:
        match = eval_pattern.search(line)
        if match:
            game = match.group(1)
            step = int(match.group(2))
            reward = float(match.group(3))
            data[game].append((step, reward))

    return dict(data)


def smooth(values: np.ndarray, window: int = 5) -> np.ndarray:
    """Simple moving average smoothing."""
    if window <= 1 or len(values) < window:
        return values
    kernel = np.ones(window) / window
    # Pad to avoid edge effects
    padded = np.concatenate([
        np.full(window // 2, values[0]),
        values,
        np.full(window // 2, values[-1]),
    ])
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed[:len(values)]


def plot_individual_curves(
    data: Dict[str, List[Tuple[int, float]]],
    output_dir: str,
    smooth_window: int = 5,
):
    """Generate one reward curve per game (separate figures)."""
    import matplotlib.pyplot as plt

    os.makedirs(os.path.join(output_dir, "png"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "svg"), exist_ok=True)

    for game, entries in data.items():
        steps = np.array([s for s, _ in entries])
        rewards = np.array([r for _, r in entries])
        smoothed = smooth(rewards, smooth_window)
        best_idx = np.argmax(rewards)

        style = GAME_STYLE.get(game, {"color": COLORS[0], "marker": "o"})

        fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)

        # Raw data (faded)
        ax.plot(
            steps, rewards,
            color=style["color"], alpha=0.3, linewidth=0.8,
            label="Raw",
        )

        # Smoothed curve
        ax.plot(
            steps, smoothed,
            color=style["color"], marker=style["marker"],
            markeredgecolor=EDGE_COLOR, markeredgewidth=0.8,
            markersize=4, linewidth=2.0, markevery=max(1, len(steps) // 15),
            label=f"Smoothed (w={smooth_window})",
        )

        # Best point
        ax.plot(
            steps[best_idx], rewards[best_idx],
            marker="*", color="#FF4444", markersize=14,
            markeredgecolor=EDGE_COLOR, markeredgewidth=1.0,
            zorder=10, label=f"Best: {rewards[best_idx]:.1f}",
        )

        ax.set_xlabel("Environment Steps")
        ax.set_ylabel("Mean Evaluation Reward")
        ax.set_title(f"{game} Expert — Training Reward Curve")
        ax.legend(loc="best")

        # Format x-axis in thousands/millions
        ax.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}k")
        )

        fname = f"expert_{game}_reward_curve"
        fig.savefig(
            os.path.join(output_dir, "png", f"{fname}.png"),
            dpi=300, bbox_inches="tight", facecolor="white",
        )
        fig.savefig(
            os.path.join(output_dir, "svg", f"{fname}.svg"),
            bbox_inches="tight", facecolor="white",
        )
        plt.close(fig)
        print(f"  Saved: {fname}.png / .svg")


def plot_combined_curve(
    data: Dict[str, List[Tuple[int, float]]],
    output_dir: str,
    smooth_window: int = 5,
):
    """Generate a combined multi-panel figure with all games."""
    import matplotlib.pyplot as plt

    n_games = len(data)
    fig, axes = plt.subplots(
        1, n_games, figsize=(5.5 * n_games, 4.5), constrained_layout=True,
    )
    if n_games == 1:
        axes = [axes]

    for ax, (game, entries) in zip(axes, data.items()):
        steps = np.array([s for s, _ in entries])
        rewards = np.array([r for _, r in entries])
        smoothed = smooth(rewards, smooth_window)
        best_idx = np.argmax(rewards)
        style = GAME_STYLE.get(game, {"color": COLORS[0], "marker": "o"})

        # Raw (faded)
        ax.plot(steps, rewards, color=style["color"], alpha=0.3, linewidth=0.8)

        # Smoothed
        ax.plot(
            steps, smoothed,
            color=style["color"], marker=style["marker"],
            markeredgecolor=EDGE_COLOR, markeredgewidth=0.8,
            markersize=4, linewidth=2.0, markevery=max(1, len(steps) // 12),
        )

        # Best point
        ax.plot(
            steps[best_idx], rewards[best_idx],
            marker="*", color="#FF4444", markersize=14,
            markeredgecolor=EDGE_COLOR, markeredgewidth=1.0,
            zorder=10,
        )

        ax.set_xlabel("Environment Steps")
        ax.set_ylabel("Mean Eval Reward")
        ax.set_title(f"{game} (Best: {rewards[best_idx]:.1f})")

        ax.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}k")
        )

    fig.suptitle(
        "Expert DQN Training — Evaluation Reward Curves",
        fontsize=14, fontweight="bold", y=1.02,
    )

    fname = "expert_all_reward_curves"
    fig.savefig(
        os.path.join(output_dir, "png", f"{fname}.png"),
        dpi=300, bbox_inches="tight", facecolor="white",
    )
    fig.savefig(
        os.path.join(output_dir, "svg", f"{fname}.svg"),
        bbox_inches="tight", facecolor="white",
    )
    plt.close(fig)
    print(f"  Saved: {fname}.png / .svg")


def plot_normalized_overlay(
    data: Dict[str, List[Tuple[int, float]]],
    output_dir: str,
    smooth_window: int = 5,
):
    """Overlay all games on a single axis with min-max normalized rewards.

    Useful for comparing relative training progress across games with
    different reward scales.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    markers = ["o", "s", "^", "D", "v", "P"]

    for i, (game, entries) in enumerate(data.items()):
        steps = np.array([s for s, _ in entries])
        rewards = np.array([r for _, r in entries])
        smoothed = smooth(rewards, smooth_window)

        # Min-max normalize
        rmin, rmax = rewards.min(), rewards.max()
        if rmax - rmin > 0:
            norm_smoothed = (smoothed - rmin) / (rmax - rmin)
        else:
            norm_smoothed = np.zeros_like(smoothed)

        style = GAME_STYLE.get(game, {"color": COLORS[i], "marker": markers[i]})

        ax.plot(
            steps, norm_smoothed,
            color=style["color"], marker=style["marker"],
            markeredgecolor=EDGE_COLOR, markeredgewidth=0.8,
            markersize=5, linewidth=2.0,
            markevery=max(1, len(steps) // 12),
            label=game,
        )

    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Normalized Reward (min-max)")
    ax.set_title("Expert Training Progress — Normalized Comparison")
    ax.legend(loc="best")
    ax.set_ylim(-0.05, 1.1)

    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}k")
    )

    fname = "expert_normalized_overlay"
    fig.savefig(
        os.path.join(output_dir, "png", f"{fname}.png"),
        dpi=300, bbox_inches="tight", facecolor="white",
    )
    fig.savefig(
        os.path.join(output_dir, "svg", f"{fname}.svg"),
        bbox_inches="tight", facecolor="white",
    )
    plt.close(fig)
    print(f"  Saved: {fname}.png / .svg")


def main():
    parser = argparse.ArgumentParser(
        description="Generate expert training reward curves."
    )
    parser.add_argument(
        "--log", type=str, default=None,
        help="Path to training.log. Auto-detected if omitted.",
    )
    parser.add_argument(
        "--tag", type=str, default="default",
        help="Experiment tag (for auto-detecting log path).",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results/figures",
        help="Output directory for figures.",
    )
    parser.add_argument(
        "--smooth", type=int, default=5,
        help="Smoothing window size (default: 5).",
    )
    args = parser.parse_args()

    # Resolve log path
    if args.log:
        log_path = args.log
    else:
        log_path = os.path.join("results", "logs", f"experts_{args.tag}", "training.log")

    if not os.path.exists(log_path):
        print(f"Error: Log file not found: {log_path}")
        sys.exit(1)

    print(f"Parsing log: {log_path}")

    # Apply matplotlib style
    setup_mpl_style()

    # Parse
    data = parse_log(log_path)
    if not data:
        print("Error: No eval data found in log.")
        sys.exit(1)

    for game, entries in data.items():
        rewards = [r for _, r in entries]
        print(f"  {game}: {len(entries)} eval points, "
              f"range [{min(rewards):.1f}, {max(rewards):.1f}], "
              f"best={max(rewards):.1f}")

    # Generate plots
    print("\nGenerating individual curves...")
    plot_individual_curves(data, args.output_dir, args.smooth)

    print("\nGenerating combined multi-panel figure...")
    plot_combined_curve(data, args.output_dir, args.smooth)

    print("\nGenerating normalized overlay...")
    plot_normalized_overlay(data, args.output_dir, args.smooth)

    print(f"\nAll figures saved to {args.output_dir}/png/ and {args.output_dir}/svg/")


if __name__ == "__main__":
    main()
