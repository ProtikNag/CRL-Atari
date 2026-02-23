"""
Compare expert models vs consolidated models across all tasks.

Generates comparison tables and publication-quality plots.

Usage:
    python scripts/compare.py [--debug] [--tag TAG]
"""

import argparse
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

from src.models.dqn import DQNNetwork
from src.data.atari_wrappers import get_valid_actions
from src.utils.config import get_effective_config
from src.utils.seed import set_seed
from scripts.evaluate import evaluate_on_task, build_model

import torch

# ── Visualization palette and style ──────────────────────────────────────────

PALETTE = {
    "pastel_blue": "#A8D8EA",
    "pastel_pink": "#F4B6C2",
    "pastel_green": "#B5EAD7",
    "pastel_yellow": "#FFEEAD",
    "pastel_purple": "#C3B1E1",
    "pastel_orange": "#FFD8B1",
    "pastel_red": "#F5A6A6",
    "pastel_teal": "#A0D2DB",
}
COLORS = list(PALETTE.values())
EDGE_COLOR = "#1a1a1a"
HATCHES = [None, "//", "\\\\", "xx", "..", "||", "--", "++"]

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.edgecolor": EDGE_COLOR,
        "axes.linewidth": 1.2,
        "axes.facecolor": "#FAFAFA",
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#E0E0E0",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.7,
        "figure.facecolor": "white",
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "legend.frameon": True,
        "legend.edgecolor": EDGE_COLOR,
        "legend.fancybox": False,
        "legend.framealpha": 0.9,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
    }
)


def load_model_checkpoint(path: str, config: dict, device: str) -> DQNNetwork:
    """Load a model from checkpoint path."""
    model = build_model(config, device)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "policy_net" in checkpoint:
        model.load_state_dict(checkpoint["policy_net"])
    else:
        model.load_state_dict(checkpoint)
    return model


def plot_comparison_bar(
    results: dict,
    figure_dir: str,
    filename: str = "comparison",
):
    """Create a grouped bar chart comparing methods across games.

    Args:
        results: Dict mapping method_name -> list of {game, mean_reward, std_reward}.
        figure_dir: Output directory for figures.
        filename: Base filename (without extension).
    """
    methods = list(results.keys())
    games = [r["game_name"] for r in results[methods[0]]]
    num_games = len(games)
    num_methods = len(methods)

    fig, ax = plt.subplots(figsize=(max(7, num_games * 2.5), 5))

    bar_width = 0.8 / num_methods
    x = np.arange(num_games)

    for i, method in enumerate(methods):
        means = [r["mean_reward"] for r in results[method]]
        stds = [r["std_reward"] for r in results[method]]
        offset = (i - num_methods / 2 + 0.5) * bar_width

        bars = ax.bar(
            x + offset,
            means,
            bar_width,
            yerr=stds,
            label=method,
            color=COLORS[i % len(COLORS)],
            edgecolor=EDGE_COLOR,
            linewidth=1.2,
            hatch=HATCHES[i % len(HATCHES)],
            capsize=4,
            error_kw={"linewidth": 1.0, "capthick": 1.0},
        )

    ax.set_xlabel("Atari Game")
    ax.set_ylabel("Mean Episode Reward")
    ax.set_title("Expert vs. Consolidated Model Performance")
    ax.set_xticks(x)
    ax.set_xticklabels(games)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))

    os.makedirs(figure_dir, exist_ok=True)
    fig.savefig(
        os.path.join(figure_dir, f"{filename}.png"),
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    fig.savefig(
        os.path.join(figure_dir, f"{filename}.svg"),
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)
    print(f"Saved comparison plot to {figure_dir}/{filename}.png/.svg")


def plot_forgetting_heatmap(
    results: dict,
    figure_dir: str,
    filename: str = "forgetting_heatmap",
):
    """Create a heatmap showing performance of each method on each task.

    Args:
        results: Dict mapping method_name -> list of {game, mean_reward}.
        figure_dir: Output directory.
        filename: Base filename.
    """
    methods = list(results.keys())
    games = [r["game_name"] for r in results[methods[0]]]

    # Build matrix
    matrix = np.zeros((len(methods), len(games)))
    for i, method in enumerate(methods):
        for j, r in enumerate(results[method]):
            matrix[i, j] = r["mean_reward"]

    # Normalize per-game (show relative to expert)
    # Expert is always the first method
    if "Expert" in methods:
        expert_idx = methods.index("Expert")
        expert_vals = matrix[expert_idx]
        # Compute ratio to expert (percentage retained)
        norm_matrix = np.zeros_like(matrix)
        for i in range(len(methods)):
            for j in range(len(games)):
                if abs(expert_vals[j]) > 1e-6:
                    norm_matrix[i, j] = matrix[i, j] / abs(expert_vals[j]) * 100
                else:
                    norm_matrix[i, j] = 100.0
    else:
        norm_matrix = matrix

    pastel_cmap = LinearSegmentedColormap.from_list(
        "pastel_heat", ["#F5A6A6", "#FFEEAD", "#B5EAD7"], N=256
    )

    fig, ax = plt.subplots(figsize=(max(6, len(games) * 2), max(4, len(methods) * 1.2)))
    im = ax.imshow(norm_matrix, cmap=pastel_cmap, aspect="auto")

    # Annotate cells
    for i in range(len(methods)):
        for j in range(len(games)):
            raw_val = matrix[i, j]
            pct = norm_matrix[i, j]
            text_color = "black" if pct > 50 else "white"
            ax.text(
                j, i, f"{raw_val:.1f}\n({pct:.0f}%)",
                ha="center", va="center", fontsize=9, color=text_color,
            )

    ax.set_xticks(range(len(games)))
    ax.set_xticklabels(games)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods)
    ax.set_title("Performance Matrix (Raw Reward & % of Expert)")

    plt.colorbar(im, ax=ax, label="% of Expert Performance")

    os.makedirs(figure_dir, exist_ok=True)
    fig.savefig(
        os.path.join(figure_dir, f"{filename}.png"),
        dpi=300, bbox_inches="tight", facecolor="white",
    )
    fig.savefig(
        os.path.join(figure_dir, f"{filename}.svg"),
        bbox_inches="tight", facecolor="white",
    )
    plt.close(fig)
    print(f"Saved heatmap to {figure_dir}/{filename}.png/.svg")


def main():
    parser = argparse.ArgumentParser(description="Compare expert vs consolidated models.")
    parser.add_argument(
        "--config", type=str, default="configs/base.yaml", help="Config file."
    )
    parser.add_argument("--debug", action="store_true", help="Debug mode.")
    parser.add_argument("--device", type=str, default=None, help="Device.")
    parser.add_argument("--tag", type=str, default="default", help="Experiment tag.")
    parser.add_argument(
        "--episodes", type=int, default=None, help="Eval episodes per task."
    )
    args = parser.parse_args()

    config = get_effective_config(args.config, debug=args.debug)

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    set_seed(config["seed"])
    num_episodes = args.episodes or config["evaluation"]["episodes"]
    checkpoint_dir = config["logging"]["checkpoint_dir"]
    figure_dir = config["logging"]["figure_dir"]

    task_sequence = config["task_sequence"]
    all_results = {}

    # ── 1. Evaluate each expert on ALL tasks ──
    print("\n" + "=" * 60)
    print("Evaluating Expert Models")
    print("=" * 60)

    expert_per_task_results = []
    for env_id in task_sequence:
        game_name = env_id.replace("NoFrameskip-v4", "")
        ckpt_path = os.path.join(checkpoint_dir, args.tag, f"expert_{game_name}_best.pt")
        if not os.path.exists(ckpt_path):
            ckpt_path = os.path.join(
                checkpoint_dir, args.tag, f"expert_{game_name}_final.pt"
            )

        if not os.path.exists(ckpt_path):
            print(f"WARNING: Expert checkpoint not found for {game_name}, skipping.")
            continue

        model = load_model_checkpoint(ckpt_path, config, device)

        # Evaluate this expert on its own task
        result = evaluate_on_task(model, env_id, config, device, num_episodes)
        expert_per_task_results.append(result)
        print(
            f"  Expert ({game_name}) on {game_name}: "
            f"{result['mean_reward']:.2f} +/- {result['std_reward']:.2f}"
        )

    all_results["Expert"] = expert_per_task_results

    # ── 2. Evaluate consolidated models on ALL tasks ──
    methods = ["ewc", "distillation", "htcl"]
    method_names = {"ewc": "EWC", "distillation": "Distillation", "htcl": "HTCL"}

    for method in methods:
        ckpt_path = os.path.join(
            checkpoint_dir, args.tag, f"consolidated_{method}.pt"
        )
        if not os.path.exists(ckpt_path):
            print(f"\nWARNING: Consolidated model not found for {method}, skipping.")
            continue

        print(f"\n{'='*60}")
        print(f"Evaluating {method_names[method]} Consolidated Model")
        print(f"{'='*60}")

        model = load_model_checkpoint(ckpt_path, config, device)
        method_results = []

        for env_id in task_sequence:
            game_name = env_id.replace("NoFrameskip-v4", "")
            result = evaluate_on_task(model, env_id, config, device, num_episodes)
            method_results.append(result)
            print(
                f"  {method_names[method]} on {game_name}: "
                f"{result['mean_reward']:.2f} +/- {result['std_reward']:.2f}"
            )

        all_results[method_names[method]] = method_results

    # ── 3. Generate plots ──
    if len(all_results) > 1:
        print("\nGenerating comparison plots...")
        plot_comparison_bar(all_results, figure_dir, "comparison_bar")
        plot_forgetting_heatmap(all_results, figure_dir, "performance_heatmap")

    # ── 4. Save numerical results ──
    results_path = os.path.join(figure_dir, f"comparison_results_{args.tag}.json")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    # Convert for JSON serialization
    json_results = {}
    for method, results_list in all_results.items():
        json_results[method] = [
            {k: v for k, v in r.items() if k != "all_rewards"}
            for r in results_list
        ]

    with open(results_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"\nNumerical results saved to {results_path}")

    # ── 5. Print summary table ──
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    header = f"{'Method':<20}"
    for env_id in task_sequence:
        game = env_id.replace("NoFrameskip-v4", "")
        header += f"{'   ' + game:>15}"
    header += f"{'   Average':>15}"
    print(header)
    print("-" * 80)

    for method, results_list in all_results.items():
        row = f"{method:<20}"
        rewards = []
        for r in results_list:
            row += f"{r['mean_reward']:>15.2f}"
            rewards.append(r["mean_reward"])
        row += f"{np.mean(rewards):>15.2f}"
        print(row)
    print("=" * 80)


if __name__ == "__main__":
    main()
