"""
Comprehensive comparison of Expert vs Consolidated models.

Generates publication-quality visualisations from multiple perspectives:
  1. Grouped bar chart  -- mean reward per game per method
  2. Performance retention heatmap  -- % of expert score retained
  3. Box plots  -- per-episode reward distributions
  4. Radar / spider chart  -- normalised multi-game profile
  5. Forgetting analysis  -- reward gap (expert - consolidated)
  6. Relative performance bar  -- % of expert per game
  7. Summary statistics table (also LaTeX-ready)

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
from src.data.atari_wrappers import get_valid_actions, compute_union_action_space
from src.utils.config import get_effective_config
from src.utils.seed import set_seed
from scripts.evaluate import evaluate_on_task, build_model

import torch

# ── Visualisation palette and global rcParams ────────────────────────────────

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
HATCHES = [None, "//", "\\\\", "xx", "..", "||", "--", "++"]
MARKERS = ["o", "s", "^", "D", "v", "P"]

plt.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "DejaVu Serif"],
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.labelsize":    12,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "legend.fontsize":   10,
    "axes.edgecolor":    EDGE_COLOR,
    "axes.linewidth":    1.2,
    "axes.facecolor":    "#FAFAFA",
    "axes.grid":         True,
    "axes.axisbelow":    True,
    "grid.color":        "#E0E0E0",
    "grid.linewidth":    0.6,
    "grid.alpha":        0.7,
    "figure.facecolor":  "white",
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "savefig.facecolor": "white",
    "legend.frameon":    True,
    "legend.edgecolor":  EDGE_COLOR,
    "legend.fancybox":   False,
    "legend.framealpha": 0.9,
    "xtick.direction":   "out",
    "ytick.direction":   "out",
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
})


# ── Utility helpers ──────────────────────────────────────────────────────────

def _save_fig(fig: plt.Figure, figure_dir: str, filename: str) -> None:
    """Save a figure to both png/ and svg/ sub-directories."""
    for fmt in ("png", "svg"):
        out_dir = os.path.join(figure_dir, fmt)
        os.makedirs(out_dir, exist_ok=True)
        fig.savefig(
            os.path.join(out_dir, f"{filename}.{fmt}"),
            dpi=300, bbox_inches="tight", facecolor="white",
        )
    plt.close(fig)
    print(f"  Saved → {figure_dir}/{{png,svg}}/{filename}")


def _method_color(idx: int) -> str:
    return COLORS[idx % len(COLORS)]


def load_model_checkpoint(path: str, config: dict, device: str) -> DQNNetwork:
    """Load a model from checkpoint path."""
    model = build_model(config, device)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "policy_net" in checkpoint:
        model.load_state_dict(checkpoint["policy_net"])
    else:
        model.load_state_dict(checkpoint)
    return model


# ── Plot 1: Grouped bar chart ───────────────────────────────────────────────

def plot_grouped_bar(
    results: dict, figure_dir: str, filename: str = "comparison_bar",
) -> None:
    """Grouped bar chart comparing mean rewards across methods and games."""
    methods = list(results.keys())
    games = [r["game_name"] for r in results[methods[0]]]
    n_games = len(games)
    n_methods = len(methods)

    fig, ax = plt.subplots(figsize=(max(7, n_games * 2.5), 5))
    bw = 0.8 / n_methods
    x = np.arange(n_games)

    for i, method in enumerate(methods):
        means = [r["mean_reward"] for r in results[method]]
        stds  = [r["std_reward"]  for r in results[method]]
        offset = (i - n_methods / 2 + 0.5) * bw
        ax.bar(
            x + offset, means, bw, yerr=stds, label=method,
            color=_method_color(i), edgecolor=EDGE_COLOR, linewidth=1.2,
            capsize=4, error_kw={"linewidth": 1.0, "capthick": 1.0},
        )

    ax.set_xlabel("Atari Game")
    ax.set_ylabel("Mean Episode Reward")
    ax.set_yscale("symlog", linthresh=10)
    ax.set_title("Expert vs Consolidated Model Performance")
    ax.set_xticks(x)
    ax.set_xticklabels(games)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    _save_fig(fig, figure_dir, filename)


# ── Plot 2: Performance heatmap ─────────────────────────────────────────────

def plot_performance_heatmap(
    results: dict, figure_dir: str, filename: str = "performance_heatmap",
) -> None:
    """Heatmap showing raw rewards and percentage of expert performance."""
    methods = list(results.keys())
    games = [r["game_name"] for r in results[methods[0]]]

    matrix = np.zeros((len(methods), len(games)))
    for i, method in enumerate(methods):
        for j, r in enumerate(results[method]):
            matrix[i, j] = r["mean_reward"]

    # Normalise to expert
    norm = np.full_like(matrix, 100.0)
    if "Expert" in methods:
        eidx = methods.index("Expert")
        for j in range(len(games)):
            ev = abs(matrix[eidx, j])
            if ev > 1e-6:
                norm[:, j] = matrix[:, j] / ev * 100

    cmap = LinearSegmentedColormap.from_list(
        "pastel_heat", ["#F5A6A6", "#FFEEAD", "#B5EAD7"], N=256,
    )
    fig, ax = plt.subplots(
        figsize=(max(6, len(games) * 2), max(4, len(methods) * 1.2)),
    )
    im = ax.imshow(norm, cmap=cmap, aspect="auto", vmin=0, vmax=150)

    for i in range(len(methods)):
        for j in range(len(games)):
            tc = "black" if norm[i, j] > 50 else "white"
            ax.text(
                j, i, f"{matrix[i, j]:.1f}\n({norm[i, j]:.0f}%)",
                ha="center", va="center", fontsize=9, color=tc,
            )

    ax.set_xticks(range(len(games)))
    ax.set_xticklabels(games)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods)
    ax.set_title("Performance Matrix  (Raw Reward & % of Expert)")
    plt.colorbar(im, ax=ax, label="% of Expert Performance")
    _save_fig(fig, figure_dir, filename)


# ── Plot 3: Box-and-whisker distributions ────────────────────────────────────

def plot_box_distributions(
    results: dict, figure_dir: str, filename: str = "reward_distributions",
) -> None:
    """Box plots of per-episode reward distributions for every game."""
    methods = list(results.keys())
    games = [r["game_name"] for r in results[methods[0]]]
    n_games = len(games)

    fig, axes = plt.subplots(
        1, n_games, figsize=(5 * n_games, 5), sharey=False,
        constrained_layout=True,
    )
    if n_games == 1:
        axes = [axes]

    for gidx, game in enumerate(games):
        ax = axes[gidx]
        data, labels = [], []
        for midx, method in enumerate(methods):
            ep = results[method][gidx].get("all_rewards", [])
            if ep:
                data.append(ep)
                labels.append(method)

        if not data:
            continue

        bp = ax.boxplot(
            data, patch_artist=True, notch=True, widths=0.55,
            medianprops=dict(color="black", linewidth=1.5),
            whiskerprops=dict(color=EDGE_COLOR, linewidth=1.0),
            capprops=dict(color=EDGE_COLOR, linewidth=1.0),
            flierprops=dict(
                marker="o", markerfacecolor="#CCCCCC",
                markeredgecolor=EDGE_COLOR, markersize=4, alpha=0.6,
            ),
        )
        for pidx, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(_method_color(pidx))
            patch.set_edgecolor(EDGE_COLOR)
            patch.set_linewidth(1.2)

        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_title(game)
        ax.set_ylabel("Episode Reward")

    fig.suptitle("Per-Episode Reward Distributions", fontsize=14, y=1.02)
    _save_fig(fig, figure_dir, filename)


# ── Plot 4: Radar / spider chart ────────────────────────────────────────────

def plot_radar(
    results: dict, figure_dir: str, filename: str = "radar_chart",
) -> None:
    """Radar chart normalised to expert performance per game."""
    methods = list(results.keys())
    games = [r["game_name"] for r in results[methods[0]]]
    n_games = len(games)

    expert_vals = np.array(
        [r["mean_reward"] for r in results.get("Expert", results[methods[0]])]
    )
    norm_scores = {}
    for method in methods:
        raw = np.array([r["mean_reward"] for r in results[method]])
        safe = np.where(np.abs(expert_vals) > 1e-6, expert_vals, 1.0)
        norm_scores[method] = np.clip(raw / safe, 0, 1.5)

    angles = np.linspace(0, 2 * np.pi, n_games, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    for midx, method in enumerate(methods):
        vals = norm_scores[method].tolist() + [norm_scores[method][0]]
        ax.plot(
            angles, vals, color=COLORS[midx % len(COLORS)],
            linewidth=2.0, marker=MARKERS[midx % len(MARKERS)],
            markeredgecolor=EDGE_COLOR, markeredgewidth=0.8,
            markersize=7, label=method,
        )
        ax.fill(angles, vals, color=COLORS[midx % len(COLORS)], alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(games, fontsize=11)
    ax.set_ylim(0, 1.3)
    ax.set_yticks([0.01, 0.05, 0.1, 0.25, 0.5, 1.0])
    ax.set_yticklabels(["1%", "5%", "10%", "25%", "50%", "100%"],
                       fontsize=8, color="gray")
    ax.set_rscale("symlog", linthresh=0.05)
    ax.set_title("Normalised Performance Profile\n(100% = Expert)", y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1))
    _save_fig(fig, figure_dir, filename)


# ── Plot 5: Forgetting gap ──────────────────────────────────────────────────

def plot_forgetting_gap(
    results: dict, figure_dir: str, filename: str = "forgetting_gap",
) -> None:
    """Horizontal bar chart of reward gap (expert − consolidated) per game."""
    if "Expert" not in results:
        return
    methods = [m for m in results if m != "Expert"]
    if not methods:
        return

    games = [r["game_name"] for r in results["Expert"]]
    n_games = len(games)
    n_methods = len(methods)
    expert_means = np.array([r["mean_reward"] for r in results["Expert"]])

    fig, ax = plt.subplots(figsize=(8, max(4, n_games * 0.9)))
    bh = 0.8 / n_methods
    y = np.arange(n_games)

    for midx, method in enumerate(methods):
        m_means = np.array([r["mean_reward"] for r in results[method]])
        gaps = expert_means - m_means
        offset = (midx - n_methods / 2 + 0.5) * bh
        ax.barh(
            y + offset, gaps, bh, label=method,
            color=_method_color(midx + 1),
            edgecolor=EDGE_COLOR, linewidth=1.0,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(games)
    ax.set_xlabel("Reward Gap  (Expert − Consolidated)")
    ax.set_xscale("symlog", linthresh=10)
    ax.set_title("Forgetting Analysis  (positive = forgetting, negative = improvement)")
    ax.axvline(0, color=EDGE_COLOR, linewidth=0.8)
    ax.legend(loc="lower right")
    _save_fig(fig, figure_dir, filename)


# ── Plot 6: Relative performance bar ────────────────────────────────────────

def plot_relative_bar(
    results: dict, figure_dir: str, filename: str = "relative_performance",
) -> None:
    """Bar chart showing % of expert performance for each consolidated method."""
    if "Expert" not in results:
        return
    methods = [m for m in results if m != "Expert"]
    if not methods:
        return

    games = [r["game_name"] for r in results["Expert"]]
    expert_means = np.array([r["mean_reward"] for r in results["Expert"]])
    n_games = len(games)
    n_methods = len(methods)

    fig, ax = plt.subplots(figsize=(max(7, n_games * 2.5), 5))
    bw = 0.8 / n_methods
    x = np.arange(n_games)

    for midx, method in enumerate(methods):
        m_means = np.array([r["mean_reward"] for r in results[method]])
        safe = np.where(np.abs(expert_means) > 1e-6, expert_means, 1.0)
        pcts = m_means / safe * 100
        offset = (midx - n_methods / 2 + 0.5) * bw
        ax.bar(
            x + offset, pcts, bw, label=method,
            color=_method_color(midx + 1),
            edgecolor=EDGE_COLOR, linewidth=1.2,
        )
        for xi, pct in zip(x + offset, pcts):
            ax.text(xi, pct + 1, f"{pct:.0f}%", ha="center", va="bottom", fontsize=8)

    ax.axhline(100, color=EDGE_COLOR, linewidth=0.8, linestyle="--",
               label="Expert (100%)")
    ax.set_xlabel("Atari Game")
    ax.set_ylabel("% of Expert Reward")
    ax.set_title("Performance Retention  (% of Expert)")
    ax.set_xticks(x)
    ax.set_xticklabels(games)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    _save_fig(fig, figure_dir, filename)


# ── Plot 7: Summary table (rendered as figure) ──────────────────────────────

def plot_summary_table(
    results: dict, figure_dir: str, filename: str = "summary_table",
) -> None:
    """Render a publication-ready summary table of mean rewards as a figure.

    Rows correspond to methods (Expert, Distillation, HTCL per-lambda, etc.)
    and columns to per-game rewards plus an Avg % of Expert column.
    The best HTCL lambda row (by avg % retention) is highlighted in green.
    """
    methods = list(results.keys())
    games = [r["game_name"] for r in results[methods[0]]]

    # Compute expert baselines for % calculation
    expert_means = None
    if "Expert" in results:
        expert_means = np.array([r["mean_reward"] for r in results["Expert"]])

    col_labels = games + ["Avg % Expert"]
    cell_text = []
    row_labels = []

    best_htcl_pct = -float("inf")
    best_htcl_key = None

    for method in methods:
        row = []
        rewards = []
        for r in results[method]:
            row.append(f"{r['mean_reward']:.1f} \u00b1 {r['std_reward']:.1f}")
            rewards.append(r["mean_reward"])

        # Compute average % of expert
        if expert_means is not None and method != "Expert":
            safe_expert = np.where(
                np.abs(expert_means) > 1e-6, expert_means, 1.0,
            )
            pcts = np.array(rewards) / safe_expert * 100
            avg_pct = np.mean(pcts)
            row.append(f"{avg_pct:.1f}%")
        elif method == "Expert":
            avg_pct = 100.0
            row.append("100.0%")
        else:
            avg_pct = np.mean(rewards)
            row.append(f"{avg_pct:.1f}")

        cell_text.append(row)
        row_labels.append(method)

        if "HTCL" in method and "\u03bb" in method and avg_pct > best_htcl_pct:
            best_htcl_pct = avg_pct
            best_htcl_key = method

    n_rows = len(row_labels)
    fig, ax = plt.subplots(
        figsize=(
            max(7, len(col_labels) * 2.2),
            max(2, n_rows * 0.7 + 1),
        ),
    )
    ax.axis("off")
    table = ax.table(
        cellText=cell_text, rowLabels=row_labels, colLabels=col_labels,
        loc="center", cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.6)

    # Style header row
    for j in range(len(col_labels)):
        table[0, j].set_facecolor(PALETTE["pastel_blue"])
        table[0, j].set_edgecolor(EDGE_COLOR)

    # Style data rows
    for i, method in enumerate(row_labels):
        data_row = i + 1
        table[data_row, -1].set_facecolor(PALETTE["pastel_yellow"])
        if method == best_htcl_key:
            for j in range(-1, len(col_labels)):
                try:
                    table[data_row, j].set_facecolor(PALETTE["pastel_green"])
                    table[data_row, j].set_edgecolor(EDGE_COLOR)
                except KeyError:
                    pass
        elif method.startswith("Expert"):
            for j in range(-1, len(col_labels)):
                try:
                    table[data_row, j].set_facecolor(PALETTE["pastel_teal"])
                    table[data_row, j].set_edgecolor(EDGE_COLOR)
                except KeyError:
                    pass

    ax.set_title(
        "Reward Comparison (Mean \u00b1 Std, Avg as % of Expert)",
        fontsize=13, pad=20,
    )
    _save_fig(fig, figure_dir, filename)


# ── Plot 10: KL divergence between consolidated and expert policies ─────────

def plot_kl_divergence(
    all_results: dict,
    config: dict,
    device: str,
    union_actions: list,
    figure_dir: str,
) -> None:
    """Compute and plot KL(expert || consolidated) per task per method.

    Uses the same high-confidence state filtering as HTCL consolidation.
    Requires that expert and consolidated checkpoints exist.
    """
    import torch.nn.functional as F2  # local import to not shadow module-level
    from src.data.replay_buffer import ReplayBuffer
    from src.data.atari_wrappers import make_atari_env, get_valid_actions

    task_sequence = config["task_sequence"]
    checkpoint_dir = config["logging"]["checkpoint_dir"]
    tag = config.get("_tag", "default")

    debug_enabled = config.get("debug", {}).get("enabled", False)
    buffer_size = (
        config["debug"].get("buffer_size_per_task", 2000)
        if debug_enabled
        else config.get("buffer_size_per_task", 50_000)
    )
    filtered_size = min(1000, buffer_size // 5)  # Small sample for KL

    methods = [m for m in all_results if m != "Expert"]
    if not methods:
        print("  No consolidated methods available for KL comparison.")
        return

    # Map display names to checkpoint filenames
    METHOD_CKPT_MAP = {
        "Distillation":  "consolidated_distillation.pt",
        "One-Shot":      "consolidated_oneshot.pt",
        "Iterative":     "consolidated_iterative.pt",
        "Hybrid":        "consolidated_hybrid.pt",
    }

    # ── Collect states and expert policies ──
    kl_data: dict = {}  # method -> {game: kl_value}

    for method in methods:
        kl_data[method] = {}

        # Resolve checkpoint path
        if method in METHOD_CKPT_MAP:
            ckpt_fname = METHOD_CKPT_MAP[method]
        elif "HTCL" in method and "\u03bb" in method:
            # Legacy HTCL per-lambda: HTCL (λ=100) -> consolidated_htcl_lam100.0.pt
            lam_str = method.split("=")[1].rstrip(")")
            ckpt_fname = f"consolidated_htcl_lam{lam_str}.pt"
        else:
            method_key = method.lower().replace(" ", "_")
            ckpt_fname = f"consolidated_{method_key}.pt"

        ckpt_path = os.path.join(checkpoint_dir, tag, ckpt_fname)
        if not os.path.exists(ckpt_path):
            continue

        consol_model = load_model_checkpoint(ckpt_path, config, device)
        consol_model.eval()

        for env_id in task_sequence:
            game = env_id.replace("NoFrameskip-v4", "")
            expert_ckpt = os.path.join(
                checkpoint_dir, tag, f"expert_{game}_best.pt",
            )
            if not os.path.exists(expert_ckpt):
                continue

            expert_model = load_model_checkpoint(expert_ckpt, config, device)
            expert_model.eval()
            valid_actions = get_valid_actions(env_id, union_actions)

            # Collect small replay and get filtered states
            from scripts.consolidate import collect_replay_data

            replay_buf = collect_replay_data(
                env_id, expert_model, config, device, filtered_size,
                union_actions,
            )
            states = replay_buf.sample_states(filtered_size)

            # Compute KL(expert || consolidated)
            mask = torch.full(
                (consol_model.unified_action_dim,),
                float("-inf"), device=device,
            )
            mask[valid_actions] = 0.0

            total_kl = 0.0
            n_batches = 0
            with torch.no_grad():
                for start in range(0, len(states), 256):
                    batch = states[start : start + 256]
                    e_q = expert_model(batch)
                    e_probs = F2.softmax(e_q + mask.unsqueeze(0), dim=1)
                    c_q = consol_model(batch)
                    c_log_p = F2.log_softmax(c_q + mask.unsqueeze(0), dim=1)
                    # Restrict to valid actions only to avoid NaN
                    # from 0 * log(0) at masked positions
                    kl = F2.kl_div(
                        c_log_p[:, valid_actions],
                        e_probs[:, valid_actions],
                        reduction="batchmean",
                    )
                    total_kl += kl.item()
                    n_batches += 1

            kl_data[method][game] = total_kl / max(n_batches, 1)

    # ── Plot ──
    games = [e.replace("NoFrameskip-v4", "") for e in task_sequence]
    valid_methods = [m for m in methods if kl_data.get(m)]
    if not valid_methods:
        print("  No KL data computed — skipping plot.")
        return

    n_methods = len(valid_methods)
    fig, ax = plt.subplots(figsize=(max(6, len(games) * 2), 5))
    x = np.arange(len(games))
    bw = 0.8 / n_methods

    for i, method in enumerate(valid_methods):
        vals = [kl_data[method].get(g, 0) for g in games]
        offset = (i - n_methods / 2 + 0.5) * bw
        ax.bar(
            x + offset, vals, bw, label=method,
            color=_method_color(i + 1), edgecolor=EDGE_COLOR, linewidth=1.2,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(games)
    ax.set_ylabel("KL(Expert ‖ Consolidated)")
    ax.set_yscale("symlog", linthresh=0.1)
    ax.set_title("Policy Divergence: Expert vs Consolidated")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    _save_fig(fig, figure_dir, "kl_divergence_comparison")
    print(f"  KL divergence plot saved to {figure_dir}")



# ── Main evaluation pipeline ────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive comparison: expert vs consolidated models.",
    )
    parser.add_argument(
        "--config", type=str, default="configs/base.yaml", help="Config file.",
    )
    parser.add_argument("--debug", action="store_true", help="Debug mode.")
    parser.add_argument("--device", type=str, default=None, help="Device.")
    parser.add_argument("--tag", type=str, default="default", help="Experiment tag.")
    parser.add_argument(
        "--episodes", type=int, default=None, help="Eval episodes per task.",
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

    union_actions = compute_union_action_space(config["task_sequence"])
    config["model"]["unified_action_dim"] = len(union_actions)

    task_sequence = config["task_sequence"]
    all_results: dict = {}

    # ── 1. Evaluate each expert on its own task ─────────────────────────────
    print("\n" + "=" * 60)
    print("Evaluating Expert Models")
    print("=" * 60)

    expert_results = []
    for env_id in task_sequence:
        game = env_id.replace("NoFrameskip-v4", "")
        ckpt = os.path.join(checkpoint_dir, args.tag, f"expert_{game}_best.pt")
        if not os.path.exists(ckpt):
            print(f"WARNING: Expert checkpoint not found for {game}, skipping.")
            continue

        model = load_model_checkpoint(ckpt, config, device)
        result = evaluate_on_task(
            model, env_id, config, device, union_actions, num_episodes,
        )
        expert_results.append(result)
        print(
            f"  Expert ({game}): "
            f"{result['mean_reward']:.2f} ± {result['std_reward']:.2f}"
        )

    all_results["Expert"] = expert_results

    # ── 2. Evaluate consolidated models on ALL tasks ────────────────────────

    # Named consolidated methods to discover
    CONSOLIDATED_METHODS = [
        ("Distillation",  "consolidated_distillation.pt"),
        ("One-Shot",      "consolidated_oneshot.pt"),
        ("Iterative",     "consolidated_iterative.pt"),
        ("Hybrid",        "consolidated_hybrid.pt"),
    ]

    for display_name, ckpt_fname in CONSOLIDATED_METHODS:
        ckpt_path = os.path.join(checkpoint_dir, args.tag, ckpt_fname)
        if not os.path.exists(ckpt_path):
            print(f"\nWARNING: {display_name} checkpoint not found, skipping.")
            continue

        print(f"\n{'=' * 60}")
        print(f"Evaluating {display_name} Consolidated Model")
        print("=" * 60)

        model = load_model_checkpoint(ckpt_path, config, device)
        method_results = []
        for env_id in task_sequence:
            game = env_id.replace("NoFrameskip-v4", "")
            result = evaluate_on_task(
                model, env_id, config, device, union_actions, num_episodes,
            )
            method_results.append(result)
            print(
                f"  {display_name} on {game}: "
                f"{result['mean_reward']:.2f} \u00b1 {result['std_reward']:.2f}"
            )
        all_results[display_name] = method_results

    # Also pick up any HTCL per-lambda checkpoints (legacy grid search)
    import glob as _glob

    htcl_pattern = os.path.join(
        checkpoint_dir, args.tag, "consolidated_htcl_lam*.pt",
    )
    htcl_ckpts = sorted(_glob.glob(htcl_pattern))

    for ckpt_path in htcl_ckpts:
        fname = os.path.basename(ckpt_path)
        lam_str = fname.replace("consolidated_htcl_lam", "").replace(".pt", "")
        try:
            lam_val = float(lam_str)
        except ValueError:
            continue

        display = f"HTCL (\u03bb={lam_val:g})"
        print(f"\n{'=' * 60}")
        print(f"Evaluating {display}")
        print("=" * 60)

        model = load_model_checkpoint(ckpt_path, config, device)
        method_results = []
        for env_id in task_sequence:
            game = env_id.replace("NoFrameskip-v4", "")
            result = evaluate_on_task(
                model, env_id, config, device, union_actions, num_episodes,
            )
            method_results.append(result)
            print(
                f"  {display} on {game}: "
                f"{result['mean_reward']:.2f} \u00b1 {result['std_reward']:.2f}"
            )
        all_results[display] = method_results

    # ── 3. Generate all visualisations ──────────────────────────────────────
    if len(all_results) > 1:
        print("\nGenerating comparison visualisations...")
        plot_grouped_bar(all_results, figure_dir, "comparison_bar")
        plot_performance_heatmap(all_results, figure_dir, "performance_heatmap")
        plot_box_distributions(all_results, figure_dir, "reward_distributions")
        plot_radar(all_results, figure_dir, "radar_chart")
        plot_forgetting_gap(all_results, figure_dir, "forgetting_gap")
        plot_relative_bar(all_results, figure_dir, "relative_performance")
        plot_summary_table(all_results, figure_dir, "summary_table")

    # KL divergence analysis
    print("\nComputing KL divergence between consolidated and expert policies...")
    config["_tag"] = args.tag  # Pass tag through config for convenience
    plot_kl_divergence(all_results, config, device, union_actions, figure_dir)

    # ── 4. Save numerical results (JSON) ────────────────────────────────────
    results_path = os.path.join(figure_dir, f"comparison_results_{args.tag}.json")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    json_results = {}
    for method, rlist in all_results.items():
        json_results[method] = [
            {k: v for k, v in r.items() if k != "all_rewards"} for r in rlist
        ]

    # Compute and save retention + geometric mean (Rusu et al. 2016)
    if "Expert" in all_results:
        expert_means = [r["mean_reward"] for r in all_results["Expert"]]
        retention_summary = {}
        for method, rlist in all_results.items():
            if method == "Expert":
                continue
            m_means = [r["mean_reward"] for r in rlist]
            pcts = [
                mm / abs(em) * 100 if abs(em) > 1e-6 else 100.0
                for em, mm in zip(expert_means, m_means)
            ]
            pcts_clamped = [max(p, 0.01) for p in pcts]
            geo_mean_pct = float(np.exp(np.mean(np.log(pcts_clamped))))
            retention_summary[method] = {
                "per_task_pct": pcts,
                "avg_pct": float(np.mean(pcts)),
                "geometric_mean_pct": geo_mean_pct,
            }
        json_results["_retention_summary"] = retention_summary

    with open(results_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"\nNumerical results saved to {results_path}")

    # Full results with all_rewards for follow-up analysis
    full_path = os.path.join(figure_dir, f"comparison_full_{args.tag}.json")
    with open(full_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # ── 5. Print summary table ──────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)

    header = f"{'Method':<20}"
    for env_id in task_sequence:
        g = env_id.replace("NoFrameskip-v4", "")
        header += f"{'   ' + g:>15}"
    header += f"{'   Average':>15}"
    print(header)
    print("-" * 80)

    for method, rlist in all_results.items():
        row = f"{method:<20}"
        rewards = []
        for r in rlist:
            row += f"{r['mean_reward']:>15.2f}"
            rewards.append(r["mean_reward"])
        row += f"{np.mean(rewards):>15.2f}"
        print(row)
    print("=" * 80)

    # Retention percentages
    if "Expert" in all_results:
        expert_means = [r["mean_reward"] for r in all_results["Expert"]]
        print("\nRetention (% of Expert):")
        for method, rlist in all_results.items():
            if method == "Expert":
                continue
            m_means = [r["mean_reward"] for r in rlist]
            pcts = [
                mm / abs(em) * 100 if abs(em) > 1e-6 else 100.0
                for em, mm in zip(expert_means, m_means)
            ]
            avg_pct = np.mean(pcts)
            # Geometric mean of retention (Rusu et al. 2016, Policy Distillation)
            # Clamp negatives to a small positive value to allow gmean computation
            pcts_clamped = [max(p, 0.01) for p in pcts]
            geo_mean_pct = float(np.exp(np.mean(np.log(pcts_clamped))))
            pct_str = "  ".join(f"{p:6.1f}%" for p in pcts)
            print(
                f"  {method:<20} {pct_str}  "
                f"(avg {avg_pct:.1f}%, gmean {geo_mean_pct:.1f}%)"
            )

    # LaTeX table (with geometric mean column)
    print("\nLaTeX table:")
    print("\\begin{tabular}{l" + "c" * len(task_sequence) + "cc}")
    print("\\toprule")
    cols = " & ".join(
        env_id.replace("NoFrameskip-v4", "") for env_id in task_sequence
    )
    print(f"Method & {cols} & Average & Geo. Mean \\\\")
    print("\\midrule")
    for method, rlist in all_results.items():
        rewards = [r["mean_reward"] for r in rlist]
        vals = " & ".join(f"{r:.1f}" for r in rewards)
        avg_r = np.mean(rewards)
        # Geometric mean retention for consolidated methods
        if method == "Expert" or "Expert" not in all_results:
            print(f"{method} & {vals} & {avg_r:.1f} & --- \\\\")
        else:
            m_means = [r["mean_reward"] for r in rlist]
            pcts = [
                mm / abs(em) * 100 if abs(em) > 1e-6 else 100.0
                for em, mm in zip(expert_means, m_means)
            ]
            pcts_clamped = [max(p, 0.01) for p in pcts]
            geo_mean_pct = float(np.exp(np.mean(np.log(pcts_clamped))))
            print(f"{method} & {vals} & {avg_r:.1f} & {geo_mean_pct:.1f}\\% \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")


if __name__ == "__main__":
    main()
