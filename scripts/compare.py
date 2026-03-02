"""
Comprehensive comparison of Expert vs Consolidated models.

Generates publication-quality visualisations from multiple perspectives:
  1. Grouped bar chart  — mean reward per game per method
  2. Performance retention heatmap  — % of expert score retained
  3. Box plots  — per-episode reward distributions
  4. Radar / spider chart  — normalised multi-game profile
  5. Forgetting analysis  — reward gap (expert − consolidated)
  6. Relative performance bar  — % of expert per game
  7. Summary statistics table (also LaTeX-ready)
  8. Fisher / Hessian diagnostic plots (if htcl_fisher_log.json exists)

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
            hatch=HATCHES[i % len(HATCHES)],
            capsize=4, error_kw={"linewidth": 1.0, "capthick": 1.0},
        )

    ax.set_xlabel("Atari Game")
    ax.set_ylabel("Mean Episode Reward")
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
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=8, color="gray")
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
            hatch=HATCHES[(midx + 1) % len(HATCHES)],
        )

    ax.set_yticks(y)
    ax.set_yticklabels(games)
    ax.set_xlabel("Reward Gap  (Expert − Consolidated)")
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
            hatch=HATCHES[(midx + 1) % len(HATCHES)],
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
    """Render a publication-ready summary table as a figure image."""
    methods = list(results.keys())
    games = [r["game_name"] for r in results[methods[0]]]

    col_labels = games + ["Average"]
    cell_text = []
    for method in methods:
        row = []
        rewards = []
        for r in results[method]:
            row.append(f"{r['mean_reward']:.1f} ± {r['std_reward']:.1f}")
            rewards.append(r["mean_reward"])
        row.append(f"{np.mean(rewards):.1f}")
        cell_text.append(row)

    fig, ax = plt.subplots(
        figsize=(max(7, len(col_labels) * 2.2), max(2, len(methods) * 0.8 + 1)),
    )
    ax.axis("off")
    table = ax.table(
        cellText=cell_text, rowLabels=methods, colLabels=col_labels,
        loc="center", cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.6)

    # Style header
    for j in range(len(col_labels)):
        table[0, j].set_facecolor(PALETTE["pastel_blue"])
        table[0, j].set_edgecolor(EDGE_COLOR)
    for i in range(len(methods)):
        table[i + 1, -1].set_facecolor(PALETTE["pastel_yellow"])

    ax.set_title("Summary Statistics", fontsize=13, pad=20)
    _save_fig(fig, figure_dir, filename)


# ── Plot 8: Fisher / Hessian diagnostics ─────────────────────────────────────

def plot_fisher_diagnostics(fisher_log_path: str, figure_dir: str) -> None:
    """Plot Fisher diagnostics from the saved HTCL JSON log.

    Generates:
      * fisher_global_stats — line plot of global Fisher statistics vs task
      * fisher_layer_heatmap — heatmap of per-layer mean Fisher (cumulative)
      * fisher_per_task_stats — per-task (non-cumulative) statistics
    """
    if not os.path.exists(fisher_log_path):
        print(f"  Fisher log not found at {fisher_log_path}, skipping Fisher plots.")
        return

    with open(fisher_log_path, "r") as f:
        fisher_log = json.load(f)

    if not fisher_log:
        return

    cumulative = [e for e in fisher_log if e["kind"] == "cumulative"]
    per_task   = [e for e in fisher_log if e["kind"] == "task"]

    # ── 8a: Cumulative global stats ──
    if cumulative:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
        labels = [e["game_name"] for e in cumulative]
        xs = list(range(1, len(cumulative) + 1))

        for ax, stat, title in zip(
            axes,
            ["mean", "max", "nonzero_frac"],
            ["Mean Fisher (cumulative)", "Max Fisher (cumulative)",
             "Non-zero Fraction"],
        ):
            vals = [e["global"][stat] for e in cumulative]
            ax.plot(xs, vals, color=PALETTE["pastel_purple"],
                    marker="o", markeredgecolor=EDGE_COLOR,
                    markeredgewidth=0.8, markersize=8, linewidth=2.0)
            ax.set_xticks(xs)
            ax.set_xticklabels(labels, rotation=20, ha="right")
            ax.set_title(title)
            ax.set_xlabel("Task")

        fig.suptitle(
            "Cumulative Fisher (Hessian Diagonal) Diagnostics",
            fontsize=14, y=1.02,
        )
        _save_fig(fig, figure_dir, "fisher_global_stats")

    # ── 8b: Per-layer Fisher heatmap ──
    if cumulative:
        all_layers = list(cumulative[0]["per_layer"].keys())
        short = [
            n.replace("features.", "F.").replace("head.", "H.")
            for n in all_layers
        ]
        labels = [e["game_name"] for e in cumulative]

        mat = np.zeros((len(all_layers), len(cumulative)))
        for tidx, entry in enumerate(cumulative):
            for lidx, layer in enumerate(all_layers):
                mat[lidx, tidx] = entry["per_layer"].get(
                    layer, {},
                ).get("mean", 0.0)

        log_mat = np.log10(mat + 1e-12)
        cmap = LinearSegmentedColormap.from_list(
            "fisher_heat",
            ["#FFFFFF", "#A8D8EA", "#C3B1E1", "#F4B6C2"],
            N=256,
        )
        fig, ax = plt.subplots(
            figsize=(
                max(6, len(labels) * 2),
                max(5, len(all_layers) * 0.35),
            ),
        )
        im = ax.imshow(log_mat, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticks(range(len(all_layers)))
        ax.set_yticklabels(short, fontsize=8)
        ax.set_title("Per-Layer Mean Fisher  (log₁₀, cumulative)")
        plt.colorbar(im, ax=ax, label="log₁₀(mean Fisher)")
        _save_fig(fig, figure_dir, "fisher_layer_heatmap")

    # ── 8c: Per-task (non-cumulative) stats ──
    if per_task:
        labels = [e["game_name"] for e in per_task]
        xs = list(range(len(labels)))
        fig, ax = plt.subplots(figsize=(max(6, len(labels) * 2.5), 5))

        for stat, col, marker, lbl in [
            ("mean", PALETTE["pastel_blue"],  "o", "Mean"),
            ("max",  PALETTE["pastel_pink"],  "s", "Max"),
            ("std",  PALETTE["pastel_green"], "^", "Std"),
        ]:
            vals = [e["global"][stat] for e in per_task]
            ax.plot(
                xs, vals, color=col, marker=marker,
                markeredgecolor=EDGE_COLOR, markeredgewidth=0.8,
                markersize=7, linewidth=2.0, label=lbl,
            )

        ax.set_xticks(xs)
        ax.set_xticklabels(labels)
        ax.set_xlabel("Task")
        ax.set_ylabel("Fisher Statistic")
        ax.set_title("Per-Task Fisher Statistics  (not cumulative)")
        ax.legend()
        _save_fig(fig, figure_dir, "fisher_per_task_stats")

    print(f"  Fisher diagnostic plots saved to {figure_dir}")


# ── Plot 9: Lambda grid search results ──────────────────────────────────────

def plot_lambda_grid(
    grid_log_path: str, figure_dir: str,
) -> None:
    """Visualise HTCL lambda grid search results.

    Generates:
        (a) Lambda selection curve — avg KL vs log(lambda)
        (b) Per-task KL bar chart for each lambda candidate
    """
    if not os.path.exists(grid_log_path):
        print(f"  Lambda grid log not found at {grid_log_path}, skipping.")
        return

    with open(grid_log_path, "r") as f:
        grid_data = json.load(f)

    if not grid_data:
        print("  Lambda grid log is empty, skipping.")
        return

    lambdas = [e["lambda"] for e in grid_data]
    avg_kls = [e["avg_kl"] for e in grid_data]
    games = list(grid_data[0]["kl_per_task"].keys())

    # ── (a) Lambda selection curve ──
    fig, ax = plt.subplots(figsize=(6, 4))

    # Average KL
    ax.plot(
        lambdas, avg_kls,
        color=PALETTE["pastel_purple"], marker="o",
        markeredgecolor=EDGE_COLOR, markeredgewidth=1.0,
        markersize=8, linewidth=2.5, label="Avg KL",
        zorder=5,
    )

    # Per-task KL curves
    for gi, game in enumerate(games):
        task_kls = [e["kl_per_task"][game] for e in grid_data]
        ax.plot(
            lambdas, task_kls,
            color=COLORS[gi % len(COLORS)], marker=MARKERS[gi % len(MARKERS)],
            markeredgecolor=EDGE_COLOR, markeredgewidth=0.8,
            markersize=6, linewidth=1.5, alpha=0.7, label=game,
        )

    # Mark best lambda
    best_idx = int(np.argmin(avg_kls))
    ax.axvline(
        lambdas[best_idx], color=PALETTE["pastel_red"],
        linestyle="--", linewidth=1.5, alpha=0.8,
        label=f"Best λ={lambdas[best_idx]:.1f}",
    )

    ax.set_xscale("log")
    ax.set_xlabel("Lambda (λ)")
    ax.set_ylabel("KL Divergence")
    ax.set_title("HTCL Lambda Grid Search")
    ax.legend(fontsize=9, loc="best")
    _save_fig(fig, figure_dir, "lambda_selection_curve")

    # ── (b) Per-lambda bar charts ──
    for entry in grid_data:
        lam = entry["lambda"]
        kl_vals = [entry["kl_per_task"].get(g, 0) for g in games]

        fig, ax = plt.subplots(figsize=(max(5, len(games) * 1.5), 4))
        x = np.arange(len(games))
        bars = ax.bar(
            x, kl_vals,
            color=[COLORS[i % len(COLORS)] for i in range(len(games))],
            edgecolor=EDGE_COLOR, linewidth=1.2,
        )
        for bar_rect, val in zip(bars, kl_vals):
            ax.text(
                bar_rect.get_x() + bar_rect.get_width() / 2, val,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(games)
        ax.set_ylabel("KL(Expert ‖ Consolidated)")
        ax.set_title(f"KL Divergence per Task — λ = {lam}")
        _save_fig(fig, figure_dir, f"lambda_kl_lam{lam:.1f}")

    print(f"  Lambda grid plots saved to {figure_dir}")


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

    # ── Collect states and expert policies ──
    kl_data: dict = {}  # method -> {game: kl_value}

    for method in methods:
        kl_data[method] = {}
        method_key = method.lower()
        ckpt_path = os.path.join(
            checkpoint_dir, tag, f"consolidated_{method_key}.pt",
        )
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
                    kl = F2.kl_div(c_log_p, e_probs, reduction="batchmean")
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
            hatch=HATCHES[(i + 1) % len(HATCHES)],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(games)
    ax.set_ylabel("KL(Expert ‖ Consolidated)")
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
    methods = ["distillation", "htcl"]
    display_names = {"distillation": "Distillation", "htcl": "HTCL"}

    for method in methods:
        ckpt = os.path.join(
            checkpoint_dir, args.tag, f"consolidated_{method}.pt",
        )
        if not os.path.exists(ckpt):
            print(f"\nWARNING: Consolidated model not found for {method}, skipping.")
            continue

        display = display_names[method]
        print(f"\n{'=' * 60}")
        print(f"Evaluating {display} Consolidated Model")
        print(f"{'=' * 60}")

        model = load_model_checkpoint(ckpt, config, device)
        method_results = []

        for env_id in task_sequence:
            game = env_id.replace("NoFrameskip-v4", "")
            result = evaluate_on_task(
                model, env_id, config, device, union_actions, num_episodes,
            )
            method_results.append(result)
            print(
                f"  {display} on {game}: "
                f"{result['mean_reward']:.2f} ± {result['std_reward']:.2f}"
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

    # Fisher / Hessian diagnostics
    fisher_path = os.path.join(checkpoint_dir, args.tag, "htcl_fisher_log.json")
    print("\nGenerating Fisher / Hessian diagnostic plots...")
    plot_fisher_diagnostics(fisher_path, figure_dir)

    # Lambda grid search results
    grid_path = os.path.join(checkpoint_dir, args.tag, "htcl_lambda_grid.json")
    print("\nGenerating lambda grid search plots...")
    plot_lambda_grid(grid_path, figure_dir)

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
            pct_str = "  ".join(f"{p:6.1f}%" for p in pcts)
            print(f"  {method:<20} {pct_str}  (avg {avg_pct:.1f}%)")

    # LaTeX table
    print("\nLaTeX table:")
    print("\\begin{tabular}{l" + "c" * len(task_sequence) + "c}")
    print("\\toprule")
    cols = " & ".join(
        env_id.replace("NoFrameskip-v4", "") for env_id in task_sequence
    )
    print(f"Method & {cols} & Average \\\\")
    print("\\midrule")
    for method, rlist in all_results.items():
        rewards = [r["mean_reward"] for r in rlist]
        vals = " & ".join(f"{r:.1f}" for r in rewards)
        print(f"{method} & {vals} & {np.mean(rewards):.1f} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")


if __name__ == "__main__":
    main()
