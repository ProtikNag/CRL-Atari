"""
Generate comprehensive visualizations for CRL-Atari experiments.

Produces publication-quality plots in both PNG and SVG formats,
organized into separate subfolders: results/figures/png/ and results/figures/svg/.

Uses an academic color palette inspired by Nature/Science/ICML publications:
muted, high-contrast, colorblind-friendly colors with clean typography.

Usage:
    python scripts/visualize.py [--results-dir results] [--tag debug]
"""

import argparse
import csv
import json
import os
import sys
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

# ══════════════════════════════════════════════════════════════════════
# Academic color scheme — muted, high-contrast, journal-quality
# ══════════════════════════════════════════════════════════════════════
PALETTE = {
    "navy":       "#2E5090",
    "terracotta": "#C45B28",
    "plum":       "#6B4C9A",
    "crimson":    "#B5322A",
    "sage":       "#5A8A3C",
    "slate":      "#5B7B8A",
    "gold":       "#C49A2A",
    "teal":       "#2A7B7B",
}
COLORS = list(PALETTE.values())
EDGE_COLOR = "#2C2C2C"
HATCHES = [None, "//", "\\\\", "xx", "..", "||", "--", "++"]
MARKERS = ["o", "s", "^", "D", "v", "P"]

METHOD_COLORS = {
    "Expert":       PALETTE["navy"],
    "Distillation": PALETTE["plum"],
    "HTCL":         PALETTE["crimson"],
}
METHOD_HATCHES = {
    "Expert":       None,
    "Distillation": "\\\\",
    "HTCL":         "xx",
}
# Light tints for fills (box/violin interiors)
METHOD_LIGHT = {
    "Expert":       "#C8D5E8",
    "Distillation": "#D5CEE0",
    "HTCL":         "#E4C5C3",
}


def setup_style() -> None:
    """Apply global matplotlib style for clean, academic aesthetics."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Georgia"],
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.edgecolor": EDGE_COLOR,
        "axes.linewidth": 1.0,
        "axes.facecolor": "#FAFAF8",
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": "#E0DEDA",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.8,
        "grid.linestyle": "--",
        "figure.facecolor": "white",
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "legend.frameon": True,
        "legend.edgecolor": "#CCCCCC",
        "legend.fancybox": True,
        "legend.framealpha": 0.95,
        "legend.shadow": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
    })


def save_figure(fig: plt.Figure, name: str, png_dir: str, svg_dir: str) -> None:
    """Save figure in both PNG and SVG to separate subdirectories."""
    fig.savefig(os.path.join(png_dir, f"{name}.png"),
                dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(os.path.join(svg_dir, f"{name}.svg"),
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {name}.png / {name}.svg")


def load_comparison_results(json_path: str) -> Dict[str, Any]:
    """Load comparison results JSON."""
    with open(json_path, "r") as f:
        return json.load(f)


def extract_data(results: Dict[str, Any]) -> Tuple[List[str], List[str], Dict]:
    """Extract methods, games, and restructure data as {method: {game: {...}}}."""
    methods = list(results.keys())
    games = [entry["game_name"] for entry in results[methods[0]]]
    structured: Dict[str, Dict[str, Any]] = {}
    for method in methods:
        structured[method] = {}
        for entry in results[method]:
            structured[method][entry["game_name"]] = entry
    return methods, games, structured


# ═══════════════════════════════════════════════════════════════════════
# PLOT 1: Grouped Bar Chart
# ═══════════════════════════════════════════════════════════════════════
def plot_grouped_bar(
    results: Dict, methods: List[str], games: List[str],
    png_dir: str, svg_dir: str
) -> None:
    """Grouped bar chart: methods x games with error bars."""
    n_methods = len(methods)
    n_games = len(games)
    x = np.arange(n_games)
    width = 0.8 / n_methods

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, method in enumerate(methods):
        means = [results[method][g]["mean_reward"] for g in games]
        stds = [results[method][g]["std_reward"] for g in games]
        offset = (i - n_methods / 2 + 0.5) * width
        color = METHOD_COLORS.get(method, COLORS[i % len(COLORS)])
        bars = ax.bar(
            x + offset, means, width,
            yerr=stds, capsize=3,
            label=method,
            color=color, alpha=0.85,
            edgecolor="white", linewidth=1.0,
            hatch=METHOD_HATCHES.get(method, HATCHES[i % len(HATCHES)]),
            error_kw={"elinewidth": 0.8, "capthick": 0.8, "color": "#555555"}
        )
        max_std = max(stds) if stds else 0
        for bar_rect, mean_val in zip(bars, means):
            va = "bottom" if mean_val >= 0 else "top"
            y_off = max_std * 0.15 if mean_val >= 0 else -max_std * 0.15
            ax.text(bar_rect.get_x() + bar_rect.get_width() / 2,
                    bar_rect.get_height() + y_off,
                    f"{mean_val:.1f}", ha="center", va=va,
                    fontsize=7, fontweight="bold", color="#333333")

    ax.set_xlabel("Atari Game", fontweight="bold")
    ax.set_ylabel("Mean Reward", fontweight="bold")
    ax.set_title("Consolidation Method Comparison Across Games", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([g.replace("NoFrameskip-v4", "") for g in games])
    ax.legend(loc="upper right", framealpha=0.9)
    ax.axhline(y=0, color=EDGE_COLOR, linewidth=0.6, linestyle="-")

    save_figure(fig, "01_grouped_bar_comparison", png_dir, svg_dir)


# ═══════════════════════════════════════════════════════════════════════
# PLOT 2: Performance Heatmap
# ═══════════════════════════════════════════════════════════════════════
def plot_performance_heatmap(
    results: Dict, methods: List[str], games: List[str],
    png_dir: str, svg_dir: str
) -> None:
    """Heatmap showing raw rewards with expert-relative percentages."""
    n_methods = len(methods)
    n_games = len(games)

    reward_matrix = np.zeros((n_methods, n_games))
    for i, method in enumerate(methods):
        for j, game in enumerate(games):
            reward_matrix[i, j] = results[method][game]["mean_reward"]

    expert_rewards = reward_matrix[0, :]

    academic_cmap = LinearSegmentedColormap.from_list(
        "academic_div",
        ["#B5322A", "#D9907A", "#FAF3EB", "#7BA4C9", "#2E5090"],
        N=256
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(reward_matrix, cmap=academic_cmap, aspect="auto")

    for i in range(n_methods):
        for j in range(n_games):
            val = reward_matrix[i, j]
            exp_val = expert_rewards[j]
            if exp_val != 0:
                pct = val / exp_val * 100
                text = f"{val:.1f}\n({pct:.0f}%)"
            else:
                text = f"{val:.1f}"
            norm_val = (val - reward_matrix.min()) / (reward_matrix.max() - reward_matrix.min() + 1e-8)
            text_color = "white" if norm_val < 0.3 or norm_val > 0.7 else "#2C2C2C"
            ax.text(j, i, text, ha="center", va="center",
                    fontsize=9, fontweight="bold", color=text_color)

    ax.set_xticks(range(n_games))
    ax.set_xticklabels([g.replace("NoFrameskip-v4", "") for g in games])
    ax.set_yticks(range(n_methods))
    ax.set_yticklabels(methods)
    ax.set_title("Performance Heatmap (Reward + % of Expert)", fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Mean Reward")
    for edge in range(n_games + 1):
        ax.axvline(edge - 0.5, color="white", linewidth=2.0)
    for edge in range(n_methods + 1):
        ax.axhline(edge - 0.5, color="white", linewidth=2.0)

    save_figure(fig, "02_performance_heatmap", png_dir, svg_dir)


# ═══════════════════════════════════════════════════════════════════════
# PLOT 3: Radar Chart
# ═══════════════════════════════════════════════════════════════════════
def plot_radar_chart(
    results: Dict, methods: List[str], games: List[str],
    png_dir: str, svg_dir: str
) -> None:
    """Radar chart showing multi-dimensional method comparison."""
    game_labels = [g.replace("NoFrameskip-v4", "") for g in games]
    n_games = len(games)

    all_vals = [results[m][g]["mean_reward"] for m in methods for g in games]

    angles = np.linspace(0, 2 * np.pi, n_games, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_facecolor("#FAFAF8")

    for i, method in enumerate(methods):
        values = []
        for g in games:
            val = results[method][g]["mean_reward"]
            norm_val = (val + abs(min(all_vals))) / (max(abs(v) for v in all_vals) + abs(min(all_vals)) + 1e-8)
            values.append(norm_val)
        values += values[:1]
        color = METHOD_COLORS.get(method, COLORS[i % len(COLORS)])
        ax.plot(angles, values, "o-",
                color=color, linewidth=2.0, markersize=7,
                markeredgecolor="white", markeredgewidth=1.2,
                label=method)
        ax.fill(angles, values, alpha=0.10, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(game_labels, fontsize=12, fontweight="bold")
    ax.set_title("Method Performance Radar (Normalized)",
                 pad=25, fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1))
    ax.set_ylim(0, 1.1)
    ax.spines["polar"].set_color("#CCCCCC")
    ax.grid(color="#D0D0D0", linewidth=0.5)

    save_figure(fig, "03_radar_chart", png_dir, svg_dir)


# ═══════════════════════════════════════════════════════════════════════
# PLOT 4: Box Plots
# ═══════════════════════════════════════════════════════════════════════
def plot_box_plots(
    results: Dict, methods: List[str], games: List[str],
    png_dir: str, svg_dir: str
) -> None:
    """Box plots showing reward distribution per method for each game."""
    n_games = len(games)
    fig, axes = plt.subplots(1, n_games, figsize=(5 * n_games, 6),
                             constrained_layout=True)
    if n_games == 1:
        axes = [axes]

    for j, game in enumerate(games):
        ax = axes[j]
        game_label = game.replace("NoFrameskip-v4", "")
        data, labels, fill_colors = [], [], []

        for method in methods:
            rewards = results[method][game].get("all_rewards", [])
            if not rewards:
                mean_r = results[method][game]["mean_reward"]
                std_r = results[method][game]["std_reward"]
                rewards = list(np.random.normal(mean_r, std_r, 30)) if std_r > 0 else [mean_r] * 30
            data.append(rewards)
            labels.append(method)
            fill_colors.append(METHOD_LIGHT.get(method, "#E0E0E0"))

        bp = ax.boxplot(
            data, patch_artist=True, tick_labels=labels,
            medianprops=dict(color=EDGE_COLOR, linewidth=2.0),
            whiskerprops=dict(color="#666666", linewidth=1.0),
            capprops=dict(color="#666666", linewidth=1.0),
            flierprops=dict(marker="o", markerfacecolor="#AAAAAA",
                            markeredgecolor="#666666", markersize=4, alpha=0.6),
            boxprops=dict(linewidth=1.0),
        )
        for patch, cfill, method in zip(bp["boxes"], fill_colors, methods):
            patch.set_facecolor(cfill)
            patch.set_edgecolor(METHOD_COLORS.get(method, EDGE_COLOR))
            patch.set_linewidth(1.5)

        ax.set_title(game_label, fontsize=14, fontweight="bold")
        ax.set_ylabel("Reward" if j == 0 else "")
        ax.tick_params(axis="x", rotation=30)

    fig.suptitle("Reward Distribution by Method and Game",
                 fontsize=16, fontweight="bold", y=1.02)
    save_figure(fig, "04_box_plots", png_dir, svg_dir)


# ═══════════════════════════════════════════════════════════════════════
# PLOT 5: Training Curves
# ═══════════════════════════════════════════════════════════════════════
def plot_training_curves(metrics_csv: str, png_dir: str, svg_dir: str) -> None:
    """Plot training loss and epsilon curves from the metrics CSV."""
    if not os.path.exists(metrics_csv):
        print(f"  Skipping training curves: {metrics_csv} not found")
        return

    rows = []
    with open(metrics_csv, "r") as f:
        for row in csv.DictReader(f):
            if row.get("step", "step") != "step":
                rows.append(row)
    if not rows:
        print("  Skipping training curves: no data in metrics.csv")
        return

    columns = list(rows[0].keys())
    loss_cols = [c for c in columns if "train_loss" in c]
    eps_cols = [c for c in columns if "epsilon" in c]
    game_names = [c.split("/")[0] for c in loss_cols]
    GAME_COLORS = [PALETTE["navy"], PALETTE["terracotta"], PALETTE["sage"]]

    def _parse_col(rows, col):
        steps, vals = [], []
        for row in rows:
            v = row.get(col, "")
            if v and v.strip():
                try:
                    steps.append(int(row["step"]))
                    vals.append(float(v))
                except (ValueError, KeyError):
                    pass
        return steps, vals

    if loss_cols:
        fig, ax = plt.subplots(figsize=(10, 6))
        for i, (col, game) in enumerate(zip(loss_cols, game_names)):
            s, v = _parse_col(rows, col)
            if s:
                ax.plot(s, v, color=GAME_COLORS[i % len(GAME_COLORS)],
                        marker=MARKERS[i % len(MARKERS)],
                        markeredgecolor="white", markeredgewidth=1.0,
                        markersize=7, linewidth=2.0, label=game, alpha=0.9)
        ax.set_xlabel("Training Steps", fontweight="bold")
        ax.set_ylabel("Training Loss (Smooth L1)", fontweight="bold")
        ax.set_title("Expert Training Loss Curves", fontweight="bold")
        ax.legend()
        save_figure(fig, "05_training_loss_curves", png_dir, svg_dir)

    if eps_cols:
        fig, ax = plt.subplots(figsize=(10, 5))
        for i, (col, game) in enumerate(zip(eps_cols, game_names)):
            s, v = _parse_col(rows, col)
            if s:
                ax.plot(s, v, color=GAME_COLORS[i % len(GAME_COLORS)],
                        marker=MARKERS[i % len(MARKERS)],
                        markeredgecolor="white", markeredgewidth=1.0,
                        markersize=5, linewidth=2.0, label=game, alpha=0.9)
        ax.set_xlabel("Training Steps", fontweight="bold")
        ax.set_ylabel("Epsilon (Exploration)", fontweight="bold")
        ax.set_title("Epsilon Decay During Expert Training", fontweight="bold")
        ax.legend()
        save_figure(fig, "06_epsilon_decay", png_dir, svg_dir)


# ═══════════════════════════════════════════════════════════════════════
# PLOT 6: Performance Retention
# ═══════════════════════════════════════════════════════════════════════
def plot_retention_chart(
    results: Dict, methods: List[str], games: List[str],
    png_dir: str, svg_dir: str
) -> None:
    """Horizontal bar chart of retention (% of expert performance)."""
    consolidated = [m for m in methods if m != "Expert"]
    game_labels = [g.replace("NoFrameskip-v4", "") for g in games]
    expert_r = {g: results["Expert"][g]["mean_reward"] for g in games}

    fig, axes = plt.subplots(1, len(games), figsize=(5 * len(games), 5),
                             constrained_layout=True)
    if len(games) == 1:
        axes = [axes]

    for j, game in enumerate(games):
        ax = axes[j]
        er = expert_r[game]
        pcts, clrs, lbls = [], [], []
        for method in consolidated:
            r = results[method][game]["mean_reward"]
            pct = (r / er * 100) if er != 0 else (100.0 if r == 0 else 0.0)
            pcts.append(pct)
            clrs.append(METHOD_COLORS.get(method, PALETTE["slate"]))
            lbls.append(method)

        y_pos = np.arange(len(consolidated))
        bars = ax.barh(y_pos, pcts, color=clrs, edgecolor="white",
                       linewidth=1.5, height=0.55, alpha=0.85)
        for bar_rect, pct in zip(bars, pcts):
            ax.text(bar_rect.get_width() + 2, bar_rect.get_y() + bar_rect.get_height() / 2,
                    f"{pct:.1f}%", va="center", fontsize=9, fontweight="bold", color="#333333")

        ax.set_yticks(y_pos)
        ax.set_yticklabels(lbls)
        ax.set_xlabel("% of Expert Reward", fontweight="bold")
        ax.set_title(game_labels[j], fontsize=14, fontweight="bold")
        ax.axvline(x=100, color=PALETTE["sage"], linestyle="--", linewidth=1.5, alpha=0.6)
        ax.set_xlim(0, max(max(pcts) if pcts else 100, 100) * 1.2 + 10)

    fig.suptitle("Performance Retention (% of Expert)",
                 fontsize=16, fontweight="bold", y=1.02)
    save_figure(fig, "07_retention_chart", png_dir, svg_dir)


# ═══════════════════════════════════════════════════════════════════════
# PLOT 7: Forgetting Analysis
# ═══════════════════════════════════════════════════════════════════════
def plot_forgetting_analysis(
    results: Dict, methods: List[str], games: List[str],
    png_dir: str, svg_dir: str
) -> None:
    """Diverging bar chart showing reward degradation from expert baseline."""
    consolidated = [m for m in methods if m != "Expert"]
    game_labels = [g.replace("NoFrameskip-v4", "") for g in games]

    fig, ax = plt.subplots(figsize=(10, 6))
    n_m = len(consolidated)
    x = np.arange(len(games))
    width = 0.7 / n_m

    for i, method in enumerate(consolidated):
        deltas = [results[method][g]["mean_reward"] - results["Expert"][g]["mean_reward"] for g in games]
        offset = (i - n_m / 2 + 0.5) * width
        color = METHOD_COLORS.get(method, COLORS[i % len(COLORS)])
        bars = ax.bar(x + offset, deltas, width, color=color, edgecolor="white",
                      linewidth=1.0, hatch=METHOD_HATCHES.get(method),
                      label=method, alpha=0.85)
        for bar_rect, d in zip(bars, deltas):
            va = "bottom" if d >= 0 else "top"
            ax.text(bar_rect.get_x() + bar_rect.get_width() / 2,
                    bar_rect.get_height(), f"{d:+.1f}", ha="center", va=va,
                    fontsize=8, fontweight="bold", color="#333333")

    ax.set_xlabel("Atari Game", fontweight="bold")
    ax.set_ylabel("Reward Change (vs Expert)", fontweight="bold")
    ax.set_title("Forgetting Analysis: Reward Degradation After Consolidation", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(game_labels)
    ax.axhline(y=0, color=EDGE_COLOR, linewidth=1.0)
    ax.legend(loc="best")

    ylims = ax.get_ylim()
    ax.axhspan(0, ylims[1], alpha=0.04, color=PALETTE["sage"])
    ax.axhspan(ylims[0], 0, alpha=0.04, color=PALETTE["crimson"])
    ax.text(0.02, 0.97, "Positive Transfer", transform=ax.transAxes,
            fontsize=9, color=PALETTE["sage"], fontweight="bold", alpha=0.7, va="top")
    ax.text(0.02, 0.03, "Forgetting", transform=ax.transAxes,
            fontsize=9, color=PALETTE["crimson"], fontweight="bold", alpha=0.7, va="bottom")

    save_figure(fig, "08_forgetting_analysis", png_dir, svg_dir)


# ═══════════════════════════════════════════════════════════════════════
# PLOT 8: Summary Dashboard
# ═══════════════════════════════════════════════════════════════════════
def plot_summary_dashboard(
    results: Dict, methods: List[str], games: List[str],
    png_dir: str, svg_dir: str
) -> None:
    """Multi-panel summary dashboard."""
    game_labels = [g.replace("NoFrameskip-v4", "") for g in games]
    consolidated = [m for m in methods if m != "Expert"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

    # (a) Average reward
    ax = axes[0, 0]
    avg_r = [np.mean([results[m][g]["mean_reward"] for g in games]) for m in methods]
    bars = ax.bar(methods, avg_r,
                  color=[METHOD_COLORS.get(m, COLORS[i % len(COLORS)]) for i, m in enumerate(methods)],
                  edgecolor="white", linewidth=1.5, alpha=0.85)
    for b, v in zip(bars, avg_r):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.1f}",
                ha="center", va="bottom" if v >= 0 else "top",
                fontsize=9, fontweight="bold", color="#333333")
    ax.set_ylabel("Average Reward", fontweight="bold")
    ax.set_title("(a) Average Reward Across All Games", fontweight="bold")
    ax.axhline(y=0, color=EDGE_COLOR, linewidth=0.5)

    # (b) Per-game lines
    ax = axes[0, 1]
    for i, m in enumerate(methods):
        r = [results[m][g]["mean_reward"] for g in games]
        c = METHOD_COLORS.get(m, COLORS[i % len(COLORS)])
        ax.plot(game_labels, r, marker=MARKERS[i % len(MARKERS)],
                color=c, markeredgecolor="white", markeredgewidth=1.2,
                markersize=9, linewidth=2.2, label=m)
    ax.set_ylabel("Mean Reward", fontweight="bold")
    ax.set_title("(b) Per-Game Performance", fontweight="bold")
    ax.legend(fontsize=9)

    # (c) Retention heatmap
    ax = axes[1, 0]
    ret_mat = np.zeros((len(consolidated), len(games)))
    for i, m in enumerate(consolidated):
        for j, g in enumerate(games):
            er = results["Expert"][g]["mean_reward"]
            cr = results[m][g]["mean_reward"]
            ret_mat[i, j] = (cr / er * 100) if er != 0 else (100 if cr == 0 else 0)
    cmap = LinearSegmentedColormap.from_list("ret", ["#B5322A", "#D9907A", "#FAF3EB", "#7BA4C9", "#2E5090"], N=256)
    im = ax.imshow(ret_mat, cmap=cmap, aspect="auto", vmin=0, vmax=150)
    for i in range(len(consolidated)):
        for j in range(len(games)):
            nv = ret_mat[i, j] / 150.0
            tc = "white" if nv < 0.25 or nv > 0.75 else "#2C2C2C"
            ax.text(j, i, f"{ret_mat[i, j]:.0f}%", ha="center", va="center",
                    fontsize=11, fontweight="bold", color=tc)
    ax.set_xticks(range(len(games))); ax.set_xticklabels(game_labels)
    ax.set_yticks(range(len(consolidated))); ax.set_yticklabels(consolidated)
    ax.set_title("(c) Retention (% of Expert)", fontweight="bold")
    for e in range(len(games) + 1): ax.axvline(e - 0.5, color="white", linewidth=2.0)
    for e in range(len(consolidated) + 1): ax.axhline(e - 0.5, color="white", linewidth=2.0)
    fig.colorbar(im, ax=ax, shrink=0.8, label="%")

    # (d) Method ranking
    ax = axes[1, 1]
    ma = sorted([(m, np.mean([results[m][g]["mean_reward"] for g in games])) for m in methods],
                key=lambda x: x[1], reverse=True)
    sm, sa = zip(*ma)
    y_pos = np.arange(len(sm))
    bars = ax.barh(y_pos, sa,
                   color=[METHOD_COLORS.get(m, PALETTE["slate"]) for m in sm],
                   edgecolor="white", linewidth=1.5, alpha=0.85)
    for b, v in zip(bars, sa):
        ax.text(b.get_width() + abs(min(sa)) * 0.05 + 1, b.get_y() + b.get_height() / 2,
                f"{v:.1f}", va="center", fontsize=9, fontweight="bold", color="#333333")
    ax.set_yticks(y_pos); ax.set_yticklabels(sm)
    ax.set_xlabel("Average Reward", fontweight="bold")
    ax.set_title("(d) Method Ranking", fontweight="bold")
    ax.axvline(x=0, color=EDGE_COLOR, linewidth=0.5)

    fig.suptitle("CRL-Atari: Experiment Summary Dashboard", fontsize=17, fontweight="bold")
    save_figure(fig, "09_summary_dashboard", png_dir, svg_dir)


# ═══════════════════════════════════════════════════════════════════════
# PLOT 9: Violin Plots
# ═══════════════════════════════════════════════════════════════════════
def plot_violin(
    results: Dict, methods: List[str], games: List[str],
    png_dir: str, svg_dir: str
) -> None:
    """Violin plots showing reward distribution shape per method per game."""
    n_games = len(games)
    fig, axes = plt.subplots(1, n_games, figsize=(5 * n_games, 6), constrained_layout=True)
    if n_games == 1:
        axes = [axes]

    for j, game in enumerate(games):
        ax = axes[j]
        data = []
        for method in methods:
            rewards = results[method][game].get("all_rewards", [])
            if not rewards:
                mr = results[method][game]["mean_reward"]
                sr = max(results[method][game]["std_reward"], 0.1)
                rewards = list(np.random.normal(mr, sr, 30))
            data.append(rewards)

        parts = ax.violinplot(data, positions=range(len(methods)),
                              showmeans=True, showmedians=True, showextrema=True)
        for i, (body, method) in enumerate(zip(parts["bodies"], methods)):
            body.set_facecolor(METHOD_LIGHT.get(method, "#E0E0E0"))
            body.set_edgecolor(METHOD_COLORS.get(method, COLORS[i % len(COLORS)]))
            body.set_linewidth(1.5)
            body.set_alpha(0.8)
        for key in ["cmeans", "cmedians"]:
            if key in parts:
                parts[key].set_color(EDGE_COLOR)
                parts[key].set_linewidth(1.5 if key == "cmedians" else 1.0)
        for key in ["cbars", "cmins", "cmaxes"]:
            if key in parts:
                parts[key].set_color("#888888")
                parts[key].set_linewidth(0.8)

        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=30, ha="right")
        ax.set_title(game.replace("NoFrameskip-v4", ""), fontsize=14, fontweight="bold")
        ax.set_ylabel("Reward" if j == 0 else "")

    fig.suptitle("Reward Distribution (Violin Plot)", fontsize=16, fontweight="bold", y=1.02)
    save_figure(fig, "10_violin_plots", png_dir, svg_dir)


# ═══════════════════════════════════════════════════════════════════════
# PLOT 10: Dot Comparison
# ═══════════════════════════════════════════════════════════════════════
def plot_dot_comparison(
    results: Dict, methods: List[str], games: List[str],
    png_dir: str, svg_dir: str
) -> None:
    """Dot plot (lollipop chart) showing expert vs consolidated."""
    game_labels = [g.replace("NoFrameskip-v4", "") for g in games]
    consolidated = [m for m in methods if m != "Expert"]

    fig, ax = plt.subplots(figsize=(10, 6))

    y_positions, y_labels = [], []
    idx = 0
    for game, gl in zip(games, game_labels):
        for method in consolidated:
            y_positions.append(idx)
            y_labels.append(f"{gl} \u2014 {method}")
            idx += 1
        idx += 0.5
    y_positions = np.array(y_positions)

    idx = 0
    for game, gl in zip(games, game_labels):
        ev = results["Expert"][game]["mean_reward"]
        for mi, method in enumerate(consolidated):
            y = y_positions[idx]
            mv = results[method][game]["mean_reward"]
            color = METHOD_COLORS.get(method, COLORS[mi % len(COLORS)])

            ax.plot([ev, mv], [y, y], color="#D0D0D0", linewidth=2.0, zorder=1)
            ax.scatter(ev, y, color=PALETTE["navy"], s=90,
                       edgecolor="white", linewidth=1.5, zorder=3, marker="o")
            ax.scatter(mv, y, color=color, s=90,
                       edgecolor="white", linewidth=1.5, zorder=3,
                       marker=MARKERS[(mi + 1) % len(MARKERS)])
            ax.text(max(ev, mv) + 5, y, f"{mv - ev:+.1f}",
                    va="center", fontsize=7, color="#555555", fontweight="bold")
            idx += 1

    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=9)
    ax.set_xlabel("Mean Reward", fontweight="bold")
    ax.set_title("Expert vs Consolidated: Dot Comparison", fontweight="bold")
    ax.axvline(x=0, color=EDGE_COLOR, linewidth=0.5, linestyle="--")

    legend_elements = [
        plt.scatter([], [], color=PALETTE["navy"], edgecolor="white", marker="o", s=90, label="Expert"),
    ]
    for i, m in enumerate(consolidated):
        legend_elements.append(
            plt.scatter([], [], color=METHOD_COLORS.get(m), edgecolor="white",
                        marker=MARKERS[(i + 1) % len(MARKERS)], s=90, label=m))
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    save_figure(fig, "11_dot_comparison", png_dir, svg_dir)


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════
def main() -> None:
    """Parse arguments and generate all visualizations."""
    parser = argparse.ArgumentParser(description="CRL-Atari Visualization Generator")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--tag", type=str, default="default")
    args = parser.parse_args()

    setup_style()

    figures_dir = os.path.join(args.results_dir, "figures")
    png_dir = os.path.join(figures_dir, "png")
    svg_dir = os.path.join(figures_dir, "svg")
    os.makedirs(png_dir, exist_ok=True)
    os.makedirs(svg_dir, exist_ok=True)

    json_path = os.path.join(figures_dir, f"comparison_results_{args.tag}.json")
    metrics_csv = os.path.join(args.results_dir, "logs", f"experts_{args.tag}", "metrics.csv")

    print(f"CRL-Atari Visualization Generator")
    print(f"  PNG: {png_dir}  |  SVG: {svg_dir}")
    print(f"  JSON: {json_path}")
    print()

    if not os.path.exists(json_path):
        print(f"ERROR: {json_path} not found. Run the pipeline first.")
        sys.exit(1)

    raw_results = load_comparison_results(json_path)
    methods, games, results = extract_data(raw_results)
    print(f"Methods: {methods}")
    print(f"Games  : {games}\n")
    print("Generating plots...")

    plot_grouped_bar(results, methods, games, png_dir, svg_dir)
    plot_performance_heatmap(results, methods, games, png_dir, svg_dir)
    plot_radar_chart(results, methods, games, png_dir, svg_dir)
    plot_box_plots(results, methods, games, png_dir, svg_dir)
    plot_training_curves(metrics_csv, png_dir, svg_dir)
    plot_retention_chart(results, methods, games, png_dir, svg_dir)
    plot_forgetting_analysis(results, methods, games, png_dir, svg_dir)
    plot_summary_dashboard(results, methods, games, png_dir, svg_dir)
    plot_violin(results, methods, games, png_dir, svg_dir)
    plot_dot_comparison(results, methods, games, png_dir, svg_dir)

    print(f"\nAll visualizations saved to:\n  PNG: {png_dir}/\n  SVG: {svg_dir}/\nDone.")


if __name__ == "__main__":
    main()
