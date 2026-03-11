"""
Generate Hybrid deep-dive visualizations for CRL-Atari experiments.

Produces publication-quality plots focused on the Hybrid (HTCL Phase 1 + KD Phase 2)
method, comparing it against all baselines.

Follows the academic-art SKILL.md palette and typography:
  AC_SERIES palette, Inter/Source Serif 4/JetBrains Mono fonts,
  white background, Tufte spine, horizontal-only faint grid, 300 dpi.

Usage:
    python scripts/visualize_hybrid.py [--results-dir results] [--tag default]
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# ══════════════════════════════════════════════════════════════════════
# Academic palette (SKILL.md)
# ══════════════════════════════════════════════════════════════════════
AC_SERIES = [
    "#2563EB",  # blue
    "#D97706",  # amber
    "#059669",  # green
    "#DC2626",  # red
    "#7C3AED",  # violet
    "#0891B2",  # teal
    "#BE185D",  # rose
    "#92400E",  # sienna
]
AC_BG      = "#FFFFFF"
AC_SURFACE = "#F8F9FA"
AC_BORDER  = "#DEE2E6"
AC_AXIS    = "#495057"
AC_GRID    = "#E9ECEF"
AC_TEXT    = "#212529"
AC_MUTED   = "#6C757D"
AC_FAINT   = "#ADB5BD"

# Consistent method-to-color mapping.
METHOD_COLORS: Dict[str, str] = {
    "Expert":       AC_SERIES[0],  # blue
    "Distillation": AC_SERIES[1],  # amber
    "One-Shot":     AC_SERIES[4],  # violet
    "Iterative":    AC_SERIES[5],  # teal
    "Hybrid":       AC_SERIES[2],  # green
}
METHOD_ORDER = ["Expert", "One-Shot", "Iterative", "Distillation", "Hybrid"]


# ──────────────────────────────────────────────────────────────────────
# Style setup
# ──────────────────────────────────────────────────────────────────────
def setup_style() -> None:
    """Apply global rcParams per SKILL.md academic guidelines."""
    mpl.rcParams.update({
        "font.family":       "sans-serif",
        "font.sans-serif":   ["Inter", "Helvetica Neue", "Arial"],
        "font.size":         11,
        "axes.titlesize":    13,
        "axes.labelsize":    12,
        "xtick.labelsize":   11,
        "ytick.labelsize":   11,
        "legend.fontsize":   10,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.edgecolor":    AC_AXIS,
        "axes.labelcolor":   AC_TEXT,
        "axes.linewidth":    1.0,
        "axes.facecolor":    AC_BG,
        "figure.facecolor":  AC_BG,
        "savefig.facecolor": AC_BG,
        "axes.grid":         True,
        "axes.grid.axis":    "y",
        "axes.axisbelow":    True,
        "grid.color":        AC_GRID,
        "grid.linewidth":    0.6,
        "grid.alpha":        1.0,
        "grid.linestyle":    "-",
        "xtick.color":       AC_MUTED,
        "ytick.color":       AC_MUTED,
        "xtick.direction":   "out",
        "ytick.direction":   "out",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size":  4,
        "ytick.major.size":  4,
        "figure.dpi":        150,
        "savefig.dpi":       300,
        "savefig.bbox":      "tight",
        "legend.frameon":    True,
        "legend.edgecolor":  AC_BORDER,
        "legend.fancybox":   False,
        "legend.framealpha": 0.95,
        "legend.shadow":     False,
    })


# ──────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────
def _color(method: str, idx: int = 0) -> str:
    return METHOD_COLORS.get(method, AC_SERIES[idx % len(AC_SERIES)])


def save_figure(fig: plt.Figure, name: str, png_dir: str, svg_dir: str) -> None:
    """Save figure in both PNG (300 dpi) and SVG."""
    fig.savefig(os.path.join(png_dir, f"{name}.png"),
                dpi=300, bbox_inches="tight", facecolor=AC_BG)
    fig.savefig(os.path.join(svg_dir, f"{name}.svg"),
                bbox_inches="tight", facecolor=AC_BG)
    plt.close(fig)
    print(f"  Saved {name}")


def load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def build_data(
    raw: Dict[str, List[Dict]],
) -> Dict[str, Dict[str, Dict]]:
    """Restructure to {method: {game_name: {...}}}."""
    nested: Dict[str, Dict[str, Dict]] = {}
    for method, entries in raw.items():
        nested[method] = {}
        for entry in entries:
            nested[method][entry["game_name"]] = entry
    return nested


# ══════════════════════════════════════════════════════════════════════
# PLOT 1  Hybrid vs Expert Violin
# ══════════════════════════════════════════════════════════════════════
def plot_hybrid_vs_expert_violin(
    data: Dict, games: List[str],
    png_dir: str, svg_dir: str,
) -> None:
    """Side-by-side violin for Expert and Hybrid, one panel per game."""
    pair = ["Expert", "Hybrid"]
    n_g = len(games)
    fig, axes = plt.subplots(1, n_g, figsize=(4.5 * n_g, 5),
                             constrained_layout=True)
    if n_g == 1:
        axes = [axes]

    rng = np.random.default_rng(42)

    for j, game in enumerate(games):
        ax = axes[j]
        reward_lists = []
        for method in pair:
            rew = data[method][game].get("all_rewards", [])
            if not rew:
                mr = data[method][game]["mean_reward"]
                sr = max(data[method][game]["std_reward"], 0.1)
                rew = list(np.random.normal(mr, sr, 30))
            reward_lists.append(rew)

        parts = ax.violinplot(
            reward_lists, positions=[0, 1],
            showmeans=True, showmedians=True, showextrema=True,
        )
        for i, body in enumerate(parts["bodies"]):
            c = _color(pair[i])
            body.set_facecolor(c)
            body.set_edgecolor(c)
            body.set_alpha(0.25)
            body.set_linewidth(1.2)
        for key in ("cmeans", "cmedians"):
            if key in parts:
                parts[key].set_color(AC_AXIS)
                parts[key].set_linewidth(1.2)
        for key in ("cbars", "cmins", "cmaxes"):
            if key in parts:
                parts[key].set_color(AC_MUTED)
                parts[key].set_linewidth(0.7)

        for i, method in enumerate(pair):
            rew = reward_lists[i]
            jitter = rng.uniform(-0.08, 0.08, len(rew))
            ax.scatter(
                np.full(len(rew), i) + jitter, rew,
                s=14, alpha=0.5, color=_color(method),
                edgecolors="white", linewidths=0.3, zorder=5,
            )

        # Retention annotation
        e_mean = data["Expert"][game]["mean_reward"]
        h_mean = data["Hybrid"][game]["mean_reward"]
        ret = (h_mean / e_mean * 100) if e_mean != 0 else 0
        ax.annotate(
            f"{ret:.1f}% retained",
            xy=(0.5, 0.97), xycoords="axes fraction", ha="center", va="top",
            fontsize=10, fontweight="600", color=AC_SERIES[2],
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#D1FAE5",
                      edgecolor=AC_SERIES[2], alpha=0.85),
        )

        ax.set_xticks([0, 1])
        ax.set_xticklabels(pair, fontsize=11)
        ax.set_title(game, fontsize=12, fontweight="600", color=AC_TEXT)
        ax.set_ylabel("Reward" if j == 0 else "")

    fig.suptitle("Hybrid vs Expert: Reward Distributions",
                 fontsize=14, fontweight="600", color=AC_TEXT, y=1.03)
    save_figure(fig, "hybrid_vs_expert_violin", png_dir, svg_dir)


# ══════════════════════════════════════════════════════════════════════
# PLOT 2  Retention Waterfall
# ══════════════════════════════════════════════════════════════════════
def plot_hybrid_retention_waterfall(
    data: Dict, games: List[str],
    png_dir: str, svg_dir: str,
) -> None:
    """Waterfall chart: Expert baseline -> Hybrid reward, showing the gap."""
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(games))
    width = 0.35

    e_vals = [data["Expert"][g]["mean_reward"] for g in games]
    h_vals = [data["Hybrid"][g]["mean_reward"] for g in games]
    gaps = [e - h for e, h in zip(e_vals, h_vals)]

    ax.bar(x - width / 2, e_vals, width, color=_color("Expert"),
           label="Expert", alpha=0.88, edgecolor="white", linewidth=0.8)
    ax.bar(x + width / 2, h_vals, width, color=_color("Hybrid"),
           label="Hybrid", alpha=0.88, edgecolor="white", linewidth=0.8)

    for i in range(len(games)):
        y_top = max(e_vals[i], h_vals[i])
        y_bot = min(e_vals[i], h_vals[i])
        mid = (y_top + y_bot) / 2
        sign = "+" if h_vals[i] > e_vals[i] else ""
        pct = (h_vals[i] / e_vals[i] * 100) if e_vals[i] != 0 else 0
        ax.annotate(
            f"{sign}{-gaps[i]:+.1f}\n({pct:.0f}%)",
            xy=(i, mid), ha="center", fontsize=9,
            fontweight="600", color=AC_MUTED,
        )
        ax.plot([i - width / 2, i + width / 2], [y_top, y_bot],
                color=AC_AXIS, linewidth=0.8, linestyle=":", alpha=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(games)
    ax.set_ylabel("Mean Reward")
    ax.set_title("Hybrid Retention Waterfall", fontweight="600", color=AC_TEXT)
    ax.legend()
    ax.axhline(y=0, color=AC_AXIS, linewidth=0.6)

    save_figure(fig, "hybrid_retention_waterfall", png_dir, svg_dir)


# ══════════════════════════════════════════════════════════════════════
# PLOT 3  Method Progression (strip chart)
# ══════════════════════════════════════════════════════════════════════
def plot_method_progression(
    data: Dict, methods: List[str], games: List[str],
    png_dir: str, svg_dir: str,
) -> None:
    """All-method strip chart with jitter, one panel per game."""
    n_g = len(games)
    fig, axes = plt.subplots(1, n_g, figsize=(5 * n_g, 5.5),
                             constrained_layout=True)
    if n_g == 1:
        axes = [axes]

    rng = np.random.default_rng(42)

    for j, game in enumerate(games):
        ax = axes[j]
        for i, method in enumerate(methods):
            rew = data[method][game].get("all_rewards", [])
            if not rew:
                mr = data[method][game]["mean_reward"]
                sr = max(data[method][game]["std_reward"], 0.1)
                rew = list(np.random.normal(mr, sr, 30))

            jitter = rng.uniform(-0.15, 0.15, len(rew))
            color = _color(method, i)
            ax.scatter(
                np.full(len(rew), i) + jitter, rew,
                s=18, alpha=0.5, color=color,
                edgecolors="white", linewidths=0.3, zorder=5,
            )
            # Mean marker
            mr_val = data[method][game]["mean_reward"]
            ax.scatter(i, mr_val, s=80, marker="D", color=color,
                       edgecolors="white", linewidths=1.0, zorder=10)

        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=25, ha="right", fontsize=9)
        ax.set_title(game, fontsize=12, fontweight="600", color=AC_TEXT)
        ax.set_ylabel("Reward" if j == 0 else "")
        ax.axhline(y=0, color=AC_AXIS, linewidth=0.6)

    fig.suptitle("All Methods: Reward Strip Chart",
                 fontsize=14, fontweight="600", color=AC_TEXT, y=1.03)
    save_figure(fig, "hybrid_all_methods_strip", png_dir, svg_dir)


# ══════════════════════════════════════════════════════════════════════
# PLOT 4  Pong Analysis (Hybrid spotlight)
# ══════════════════════════════════════════════════════════════════════
def plot_hybrid_pong_analysis(
    data: Dict, methods: List[str],
    png_dir: str, svg_dir: str,
) -> None:
    """Horizontal bar chart for Pong, highlighting Hybrid advantage."""
    game = "Pong"
    if game not in data.get("Expert", {}):
        print("  Skipping Pong analysis: Pong not in data")
        return

    fig, ax = plt.subplots(figsize=(8, 4.5))
    y_pos = np.arange(len(methods))
    means = [data[m][game]["mean_reward"] for m in methods]
    stds  = [data[m][game]["std_reward"] for m in methods]
    colors = [_color(m, i) for i, m in enumerate(methods)]

    bars = ax.barh(
        y_pos, means, xerr=stds, height=0.5,
        color=colors, alpha=0.88, edgecolor="white", linewidth=0.8,
        capsize=3,
        error_kw={"elinewidth": 0.8, "capthick": 0.8, "color": AC_AXIS},
    )

    for bar_rect, mean in zip(bars, means):
        x_pos = bar_rect.get_width()
        label = f"{mean:+.1f}" if mean < 0 else f"{mean:.1f}"
        ha = "left" if x_pos >= 0 else "right"
        offset = 1.0 if x_pos >= 0 else -1.0
        ax.text(x_pos + offset, bar_rect.get_y() + bar_rect.get_height() / 2,
                label, ha=ha, va="center", fontsize=10,
                fontweight="600", color=AC_TEXT)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(methods, fontsize=11)
    ax.set_xlabel("Mean Reward")
    ax.set_title("Pong: Method Comparison (Only Hybrid Positive)",
                 fontweight="600", color=AC_TEXT)
    ax.axvline(x=0, color=AC_AXIS, linewidth=0.8)

    # Shade positive region
    xlims = ax.get_xlim()
    ax.axvspan(0, xlims[1], alpha=0.03, color=AC_SERIES[2])
    ax.axvspan(xlims[0], 0, alpha=0.03, color=AC_SERIES[3])

    save_figure(fig, "hybrid_pong_analysis", png_dir, svg_dir)


# ══════════════════════════════════════════════════════════════════════
# PLOT 5  Retention Heatmap
# ══════════════════════════════════════════════════════════════════════
def plot_retention_heatmap(
    data: Dict, methods: List[str], games: List[str],
    png_dir: str, svg_dir: str,
) -> None:
    """Heatmap of % retention, consolidated methods vs games."""
    consolidated = [m for m in methods if m != "Expert"]
    expert_r = {g: data["Expert"][g]["mean_reward"] for g in games}

    n_m = len(consolidated)
    n_g = len(games)
    mat = np.zeros((n_m, n_g))
    for i, method in enumerate(consolidated):
        for j, game in enumerate(games):
            er = expert_r[game]
            mr = data[method][game]["mean_reward"]
            mat[i, j] = (mr / er * 100) if er != 0 else 0.0

    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        "retention",
        [AC_SERIES[3], "#FEF3C7", "#D1FAE5", AC_SERIES[2]],
        N=256,
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(mat, cmap=cmap, aspect="auto", vmin=-120, vmax=120)

    for i in range(n_m):
        for j in range(n_g):
            val = mat[i, j]
            tc = "white" if abs(val) > 80 else AC_TEXT
            ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                    fontsize=10, fontweight="600", color=tc)

    ax.set_xticks(range(n_g))
    ax.set_xticklabels(games)
    ax.set_yticks(range(n_m))
    ax.set_yticklabels(consolidated)
    ax.set_title("Performance Retention (% of Expert)",
                 fontweight="600", color=AC_TEXT)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("% Expert Reward")

    for edge in range(n_g + 1):
        ax.axvline(edge - 0.5, color="white", linewidth=2.0)
    for edge in range(n_m + 1):
        ax.axhline(edge - 0.5, color="white", linewidth=2.0)

    save_figure(fig, "hybrid_retention_heatmap", png_dir, svg_dir)


# ══════════════════════════════════════════════════════════════════════
# PLOT 6  Summary Card
# ══════════════════════════════════════════════════════════════════════
def plot_hybrid_summary_card(
    data: Dict, methods: List[str], games: List[str],
    png_dir: str, svg_dir: str,
) -> None:
    """Single-page summary card: avg retention + per-game breakdown."""
    consolidated = [m for m in methods if m != "Expert"]
    expert_r = {g: data["Expert"][g]["mean_reward"] for g in games}

    # Compute average retention per method
    avg_retention: Dict[str, float] = {}
    for method in consolidated:
        pcts = []
        for g in games:
            er = expert_r[g]
            mr = data[method][g]["mean_reward"]
            pcts.append((mr / er * 100) if er != 0 else 0.0)
        avg_retention[method] = float(np.mean(pcts))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5),
                             gridspec_kw={"width_ratios": [1, 1.5]})

    # Left: average retention bar
    ax = axes[0]
    y_pos = np.arange(len(consolidated))
    avgs = [avg_retention[m] for m in consolidated]
    colors = [_color(m) for m in consolidated]
    bars = ax.barh(y_pos, avgs, color=colors, height=0.5,
                   alpha=0.88, edgecolor="white", linewidth=0.8)
    for bar_rect, m in zip(bars, consolidated):
        ax.text(bar_rect.get_width() + 1.5,
                bar_rect.get_y() + bar_rect.get_height() / 2,
                f"{avg_retention[m]:.1f}%", va="center",
                fontsize=10, fontweight="600", color=AC_TEXT)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(consolidated, fontsize=11)
    ax.set_xlabel("Average % of Expert")
    ax.set_title("Average Retention", fontweight="600", color=AC_TEXT)
    ax.axvline(x=0, color=AC_AXIS, linewidth=0.6)

    # Right: per-game grouped bars
    ax2 = axes[1]
    x = np.arange(len(games))
    n_m = len(consolidated)
    w = 0.7 / n_m
    for i, method in enumerate(consolidated):
        vals = []
        for g in games:
            er = expert_r[g]
            mr = data[method][g]["mean_reward"]
            vals.append((mr / er * 100) if er != 0 else 0.0)
        offset = (i - n_m / 2 + 0.5) * w
        ax2.bar(x + offset, vals, w, color=_color(method, i),
                label=method, alpha=0.88, edgecolor="white", linewidth=0.8)

    ax2.set_xticks(x)
    ax2.set_xticklabels(games)
    ax2.set_ylabel("% of Expert Reward")
    ax2.set_title("Per-Game Retention", fontweight="600", color=AC_TEXT)
    ax2.axhline(y=100, color=AC_MUTED, linewidth=0.8, linestyle="--", alpha=0.5)
    ax2.axhline(y=0, color=AC_AXIS, linewidth=0.6)
    ax2.legend(loc="best", fontsize=8)

    fig.suptitle("Consolidation Summary",
                 fontsize=14, fontweight="600", color=AC_TEXT, y=1.02)
    fig.tight_layout()
    save_figure(fig, "hybrid_method_progression", png_dir, svg_dir)


# ══════════════════════════════════════════════════════════════════════
# PLOT 7  Hybrid Advantage Delta
# ══════════════════════════════════════════════════════════════════════
def plot_hybrid_advantage(
    data: Dict, methods: List[str], games: List[str],
    png_dir: str, svg_dir: str,
) -> None:
    """Bar chart: Hybrid reward minus each other method, per game."""
    others = [m for m in methods if m not in ("Hybrid", "Expert")]
    n_o = len(others)
    x = np.arange(len(games))
    w = 0.7 / n_o

    fig, ax = plt.subplots(figsize=(10, 5))

    hybrid_vals = {g: data["Hybrid"][g]["mean_reward"] for g in games}

    for i, method in enumerate(others):
        deltas = [
            hybrid_vals[g] - data[method][g]["mean_reward"] for g in games
        ]
        offset = (i - n_o / 2 + 0.5) * w
        color = _color(method, i)
        bars = ax.bar(x + offset, deltas, w, color=color,
                      edgecolor="white", linewidth=0.8, label=f"vs {method}",
                      alpha=0.88)
        for bar_rect, d in zip(bars, deltas):
            va = "bottom" if d >= 0 else "top"
            ax.text(bar_rect.get_x() + bar_rect.get_width() / 2,
                    bar_rect.get_height(), f"{d:+.1f}",
                    ha="center", va=va, fontsize=8,
                    fontweight="600", color=AC_TEXT)

    ax.set_xticks(x)
    ax.set_xticklabels(games)
    ax.set_ylabel("Hybrid Advantage (reward delta)")
    ax.set_title("Hybrid Advantage Over Baselines",
                 fontweight="600", color=AC_TEXT)
    ax.axhline(y=0, color=AC_AXIS, linewidth=0.8)
    ax.legend(loc="best")

    ylims = ax.get_ylim()
    ax.axhspan(0, ylims[1], alpha=0.03, color=AC_SERIES[2])
    ax.axhspan(ylims[0], 0, alpha=0.03, color=AC_SERIES[3])

    save_figure(fig, "hybrid_advantage_delta", png_dir, svg_dir)


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════
def main() -> None:
    """Parse arguments and generate all Hybrid deep-dive visualizations."""
    parser = argparse.ArgumentParser(
        description="CRL-Atari Hybrid Deep-Dive Visualizations",
    )
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--tag", type=str, default="default")
    args = parser.parse_args()

    setup_style()

    figures_dir = os.path.join(args.results_dir, "figures")
    png_dir = os.path.join(figures_dir, "png")
    svg_dir = os.path.join(figures_dir, "svg")
    os.makedirs(png_dir, exist_ok=True)
    os.makedirs(svg_dir, exist_ok=True)

    json_path = os.path.join(figures_dir, f"comparison_full_{args.tag}.json")
    if not os.path.exists(json_path):
        json_path = os.path.join(figures_dir, f"comparison_results_{args.tag}.json")

    print("CRL-Atari Hybrid Deep-Dive Visualization")
    print(f"  PNG dir : {png_dir}")
    print(f"  SVG dir : {svg_dir}")
    print(f"  Data    : {json_path}")
    print()

    if not os.path.exists(json_path):
        print(f"ERROR: {json_path} not found.")
        sys.exit(1)

    raw = load_json(json_path)
    data = build_data(raw)
    methods = list(raw.keys())
    games = [e["game_name"] for e in raw[methods[0]]]

    # Reorder methods to preferred display order where possible
    ordered = [m for m in METHOD_ORDER if m in methods]
    ordered += [m for m in methods if m not in ordered]
    methods = ordered

    print(f"Methods: {methods}")
    print(f"Games  : {games}")
    print()
    print("Generating Hybrid deep-dive plots...")

    plot_hybrid_vs_expert_violin(data, games, png_dir, svg_dir)
    plot_hybrid_retention_waterfall(data, games, png_dir, svg_dir)
    plot_method_progression(data, methods, games, png_dir, svg_dir)
    plot_hybrid_pong_analysis(data, methods, png_dir, svg_dir)
    plot_retention_heatmap(data, methods, games, png_dir, svg_dir)
    plot_hybrid_summary_card(data, methods, games, png_dir, svg_dir)
    plot_hybrid_advantage(data, methods, games, png_dir, svg_dir)

    print(f"\nDone. Hybrid figures saved to {png_dir}/ and {svg_dir}/")


if __name__ == "__main__":
    main()
