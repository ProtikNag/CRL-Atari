#!/usr/bin/env python3
"""Regenerate classic control visualizations from existing evaluation JSONs."""

import json
import os
import sys
import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────
RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results_classic_control", "figures",
)

GAME_NAMES = ["CartPole", "Acrobot", "LunarLander"]

# Map JSON filenames to display names
METHOD_MAP = {
    "eval_expert": "Expert",
    "eval_multitask": "Multi-Task",
    "eval_whc": "WHC",
    "eval_distillation": "Distillation",
    "eval_hybrid": "Hybrid",
    "eval_ewc": "EWC",
    "eval_pc": "Progress & Compress",
    "eval_trac": "TRAC",
    "eval_cchain": "C-CHAIN",
}


def load_all_evals(results_dir: str, tag: str = "default"):
    """Load all eval JSONs into {method_name: [per-game dicts]}."""
    all_eval_data = {}
    expert_rewards = {}

    for fname in sorted(os.listdir(results_dir)):
        if not fname.endswith(f"_{tag}.json"):
            continue

        fpath = os.path.join(results_dir, fname)
        with open(fpath) as f:
            data = json.load(f)

        # Identify method from filename
        prefix = fname.replace(f"_{tag}.json", "")

        # Handle expert files (one per game)
        if prefix.startswith("eval_expert_"):
            game = prefix.replace("eval_expert_", "")
            if "Expert" not in all_eval_data:
                all_eval_data["Expert"] = []
            all_eval_data["Expert"].extend(data)
            for entry in data:
                expert_rewards[entry["game_name"]] = entry["mean_reward"]
            continue

        # Handle method files
        method_name = METHOD_MAP.get(prefix)
        if method_name and method_name != "Expert":
            all_eval_data[method_name] = data

    return all_eval_data, expert_rewards


def generate_heatmap(all_eval_data, expert_rewards, figure_dir, game_names):
    """Generate retention heatmap (% of expert reward)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from matplotlib.colors import TwoSlopeNorm

    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Inter', 'Helvetica', 'Arial'],
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': True, 'grid.color': '#E9ECEF', 'grid.linewidth': 0.6,
        'axes.edgecolor': '#495057', 'axes.labelcolor': '#212529',
        'xtick.color': '#6C757D', 'ytick.color': '#6C757D',
        'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    })

    method_order = ["Multi-Task", "WHC", "Distillation", "Hybrid",
                    "EWC", "Progress & Compress", "TRAC", "C-CHAIN"]
    methods = [m for m in method_order if m in all_eval_data]
    n_m = len(methods)
    n_g = len(game_names)

    if n_m == 0:
        print("  No methods to plot in heatmap.")
        return

    ret_matrix = np.zeros((n_m, n_g))
    raw_matrix = np.zeros((n_m, n_g))

    for i, method in enumerate(methods):
        evals = all_eval_data[method]
        eval_dict = {e["game_name"]: e for e in evals}
        for j, game in enumerate(game_names):
            if game in eval_dict:
                raw = eval_dict[game]["mean_reward"]
                exp = expert_rewards.get(game, 1.0)
                raw_matrix[i, j] = raw

                # Acrobot: higher (less negative) is better, normalize accordingly
                if game == "Acrobot":
                    # Scale: -500 (worst) -> 0%, expert -> 100%
                    worst = -500.0
                    if exp != worst:
                        ret_matrix[i, j] = (raw - worst) / (exp - worst) * 100
                    else:
                        ret_matrix[i, j] = 0.0
                elif exp != 0:
                    ret_matrix[i, j] = raw / exp * 100
                else:
                    ret_matrix[i, j] = 100.0 if raw == 0 else 0.0

    fig, ax = plt.subplots(figsize=(8, 0.6 * n_m + 2.5))

    vmin = min(0, np.nanmin(ret_matrix))
    vmax = max(100, np.nanmax(ret_matrix) * 1.05)
    vcenter = 50
    if vmin >= vcenter:
        vcenter = (vmin + vmax) / 2
    norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    cmap = mpl.colormaps.get_cmap("RdYlGn")

    im = ax.imshow(ret_matrix, cmap=cmap, norm=norm, aspect="auto")

    for i in range(n_m):
        for j in range(n_g):
            pct = ret_matrix[i, j]
            raw = raw_matrix[i, j]
            text_color = "#FFFFFF" if pct < 20 or pct > 80 else "#212529"
            ax.text(j, i, f"{raw:.1f}\n({pct:.0f}%)",
                    ha="center", va="center", fontsize=9, fontweight="600",
                    color=text_color)

    ax.set_xticks(range(n_g))
    ax.set_xticklabels(game_names, fontsize=11)
    ax.set_yticks(range(n_m))
    ax.set_yticklabels(methods, fontsize=10)

    for e in range(n_g + 1):
        ax.axvline(e - 0.5, color="white", linewidth=2.5)
    for e in range(n_m + 1):
        ax.axhline(e - 0.5, color="white", linewidth=2.5)

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("% of Expert Reward", fontsize=11)

    ax.set_title("Classic Control: Consolidation Performance (% of Expert)",
                 fontsize=14, fontweight="600", color="#212529", pad=14,
                 fontfamily="serif")

    for spine in ax.spines.values():
        spine.set_visible(False)

    for fmt in ("png", "svg"):
        out_dir = os.path.join(figure_dir, fmt)
        os.makedirs(out_dir, exist_ok=True)
        fig.savefig(os.path.join(out_dir, f"02_retention_heatmap.{fmt}"),
                    dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Heatmap saved to {figure_dir}/{{png,svg}}/")


def generate_reward_distributions(all_eval_data, figure_dir, game_names):
    """Generate reward distribution box plots per game."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    AC_SERIES = ['#2563EB', '#D97706', '#059669', '#DC2626',
                 '#7C3AED', '#0891B2', '#BE185D', '#92400E']

    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Inter', 'Helvetica', 'Arial'],
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': True, 'grid.color': '#E9ECEF', 'grid.linewidth': 0.6,
        'axes.edgecolor': '#495057', 'axes.labelcolor': '#212529',
        'xtick.color': '#6C757D', 'ytick.color': '#6C757D',
        'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    })

    method_colors = {
        "Expert": AC_SERIES[0], "Multi-Task": AC_SERIES[1],
        "WHC": AC_SERIES[7], "Distillation": AC_SERIES[3],
        "Hybrid": AC_SERIES[2], "EWC": AC_SERIES[6],
        "Progress & Compress": AC_SERIES[5], "TRAC": AC_SERIES[4],
        "C-CHAIN": "#92400E",
    }

    method_order = ["Expert", "Multi-Task", "WHC", "Distillation", "Hybrid",
                    "EWC", "Progress & Compress", "TRAC", "C-CHAIN"]
    methods = [m for m in method_order if m in all_eval_data]

    n = len(game_names)
    fig, axes = plt.subplots(1, n, figsize=(max(4.5 * n, 12), 5.0), constrained_layout=True)
    if n == 1:
        axes = [axes]

    rng = np.random.default_rng(42)

    for j, game in enumerate(game_names):
        ax = axes[j]
        positions, box_data, colors_list, labels = [], [], [], []

        for i, method in enumerate(methods):
            evals = all_eval_data[method]
            eval_dict = {e["game_name"]: e for e in evals}
            if game not in eval_dict:
                continue
            entry = eval_dict[game]
            rewards = entry.get("all_rewards", [entry["mean_reward"]] * 30)
            positions.append(i)
            box_data.append(rewards)
            colors_list.append(method_colors.get(method, AC_SERIES[i % 8]))
            labels.append(method)

        if not box_data:
            continue

        bp = ax.boxplot(box_data, positions=positions, widths=0.45, patch_artist=True,
                        showfliers=False, medianprops=dict(color="#212529", linewidth=1.5),
                        whiskerprops=dict(color="#6C757D"), capprops=dict(color="#6C757D"),
                        boxprops=dict(linewidth=1.0), zorder=1)
        for patch, c in zip(bp["boxes"], colors_list):
            patch.set_facecolor(mpl.colors.to_rgba(c, alpha=0.25))
            patch.set_edgecolor(c)

        for k, (pos, rewards, c) in enumerate(zip(positions, box_data, colors_list)):
            jitter = rng.uniform(-0.15, 0.15, size=len(rewards))
            ax.scatter(pos + jitter, rewards, color=c, s=14, alpha=0.55,
                       edgecolors="none", zorder=2)

        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.set_title(game, fontweight="600")
        if j == 0:
            ax.set_ylabel("Episode Reward")

    fig.suptitle("Classic Control: Reward Distributions by Method",
                 fontsize=14, fontweight="600", y=1.03, fontfamily="serif")

    for fmt in ("png", "svg"):
        out_dir = os.path.join(figure_dir, fmt)
        os.makedirs(out_dir, exist_ok=True)
        fig.savefig(os.path.join(out_dir, f"04_reward_distributions.{fmt}"),
                    dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Distributions saved to {figure_dir}/{{png,svg}}/")


def generate_bar_chart(all_eval_data, expert_rewards, figure_dir, game_names):
    """Generate grouped bar chart of mean rewards."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    AC_SERIES = ['#2563EB', '#D97706', '#059669', '#DC2626',
                 '#7C3AED', '#0891B2', '#BE185D', '#92400E']

    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Inter', 'Helvetica', 'Arial'],
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': True, 'grid.color': '#E9ECEF', 'grid.linewidth': 0.6,
        'axes.edgecolor': '#495057', 'axes.labelcolor': '#212529',
        'xtick.color': '#6C757D', 'ytick.color': '#6C757D',
        'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    })

    method_order = ["Expert", "Multi-Task", "WHC", "Distillation", "Hybrid",
                    "EWC", "Progress & Compress", "TRAC", "C-CHAIN"]
    method_colors = {
        "Expert": AC_SERIES[0], "Multi-Task": AC_SERIES[1],
        "WHC": AC_SERIES[7], "Distillation": AC_SERIES[3],
        "Hybrid": AC_SERIES[2], "EWC": AC_SERIES[6],
        "Progress & Compress": AC_SERIES[5], "TRAC": AC_SERIES[4],
        "C-CHAIN": "#92400E",
    }

    methods = [m for m in method_order if m in all_eval_data]
    n_m = len(methods)
    n_g = len(game_names)

    x = np.arange(n_g)
    width = 0.8 / n_m

    fig, ax = plt.subplots(figsize=(12, 5.5))

    for i, method in enumerate(methods):
        evals = all_eval_data[method]
        eval_dict = {e["game_name"]: e for e in evals}
        means = []
        stds = []
        for game in game_names:
            if game in eval_dict:
                means.append(eval_dict[game]["mean_reward"])
                stds.append(eval_dict[game]["std_reward"])
            else:
                means.append(0)
                stds.append(0)

        offset = (i - n_m / 2 + 0.5) * width
        color = method_colors.get(method, AC_SERIES[i % 8])
        ax.bar(x + offset, means, width * 0.9, yerr=stds,
               label=method, color=mpl.colors.to_rgba(color, 0.7),
               edgecolor=color, linewidth=0.8,
               error_kw=dict(elinewidth=0.8, capsize=2, capthick=0.8, color="#495057"))

    ax.set_xticks(x)
    ax.set_xticklabels(game_names, fontsize=12)
    ax.set_ylabel("Mean Reward", fontsize=12)
    ax.legend(fontsize=8, ncol=3, loc="upper right", framealpha=0.9)
    ax.axhline(0, color="#495057", linewidth=0.5, zorder=0)

    ax.set_title("Classic Control: Mean Reward by Method",
                 fontsize=14, fontweight="600", color="#212529", pad=14,
                 fontfamily="serif")

    for fmt in ("png", "svg"):
        out_dir = os.path.join(figure_dir, fmt)
        os.makedirs(out_dir, exist_ok=True)
        fig.savefig(os.path.join(out_dir, f"01_comparison_bar.{fmt}"),
                    dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Bar chart saved to {figure_dir}/{{png,svg}}/")


def main():
    print(f"Loading evaluation data from: {RESULTS_DIR}")
    all_eval_data, expert_rewards = load_all_evals(RESULTS_DIR)

    print(f"  Found methods: {list(all_eval_data.keys())}")
    print(f"  Expert rewards: {expert_rewards}")
    print()

    print("Generating bar chart...")
    generate_bar_chart(all_eval_data, expert_rewards, RESULTS_DIR, GAME_NAMES)

    print("Generating retention heatmap...")
    generate_heatmap(all_eval_data, expert_rewards, RESULTS_DIR, GAME_NAMES)

    print("Generating reward distributions...")
    generate_reward_distributions(all_eval_data, RESULTS_DIR, GAME_NAMES)

    print("\nDone. All figures saved to:")
    print(f"  {RESULTS_DIR}/png/")
    print(f"  {RESULTS_DIR}/svg/")


if __name__ == "__main__":
    main()
