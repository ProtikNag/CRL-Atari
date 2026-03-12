"""
Q-value distribution visualization across Atari games.

Collects states from each game via expert policy rollout, computes Q-values,
then generates:
  1. UMAP of Q-value vectors (6-dim -> 2D), colored by game
  2. Ridge / KDE plots of max-Q per game
  3. Per-action Q-value heatmap showing action importance per game
  4. Violin plots of Q-values broken down by action and game

Usage:
    python scripts/visualize_qvalues.py [--config configs/base.yaml] [--tag default]
        [--samples-per-game 2000] [--device mps]
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import umap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.dqn import DQNNetwork
from src.data.atari_wrappers import (
    ALE_ACTION_MEANINGS,
    compute_union_action_space,
    get_valid_actions,
    make_atari_env,
)
from src.utils.config import get_effective_config
from src.utils.seed import set_seed


# ── Modern palette ───────────────────────────────────────────────────
# Vibrant, balanced triad inspired by modern data-vis (Observable, Figma)
GAME_COLORS = {
    "Pong":          "#6366F1",   # indigo-500
    "Breakout":      "#F59E0B",   # amber-500
    "SpaceInvaders": "#10B981",   # emerald-500
}
GAME_COLORS_LIGHT = {
    "Pong":          "#A5B4FC",   # indigo-300
    "Breakout":      "#FCD34D",   # amber-300
    "SpaceInvaders": "#6EE7B7",   # emerald-300
}
BG        = "#0F172A"   # slate-900
BG_PANEL  = "#1E293B"   # slate-800
TEXT      = "#F8FAFC"   # slate-50
TEXT_DIM  = "#94A3B8"   # slate-400
GRID_CLR  = "#334155"   # slate-700
ACCENT    = "#38BDF8"   # sky-400

# Heatmap colormap: deep navy -> electric cyan -> warm white
HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "cyber", ["#0F172A", "#1E3A5F", "#0EA5E9", "#38BDF8", "#BAE6FD", "#F0F9FF"],
)


def _apply_modern_style() -> None:
    """Set a dark modern matplotlib style."""
    plt.rcParams.update({
        "figure.facecolor":    BG,
        "axes.facecolor":      BG_PANEL,
        "axes.edgecolor":      GRID_CLR,
        "axes.labelcolor":     TEXT,
        "axes.grid":           True,
        "grid.color":          GRID_CLR,
        "grid.linewidth":      0.4,
        "grid.alpha":          0.5,
        "text.color":          TEXT,
        "xtick.color":         TEXT_DIM,
        "ytick.color":         TEXT_DIM,
        "xtick.labelsize":     10,
        "ytick.labelsize":     10,
        "legend.facecolor":    BG_PANEL,
        "legend.edgecolor":    GRID_CLR,
        "legend.fontsize":     11,
        "font.family":         "sans-serif",
        "font.sans-serif":     ["Inter", "SF Pro Display", "Helvetica Neue",
                                "Segoe UI", "Arial"],
        "figure.dpi":          150,
        "savefig.dpi":         300,
        "savefig.bbox":        "tight",
        "savefig.facecolor":   BG,
    })


def build_model(config: dict, device: str) -> DQNNetwork:
    """Construct a DQN model from config."""
    model_cfg = config["model"]
    return DQNNetwork(
        in_channels=config["env"]["frame_stack"],
        conv_channels=model_cfg["conv_channels"],
        conv_kernels=model_cfg["conv_kernels"],
        conv_strides=model_cfg["conv_strides"],
        fc_hidden=model_cfg["fc_hidden"],
        unified_action_dim=model_cfg["unified_action_dim"],
        dueling=model_cfg.get("dueling", False),
    ).to(device)


def collect_states_and_qvalues(
    env_id: str,
    model: DQNNetwork,
    config: dict,
    device: str,
    num_samples: int,
    union_actions: list,
) -> Tuple[np.ndarray, np.ndarray]:
    """Collect states and corresponding Q-values by running the expert policy.

    Returns:
        states: (N, C, H, W) uint8
        qvalues: (N, num_actions) float32
    """
    env_cfg = config["env"]
    env = make_atari_env(
        env_id=env_id,
        union_actions=union_actions,
        seed=config["seed"],
        frame_stack=env_cfg["frame_stack"],
        frame_skip=env_cfg["frame_skip"],
        screen_size=env_cfg["screen_size"],
        noop_max=env_cfg["noop_max"],
        episodic_life=env_cfg["episodic_life"],
        clip_reward=env_cfg["clip_reward"],
    )

    valid_actions = get_valid_actions(env_id, union_actions)
    model.eval()

    states: List[np.ndarray] = []
    qvalues: List[np.ndarray] = []
    state, _ = env.reset()
    collected = 0

    while collected < num_samples:
        with torch.no_grad():
            state_t = (
                torch.from_numpy(state).float().unsqueeze(0).to(device) / 255.0
            )
            q_vals = model(state_t).cpu().numpy().squeeze(0)

        states.append(state.copy())
        qvalues.append(q_vals)
        collected += 1

        # Take greedy action (masking invalid)
        masked_q = q_vals.copy()
        invalid = [i for i in range(len(masked_q)) if i not in valid_actions]
        masked_q[invalid] = -np.inf
        action = int(np.argmax(masked_q))

        next_state, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            state, _ = env.reset()
        else:
            state = next_state

    env.close()
    return np.array(states[:num_samples]), np.array(qvalues[:num_samples])


def plot_umap_qvalues(
    qvalues_dict: Dict[str, np.ndarray],
    action_labels: List[str],
    save_dir: str,
    tag: str,
    umap_neighbors: int = 15,
    umap_min_dist: float = 0.1,
    seed: int = 42,
) -> None:
    """UMAP of Q-value vectors colored by game."""
    all_q = []
    all_labels = []
    for game, qv in qvalues_dict.items():
        all_q.append(qv)
        all_labels.extend([game] * len(qv))
    all_q = np.concatenate(all_q, axis=0)
    all_labels = np.array(all_labels)

    print(f"Running UMAP on Q-values ({all_q.shape})...")
    reducer = umap.UMAP(
        n_neighbors=umap_neighbors,
        min_dist=umap_min_dist,
        n_components=2,
        metric="euclidean",
        random_state=seed,
    )
    embedding = reducer.fit_transform(all_q)

    fig, ax = plt.subplots(figsize=(9, 8))

    games = list(qvalues_dict.keys())
    for game in games:
        mask = all_labels == game
        color = GAME_COLORS.get(game, ACCENT)
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            c=color,
            s=10,
            alpha=0.5,
            label=game,
            edgecolors="none",
            rasterized=True,
        )

    ax.set_xlabel("UMAP 1", fontsize=12)
    ax.set_ylabel("UMAP 2", fontsize=12)
    ax.set_title(
        "Q-Value Landscape  ·  UMAP Projection",
        fontsize=18, fontweight="bold", pad=16,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.grid(False)

    legend = ax.legend(
        loc="upper right", fontsize=12, frameon=True,
        framealpha=0.85, markerscale=2.5, borderpad=0.8,
    )
    for handle in legend.legend_handles:
        handle.set_alpha(0.9)

    _save(fig, save_dir, f"qvalue_umap_{tag}")
    plt.close(fig)


def plot_maxq_distributions(
    qvalues_dict: Dict[str, np.ndarray],
    save_dir: str,
    tag: str,
) -> None:
    """Ridge / overlapping KDE of max-Q values per game."""
    fig, ax = plt.subplots(figsize=(10, 5))

    games = list(qvalues_dict.keys())
    for i, game in enumerate(games):
        max_q = qvalues_dict[game].max(axis=1)
        color = GAME_COLORS.get(game, ACCENT)
        light = GAME_COLORS_LIGHT.get(game, color)

        sns.kdeplot(
            max_q, ax=ax, color=color, linewidth=2.2, label=game,
            fill=True, alpha=0.18, common_norm=False,
        )

    ax.set_xlabel("Max Q-Value", fontsize=13, labelpad=8)
    ax.set_ylabel("Density", fontsize=13, labelpad=8)
    ax.set_title(
        "Distribution of Max Q-Values by Game",
        fontsize=17, fontweight="bold", pad=14,
    )
    ax.legend(fontsize=12, framealpha=0.85)
    ax.grid(axis="y", alpha=0.3)
    ax.grid(axis="x", alpha=0.15)
    sns.despine(ax=ax)

    _save(fig, save_dir, f"qvalue_maxq_kde_{tag}")
    plt.close(fig)


def plot_action_heatmap(
    qvalues_dict: Dict[str, np.ndarray],
    action_labels: List[str],
    union_actions: List[int],
    valid_actions_map: Dict[str, List[int]],
    save_dir: str,
    tag: str,
) -> None:
    """Heatmap of mean Q-values per action per game."""
    games = list(qvalues_dict.keys())
    n_actions = len(action_labels)

    # Build matrix: rows=games, cols=actions
    mean_q = np.full((len(games), n_actions), np.nan)
    for i, game in enumerate(games):
        valid = valid_actions_map[game]
        qv = qvalues_dict[game]
        for j in range(n_actions):
            if j in valid:
                mean_q[i, j] = qv[:, j].mean()

    fig, ax = plt.subplots(figsize=(10, 4))

    # Mask NaN for display
    masked = np.ma.masked_invalid(mean_q)
    im = ax.imshow(
        masked, cmap=HEATMAP_CMAP, aspect="auto",
        interpolation="nearest",
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.04)
    cbar.set_label("Mean Q-Value", fontsize=11, color=TEXT)
    cbar.ax.yaxis.set_tick_params(color=TEXT_DIM)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT_DIM)

    ax.set_xticks(range(n_actions))
    ax.set_xticklabels(action_labels, rotation=35, ha="right", fontsize=11)
    ax.set_yticks(range(len(games)))
    ax.set_yticklabels(games, fontsize=12)

    # Annotate cells
    for i in range(len(games)):
        for j in range(n_actions):
            val = mean_q[i, j]
            if not np.isnan(val):
                text_c = TEXT if val < (np.nanmax(mean_q) * 0.6) else "#0F172A"
                ax.text(
                    j, i, f"{val:.1f}",
                    ha="center", va="center", fontsize=10,
                    fontweight="medium", color=text_c,
                )
            else:
                ax.text(
                    j, i, "—",
                    ha="center", va="center", fontsize=10,
                    color=GRID_CLR,
                )

    ax.set_title(
        "Mean Q-Value per Action × Game",
        fontsize=17, fontweight="bold", pad=14,
    )
    ax.tick_params(length=0)

    _save(fig, save_dir, f"qvalue_action_heatmap_{tag}")
    plt.close(fig)


def plot_per_action_violins(
    qvalues_dict: Dict[str, np.ndarray],
    action_labels: List[str],
    valid_actions_map: Dict[str, List[int]],
    save_dir: str,
    tag: str,
) -> None:
    """Violin plots of Q-values per action, faceted by game."""
    import pandas as pd

    games = list(qvalues_dict.keys())
    rows = []
    for game in games:
        valid = valid_actions_map[game]
        qv = qvalues_dict[game]
        for j in valid:
            for val in qv[:, j]:
                rows.append({
                    "Game": game,
                    "Action": action_labels[j],
                    "Q-Value": float(val),
                })
    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(
        1, len(games), figsize=(6 * len(games), 5.5), sharey=False,
    )
    if len(games) == 1:
        axes = [axes]

    for ax, game in zip(axes, games):
        sub = df[df["Game"] == game]
        color = GAME_COLORS.get(game, ACCENT)
        light = GAME_COLORS_LIGHT.get(game, color)

        valid = valid_actions_map[game]
        valid_labels = [action_labels[j] for j in valid]

        parts = ax.violinplot(
            [sub[sub["Action"] == lbl]["Q-Value"].values for lbl in valid_labels],
            showmeans=True, showextrema=False, showmedians=True,
        )
        for pc in parts["bodies"]:
            pc.set_facecolor(color)
            pc.set_alpha(0.35)
            pc.set_edgecolor(color)
            pc.set_linewidth(1.0)
        parts["cmeans"].set_color(ACCENT)
        parts["cmeans"].set_linewidth(1.5)
        parts["cmedians"].set_color(TEXT)
        parts["cmedians"].set_linewidth(1.2)

        ax.set_xticks(range(1, len(valid_labels) + 1))
        ax.set_xticklabels(valid_labels, rotation=35, ha="right", fontsize=10)
        ax.set_title(game, fontsize=14, fontweight="bold", color=color, pad=10)
        ax.set_ylabel("Q-Value" if ax == axes[0] else "", fontsize=12)
        ax.grid(axis="y", alpha=0.3)
        ax.grid(axis="x", alpha=0.0)
        sns.despine(ax=ax, left=False, bottom=True)

    fig.suptitle(
        "Q-Value Distribution per Action",
        fontsize=18, fontweight="bold", y=1.02,
    )
    fig.tight_layout()

    _save(fig, save_dir, f"qvalue_action_violins_{tag}")
    plt.close(fig)


def plot_combined_dashboard(
    qvalues_dict: Dict[str, np.ndarray],
    action_labels: List[str],
    union_actions: List[int],
    valid_actions_map: Dict[str, List[int]],
    embedding: np.ndarray,
    all_labels: np.ndarray,
    save_dir: str,
    tag: str,
) -> None:
    """Combined 2x2 dashboard figure."""
    games = list(qvalues_dict.keys())
    n_actions = len(action_labels)

    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(
        2, 2, figure=fig, hspace=0.32, wspace=0.28,
        left=0.06, right=0.96, top=0.93, bottom=0.06,
    )

    # ── Panel A: UMAP ────────────────────────────────────────────────
    ax_umap = fig.add_subplot(gs[0, 0])
    for game in games:
        mask = all_labels == game
        ax_umap.scatter(
            embedding[mask, 0], embedding[mask, 1],
            c=GAME_COLORS.get(game, ACCENT), s=6, alpha=0.45,
            label=game, edgecolors="none", rasterized=True,
        )
    ax_umap.set_title("A   Q-Value UMAP", fontsize=14, fontweight="bold",
                       loc="left", pad=10)
    ax_umap.set_xticks([]); ax_umap.set_yticks([])
    ax_umap.spines["left"].set_visible(False)
    ax_umap.spines["bottom"].set_visible(False)
    ax_umap.grid(False)
    ax_umap.legend(fontsize=10, markerscale=2.5, framealpha=0.8)

    # ── Panel B: Max-Q KDE ───────────────────────────────────────────
    ax_kde = fig.add_subplot(gs[0, 1])
    for game in games:
        max_q = qvalues_dict[game].max(axis=1)
        sns.kdeplot(
            max_q, ax=ax_kde, color=GAME_COLORS.get(game, ACCENT),
            linewidth=2.0, label=game, fill=True, alpha=0.15,
            common_norm=False,
        )
    ax_kde.set_title("B   Max Q-Value Density", fontsize=14,
                      fontweight="bold", loc="left", pad=10)
    ax_kde.set_xlabel("Max Q", fontsize=11)
    ax_kde.set_ylabel("Density", fontsize=11)
    ax_kde.legend(fontsize=10, framealpha=0.8)
    ax_kde.grid(axis="y", alpha=0.25)
    sns.despine(ax=ax_kde)

    # ── Panel C: Action heatmap ──────────────────────────────────────
    ax_heat = fig.add_subplot(gs[1, 0])
    mean_q = np.full((len(games), n_actions), np.nan)
    for i, game in enumerate(games):
        valid = valid_actions_map[game]
        qv = qvalues_dict[game]
        for j in range(n_actions):
            if j in valid:
                mean_q[i, j] = qv[:, j].mean()

    masked = np.ma.masked_invalid(mean_q)
    im = ax_heat.imshow(masked, cmap=HEATMAP_CMAP, aspect="auto",
                         interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax_heat, fraction=0.025, pad=0.03)
    cbar.set_label("Mean Q", fontsize=10, color=TEXT)
    cbar.ax.yaxis.set_tick_params(color=TEXT_DIM)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT_DIM, fontsize=9)

    ax_heat.set_xticks(range(n_actions))
    ax_heat.set_xticklabels(action_labels, rotation=35, ha="right", fontsize=10)
    ax_heat.set_yticks(range(len(games)))
    ax_heat.set_yticklabels(games, fontsize=11)
    ax_heat.tick_params(length=0)
    ax_heat.set_title("C   Mean Q per Action", fontsize=14,
                       fontweight="bold", loc="left", pad=10)
    for i in range(len(games)):
        for j in range(n_actions):
            val = mean_q[i, j]
            if not np.isnan(val):
                tc = TEXT if val < (np.nanmax(mean_q) * 0.6) else "#0F172A"
                ax_heat.text(j, i, f"{val:.1f}", ha="center", va="center",
                             fontsize=9, fontweight="medium", color=tc)
            else:
                ax_heat.text(j, i, "—", ha="center", va="center",
                             fontsize=9, color=GRID_CLR)

    # ── Panel D: Std-dev heatmap (Q-value spread per action) ─────────
    ax_std = fig.add_subplot(gs[1, 1])
    std_q = np.full((len(games), n_actions), np.nan)
    for i, game in enumerate(games):
        valid = valid_actions_map[game]
        qv = qvalues_dict[game]
        for j in range(n_actions):
            if j in valid:
                std_q[i, j] = qv[:, j].std()

    masked_std = np.ma.masked_invalid(std_q)
    std_cmap = LinearSegmentedColormap.from_list(
        "warmth", ["#1E293B", "#92400E", "#F59E0B", "#FCD34D", "#FFFBEB"],
    )
    im2 = ax_std.imshow(masked_std, cmap=std_cmap, aspect="auto",
                         interpolation="nearest")
    cbar2 = fig.colorbar(im2, ax=ax_std, fraction=0.025, pad=0.03)
    cbar2.set_label("Std Q", fontsize=10, color=TEXT)
    cbar2.ax.yaxis.set_tick_params(color=TEXT_DIM)
    plt.setp(cbar2.ax.yaxis.get_ticklabels(), color=TEXT_DIM, fontsize=9)

    ax_std.set_xticks(range(n_actions))
    ax_std.set_xticklabels(action_labels, rotation=35, ha="right", fontsize=10)
    ax_std.set_yticks(range(len(games)))
    ax_std.set_yticklabels(games, fontsize=11)
    ax_std.tick_params(length=0)
    ax_std.set_title("D   Q-Value Spread (σ) per Action", fontsize=14,
                      fontweight="bold", loc="left", pad=10)
    for i in range(len(games)):
        for j in range(n_actions):
            val = std_q[i, j]
            if not np.isnan(val):
                tc = TEXT if val < (np.nanmax(std_q) * 0.6) else "#0F172A"
                ax_std.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=9, fontweight="medium", color=tc)
            else:
                ax_std.text(j, i, "—", ha="center", va="center",
                            fontsize=9, color=GRID_CLR)

    fig.suptitle(
        "Q-Value Analysis Dashboard",
        fontsize=22, fontweight="bold", color=TEXT, y=0.97,
    )

    _save(fig, save_dir, f"qvalue_dashboard_{tag}")
    plt.close(fig)


def _save(fig: plt.Figure, save_dir: str, name: str) -> None:
    """Save figure as PNG and SVG."""
    png_dir = os.path.join(save_dir, "png")
    svg_dir = os.path.join(save_dir, "svg")
    os.makedirs(png_dir, exist_ok=True)
    os.makedirs(svg_dir, exist_ok=True)

    png_path = os.path.join(png_dir, f"{name}.png")
    svg_path = os.path.join(svg_dir, f"{name}.svg")
    fig.savefig(png_path, dpi=300, facecolor=fig.get_facecolor())
    fig.savefig(svg_path, facecolor=fig.get_facecolor())
    print(f"  Saved: {png_path}")
    print(f"  Saved: {svg_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Q-value distribution visualization across Atari games.",
    )
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--tag", type=str, default="default")
    parser.add_argument("--samples-per-game", type=int, default=2000)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--umap-neighbors", type=int, default=15)
    parser.add_argument("--umap-min-dist", type=float, default=0.1)
    args = parser.parse_args()

    _apply_modern_style()

    config = get_effective_config(args.config)

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    set_seed(config["seed"])

    union_actions = compute_union_action_space(config["task_sequence"])
    config["model"]["unified_action_dim"] = len(union_actions)
    action_labels = [ALE_ACTION_MEANINGS[a] for a in union_actions]
    print(f"Union actions: {union_actions}")
    print(f"Action labels: {action_labels}")
    print(f"Device: {device}\n")

    checkpoint_dir = config["logging"]["checkpoint_dir"]
    figure_dir = config["logging"]["figure_dir"]

    summary_path = os.path.join(checkpoint_dir, args.tag, "expert_summary.json")
    with open(summary_path) as f:
        summary = json.load(f)

    # ── Collect data ─────────────────────────────────────────────────
    qvalues_dict: Dict[str, np.ndarray] = {}
    valid_actions_map: Dict[str, List[int]] = {}
    all_labels_list: List[str] = []

    for expert_info in summary["expert_results"]:
        game_name = expert_info["game_name"]
        env_id = expert_info["env_id"]

        ckpt_path = os.path.join(
            checkpoint_dir, args.tag, f"expert_{game_name}_best.pt",
        )
        print(f"Loading expert: {game_name}")
        model = build_model(config, device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["policy_net"])
        model.eval()

        print(f"  Collecting {args.samples_per_game} states + Q-values...")
        _, qvalues = collect_states_and_qvalues(
            env_id, model, config, device,
            args.samples_per_game, union_actions,
        )
        print(f"  Q-values shape: {qvalues.shape} "
              f"(range: [{qvalues.min():.2f}, {qvalues.max():.2f}])")

        valid = get_valid_actions(env_id, union_actions)
        qvalues_dict[game_name] = qvalues
        valid_actions_map[game_name] = valid
        all_labels_list.extend([game_name] * len(qvalues))

    # ── UMAP embedding (compute once, reuse) ─────────────────────────
    all_q_concat = np.concatenate(list(qvalues_dict.values()), axis=0)
    all_labels_arr = np.array(all_labels_list)

    print(f"\nRunning UMAP on Q-values {all_q_concat.shape}...")
    reducer = umap.UMAP(
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
        n_components=2,
        metric="euclidean",
        random_state=config["seed"],
    )
    embedding = reducer.fit_transform(all_q_concat)
    print(f"Embedding shape: {embedding.shape}\n")

    # ── Generate figures ─────────────────────────────────────────────
    print("Generating figures...")

    plot_umap_qvalues(
        qvalues_dict, action_labels, figure_dir, args.tag,
        args.umap_neighbors, args.umap_min_dist, config["seed"],
    )

    plot_maxq_distributions(qvalues_dict, figure_dir, args.tag)

    plot_action_heatmap(
        qvalues_dict, action_labels, union_actions,
        valid_actions_map, figure_dir, args.tag,
    )

    plot_per_action_violins(
        qvalues_dict, action_labels,
        valid_actions_map, figure_dir, args.tag,
    )

    plot_combined_dashboard(
        qvalues_dict, action_labels, union_actions,
        valid_actions_map, embedding, all_labels_arr,
        figure_dir, args.tag,
    )

    print("\nDone. All figures saved.")


if __name__ == "__main__":
    main()
