"""
Loss landscape visualization for Atari CRL consolidation.

Projects expert and consolidated parameter vectors onto a 2D PCA plane,
then evaluates the DQN Bellman (TD) loss at a grid of interpolated model
weights.

Produces:
  1. Per-game 2D contour plots showing each game's loss landscape with
     the three expert minima marked (star markers)
  2. Combined 2D contour of across-game summed loss with all models
     plotted (experts + one-shot + iterative + hybrid + distillation)
  3. Combined 3D surface with the same model positions on the surface

Methodology follows Li et al. (2018) "Visualizing the Loss Landscape of
Neural Nets" with PCA directions derived from expert weight vectors and
filter-wise normalization.

Usage:
    python scripts/visualize_loss_landscape.py [--config configs/base.yaml]
        [--tag default] [--grid-size 31] [--range 1.0]
        [--eval-samples 256] [--device mps]
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.dqn import DQNNetwork
from src.data.atari_wrappers import (
    compute_union_action_space,
    get_valid_actions,
    make_atari_env,
)
from src.data.replay_buffer import ReplayBuffer
from src.utils.config import get_effective_config
from src.utils.seed import set_seed


# -- Palette: light warm background, saturated accents -----------------

BG        = "#F9FAFB"   # gray-50  (very light, not pure white)
BG_PANEL  = "#F3F4F6"   # gray-100
TEXT      = "#111827"    # gray-900
TEXT_DIM  = "#6B7280"    # gray-500
GRID_CLR  = "#E5E7EB"   # gray-200
BORDER    = "#D1D5DB"    # gray-300

# Expert colors (same indigo / amber / emerald from previous visualizations)
GAME_COLORS = {
    "Pong":          "#6366F1",   # indigo-500
    "Breakout":      "#F59E0B",   # amber-500
    "SpaceInvaders": "#10B981",   # emerald-500
}

# Consolidated method colors and markers
METHOD_STYLE = {
    "One-Shot":     {"color": "#EF4444", "marker": "D", "size": 110},  # red-500
    "Iterative":    {"color": "#8B5CF6", "marker": "s", "size": 100},  # violet-500
    "Hybrid":       {"color": "#EC4899", "marker": "P", "size": 120},  # pink-500
    "Distillation": {"color": "#0EA5E9", "marker": "^", "size": 110},  # sky-500
}

# Per-game contour colormaps (light bg friendly)
GAME_CMAPS = {
    "Pong": LinearSegmentedColormap.from_list(
        "indigo_l", ["#EEF2FF", "#C7D2FE", "#818CF8", "#6366F1",
                      "#4338CA", "#312E81"]),
    "Breakout": LinearSegmentedColormap.from_list(
        "amber_l", ["#FFFBEB", "#FDE68A", "#FBBF24", "#F59E0B",
                     "#B45309", "#78350F"]),
    "SpaceInvaders": LinearSegmentedColormap.from_list(
        "emerald_l", ["#ECFDF5", "#A7F3D0", "#34D399", "#10B981",
                       "#047857", "#064E3B"]),
}
COMBINED_CMAP = LinearSegmentedColormap.from_list(
    "warm_depth", ["#F9FAFB", "#DBEAFE", "#93C5FD", "#60A5FA",
                    "#FBBF24", "#F97316", "#EF4444", "#991B1B"],
)


def _apply_style() -> None:
    """Light, minimalistic matplotlib style."""
    plt.rcParams.update({
        "figure.facecolor":    BG,
        "axes.facecolor":      BG_PANEL,
        "axes.edgecolor":      BORDER,
        "axes.labelcolor":     TEXT,
        "axes.grid":           False,
        "text.color":          TEXT,
        "xtick.color":         TEXT_DIM,
        "ytick.color":         TEXT_DIM,
        "xtick.labelsize":     10,
        "ytick.labelsize":     10,
        "legend.facecolor":    BG,
        "legend.edgecolor":    BORDER,
        "legend.fontsize":     11,
        "font.family":         "sans-serif",
        "font.sans-serif":     ["Inter", "SF Pro Display", "Helvetica Neue",
                                "Segoe UI", "Arial"],
        "figure.dpi":          150,
        "savefig.dpi":         300,
        "savefig.bbox":        "tight",
        "savefig.facecolor":   BG,
    })


# -- Utilities ---------------------------------------------------------

def build_model(config: dict, device: str) -> DQNNetwork:
    """Construct a DQN model from config."""
    m = config["model"]
    return DQNNetwork(
        in_channels=config["env"]["frame_stack"],
        conv_channels=m["conv_channels"], conv_kernels=m["conv_kernels"],
        conv_strides=m["conv_strides"], fc_hidden=m["fc_hidden"],
        unified_action_dim=m["unified_action_dim"],
        dueling=m.get("dueling", False),
    ).to(device)


def flatten_params(sd: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Flatten a state dict into a single 1-D vector."""
    return torch.cat([v.flatten().float() for v in sd.values()])


def unflatten_params(flat: torch.Tensor,
                     ref: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Reshape a flat vector back into a state dict."""
    sd, offset = {}, 0
    for name, t in ref.items():
        n = t.numel()
        sd[name] = flat[offset:offset + n].reshape(t.shape)
        offset += n
    return sd


def filter_normalize(direction: torch.Tensor, ref_flat: torch.Tensor,
                     shapes: List[Tuple[str, torch.Size]]) -> torch.Tensor:
    """Apply filter-wise normalization (Li et al. 2018, Sec 4)."""
    normed = torch.zeros_like(direction)
    offset = 0
    for _, shape in shapes:
        n = shape.numel()
        d = direction[offset:offset + n].reshape(shape)
        w = ref_flat[offset:offset + n].reshape(shape)
        if len(shape) >= 2:
            for i in range(shape[0]):
                wn, dn = w[i].norm(), d[i].norm()
                if dn > 1e-10 and wn > 1e-10:
                    d[i] = d[i] * (wn / dn)
        else:
            wn, dn = w.norm(), d.norm()
            if dn > 1e-10 and wn > 1e-10:
                d = d * (wn / dn)
        normed[offset:offset + n] = d.flatten()
        offset += n
    return normed


def compute_pca_directions(
    expert_sds: List[Dict[str, torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute two PCA directions from the expert weight vectors.

    Returns:
        (dir1, dir2, center)
    """
    mat = torch.stack([flatten_params(sd) for sd in expert_sds], dim=0).float()
    center = mat.mean(dim=0)
    centered = mat - center.unsqueeze(0)
    U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
    d1 = Vh[0] / Vh[0].norm()
    d2 = (Vh[1] / Vh[1].norm()) if Vh.shape[0] > 1 else torch.randn_like(d1)
    return d1, d2, center


def project(sd: Dict[str, torch.Tensor], center: torch.Tensor,
            d1: torch.Tensor, d2: torch.Tensor) -> Tuple[float, float]:
    """Project a state dict onto the 2D PCA plane."""
    diff = flatten_params(sd) - center
    return torch.dot(diff, d1).item(), torch.dot(diff, d2).item()


def collect_eval_data(
    env_id: str, model: DQNNetwork, config: dict, device: str,
    num_samples: int, union_actions: list,
) -> ReplayBuffer:
    """Collect transitions by running the expert policy."""
    ec = config["env"]
    env = make_atari_env(
        env_id=env_id, union_actions=union_actions, seed=config["seed"],
        frame_stack=ec["frame_stack"], frame_skip=ec["frame_skip"],
        screen_size=ec["screen_size"], noop_max=ec["noop_max"],
        episodic_life=ec["episodic_life"], clip_reward=ec["clip_reward"],
    )
    valid = get_valid_actions(env_id, union_actions)
    buf = ReplayBuffer(
        capacity=num_samples, frame_stack=ec["frame_stack"],
        frame_shape=(ec["screen_size"], ec["screen_size"]), device=device,
    )
    model.eval()
    state, _ = env.reset()
    while buf.size < num_samples:
        with torch.no_grad():
            st = torch.from_numpy(state).float().unsqueeze(0).to(device) / 255.0
            q = model(st)
            mask = torch.full((model.unified_action_dim,), float("-inf"),
                              device=device)
            mask[valid] = 0.0
            action = (q + mask.unsqueeze(0)).argmax(dim=1).item()
        ns, rew, term, trunc, _ = env.step(action)
        buf.push(state, action, rew, ns, term)
        state = (env.reset()[0] if term or trunc else ns)
    env.close()
    return buf


def evaluate_loss(
    model: DQNNetwork, sd: Dict[str, torch.Tensor],
    buf: ReplayBuffer, device: str,
    num_samples: int = 256, gamma: float = 0.99,
) -> float:
    """Evaluate Bellman (smooth-L1) loss at given weights."""
    model.load_state_dict(sd)
    model.eval()
    states, actions, rewards, next_states, dones = buf.sample(
        min(num_samples, buf.size))
    with torch.no_grad():
        q = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        q_next_max = model(next_states).max(dim=1).values
        target = rewards + gamma * q_next_max * (1.0 - dones)
        return F.smooth_l1_loss(q, target).item()


def _save(fig: plt.Figure, save_dir: str, name: str) -> None:
    """Save figure as PNG and SVG."""
    for sub, ext in [("png", "png"), ("svg", "svg")]:
        d = os.path.join(save_dir, sub)
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, f"{name}.{ext}")
        fig.savefig(p, dpi=300, facecolor=fig.get_facecolor())
        print(f"  Saved: {p}")


# -- Main --------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Loss landscape visualization for Atari CRL.",
    )
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--tag", type=str, default="default")
    parser.add_argument("--grid-size", type=int, default=31)
    parser.add_argument("--range", type=float, default=1.0)
    parser.add_argument("--eval-samples", type=int, default=256)
    parser.add_argument("--collect-samples", type=int, default=1000)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--gamma", type=float, default=0.99)
    args = parser.parse_args()

    _apply_style()
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
    print(f"Union actions: {union_actions} ({len(union_actions)})")
    print(f"Device: {device}")
    print(f"Grid: {args.grid_size}x{args.grid_size}\n")

    ckpt_dir = os.path.join(config["logging"]["checkpoint_dir"], args.tag)
    fig_dir = config["logging"]["figure_dir"]

    with open(os.path.join(ckpt_dir, "expert_summary.json")) as f:
        summary = json.load(f)

    # -- Load experts ---------------------------------------------------
    expert_sds: List[Dict[str, torch.Tensor]] = []
    game_names: List[str] = []
    game_buffers: Dict[str, ReplayBuffer] = {}

    for info in summary["expert_results"]:
        gname, env_id = info["game_name"], info["env_id"]
        game_names.append(gname)
        ckpt = torch.load(
            os.path.join(ckpt_dir, f"expert_{gname}_best.pt"),
            map_location=device, weights_only=False,
        )
        sd = ckpt["policy_net"]
        expert_sds.append(sd)

        model = build_model(config, device)
        model.load_state_dict(sd)
        print(f"Collecting {args.collect_samples} transitions for {gname}...")
        game_buffers[gname] = collect_eval_data(
            env_id, model, config, device, args.collect_samples, union_actions,
        )

    # -- Load consolidated models ---------------------------------------
    CONSOLIDATED = {
        "One-Shot":     "consolidated_oneshot.pt",
        "Iterative":    "consolidated_iterative.pt",
        "Hybrid":       "consolidated_hybrid.pt",
        "Distillation": "consolidated_distillation.pt",
    }
    consol_sds: Dict[str, Dict[str, torch.Tensor]] = {}
    for label, fname in CONSOLIDATED.items():
        fpath = os.path.join(ckpt_dir, fname)
        if os.path.exists(fpath):
            sd = torch.load(fpath, map_location=device, weights_only=False)
            if isinstance(sd, dict) and "policy_net" in sd:
                sd = sd["policy_net"]
            consol_sds[label] = sd
            print(f"Loaded consolidated: {label}")
        else:
            print(f"  (not found: {fname})")

    # -- PCA directions -------------------------------------------------
    print("\nComputing PCA directions...")
    d1, d2, center = compute_pca_directions(expert_sds)
    shapes = [(n, p.shape) for n, p in expert_sds[0].items()]
    d1 = filter_normalize(d1, center, shapes)
    d2 = filter_normalize(d2, center, shapes)

    expert_coords = {gn: project(sd, center, d1, d2)
                     for gn, sd in zip(game_names, expert_sds)}
    consol_coords = {lab: project(sd, center, d1, d2)
                     for lab, sd in consol_sds.items()}

    print("Expert projections:")
    for gn, (x, y) in expert_coords.items():
        print(f"  {gn}: ({x:.2f}, {y:.2f})")
    for lab, (x, y) in consol_coords.items():
        print(f"  {lab}: ({x:.2f}, {y:.2f})")

    # -- Build grid -----------------------------------------------------
    all_xy = list(expert_coords.values()) + list(consol_coords.values())
    extent = max(max(abs(c[0]) for c in all_xy),
                 max(abs(c[1]) for c in all_xy)) * 1.5 * args.range

    alphas = np.linspace(-extent, extent, args.grid_size)
    betas = np.linspace(-extent, extent, args.grid_size)
    A, B = np.meshgrid(alphas, betas)

    # -- Evaluate grid --------------------------------------------------
    ref_sd = expert_sds[0]
    loss_grids: Dict[str, np.ndarray] = {g: np.zeros_like(A) for g in game_names}
    eval_model = build_model(config, device)

    total = args.grid_size ** 2
    print(f"\nEvaluating {total} grid points x {len(game_names)} games...")
    for i in range(args.grid_size):
        if (i + 1) % 5 == 0 or i == 0:
            print(f"  Row {i + 1}/{args.grid_size}...")
        for j in range(args.grid_size):
            flat = center + A[i, j] * d1 + B[i, j] * d2
            grid_sd = {k: v.to(device)
                       for k, v in unflatten_params(flat, ref_sd).items()}
            for gn in game_names:
                loss_grids[gn][i, j] = evaluate_loss(
                    eval_model, grid_sd, game_buffers[gn],
                    device, args.eval_samples, args.gamma,
                )

    summed = sum(loss_grids[g] for g in game_names)
    log_grids = {g: np.log1p(loss_grids[g]) for g in game_names}
    log_summed = np.log1p(summed)

    # ================================================================
    # Figure 1: Per-game contour plots (1x3) -- experts only
    # ================================================================
    print("\nGenerating per-game contour plots...")
    fig, axes = plt.subplots(1, 3, figsize=(21, 6.5))

    for idx, (gn, ax) in enumerate(zip(game_names, axes)):
        Z = log_grids[gn]
        cmap = GAME_CMAPS.get(gn, COMBINED_CMAP)

        cf = ax.contourf(A, B, Z, levels=30, cmap=cmap, alpha=0.92)
        ax.contour(A, B, Z, levels=15, colors=["#374151"],
                   linewidths=0.25, alpha=0.3)

        # Mark all three experts
        for k, (egn, (ex, ey)) in enumerate(expert_coords.items()):
            is_this = (egn == gn)
            ax.scatter(
                ex, ey, c=GAME_COLORS[egn],
                s=200 if is_this else 80,
                edgecolors="white" if is_this else "#374151",
                linewidths=2.5 if is_this else 1,
                zorder=10, marker="*",
                label=f"{egn}" if idx == 0 else None,
            )
            if is_this:
                ax.annotate(
                    f"{gn}\nminimum",
                    (ex, ey), textcoords="offset points",
                    xytext=(12, 12), fontsize=9, fontweight="bold",
                    color=GAME_COLORS[gn],
                    bbox=dict(boxstyle="round,pad=0.3", fc=BG, ec=BORDER,
                              alpha=0.85),
                    arrowprops=dict(arrowstyle="-|>", color=GAME_COLORS[gn],
                                    lw=1.2),
                )

        ax.set_title(gn, fontsize=15, fontweight="bold",
                     color=GAME_COLORS[gn], pad=10)
        ax.set_xlabel("PC 1", fontsize=11)
        ax.set_ylabel("PC 2" if idx == 0 else "", fontsize=11)
        ax.tick_params(length=0)

        cbar = fig.colorbar(cf, ax=ax, fraction=0.04, pad=0.03)
        cbar.set_label("log(1 + loss)", fontsize=9, color=TEXT_DIM)
        cbar.ax.yaxis.set_tick_params(color=TEXT_DIM)
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT_DIM, fontsize=8)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=11,
               framealpha=0.9, borderpad=0.6, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Per-Game Loss Landscape  \u00b7  Expert Minima",
                 fontsize=19, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, fig_dir, f"loss_landscape_per_game_{args.tag}")
    plt.close(fig)

    # ================================================================
    # Figure 2: Combined 2D contour -- all models
    # ================================================================
    print("Generating combined 2D contour...")
    fig, ax = plt.subplots(figsize=(11, 9))

    cf = ax.contourf(A, B, log_summed, levels=40, cmap=COMBINED_CMAP, alpha=0.92)
    ax.contour(A, B, log_summed, levels=20, colors=["#374151"],
               linewidths=0.25, alpha=0.25)

    # Experts (stars)
    for gn, (ex, ey) in expert_coords.items():
        ax.scatter(ex, ey, c=GAME_COLORS[gn], s=220, edgecolors="white",
                   linewidths=2.5, zorder=10, marker="*",
                   label=f"{gn} Expert")
        ax.annotate(gn, (ex, ey), textcoords="offset points",
                    xytext=(12, 10), fontsize=11, fontweight="bold",
                    color=GAME_COLORS[gn],
                    bbox=dict(boxstyle="round,pad=0.25", fc=BG, ec="none",
                              alpha=0.8))

    # Consolidated methods (distinct markers)
    for lab, (cx, cy) in consol_coords.items():
        st = METHOD_STYLE[lab]
        ax.scatter(cx, cy, c=st["color"], s=st["size"],
                   edgecolors="white", linewidths=1.8, zorder=10,
                   marker=st["marker"], label=lab)
        ax.annotate(lab, (cx, cy), textcoords="offset points",
                    xytext=(10, -14), fontsize=9.5, fontweight="medium",
                    color=st["color"],
                    bbox=dict(boxstyle="round,pad=0.2", fc=BG, ec="none",
                              alpha=0.8))

    ax.set_xlabel("PC 1", fontsize=13)
    ax.set_ylabel("PC 2", fontsize=13)
    ax.set_title("Combined Loss Landscape  \u00b7  All Models",
                 fontsize=18, fontweight="bold", pad=14)
    ax.legend(fontsize=10.5, framealpha=0.92, loc="upper right",
              borderpad=0.7, handletextpad=0.6)
    ax.tick_params(length=0)

    cbar = fig.colorbar(cf, ax=ax, fraction=0.03, pad=0.03)
    cbar.set_label("log(1 + \u03a3 losses)", fontsize=11, color=TEXT_DIM)
    cbar.ax.yaxis.set_tick_params(color=TEXT_DIM)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT_DIM, fontsize=9)

    fig.tight_layout()
    _save(fig, fig_dir, f"loss_landscape_combined_{args.tag}")
    plt.close(fig)

    # ================================================================
    # Figure 3: Combined 3D surface -- all models
    # ================================================================
    print("Generating 3D surface...")
    fig = plt.figure(figsize=(13, 10))
    ax3 = fig.add_subplot(111, projection="3d")
    ax3.set_facecolor(BG_PANEL)

    surf = ax3.plot_surface(
        A, B, log_summed, cmap=COMBINED_CMAP, alpha=0.82,
        edgecolor="none", antialiased=True,
        rcount=args.grid_size, ccount=args.grid_size,
    )

    z_range = log_summed.max() - log_summed.min()

    # Experts on surface
    for gn, (ex, ey) in expert_coords.items():
        ix = np.argmin(np.abs(alphas - ex))
        iy = np.argmin(np.abs(betas - ey))
        ez = log_summed[iy, ix]
        ax3.scatter([ex], [ey], [ez], c=GAME_COLORS[gn], s=200,
                    edgecolors="white", linewidths=2, zorder=10, marker="*")
        ax3.text(ex, ey, ez + z_range * 0.04, gn,
                 fontsize=10, fontweight="bold", color=GAME_COLORS[gn],
                 ha="center")

    # Consolidated on surface
    for lab, (cx, cy) in consol_coords.items():
        st = METHOD_STYLE[lab]
        ix = np.argmin(np.abs(alphas - cx))
        iy = np.argmin(np.abs(betas - cy))
        cz = log_summed[iy, ix]
        ax3.scatter([cx], [cy], [cz], c=st["color"], s=st["size"],
                    edgecolors="white", linewidths=1.5, zorder=10,
                    marker=st["marker"])
        ax3.text(cx, cy, cz + z_range * 0.03, lab,
                 fontsize=8.5, fontweight="medium", color=st["color"],
                 ha="center")

    ax3.set_xlabel("PC 1", fontsize=11, labelpad=8)
    ax3.set_ylabel("PC 2", fontsize=11, labelpad=8)
    ax3.set_zlabel("log(1 + \u03a3 losses)", fontsize=11, labelpad=8)
    ax3.set_title("3D Combined Loss Landscape",
                  fontsize=17, fontweight="bold", pad=18)
    ax3.view_init(elev=28, azim=-55)
    ax3.tick_params(labelsize=8)
    ax3.xaxis.pane.fill = False
    ax3.yaxis.pane.fill = False
    ax3.zaxis.pane.fill = False
    ax3.xaxis.pane.set_edgecolor(BORDER)
    ax3.yaxis.pane.set_edgecolor(BORDER)
    ax3.zaxis.pane.set_edgecolor(BORDER)

    _save(fig, fig_dir, f"loss_landscape_3d_{args.tag}")
    plt.close(fig)

    # -- Print minima analysis ------------------------------------------
    print("\n" + "=" * 60)
    print("Loss Landscape Analysis")
    print("=" * 60)

    for gn in game_names:
        Z = loss_grids[gn]
        mi = np.unravel_index(Z.argmin(), Z.shape)
        ex, ey = expert_coords[gn]
        ix = np.argmin(np.abs(alphas - ex))
        iy = np.argmin(np.abs(betas - ey))
        print(f"\n{gn}:")
        print(f"  Grid min at PC1={A[mi]:.2f}, PC2={B[mi]:.2f}, loss={Z[mi]:.4f}")
        print(f"  Expert at PC1={ex:.2f}, PC2={ey:.2f}, loss={Z[iy, ix]:.4f}")

    print(f"\nCombined (summed) landscape:")
    mi = np.unravel_index(summed.argmin(), summed.shape)
    print(f"  Grid min at PC1={A[mi]:.2f}, PC2={B[mi]:.2f}, "
          f"summed loss={summed[mi]:.4f}")
    for lab, (cx, cy) in consol_coords.items():
        ix = np.argmin(np.abs(alphas - cx))
        iy = np.argmin(np.abs(betas - cy))
        print(f"  {lab}: summed loss={summed[iy, ix]:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
