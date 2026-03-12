"""
Loss landscape visualization for Atari expert models.

Projects the three expert parameter vectors onto a 2D plane via PCA,
then evaluates the DQN Bellman (TD) loss at a grid of interpolated
model weights. Produces:
  1. Per-game loss contour plots with expert positions marked
  2. Combined contour overlay showing all three landscapes
  3. 3D surface plot of the summed loss landscape

Follows the methodology of Li et al. (2018) "Visualizing the Loss
Landscape of Neural Nets" with filter-normalized random directions
replaced by PCA of the actual expert weight vectors.

Usage:
    python scripts/visualize_loss_landscape.py [--config configs/base.yaml]
        [--tag default] [--grid-size 41] [--range 1.0]
        [--eval-samples 512] [--device mps]
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import matplotlib.gridspec as gridspec

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


# ── Modern dark palette ──────────────────────────────────────────────
GAME_COLORS = {
    "Pong":          "#6366F1",   # indigo
    "Breakout":      "#F59E0B",   # amber
    "SpaceInvaders": "#10B981",   # emerald
}
CONSOLIDATED_COLOR = "#F43F5E"    # rose-500

BG        = "#0F172A"
BG_PANEL  = "#1E293B"
TEXT      = "#F8FAFC"
TEXT_DIM  = "#94A3B8"
GRID_CLR  = "#334155"


# Per-game contour colormaps (light to saturated)
GAME_CMAPS = {
    "Pong": LinearSegmentedColormap.from_list(
        "indigo", ["#0F172A", "#312E81", "#4338CA", "#6366F1", "#A5B4FC", "#E0E7FF"]),
    "Breakout": LinearSegmentedColormap.from_list(
        "amber", ["#0F172A", "#78350F", "#B45309", "#F59E0B", "#FCD34D", "#FFFBEB"]),
    "SpaceInvaders": LinearSegmentedColormap.from_list(
        "emerald", ["#0F172A", "#064E3B", "#047857", "#10B981", "#6EE7B7", "#D1FAE5"]),
}
COMBINED_CMAP = LinearSegmentedColormap.from_list(
    "slate_fire", ["#0F172A", "#1E3A5F", "#475569", "#94A3B8", "#F59E0B", "#EF4444"],
)


def _apply_style() -> None:
    """Dark modern matplotlib style."""
    plt.rcParams.update({
        "figure.facecolor":    BG,
        "axes.facecolor":      BG_PANEL,
        "axes.edgecolor":      GRID_CLR,
        "axes.labelcolor":     TEXT,
        "axes.grid":           False,
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


# ── Utilities ────────────────────────────────────────────────────────

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


def flatten_params(state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Flatten a state dict into a single 1-D vector."""
    return torch.cat([v.flatten().float() for v in state_dict.values()])


def unflatten_params(
    flat: torch.Tensor,
    reference_sd: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Reshape a flat vector back into a state dict."""
    sd = {}
    offset = 0
    for name, ref in reference_sd.items():
        numel = ref.numel()
        sd[name] = flat[offset:offset + numel].reshape(ref.shape)
        offset += numel
    return sd


def filter_normalize(direction: torch.Tensor, ref_flat: torch.Tensor,
                     shapes: List[Tuple[str, torch.Size]]) -> torch.Tensor:
    """Apply filter-wise normalization (Li et al. 2018, Sec 4).

    For conv layers, normalize each filter; for FC, normalize each row.
    This prevents directions from being dominated by layers with large norms.
    """
    normed = torch.zeros_like(direction)
    offset = 0
    for name, shape in shapes:
        numel = shape.numel()
        d_slice = direction[offset:offset + numel].reshape(shape)
        w_slice = ref_flat[offset:offset + numel].reshape(shape)

        if len(shape) >= 2:
            # Normalize per filter/row
            for i in range(shape[0]):
                d_filter = d_slice[i]
                w_filter = w_slice[i]
                w_norm = w_filter.norm()
                d_norm = d_filter.norm()
                if d_norm > 1e-10 and w_norm > 1e-10:
                    d_slice[i] = d_filter * (w_norm / d_norm)
        else:
            # 1-D params (bias): normalize globally
            w_norm = w_slice.norm()
            d_norm = d_slice.norm()
            if d_norm > 1e-10 and w_norm > 1e-10:
                d_slice = d_slice * (w_norm / d_norm)

        normed[offset:offset + numel] = d_slice.flatten()
        offset += numel
    return normed


def compute_pca_directions(
    expert_sds: List[Dict[str, torch.Tensor]],
    reference_sd: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]:
    """Compute 2 PCA directions from the expert weight vectors.

    Centers at the mean of the experts, then runs SVD on the matrix
    of centered parameter vectors.

    Returns:
        (dir1, dir2, center, [proj_coords for each expert])
    """
    flat_vecs = [flatten_params(sd) for sd in expert_sds]
    mat = torch.stack(flat_vecs, dim=0).float()  # (N, D)
    center = mat.mean(dim=0)

    centered = mat - center.unsqueeze(0)  # (N, D)

    # SVD on the centered matrix
    U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
    # Vh[0] and Vh[1] are the two principal directions
    dir1 = Vh[0]  # (D,)
    dir2 = Vh[1] if Vh.shape[0] > 1 else torch.randn_like(dir1)

    # Normalize directions to unit norm
    dir1 = dir1 / dir1.norm()
    dir2 = dir2 / dir2.norm()

    # Project each expert onto the 2D plane
    proj_coords = []
    for v in flat_vecs:
        diff = v - center
        x = torch.dot(diff, dir1).item()
        y = torch.dot(diff, dir2).item()
        proj_coords.append((x, y))

    return dir1, dir2, center, proj_coords


def collect_eval_data(
    env_id: str,
    model: DQNNetwork,
    config: dict,
    device: str,
    num_samples: int,
    union_actions: list,
) -> ReplayBuffer:
    """Collect transitions by running the expert policy."""
    env_cfg = config["env"]
    env = make_atari_env(
        env_id=env_id, union_actions=union_actions, seed=config["seed"],
        frame_stack=env_cfg["frame_stack"], frame_skip=env_cfg["frame_skip"],
        screen_size=env_cfg["screen_size"], noop_max=env_cfg["noop_max"],
        episodic_life=env_cfg["episodic_life"], clip_reward=env_cfg["clip_reward"],
    )

    valid_actions = get_valid_actions(env_id, union_actions)
    buf = ReplayBuffer(
        capacity=num_samples, frame_stack=env_cfg["frame_stack"],
        frame_shape=(env_cfg["screen_size"], env_cfg["screen_size"]),
        device=device,
    )
    model.eval()
    state, _ = env.reset()

    while buf.size < num_samples:
        with torch.no_grad():
            state_t = torch.from_numpy(state).float().unsqueeze(0).to(device) / 255.0
            q_vals = model(state_t)
            mask = torch.full((model.unified_action_dim,), float("-inf"), device=device)
            mask[valid_actions] = 0.0
            action = (q_vals + mask.unsqueeze(0)).argmax(dim=1).item()

        next_state, reward, terminated, truncated, _ = env.step(action)
        buf.push(state, action, reward, next_state, terminated)

        if terminated or truncated:
            state, _ = env.reset()
        else:
            state = next_state

    env.close()
    return buf


def evaluate_bellman_loss(
    model: DQNNetwork,
    state_dict: Dict[str, torch.Tensor],
    replay_buffer: ReplayBuffer,
    device: str,
    num_samples: int = 512,
    gamma: float = 0.99,
) -> float:
    """Evaluate the Huber (smooth-L1) Bellman loss at given weights.

    Uses the same weights as both policy and target network (no separate
    target net), which means minima correspond to fixed points of the
    Bellman operator.
    """
    model.load_state_dict(state_dict)
    model.eval()

    states, actions, rewards, next_states, dones = replay_buffer.sample(
        min(num_samples, replay_buffer.size),
    )

    with torch.no_grad():
        # Current Q
        q_all = model(states)
        q_taken = q_all.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Bootstrap from same network (simplified; no target net)
        q_next = model(next_states)
        max_q_next = q_next.max(dim=1).values
        target = rewards + gamma * max_q_next * (1.0 - dones)

        loss = F.smooth_l1_loss(q_taken, target)

    return loss.item()


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


# ── Main ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Loss landscape visualization for Atari expert models.",
    )
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--tag", type=str, default="default")
    parser.add_argument(
        "--grid-size", type=int, default=41,
        help="Number of grid points per axis (41 -> 1681 evaluations per game).",
    )
    parser.add_argument(
        "--range", type=float, default=1.0,
        help="Multiplier on the coordinate range. 1.0 means the grid extends "
             "to ~1x the max expert projection distance from center.",
    )
    parser.add_argument(
        "--eval-samples", type=int, default=512,
        help="Number of replay transitions for loss evaluation at each grid point.",
    )
    parser.add_argument(
        "--collect-samples", type=int, default=2000,
        help="Transitions to collect per game for evaluation.",
    )
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
    print(f"Union actions: {union_actions} ({len(union_actions)} actions)")
    print(f"Device: {device}")
    print(f"Grid: {args.grid_size}x{args.grid_size} = "
          f"{args.grid_size**2} points per game\n")

    checkpoint_dir = config["logging"]["checkpoint_dir"]
    figure_dir = config["logging"]["figure_dir"]

    summary_path = os.path.join(checkpoint_dir, args.tag, "expert_summary.json")
    with open(summary_path) as f:
        summary = json.load(f)

    # ── Load expert models and collect evaluation data ────────────────
    expert_sds: List[Dict[str, torch.Tensor]] = []
    game_names: List[str] = []
    game_buffers: Dict[str, ReplayBuffer] = {}
    reference_model = build_model(config, device)

    for expert_info in summary["expert_results"]:
        game_name = expert_info["game_name"]
        env_id = expert_info["env_id"]
        game_names.append(game_name)

        ckpt_path = os.path.join(
            checkpoint_dir, args.tag, f"expert_{game_name}_best.pt",
        )
        print(f"Loading expert: {game_name}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        sd = ckpt["policy_net"]
        expert_sds.append(sd)

        # Collect evaluation replay data
        model = build_model(config, device)
        model.load_state_dict(sd)
        print(f"  Collecting {args.collect_samples} transitions...")
        buf = collect_eval_data(
            env_id, model, config, device,
            args.collect_samples, union_actions,
        )
        game_buffers[game_name] = buf
        print(f"  Buffer size: {buf.size}")

    # ── Also load consolidated models if available ────────────────────
    consolidated_models: Dict[str, Dict[str, torch.Tensor]] = {}
    for cname in ["consolidated_distillation", "consolidated_htcl_lam0.1",
                   "consolidated_htcl_lam1.0", "consolidated_htcl_lam10.0"]:
        cpath = os.path.join(checkpoint_dir, args.tag, f"{cname}.pt")
        if os.path.exists(cpath):
            csd = torch.load(cpath, map_location=device, weights_only=False)
            # Some checkpoints are raw state dicts, others are wrapped
            if isinstance(csd, dict) and "policy_net" in csd:
                csd = csd["policy_net"]
            consolidated_models[cname] = csd
            print(f"  Loaded consolidated: {cname}")

    # ── Compute PCA directions ────────────────────────────────────────
    print("\nComputing PCA directions from expert weight vectors...")
    dir1, dir2, center, expert_coords = compute_pca_directions(
        expert_sds, expert_sds[0],
    )

    # Apply filter normalization
    ref_sd = expert_sds[0]
    shapes = [(name, param.shape) for name, param in ref_sd.items()]
    dir1 = filter_normalize(dir1, center, shapes)
    dir2 = filter_normalize(dir2, center, shapes)

    # Re-project after normalization
    expert_coords = []
    for sd in expert_sds:
        diff = flatten_params(sd) - center
        x = torch.dot(diff, dir1).item()
        y = torch.dot(diff, dir2).item()
        expert_coords.append((x, y))

    # Project consolidated models
    consolidated_coords: Dict[str, Tuple[float, float]] = {}
    for cname, csd in consolidated_models.items():
        diff = flatten_params(csd) - center
        x = torch.dot(diff, dir1).item()
        y = torch.dot(diff, dir2).item()
        consolidated_coords[cname] = (x, y)

    print("Expert projections:")
    for gn, (x, y) in zip(game_names, expert_coords):
        print(f"  {gn}: ({x:.4f}, {y:.4f})")
    for cname, (x, y) in consolidated_coords.items():
        print(f"  {cname}: ({x:.4f}, {y:.4f})")

    # ── Build evaluation grid ─────────────────────────────────────────
    all_coords = expert_coords + list(consolidated_coords.values())
    max_extent = max(
        max(abs(c[0]) for c in all_coords),
        max(abs(c[1]) for c in all_coords),
    ) * 1.5 * args.range

    alphas = np.linspace(-max_extent, max_extent, args.grid_size)
    betas = np.linspace(-max_extent, max_extent, args.grid_size)
    A, B = np.meshgrid(alphas, betas)

    # ── Evaluate losses on the grid ──────────────────────────────────
    # loss_grids[game_name] = 2D array of loss values
    loss_grids: Dict[str, np.ndarray] = {g: np.zeros_like(A) for g in game_names}

    total_points = args.grid_size * args.grid_size
    eval_model = build_model(config, device)

    print(f"\nEvaluating {total_points} grid points x {len(game_names)} games...")
    for i in range(args.grid_size):
        if (i + 1) % 5 == 0 or i == 0:
            print(f"  Row {i + 1}/{args.grid_size}...")

        for j in range(args.grid_size):
            alpha = A[i, j]
            beta = B[i, j]

            # Construct model weights at this grid point
            flat_weights = center + alpha * dir1 + beta * dir2
            grid_sd = unflatten_params(flat_weights, ref_sd)
            # Move to device
            grid_sd = {k: v.to(device) for k, v in grid_sd.items()}

            # Evaluate on each game
            for game_name in game_names:
                loss = evaluate_bellman_loss(
                    eval_model, grid_sd, game_buffers[game_name],
                    device, args.eval_samples, args.gamma,
                )
                loss_grids[game_name][i, j] = loss

    # ── Compute summed loss landscape ─────────────────────────────────
    summed_loss = sum(loss_grids[g] for g in game_names)

    # ── Log-scale for better visualization ────────────────────────────
    # Add small epsilon then take log for smoother contours
    log_grids = {}
    for g in game_names:
        log_grids[g] = np.log1p(loss_grids[g])
    log_summed = np.log1p(summed_loss)

    # ── Figure 1: Per-game contour plots (1x3) ───────────────────────
    print("\nGenerating per-game contour plots...")
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5))

    for idx, (game_name, ax) in enumerate(zip(game_names, axes)):
        Z = log_grids[game_name]
        cmap = GAME_CMAPS.get(game_name, COMBINED_CMAP)

        cf = ax.contourf(A, B, Z, levels=30, cmap=cmap, alpha=0.9)
        ax.contour(A, B, Z, levels=15, colors=["#FFFFFF"], linewidths=0.3, alpha=0.25)

        # Mark all expert positions
        for k, (gn, (ex, ey)) in enumerate(zip(game_names, expert_coords)):
            is_this = (gn == game_name)
            ax.scatter(
                ex, ey,
                c=GAME_COLORS[gn],
                s=120 if is_this else 60,
                edgecolors="white",
                linewidths=2 if is_this else 1,
                zorder=10,
                marker="*" if is_this else "o",
                label=f"{gn} expert" if idx == 0 else None,
            )

        # Mark consolidated models
        for cname, (cx, cy) in consolidated_coords.items():
            short_name = cname.replace("consolidated_", "").replace("_", " ")
            ax.scatter(
                cx, cy, c=CONSOLIDATED_COLOR, s=70,
                edgecolors="white", linewidths=1.2,
                zorder=10, marker="D",
                label=short_name if idx == 0 else None,
            )

        ax.set_title(
            f"{game_name} Loss",
            fontsize=14, fontweight="bold",
            color=GAME_COLORS[game_name], pad=10,
        )
        ax.set_xlabel("PC 1", fontsize=11)
        ax.set_ylabel("PC 2" if idx == 0 else "", fontsize=11)
        ax.tick_params(length=0)

        # Colorbar
        cbar = fig.colorbar(cf, ax=ax, fraction=0.04, pad=0.03)
        cbar.set_label("log(1 + loss)", fontsize=9, color=TEXT_DIM)
        cbar.ax.yaxis.set_tick_params(color=TEXT_DIM)
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT_DIM, fontsize=8)

    # Shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=len(handles),
        fontsize=10, framealpha=0.85, borderpad=0.6,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.suptitle(
        "Per-Game Loss Landscape (PCA Projection)",
        fontsize=18, fontweight="bold", color=TEXT, y=1.02,
    )
    fig.tight_layout()
    _save(fig, figure_dir, f"loss_landscape_per_game_{args.tag}")
    plt.close(fig)

    # ── Figure 2: Combined contour with expert markers ────────────────
    print("Generating combined contour plot...")
    fig, ax = plt.subplots(figsize=(10, 8.5))

    cf = ax.contourf(A, B, log_summed, levels=40, cmap=COMBINED_CMAP, alpha=0.9)
    ax.contour(A, B, log_summed, levels=20, colors=["#FFFFFF"], linewidths=0.3, alpha=0.2)

    for gn, (ex, ey) in zip(game_names, expert_coords):
        ax.scatter(
            ex, ey, c=GAME_COLORS[gn], s=180,
            edgecolors="white", linewidths=2.5, zorder=10,
            marker="*", label=f"{gn}",
        )
        ax.annotate(
            gn, (ex, ey), textcoords="offset points",
            xytext=(10, 10), fontsize=11, fontweight="bold",
            color=GAME_COLORS[gn],
            bbox=dict(boxstyle="round,pad=0.2", fc=BG, ec="none", alpha=0.7),
        )

    for cname, (cx, cy) in consolidated_coords.items():
        short = cname.replace("consolidated_", "")
        ax.scatter(
            cx, cy, c=CONSOLIDATED_COLOR, s=100,
            edgecolors="white", linewidths=1.5, zorder=10, marker="D",
            label=short,
        )
        ax.annotate(
            short, (cx, cy), textcoords="offset points",
            xytext=(10, -12), fontsize=9, color=CONSOLIDATED_COLOR,
            bbox=dict(boxstyle="round,pad=0.2", fc=BG, ec="none", alpha=0.7),
        )

    ax.set_xlabel("PC 1", fontsize=13)
    ax.set_ylabel("PC 2", fontsize=13)
    ax.set_title(
        "Combined Loss Landscape",
        fontsize=18, fontweight="bold", pad=14,
    )
    ax.legend(fontsize=11, framealpha=0.85, loc="upper right")
    ax.tick_params(length=0)

    cbar = fig.colorbar(cf, ax=ax, fraction=0.03, pad=0.03)
    cbar.set_label("log(1 + Σ losses)", fontsize=11, color=TEXT_DIM)
    cbar.ax.yaxis.set_tick_params(color=TEXT_DIM)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT_DIM, fontsize=9)

    fig.tight_layout()
    _save(fig, figure_dir, f"loss_landscape_combined_{args.tag}")
    plt.close(fig)

    # ── Figure 3: 3D surface plot ─────────────────────────────────────
    print("Generating 3D surface plot...")
    fig = plt.figure(figsize=(12, 9))
    ax3d = fig.add_subplot(111, projection="3d")
    ax3d.set_facecolor(BG_PANEL)

    surf = ax3d.plot_surface(
        A, B, log_summed,
        cmap=COMBINED_CMAP, alpha=0.85,
        edgecolor="none", antialiased=True,
        rcount=args.grid_size, ccount=args.grid_size,
    )

    # Plot expert positions on the surface
    for gn, (ex, ey) in zip(game_names, expert_coords):
        # Find closest grid point for z-value
        ix = np.argmin(np.abs(alphas - ex))
        iy = np.argmin(np.abs(betas - ey))
        ez = log_summed[iy, ix]
        ax3d.scatter(
            [ex], [ey], [ez], c=GAME_COLORS[gn], s=150,
            edgecolors="white", linewidths=2, zorder=10, marker="*",
        )
        ax3d.text(
            ex, ey, ez + (log_summed.max() - log_summed.min()) * 0.05,
            gn, fontsize=10, fontweight="bold", color=GAME_COLORS[gn],
            ha="center",
        )

    # Consolidated models on the surface
    for cname, (cx, cy) in consolidated_coords.items():
        ix = np.argmin(np.abs(alphas - cx))
        iy = np.argmin(np.abs(betas - cy))
        cz = log_summed[iy, ix]
        short = cname.replace("consolidated_", "")
        ax3d.scatter(
            [cx], [cy], [cz], c=CONSOLIDATED_COLOR, s=80,
            edgecolors="white", linewidths=1.5, zorder=10, marker="D",
        )

    ax3d.set_xlabel("PC 1", fontsize=11, labelpad=8)
    ax3d.set_ylabel("PC 2", fontsize=11, labelpad=8)
    ax3d.set_zlabel("log(1 + Σ losses)", fontsize=11, labelpad=8)
    ax3d.set_title(
        "3D Combined Loss Landscape",
        fontsize=17, fontweight="bold", pad=20,
    )
    ax3d.view_init(elev=30, azim=-50)
    ax3d.tick_params(labelsize=8)
    ax3d.xaxis.pane.fill = False
    ax3d.yaxis.pane.fill = False
    ax3d.zaxis.pane.fill = False
    ax3d.xaxis.pane.set_edgecolor(GRID_CLR)
    ax3d.yaxis.pane.set_edgecolor(GRID_CLR)
    ax3d.zaxis.pane.set_edgecolor(GRID_CLR)

    _save(fig, figure_dir, f"loss_landscape_3d_{args.tag}")
    plt.close(fig)

    # ── Figure 4: 2x2 dashboard ──────────────────────────────────────
    print("Generating dashboard...")
    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.30, wspace=0.25,
                           left=0.06, right=0.96, top=0.93, bottom=0.06)

    # Panel A, B, C: per-game contours
    panel_labels = ["A", "B", "C"]
    for idx, game_name in enumerate(game_names):
        if idx < 2:
            ax = fig.add_subplot(gs[0, idx])
        else:
            ax = fig.add_subplot(gs[1, 0])

        Z = log_grids[game_name]
        cmap = GAME_CMAPS.get(game_name, COMBINED_CMAP)
        cf = ax.contourf(A, B, Z, levels=25, cmap=cmap, alpha=0.9)
        ax.contour(A, B, Z, levels=12, colors=["#FFFFFF"], linewidths=0.3, alpha=0.2)

        for gn, (ex, ey) in zip(game_names, expert_coords):
            is_this = gn == game_name
            ax.scatter(
                ex, ey, c=GAME_COLORS[gn],
                s=100 if is_this else 40,
                edgecolors="white", linewidths=1.5 if is_this else 0.8,
                zorder=10, marker="*" if is_this else "o",
            )

        for cname, (cx, cy) in consolidated_coords.items():
            ax.scatter(cx, cy, c=CONSOLIDATED_COLOR, s=50,
                       edgecolors="white", linewidths=1, zorder=10, marker="D")

        ax.set_title(
            f"{panel_labels[idx]}   {game_name}",
            fontsize=13, fontweight="bold",
            color=GAME_COLORS[game_name], loc="left", pad=8,
        )
        ax.set_xlabel("PC 1", fontsize=10)
        ax.set_ylabel("PC 2", fontsize=10)
        ax.tick_params(length=0)

    # Panel D: combined
    ax_comb = fig.add_subplot(gs[1, 1])
    cf = ax_comb.contourf(A, B, log_summed, levels=30, cmap=COMBINED_CMAP, alpha=0.9)
    ax_comb.contour(A, B, log_summed, levels=15, colors=["#FFFFFF"],
                     linewidths=0.3, alpha=0.2)

    for gn, (ex, ey) in zip(game_names, expert_coords):
        ax_comb.scatter(ex, ey, c=GAME_COLORS[gn], s=100,
                        edgecolors="white", linewidths=1.5, zorder=10, marker="*")
    for cname, (cx, cy) in consolidated_coords.items():
        ax_comb.scatter(cx, cy, c=CONSOLIDATED_COLOR, s=60,
                        edgecolors="white", linewidths=1, zorder=10, marker="D")

    ax_comb.set_title("D   Combined Loss", fontsize=13, fontweight="bold",
                       loc="left", pad=8)
    ax_comb.set_xlabel("PC 1", fontsize=10)
    ax_comb.set_ylabel("PC 2", fontsize=10)
    ax_comb.tick_params(length=0)

    fig.suptitle(
        "Loss Landscape Dashboard",
        fontsize=20, fontweight="bold", color=TEXT, y=0.97,
    )
    _save(fig, figure_dir, f"loss_landscape_dashboard_{args.tag}")
    plt.close(fig)

    # ── Print minima info ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Grid Minima Analysis")
    print("=" * 60)
    for game_name in game_names:
        Z = loss_grids[game_name]
        min_idx = np.unravel_index(Z.argmin(), Z.shape)
        min_alpha = A[min_idx]
        min_beta = B[min_idx]
        min_loss = Z[min_idx]
        print(f"\n{game_name}:")
        print(f"  Grid minimum at PC1={min_alpha:.4f}, PC2={min_beta:.4f}")
        print(f"  Loss at minimum: {min_loss:.4f}")
        ex, ey = expert_coords[game_names.index(game_name)]
        print(f"  Expert position: PC1={ex:.4f}, PC2={ey:.4f}")
        # Loss at expert position
        ix = np.argmin(np.abs(alphas - ex))
        iy = np.argmin(np.abs(betas - ey))
        print(f"  Loss at expert: {Z[iy, ix]:.4f}")

    print(f"\nCombined landscape:")
    min_idx = np.unravel_index(summed_loss.argmin(), summed_loss.shape)
    print(f"  Grid minimum at PC1={A[min_idx]:.4f}, PC2={B[min_idx]:.4f}")
    print(f"  Summed loss at minimum: {summed_loss[min_idx]:.4f}")

    for cname, (cx, cy) in consolidated_coords.items():
        ix = np.argmin(np.abs(alphas - cx))
        iy = np.argmin(np.abs(betas - cy))
        print(f"  {cname}: summed loss = {summed_loss[iy, ix]:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
