"""
Consolidate expert models using one of four methods.

Methods:
    distillation  -- Knowledge Distillation (soft Q-value matching)
    oneshot       -- One-Shot Joint Consolidation (single Taylor step over all tasks)
    iterative     -- Multi-Round Iterative Consolidation (sequential Taylor + multi-pass)
    hybrid        -- Hybrid Consolidation (iterative HTCL then KD refinement)

Loads trained expert checkpoints and merges them into a single global model.

Usage:
    python scripts/consolidate.py --method distillation [--debug]
    python scripts/consolidate.py --method oneshot [--debug]
    python scripts/consolidate.py --method iterative [--debug]
    python scripts/consolidate.py --method hybrid [--debug]
"""

import argparse
import os
import sys
import json
import time as _time

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.dqn import DQNNetwork
from src.data.replay_buffer import ReplayBuffer
from src.data.atari_wrappers import (
    get_valid_actions,
    make_atari_env,
    compute_union_action_space,
)
from src.consolidation.distillation import DistillationConsolidator
from src.consolidation.oneshot import OneShotConsolidator
from src.consolidation.iterative import IterativeConsolidator
from src.consolidation.hybrid import HybridConsolidator
from src.consolidation.whc import WHCConsolidator
from src.utils.config import get_effective_config, save_config
from src.utils.seed import set_seed
from src.utils.logger import setup_logger

# -- Supported methods ---------------------------------------------------------

ALL_METHODS = ["distillation", "oneshot", "iterative", "hybrid", "whc"]

# Methods that need Fisher/gradient infrastructure (high-confidence states,
# frozen expert models, etc.)
TAYLOR_METHODS = {"oneshot", "iterative", "hybrid", "whc"}


# -- Helpers -------------------------------------------------------------------

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


def collect_replay_data(
    env_id: str,
    model: DQNNetwork,
    config: dict,
    device: str,
    num_samples: int,
    union_actions: list,
) -> ReplayBuffer:
    """Collect replay data by running the expert in the environment.

    Args:
        env_id: Environment ID.
        model: Expert model.
        config: Configuration dictionary.
        device: Device string.
        num_samples: Number of transitions to collect.
        union_actions: Union action list.

    Returns:
        Filled replay buffer.
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
    buffer = ReplayBuffer(
        capacity=num_samples,
        frame_stack=env_cfg["frame_stack"],
        frame_shape=(env_cfg["screen_size"], env_cfg["screen_size"]),
        device=device,
    )

    model.eval()
    state, _ = env.reset()
    collected = 0

    while collected < num_samples:
        with torch.no_grad():
            state_tensor = (
                torch.from_numpy(state).float().unsqueeze(0).to(device) / 255.0
            )
            q_values = model(state_tensor)
            mask = torch.full(
                (model.unified_action_dim,), float("-inf"), device=device,
            )
            mask[valid_actions] = 0.0
            masked_q = q_values + mask.unsqueeze(0)
            action = masked_q.argmax(dim=1).item()

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        buffer.push(state, action, reward, next_state, done)
        collected += 1

        if done:
            state, _ = env.reset()
        else:
            state = next_state

    env.close()
    return buffer


def load_expert_results(
    config: dict,
    tag: str,
    device: str,
    union_actions: list,
) -> list:
    """Load expert checkpoints and rebuild results structure.

    Args:
        config: Configuration dictionary.
        tag: Experiment tag.
        device: Device string.
        union_actions: Union action list.

    Returns:
        List of expert result dictionaries.
    """
    checkpoint_dir = config["logging"]["checkpoint_dir"]
    summary_path = os.path.join(checkpoint_dir, tag, "expert_summary.json")

    with open(summary_path, "r") as f:
        summary = json.load(f)

    results = []
    for expert_info in summary["expert_results"]:
        game_name = expert_info["game_name"]
        env_id = expert_info["env_id"]
        valid_actions = expert_info["valid_actions"]

        ckpt_path = os.path.join(
            checkpoint_dir, tag, f"expert_{game_name}_best.pt",
        )
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"No best checkpoint found for {game_name} at {ckpt_path}"
            )

        checkpoint = torch.load(
            ckpt_path, map_location=device, weights_only=False,
        )
        policy_sd = checkpoint["policy_net"]

        # Build model and collect replay data for Fisher / distillation
        model = build_model(config, device)
        model.load_state_dict(policy_sd)

        debug_enabled = config.get("debug", {}).get("enabled", False)
        buffer_size = (
            config["debug"].get("buffer_size_per_task", 2000)
            if debug_enabled
            else config.get("buffer_size_per_task", 50_000)
        )

        print(f"Collecting {buffer_size} replay samples for {game_name}...")
        replay_buffer = collect_replay_data(
            env_id, model, config, device, buffer_size, union_actions,
        )

        results.append(
            {
                "game_name": game_name,
                "env_id": env_id,
                "valid_actions": valid_actions,
                "policy_state_dict": policy_sd,
                "replay_buffer": replay_buffer,
                "final_reward": expert_info["final_reward"],
                "best_reward": expert_info["best_reward"],
            }
        )

    return results


def filter_replay_states(
    config: dict,
    expert_results: list,
    device: str,
    logger,
) -> tuple:
    """Build frozen expert models and filter replay buffers.

    Returns:
        (filtered_states_list, expert_models)
    """
    debug_enabled = config.get("debug", {}).get("enabled", False)

    # Pick the right filtered_buffer_size from htcl config (shared)
    htcl_cfg = config.get("htcl", {})
    filtered_size = (
        config["debug"].get("filtered_buffer_size", 500)
        if debug_enabled
        else htcl_cfg.get("filtered_buffer_size", 10_000)
    )

    logger.info(
        f"Filtering replay buffers to top-{filtered_size} "
        f"high-confidence states per expert..."
    )

    filtered_states_list = []
    expert_models = []

    for r in expert_results:
        expert_model = build_model(config, device)
        expert_model.load_state_dict(r["policy_state_dict"])
        expert_model.eval()
        for p in expert_model.parameters():
            p.requires_grad_(False)
        expert_models.append(expert_model)

        filt_states = r["replay_buffer"].filter_by_confidence(
            expert_model, r["valid_actions"], top_k=filtered_size,
        )
        filtered_states_list.append(filt_states)
        logger.info(
            f"  {r['game_name']}: {filt_states.shape[0]} states filtered "
            f"from {r['replay_buffer'].size} raw transitions"
        )

    return filtered_states_list, expert_models


# -- Per-method consolidation runners -----------------------------------------

def run_distillation(
    config, expert_results, device, logger, tag,
    snapshot_epochs=None,
):
    """Run Knowledge Distillation consolidation."""
    logger.info("Initializing global model as parameter average of experts...")
    global_model = build_model(config, device)
    first_sd = expert_results[0]["policy_state_dict"]
    init_sd = {}
    for key in first_sd:
        init_sd[key] = torch.stack(
            [r["policy_state_dict"][key].float().to(device) for r in expert_results]
        ).mean(dim=0)
    global_model.load_state_dict(init_sd)

    snap_dir = os.path.join(config["logging"]["checkpoint_dir"], tag)
    consolidator = DistillationConsolidator(config, device=device, logger=logger)
    consolidated = consolidator.consolidate(
        global_model, expert_results,
        snapshot_epochs=snapshot_epochs,
        snapshot_dir=snap_dir,
        snapshot_prefix="consolidated_distillation",
    )

    save_path = os.path.join(snap_dir, "consolidated_distillation.pt")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(consolidated.state_dict(), save_path)
    logger.info(f"Distillation model saved to {save_path}")
    return consolidated


def _ensemble_mean_sd(
    expert_results: list, device: str,
) -> dict:
    """Compute ensemble-mean state dict  w_bar = (1/N) Sigma_i w_e^(i).

    Paper Definition 3.1 / Algorithm 1 line 1 / Algorithm 2 line 2.
    """
    sds = [r["policy_state_dict"] for r in expert_results]
    keys = sds[0].keys()
    return {
        k: torch.stack([sd[k].float().to(device) for sd in sds]).mean(dim=0)
        for k in keys
    }


def run_oneshot(
    config, expert_results, filtered_states_list, expert_models,
    device, logger, tag,
):
    """Run One-Shot Joint Consolidation."""
    global_model = build_model(config, device)
    init_sd = _ensemble_mean_sd(expert_results, device)
    global_model.load_state_dict(init_sd)
    logger.info(
        "Initialising global model at ensemble-mean anchor "
        "(Definition 3.1)..."
    )

    consolidator = OneShotConsolidator(config, device=device, logger=logger)
    consolidated = consolidator.consolidate(
        global_model, expert_results,
        filtered_states_list=filtered_states_list,
        expert_models=expert_models,
    )

    save_path = os.path.join(
        config["logging"]["checkpoint_dir"], tag, "consolidated_oneshot.pt",
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(consolidated.state_dict(), save_path)
    logger.info(f"One-Shot model saved to {save_path}")
    return consolidated


def run_iterative(
    config, expert_results, filtered_states_list, expert_models,
    device, logger, tag,
):
    """Run Multi-Round Iterative Consolidation."""
    global_model = build_model(config, device)
    init_sd = _ensemble_mean_sd(expert_results, device)
    global_model.load_state_dict(init_sd)
    logger.info(
        "Initialising global model at ensemble-mean anchor "
        "(Algorithm 1, line 1)..."
    )

    consolidator = IterativeConsolidator(config, device=device, logger=logger)
    consolidated = consolidator.consolidate(
        global_model, expert_results,
        filtered_states_list=filtered_states_list,
        expert_models=expert_models,
    )

    save_path = os.path.join(
        config["logging"]["checkpoint_dir"], tag, "consolidated_iterative.pt",
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(consolidated.state_dict(), save_path)
    logger.info(f"Iterative model saved to {save_path}")
    return consolidated


def run_hybrid(
    config, expert_results, filtered_states_list, expert_models,
    device, logger, tag, snapshot_epochs=None,
):
    """Run Hybrid Consolidation (HTCL + KD)."""
    global_model = build_model(config, device)
    init_sd = _ensemble_mean_sd(expert_results, device)
    global_model.load_state_dict(init_sd)
    logger.info(
        "Initialising global model at ensemble-mean anchor "
        "(Algorithm 2, line 2)..."
    )

    # Apply debug override for hybrid KD epochs
    debug_enabled = config.get("debug", {}).get("enabled", False)
    if debug_enabled:
        hybrid_cfg = config.setdefault("hybrid", {})
        hybrid_cfg["kd_epochs"] = config["debug"].get("hybrid_kd_epochs", 3)

    snap_dir = os.path.join(config["logging"]["checkpoint_dir"], tag)
    consolidator = HybridConsolidator(config, device=device, logger=logger)
    consolidated = consolidator.consolidate(
        global_model, expert_results,
        filtered_states_list=filtered_states_list,
        expert_models=expert_models,
        snapshot_epochs=snapshot_epochs,
        snapshot_dir=snap_dir,
        snapshot_prefix="consolidated_hybrid",
    )

    save_path = os.path.join(snap_dir, "consolidated_hybrid.pt")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(consolidated.state_dict(), save_path)
    logger.info(f"Hybrid model saved to {save_path}")
    return consolidated


def run_whc(
    config, expert_results, filtered_states_list, expert_models,
    device, logger, tag,
):
    """Run Weighted Hessian Consolidation."""
    global_model = build_model(config, device)
    logger.info(
        "Initialising WHC (Fisher at each expert's own optimum)..."
    )

    consolidator = WHCConsolidator(config, device=device, logger=logger)
    consolidated = consolidator.consolidate(
        global_model, expert_results,
        filtered_states_list=filtered_states_list,
        expert_models=expert_models,
    )

    save_path = os.path.join(
        config["logging"]["checkpoint_dir"], tag, "consolidated_whc.pt",
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(consolidated.state_dict(), save_path)
    logger.info(f"WHC model saved to {save_path}")
    return consolidated


# -- Main ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Consolidate expert models.")
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=ALL_METHODS,
        help="Consolidation method.",
    )
    parser.add_argument(
        "--config", type=str, default="configs/base.yaml", help="Config file.",
    )
    parser.add_argument(
        "--override-config", type=str, default=None, help="Override config.",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug mode.",
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Device.",
    )
    parser.add_argument(
        "--tag", type=str, default="default", help="Experiment tag.",
    )
    parser.add_argument(
        "--snapshot-epochs",
        type=str,
        default=None,
        help=(
            "Comma-separated epoch milestones for intermediate checkpoints. "
            "Train once for max(milestones) epochs and snapshot at each. "
            "Example: '10,100,500,5000,10000'"
        ),
    )
    args = parser.parse_args()

    config = get_effective_config(
        args.config, args.override_config, debug=args.debug,
    )

    # Parse snapshot epochs and override total epochs to max milestone
    snapshot_epochs = None
    if args.snapshot_epochs:
        snapshot_epochs = sorted(int(x) for x in args.snapshot_epochs.split(","))
        max_ep = max(snapshot_epochs)
        config.setdefault("distillation", {})["distill_epochs"] = max_ep
        config.setdefault("hybrid", {})["kd_epochs"] = max_ep

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    set_seed(config["seed"])

    logger = setup_logger(
        log_dir=config["logging"]["log_dir"],
        experiment_name=f"consolidation_{args.method}_{args.tag}",
        use_tensorboard=config["logging"]["use_tensorboard"],
    )

    logger.info(f"Consolidation method: {args.method}")
    logger.info(f"Device: {device}")
    logger.info(f"Debug: {config['debug']['enabled']}")
    logger.info(f"Tag: {args.tag}")
    logger.info(f"Config: {args.config}")
    logger.info(f"Task sequence: {config['task_sequence']}")

    # Compute union action space
    union_actions = compute_union_action_space(config["task_sequence"])
    config["model"]["unified_action_dim"] = len(union_actions)
    logger.info(
        f"Union action space: {union_actions} ({len(union_actions)} actions)"
    )

    # Log method-specific hyperparameters
    if args.method == "distillation":
        dist_cfg = config["distillation"]
        logger.info(
            f"Distillation config: temperature={dist_cfg['temperature']}, "
            f"alpha={dist_cfg['alpha']}, epochs={dist_cfg['distill_epochs']}, "
            f"lr={dist_cfg['distill_lr']}, "
            f"batch_size={dist_cfg['distill_batch_size']}"
        )
    elif args.method == "oneshot":
        os_cfg = config.get("oneshot", config.get("htcl", {}))
        logger.info(
            f"One-Shot config: lambda={os_cfg.get('lambda_htcl', 100.0)}, "
            f"eta={os_cfg.get('eta', 0.9)}, "
            f"fisher_samples={os_cfg.get('fisher_samples', 5000)}"
        )
    elif args.method == "iterative":
        it_cfg = config.get("iterative", config.get("htcl", {}))
        logger.info(
            f"Iterative config: lambda={it_cfg.get('lambda_htcl', 100.0)}, "
            f"eta_0={it_cfg.get('eta', 0.9)}, "
            f"gamma={it_cfg.get('gamma', 0.5)}, "
            f"num_rounds={it_cfg.get('num_rounds', it_cfg.get('num_passes', 3))}, "
            f"fisher_samples={it_cfg.get('fisher_samples', 5000)}, "
            f"recompute_fisher={it_cfg.get('recompute_fisher', False)}"
        )
    elif args.method == "hybrid":
        hy_cfg = config.get("hybrid", {})
        logger.info(
            f"Hybrid config: lambda={hy_cfg.get('lambda_htcl', 100.0)}, "
            f"K={hy_cfg.get('num_rounds', hy_cfg.get('num_passes', 3))}, "
            f"recompute_fisher={hy_cfg.get('recompute_fisher', False)} | "
            f"KD epochs={hy_cfg.get('kd_epochs', 25)}, "
            f"lr={hy_cfg.get('kd_lr', 2.5e-5)}"
        )
    elif args.method == "whc":
        whc_cfg = config.get("whc", {})
        logger.info(
            f"WHC config: lambda_reg={whc_cfg.get('lambda_reg', 1.0)}, "
            f"fisher_samples={whc_cfg.get('fisher_samples', 20000)}"
        )

    # -- Load expert results --
    _t0 = _time.time()
    logger.info("Loading expert checkpoints and collecting replay data...")
    expert_results = load_expert_results(config, args.tag, device, union_actions)
    logger.info(
        f"Loaded {len(expert_results)} expert models "
        f"in {_time.time() - _t0:.1f}s."
    )
    for r in expert_results:
        logger.info(
            f"  {r['game_name']}: best_reward={r['best_reward']:.2f}, "
            f"valid_actions={r['valid_actions']}, "
            f"buffer_size={r['replay_buffer'].size}"
        )

    # -- Filter replay states (needed for Taylor-based methods) --
    filtered_states_list = None
    expert_models = None

    if args.method in TAYLOR_METHODS:
        filtered_states_list, expert_models = filter_replay_states(
            config, expert_results, device, logger,
        )

    # -- Run consolidation --
    if args.method == "distillation":
        consolidated = run_distillation(
            config, expert_results, device, logger, args.tag,
            snapshot_epochs=snapshot_epochs,
        )
    elif args.method == "oneshot":
        consolidated = run_oneshot(
            config, expert_results, filtered_states_list, expert_models,
            device, logger, args.tag,
        )
    elif args.method == "iterative":
        consolidated = run_iterative(
            config, expert_results, filtered_states_list, expert_models,
            device, logger, args.tag,
        )
    elif args.method == "hybrid":
        consolidated = run_hybrid(
            config, expert_results, filtered_states_list, expert_models,
            device, logger, args.tag, snapshot_epochs=snapshot_epochs,
        )
    elif args.method == "whc":
        consolidated = run_whc(
            config, expert_results, filtered_states_list, expert_models,
            device, logger, args.tag,
        )
    else:
        raise ValueError(f"Unknown method: {args.method}")

    # -- Post-consolidation statistics --
    total_norm = sum(
        p.data.norm().item() ** 2 for p in consolidated.parameters()
    ) ** 0.5
    logger.info(f"Consolidated model param norm: {total_norm:.4f}")

    _total_time = _time.time() - _t0
    logger.info(
        f"Total consolidation time ({args.method}): {_total_time:.1f}s"
    )

    logger.close()


if __name__ == "__main__":
    main()
