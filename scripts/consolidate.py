"""
Consolidate expert models using EWC, Distillation, or HTCL.

Loads trained expert checkpoints and merges them into a single global model.

Usage:
    python scripts/consolidate.py --method ewc [--debug] [--config CONFIG_PATH]
    python scripts/consolidate.py --method distillation [--debug]
    python scripts/consolidate.py --method htcl [--debug]
"""

import argparse
import os
import sys
import json
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.dqn import DQNNetwork
from src.data.replay_buffer import ReplayBuffer
from src.data.atari_wrappers import get_valid_actions, make_atari_env, compute_union_action_space
from src.consolidation.ewc import EWCConsolidator
from src.consolidation.distillation import DistillationConsolidator
from src.consolidation.htcl import HTCLConsolidator
from src.utils.config import get_effective_config, save_config
from src.utils.seed import set_seed
from src.utils.logger import setup_logger

import numpy as np


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
    env_id: str, model: DQNNetwork, config: dict, device: str, num_samples: int,
    union_actions: list,
) -> ReplayBuffer:
    """Collect replay data by running the expert in the environment.

    Args:
        env_id: Environment ID.
        model: Expert model.
        config: Configuration dictionary.
        device: Device string.
        num_samples: Number of transitions to collect.

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
                (model.unified_action_dim,), float("-inf"), device=device
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
    config: dict, tag: str, device: str, union_actions: list,
) -> list:
    """Load expert checkpoints and rebuild results structure.

    Args:
        config: Configuration dictionary.
        tag: Experiment tag.
        device: Device string.

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
            checkpoint_dir, tag, f"expert_{game_name}_best.pt"
        )
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"No best checkpoint found for {game_name} at {ckpt_path}"
            )

        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        policy_sd = checkpoint["policy_net"]

        # Build model and collect replay data for Fisher/distillation
        model = build_model(config, device)
        model.load_state_dict(policy_sd)

        debug_enabled = config.get("debug", {}).get("enabled", False)
        buffer_size = (
            config["debug"].get("distill_buffer_size", 500)
            if debug_enabled
            else config["distillation"].get("buffer_size_per_task", 10_000)
        )

        print(f"Collecting {buffer_size} replay samples for {game_name}...")
        replay_buffer = collect_replay_data(
            env_id, model, config, device, buffer_size, union_actions
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


def main():
    parser = argparse.ArgumentParser(description="Consolidate expert models.")
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["ewc", "distillation", "htcl"],
        help="Consolidation method.",
    )
    parser.add_argument(
        "--config", type=str, default="configs/base.yaml", help="Config file."
    )
    parser.add_argument(
        "--override-config", type=str, default=None, help="Override config."
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug mode."
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Device."
    )
    parser.add_argument(
        "--tag", type=str, default="default", help="Experiment tag."
    )
    args = parser.parse_args()

    config = get_effective_config(args.config, args.override_config, debug=args.debug)

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
    logger.info(f"Union action space: {union_actions} ({len(union_actions)} actions)")

    # Log method-specific hyperparameters
    if args.method == "ewc":
        ewc_cfg = config["ewc"]
        logger.info(f"EWC config: lambda={ewc_cfg['lambda_ewc']}, "
                    f"fisher_samples={ewc_cfg['fisher_samples']}, "
                    f"online={ewc_cfg['online']}, gamma={ewc_cfg['gamma_ewc']}")
    elif args.method == "distillation":
        dist_cfg = config["distillation"]
        logger.info(f"Distillation config: temperature={dist_cfg['temperature']}, "
                    f"alpha={dist_cfg['alpha']}, epochs={dist_cfg['distill_epochs']}, "
                    f"lr={dist_cfg['distill_lr']}, batch_size={dist_cfg['distill_batch_size']}")
    elif args.method == "htcl":
        htcl_cfg = config["htcl"]
        logger.info(f"HTCL config: lambda={htcl_cfg['lambda_htcl']}, "
                    f"lambda_auto={htcl_cfg['lambda_auto']}, "
                    f"fisher_samples={htcl_cfg['fisher_samples']}, "
                    f"catch_up_iters={htcl_cfg['catch_up_iterations']}, "
                    f"eta={htcl_cfg['eta']}")

    # Load expert results
    import time as _time
    _t0 = _time.time()
    logger.info("Loading expert checkpoints and collecting replay data...")
    expert_results = load_expert_results(config, args.tag, device, union_actions)
    logger.info(f"Loaded {len(expert_results)} expert models in {_time.time()-_t0:.1f}s.")
    for r in expert_results:
        logger.info(f"  {r['game_name']}: best_reward={r['best_reward']:.2f}, "
                    f"valid_actions={r['valid_actions']}, "
                    f"buffer_size={r['replay_buffer'].size}")

    # Build global model (initialized randomly or from average)
    global_model = build_model(config, device)

    # Initialize global model as average of experts
    logger.info("Initializing global model as parameter average of experts...")
    avg_sd = {}
    first_sd = expert_results[0]["policy_state_dict"]
    for key in first_sd:
        avg_sd[key] = torch.stack(
            [r["policy_state_dict"][key].float().to(device) for r in expert_results]
        ).mean(dim=0)
    global_model.load_state_dict(avg_sd)

    # Log parameter statistics of the averaged model
    total_norm = sum(p.data.norm().item() ** 2 for p in global_model.parameters()) ** 0.5
    logger.info(f"Global model param norm: {total_norm:.4f}")

    # Run consolidation
    if args.method == "ewc":
        consolidator = EWCConsolidator(config, device=device, logger=logger)
        consolidated_model = consolidator.consolidate(
            global_model, expert_results,
            consolidation_steps=5000 if not config["debug"]["enabled"] else 500,
        )
    elif args.method == "distillation":
        consolidator = DistillationConsolidator(config, device=device, logger=logger)
        consolidated_model = consolidator.consolidate(global_model, expert_results)
    elif args.method == "htcl":
        consolidator = HTCLConsolidator(config, device=device, logger=logger)
        consolidated_model = consolidator.consolidate(global_model, expert_results)
    else:
        raise ValueError(f"Unknown method: {args.method}")

    # Log post-consolidation statistics
    total_norm_after = sum(
        p.data.norm().item() ** 2 for p in consolidated_model.parameters()
    ) ** 0.5
    logger.info(f"Consolidated model param norm: {total_norm_after:.4f} "
                f"(delta: {total_norm_after - total_norm:+.4f})")

    # Compute weight drift from average init
    drift_norms = {}
    for key, param in consolidated_model.named_parameters():
        drift = (param.data - avg_sd[key].to(device)).norm().item()
        drift_norms[key.split('.')[0]] = drift_norms.get(key.split('.')[0], 0) + drift
    for layer, drift in drift_norms.items():
        logger.info(f"  Weight drift [{layer}]: {drift:.6f}")

    # Save consolidated model
    save_path = os.path.join(
        config["logging"]["checkpoint_dir"],
        args.tag,
        f"consolidated_{args.method}.pt",
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(consolidated_model.state_dict(), save_path)
    logger.info(f"Consolidated model saved to {save_path}")
    _total_time = _time.time() - _t0
    logger.info(f"Total consolidation time ({args.method}): {_total_time:.1f}s")

    logger.close()


if __name__ == "__main__":
    main()
