"""
Train expert DQN agents sequentially on Atari tasks.

Each expert is initialized from the current global model state
(to keep local models close to global, per HTCL paper).

Usage:
    python scripts/train_experts.py [--debug] [--config CONFIG_PATH]
"""

import argparse
import os
import sys
import json
import glob
import torch
import subprocess

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.dqn import DQNNetwork
from src.trainers.expert_trainer import ExpertTrainer
from src.data.atari_wrappers import get_valid_actions
from src.utils.config import get_effective_config, save_config
from src.utils.seed import set_seed
from src.utils.logger import setup_logger


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


def main():
    parser = argparse.ArgumentParser(description="Train expert DQN agents.")
    parser.add_argument(
        "--config", type=str, default="configs/base.yaml", help="Config file path."
    )
    parser.add_argument(
        "--override-config", type=str, default=None, help="Override config file."
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug mode (fast training)."
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Device (auto-detected if omitted)."
    )
    parser.add_argument(
        "--tag", type=str, default="default", help="Experiment tag."
    )
    args = parser.parse_args()

    # Load config
    config = get_effective_config(args.config, args.override_config, debug=args.debug)

    # Device
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # Seed
    set_seed(config["seed"])

    # Logger
    logger = setup_logger(
        log_dir=config["logging"]["log_dir"],
        experiment_name=f"experts_{args.tag}",
        use_tensorboard=config["logging"]["use_tensorboard"],
        use_wandb=config["logging"]["use_wandb"],
    )

    logger.info(f"Device: {device}")
    logger.info(f"Debug mode: {config['debug']['enabled']}")
    logger.info(f"Task sequence: {config['task_sequence']}")

    # Save effective config
    config_save_path = os.path.join(
        config["logging"]["log_dir"], f"experts_{args.tag}", "effective_config.yaml"
    )
    save_config(config, config_save_path)

    # Log environment info
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        logger.info(f"Git commit: {git_hash}")
    except Exception:
        pass

    # Build initial global model
    global_model = build_model(config, device)
    logger.info(f"Model parameters: {global_model.num_parameters:,}")

    # Initialize from checkpoint if available
    init_from_global = config["consolidation"].get("init_from_global", True)
    global_weights = global_model.state_dict()

    # Train experts sequentially
    all_results = []
    checkpoint_dir = config["logging"]["checkpoint_dir"]
    for task_idx, env_id in enumerate(config["task_sequence"]):
        game_name = env_id.replace("NoFrameskip-v4", "")
        logger.info(f"\n{'='*60}")
        logger.info(f"Training Expert {task_idx + 1}/{len(config['task_sequence'])}: {env_id}")
        logger.info(f"{'='*60}")

        # Check for existing best checkpoint to resume from
        best_ckpt_path = os.path.join(
            checkpoint_dir, args.tag, f"expert_{game_name}_best.pt"
        )
        resume_ckpt = best_ckpt_path if os.path.exists(best_ckpt_path) else None
        if resume_ckpt:
            logger.info(f"Found existing best checkpoint: {resume_ckpt}")

        # Initialize expert from global state (only if not resuming)
        expert_init_weights = global_weights if (init_from_global and resume_ckpt is None) else None

        trainer = ExpertTrainer(
            config=config,
            env_id=env_id,
            logger=logger,
            device=device,
            global_weights=expert_init_weights,
            experiment_tag=args.tag,
            resume_checkpoint=resume_ckpt,
        )

        result = trainer.train()
        all_results.append(result)

        # Clean up: remove step checkpoints and final checkpoint, keep only best
        step_ckpts = glob.glob(
            os.path.join(checkpoint_dir, args.tag, f"expert_{game_name}_step*.pt")
        )
        final_ckpt = os.path.join(
            checkpoint_dir, args.tag, f"expert_{game_name}_final.pt"
        )
        removed_count = 0
        for ckpt in step_ckpts:
            os.remove(ckpt)
            removed_count += 1
        if os.path.exists(final_ckpt):
            os.remove(final_ckpt)
            removed_count += 1
        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} intermediate checkpoints for {game_name}.")

        # Update global weights to the average of all experts so far
        # (simple baseline; HTCL/EWC will do better consolidation)
        if init_from_global:
            avg_sd = {}
            for key in global_weights:
                avg_sd[key] = torch.stack(
                    [r["policy_state_dict"][key].float() for r in all_results]
                ).mean(dim=0)
            global_weights = avg_sd
            global_model.load_state_dict(global_weights)

        logger.info(
            f"Expert {task_idx + 1} complete: "
            f"{result['game_name']} | "
            f"Final: {result['final_reward']:.2f} | "
            f"Best: {result['best_reward']:.2f}"
        )

    # Save summary
    summary = {
        "task_sequence": config["task_sequence"],
        "expert_results": [
            {
                "game_name": r["game_name"],
                "env_id": r["env_id"],
                "final_reward": r["final_reward"],
                "best_reward": r["best_reward"],
                "valid_actions": r["valid_actions"],
            }
            for r in all_results
        ],
    }

    # Generate combined reward curves for all experts
    figure_dir = config["logging"].get("figure_dir", "results/figures")
    ExpertTrainer.save_combined_reward_curves(all_results, figure_dir)
    logger.info(f"Combined reward curves saved to {figure_dir}/")

    summary_path = os.path.join(
        config["logging"]["checkpoint_dir"], args.tag, "expert_summary.json"
    )
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\nExpert training complete. Summary saved to {summary_path}")
    logger.close()


if __name__ == "__main__":
    main()
