"""
Train a DQN model on all Atari tasks simultaneously (multi-task baseline).

This is the standard "upper bound" in continual learning: the model sees
data from every task concurrently, so catastrophic forgetting does not
apply. The result serves as a ceiling for post-hoc consolidation methods.

The default budget is 5M total environment steps distributed round-robin
across tasks (~1.67M per task).  This is configurable via the
``multitask.total_timesteps`` config key or the ``--total-steps`` flag.

Usage:
    python scripts/train_multitask.py [--debug] [--tag default]
    python scripts/train_multitask.py --total-steps 3000000
"""

import argparse
import os
import sys
import subprocess
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.baselines.multitask import MultiTaskTrainer
from src.data.atari_wrappers import compute_union_action_space
from src.utils.config import get_effective_config, save_config
from src.utils.seed import set_seed
from src.utils.logger import setup_logger


def main():
    parser = argparse.ArgumentParser(
        description="Train DQN jointly on all Atari tasks (multi-task baseline)."
    )
    parser.add_argument(
        "--config", type=str, default="configs/base.yaml", help="Config file."
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug mode (fast training)."
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Force device: cpu, cuda, mps."
    )
    parser.add_argument(
        "--tag", type=str, default="default", help="Experiment tag."
    )
    parser.add_argument(
        "--total-steps", type=int, default=None,
        help="Override total environment steps (default: 5M from config).",
    )
    args = parser.parse_args()

    # ── Config ───────────────────────────────────────────────────────────
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

    # Override total steps from CLI
    if "multitask" not in config:
        config["multitask"] = {}
    if args.total_steps is not None:
        config["multitask"]["total_timesteps"] = args.total_steps

    # ── Logger ───────────────────────────────────────────────────────────
    logger = setup_logger(
        log_dir=config["logging"]["log_dir"],
        experiment_name=f"baseline_multitask_{args.tag}",
        use_tensorboard=config["logging"]["use_tensorboard"],
        use_wandb=config["logging"]["use_wandb"],
    )

    logger.info(f"Device: {device}")
    logger.info(f"Debug mode: {config['debug']['enabled']}")
    logger.info(f"Task sequence: {config['task_sequence']}")

    mt_cfg = config.get("multitask", {})
    total_steps = mt_cfg.get("total_timesteps", 5_000_000)
    logger.info(f"Total steps: {total_steps}")
    logger.info(
        f"Per-task steps (approx): "
        f"{total_steps // len(config['task_sequence'])}"
    )

    # ── Union action space ───────────────────────────────────────────────
    union_actions = compute_union_action_space(config["task_sequence"])
    config["model"]["unified_action_dim"] = len(union_actions)
    logger.info(
        f"Union action space: {union_actions} ({len(union_actions)} actions)"
    )

    # Git hash
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        logger.info(f"Git commit: {git_hash}")
    except Exception:
        pass

    # Save config
    config_path = os.path.join(
        config["logging"]["log_dir"],
        f"baseline_multitask_{args.tag}",
        "effective_config.yaml",
    )
    save_config(config, config_path)

    # ── Train ────────────────────────────────────────────────────────────
    trainer = MultiTaskTrainer(
        config=config,
        union_actions=union_actions,
        device=device,
        logger=logger,
        tag=args.tag,
    )

    model = trainer.train(task_sequence=config["task_sequence"])

    # ── Summary ──────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("Multi-Task Joint Training Complete")
    logger.info("=" * 60)

    ckpt_dir = os.path.join(config["logging"]["checkpoint_dir"], args.tag)
    final_path = os.path.join(ckpt_dir, "consolidated_multitask.pt")
    logger.info(f"Final model: {final_path}")
    logger.info(
        "Evaluate with:\n"
        f"  python scripts/evaluate.py --model-path {final_path} --all-tasks"
    )

    logger.close()


if __name__ == "__main__":
    main()
