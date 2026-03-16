"""
Train a DQN model sequentially on Atari tasks using EWC.

This implements the sequential continual learning protocol:
    Task 1 (Breakout) → Task 2 (SpaceInvaders) → Task 3 (Pong)

Task 1 is loaded from an existing expert checkpoint (skipping ~5h of
training). Tasks 2 and 3 are trained from the previous task's solution
with the EWC Fisher-weighted penalty applied.

The final model checkpoint is saved as a bare state_dict compatible with
``scripts/evaluate.py``.

Usage:
    python scripts/train_ewc.py [--debug] [--tag default]
    python scripts/train_ewc.py --online-ewc --tag ewc_online
"""

import argparse
import os
import sys
import json
import subprocess
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.baselines.ewc import EWCTrainer
from src.data.atari_wrappers import compute_union_action_space
from src.utils.config import get_effective_config, save_config
from src.utils.seed import set_seed
from src.utils.logger import setup_logger


def main():
    parser = argparse.ArgumentParser(
        description="Train DQN sequentially with EWC regularization."
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
        "--first-task-expert", type=str, default=None,
        help=(
            "Path to expert checkpoint for Task 1.  If omitted, auto-detects "
            "from results/checkpoints/<tag>/expert_<game>_best.pt."
        ),
    )
    parser.add_argument(
        "--online-ewc", action="store_true",
        help="Use Online EWC (running Fisher average) instead of standard EWC.",
    )
    parser.add_argument(
        "--gamma-ewc", type=float, default=None,
        help="Decay factor for Online EWC (default from config or 0.95).",
    )
    parser.add_argument(
        "--ewc-lambda", type=float, default=None,
        help="Override EWC lambda (penalty strength).",
    )
    args = parser.parse_args()

    # ── Config ───────────────────────────────────────────────────────────
    config = get_effective_config(args.config, debug=args.debug)

    # Device
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    set_seed(config["seed"])

    # Override EWC config from CLI
    if "ewc" not in config:
        config["ewc"] = {}
    if args.ewc_lambda is not None:
        config["ewc"]["lambda"] = args.ewc_lambda
    if args.gamma_ewc is not None:
        config["ewc"]["gamma_ewc"] = args.gamma_ewc

    # ── Logger ───────────────────────────────────────────────────────────
    variant = "online_ewc" if args.online_ewc else "ewc"
    logger = setup_logger(
        log_dir=config["logging"]["log_dir"],
        experiment_name=f"baseline_{variant}_{args.tag}",
        use_tensorboard=config["logging"]["use_tensorboard"],
        use_wandb=config["logging"]["use_wandb"],
    )

    logger.info(f"Device: {device}")
    logger.info(f"Debug mode: {config['debug']['enabled']}")
    logger.info(f"Variant: {variant}")
    logger.info(f"Task sequence: {config['task_sequence']}")

    # ── Union action space ───────────────────────────────────────────────
    union_actions = compute_union_action_space(config["task_sequence"])
    config["model"]["unified_action_dim"] = len(union_actions)
    logger.info(f"Union action space: {union_actions} ({len(union_actions)} actions)")

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
        f"baseline_{variant}_{args.tag}",
        "effective_config.yaml",
    )
    save_config(config, config_path)

    # ── Find Task 1 expert checkpoint ────────────────────────────────────
    first_env = config["task_sequence"][0]
    first_game = first_env.replace("NoFrameskip-v4", "")

    if args.first_task_expert:
        expert_path = args.first_task_expert
    else:
        expert_path = os.path.join(
            config["logging"]["checkpoint_dir"],
            args.tag,
            f"expert_{first_game}_best.pt",
        )

    if not os.path.exists(expert_path):
        logger.info(f"ERROR: Expert checkpoint not found: {expert_path}")
        logger.info(
            "Train experts first with: python scripts/train_experts.py"
        )
        sys.exit(1)

    logger.info(f"Task 1 expert checkpoint: {expert_path}")

    # ── EWC config summary ───────────────────────────────────────────────
    ewc_cfg = config.get("ewc", {})
    logger.info(f"EWC lambda: {ewc_cfg.get('lambda', 5000.0)}")
    logger.info(f"Fisher samples: {ewc_cfg.get('fisher_samples', 5000)}")
    logger.info(f"Online EWC: {args.online_ewc}")
    if args.online_ewc:
        logger.info(
            f"Online EWC gamma: {ewc_cfg.get('gamma_ewc', 0.95)}"
        )

    # ── Train ────────────────────────────────────────────────────────────
    trainer = EWCTrainer(
        config=config,
        union_actions=union_actions,
        device=device,
        logger=logger,
        tag=args.tag,
        online_ewc=args.online_ewc,
        gamma_ewc=ewc_cfg.get("gamma_ewc", 0.95),
    )

    model = trainer.train_sequential(
        task_sequence=config["task_sequence"],
        first_task_checkpoint=expert_path,
    )

    # ── Summary ──────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("EWC Sequential Training Complete")
    logger.info("=" * 60)

    ckpt_dir = os.path.join(config["logging"]["checkpoint_dir"], args.tag)
    final_path = os.path.join(ckpt_dir, "consolidated_ewc.pt")
    logger.info(f"Final model: {final_path}")
    logger.info(
        "Evaluate with:\n"
        f"  python scripts/evaluate.py --model-path {final_path} --all-tasks"
    )

    logger.close()


if __name__ == "__main__":
    main()
