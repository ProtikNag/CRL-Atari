"""
CRL-Atari: Continual Reinforcement Learning on Atari Games.

Main entry point for quick local debugging and testing.
Run this file directly for a fast debug cycle without the full shell pipeline.

Usage:
    python main.py                    # Debug mode (fast)
    python main.py --no-debug         # Full training
    python main.py --step train       # Only train experts
    python main.py --step consolidate # Only consolidate (needs trained experts)
    python main.py --step evaluate    # Only evaluate (needs checkpoints)
    python main.py --step compare     # Only compare (needs all checkpoints)
"""

import argparse
import os
import sys
import torch

from src.utils.config import get_effective_config, save_config
from src.utils.seed import set_seed
from src.utils.logger import setup_logger


def get_device() -> str:
    """Auto-detect best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def run_train_experts(config: dict, device: str, tag: str) -> None:
    """Train expert agents on all tasks."""
    from src.models.dqn import DQNNetwork
    from src.trainers.expert_trainer import ExpertTrainer
    import json

    logger = setup_logger(
        log_dir=config["logging"]["log_dir"],
        experiment_name=f"experts_{tag}",
        use_tensorboard=config["logging"]["use_tensorboard"],
    )

    model_cfg = config["model"]
    global_model = DQNNetwork(
        in_channels=config["env"]["frame_stack"],
        conv_channels=model_cfg["conv_channels"],
        conv_kernels=model_cfg["conv_kernels"],
        conv_strides=model_cfg["conv_strides"],
        fc_hidden=model_cfg["fc_hidden"],
        unified_action_dim=model_cfg["unified_action_dim"],
        dueling=model_cfg.get("dueling", False),
    ).to(device)

    logger.info(f"Device: {device}")
    logger.info(f"Model parameters: {global_model.num_parameters:,}")
    logger.info(f"Debug: {config['debug']['enabled']}")

    init_from_global = config["consolidation"].get("init_from_global", True)
    global_weights = global_model.state_dict()
    all_results = []

    for task_idx, env_id in enumerate(config["task_sequence"]):
        logger.info(f"\n{'='*60}")
        logger.info(
            f"Training Expert {task_idx + 1}/{len(config['task_sequence'])}: {env_id}"
        )
        logger.info(f"{'='*60}")

        expert_init_weights = global_weights if init_from_global else None

        trainer = ExpertTrainer(
            config=config,
            env_id=env_id,
            logger=logger,
            device=device,
            global_weights=expert_init_weights,
            experiment_tag=tag,
        )

        result = trainer.train()
        all_results.append(result)

        if init_from_global:
            avg_sd = {}
            for key in global_weights:
                avg_sd[key] = torch.stack(
                    [r["policy_state_dict"][key].float() for r in all_results]
                ).mean(dim=0)
            global_weights = avg_sd

        logger.info(
            f"Expert {task_idx + 1}: {result['game_name']} | "
            f"Final: {result['final_reward']:.2f} | Best: {result['best_reward']:.2f}"
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
    summary_path = os.path.join(
        config["logging"]["checkpoint_dir"], tag, "expert_summary.json"
    )
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Expert summary saved to {summary_path}")
    logger.close()


def run_consolidate(config: dict, device: str, tag: str, method: str) -> None:
    """Consolidate experts using specified method."""
    # Build sys.argv for the consolidation script
    saved_argv = sys.argv
    sys.argv = [
        "consolidate.py",
        "--method", method,
        "--config", "configs/base.yaml",
        "--tag", tag,
    ]
    if config["debug"]["enabled"]:
        sys.argv.append("--debug")

    from scripts.consolidate import main as consolidate_main
    consolidate_main()
    sys.argv = saved_argv


def run_compare(config: dict, device: str, tag: str) -> None:
    """Run comparison across all methods."""
    saved_argv = sys.argv
    sys.argv = [
        "compare.py",
        "--config", "configs/base.yaml",
        "--tag", tag,
    ]
    if config["debug"]["enabled"]:
        sys.argv.append("--debug")

    from scripts.compare import main as compare_main
    compare_main()
    sys.argv = saved_argv


def main():
    parser = argparse.ArgumentParser(
        description="CRL-Atari: debug-friendly main entry point."
    )
    parser.add_argument(
        "--no-debug",
        action="store_true",
        help="Disable debug mode (run full training).",
    )
    parser.add_argument(
        "--step",
        type=str,
        default="all",
        choices=["all", "train", "consolidate", "evaluate", "compare"],
        help="Which pipeline step to run.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        choices=["ewc", "distillation", "htcl"],
        help="Consolidation method (for consolidate step).",
    )
    parser.add_argument(
        "--tag", type=str, default="debug", help="Experiment tag."
    )
    args = parser.parse_args()

    import time as _time
    _pipeline_start = _time.time()

    debug = not args.no_debug
    config = get_effective_config("configs/base.yaml", debug=debug)
    device = get_device()
    set_seed(config["seed"])

    print(f"CRL-Atari | Debug: {debug} | Device: {device} | Tag: {args.tag}")
    print(f"Tasks: {config['task_sequence']}")
    print(f"Seed: {config['seed']}")
    print(f"PyTorch: {torch.__version__}")
    if device == "cuda":
        print(f"CUDA: {torch.cuda.get_device_name(0)}")
    print(f"Total timesteps per expert: {config['training']['total_timesteps']}")

    if args.step in ("all", "train"):
        print("\n>>> Training Experts <<<")
        run_train_experts(config, device, args.tag)

    if args.step in ("all", "consolidate"):
        methods = [args.method] if args.method else ["ewc", "distillation", "htcl"]
        for method in methods:
            print(f"\n>>> Consolidating with {method.upper()} <<<")
            run_consolidate(config, device, args.tag, method)

    if args.step in ("all", "compare", "evaluate"):
        print("\n>>> Comparing Models <<<")
        run_compare(config, device, args.tag)

    _total = _time.time() - _pipeline_start
    print(f"\nDone! Total pipeline time: {_total:.1f}s ({_total/60:.1f}m)")


if __name__ == "__main__":
    main()
