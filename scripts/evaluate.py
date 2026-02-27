"""
Evaluate a model (expert or consolidated) on one or all Atari tasks.

Usage:
    python scripts/evaluate.py --model-path PATH --env PongNoFrameskip-v4 [--debug]
    python scripts/evaluate.py --model-path PATH --all-tasks [--debug]
"""

import argparse
import os
import sys
import json
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.dqn import DQNNetwork
from src.data.atari_wrappers import make_atari_env, get_valid_actions, compute_union_action_space
from src.utils.config import get_effective_config
from src.utils.seed import set_seed


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


def evaluate_on_task(
    model: DQNNetwork,
    env_id: str,
    config: dict,
    device: str,
    union_actions: list,
    num_episodes: int = 30,
) -> dict:
    """Evaluate model on a single task.

    Args:
        model: DQN model to evaluate.
        env_id: Environment ID.
        config: Configuration dictionary.
        device: Device string.
        num_episodes: Number of evaluation episodes.

    Returns:
        Dictionary with evaluation metrics.
    """
    env_cfg = config["env"]
    env = make_atari_env(
        env_id=env_id,
        union_actions=union_actions,
        seed=config["seed"] + 2000,
        frame_stack=env_cfg["frame_stack"],
        frame_skip=env_cfg["frame_skip"],
        screen_size=env_cfg["screen_size"],
        noop_max=env_cfg["noop_max"],
        episodic_life=False,
        clip_reward=False,
    )

    valid_actions = get_valid_actions(env_id, union_actions)
    model.eval()
    rewards = []

    for ep in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0.0
        done = False
        steps = 0

        max_steps = 27000  # Safety limit (~108k frames with skip=4)
        while not done and steps < max_steps:
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

            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
            steps += 1

        rewards.append(total_reward)

    env.close()

    return {
        "env_id": env_id,
        "game_name": env_id.replace("NoFrameskip-v4", ""),
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "min_reward": float(np.min(rewards)),
        "max_reward": float(np.max(rewards)),
        "median_reward": float(np.median(rewards)),
        "num_episodes": num_episodes,
        "all_rewards": [float(r) for r in rewards],
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate a model.")
    parser.add_argument(
        "--model-path", type=str, required=True, help="Path to model checkpoint."
    )
    parser.add_argument(
        "--env", type=str, default=None, help="Single environment to evaluate on."
    )
    parser.add_argument(
        "--all-tasks", action="store_true", help="Evaluate on all tasks in sequence."
    )
    parser.add_argument(
        "--config", type=str, default="configs/base.yaml", help="Config file."
    )
    parser.add_argument(
        "--debug", action="store_true", help="Debug mode."
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Device."
    )
    parser.add_argument(
        "--episodes", type=int, default=None, help="Number of eval episodes."
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Output JSON path."
    )
    args = parser.parse_args()

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
    num_episodes = args.episodes or config["evaluation"]["episodes"]

    # Compute union action space
    union_actions = compute_union_action_space(config["task_sequence"])
    config["model"]["unified_action_dim"] = len(union_actions)
    print(f"Union action space: {union_actions} ({len(union_actions)} actions)")

    # Load model
    model = build_model(config, device)

    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    # Handle both full checkpoint and bare state_dict
    if isinstance(checkpoint, dict) and "policy_net" in checkpoint:
        model.load_state_dict(checkpoint["policy_net"])
    else:
        model.load_state_dict(checkpoint)

    print(f"Model loaded from {args.model_path}")
    print(f"Device: {device}")

    # Determine tasks to evaluate
    if args.all_tasks:
        env_ids = config["task_sequence"]
    elif args.env:
        env_ids = [args.env]
    else:
        env_ids = config["task_sequence"]

    # Evaluate
    all_results = []
    for env_id in env_ids:
        print(f"\nEvaluating on {env_id}...")
        result = evaluate_on_task(model, env_id, config, device, union_actions, num_episodes)
        all_results.append(result)
        print(
            f"  {result['game_name']}: "
            f"mean={result['mean_reward']:.2f} +/- {result['std_reward']:.2f} "
            f"(min={result['min_reward']:.1f}, max={result['max_reward']:.1f})"
        )

    # Save results
    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
