"""Evaluation metrics for continual RL experiments."""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional


def evaluate_agent(
    model: nn.Module,
    env,
    action_mask: np.ndarray,
    device: torch.device,
    num_episodes: int = 30,
    epsilon: float = 0.01,
) -> Dict[str, float]:
    """
    Evaluate a model on a single environment.

    Returns dict with mean_reward, std_reward, min_reward, max_reward.
    """
    model.eval()
    episode_rewards = []

    for _ in range(num_episodes):
        obs, _ = env.reset()
        obs = np.array(obs)
        total_reward = 0.0
        done = False

        while not done:
            action = model.select_action(obs, action_mask, epsilon, device)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
            obs = np.array(next_obs)

        episode_rewards.append(total_reward)

    return {
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "min_reward": float(np.min(episode_rewards)),
        "max_reward": float(np.max(episode_rewards)),
        "episodes": episode_rewards,
    }


def evaluate_on_all_games(
    model: nn.Module,
    eval_envs: List,
    action_masks: List[np.ndarray],
    game_ids: List[str],
    device: torch.device,
    num_episodes: int = 30,
    epsilon: float = 0.01,
) -> Dict[str, Dict[str, float]]:
    """Evaluate a single model on all games. Returns {game_id: metrics}."""
    results = {}
    for env, mask, gid in zip(eval_envs, action_masks, game_ids):
        results[gid] = evaluate_agent(model, env, mask, device, num_episodes, epsilon)
    return results


def compute_forgetting(
    performance_matrix: Dict[str, Dict[str, float]],
    game_ids: List[str],
) -> Dict[str, float]:
    """
    Compute forgetting from the performance matrix.

    performance_matrix[after_game_i][game_j] = mean reward on game j
    after training on game i.

    Forgetting for game j = max reward on j (across all stages) - final reward on j.
    """
    final_stage = game_ids[-1]
    forgetting = {}

    for gid in game_ids:
        rewards_across_stages = [
            performance_matrix[stage][gid]
            for stage in game_ids
            if gid in performance_matrix.get(stage, {})
        ]
        if not rewards_across_stages:
            continue
        peak = max(rewards_across_stages)
        final = performance_matrix[final_stage].get(gid, 0.0)
        forgetting[gid] = peak - final

    return forgetting


def compute_forward_transfer(
    performance_matrix: Dict[str, Dict[str, float]],
    game_ids: List[str],
    random_baselines: Dict[str, float],
) -> Dict[str, float]:
    """
    Forward transfer: performance on game j before training on it,
    minus random baseline.
    """
    fwt = {}
    for idx, gid in enumerate(game_ids):
        if idx == 0:
            continue
        prev_stage = game_ids[idx - 1]
        if gid in performance_matrix.get(prev_stage, {}):
            fwt[gid] = performance_matrix[prev_stage][gid] - random_baselines.get(gid, 0.0)
    return fwt


def aggregate_metrics(
    method_results: Dict[str, Dict[str, float]],
    game_ids: List[str],
) -> Dict[str, float]:
    """
    Compute aggregate metrics across games for a single method.

    Returns mean_reward, std_across_games, and per-game rewards.
    """
    rewards = [
        method_results[gid]["mean_reward"]
        for gid in game_ids
        if gid in method_results
    ]
    return {
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "std_reward": float(np.std(rewards)) if rewards else 0.0,
        "per_game": {gid: method_results[gid]["mean_reward"] for gid in game_ids if gid in method_results},
    }
