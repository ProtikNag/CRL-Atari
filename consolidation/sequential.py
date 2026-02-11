"""
Sequential fine-tuning baseline.

Trains a single DQN on games one after another without any
continual learning mechanism. This demonstrates catastrophic
forgetting as the lower-bound reference.
"""

import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple

from agents.dqn import ReplayBuffer, train_dqn_on_env


def train_sequential(
    model: nn.Module,
    envs: List,
    action_masks: List[np.ndarray],
    device,
    dqn_config: Dict,
    game_ids: List[str],
    log_interval: int = 10000,
) -> Tuple[nn.Module, Dict[str, ReplayBuffer], List[List[Dict]]]:
    """
    Train a single model sequentially on all games (naive fine-tuning).

    Returns:
        (final_model, buffers_per_game, logs_per_game)
    """
    buffers = {}
    all_logs = []

    for idx, (env, mask, gid) in enumerate(zip(envs, action_masks, game_ids)):
        print(f"\n[Sequential] Training on game {idx+1}/{len(envs)}: {gid}")
        model, buffer, log = train_dqn_on_env(
            model, env, mask, device, dqn_config,
            log_interval=log_interval,
        )
        buffers[gid] = buffer
        all_logs.append(log)

    return model, buffers, all_logs
