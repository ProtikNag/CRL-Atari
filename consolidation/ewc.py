"""
Elastic Weight Consolidation (EWC) for continual DQN.

After training on each game, the Fisher information matrix (diagonal)
is computed and stored. When training on subsequent games, a penalty
is added to the DQN loss proportional to the deviation from the
parameters important to previous games.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy
from typing import Dict, List, Optional, Tuple

from agents.dqn import ReplayBuffer, EpsilonSchedule
from consolidation.taylor import dqn_loss_on_batch


def compute_fisher(
    model: nn.Module,
    transitions: Dict[str, np.ndarray],
    device: torch.device,
    gamma: float = 0.99,
    n_samples: int = 5000,
    batch_size: int = 256,
) -> Dict[str, torch.Tensor]:
    """
    Compute diagonal Fisher Information Matrix using DQN loss.

    The Fisher is approximated as the mean of squared gradients
    over sampled transitions.
    """
    model.eval()
    fisher = {
        name: torch.zeros_like(param, device=device)
        for name, param in model.named_parameters()
        if param.requires_grad
    }

    n = min(n_samples, len(transitions["states"]))
    indices = np.random.choice(len(transitions["states"]), size=n, replace=False)

    states = torch.from_numpy(transitions["states"][indices]).to(device)
    actions = torch.from_numpy(transitions["actions"][indices]).long().to(device)
    rewards = torch.from_numpy(transitions["rewards"][indices]).to(device)
    next_states = torch.from_numpy(transitions["next_states"][indices]).to(device)
    dones = torch.from_numpy(transitions["dones"][indices]).to(device)

    n_batches = 0
    for i in range(0, n, batch_size):
        j = min(i + batch_size, n)
        model.zero_grad()

        loss = dqn_loss_on_batch(
            model,
            states[i:j], actions[i:j], rewards[i:j],
            next_states[i:j], dones[i:j], gamma,
        )
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                fisher[name] += param.grad.detach().pow(2)
        n_batches += 1

    for name in fisher:
        fisher[name] /= max(n_batches, 1)

    return fisher


def ewc_penalty(
    model: nn.Module,
    fisher_list: List[Dict[str, torch.Tensor]],
    star_params_list: List[Dict[str, torch.Tensor]],
) -> torch.Tensor:
    """
    Compute EWC penalty: sum over previous tasks of
        F_k * (theta - theta_k*)^2
    """
    penalty = torch.tensor(0.0, device=next(model.parameters()).device)

    for fisher, star_params in zip(fisher_list, star_params_list):
        for name, param in model.named_parameters():
            if not param.requires_grad or name not in fisher:
                continue
            penalty += (fisher[name] * (param - star_params[name]).pow(2)).sum()

    return penalty


def train_dqn_with_ewc(
    model: nn.Module,
    env,
    action_mask: np.ndarray,
    device: torch.device,
    config: Dict,
    fisher_list: List[Dict[str, torch.Tensor]],
    star_params_list: List[Dict[str, torch.Tensor]],
    lambda_ewc: float = 5000.0,
    log_interval: int = 10000,
) -> Tuple[nn.Module, ReplayBuffer, List[Dict]]:
    """
    Train DQN on a single game with EWC regularization from previous games.
    """
    cfg = config
    obs_shape = env.observation_space.shape
    buffer = ReplayBuffer(cfg["buffer_size"], obs_shape)

    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])
    target_net = copy.deepcopy(model).to(device)
    target_net.eval()
    gamma = cfg["gamma"]
    grad_clip = cfg["grad_clip"]

    epsilon_schedule = EpsilonSchedule(
        cfg["epsilon_start"], cfg["epsilon_end"], cfg["epsilon_decay_steps"]
    )

    training_log = []
    episode_reward = 0.0
    episode_count = 0
    recent_rewards: List[float] = []
    train_steps = 0

    obs, _ = env.reset()
    obs = np.array(obs)

    for step in range(1, cfg["train_steps"] + 1):
        epsilon = epsilon_schedule(step)
        action = model.select_action(obs, action_mask, epsilon, device)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        next_obs = np.array(next_obs)
        done = terminated or truncated

        buffer.push(obs, action, np.clip(reward, -1.0, 1.0), next_obs, done)
        episode_reward += reward

        if done:
            recent_rewards.append(episode_reward)
            episode_reward = 0.0
            episode_count += 1
            obs, _ = env.reset()
            obs = np.array(obs)
        else:
            obs = next_obs

        if len(buffer) >= cfg["learning_starts"] and step % cfg["train_freq"] == 0:
            batch = buffer.sample(cfg["batch_size"])

            states = torch.from_numpy(batch["states"]).to(device)
            actions_t = torch.from_numpy(batch["actions"]).long().to(device)
            rewards_t = torch.from_numpy(batch["rewards"]).to(device)
            next_states = torch.from_numpy(batch["next_states"]).to(device)
            dones_t = torch.from_numpy(batch["dones"]).to(device)

            q_values = model(states).gather(1, actions_t.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                next_q = target_net(next_states).max(1)[0]
                targets = rewards_t + gamma * (1.0 - dones_t) * next_q

            td_loss = F.smooth_l1_loss(q_values, targets)
            ewc_loss = ewc_penalty(model, fisher_list, star_params_list)
            loss = td_loss + (lambda_ewc / 2.0) * ewc_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            train_steps += 1
            if train_steps % cfg["target_update_freq"] == 0:
                target_net.load_state_dict(model.state_dict())

        if step % log_interval == 0:
            mean_r = np.mean(recent_rewards[-20:]) if recent_rewards else 0.0
            entry = {
                "step": step,
                "episodes": episode_count,
                "mean_reward_20": float(mean_r),
                "epsilon": epsilon,
            }
            training_log.append(entry)
            print(f"  Step {step:>7d} | Eps {epsilon:.3f} | "
                  f"Ep {episode_count:>4d} | Mean20 {mean_r:>8.1f}")

    return model, buffer, training_log
