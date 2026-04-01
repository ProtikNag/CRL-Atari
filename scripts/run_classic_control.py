#!/usr/bin/env python3
"""
Unified CRL experiment runner for Classic Control environments.

Trains DQN experts on CartPole, Acrobot, and LunarLander, then runs all
consolidation and sequential methods, evaluates everything, and produces
the retention heatmap.

Usage:
    python scripts/run_classic_control.py
    python scripts/run_classic_control.py --debug
    python scripts/run_classic_control.py --config configs/classic_control.yaml
"""

import argparse
import copy
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Project imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.mlp_dqn import MLPDQNNetwork
from src.models.mlp_progressive import MLPProgressiveColumn
from src.data.vector_replay_buffer import VectorReplayBuffer
from src.data.classic_control_wrappers import (
    compute_union_action_space_classic,
    get_valid_actions_classic,
    compute_max_state_dim,
    make_classic_control_env,
    TASK_INFO,
)
from src.utils.seed import set_seed
from src.utils.logger import Logger, setup_logger

# Consolidation methods (reuse existing implementations)
from src.consolidation.whc import WHCConsolidator
from src.consolidation.distillation import DistillationConsolidator
from src.consolidation.hybrid import HybridConsolidator

# Sequential baseline methods
from src.baselines.trac import TRACTrainer
from src.baselines.cchain import CChainTrainer


# ═══════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════

def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration."""
    import yaml
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def apply_debug_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    """Apply debug mode overrides if enabled."""
    debug = config.get("debug", {})
    if not debug.get("enabled", False):
        return config
    config = copy.deepcopy(config)
    train = config["training"]
    train["total_timesteps"] = debug.get("total_timesteps", 5000)
    train["buffer_size"] = debug.get("buffer_size", 500)
    train["min_buffer_size"] = debug.get("min_buffer_size", 200)
    train["eval_episodes"] = debug.get("eval_episodes", 2)
    train["eval_freq"] = debug.get("eval_freq", 1000)
    train["save_freq"] = debug.get("save_freq", 2500)
    config["buffer_size_per_task"] = debug.get("buffer_size_per_task", 500)
    config["distillation"]["distill_epochs"] = debug.get("distill_epochs", 5)
    config["hybrid"]["kd_epochs"] = debug.get("hybrid_kd_epochs", 3)
    config["multitask"]["total_timesteps"] = debug.get("multitask_total_timesteps", 5000)
    config["progress_compress"]["progress_steps"] = debug.get("pc_progress_steps", 5000)
    config["progress_compress"]["compress_epochs"] = debug.get("pc_compress_epochs", 5)
    config["htcl"]["filtered_buffer_size"] = debug.get("filtered_buffer_size", 200)
    config["htcl"]["fisher_samples"] = debug.get("fisher_samples", 50)
    config["whc"]["fisher_samples"] = debug.get("fisher_samples", 50)
    config["ewc"]["fisher_samples"] = debug.get("fisher_samples", 50)
    config["evaluation"]["episodes"] = debug.get("eval_episodes", 2)
    return config


def get_device() -> str:
    """Auto-detect best available device."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ═══════════════════════════════════════════════════════════════════════
# Model factory
# ═══════════════════════════════════════════════════════════════════════

def build_model(config: Dict[str, Any], device: str) -> MLPDQNNetwork:
    """Create a fresh MLPDQNNetwork from config."""
    model_cfg = config["model"]
    return MLPDQNNetwork(
        state_dim=config["max_state_dim"],
        hidden_dims=model_cfg.get("hidden_dims", [128, 128]),
        unified_action_dim=model_cfg["unified_action_dim"],
        dueling=model_cfg.get("dueling", False),
    ).to(device)


# ═══════════════════════════════════════════════════════════════════════
# Expert Training
# ═══════════════════════════════════════════════════════════════════════

def train_expert(
    config: Dict[str, Any],
    env_id: str,
    union_actions: List[int],
    max_state_dim: int,
    device: str,
    logger: Logger,
    tag: str = "default",
) -> Dict[str, Any]:
    """Train a single DQN expert on one classic control task.

    Returns a result dict with policy_state_dict, valid_actions,
    game_name, replay_buffer, eval data.
    """
    train_cfg = config["training"]
    total_timesteps = train_cfg["total_timesteps"]
    eval_freq = train_cfg["eval_freq"]
    eval_episodes = train_cfg["eval_episodes"]
    train_freq = train_cfg["train_freq"]
    log_interval = config["logging"].get("log_interval", 500)
    checkpoint_dir = config["logging"]["checkpoint_dir"]

    valid_actions = get_valid_actions_classic(env_id, union_actions)
    game_name = env_id.replace("-v1", "").replace("-v2", "").replace("-v3", "")

    logger.info(f"Training expert for {game_name} (valid actions: {valid_actions})")

    model = build_model(config, device)
    target_net = copy.deepcopy(model)
    target_net.eval()

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg["learning_rate"])

    replay_buffer = VectorReplayBuffer(
        capacity=train_cfg["buffer_size"],
        state_dim=max_state_dim,
        device=device,
    )

    env = make_classic_control_env(
        env_id, union_actions, max_state_dim,
        seed=config["seed"],
    )

    action_mask = torch.full(
        (config["model"]["unified_action_dim"],), float("-inf"), device=device
    )
    action_mask[valid_actions] = 0.0

    # Training loop
    state, _ = env.reset()
    episode_reward = 0.0
    episode_count = 0
    episode_rewards = []
    best_eval_reward = float("-inf")
    train_steps = 0
    start_time = time.time()

    eps_start = train_cfg["eps_start"]
    eps_end = train_cfg["eps_end"]
    eps_decay = train_cfg["eps_decay_steps"]
    gamma = train_cfg["gamma"]
    batch_size = train_cfg["batch_size"]
    min_buffer = train_cfg["min_buffer_size"]
    target_update_freq = train_cfg["target_update_freq"]
    double_dqn = train_cfg.get("double_dqn", True)
    gradient_clip = train_cfg.get("gradient_clip", 10.0)

    best_state_dict = None

    for step in range(1, total_timesteps + 1):
        eps = eps_start + min(step / eps_decay, 1.0) * (eps_end - eps_start)

        if np.random.random() < eps:
            action = np.random.choice(valid_actions)
        else:
            with torch.no_grad():
                s_t = torch.from_numpy(state).float().unsqueeze(0).to(device)
                q = model(s_t) + action_mask.unsqueeze(0)
                action = q.argmax(dim=1).item()

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        replay_buffer.push(state, action, reward, next_state, done)
        episode_reward += reward
        state = next_state

        # Train
        if step % train_freq == 0 and len(replay_buffer) >= min_buffer:
            states_b, actions_b, rewards_b, next_b, dones_b = replay_buffer.sample(batch_size)

            q_vals = model(states_b)
            q_taken = q_vals.gather(1, actions_b.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                if double_dqn:
                    next_q_policy = model(next_b) + action_mask.unsqueeze(0)
                    next_acts = next_q_policy.argmax(dim=1, keepdim=True)
                    next_q_target = target_net(next_b)
                    next_q_max = next_q_target.gather(1, next_acts).squeeze(1)
                else:
                    next_q = target_net(next_b) + action_mask.unsqueeze(0)
                    next_q_max = next_q.max(dim=1)[0]
                target_vals = rewards_b + gamma * next_q_max * (1 - dones_b)

            loss = F.smooth_l1_loss(q_taken, target_vals)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            train_steps += 1

            if train_steps % target_update_freq == 0:
                target_net.load_state_dict(model.state_dict())

        if done:
            episode_count += 1
            episode_rewards.append(episode_reward)
            if step % log_interval == 0 or episode_count % 20 == 0:
                recent = np.mean(episode_rewards[-100:])
                elapsed = time.time() - start_time
                logger.info(
                    f"[{game_name}] Step {step}/{total_timesteps} | "
                    f"Ep {episode_count} | R: {episode_reward:.1f} | "
                    f"Mean100: {recent:.1f} | Eps: {eps:.3f} | {elapsed:.0f}s"
                )
            episode_reward = 0.0
            state, _ = env.reset()

        if step % eval_freq == 0:
            eval_r = evaluate_model(
                model, env_id, union_actions, max_state_dim,
                valid_actions, config, device, eval_episodes,
            )
            logger.info(f"[{game_name}] Eval at step {step}: {eval_r:.2f}")
            if eval_r > best_eval_reward:
                best_eval_reward = eval_r
                best_state_dict = copy.deepcopy(model.state_dict())

    env.close()

    # Use best checkpoint
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    # Save checkpoint
    ckpt_dir = os.path.join(checkpoint_dir, tag)
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"expert_{game_name}_best.pt")
    torch.save(model.state_dict(), ckpt_path)
    logger.info(f"[{game_name}] Expert saved: {ckpt_path} (best eval: {best_eval_reward:.2f})")

    return {
        "policy_state_dict": copy.deepcopy(model.state_dict()),
        "valid_actions": valid_actions,
        "game_name": game_name,
        "env_id": env_id,
        "replay_buffer": replay_buffer,
        "best_reward": best_eval_reward,
    }


# ═══════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════

def evaluate_model(
    model: nn.Module,
    env_id: str,
    union_actions: List[int],
    max_state_dim: int,
    valid_actions: List[int],
    config: Dict[str, Any],
    device: str,
    num_episodes: int = 30,
) -> float:
    """Evaluate model on a single task. Returns mean reward."""
    env = make_classic_control_env(
        env_id, union_actions, max_state_dim,
        seed=config["seed"] + 3000,
    )
    n_actions = config["model"]["unified_action_dim"]
    action_mask = torch.full((n_actions,), float("-inf"), device=device)
    action_mask[valid_actions] = 0.0

    model.eval()
    rewards = []
    for _ in range(num_episodes):
        state, _ = env.reset()
        total_r = 0.0
        done = False
        steps = 0
        while not done and steps < 10000:
            with torch.no_grad():
                s_t = torch.from_numpy(state).float().unsqueeze(0).to(device)
                q = model(s_t) + action_mask.unsqueeze(0)
                action = q.argmax(dim=1).item()
            state, reward, terminated, truncated, _ = env.step(action)
            total_r += reward
            done = terminated or truncated
            steps += 1
        rewards.append(total_r)

    env.close()
    return float(np.mean(rewards))


def evaluate_all_tasks(
    model: nn.Module,
    task_sequence: List[str],
    union_actions: List[int],
    max_state_dim: int,
    config: Dict[str, Any],
    device: str,
    num_episodes: int = 30,
) -> List[Dict[str, Any]]:
    """Evaluate model on all tasks, return list of result dicts."""
    results = []
    for env_id in task_sequence:
        valid_actions = get_valid_actions_classic(env_id, union_actions)
        game_name = env_id.replace("-v1", "").replace("-v2", "").replace("-v3", "")

        rewards = []
        env = make_classic_control_env(
            env_id, union_actions, max_state_dim,
            seed=config["seed"] + 3000,
        )
        n_actions = config["model"]["unified_action_dim"]
        action_mask = torch.full((n_actions,), float("-inf"), device=device)
        action_mask[valid_actions] = 0.0

        model.eval()
        for _ in range(num_episodes):
            state, _ = env.reset()
            total_r = 0.0
            done = False
            steps = 0
            while not done and steps < 10000:
                with torch.no_grad():
                    s_t = torch.from_numpy(state).float().unsqueeze(0).to(device)
                    q = model(s_t) + action_mask.unsqueeze(0)
                    action = q.argmax(dim=1).item()
                state, reward, terminated, truncated, _ = env.step(action)
                total_r += reward
                done = terminated or truncated
                steps += 1
            rewards.append(total_r)
        env.close()

        results.append({
            "game_name": game_name,
            "env_id": env_id,
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "all_rewards": rewards,
        })
    return results


# ═══════════════════════════════════════════════════════════════════════
# EWC Sequential Training (Classic Control)
# ═══════════════════════════════════════════════════════════════════════

def compute_fisher_classic(
    model: MLPDQNNetwork,
    states: torch.Tensor,
    valid_actions: List[int],
    device: str,
    num_samples: int = 2000,
) -> Dict[str, torch.Tensor]:
    """Compute diagonal Fisher at model's current parameters."""
    model.eval()
    fisher = {n: torch.zeros_like(p, device=device)
              for n, p in model.named_parameters() if p.requires_grad}

    n = min(num_samples, states.shape[0])
    indices = torch.randperm(states.shape[0])[:n]

    action_mask = torch.full((model.unified_action_dim,), float("-inf"), device=device)
    action_mask[valid_actions] = 0.0

    for idx in indices:
        state = states[idx].unsqueeze(0).to(device)
        model.zero_grad()
        q_values = model(state)
        masked_q = q_values + action_mask.unsqueeze(0)
        best_action = masked_q.argmax(dim=1)

        valid_q = q_values[0, valid_actions]
        log_probs = torch.log_softmax(valid_q, dim=0)
        best_in_valid = valid_actions.index(best_action.item())
        log_probs[best_in_valid].backward()

        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                fisher[name] += param.grad.detach() ** 2

    for name in fisher:
        fisher[name] /= n
    model.train()
    return fisher


def train_ewc_classic(
    config: Dict[str, Any],
    task_sequence: List[str],
    union_actions: List[int],
    max_state_dim: int,
    first_expert_state_dict: Dict[str, torch.Tensor],
    device: str,
    logger: Logger,
    tag: str = "default",
) -> MLPDQNNetwork:
    """Sequential EWC training on classic control."""
    train_cfg = config["training"]
    ewc_cfg = config.get("ewc", {})
    ewc_lambda = ewc_cfg.get("lambda", 1000.0)
    fisher_samples = ewc_cfg.get("fisher_samples", 2000)
    gamma_ewc = ewc_cfg.get("gamma_ewc", 0.95)

    total_timesteps = train_cfg["total_timesteps"]
    train_freq = train_cfg["train_freq"]
    log_interval = config["logging"].get("log_interval", 500)
    checkpoint_dir = config["logging"]["checkpoint_dir"]

    model = build_model(config, device)
    target_net = copy.deepcopy(model)
    target_net.eval()

    # Online EWC state
    online_fisher = None
    online_params = None

    for task_idx, env_id in enumerate(task_sequence):
        game_name = env_id.replace("-v1", "").replace("-v2", "").replace("-v3", "")
        valid_actions = get_valid_actions_classic(env_id, union_actions)

        logger.info(f"\nEWC Task {task_idx+1}/{len(task_sequence)}: {game_name}")

        if task_idx == 0:
            model.load_state_dict(first_expert_state_dict)
            target_net.load_state_dict(model.state_dict())
            # Compute Fisher for task 1
            states = _collect_states(env_id, union_actions, max_state_dim,
                                     config, ewc_cfg.get("buffer_size_per_task", 5000))
            fisher = compute_fisher_classic(model, states, valid_actions, device, fisher_samples)
            online_fisher = {k: v.clone() for k, v in fisher.items()}
            online_params = {n: p.detach().clone() for n, p in model.named_parameters()}
            logger.info(f"  Task 1 loaded from expert. Fisher computed.")
            continue

        # Tasks 2+: train with EWC penalty
        env = make_classic_control_env(env_id, union_actions, max_state_dim,
                                       seed=config["seed"] + task_idx * 1000)

        optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg["learning_rate"])
        replay_buffer = VectorReplayBuffer(train_cfg["buffer_size"], max_state_dim, device)

        action_mask = torch.full((config["model"]["unified_action_dim"],), float("-inf"), device=device)
        action_mask[valid_actions] = 0.0

        state, _ = env.reset()
        episode_reward = 0.0
        episode_count = 0
        episode_rewards = []
        train_steps = 0

        for step in range(1, total_timesteps + 1):
            eps = train_cfg["eps_start"] + min(step / train_cfg["eps_decay_steps"], 1.0) * (
                train_cfg["eps_end"] - train_cfg["eps_start"])

            if np.random.random() < eps:
                action = np.random.choice(valid_actions)
            else:
                with torch.no_grad():
                    s_t = torch.from_numpy(state).float().unsqueeze(0).to(device)
                    q = model(s_t) + action_mask.unsqueeze(0)
                    action = q.argmax(dim=1).item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            replay_buffer.push(state, action, reward, next_state, done)
            episode_reward += reward
            state = next_state

            if step % train_freq == 0 and len(replay_buffer) >= train_cfg["min_buffer_size"]:
                states_b, actions_b, rewards_b, next_b, dones_b = replay_buffer.sample(train_cfg["batch_size"])

                q_vals = model(states_b)
                q_taken = q_vals.gather(1, actions_b.unsqueeze(1)).squeeze(1)

                with torch.no_grad():
                    next_q_policy = model(next_b) + action_mask.unsqueeze(0)
                    next_acts = next_q_policy.argmax(dim=1, keepdim=True)
                    next_q_target = target_net(next_b)
                    next_q_max = next_q_target.gather(1, next_acts).squeeze(1)
                    target_vals = rewards_b + train_cfg["gamma"] * next_q_max * (1 - dones_b)

                dqn_loss = F.smooth_l1_loss(q_taken, target_vals)

                # EWC penalty
                ewc_penalty = torch.tensor(0.0, device=device)
                if online_fisher is not None:
                    for name, param in model.named_parameters():
                        if name in online_fisher:
                            diff = param - online_params[name]
                            ewc_penalty += (online_fisher[name] * diff ** 2).sum()
                    ewc_penalty = (ewc_lambda / 2.0) * ewc_penalty

                total_loss = dqn_loss + ewc_penalty
                optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), train_cfg.get("gradient_clip", 10.0))
                optimizer.step()
                train_steps += 1

                if train_steps % train_cfg["target_update_freq"] == 0:
                    target_net.load_state_dict(model.state_dict())

            if done:
                episode_count += 1
                episode_rewards.append(episode_reward)
                if step % log_interval == 0:
                    logger.info(
                        f"[EWC-{game_name}] Step {step}/{total_timesteps} | "
                        f"R: {episode_reward:.1f} | Mean100: {np.mean(episode_rewards[-100:]):.1f}")
                episode_reward = 0.0
                state, _ = env.reset()

        env.close()

        # Compute and accumulate Fisher
        states = _collect_states(env_id, union_actions, max_state_dim,
                                 config, ewc_cfg.get("buffer_size_per_task", 5000))
        fisher = compute_fisher_classic(model, states, valid_actions, device, fisher_samples)
        for name in online_fisher:
            online_fisher[name] = gamma_ewc * online_fisher[name] + fisher[name]
        online_params = {n: p.detach().clone() for n, p in model.named_parameters()}

    # Save final model
    ckpt_dir = os.path.join(checkpoint_dir, tag)
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(ckpt_dir, "consolidated_ewc.pt"))
    logger.info("EWC training complete.")
    return model


# ═══════════════════════════════════════════════════════════════════════
# Progress & Compress (Classic Control)
# ═══════════════════════════════════════════════════════════════════════

def train_progress_compress_classic(
    config: Dict[str, Any],
    task_sequence: List[str],
    union_actions: List[int],
    max_state_dim: int,
    device: str,
    logger: Logger,
    tag: str = "default",
) -> MLPDQNNetwork:
    """Progress & Compress on classic control."""
    pc_cfg = config.get("progress_compress", {})
    train_cfg = config["training"]
    progress_steps = pc_cfg.get("progress_steps", 100_000)
    compress_epochs = pc_cfg.get("compress_epochs", 1000)
    compress_lr = pc_cfg.get("compress_lr", 1e-3)
    compress_batch_size = pc_cfg.get("compress_batch_size", 128)
    temperature = pc_cfg.get("temperature", 1.0)
    lambda_ewc = pc_cfg.get("lambda_ewc", 1000.0)
    gamma_ewc = pc_cfg.get("gamma_ewc", 0.95)
    fisher_samples = pc_cfg.get("fisher_samples", 2000)
    buffer_size_per_task = pc_cfg.get("buffer_size_per_task", 5000)

    checkpoint_dir = config["logging"]["checkpoint_dir"]
    log_interval = config["logging"].get("log_interval", 500)
    n_actions = config["model"]["unified_action_dim"]

    # Build KB
    kb = build_model(config, device)

    # Online EWC state
    fisher_accum = None
    kb_star = None

    for task_idx, env_id in enumerate(task_sequence):
        game_name = env_id.replace("-v1", "").replace("-v2", "").replace("-v3", "")
        valid_actions = get_valid_actions_classic(env_id, union_actions)

        logger.info(f"\nP&C Task {task_idx+1}/{len(task_sequence)}: {game_name}")

        # ── Progress phase: train Active Column ──
        logger.info(f"  Progress phase ({progress_steps} steps)...")
        for p in kb.parameters():
            p.requires_grad_(False)
        kb.eval()

        ac = MLPProgressiveColumn(
            kb=kb,
            state_dim=max_state_dim,
            hidden_dims=config["model"].get("hidden_dims", [128, 128]),
            unified_action_dim=n_actions,
        ).to(device)

        target_ac = copy.deepcopy(ac)
        # Share the same KB reference
        object.__setattr__(target_ac, '_kb_ref', object.__getattribute__(ac, '_kb_ref'))
        target_ac.eval()

        optimizer = torch.optim.AdamW(ac.parameters(), lr=train_cfg["learning_rate"])
        replay_buffer = VectorReplayBuffer(train_cfg["buffer_size"], max_state_dim, device)

        action_mask = torch.full((n_actions,), float("-inf"), device=device)
        action_mask[valid_actions] = 0.0

        env = make_classic_control_env(env_id, union_actions, max_state_dim,
                                       seed=config["seed"] + task_idx * 1000)

        state, _ = env.reset()
        episode_reward = 0.0
        episode_count = 0
        episode_rewards = []
        train_steps = 0

        for step in range(1, progress_steps + 1):
            eps = train_cfg["eps_start"] + min(step / train_cfg["eps_decay_steps"], 1.0) * (
                train_cfg["eps_end"] - train_cfg["eps_start"])

            if np.random.random() < eps:
                action = np.random.choice(valid_actions)
            else:
                with torch.no_grad():
                    s_t = torch.from_numpy(state).float().unsqueeze(0).to(device)
                    q = ac(s_t) + action_mask.unsqueeze(0)
                    action = q.argmax(dim=1).item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            replay_buffer.push(state, action, reward, next_state, terminated)
            episode_reward += reward
            state = next_state

            if step % train_cfg["train_freq"] == 0 and len(replay_buffer) >= train_cfg["min_buffer_size"]:
                states_b, actions_b, rewards_b, next_b, dones_b = replay_buffer.sample(train_cfg["batch_size"])

                q_vals = ac(states_b)
                q_taken = q_vals.gather(1, actions_b.unsqueeze(1)).squeeze(1)

                with torch.no_grad():
                    next_q_policy = ac(next_b) + action_mask.unsqueeze(0)
                    next_acts = next_q_policy.argmax(dim=1, keepdim=True)
                    next_q_target = target_ac(next_b)
                    next_q_max = next_q_target.gather(1, next_acts).squeeze(1)
                    target_vals = rewards_b + train_cfg["gamma"] * next_q_max * (1 - dones_b)

                loss = F.smooth_l1_loss(q_taken, target_vals)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(ac.parameters(), train_cfg.get("gradient_clip", 10.0))
                optimizer.step()
                train_steps += 1

                if train_steps % train_cfg["target_update_freq"] == 0:
                    # Sync only AC-owned parameters
                    target_ac.trunk.load_state_dict(ac.trunk.state_dict())
                    target_ac.ac_linear.load_state_dict(ac.ac_linear.state_dict())
                    target_ac.lateral.load_state_dict(ac.lateral.state_dict())
                    target_ac.head.load_state_dict(ac.head.state_dict())

            if done:
                episode_count += 1
                episode_rewards.append(episode_reward)
                if step % log_interval == 0:
                    logger.info(
                        f"[PC-Progress-{game_name}] Step {step}/{progress_steps} | "
                        f"R: {episode_reward:.1f} | Mean100: {np.mean(episode_rewards[-100:]):.1f}")
                episode_reward = 0.0
                state, _ = env.reset()

        env.close()

        # Unfreeze KB
        for p in kb.parameters():
            p.requires_grad_(True)

        # ── Compress phase: distill AC -> KB with EWC ──
        logger.info(f"  Compress phase ({compress_epochs} epochs)...")

        compress_states = _collect_states(env_id, union_actions, max_state_dim,
                                          config, buffer_size_per_task)

        for p in ac.parameters():
            p.requires_grad_(False)
        ac.eval()

        kb_optimizer = torch.optim.AdamW(kb.parameters(), lr=compress_lr)
        kb.train()

        for epoch in range(1, compress_epochs + 1):
            idx = torch.randint(0, compress_states.shape[0], (compress_batch_size,))
            s_batch = compress_states[idx].to(device)

            with torch.no_grad():
                teacher_probs = torch.softmax(
                    ac(s_batch)[:, valid_actions] / temperature, dim=1)

            student_log_probs = torch.log_softmax(
                kb(s_batch)[:, valid_actions] / temperature, dim=1)

            kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")

            # EWC penalty
            ewc_pen = torch.tensor(0.0, device=device)
            if fisher_accum is not None:
                for name, param in kb.named_parameters():
                    if name in fisher_accum:
                        diff = param - kb_star[name]
                        ewc_pen += (fisher_accum[name] * diff ** 2).sum()
                ewc_pen = (lambda_ewc / 2.0) * ewc_pen

            total_loss = kl_loss + ewc_pen
            kb_optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(kb.parameters(), 10.0)
            kb_optimizer.step()

            if epoch % max(1, compress_epochs // 5) == 0:
                logger.info(
                    f"[PC-Compress-{game_name}] Epoch {epoch}/{compress_epochs} | "
                    f"KL: {kl_loss.item():.4f} | EWC: {ewc_pen.item():.4f}")

        # Accumulate Fisher
        fisher = compute_fisher_classic(kb, compress_states, valid_actions, device, fisher_samples)
        if fisher_accum is None:
            fisher_accum = {k: v.clone() for k, v in fisher.items()}
        else:
            for name in fisher_accum:
                fisher_accum[name] = gamma_ewc * fisher_accum[name] + fisher[name]
        kb_star = {n: p.detach().clone() for n, p in kb.named_parameters()}

    # Save
    ckpt_dir = os.path.join(checkpoint_dir, tag)
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(kb.state_dict(), os.path.join(ckpt_dir, "consolidated_pc.pt"))
    logger.info("Progress & Compress complete.")
    return kb


# ═══════════════════════════════════════════════════════════════════════
# Multi-Task Joint Training (Classic Control)
# ═══════════════════════════════════════════════════════════════════════

def train_multitask_classic(
    config: Dict[str, Any],
    task_sequence: List[str],
    union_actions: List[int],
    max_state_dim: int,
    device: str,
    logger: Logger,
    tag: str = "default",
) -> MLPDQNNetwork:
    """Joint multi-task training on all classic control tasks."""
    train_cfg = config["training"]
    mt_cfg = config.get("multitask", {})
    total_timesteps = mt_cfg.get("total_timesteps", 200_000)
    log_interval = config["logging"].get("log_interval", 500)
    checkpoint_dir = config["logging"]["checkpoint_dir"]
    n_actions = config["model"]["unified_action_dim"]

    model = build_model(config, device)
    target_net = copy.deepcopy(model)
    target_net.eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg["learning_rate"])

    num_tasks = len(task_sequence)
    envs, buffers, action_masks, valid_actions_list, game_names = [], [], [], [], []
    states, episode_rewards_list, episode_counts = [], [], []

    for task_idx, env_id in enumerate(task_sequence):
        game_name = env_id.replace("-v1", "").replace("-v2", "").replace("-v3", "")
        game_names.append(game_name)
        valid_actions = get_valid_actions_classic(env_id, union_actions)
        valid_actions_list.append(valid_actions)

        env = make_classic_control_env(env_id, union_actions, max_state_dim,
                                       seed=config["seed"] + task_idx * 1000)
        envs.append(env)

        buf = VectorReplayBuffer(
            train_cfg["buffer_size"] // num_tasks, max_state_dim, device)
        buffers.append(buf)

        mask = torch.full((n_actions,), float("-inf"), device=device)
        mask[valid_actions] = 0.0
        action_masks.append(mask)

        state, _ = env.reset()
        states.append(state)
        episode_rewards_list.append(0.0)
        episode_counts.append(0)

    logger.info(f"\nMulti-Task Joint Training ({total_timesteps} steps)")

    train_steps = 0
    best_avg = float("-inf")
    best_sd = None

    for step in range(1, total_timesteps + 1):
        task_idx = (step - 1) % num_tasks
        env = envs[task_idx]
        buf = buffers[task_idx]
        valid_actions = valid_actions_list[task_idx]
        a_mask = action_masks[task_idx]
        state = states[task_idx]

        eps = train_cfg["eps_start"] + min(step / train_cfg["eps_decay_steps"], 1.0) * (
            train_cfg["eps_end"] - train_cfg["eps_start"])

        if np.random.random() < eps:
            action = np.random.choice(valid_actions)
        else:
            with torch.no_grad():
                s_t = torch.from_numpy(state).float().unsqueeze(0).to(device)
                q = model(s_t) + a_mask.unsqueeze(0)
                action = q.argmax(dim=1).item()

        next_state, reward, terminated, truncated, _ = env.step(action)
        buf.push(state, action, reward, next_state, terminated)
        episode_rewards_list[task_idx] += reward
        states[task_idx] = next_state

        if terminated or truncated:
            episode_counts[task_idx] += 1
            episode_rewards_list[task_idx] = 0.0
            states[task_idx], _ = env.reset()

        if step % train_cfg["train_freq"] == 0:
            min_buf = train_cfg["min_buffer_size"] // num_tasks
            ready = [i for i in range(num_tasks) if len(buffers[i]) >= min_buf]
            if ready:
                t_idx = ready[np.random.randint(len(ready))]
                t_mask = action_masks[t_idx]
                states_b, actions_b, rewards_b, next_b, dones_b = buffers[t_idx].sample(train_cfg["batch_size"])

                q_vals = model(states_b)
                q_taken = q_vals.gather(1, actions_b.unsqueeze(1)).squeeze(1)

                with torch.no_grad():
                    next_q_policy = model(next_b) + t_mask.unsqueeze(0)
                    next_acts = next_q_policy.argmax(dim=1, keepdim=True)
                    next_q_target = target_net(next_b)
                    next_q_max = next_q_target.gather(1, next_acts).squeeze(1)
                    target_vals = rewards_b + train_cfg["gamma"] * next_q_max * (1 - dones_b)

                loss = F.smooth_l1_loss(q_taken, target_vals)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), train_cfg.get("gradient_clip", 10.0))
                optimizer.step()
                train_steps += 1

                if train_steps % train_cfg["target_update_freq"] == 0:
                    target_net.load_state_dict(model.state_dict())

        if step % train_cfg["eval_freq"] == 0:
            eval_rewards = []
            for i, env_id in enumerate(task_sequence):
                r = evaluate_model(model, env_id, union_actions, max_state_dim,
                                   valid_actions_list[i], config, device,
                                   train_cfg["eval_episodes"])
                eval_rewards.append(r)
            avg = np.mean(eval_rewards)
            if avg > best_avg:
                best_avg = avg
                best_sd = copy.deepcopy(model.state_dict())
            if step % log_interval == 0:
                parts = [f"{game_names[i]}={eval_rewards[i]:.1f}" for i in range(num_tasks)]
                logger.info(f"[MTL] Step {step} | {' | '.join(parts)} | avg={avg:.1f}")

    for env in envs:
        env.close()

    if best_sd is not None:
        model.load_state_dict(best_sd)

    ckpt_dir = os.path.join(checkpoint_dir, tag)
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(ckpt_dir, "consolidated_multitask.pt"))
    logger.info("Multi-Task training complete.")
    return model


# ═══════════════════════════════════════════════════════════════════════
# Helper: collect states
# ═══════════════════════════════════════════════════════════════════════

def _collect_states(
    env_id: str,
    union_actions: List[int],
    max_state_dim: int,
    config: Dict[str, Any],
    num_states: int,
) -> torch.Tensor:
    """Collect state vectors by random rollout."""
    valid_actions = get_valid_actions_classic(env_id, union_actions)
    env = make_classic_control_env(env_id, union_actions, max_state_dim,
                                   seed=config["seed"] + 5000)
    collected = []
    state, _ = env.reset()
    for _ in range(num_states):
        collected.append(state)
        action = np.random.choice(valid_actions)
        next_state, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            state, _ = env.reset()
        else:
            state = next_state
    env.close()
    return torch.from_numpy(np.array(collected)).float()


# ═══════════════════════════════════════════════════════════════════════
# Visualization: Retention Heatmap
# ═══════════════════════════════════════════════════════════════════════

def generate_heatmap(
    all_eval_data: Dict[str, List[Dict[str, Any]]],
    expert_rewards: Dict[str, float],
    figure_dir: str,
    game_names: List[str],
) -> None:
    """Generate retention heatmap (% of expert reward)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    AC_SERIES = ['#2563EB', '#D97706', '#059669', '#DC2626',
                 '#7C3AED', '#0891B2', '#BE185D', '#92400E']

    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Inter', 'Helvetica', 'Arial'],
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': True, 'grid.color': '#E9ECEF', 'grid.linewidth': 0.6,
        'axes.edgecolor': '#495057', 'axes.labelcolor': '#212529',
        'xtick.color': '#6C757D', 'ytick.color': '#6C757D',
        'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    })

    # Method order (no One-Shot, no Iterative)
    method_order = ["Multi-Task", "WHC", "Distillation", "Hybrid",
                    "EWC", "Progress & Compress", "TRAC", "C-CHAIN"]

    methods = [m for m in method_order if m in all_eval_data]
    n_m = len(methods)
    n_g = len(game_names)

    if n_m == 0:
        print("  No methods to plot in heatmap.")
        return

    ret_matrix = np.zeros((n_m, n_g))
    raw_matrix = np.zeros((n_m, n_g))

    for i, method in enumerate(methods):
        evals = all_eval_data[method]
        eval_dict = {e["game_name"]: e for e in evals}
        for j, game in enumerate(game_names):
            if game in eval_dict:
                raw = eval_dict[game]["mean_reward"]
                exp = expert_rewards.get(game, 1.0)
                raw_matrix[i, j] = raw
                if exp != 0:
                    ret_matrix[i, j] = raw / exp * 100
                else:
                    ret_matrix[i, j] = 100.0 if raw == 0 else 0.0

    fig, ax = plt.subplots(figsize=(8, 0.6 * n_m + 2.5))

    from matplotlib.colors import TwoSlopeNorm
    vmin, vmax = min(0, np.nanmin(ret_matrix)), max(100, np.nanmax(ret_matrix) * 1.05)
    vcenter = 50
    if vmin >= vcenter:
        vcenter = (vmin + vmax) / 2
    norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    cmap = mpl.colormaps.get_cmap("RdYlGn")

    im = ax.imshow(ret_matrix, cmap=cmap, norm=norm, aspect="auto")

    for i in range(n_m):
        for j in range(n_g):
            pct = ret_matrix[i, j]
            raw = raw_matrix[i, j]
            text_color = "#FFFFFF" if pct < 20 or pct > 80 else "#212529"
            ax.text(j, i, f"{raw:.1f}\n({pct:.0f}%)",
                    ha="center", va="center", fontsize=9, fontweight="600",
                    color=text_color)

    ax.set_xticks(range(n_g))
    ax.set_xticklabels(game_names, fontsize=11)
    ax.set_yticks(range(n_m))
    ax.set_yticklabels(methods, fontsize=10)

    for e in range(n_g + 1):
        ax.axvline(e - 0.5, color="white", linewidth=2.5)
    for e in range(n_m + 1):
        ax.axhline(e - 0.5, color="white", linewidth=2.5)

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("% of Expert Reward", fontsize=11)

    ax.set_title("Consolidation Performance (% of Expert)",
                 fontsize=14, fontweight="600", color="#212529", pad=14,
                 fontfamily="serif")

    for spine in ax.spines.values():
        spine.set_visible(False)

    for fmt in ("png", "svg"):
        out_dir = os.path.join(figure_dir, fmt)
        os.makedirs(out_dir, exist_ok=True)
        fig.savefig(os.path.join(out_dir, f"02_retention_heatmap.{fmt}"),
                    dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Heatmap saved to {figure_dir}/{{png,svg}}/")


def generate_reward_distributions(
    all_eval_data: Dict[str, List[Dict[str, Any]]],
    figure_dir: str,
    game_names: List[str],
) -> None:
    """Generate reward distribution box plots per game."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    AC_SERIES = ['#2563EB', '#D97706', '#059669', '#DC2626',
                 '#7C3AED', '#0891B2', '#BE185D', '#92400E']

    method_colors = {
        "Expert": AC_SERIES[0], "Multi-Task": AC_SERIES[1],
        "WHC": AC_SERIES[7], "Distillation": AC_SERIES[3],
        "Hybrid": AC_SERIES[2], "EWC": AC_SERIES[6],
        "Progress & Compress": AC_SERIES[5], "TRAC": AC_SERIES[4],
        "C-CHAIN": "#92400E",
    }

    method_order = ["Expert", "Multi-Task", "WHC", "Distillation", "Hybrid",
                    "EWC", "Progress & Compress", "TRAC", "C-CHAIN"]
    methods = [m for m in method_order if m in all_eval_data]

    n = len(game_names)
    fig, axes = plt.subplots(1, n, figsize=(max(4.5 * n, 12), 5.0), constrained_layout=True)
    if n == 1:
        axes = [axes]

    rng = np.random.default_rng(42)

    for j, game in enumerate(game_names):
        ax = axes[j]
        positions, box_data, colors_list, labels = [], [], [], []

        for i, method in enumerate(methods):
            evals = all_eval_data[method]
            eval_dict = {e["game_name"]: e for e in evals}
            if game not in eval_dict:
                continue
            entry = eval_dict[game]
            rewards = entry.get("all_rewards", [entry["mean_reward"]] * 30)
            positions.append(i)
            box_data.append(rewards)
            colors_list.append(method_colors.get(method, AC_SERIES[i % 8]))
            labels.append(method)

        if not box_data:
            continue

        bp = ax.boxplot(box_data, positions=positions, widths=0.45, patch_artist=True,
                        showfliers=False, medianprops=dict(color="#212529", linewidth=1.5),
                        whiskerprops=dict(color="#6C757D"), capprops=dict(color="#6C757D"),
                        boxprops=dict(linewidth=1.0), zorder=1)
        for patch, c in zip(bp["boxes"], colors_list):
            patch.set_facecolor(mpl.colors.to_rgba(c, alpha=0.25))
            patch.set_edgecolor(c)

        for k, (pos, rewards, c) in enumerate(zip(positions, box_data, colors_list)):
            jitter = rng.uniform(-0.15, 0.15, size=len(rewards))
            ax.scatter(pos + jitter, rewards, color=c, s=14, alpha=0.55,
                       edgecolors="none", zorder=2)

        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.set_title(game, fontweight="600")
        if j == 0:
            ax.set_ylabel("Episode Reward")

    fig.suptitle("Reward Distributions by Method", fontsize=14, fontweight="600",
                 y=1.03, fontfamily="serif")

    for fmt in ("png", "svg"):
        out_dir = os.path.join(figure_dir, fmt)
        os.makedirs(out_dir, exist_ok=True)
        fig.savefig(os.path.join(out_dir, f"04_reward_distributions.{fmt}"),
                    dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Distributions saved to {figure_dir}/{{png,svg}}/")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="CRL Classic Control Runner")
    parser.add_argument("--config", type=str, default="configs/classic_control.yaml")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--tag", type=str, default="default")
    parser.add_argument("--skip-experts", action="store_true",
                        help="Skip expert training (load from checkpoints)")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    if args.debug:
        config["debug"]["enabled"] = True
    config = apply_debug_overrides(config)

    # Setup
    device = get_device()
    set_seed(config["seed"])
    tag = args.tag

    task_sequence = config["task_sequence"]
    union_actions = compute_union_action_space_classic(task_sequence)
    max_state_dim = compute_max_state_dim(task_sequence)

    config["model"]["unified_action_dim"] = len(union_actions)
    config["max_state_dim"] = max_state_dim

    checkpoint_dir = config["logging"]["checkpoint_dir"]
    figure_dir = config["logging"]["figure_dir"]
    os.makedirs(os.path.join(checkpoint_dir, tag), exist_ok=True)
    os.makedirs(os.path.join(figure_dir, "png"), exist_ok=True)
    os.makedirs(os.path.join(figure_dir, "svg"), exist_ok=True)

    logger = setup_logger(
        log_dir=config["logging"]["log_dir"],
        experiment_name=f"classic_control_{tag}",
        use_tensorboard=config["logging"].get("use_tensorboard", False),
    )

    logger.info("=" * 60)
    logger.info("CRL Classic Control Experiment")
    logger.info("=" * 60)
    logger.info(f"Device: {device}")
    logger.info(f"Tasks: {task_sequence}")
    logger.info(f"Union actions: {union_actions} ({len(union_actions)} actions)")
    logger.info(f"Max state dim: {max_state_dim}")
    logger.info(f"Debug mode: {config['debug'].get('enabled', False)}")

    eval_episodes = config["evaluation"]["episodes"]
    game_names = [env_id.replace("-v1", "").replace("-v2", "").replace("-v3", "")
                  for env_id in task_sequence]

    # ══════════════════════════════════════════════════════════════════
    # STEP 1: Train Experts
    # ══════════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info("STEP 1: Training Experts")
    logger.info("=" * 60)

    expert_results = []
    if args.skip_experts:
        logger.info("Loading experts from checkpoints...")
        for env_id in task_sequence:
            game_name = env_id.replace("-v1", "").replace("-v2", "").replace("-v3", "")
            ckpt_path = os.path.join(checkpoint_dir, tag, f"expert_{game_name}_best.pt")
            model = build_model(config, device)
            model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))

            valid_actions = get_valid_actions_classic(env_id, union_actions)
            # Collect replay buffer
            replay_buffer = VectorReplayBuffer(
                config["training"]["buffer_size"], max_state_dim, device)
            env = make_classic_control_env(env_id, union_actions, max_state_dim,
                                           seed=config["seed"])
            state, _ = env.reset()
            for _ in range(min(config["training"]["buffer_size"], 5000)):
                action = np.random.choice(valid_actions)
                next_state, reward, terminated, truncated, _ = env.step(action)
                replay_buffer.push(state, action, reward, next_state, terminated or truncated)
                if terminated or truncated:
                    state, _ = env.reset()
                else:
                    state = next_state
            env.close()

            eval_r = evaluate_model(model, env_id, union_actions, max_state_dim,
                                    valid_actions, config, device, eval_episodes)
            expert_results.append({
                "policy_state_dict": copy.deepcopy(model.state_dict()),
                "valid_actions": valid_actions,
                "game_name": game_name,
                "env_id": env_id,
                "replay_buffer": replay_buffer,
                "best_reward": eval_r,
            })
            logger.info(f"  Loaded {game_name}: eval={eval_r:.2f}")
    else:
        for env_id in task_sequence:
            result = train_expert(
                config, env_id, union_actions, max_state_dim,
                device, logger, tag,
            )
            expert_results.append(result)

    # Expert evaluation
    expert_rewards = {}
    all_eval_data = {}
    expert_evals = []
    for result in expert_results:
        env_id = result["env_id"]
        game_name = result["game_name"]
        model = build_model(config, device)
        model.load_state_dict(result["policy_state_dict"])
        evals = evaluate_all_tasks(model, [env_id], union_actions, max_state_dim,
                                   config, device, eval_episodes)
        expert_rewards[game_name] = evals[0]["mean_reward"]
        expert_evals.append(evals[0])
        logger.info(f"  Expert {game_name}: {evals[0]['mean_reward']:.2f}")

    all_eval_data["Expert"] = expert_evals
    # Save expert eval JSONs
    for ev in expert_evals:
        path = os.path.join(figure_dir, f"eval_expert_{ev['game_name']}_{tag}.json")
        with open(path, "w") as f:
            json.dump([ev], f, indent=2, default=lambda x: x if not isinstance(x, np.floating) else float(x))

    # ══════════════════════════════════════════════════════════════════
    # STEP 2: Collect filtered states
    # ══════════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: Collecting filtered states")
    logger.info("=" * 60)

    filtered_states_list = []
    htcl_cfg = config.get("htcl", {})
    filtered_buffer_size = htcl_cfg.get("filtered_buffer_size", 5000)

    for result in expert_results:
        game_name = result["game_name"]
        model = build_model(config, device)
        model.load_state_dict(result["policy_state_dict"])
        filt = result["replay_buffer"].filter_by_confidence(
            model, result["valid_actions"], filtered_buffer_size)
        filtered_states_list.append(filt)
        logger.info(f"  {game_name}: {filt.shape[0]} high-confidence states")

    # ══════════════════════════════════════════════════════════════════
    # STEP 3: Run Consolidation Methods
    # ══════════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: Running Consolidation Methods")
    logger.info("=" * 60)

    ckpt_dir = os.path.join(checkpoint_dir, tag)

    # ── 3a: WHC ──
    logger.info("\n--- WHC ---")
    global_model = build_model(config, device)
    whc = WHCConsolidator(config, device=device, logger=logger)
    whc_model = whc.consolidate(global_model, expert_results, filtered_states_list)
    torch.save(whc_model.state_dict(), os.path.join(ckpt_dir, "consolidated_whc.pt"))

    whc_evals = evaluate_all_tasks(whc_model, task_sequence, union_actions,
                                   max_state_dim, config, device, eval_episodes)
    all_eval_data["WHC"] = whc_evals
    for ev in whc_evals:
        logger.info(f"  WHC {ev['game_name']}: {ev['mean_reward']:.2f}")

    # ── 3b: Distillation ──
    logger.info("\n--- Distillation ---")
    global_model = build_model(config, device)
    distiller = DistillationConsolidator(config, device=device, logger=logger)
    dist_model = distiller.consolidate(global_model, expert_results)
    torch.save(dist_model.state_dict(), os.path.join(ckpt_dir, "consolidated_distillation.pt"))

    dist_evals = evaluate_all_tasks(dist_model, task_sequence, union_actions,
                                    max_state_dim, config, device, eval_episodes)
    all_eval_data["Distillation"] = dist_evals
    for ev in dist_evals:
        logger.info(f"  Distillation {ev['game_name']}: {ev['mean_reward']:.2f}")

    # ── 3c: Hybrid ──
    logger.info("\n--- Hybrid ---")
    # Initialize to ensemble mean
    global_model = build_model(config, device)
    mean_sd = {}
    for key in expert_results[0]["policy_state_dict"]:
        mean_sd[key] = torch.stack(
            [r["policy_state_dict"][key].float() for r in expert_results]
        ).mean(dim=0)
    global_model.load_state_dict(mean_sd)

    hybrid = HybridConsolidator(config, device=device, logger=logger)
    hybrid_model = hybrid.consolidate(global_model, expert_results, filtered_states_list)
    torch.save(hybrid_model.state_dict(), os.path.join(ckpt_dir, "consolidated_hybrid.pt"))

    hybrid_evals = evaluate_all_tasks(hybrid_model, task_sequence, union_actions,
                                      max_state_dim, config, device, eval_episodes)
    all_eval_data["Hybrid"] = hybrid_evals
    for ev in hybrid_evals:
        logger.info(f"  Hybrid {ev['game_name']}: {ev['mean_reward']:.2f}")

    # ══════════════════════════════════════════════════════════════════
    # STEP 4: Run Sequential Methods
    # ══════════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info("STEP 4: Running Sequential Methods")
    logger.info("=" * 60)

    first_expert_sd = expert_results[0]["policy_state_dict"]
    first_expert_ckpt = os.path.join(ckpt_dir, f"expert_{game_names[0]}_best.pt")

    # ── 4a: EWC ──
    logger.info("\n--- EWC ---")
    ewc_model = train_ewc_classic(
        config, task_sequence, union_actions, max_state_dim,
        first_expert_sd, device, logger, tag,
    )
    ewc_evals = evaluate_all_tasks(ewc_model, task_sequence, union_actions,
                                   max_state_dim, config, device, eval_episodes)
    all_eval_data["EWC"] = ewc_evals
    for ev in ewc_evals:
        logger.info(f"  EWC {ev['game_name']}: {ev['mean_reward']:.2f}")

    # ── 4b: Progress & Compress ──
    logger.info("\n--- Progress & Compress ---")
    pc_model = train_progress_compress_classic(
        config, task_sequence, union_actions, max_state_dim,
        device, logger, tag,
    )
    pc_evals = evaluate_all_tasks(pc_model, task_sequence, union_actions,
                                  max_state_dim, config, device, eval_episodes)
    all_eval_data["Progress & Compress"] = pc_evals
    for ev in pc_evals:
        logger.info(f"  P&C {ev['game_name']}: {ev['mean_reward']:.2f}")

    # ── 4c: TRAC ──
    logger.info("\n--- TRAC ---")
    trac_trainer = TRACTrainer(config, union_actions, device, logger, tag)
    trac_model = trac_trainer.train_sequential(task_sequence, first_expert_ckpt)
    trac_evals = evaluate_all_tasks(trac_model, task_sequence, union_actions,
                                    max_state_dim, config, device, eval_episodes)
    all_eval_data["TRAC"] = trac_evals
    for ev in trac_evals:
        logger.info(f"  TRAC {ev['game_name']}: {ev['mean_reward']:.2f}")

    # ── 4d: C-CHAIN ──
    logger.info("\n--- C-CHAIN ---")
    cchain_trainer = CChainTrainer(config, union_actions, device, logger, tag)
    cchain_model = cchain_trainer.train_sequential(task_sequence, first_expert_ckpt)
    cchain_evals = evaluate_all_tasks(cchain_model, task_sequence, union_actions,
                                      max_state_dim, config, device, eval_episodes)
    all_eval_data["C-CHAIN"] = cchain_evals
    for ev in cchain_evals:
        logger.info(f"  C-CHAIN {ev['game_name']}: {ev['mean_reward']:.2f}")

    # ══════════════════════════════════════════════════════════════════
    # STEP 5: Multi-Task
    # ══════════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info("STEP 5: Multi-Task Joint Training")
    logger.info("=" * 60)

    mt_model = train_multitask_classic(
        config, task_sequence, union_actions, max_state_dim,
        device, logger, tag,
    )
    mt_evals = evaluate_all_tasks(mt_model, task_sequence, union_actions,
                                  max_state_dim, config, device, eval_episodes)
    all_eval_data["Multi-Task"] = mt_evals
    for ev in mt_evals:
        logger.info(f"  Multi-Task {ev['game_name']}: {ev['mean_reward']:.2f}")

    # ══════════════════════════════════════════════════════════════════
    # STEP 6: Save eval JSONs and generate figures
    # ══════════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info("STEP 6: Generating Figures")
    logger.info("=" * 60)

    # Save all eval data
    for method, evals in all_eval_data.items():
        if method == "Expert":
            continue
        fname_map = {
            "WHC": "whc", "Distillation": "distillation", "Hybrid": "hybrid",
            "EWC": "ewc", "Progress & Compress": "pc", "TRAC": "trac",
            "C-CHAIN": "cchain", "Multi-Task": "multitask",
        }
        fname = fname_map.get(method, method.lower().replace(" ", "_"))
        path = os.path.join(figure_dir, f"eval_{fname}_{tag}.json")

        # Convert numpy types for JSON serialization
        serializable = []
        for ev in evals:
            s_ev = {}
            for k, v in ev.items():
                if isinstance(v, (np.floating, np.integer)):
                    s_ev[k] = float(v)
                elif isinstance(v, list):
                    s_ev[k] = [float(x) if isinstance(x, (np.floating, np.integer)) else x for x in v]
                else:
                    s_ev[k] = v
            serializable.append(s_ev)
        with open(path, "w") as f:
            json.dump(serializable, f, indent=2)

    # Generate heatmap
    generate_heatmap(all_eval_data, expert_rewards, figure_dir, game_names)
    generate_reward_distributions(all_eval_data, figure_dir, game_names)

    # ══════════════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)

    header = f"{'Method':<22}"
    for g in game_names:
        header += f" {g:>12}"
    header += f" {'Avg':>8}"
    logger.info(header)
    logger.info("-" * len(header))

    for method in ["Expert", "Multi-Task", "WHC", "Distillation", "Hybrid",
                    "EWC", "Progress & Compress", "TRAC", "C-CHAIN"]:
        if method not in all_eval_data:
            continue
        evals = all_eval_data[method]
        eval_dict = {e["game_name"]: e for e in evals}
        row = f"{method:<22}"
        vals = []
        for g in game_names:
            if g in eval_dict:
                v = eval_dict[g]["mean_reward"]
                row += f" {v:>12.1f}"
                vals.append(v)
            else:
                row += f" {'N/A':>12}"
        if vals:
            row += f" {np.mean(vals):>8.1f}"
        logger.info(row)

    logger.info(f"\nAll results saved to {figure_dir}/")
    logger.info("Done.")
    logger.close()


if __name__ == "__main__":
    main()
