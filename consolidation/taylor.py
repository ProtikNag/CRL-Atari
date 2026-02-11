"""
HTCL Taylor-series consolidation adapted for DQN.

Implements Eq. (6) from the paper:
    w_g^(t) = w_g^(t-1) + (H + λI)^{-1} [λ(w_l - w_g) - g]

The gradient g and diagonal Hessian H are computed using the DQN
temporal-difference loss instead of cross-entropy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy
from typing import Dict, Optional, Tuple


def dqn_loss_on_batch(
    model: nn.Module,
    states: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    next_states: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Compute DQN TD loss (Huber) on a single batch."""
    q_values = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        next_q = model(next_states).max(1)[0]
        targets = rewards + gamma * (1.0 - dones) * next_q
    return F.smooth_l1_loss(q_values, targets)


def estimate_dqn_hessian(
    model: nn.Module,
    transitions: Dict[str, np.ndarray],
    device: torch.device,
    gamma: float = 0.99,
    batch_size: int = 256,
) -> Dict[str, torch.Tensor]:
    """
    Estimate diagonal Hessian via Fisher approximation (squared gradients)
    using DQN TD loss.
    """
    model.eval()
    hessian_diag = {
        name: torch.zeros_like(param, device=device)
        for name, param in model.named_parameters()
        if param.requires_grad
    }

    states = torch.from_numpy(transitions["states"]).to(device)
    actions = torch.from_numpy(transitions["actions"]).long().to(device)
    rewards = torch.from_numpy(transitions["rewards"]).to(device)
    next_states = torch.from_numpy(transitions["next_states"]).to(device)
    dones = torch.from_numpy(transitions["dones"]).to(device)

    n_samples = len(states)
    n_batches = 0

    for i in range(0, n_samples, batch_size):
        j = min(i + batch_size, n_samples)
        model.zero_grad()

        loss = dqn_loss_on_batch(
            model,
            states[i:j], actions[i:j], rewards[i:j],
            next_states[i:j], dones[i:j], gamma,
        )
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                hessian_diag[name] += param.grad.detach().pow(2)
        n_batches += 1

    for name in hessian_diag:
        hessian_diag[name] /= max(n_batches, 1)

    return hessian_diag


def compute_gradients(
    model: nn.Module,
    transitions: Dict[str, np.ndarray],
    device: torch.device,
    gamma: float = 0.99,
    batch_size: int = 256,
) -> Dict[str, torch.Tensor]:
    """Compute mean gradient of DQN loss over transition buffer."""
    model.train()
    grads = {
        name: torch.zeros_like(param, device=device)
        for name, param in model.named_parameters()
        if param.requires_grad
    }

    states = torch.from_numpy(transitions["states"]).to(device)
    actions = torch.from_numpy(transitions["actions"]).long().to(device)
    rewards = torch.from_numpy(transitions["rewards"]).to(device)
    next_states = torch.from_numpy(transitions["next_states"]).to(device)
    dones = torch.from_numpy(transitions["dones"]).to(device)

    n_samples = len(states)
    n_batches = 0

    for i in range(0, n_samples, batch_size):
        j = min(i + batch_size, n_samples)
        model.zero_grad()

        loss = dqn_loss_on_batch(
            model,
            states[i:j], actions[i:j], rewards[i:j],
            next_states[i:j], dones[i:j], gamma,
        )
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grads[name] += param.grad.detach()
        n_batches += 1

    for name in grads:
        grads[name] /= max(n_batches, 1)

    return grads


def taylor_update(
    global_model: nn.Module,
    local_model: nn.Module,
    transitions: Dict[str, np.ndarray],
    device: torch.device,
    gamma: float = 0.99,
    eta: float = 0.9,
    max_norm: float = 1.0,
    lambda_reg: Optional[float] = None,
    verbose: bool = False,
) -> nn.Module:
    """
    Update global model toward local model using second-order Taylor expansion.

    Implements:
        Δw* = (H + λI)^{-1} [λ(w_local - w_global) - g]
        w_global ← w_global + η · Δw*

    Args:
        global_model: Conservative global model to update
        local_model: Locally trained model (target)
        transitions: Dict of numpy arrays from consolidation buffer
        device: Torch device
        gamma: DQN discount factor
        eta: Step size for the Taylor update
        max_norm: Maximum norm for clipping the update vector
        lambda_reg: Regularization strength; auto-computed if None
        verbose: Print debug information

    Returns:
        Updated global model (modified in-place)
    """
    grads = compute_gradients(global_model, transitions, device, gamma)
    hessians = estimate_dqn_hessian(global_model, transitions, device, gamma)

    if lambda_reg is None:
        diag_entries = torch.cat([v.flatten() for v in hessians.values()])
        min_eig = float(diag_entries.min().item())
        max_eig = float(diag_entries.max().item())
        if not np.isfinite(max_eig) or max_eig <= 0:
            lambda_reg = max(1e-6, 10.0 * (abs(min_eig) + 1e-6))
        else:
            lambda_reg = 1000.0 * max_eig

    if verbose:
        print(f"    Taylor update: lambda_reg={lambda_reg:.4f}, eta={eta}")

    local_state = local_model.state_dict()

    with torch.no_grad():
        for name, param in global_model.named_parameters():
            if not param.requires_grad or name not in grads:
                continue

            h_diag = hessians[name].to(device)
            local_param = local_state[name].to(device)

            denom = h_diag + lambda_reg + 1e-8
            h_inv = 1.0 / denom

            delta_d = local_param - param
            raw_delta = h_inv * (lambda_reg * delta_d - grads[name])
            delta = eta * raw_delta

            delta_norm = delta.norm().item()
            if delta_norm > max_norm:
                delta = delta * (max_norm / (delta_norm + 1e-12))

            if torch.isnan(delta).any() or torch.isinf(delta).any():
                if verbose:
                    print(f"    Warning: NaN/Inf in delta for {name}, skipping")
                continue

            param.add_(delta)

    return global_model


def global_catchup(
    global_model: nn.Module,
    local_model: nn.Module,
    transitions: Dict[str, np.ndarray],
    device: torch.device,
    num_iterations: int = 5,
    gamma: float = 0.99,
    eta: float = 0.9,
    max_norm: float = 1.0,
    lambda_reg: float = 10000.0,
    catchup_lr: float = 0.001,
    patience: int = 2,
    verbose: bool = False,
) -> nn.Module:
    """
    Catch-up phase: iteratively pull global model toward local using
    alternating SGD fine-tuning and Taylor updates.

    Includes early stopping: if the loss increases for `patience`
    consecutive iterations, the catch-up halts and reverts to
    the best checkpoint seen so far.

    Args:
        patience: Number of consecutive loss increases before stopping.
    """
    if num_iterations <= 0:
        return global_model

    states = torch.from_numpy(transitions["states"]).to(device)
    actions = torch.from_numpy(transitions["actions"]).long().to(device)
    rewards = torch.from_numpy(transitions["rewards"]).to(device)
    next_states = torch.from_numpy(transitions["next_states"]).to(device)
    dones = torch.from_numpy(transitions["dones"]).to(device)

    # Measure initial loss as baseline
    global_model.eval()
    with torch.no_grad():
        eval_n = min(512, len(states))
        initial_loss = dqn_loss_on_batch(
            global_model, states[:eval_n], actions[:eval_n], rewards[:eval_n],
            next_states[:eval_n], dones[:eval_n], gamma,
        ).item()

    best_loss = initial_loss
    best_state = copy.deepcopy(global_model.state_dict())
    no_improve_count = 0

    if verbose:
        print(f"    Catchup initial loss: {initial_loss:.4f}")

    for iteration in range(num_iterations):
        pre_state = copy.deepcopy(global_model.state_dict())

        # SGD fine-tuning step on a temporary copy
        temp_local = copy.deepcopy(global_model).to(device)
        temp_opt = optim.Adam(temp_local.parameters(), lr=catchup_lr)
        temp_local.train()
        batch_size = 256
        for i in range(0, len(states), batch_size):
            j = min(i + batch_size, len(states))
            temp_opt.zero_grad()
            loss = dqn_loss_on_batch(
                temp_local,
                states[i:j], actions[i:j], rewards[i:j],
                next_states[i:j], dones[i:j], gamma,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(temp_local.parameters(), 10.0)
            temp_opt.step()

        # Conservative Taylor update toward the fine-tuned model
        taylor_update(
            global_model, temp_local, transitions, device,
            gamma=gamma, eta=eta, max_norm=max_norm,
            lambda_reg=lambda_reg, verbose=False,
        )
        del temp_local

        # Evaluate current loss
        global_model.eval()
        with torch.no_grad():
            current_loss = dqn_loss_on_batch(
                global_model, states[:eval_n], actions[:eval_n], rewards[:eval_n],
                next_states[:eval_n], dones[:eval_n], gamma,
            ).item()

        if verbose:
            status = "improved" if current_loss < best_loss else "worse"
            print(f"    Catchup {iteration+1}/{num_iterations}: "
                  f"loss={current_loss:.4f} ({status})")

        if current_loss < best_loss:
            best_loss = current_loss
            best_state = copy.deepcopy(global_model.state_dict())
            no_improve_count = 0
        else:
            no_improve_count += 1
            # Revert to pre-step state to avoid compounding bad updates
            global_model.load_state_dict(pre_state)

            if no_improve_count >= patience:
                if verbose:
                    print(f"    Early stopping at iteration {iteration+1} "
                          f"(no improvement for {patience} steps)")
                break

    global_model.load_state_dict(best_state)
    return global_model


class ConsolidationBuffer:
    """
    Accumulates transitions from all seen games for use in Taylor updates.

    Each game contributes up to `max_per_game` transitions. When computing
    gradients and Hessians, all stored transitions are combined.
    """

    def __init__(self, max_per_game: int = 5000):
        self.max_per_game = max_per_game
        self.game_data: Dict[str, Dict[str, np.ndarray]] = {}

    def add_game(self, game_id: str, replay_buffer) -> None:
        """Sample and store transitions from a game's replay buffer."""
        self.game_data[game_id] = replay_buffer.sample_all(self.max_per_game)

    def get_combined(self) -> Dict[str, np.ndarray]:
        """Concatenate all stored transitions into a single dict."""
        if not self.game_data:
            raise ValueError("No game data in consolidation buffer")

        combined = {}
        for key in ["states", "actions", "rewards", "next_states", "dones"]:
            arrays = [data[key] for data in self.game_data.values()]
            combined[key] = np.concatenate(arrays, axis=0)

        indices = np.random.permutation(len(combined["states"]))
        for key in combined:
            combined[key] = combined[key][indices]

        return combined

    @property
    def num_games(self) -> int:
        return len(self.game_data)

    @property
    def total_transitions(self) -> int:
        return sum(len(d["states"]) for d in self.game_data.values())
