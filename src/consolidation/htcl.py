"""
Hierarchical Taylor-based Continual Learning (HTCL) Consolidation.

Implements the second-order Taylor expansion consolidation mechanism from:
"Mitigating Task-Order Sensitivity and Forgetting via Hierarchical Second-Order
Consolidation" (Nag, Raghavan, Narayanan, 2026).

Key properties:
  - Local models initialized from the global (hierarchical) model
  - Diagonal Fisher approximation of the Hessian
  - Lambda must satisfy: lambda > -mu_min(H) for positive definiteness
  - Catch-up iterations refine the hierarchical model after each merge
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import os
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

from src.models.dqn import DQNNetwork
from src.data.replay_buffer import ReplayBuffer
from src.utils.logger import Logger


class HTCLConsolidator:
    """HTCL consolidation for merging expert models into a global model.

    Uses a second-order Taylor expansion around the current global parameters
    to find an optimal update direction that balances stability (preserving
    past knowledge, captured by the Fisher/Hessian) with plasticity (absorbing
    new expert knowledge).

    The closed-form update is:
        w_global^(t) = w_global^(t-1) + (H + lambda * I)^{-1} [lambda * delta_d - g]

    where:
        delta_d = w_local^(t) - w_global^(t-1)  (expert drift)
        g = gradient of past loss at w_global     (gradient on old tasks)
        H = diagonal Fisher approximation          (Hessian on old tasks)

    Args:
        config: Configuration dictionary.
        device: Torch device string.
        logger: Logger instance.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        device: str = "cpu",
        logger: Optional[Logger] = None,
    ):
        self.config = config
        self.device = device
        self.logger = logger
        self.htcl_cfg = config["htcl"]

        self.lambda_htcl = self.htcl_cfg["lambda_htcl"]
        self.lambda_auto = self.htcl_cfg.get("lambda_auto", True)
        self.lambda_margin = self.htcl_cfg.get("lambda_margin", 0.1)
        self.fisher_samples = self.htcl_cfg["fisher_samples"]
        self.catch_up_iterations = self.htcl_cfg["catch_up_iterations"]
        self.eta = self.htcl_cfg.get("eta", 0.9)
        self.diagonal_fisher = self.htcl_cfg.get("diagonal_fisher", True)

        # Storage for cumulative Fisher (diagonal) across all registered tasks
        self.cumulative_fisher: Optional[Dict[str, torch.Tensor]] = None
        # Cumulative gradient at global params
        self.cumulative_gradient: Optional[Dict[str, torch.Tensor]] = None
        # Number of tasks registered
        self.num_registered_tasks = 0

    def compute_diagonal_fisher(
        self,
        model: DQNNetwork,
        replay_buffer: ReplayBuffer,
        valid_actions: List[int],
        num_samples: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute diagonal Fisher Information Matrix.

        Uses the empirical Fisher: E[grad(log pi(a|s))^2] as a diagonal
        approximation of the Hessian.

        Args:
            model: Network to compute Fisher for.
            replay_buffer: Replay data.
            valid_actions: Valid action indices.
            num_samples: Number of samples.

        Returns:
            Dictionary of parameter name -> diagonal Fisher tensor.
        """
        if num_samples is None:
            num_samples = self.fisher_samples

        model.eval()
        fisher = {
            name: torch.zeros_like(param)
            for name, param in model.named_parameters()
            if param.requires_grad
        }

        states = replay_buffer.sample_states(num_samples)
        batch_size = 64

        for i in range(0, len(states), batch_size):
            batch = states[i : i + batch_size]
            model.zero_grad()

            q_values = model(batch)
            mask = torch.full(
                (model.unified_action_dim,), float("-inf"), device=self.device
            )
            mask[valid_actions] = 0.0
            masked_q = q_values + mask.unsqueeze(0)

            # Sample actions from softmax policy (not argmax) for proper
            # empirical Fisher. Argmax yields near-zero gradients when one
            # Q-value dominates, causing Fisher diagonal entries to be 0.
            probs = F.softmax(masked_q, dim=1)
            sampled_actions = torch.multinomial(probs, num_samples=1).squeeze(1)

            log_probs = F.log_softmax(masked_q, dim=1)
            selected_log_probs = log_probs.gather(1, sampled_actions.unsqueeze(1)).squeeze(1)

            # Compute Fisher per sample to avoid gradient cancellation
            for j in range(len(batch)):
                model.zero_grad()
                selected_log_probs[j].backward(retain_graph=(j < len(batch) - 1))
                for name, param in model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        fisher[name] += (param.grad.data ** 2) / num_samples

        model.train()
        return fisher

    def compute_gradient(
        self,
        model: DQNNetwork,
        replay_buffer: ReplayBuffer,
        valid_actions: List[int],
        num_samples: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute gradient of loss at current model parameters.

        This represents g^{(t-1)} = nabla J(w_global^{(t-1)}) evaluated
        on replay data from past tasks.

        Args:
            model: Global model.
            replay_buffer: Replay buffer with past task data.
            valid_actions: Valid actions.
            num_samples: Number of samples.

        Returns:
            Dictionary of parameter name -> gradient tensor.
        """
        if num_samples is None:
            num_samples = self.fisher_samples

        model.eval()
        gradient = {
            name: torch.zeros_like(param)
            for name, param in model.named_parameters()
            if param.requires_grad
        }

        states = replay_buffer.sample_states(num_samples)
        batch_size = 64

        for i in range(0, len(states), batch_size):
            batch = states[i : i + batch_size]
            model.zero_grad()

            q_values = model(batch)
            mask = torch.full(
                (model.unified_action_dim,), float("-inf"), device=self.device
            )
            mask[valid_actions] = 0.0
            masked_q = q_values + mask.unsqueeze(0)

            # Negative log-likelihood loss
            log_probs = F.log_softmax(masked_q, dim=1)
            actions = masked_q.argmax(dim=1)
            loss = -log_probs.gather(1, actions.unsqueeze(1)).squeeze(1).mean()
            loss.backward()

            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    gradient[name] += param.grad.data * (len(batch) / num_samples)

        model.train()
        return gradient

    def _ensure_lambda_constraint(
        self, fisher: Dict[str, torch.Tensor]
    ) -> float:
        """Ensure lambda > -mu_min(H) for positive definiteness.

        For diagonal Fisher, mu_min is simply the minimum diagonal entry.
        We set lambda = max(self.lambda_htcl, -mu_min + margin).

        Args:
            fisher: Diagonal Fisher dictionary.

        Returns:
            Valid lambda value satisfying the constraint.
        """
        # Find minimum eigenvalue (min diagonal entry for diagonal Fisher)
        mu_min = float("inf")
        for name, f_diag in fisher.items():
            local_min = f_diag.min().item()
            if local_min < mu_min:
                mu_min = local_min

        # Constraint: lambda > -mu_min
        # Since Fisher is PSD, mu_min >= 0, so lambda > 0 always works.
        # But after accumulation with gradients, numerical issues can arise.
        lambda_lower_bound = -mu_min + self.lambda_margin

        if self.lambda_auto:
            effective_lambda = max(self.lambda_htcl, lambda_lower_bound)
        else:
            effective_lambda = self.lambda_htcl
            if effective_lambda <= -mu_min:
                if self.logger:
                    self.logger.warning(
                        f"HTCL: lambda={effective_lambda:.4f} violates constraint "
                        f"lambda > -mu_min(H) = {-mu_min:.4f}. "
                        f"Consider enabling lambda_auto or increasing lambda."
                    )

        if self.logger:
            self.logger.info(
                f"HTCL: mu_min(H) = {mu_min:.6f} | "
                f"Constraint: lambda > {-mu_min:.6f} | "
                f"Using lambda = {effective_lambda:.6f}"
            )

        return effective_lambda

    def _taylor_update(
        self,
        global_state_dict: Dict[str, torch.Tensor],
        local_state_dict: Dict[str, torch.Tensor],
        fisher: Dict[str, torch.Tensor],
        gradient: Dict[str, torch.Tensor],
        lambda_val: float,
    ) -> Dict[str, torch.Tensor]:
        """Compute the HTCL closed-form parameter update.

        w_new = w_global + (H + lambda*I)^{-1} * [lambda * delta_d - g]

        For diagonal Fisher, the inverse is trivially element-wise 1/(H_ii + lambda).

        Args:
            global_state_dict: Current global model parameters.
            local_state_dict: Expert (local) model parameters.
            fisher: Diagonal Fisher / Hessian approximation.
            gradient: Gradient of past loss at global params.
            lambda_val: Validated lambda value.

        Returns:
            Updated state dict.
        """
        updated = {}
        for name in global_state_dict:
            if name in fisher:
                w_global = global_state_dict[name].to(self.device)
                w_local = local_state_dict[name].to(self.device)
                h_diag = fisher[name].to(self.device)
                g = gradient[name].to(self.device)

                # delta_d = w_local - w_global
                delta_d = w_local - w_global

                # Numerator: lambda * delta_d - g
                numerator = lambda_val * delta_d - g

                # Denominator: H_diag + lambda (element-wise for diagonal)
                denominator = h_diag + lambda_val

                # Update: w_new = w_global + (H + lambda*I)^{-1} * numerator
                update = numerator / (denominator + 1e-8)

                # Scale by eta (step size)
                updated[name] = w_global + self.eta * update
            else:
                # Non-trainable params (e.g., BN stats): average
                updated[name] = (
                    global_state_dict[name].to(self.device)
                    + local_state_dict[name].to(self.device)
                ) / 2.0

        return updated

    def consolidate(
        self,
        global_model: DQNNetwork,
        expert_results: List[Dict[str, Any]],
    ) -> DQNNetwork:
        """Consolidate expert models into global model via HTCL.

        For each expert (task), applies the second-order Taylor update to
        merge expert knowledge into the global model, then performs catch-up
        iterations.

        Args:
            global_model: The current global (hierarchical) model.
            expert_results: List of expert training results.

        Returns:
            The consolidated global model.
        """
        if self.logger:
            self.logger.info("Starting HTCL consolidation...")

        consolidated = copy.deepcopy(global_model).to(self.device)
        global_sd = {
            name: param.clone()
            for name, param in consolidated.state_dict().items()
        }

        for task_idx, result in enumerate(expert_results):
            game_name = result["game_name"]
            local_sd = result["policy_state_dict"]
            valid_actions = result["valid_actions"]
            replay_buffer = result["replay_buffer"]

            if self.logger:
                self.logger.info(
                    f"HTCL: Consolidating task {task_idx + 1}/{len(expert_results)} "
                    f"({game_name})..."
                )

            # Compute Fisher (Hessian approximation) at current global params
            # We need Fisher from ALL previously seen tasks
            consolidated.load_state_dict(global_sd)
            task_fisher = self.compute_diagonal_fisher(
                consolidated, replay_buffer, valid_actions
            )

            # Accumulate Fisher across tasks
            if self.cumulative_fisher is None:
                self.cumulative_fisher = {
                    name: f.clone() for name, f in task_fisher.items()
                }
            else:
                for name in task_fisher:
                    self.cumulative_fisher[name] += task_fisher[name]

            # Compute gradient at global params on current task data
            task_gradient = self.compute_gradient(
                consolidated, replay_buffer, valid_actions
            )

            # Accumulate gradient
            if self.cumulative_gradient is None:
                self.cumulative_gradient = {
                    name: g.clone() for name, g in task_gradient.items()
                }
            else:
                for name in task_gradient:
                    self.cumulative_gradient[name] += task_gradient[name]

            # Ensure lambda constraint: lambda > -mu_min(H)
            effective_lambda = self._ensure_lambda_constraint(self.cumulative_fisher)

            # Log lambda to TensorBoard
            if self.logger:
                self.logger.info(
                    f"HTCL: Task {task_idx + 1} ({game_name}) | "
                    f"effective_lambda = {effective_lambda:.2f}"
                )
                self.logger.log_scalar(
                    "htcl/effective_lambda", effective_lambda, task_idx + 1
                )
                self.logger.log_scalar(
                    "htcl/num_tasks_consolidated", task_idx + 1, task_idx + 1
                )

            # Apply Taylor update
            updated_sd = self._taylor_update(
                global_sd,
                local_sd,
                self.cumulative_fisher,
                self.cumulative_gradient,
                effective_lambda,
            )

            # Log parameter drift and update magnitude
            if self.logger:
                drift_norm = sum(
                    (local_sd[n].to(self.device) - global_sd[n].to(self.device)).norm().item() ** 2
                    for n in self.cumulative_fisher
                ) ** 0.5
                update_norm = sum(
                    (updated_sd[n] - global_sd[n].to(self.device)).norm().item() ** 2
                    for n in self.cumulative_fisher
                ) ** 0.5
                self.logger.info(
                    f"HTCL: {game_name} | expert_drift_norm = {drift_norm:.4f} | "
                    f"update_norm = {update_norm:.4f}"
                )
                self.logger.log_scalar(
                    f"htcl/{game_name}/expert_drift_norm", drift_norm, task_idx + 1
                )
                self.logger.log_scalar(
                    f"htcl/{game_name}/update_norm", update_norm, task_idx + 1
                )

            # Update global state dict
            global_sd = updated_sd

            # Catch-up iterations: refine the global model
            for catch_iter in range(self.catch_up_iterations):
                # Recompute gradient at the new position
                consolidated.load_state_dict(global_sd)
                refined_gradient = self.compute_gradient(
                    consolidated, replay_buffer, valid_actions
                )

                # Apply another Taylor update with same Fisher but new gradient
                global_sd = self._taylor_update(
                    global_sd,
                    local_sd,
                    self.cumulative_fisher,
                    refined_gradient,
                    effective_lambda,
                )

                if self.logger:
                    self.logger.info(
                        f"HTCL: Catch-up iteration {catch_iter + 1}/"
                        f"{self.catch_up_iterations} for {game_name}"
                    )

            self.num_registered_tasks += 1

        # Load final consolidated weights
        consolidated.load_state_dict(global_sd)
        consolidated.eval()

        if self.logger:
            self.logger.info("HTCL consolidation complete.")
        return consolidated

    def get_global_weights(self, model: DQNNetwork) -> dict:
        """Get a copy of the global model weights for local initialization.

        This implements the key HTCL property: local models are initialized
        from the current global state to prevent excessive drift.

        Args:
            model: The current global model.

        Returns:
            Copy of the model's state dict.
        """
        return copy.deepcopy(model.state_dict())

    def save(self, path: str) -> None:
        """Save HTCL state."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "cumulative_fisher": self.cumulative_fisher,
                "cumulative_gradient": self.cumulative_gradient,
                "num_registered_tasks": self.num_registered_tasks,
                "lambda_htcl": self.lambda_htcl,
            },
            path,
        )

    def load(self, path: str) -> None:
        """Load HTCL state."""
        data = torch.load(path, map_location=self.device, weights_only=False)
        self.cumulative_fisher = data["cumulative_fisher"]
        self.cumulative_gradient = data["cumulative_gradient"]
        self.num_registered_tasks = data["num_registered_tasks"]
        self.lambda_htcl = data["lambda_htcl"]
