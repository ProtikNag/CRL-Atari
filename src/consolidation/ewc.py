"""
Elastic Weight Consolidation (EWC) for continual RL.

Computes a diagonal Fisher Information Matrix for each task's expert,
then merges expert weights into a global model by penalizing movement
away from important parameters of previously learned tasks.

Reference: Kirkpatrick et al., "Overcoming catastrophic forgetting in
neural networks" (PNAS 2017).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import os
from typing import Dict, Any, List, Optional

from src.models.dqn import DQNNetwork
from src.data.replay_buffer import ReplayBuffer
from src.utils.logger import Logger


class EWCConsolidator:
    """EWC-based consolidation for merging expert models.

    Maintains a diagonal Fisher Information Matrix for each task. The
    consolidated model is obtained by fine-tuning toward each expert's
    weights while penalizing deviation from important parameters of
    previous tasks.

    Supports online EWC (running Fisher with decay) for efficiency.

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
        self.ewc_cfg = config["ewc"]

        self.lambda_ewc = self.ewc_cfg["lambda_ewc"]
        self.fisher_samples = self.ewc_cfg["fisher_samples"]
        self.online = self.ewc_cfg.get("online", True)
        self.gamma_ewc = self.ewc_cfg.get("gamma_ewc", 0.95)

        # Storage for per-task Fisher and optimal params
        self.fisher_matrices: List[Dict[str, torch.Tensor]] = []
        self.optimal_params: List[Dict[str, torch.Tensor]] = []

        # For online EWC: cumulative Fisher
        self.cumulative_fisher: Optional[Dict[str, torch.Tensor]] = None

    def compute_fisher(
        self,
        model: DQNNetwork,
        replay_buffer: ReplayBuffer,
        valid_actions: List[int],
        num_samples: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute diagonal Fisher Information Matrix from replay data.

        Uses the empirical Fisher approximation: average of squared gradients
        of the log-likelihood (Q-value for taken actions).

        Args:
            model: Trained expert network.
            replay_buffer: Replay buffer with task data.
            valid_actions: Valid actions for this task.
            num_samples: Number of samples to use.

        Returns:
            Dictionary mapping parameter names to diagonal Fisher values.
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
            # Mask to valid actions and sample from softmax (empirical Fisher)
            mask = torch.full(
                (model.unified_action_dim,), float("-inf"), device=self.device
            )
            mask[valid_actions] = 0.0
            masked_q = q_values + mask.unsqueeze(0)
            log_probs = F.log_softmax(masked_q, dim=1)

            # Use max Q action as the "label"
            actions = masked_q.argmax(dim=1)
            selected_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
            loss = selected_log_probs.mean()
            loss.backward()

            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += (param.grad.data ** 2) * (len(batch) / num_samples)

        model.train()
        return fisher

    def register_task(
        self,
        model: DQNNetwork,
        replay_buffer: ReplayBuffer,
        valid_actions: List[int],
    ) -> None:
        """Register a completed task: compute and store Fisher + optimal params.

        Args:
            model: The trained expert model.
            replay_buffer: Replay buffer from training.
            valid_actions: Valid actions for this task.
        """
        if self.logger:
            self.logger.info("Computing Fisher Information Matrix for EWC...")

        fisher = self.compute_fisher(model, replay_buffer, valid_actions)

        if self.online and self.cumulative_fisher is not None:
            # Online EWC: decay old Fisher and add new
            for name in fisher:
                self.cumulative_fisher[name] = (
                    self.gamma_ewc * self.cumulative_fisher[name] + fisher[name]
                )
        elif self.online:
            self.cumulative_fisher = fisher
        else:
            self.fisher_matrices.append(fisher)

        # Store optimal parameters
        optimal = {
            name: param.data.clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        self.optimal_params.append(optimal)

        if self.logger:
            self.logger.info(
                f"EWC: Registered task {len(self.optimal_params)}. "
                f"Fisher norm: {sum(f.sum().item() for f in fisher.values()):.2f}"
            )

    def consolidate(
        self,
        global_model: DQNNetwork,
        expert_results: List[Dict[str, Any]],
        consolidation_steps: int = 5000,
        learning_rate: float = 1e-4,
    ) -> DQNNetwork:
        """Consolidate expert knowledge into global model using EWC penalty.

        Iteratively fine-tunes the global model to match each expert while
        penalizing deviation from important parameters.

        Args:
            global_model: The current global/consolidated model.
            expert_results: List of expert training results (with state dicts, buffers).
            consolidation_steps: Number of gradient steps for consolidation.
            learning_rate: Learning rate for consolidation.

        Returns:
            The consolidated model.
        """
        if self.logger:
            self.logger.info("Starting EWC consolidation...")

        # First, register all tasks
        for result in expert_results:
            temp_model = copy.deepcopy(global_model)
            temp_model.load_state_dict(result["policy_state_dict"])
            self.register_task(
                temp_model, result["replay_buffer"], result["valid_actions"]
            )

        # Now fine-tune global model
        consolidated = copy.deepcopy(global_model)
        consolidated.train()
        optimizer = torch.optim.AdamW(consolidated.parameters(), lr=learning_rate)

        for step in range(consolidation_steps):
            total_loss = torch.tensor(0.0, device=self.device)

            # EWC penalty: penalize deviations from each task's optimal params
            for task_idx, optimal in enumerate(self.optimal_params):
                if self.online:
                    fisher = self.cumulative_fisher
                else:
                    fisher = self.fisher_matrices[task_idx]

                for name, param in consolidated.named_parameters():
                    if param.requires_grad and name in fisher:
                        total_loss += (
                            fisher[name] * (param - optimal[name]) ** 2
                        ).sum()

            total_loss = self.lambda_ewc * total_loss / (2 * len(self.optimal_params))

            # Also add a direct parameter averaging pull
            avg_loss = torch.tensor(0.0, device=self.device)
            for result in expert_results:
                expert_sd = result["policy_state_dict"]
                for name, param in consolidated.named_parameters():
                    if param.requires_grad and name in expert_sd:
                        avg_loss += ((param - expert_sd[name].to(self.device)) ** 2).sum()
            avg_loss = avg_loss / len(expert_results)

            loss = total_loss + avg_loss
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(consolidated.parameters(), 1.0)
            optimizer.step()

            if self.logger and (step + 1) % 500 == 0:
                self.logger.info(
                    f"EWC consolidation step {step + 1}/{consolidation_steps} | "
                    f"Loss: {loss.item():.6f} (EWC: {total_loss.item():.6f}, "
                    f"Avg: {avg_loss.item():.6f})"
                )

        consolidated.eval()
        if self.logger:
            self.logger.info("EWC consolidation complete.")
        return consolidated

    def save(self, path: str) -> None:
        """Save EWC state (Fisher matrices and optimal params)."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "fisher_matrices": self.fisher_matrices,
                "optimal_params": self.optimal_params,
                "cumulative_fisher": self.cumulative_fisher,
            },
            path,
        )

    def load(self, path: str) -> None:
        """Load EWC state."""
        data = torch.load(path, map_location=self.device, weights_only=False)
        self.fisher_matrices = data["fisher_matrices"]
        self.optimal_params = data["optimal_params"]
        self.cumulative_fisher = data["cumulative_fisher"]
