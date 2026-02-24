"""
Q-Value Normalization Utilities.

Implements running-stats normalization to bring Q-values from different
tasks to comparable scales, addressing the Q-value scale imbalance problem
in multi-task / continual RL.
"""

import torch
import numpy as np
from typing import Optional


class RunningNormalizer:
    """Running mean/std normalizer for Q-values.

    Tracks per-action running statistics and normalizes Q-values so that
    different tasks' Q-scales remain comparable during consolidation.

    Uses Welford's online algorithm for numerically stable updates.

    Args:
        size: Number of values to track (unified_action_dim).
        momentum: Exponential moving average momentum.
        clip_range: Clip normalized values to [-clip_range, clip_range].
    """

    def __init__(
        self,
        size: int = 18,
        momentum: float = 0.01,
        clip_range: float = 10.0,
    ):
        self.size = size
        self.momentum = momentum
        self.clip_range = clip_range

        self.running_mean = torch.zeros(size)
        self.running_var = torch.ones(size)
        self.count = 0

    def update(self, q_values: torch.Tensor) -> None:
        """Update running statistics with a new batch of Q-values.

        Args:
            q_values: Tensor of shape (batch, size) containing Q-values.
        """
        with torch.no_grad():
            batch_mean = q_values.mean(dim=0).cpu()
            batch_var = q_values.var(dim=0).cpu()

            if self.count == 0:
                self.running_mean = batch_mean
                self.running_var = batch_var
            else:
                self.running_mean = (
                    (1 - self.momentum) * self.running_mean
                    + self.momentum * batch_mean
                )
                self.running_var = (
                    (1 - self.momentum) * self.running_var
                    + self.momentum * batch_var
                )
            self.count += 1

    def normalize(self, q_values: torch.Tensor) -> torch.Tensor:
        """Normalize Q-values using running statistics.

        Args:
            q_values: Tensor of shape (batch, size).

        Returns:
            Normalized Q-values, clipped to [-clip_range, clip_range].
        """
        mean = self.running_mean.to(q_values.device)
        std = torch.sqrt(self.running_var.to(q_values.device) + 1e-8)
        normalized = (q_values - mean) / std
        return torch.clamp(normalized, -self.clip_range, self.clip_range)

    def denormalize(self, normalized_q: torch.Tensor) -> torch.Tensor:
        """Convert normalized Q-values back to original scale.

        Args:
            normalized_q: Normalized Q-values of shape (batch, size).

        Returns:
            Denormalized Q-values.
        """
        mean = self.running_mean.to(normalized_q.device)
        std = torch.sqrt(self.running_var.to(normalized_q.device) + 1e-8)
        return normalized_q * std + mean

    def state_dict(self) -> dict:
        return {
            "running_mean": self.running_mean.clone(),
            "running_var": self.running_var.clone(),
            "count": self.count,
        }

    def load_state_dict(self, state: dict) -> None:
        self.running_mean = state["running_mean"].cpu()
        self.running_var = state["running_var"].cpu()
        self.count = state["count"]


class PopArtNormalizer:
    """PopArt (Preserving Outputs Precisely, while Adaptively Rescaling Targets).

    Applies adaptive normalization to the output layer of the network,
    preserving the target outputs while rescaling internal representations.

    Reference: van Hasselt et al., "Learning values across many orders of
    magnitude" (2016).

    Args:
        output_layer: The final Linear layer of the Q-network.
        momentum: EMA momentum for statistics update.
    """

    def __init__(
        self,
        output_layer: torch.nn.Linear,
        momentum: float = 0.01,
    ):
        self.output_layer = output_layer
        self.momentum = momentum

        out_features = output_layer.out_features
        self.running_mean = torch.zeros(out_features)
        self.running_std = torch.ones(out_features)
        self.count = 0

    def update_and_rescale(self, targets: torch.Tensor) -> torch.Tensor:
        """Update statistics and rescale network output layer to preserve outputs.

        Args:
            targets: Raw target Q-values of shape (batch, out_features) or (batch,).

        Returns:
            Normalized targets for training.
        """
        with torch.no_grad():
            if targets.dim() == 1:
                new_mean = targets.mean().cpu()
                new_std = targets.std().cpu().clamp(min=1e-4)

                old_mean = self.running_mean.clone()
                old_std = self.running_std.clone()

                # Update stats
                if self.count == 0:
                    self.running_mean = new_mean.expand_as(self.running_mean)
                    self.running_std = new_std.expand_as(self.running_std)
                else:
                    self.running_mean = (
                        (1 - self.momentum) * self.running_mean + self.momentum * new_mean
                    )
                    self.running_std = (
                        (1 - self.momentum) * self.running_std + self.momentum * new_std
                    )
                self.count += 1

                # Rescale output layer weights and biases to preserve outputs
                ratio = old_std / self.running_std
                device = self.output_layer.weight.device
                ratio = ratio.to(device)
                old_mean = old_mean.to(device)
                new_mean = self.running_mean.to(device)
                new_std = self.running_std.to(device)

                self.output_layer.weight.data *= ratio.unsqueeze(1)
                self.output_layer.bias.data = (
                    ratio * self.output_layer.bias.data
                    + (ratio * old_mean - self.running_mean.to(device)) / new_std
                )

            # Normalize targets
            mean = self.running_mean.to(targets.device)
            std = self.running_std.to(targets.device)
            if targets.dim() == 1:
                return (targets - mean[0]) / std[0]
            return (targets - mean) / std

    def state_dict(self) -> dict:
        return {
            "running_mean": self.running_mean.clone(),
            "running_std": self.running_std.clone(),
            "count": self.count,
        }

    def load_state_dict(self, state: dict) -> None:
        self.running_mean = state["running_mean"]
        self.running_std = state["running_std"]
        self.count = state["count"]
