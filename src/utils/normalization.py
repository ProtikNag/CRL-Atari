"""
Q-Value Normalization Utilities.

Implements PopArt (Preserving Outputs Precisely while Adaptively Rescaling
Targets) normalization for DQN, addressing the Q-value scale imbalance
problem in multi-task / continual RL.

Reference: van Hasselt et al., "Learning values across many orders of
magnitude" (NeurIPS 2016).
"""

import torch
import torch.nn as nn
from typing import Optional


class PopArtNormalizer:
    """PopArt normalization for DQN Q-value targets.

    Maintains running mean/std of scalar target Q-values and rescales the
    network output layer weights and biases so that the un-normalized
    predictions are preserved while the network learns on normalized targets.

    The key equations when stats change from (mu_old, sigma_old) to
    (mu_new, sigma_new):
        W_new = (sigma_old / sigma_new) * W_old
        b_new = (sigma_old * b_old + mu_old - mu_new) / sigma_new

    Args:
        output_layer: The final nn.Linear layer of the Q-network.
        momentum: EMA momentum for statistics update.
    """

    def __init__(self, output_layer: nn.Linear, momentum: float = 0.01):
        self.output_layer = output_layer
        self.momentum = momentum
        self.mu = torch.tensor(0.0)
        self.sigma = torch.tensor(1.0)
        self.count = 0

    def normalize_targets(self, targets: torch.Tensor) -> torch.Tensor:
        """Update running stats, rescale output layer, normalize targets.

        Args:
            targets: Un-normalized scalar targets of shape (batch,).

        Returns:
            Normalized targets of shape (batch,).
        """
        with torch.no_grad():
            old_mu = self.mu.clone()
            old_sigma = self.sigma.clone()

            batch_mean = targets.mean().cpu()
            batch_std = targets.std().cpu().clamp(min=1e-4)

            if self.count == 0:
                self.mu = batch_mean
                self.sigma = batch_std
            else:
                self.mu = (1 - self.momentum) * self.mu + self.momentum * batch_mean
                self.sigma = (
                    (1 - self.momentum) * self.sigma + self.momentum * batch_std
                )

            self.count += 1

            # Rescale output layer to preserve un-normalized outputs
            if self.count > 1 and old_sigma > 1e-6:
                device = self.output_layer.weight.device
                old_sigma_d = old_sigma.to(device)
                new_sigma_d = self.sigma.to(device)
                old_mu_d = old_mu.to(device)
                new_mu_d = self.mu.to(device)

                old_bias = self.output_layer.bias.data.clone()
                self.output_layer.weight.data.mul_(old_sigma_d / new_sigma_d)
                self.output_layer.bias.data = (
                    old_sigma_d * old_bias + old_mu_d - new_mu_d
                ) / new_sigma_d

        # Normalize targets
        mu = self.mu.to(targets.device)
        sigma = self.sigma.to(targets.device)
        return (targets - mu) / sigma

    def denormalize(self, normalized_q: torch.Tensor) -> torch.Tensor:
        """Convert normalized Q-values back to original scale.

        Args:
            normalized_q: Normalized Q-values (any shape).

        Returns:
            Un-normalized Q-values.
        """
        mu = self.mu.to(normalized_q.device)
        sigma = self.sigma.to(normalized_q.device)
        return sigma * normalized_q + mu

    def state_dict(self) -> dict:
        """Serialize normalizer state for checkpointing."""
        return {
            "mu": self.mu.clone(),
            "sigma": self.sigma.clone(),
            "count": self.count,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore normalizer state from checkpoint."""
        self.mu = state["mu"].cpu()
        self.sigma = state["sigma"].cpu()
        self.count = state["count"]
