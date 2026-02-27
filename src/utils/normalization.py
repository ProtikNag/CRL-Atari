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
from typing import Optional, Tuple, Union


class PopArtNormalizer:
    """PopArt normalization for DQN Q-value targets.

    Maintains running mean/std of scalar target Q-values and rescales the
    network output layer weights and biases so that the un-normalized
    predictions are preserved while the network learns on normalized targets.

    The key equations when stats change from (mu_old, sigma_old) to
    (mu_new, sigma_new):
        W_new = (sigma_old / sigma_new) * W_old
        b_new = (sigma_old * b_old + mu_old - mu_new) / sigma_new

    IMPORTANT: Both the policy network's and target network's output layers
    must be rescaled in lock-step. Otherwise, denormalize() on the target
    net produces wrong values (it uses current mu/sigma but the target net's
    weights embed stale normalization params from its last hard copy).

    Args:
        output_layer: The final nn.Linear layer of the policy Q-network.
        target_output_layer: The final nn.Linear layer of the target Q-network.
            If provided, it is rescaled identically whenever stats update.
        momentum: EMA momentum for statistics update.
    """

    def __init__(
        self,
        output_layer: nn.Linear,
        target_output_layer: Optional[nn.Linear] = None,
        momentum: float = 0.01,
    ):
        self.output_layer = output_layer
        self.target_output_layer = target_output_layer
        self.momentum = momentum
        self.mu = torch.tensor(0.0)
        self.sigma = torch.tensor(1.0)
        self.count = 0

    def _rescale_layer(
        self,
        layer: nn.Linear,
        old_sigma_d: torch.Tensor,
        new_sigma_d: torch.Tensor,
        old_mu_d: torch.Tensor,
        new_mu_d: torch.Tensor,
    ) -> None:
        """Rescale a single output layer to preserve un-normalized outputs."""
        old_bias = layer.bias.data.clone()
        layer.weight.data.mul_(old_sigma_d / new_sigma_d)
        layer.bias.data = (old_sigma_d * old_bias + old_mu_d - new_mu_d) / new_sigma_d

    def normalize_targets(
        self,
        targets: torch.Tensor,
        q_taken: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Update running stats, rescale output layer(s), normalize targets.

        When q_taken is provided: since q_taken was computed from the policy
        net BEFORE this method rescales its output layer, q_taken is in the
        OLD normalized space. This method returns a corrected q_taken that
        has been affine-transformed into the NEW normalized space so that
        the loss is computed consistently.

        Args:
            targets: Un-normalized scalar targets of shape (batch,).
            q_taken: Optional policy-net Q-values computed before rescaling.

        Returns:
            If q_taken is None: normalized targets.
            If q_taken is given: (normalized_targets, corrected_q_taken).
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

            # Rescale output layer(s) to preserve un-normalized outputs
            scale = None  # will be set if rescaling happens
            if self.count > 1 and old_sigma > 1e-6:
                device = self.output_layer.weight.device
                old_sigma_d = old_sigma.to(device)
                new_sigma_d = self.sigma.to(device)
                old_mu_d = old_mu.to(device)
                new_mu_d = self.mu.to(device)

                self._rescale_layer(
                    self.output_layer, old_sigma_d, new_sigma_d, old_mu_d, new_mu_d
                )
                if self.target_output_layer is not None:
                    self._rescale_layer(
                        self.target_output_layer,
                        old_sigma_d, new_sigma_d, old_mu_d, new_mu_d,
                    )

                scale = old_sigma_d / new_sigma_d
                shift = (old_mu_d - new_mu_d) / new_sigma_d

        # Normalize targets
        mu = self.mu.to(targets.device)
        sigma = self.sigma.to(targets.device)
        normalized = (targets - mu) / sigma

        if q_taken is not None:
            # Correct q_taken from old → new normalized space (preserves grads)
            if scale is not None:
                corrected_q = scale.to(q_taken.device) * q_taken + shift.to(q_taken.device)
            else:
                corrected_q = q_taken
            return normalized, corrected_q

        return normalized

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
