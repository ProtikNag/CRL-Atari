"""
MLP-based DQN for Classic Control CRL.

Drop-in replacement for DQNNetwork (src/models/dqn.py) when operating on
low-dimensional state vectors instead of image observations.  The interface
(forward signature, get_action_mask, masked_q_values, output_layer property)
is identical so that all training, evaluation, and consolidation code can
use either model class interchangeably.

Architecture:
    state_dim -> hidden_dims[0] -> hidden_dims[1] -> ... -> unified_action_dim

States from different environments are zero-padded to a common max_state_dim
before being fed to the network, so a single model handles the entire task
sequence.  No pixel normalization (/255) is applied since inputs are already
float-valued.
"""

import torch
import torch.nn as nn
from typing import List, Optional


class MLPDQNNetwork(nn.Module):
    """MLP Deep Q-Network with union action head for multi-task classic control CRL.

    Args:
        state_dim: Dimensionality of the (zero-padded) input state vector.
        hidden_dims: List of hidden layer widths. Default ``[128, 128]``.
        unified_action_dim: Output dimension (union action space size across tasks).
        dueling: Whether to use the Dueling DQN architecture (value + advantage).
    """

    def __init__(
        self,
        state_dim: int = 8,
        hidden_dims: Optional[List[int]] = None,
        unified_action_dim: int = 4,
        dueling: bool = False,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [128, 128]

        self.state_dim = state_dim
        self.hidden_dims = list(hidden_dims)
        self.unified_action_dim = unified_action_dim
        self.dueling = dueling

        # ── Shared feature trunk ────────────────────────────────────────
        # Build all hidden layers except the last one as a shared backbone.
        # If there is only one hidden layer, the trunk is empty and the
        # entire network lives in the head section.
        trunk_layers: List[nn.Module] = []
        prev_dim = state_dim
        for h_dim in hidden_dims[:-1]:
            trunk_layers.append(nn.Linear(prev_dim, h_dim))
            trunk_layers.append(nn.ReLU(inplace=True))
            prev_dim = h_dim
        self.trunk = nn.Sequential(*trunk_layers)

        # Dimensionality entering the head (last hidden layer input)
        last_hidden = hidden_dims[-1]

        if dueling:
            # Value stream
            self.value_stream = nn.Sequential(
                nn.Linear(prev_dim, last_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(last_hidden, 1),
            )
            # Advantage stream
            self.advantage_stream = nn.Sequential(
                nn.Linear(prev_dim, last_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(last_hidden, unified_action_dim),
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(prev_dim, last_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(last_hidden, unified_action_dim),
            )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Q-values for all actions in the union space.

        Args:
            x: Float state tensor of shape ``(batch, state_dim)``.
               No pixel normalization is applied; inputs are used as-is.

        Returns:
            Q-values of shape ``(batch, unified_action_dim)``.
        """
        features = self.trunk(x)

        if self.dueling:
            value = self.value_stream(features)
            advantage = self.advantage_stream(features)
            q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        else:
            q_values = self.fc(features)

        return q_values

    # ------------------------------------------------------------------
    # Action masking utilities (same interface as DQNNetwork)
    # ------------------------------------------------------------------
    def get_action_mask(self, valid_actions: List[int]) -> torch.Tensor:
        """Create a boolean mask for valid actions in the unified space.

        Args:
            valid_actions: List of valid action indices for the current task.

        Returns:
            Boolean tensor of shape ``(unified_action_dim,)`` with ``True``
            for valid actions.
        """
        mask = torch.zeros(self.unified_action_dim, dtype=torch.bool)
        mask[valid_actions] = True
        return mask

    def masked_q_values(
        self, x: torch.Tensor, valid_actions: List[int]
    ) -> torch.Tensor:
        """Compute Q-values with invalid actions masked to ``-inf``.

        Args:
            x: Float state tensor of shape ``(batch, state_dim)``.
            valid_actions: List of valid action indices.

        Returns:
            Masked Q-values of shape ``(batch, unified_action_dim)``.
        """
        q_values = self.forward(x)
        mask = self.get_action_mask(valid_actions).to(x.device)
        q_values[~mask.unsqueeze(0).expand_as(q_values)] = float("-inf")
        return q_values

    # ------------------------------------------------------------------
    # Properties (same interface as DQNNetwork)
    # ------------------------------------------------------------------
    @property
    def output_layer(self) -> nn.Linear:
        """Return the final Linear layer (for PopArt weight rescaling)."""
        if self.dueling:
            return self.advantage_stream[-1]
        return self.fc[-1]

    @property
    def num_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ------------------------------------------------------------------
    # Hidden feature extraction (for lateral connections)
    # ------------------------------------------------------------------
    def hidden_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the output of the shared trunk (before the head layers).

        This is used by MLPProgressiveColumn for lateral connections from
        the frozen Knowledge Base.

        Args:
            x: Float state tensor of shape ``(batch, state_dim)``.

        Returns:
            Hidden features of shape ``(batch, trunk_output_dim)``.
        """
        return self.trunk(x)
