"""
DQN Network with Unified Action Space for CRL-Atari.

The network outputs Q-values for ALL 18 Atari joystick actions.
For a given game, only the valid action subset is used during
action selection and training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class DQNNetwork(nn.Module):
    """Deep Q-Network with unified action head for multi-game CRL.

    Architecture: 3-layer CNN backbone -> FC hidden -> Q-value head (18 actions).
    Optionally supports Dueling DQN (value + advantage streams).

    Args:
        in_channels: Number of stacked frames (typically 4).
        conv_channels: List of output channels for each conv layer.
        conv_kernels: List of kernel sizes for each conv layer.
        conv_strides: List of strides for each conv layer.
        fc_hidden: Size of the hidden fully connected layer.
        unified_action_dim: Output dimension (18 for full Atari action set).
        dueling: Whether to use Dueling DQN architecture.
    """

    def __init__(
        self,
        in_channels: int = 4,
        conv_channels: Optional[List[int]] = None,
        conv_kernels: Optional[List[int]] = None,
        conv_strides: Optional[List[int]] = None,
        fc_hidden: int = 1024,
        unified_action_dim: int = 18,
        dueling: bool = False,
    ):
        super().__init__()

        if conv_channels is None:
            conv_channels = [64, 128, 128]
        if conv_kernels is None:
            conv_kernels = [8, 4, 3]
        if conv_strides is None:
            conv_strides = [4, 2, 1]

        self.unified_action_dim = unified_action_dim
        self.dueling = dueling

        # Build CNN backbone
        layers: List[nn.Module] = []
        prev_channels = in_channels
        for out_ch, k, s in zip(conv_channels, conv_kernels, conv_strides):
            layers.append(nn.Conv2d(prev_channels, out_ch, kernel_size=k, stride=s))
            layers.append(nn.ReLU(inplace=True))
            prev_channels = out_ch
        self.features = nn.Sequential(*layers)

        # Compute flattened feature size for 84x84 input
        self._feature_size = self._get_conv_output_size(in_channels)

        if dueling:
            # Value stream
            self.value_stream = nn.Sequential(
                nn.Linear(self._feature_size, fc_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(fc_hidden, 1),
            )
            # Advantage stream
            self.advantage_stream = nn.Sequential(
                nn.Linear(self._feature_size, fc_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(fc_hidden, unified_action_dim),
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(self._feature_size, fc_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(fc_hidden, unified_action_dim),
            )

    def _get_conv_output_size(self, in_channels: int) -> int:
        """Compute the flattened output size of the CNN for an 84x84 input."""
        dummy = torch.zeros(1, in_channels, 84, 84)
        with torch.no_grad():
            out = self.features(dummy)
        return int(out.view(1, -1).size(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, channels, 84, 84).
               Pixel values should be in [0, 255] (uint8) or [0, 1] (float).

        Returns:
            Q-values of shape (batch, unified_action_dim).
        """
        # Normalize pixel values to [0, 1] if they come as uint8
        if x.dtype == torch.uint8:
            x = x.float() / 255.0

        features = self.features(x)
        features = features.view(features.size(0), -1)

        if self.dueling:
            value = self.value_stream(features)
            advantage = self.advantage_stream(features)
            # Combine: Q(s,a) = V(s) + A(s,a) - mean(A(s,.))
            q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        else:
            q_values = self.fc(features)

        return q_values

    def get_action_mask(self, valid_actions: List[int]) -> torch.Tensor:
        """Create a boolean mask for valid actions in the unified space.

        Args:
            valid_actions: List of valid action indices for the current game.

        Returns:
            Boolean tensor of shape (unified_action_dim,) with True for valid actions.
        """
        mask = torch.zeros(self.unified_action_dim, dtype=torch.bool)
        mask[valid_actions] = True
        return mask

    def masked_q_values(
        self, x: torch.Tensor, valid_actions: List[int]
    ) -> torch.Tensor:
        """Compute Q-values with invalid actions masked to -inf.

        Args:
            x: Input tensor of shape (batch, channels, 84, 84).
            valid_actions: List of valid action indices.

        Returns:
            Masked Q-values of shape (batch, unified_action_dim).
        """
        q_values = self.forward(x)
        mask = self.get_action_mask(valid_actions).to(x.device)
        # Set invalid actions to -inf so they are never selected
        q_values[~mask.unsqueeze(0).expand_as(q_values)] = float("-inf")
        return q_values

    @property
    def num_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
