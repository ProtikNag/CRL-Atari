"""Neural network architectures for DQN agents."""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional


class NatureDQN(nn.Module):
    """
    Nature DQN CNN (Mnih et al., 2015) with unified action space.

    Input: (batch, frame_stack, 84, 84) uint8 images
    Output: (batch, num_actions) Q-values
    """

    def __init__(self, num_actions: int = 6, frame_stack: int = 4):
        super().__init__()
        self.num_actions = num_actions

        self.features = nn.Sequential(
            nn.Conv2d(frame_stack, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        self.head = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        x = self.features(x)
        x = x.reshape(x.size(0), -1)
        return self.head(x)

    def select_action(
        self,
        state: np.ndarray,
        action_mask: np.ndarray,
        epsilon: float,
        device: torch.device,
    ) -> int:
        """Epsilon-greedy action selection with action masking."""
        valid_actions = np.where(action_mask > 0)[0]

        if np.random.random() < epsilon:
            return int(np.random.choice(valid_actions))

        with torch.no_grad():
            state_t = torch.from_numpy(state).unsqueeze(0).to(device)
            q_values = self.forward(state_t).cpu().numpy()[0]

        q_values[action_mask == 0] = -np.inf
        return int(np.argmax(q_values))
