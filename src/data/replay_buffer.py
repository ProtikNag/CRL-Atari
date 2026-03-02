"""
Replay Buffer for DQN training.

Stores (state, action, reward, next_state, done) transitions in a circular buffer.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional


class ReplayBuffer:
    """Fixed-size circular replay buffer with numpy backend.

    Stores transitions as uint8 frames to minimize memory usage.

    Args:
        capacity: Maximum number of transitions to store.
        frame_stack: Number of stacked frames per observation.
        frame_shape: Spatial dimensions of each frame (H, W).
        device: Device to move sampled batches to.
    """

    def __init__(
        self,
        capacity: int,
        frame_stack: int = 4,
        frame_shape: Tuple[int, int] = (84, 84),
        device: str = "cpu",
    ):
        self.capacity = capacity
        self.frame_stack = frame_stack
        self.device = device

        # Allocate storage as uint8 for memory efficiency
        self.states = np.zeros(
            (capacity, frame_stack, *frame_shape), dtype=np.uint8
        )
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros(
            (capacity, frame_stack, *frame_shape), dtype=np.uint8
        )
        self.dones = np.zeros(capacity, dtype=np.bool_)

        self.position = 0
        self.size = 0

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store a single transition.

        Args:
            state: Observation array of shape (frame_stack, H, W).
            action: Action index taken.
            reward: Reward received.
            next_state: Resulting observation.
            done: Whether the episode terminated.
        """
        idx = self.position
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.dones[idx] = done

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Sample a random batch of transitions.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            Tuple of (states, actions, rewards, next_states, dones) as tensors.
        """
        indices = np.random.randint(0, self.size, size=batch_size)

        states = torch.from_numpy(self.states[indices]).float().to(self.device) / 255.0
        actions = torch.from_numpy(self.actions[indices]).long().to(self.device)
        rewards = torch.from_numpy(self.rewards[indices]).float().to(self.device)
        next_states = (
            torch.from_numpy(self.next_states[indices]).float().to(self.device) / 255.0
        )
        dones = torch.from_numpy(self.dones[indices]).float().to(self.device)

        return states, actions, rewards, next_states, dones

    def sample_states(self, num_samples: int) -> torch.Tensor:
        """Sample only states (for Fisher computation, distillation, etc.).

        Args:
            num_samples: Number of states to sample.

        Returns:
            Tensor of shape (num_samples, frame_stack, H, W) in [0, 1].
        """
        num_samples = min(num_samples, self.size)
        indices = np.random.randint(0, self.size, size=num_samples)
        states = torch.from_numpy(self.states[indices]).float().to(self.device) / 255.0
        return states

    def sample_all_data(
        self, max_samples: Optional[int] = None
    ) -> Tuple[torch.Tensor, ...]:
        """Return all stored data (or up to max_samples).

        Args:
            max_samples: Optional cap on number of samples.

        Returns:
            Tuple of (states, actions, rewards, next_states, dones) as tensors.
        """
        n = self.size if max_samples is None else min(self.size, max_samples)
        indices = np.random.choice(self.size, size=n, replace=False)

        states = torch.from_numpy(self.states[indices]).float().to(self.device) / 255.0
        actions = torch.from_numpy(self.actions[indices]).long().to(self.device)
        rewards = torch.from_numpy(self.rewards[indices]).float().to(self.device)
        next_states = (
            torch.from_numpy(self.next_states[indices]).float().to(self.device) / 255.0
        )
        dones = torch.from_numpy(self.dones[indices]).float().to(self.device)

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return self.size

    def filter_by_confidence(
        self,
        model: "DQNNetwork",  # noqa: F821 — forward ref to avoid circular import
        valid_actions: List[int],
        top_k: int,
        batch_size: int = 256,
    ) -> torch.Tensor:
        """Return the top-K most decisive states ranked by Q-value gap.

        For each stored state the *Q-value gap* is defined as
        ``max_a Q(s,a) - min_a Q(s,a)`` over the valid action subset.
        States where the expert has a strong preference (large gap) carry
        more informative Fisher / gradient signal.

        Args:
            model: Expert DQN used to score states.
            valid_actions: Indices of valid actions for this game.
            top_k: Number of high-confidence states to keep.
            batch_size: Forward-pass batch size (controls memory).

        Returns:
            Tensor of shape ``(top_k, frame_stack, H, W)`` in ``[0, 1]``.
        """
        n = self.size
        top_k = min(top_k, n)
        gaps = np.empty(n, dtype=np.float32)

        model.eval()
        with torch.no_grad():
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                states_np = self.states[start:end]
                states_t = (
                    torch.from_numpy(states_np).float().to(self.device) / 255.0
                )
                q_values = model(states_t)  # (B, num_actions)
                valid_q = q_values[:, valid_actions]  # (B, |valid|)
                gap = (valid_q.max(dim=1).values - valid_q.min(dim=1).values)
                gaps[start:end] = gap.cpu().numpy()

        # Pick the top_k indices with largest gaps
        top_indices = np.argpartition(gaps, -top_k)[-top_k:]
        # Sort them descending for determinism (optional but nice for logging)
        top_indices = top_indices[np.argsort(gaps[top_indices])[::-1]]

        states = (
            torch.from_numpy(self.states[top_indices]).float().to(self.device) / 255.0
        )
        return states
