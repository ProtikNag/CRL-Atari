"""
Replay Buffer for vector-observation environments (classic control).

Stores (state, action, reward, next_state, done) transitions in a circular
buffer. Unlike the image-based ReplayBuffer, this variant stores float32
state vectors directly without uint8 compression or /255.0 normalization.
"""

import numpy as np
import torch
from typing import List, Optional, Tuple


class VectorReplayBuffer:
    """Fixed-size circular replay buffer for vector state observations.

    Designed for classic control environments (CartPole, Acrobot, LunarLander)
    where observations are low-dimensional float vectors rather than image
    frames. The interface mirrors ``ReplayBuffer`` so that training loops
    can swap between image and vector buffers with minimal changes.

    Args:
        capacity: Maximum number of transitions to store.
        state_dim: Dimensionality of the (possibly zero-padded) state vector.
        device: Device to move sampled batches to.
    """

    def __init__(self, capacity: int, state_dim: int, device: str = "cpu"):
        self.capacity = capacity
        self.state_dim = state_dim
        self.device = device

        # Pre-allocate storage arrays
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
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
            state: Observation vector of shape ``(state_dim,)``.
            action: Action index taken.
            reward: Reward received.
            next_state: Resulting observation vector.
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
            Tuple of ``(states, actions, rewards, next_states, dones)`` as
            tensors on ``self.device``. States are float32 without any
            rescaling (no /255.0).
        """
        indices = np.random.randint(0, self.size, size=batch_size)

        states = torch.from_numpy(self.states[indices]).to(self.device)
        actions = torch.from_numpy(self.actions[indices]).long().to(self.device)
        rewards = torch.from_numpy(self.rewards[indices]).to(self.device)
        next_states = torch.from_numpy(self.next_states[indices]).to(self.device)
        dones = torch.from_numpy(self.dones[indices]).float().to(self.device)

        return states, actions, rewards, next_states, dones

    def sample_states(self, num_samples: int) -> torch.Tensor:
        """Sample only states (for Fisher computation, distillation, etc.).

        Args:
            num_samples: Number of states to sample.

        Returns:
            Tensor of shape ``(num_samples, state_dim)`` as float32 on
            ``self.device``.
        """
        num_samples = min(num_samples, self.size)
        indices = np.random.randint(0, self.size, size=num_samples)
        states = torch.from_numpy(self.states[indices]).to(self.device)
        return states

    def sample_all_data(
        self, max_samples: Optional[int] = None
    ) -> Tuple[torch.Tensor, ...]:
        """Return all stored data (or up to max_samples).

        Args:
            max_samples: Optional cap on number of samples returned.

        Returns:
            Tuple of ``(states, actions, rewards, next_states, dones)`` as
            tensors on ``self.device``.
        """
        n = self.size if max_samples is None else min(self.size, max_samples)
        indices = np.random.choice(self.size, size=n, replace=False)

        states = torch.from_numpy(self.states[indices]).to(self.device)
        actions = torch.from_numpy(self.actions[indices]).long().to(self.device)
        rewards = torch.from_numpy(self.rewards[indices]).to(self.device)
        next_states = torch.from_numpy(self.next_states[indices]).to(self.device)
        dones = torch.from_numpy(self.dones[indices]).float().to(self.device)

        return states, actions, rewards, next_states, dones

    def filter_by_confidence(
        self,
        model: "nn.Module",  # noqa: F821 -- forward ref to avoid circular import
        valid_actions: List[int],
        top_k: int,
        batch_size: int = 256,
    ) -> torch.Tensor:
        """Return the top-K most decisive states ranked by Q-value gap.

        For each stored state the *Q-value gap* is defined as
        ``max_a Q(s,a) - min_a Q(s,a)`` over the valid action subset.
        States where the expert has a strong preference (large gap) carry
        more informative Fisher / gradient signal.

        The model is expected to accept input of shape
        ``(batch, state_dim)`` and return Q-values of shape
        ``(batch, num_actions)``. This covers both ``DQNNetwork`` (for
        image inputs via a CNN) and MLP-based networks for vector states.

        Args:
            model: Expert network used to score states.
            valid_actions: Indices of valid actions for this environment.
            top_k: Number of high-confidence states to keep.
            batch_size: Forward-pass batch size (controls peak memory).

        Returns:
            Tensor of shape ``(top_k, state_dim)`` on CPU.
        """
        n = self.size
        top_k = min(top_k, n)
        gaps = np.empty(n, dtype=np.float32)

        model.eval()
        with torch.no_grad():
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                states_np = self.states[start:end]
                states_t = torch.from_numpy(states_np).to(self.device)
                q_values = model(states_t)  # (B, num_actions)
                valid_q = q_values[:, valid_actions]  # (B, |valid|)
                gap = valid_q.max(dim=1).values - valid_q.min(dim=1).values
                gaps[start:end] = gap.cpu().numpy()

        # Pick the top_k indices with the largest gaps
        top_indices = np.argpartition(gaps, -top_k)[-top_k:]
        # Sort descending for deterministic ordering
        top_indices = top_indices[np.argsort(gaps[top_indices])[::-1]]

        # Return on CPU to avoid OOM when multiple experts are filtered.
        # Downstream consumers (Fisher / gradient) batch-transfer to GPU.
        states = torch.from_numpy(self.states[top_indices])
        return states

    def __len__(self) -> int:
        return self.size
