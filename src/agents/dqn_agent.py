"""
DQN Agent for Atari games with unified action space.

Handles action selection (epsilon-greedy with action masking),
training step, and target network updates.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
import os
from typing import List, Dict, Any, Optional, Tuple

from src.models.dqn import DQNNetwork
from src.data.replay_buffer import ReplayBuffer
from src.utils.normalization import RunningNormalizer


class DQNAgent:
    """DQN agent with unified action space and Q-value normalization.

    Supports action masking for game-specific valid actions, epsilon-greedy
    exploration, target network soft/hard updates, and running-stats
    Q-value normalization.

    Args:
        config: Configuration dictionary.
        valid_actions: List of valid action indices for the current game.
        device: Torch device string.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        valid_actions: List[int],
        device: str = "cpu",
    ):
        self.config = config
        self.valid_actions = valid_actions
        self.device = device
        self.unified_action_dim = config["model"]["unified_action_dim"]

        # Build networks
        model_cfg = config["model"]
        self.policy_net = DQNNetwork(
            in_channels=config["env"]["frame_stack"],
            conv_channels=model_cfg["conv_channels"],
            conv_kernels=model_cfg["conv_kernels"],
            conv_strides=model_cfg["conv_strides"],
            fc_hidden=model_cfg["fc_hidden"],
            unified_action_dim=model_cfg["unified_action_dim"],
            dueling=model_cfg.get("dueling", False),
        ).to(device)

        self.target_net = copy.deepcopy(self.policy_net)
        self.target_net.eval()

        # Optimizer
        train_cfg = config["training"]
        self.optimizer = optim.AdamW(
            self.policy_net.parameters(), lr=train_cfg["learning_rate"]
        )

        # Replay buffer
        self.replay_buffer = ReplayBuffer(
            capacity=train_cfg["buffer_size"],
            frame_stack=config["env"]["frame_stack"],
            frame_shape=(config["env"]["screen_size"], config["env"]["screen_size"]),
            device=device,
        )

        # Q-value normalizer
        norm_cfg = config.get("normalization", {})
        self.normalizer = None
        if norm_cfg.get("enabled", False):
            self.normalizer = RunningNormalizer(
                size=model_cfg["unified_action_dim"],
                momentum=norm_cfg.get("momentum", 0.01),
                clip_range=norm_cfg.get("clip_range", 10.0),
            )

        # Exploration parameters
        self.eps_start = train_cfg["eps_start"]
        self.eps_end = train_cfg["eps_end"]
        self.eps_decay_steps = train_cfg["eps_decay_steps"]
        self.gradient_clip = train_cfg.get("gradient_clip", 10.0)
        self.gamma = train_cfg["gamma"]
        self.batch_size = train_cfg["batch_size"]
        self.target_update_freq = train_cfg["target_update_freq"]

        # Step counters
        # env_steps: incremented every environment interaction (for epsilon decay)
        # train_steps: incremented every gradient update (for target network updates)
        self.env_steps = 0
        self.train_steps = 0

    @property
    def epsilon(self) -> float:
        """Current exploration rate (linear decay based on env steps)."""
        frac = min(self.env_steps / self.eps_decay_steps, 1.0)
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> int:
        """Select action using epsilon-greedy with action masking.

        Args:
            state: Observation array of shape (frame_stack, H, W).
            deterministic: If True, always select greedy action.

        Returns:
            Selected action index from the unified action space.
        """
        if not deterministic and np.random.random() < self.epsilon:
            return np.random.choice(self.valid_actions)

        with torch.no_grad():
            state_tensor = (
                torch.from_numpy(state).float().unsqueeze(0).to(self.device) / 255.0
            )
            q_values = self.policy_net(state_tensor)

            # Mask invalid actions
            mask = torch.full(
                (self.unified_action_dim,), float("-inf"), device=self.device
            )
            mask[self.valid_actions] = 0.0
            masked_q = q_values + mask.unsqueeze(0)

            return masked_q.argmax(dim=1).item()

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store a transition in the replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)
        self.env_steps += 1

    def train_step(self) -> Optional[float]:
        """Perform a single training step (sample batch, compute loss, update).

        Returns:
            Training loss value, or None if buffer is too small.
        """
        if len(self.replay_buffer) < self.config["training"]["min_buffer_size"]:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        # Current Q-values for taken actions
        q_values = self.policy_net(states)
        q_taken = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values (only over valid actions)
        with torch.no_grad():
            next_q_values = self.target_net(next_states)
            # Mask invalid actions for target
            mask = torch.full(
                (self.unified_action_dim,), float("-inf"), device=self.device
            )
            mask[self.valid_actions] = 0.0
            masked_next_q = next_q_values + mask.unsqueeze(0)
            next_q_max = masked_next_q.max(dim=1)[0]
            target = rewards + self.gamma * next_q_max * (1 - dones)

        # Update normalizer if enabled
        if self.normalizer is not None:
            self.normalizer.update(q_values.detach())

        # Huber loss (smooth L1)
        loss = nn.functional.smooth_l1_loss(q_taken, target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            self.policy_net.parameters(), self.gradient_clip
        )
        self.optimizer.step()

        self.train_steps += 1

        # Update target network periodically (based on training steps)
        if self.train_steps % self.target_update_freq == 0:
            self.update_target_network()

        return loss.item()

    def update_target_network(self) -> None:
        """Hard update: copy policy net weights to target net."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path: str) -> None:
        """Save agent state to disk.

        Args:
            path: File path for the checkpoint.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = {
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "valid_actions": self.valid_actions,
            "env_steps": self.env_steps,
            "train_steps": self.train_steps,
        }
        if self.normalizer is not None:
            checkpoint["normalizer"] = self.normalizer.state_dict()
        torch.save(checkpoint, path)

    def load(self, path: str) -> None:
        """Load agent state from disk.

        Args:
            path: File path to the checkpoint.
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.valid_actions = checkpoint["valid_actions"]
        # Support both old (total_steps) and new (env_steps/train_steps) checkpoints
        self.env_steps = checkpoint.get("env_steps", checkpoint.get("total_steps", 0))
        self.train_steps = checkpoint.get("train_steps", checkpoint.get("total_steps", 0))
        if self.normalizer is not None and "normalizer" in checkpoint:
            self.normalizer.load_state_dict(checkpoint["normalizer"])

    def load_policy_weights(self, state_dict: dict) -> None:
        """Load only policy network weights (for global-to-local init).

        Args:
            state_dict: State dict for the policy network.
        """
        self.policy_net.load_state_dict(state_dict)
        self.target_net.load_state_dict(state_dict)

    def get_policy_state_dict(self) -> dict:
        """Return a copy of the policy network state dict."""
        return copy.deepcopy(self.policy_net.state_dict())
