"""DQN agent with experience replay and target network."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy
from typing import Dict, Optional, Tuple, List


class ReplayBuffer:
    """Fixed-size replay buffer storing transitions as pre-allocated numpy arrays."""

    def __init__(self, capacity: int, obs_shape: Tuple[int, ...]):
        self.capacity = capacity
        self.pos = 0
        self.size = 0

        self.states = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.dones = np.zeros(capacity, dtype=np.float32)

    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool):
        self.states[self.pos] = state
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_states[self.pos] = next_state
        self.dones[self.pos] = float(done)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        indices = np.random.randint(0, self.size, size=batch_size)
        return {
            "states": self.states[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "next_states": self.next_states[indices],
            "dones": self.dones[indices],
        }

    def sample_all(self, max_samples: int) -> Dict[str, np.ndarray]:
        """Sample up to max_samples transitions (for consolidation buffer)."""
        n = min(max_samples, self.size)
        indices = np.random.choice(self.size, size=n, replace=False)
        return {
            "states": self.states[indices].copy(),
            "actions": self.actions[indices].copy(),
            "rewards": self.rewards[indices].copy(),
            "next_states": self.next_states[indices].copy(),
            "dones": self.dones[indices].copy(),
        }

    def __len__(self) -> int:
        return self.size


class EpsilonSchedule:
    """Linear epsilon decay schedule."""

    def __init__(self, start: float, end: float, decay_steps: int):
        self.start = start
        self.end = end
        self.decay_steps = decay_steps

    def __call__(self, step: int) -> float:
        fraction = min(1.0, step / self.decay_steps)
        return self.start + fraction * (self.end - self.start)


class DQNAgent:
    """DQN agent with target network and epsilon-greedy exploration."""

    def __init__(
        self,
        online_net: nn.Module,
        device: torch.device,
        lr: float = 0.00025,
        gamma: float = 0.99,
        target_update_freq: int = 1000,
        grad_clip: float = 10.0,
    ):
        self.online_net = online_net.to(device)
        self.target_net = copy.deepcopy(online_net).to(device)
        self.target_net.eval()
        self.device = device
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.grad_clip = grad_clip

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.train_steps = 0

    def train_step(self, batch: Dict[str, np.ndarray]) -> float:
        """Single gradient step on a batch of transitions."""
        states = torch.from_numpy(batch["states"]).to(self.device)
        actions = torch.from_numpy(batch["actions"]).long().to(self.device)
        rewards = torch.from_numpy(batch["rewards"]).to(self.device)
        next_states = torch.from_numpy(batch["next_states"]).to(self.device)
        dones = torch.from_numpy(batch["dones"]).to(self.device)

        q_values = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            targets = rewards + self.gamma * (1.0 - dones) * next_q

        loss = F.smooth_l1_loss(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), self.grad_clip)
        self.optimizer.step()

        self.train_steps += 1
        if self.train_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return loss.item()

    def sync_target(self):
        """Force sync target network to online network."""
        self.target_net.load_state_dict(self.online_net.state_dict())


def train_dqn_on_env(
    model: nn.Module,
    env,
    action_mask: np.ndarray,
    device: torch.device,
    config: Dict,
    existing_buffer: Optional[ReplayBuffer] = None,
    eval_env=None,
    log_interval: int = 10000,
) -> Tuple[nn.Module, ReplayBuffer, List[Dict]]:
    """
    Train a DQN agent on a single Atari environment.

    Args:
        model: Q-network (will be modified in-place)
        env: Training environment
        action_mask: Binary mask for valid actions in unified space
        device: Torch device
        config: DQN hyperparameters from config["dqn"]
        existing_buffer: Optional pre-filled replay buffer
        eval_env: Optional separate env for periodic evaluation
        log_interval: Steps between logging

    Returns:
        (trained_model, replay_buffer, training_log)
    """
    cfg = config
    obs_shape = env.observation_space.shape

    buffer = existing_buffer or ReplayBuffer(cfg["buffer_size"], obs_shape)
    agent = DQNAgent(
        model, device,
        lr=cfg["lr"],
        gamma=cfg["gamma"],
        target_update_freq=cfg["target_update_freq"],
        grad_clip=cfg["grad_clip"],
    )
    epsilon_schedule = EpsilonSchedule(
        cfg["epsilon_start"], cfg["epsilon_end"], cfg["epsilon_decay_steps"]
    )

    training_log = []
    episode_reward = 0.0
    episode_count = 0
    recent_rewards: List[float] = []

    obs, _ = env.reset()
    obs = np.array(obs)

    for step in range(1, cfg["train_steps"] + 1):
        epsilon = epsilon_schedule(step)
        action = agent.online_net.select_action(obs, action_mask, epsilon, device)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        next_obs = np.array(next_obs)
        done = terminated or truncated

        clipped_reward = np.clip(reward, -1.0, 1.0)
        buffer.push(obs, action, clipped_reward, next_obs, done)
        episode_reward += reward

        if done:
            recent_rewards.append(episode_reward)
            episode_reward = 0.0
            episode_count += 1
            obs, _ = env.reset()
            obs = np.array(obs)
        else:
            obs = next_obs

        if len(buffer) >= cfg["learning_starts"] and step % cfg["train_freq"] == 0:
            batch = buffer.sample(cfg["batch_size"])
            agent.train_step(batch)

        if step % log_interval == 0:
            mean_r = np.mean(recent_rewards[-20:]) if recent_rewards else 0.0
            entry = {
                "step": step,
                "episodes": episode_count,
                "mean_reward_20": float(mean_r),
                "epsilon": epsilon,
            }
            training_log.append(entry)
            print(f"  Step {step:>7d} | Eps {epsilon:.3f} | "
                  f"Ep {episode_count:>4d} | Mean20 {mean_r:>8.1f}")

    return agent.online_net, buffer, training_log
