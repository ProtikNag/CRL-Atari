"""
Atari environment wrappers following DeepMind-style preprocessing.

Handles frame stacking, skipping, grayscale conversion, resizing,
reward clipping, episodic life, and fire-on-reset.
"""

import gymnasium as gym
import numpy as np
import cv2
from collections import deque
from typing import Optional, List, Tuple

# Register ALE environments with gymnasium
import ale_py
gym.register_envs(ale_py)


class UnifiedActionWrapper(gym.Wrapper):
    """Map unified 18-action space indices to local environment action indices.

    The agent outputs actions in the full 18-action Atari space (e.g., 0, 1, 3,
    4, 11, 12 for Pong). The ALE environment expects local indices (0..n-1).
    This wrapper performs the translation.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        # Get the minimal action set (ALE Action enums -> ints)
        minimal_actions = env.unwrapped.ale.getMinimalActionSet()
        self.unified_to_local = {
            int(a): i for i, a in enumerate(minimal_actions)
        }
        self.local_to_unified = {
            i: int(a) for i, a in enumerate(minimal_actions)
        }

    def step(self, action):
        # Convert unified action index to local environment index
        local_action = self.unified_to_local.get(action, 0)  # default NOOP
        return self.env.step(local_action)


class NoopResetEnv(gym.Wrapper):
    """Execute a random number of no-ops on reset."""

    def __init__(self, env: gym.Env, noop_max: int = 30):
        super().__init__(env)
        self.noop_max = noop_max
        self.noop_action = 0  # NOOP is always action 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        noops = np.random.randint(1, self.noop_max + 1)
        for _ in range(noops):
            obs, _, terminated, truncated, info = self.env.step(self.noop_action)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        return obs, info


class MaxAndSkipEnv(gym.Wrapper):
    """Return max over last 2 frames and skip intermediate frames."""

    def __init__(self, env: gym.Env, skip: int = 4):
        super().__init__(env)
        self._skip = skip
        self._obs_buffer = deque(maxlen=2)

    def step(self, action):
        total_reward = 0.0
        terminated = truncated = False
        for _ in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if terminated or truncated:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, terminated, truncated, info

    def reset(self, **kwargs):
        self._obs_buffer.clear()
        obs, info = self.env.reset(**kwargs)
        self._obs_buffer.append(obs)
        return obs, info


class EpisodicLifeEnv(gym.Wrapper):
    """Make end-of-life == end-of-episode (but only reset on true game over)."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.was_real_done = terminated or truncated
        lives = self.env.unwrapped.ale.lives()
        if 0 < lives < self.lives:
            terminated = True
        self.lives = lives
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        if self.was_real_done:
            obs, info = self.env.reset(**kwargs)
        else:
            # No-op step to advance from terminal/lost life state
            obs, _, _, _, info = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs, info


class FireResetEnv(gym.Wrapper):
    """Press FIRE on reset for games that require it."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == "FIRE"

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, terminated, truncated, info = self.env.step(1)
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)
        obs, _, terminated, truncated, info = self.env.step(2)
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)
        return obs, info


class ClipRewardEnv(gym.Wrapper):
    """Clip reward to {-1, 0, +1}."""

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, np.sign(reward), terminated, truncated, info


class WarpFrame(gym.ObservationWrapper):
    """Convert to grayscale and resize to (screen_size, screen_size)."""

    def __init__(self, env: gym.Env, width: int = 84, height: int = 84):
        super().__init__(env)
        self.width = width
        self.height = height
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.height, self.width, 1), dtype=np.uint8
        )

    def observation(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame[:, :, None]


class FrameStack(gym.Wrapper):
    """Stack the last k frames as a single observation. Channel-first output."""

    def __init__(self, env: gym.Env, k: int = 4):
        super().__init__(env)
        self.k = k
        self.frames = deque(maxlen=k)
        shape = env.observation_space.shape
        # Output shape: (k, H, W)
        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=(k, shape[0], shape[1]),
            dtype=np.uint8,
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.k):
            self.frames.append(obs)
        return self._get_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        frames = np.concatenate(list(self.frames), axis=-1)  # (H, W, k)
        return np.transpose(frames, (2, 0, 1))  # (k, H, W)


def get_valid_actions(env_id: str) -> List[int]:
    """Get the list of valid action indices for a given Atari game.

    Creates a temporary environment to query the available actions.

    Args:
        env_id: Gymnasium environment ID (e.g., 'PongNoFrameskip-v4').

    Returns:
        Sorted list of valid action indices in the full 18-action space.
    """
    env = gym.make(env_id)
    # ale.getMinimalActionSet gives the game-specific actions as Action enums
    minimal_actions = env.unwrapped.ale.getMinimalActionSet()
    valid = sorted([int(a) for a in minimal_actions])
    env.close()
    return valid


def make_atari_env(
    env_id: str,
    seed: int = 42,
    frame_stack: int = 4,
    frame_skip: int = 4,
    screen_size: int = 84,
    noop_max: int = 30,
    episodic_life: bool = True,
    clip_reward: bool = True,
) -> gym.Env:
    """Create a fully wrapped Atari environment.

    Args:
        env_id: Gymnasium environment ID.
        seed: Random seed.
        frame_stack: Number of frames to stack.
        frame_skip: Number of frames to skip (via MaxAndSkip).
        screen_size: Width/height of resized frames.
        noop_max: Max no-op actions on reset.
        episodic_life: Treat loss of life as episode end.
        clip_reward: Clip reward to {-1, 0, +1}.

    Returns:
        Wrapped Gymnasium environment.
    """
    env = gym.make(env_id, render_mode=None)
    # Map unified 18-action indices to local env indices FIRST
    env = UnifiedActionWrapper(env)
    env = NoopResetEnv(env, noop_max=noop_max)
    env = MaxAndSkipEnv(env, skip=frame_skip)

    if episodic_life:
        env = EpisodicLifeEnv(env)

    # Fire reset if applicable
    action_meanings = env.unwrapped.get_action_meanings()
    if "FIRE" in action_meanings and len(action_meanings) > 2:
        env = FireResetEnv(env)

    if clip_reward:
        env = ClipRewardEnv(env)

    env = WarpFrame(env, width=screen_size, height=screen_size)
    env = FrameStack(env, k=frame_stack)

    return env
