"""Atari environment wrappers for preprocessing and unified action space."""

import gymnasium as gym
import numpy as np
from typing import Dict

try:
    import ale_py
    gym.register_envs(ale_py)
except (ImportError, TypeError):
    pass


def get_action_mask(game_id: str, unified_dim: int) -> np.ndarray:
    """
    Get binary action mask for a game within the unified action space.

    Games with fewer actions than unified_dim have trailing zeros.
    """
    action_dims = {
        "ALE/Pong-v5": 6,
        "ALE/Breakout-v5": 4,
        "ALE/SpaceInvaders-v5": 6,
    }
    n_valid = action_dims.get(game_id, unified_dim)
    mask = np.zeros(unified_dim, dtype=np.float32)
    mask[:n_valid] = 1.0
    return mask


class FireResetEnv(gym.Wrapper):
    """
    Press FIRE (action 1) on reset for envs that require it to start.

    After each reset (including life-loss resets when using
    terminal_on_life_loss), takes the FIRE action so that Pong serves
    and Breakout launches the ball. Envs without FIRE in their action
    meanings are passed through unchanged.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        action_meanings = env.unwrapped.get_action_meanings()
        self.fire_needed = "FIRE" in action_meanings
        if self.fire_needed:
            self.fire_action = action_meanings.index("FIRE")

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if self.fire_needed:
            obs, _, terminated, truncated, info = self.env.step(self.fire_action)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, info


class UnifiedActionWrapper(gym.ActionWrapper):
    """
    Wraps environment to accept actions from a unified (larger) action space.

    Actions beyond the game's native action count are clamped to the last
    valid action. This is a safety net; agents should use action masking
    to avoid selecting invalid actions.
    """

    def __init__(self, env: gym.Env, unified_dim: int):
        super().__init__(env)
        self.native_n = env.action_space.n
        self.action_space = gym.spaces.Discrete(unified_dim)

    def action(self, act: int) -> int:
        return min(int(act), self.native_n - 1)


def make_atari_env(
    game_id: str,
    unified_action_dim: int = 6,
    frame_stack: int = 4,
    seed: int = 42,
    render_mode: str = None,
) -> gym.Env:
    """
    Create a fully preprocessed Atari environment.

    Pipeline:
        1. Raw ALE env (frameskip=1, no sticky actions)
        2. AtariPreprocessing (frame skip, grayscale, resize 84x84, noop reset)
        3. FireResetEnv (auto-press FIRE on reset for Pong/Breakout)
        4. FrameStack (4 frames)
        5. UnifiedActionWrapper (expand to unified action space)
    """
    env = gym.make(
        game_id,
        frameskip=1,
        repeat_action_probability=0.0,
        render_mode=render_mode,
    )
    env = gym.wrappers.AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=4,
        screen_size=84,
        terminal_on_life_loss=True,
        grayscale_obs=True,
        scale_obs=False,
    )
    env = FireResetEnv(env)
    try:
        env = gym.wrappers.FrameStackObservation(env, stack_size=frame_stack)
    except AttributeError:
        env = gym.wrappers.FrameStack(env, num_stack=frame_stack)
    env = UnifiedActionWrapper(env, unified_action_dim)
    env.reset(seed=seed)
    return env
