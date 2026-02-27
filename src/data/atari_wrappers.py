"""
Atari environment wrappers following DeepMind-style preprocessing.

Handles frame stacking, skipping, grayscale conversion, resizing,
reward clipping, episodic life, fire-on-reset, and union action mapping.
"""

import gymnasium as gym
import numpy as np
import cv2
from collections import deque
from typing import Optional, List, Tuple, Dict

# Register ALE environments with gymnasium
import ale_py
gym.register_envs(ale_py)


# Full ALE 18-action meaning table (correct indices!)
ALE_ACTION_MEANINGS: Dict[int, str] = {
    0: "NOOP", 1: "FIRE", 2: "UP", 3: "RIGHT", 4: "LEFT",
    5: "DOWN", 6: "UPRIGHT", 7: "UPLEFT", 8: "DOWNRIGHT",
    9: "DOWNLEFT", 10: "UPFIRE", 11: "RIGHTFIRE", 12: "LEFTFIRE",
    13: "DOWNFIRE", 14: "UPRIGHTFIRE", 15: "UPLEFTFIRE",
    16: "DOWNRIGHTFIRE", 17: "DOWNLEFTFIRE",
}


def compute_union_action_space(task_sequence: List[str]) -> List[int]:
    """Compute the union of minimal action sets across all tasks.

    Creates a temporary environment for each game to query its minimal
    action set, then returns the sorted union.

    Args:
        task_sequence: List of Atari env IDs, e.g. ["PongNoFrameskip-v4", ...].

    Returns:
        Sorted list of ALE action indices that form the union.
    """
    union = set()
    for env_id in task_sequence:
        env = gym.make(env_id)
        minimal = env.unwrapped.ale.getMinimalActionSet()
        union.update(int(a) for a in minimal)
        env.close()
    return sorted(union)


def get_valid_actions(env_id: str, union_actions: List[int]) -> List[int]:
    """Return indices into the union action list that are valid for env_id.

    Args:
        env_id: Atari env ID, e.g. "PongNoFrameskip-v4".
        union_actions: The union action list (sorted ALE action ints).

    Returns:
        List of indices into union_actions that the game actually uses.
        E.g. if union=[0,1,3,4,11,12] and game uses [0,1,3,4], returns [0,1,2,3].
    """
    env = gym.make(env_id)
    minimal = set(int(a) for a in env.unwrapped.ale.getMinimalActionSet())
    env.close()
    return [i for i, ale_action in enumerate(union_actions) if ale_action in minimal]


class UnionActionWrapper(gym.Wrapper):
    """Map union-space action indices to ALE action indices.

    The agent outputs actions in the union action space (e.g. indices 0..5
    for [NOOP, FIRE, RIGHT, LEFT, RIGHTFIRE, LEFTFIRE]). The ALE environment
    expects actions in its LOCAL minimal action set (0..n-1). This wrapper
    translates from union index -> ALE action -> local index.

    IMPORTANT: This wrapper must be the OUTERMOST wrapper so that all inner
    wrappers (FireResetEnv, EpisodicLifeEnv) issue local-index actions.
    """

    def __init__(self, env: gym.Env, union_actions: List[int]):
        super().__init__(env)
        # Get the minimal action set for this specific game
        minimal_actions = env.unwrapped.ale.getMinimalActionSet()
        self.ale_to_local = {int(a): i for i, a in enumerate(minimal_actions)}
        self.union_actions = union_actions

        # Override action space to union size
        self.action_space = gym.spaces.Discrete(len(union_actions))

    def step(self, action: int):
        """Translate union-space action to local action and step."""
        ale_action = self.union_actions[action]
        local_action = self.ale_to_local.get(ale_action, 0)  # fallback NOOP
        return self.env.step(local_action)


class NoopResetEnv(gym.Wrapper):
    """Execute a random number of NOOPs on reset.

    Adds stochasticity to the initial state, preventing the agent from
    memorizing fixed start positions. The NOOP action (0) is used.

    Args:
        env: The wrapped environment.
        noop_max: Maximum number of NOOPs to execute. Default 30 follows
            the DeepMind Atari benchmark convention and provides sufficient
            initial state diversity without wasting too many frames.
    """

    def __init__(self, env: gym.Env, noop_max: int = 30):
        super().__init__(env)
        self.noop_max = noop_max

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        noops = np.random.randint(1, self.noop_max + 1)
        for _ in range(noops):
            obs, _, terminated, truncated, info = self.env.step(0)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        return obs, info


class FireResetEnv(gym.Wrapper):
    """Press FIRE on reset for environments that require it (e.g. Breakout).

    Some Atari games require the FIRE action to start a new episode or life.
    This wrapper detects the FIRE action in meanings and issues it on reset.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == "FIRE"
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, terminated, truncated, _ = self.env.step(1)
        if terminated or truncated:
            obs, _ = self.env.reset(**kwargs)
        obs, _, terminated, truncated, info = self.env.step(2)
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)
        return obs, info


class EpisodicLifeEnv(gym.Wrapper):
    """Signal episode end on life loss during training.

    During training this makes every life loss a terminal signal, which helps
    the agent learn to avoid losing lives. The environment itself is NOT
    actually reset (the game continues), only the done flag is set.

    For evaluation (episodic_life=False in make_atari_env), this wrapper is
    NOT applied, so the agent plays a full game to true episode end.

    Note: This only affects games with a lives mechanic:
      - Breakout: 5 lives
      - SpaceInvaders: 3 lives
      - Pong: 0 lives (no lives mechanic, so this wrapper is inert)
    """

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
            obs, _, _, _, info = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs, info


class MaxAndSkipEnv(gym.Wrapper):
    """Return max over last 2 frames, repeat action for skip frames.

    Takes the pixel-wise maximum of the last 2 raw frames to handle
    flickering sprites (a common ALE artifact). The action is repeated
    for `skip` consecutive frames to speed up training.
    """

    def __init__(self, env: gym.Env, skip: int = 4):
        super().__init__(env)
        self._obs_buffer = np.zeros(
            (2,) + env.observation_space.shape, dtype=np.uint8
        )
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        terminated = truncated = False
        for i in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if terminated or truncated:
                break
        max_frame = self._obs_buffer.max(axis=0)
        return max_frame, total_reward, terminated, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class WarpFrame(gym.ObservationWrapper):
    """Convert RGB frames to grayscale and resize.

    Standard Atari preprocessing: 210x160 RGB -> 84x84 grayscale.
    """

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
    """Stack the last k frames as a single observation.

    Frame stacking gives the agent a sense of motion and velocity, critical
    for Atari games where a single frame is ambiguous (e.g., ball direction
    in Pong). Standard k=4 is used.
    """

    def __init__(self, env: gym.Env, k: int = 4):
        super().__init__(env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shape = env.observation_space.shape  # (H, W, 1) from WarpFrame
        # Output as channels-first: (k, H, W) for CNN and replay buffer
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

    def _get_obs(self):
        # Concatenate along last axis then transpose to channels-first (k, H, W)
        stacked = np.concatenate(list(self.frames), axis=2)  # (H, W, k)
        return np.transpose(stacked, (2, 0, 1))  # (k, H, W)


class ClipRewardEnv(gym.RewardWrapper):
    """Clip rewards to {-1, 0, +1} by sign."""

    def reward(self, reward):
        return float(np.sign(reward))


def make_atari_env(
    env_id: str,
    union_actions: List[int],
    seed: int = 42,
    frame_stack: int = 4,
    frame_skip: int = 4,
    screen_size: int = 84,
    noop_max: int = 30,
    episodic_life: bool = True,
    clip_reward: bool = True,
    render_mode: Optional[str] = None,
) -> gym.Env:
    """Create a fully-wrapped Atari environment.

    Wrapper order (inside-out):
        gym.make -> NoopResetEnv -> MaxAndSkipEnv -> [EpisodicLifeEnv]
        -> FireResetEnv -> WarpFrame -> [ClipRewardEnv] -> FrameStack
        -> UnionActionWrapper  (OUTERMOST)

    The UnionActionWrapper is outermost so that all inner wrappers
    (especially FireResetEnv) issue actions in the local action space.
    The agent interacts only through the union action space.

    Args:
        env_id: Atari env ID, e.g. "PongNoFrameskip-v4".
        union_actions: Sorted list of ALE action indices forming the union.
        seed: Random seed for environment.
        frame_stack: Number of frames to stack (default 4).
        frame_skip: Number of frames to repeat each action (default 4).
        screen_size: Resize dimension (default 84).
        noop_max: Max NOOPs on reset (default 30).
        episodic_life: If True, signal done on life loss (training only).
        clip_reward: If True, clip rewards to {-1, 0, +1}.
        render_mode: Optional render mode (e.g. "rgb_array").

    Returns:
        Fully wrapped Atari environment with union-space actions.
    """
    kwargs = {"render_mode": render_mode} if render_mode else {}
    env = gym.make(env_id, **kwargs)
    env.reset(seed=seed)

    # Inner wrappers operate on local (minimal) action space
    env = NoopResetEnv(env, noop_max=noop_max)
    env = MaxAndSkipEnv(env, skip=frame_skip)

    if episodic_life:
        env = EpisodicLifeEnv(env)

    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)

    env = WarpFrame(env, width=screen_size, height=screen_size)

    if clip_reward:
        env = ClipRewardEnv(env)

    env = FrameStack(env, k=frame_stack)

    # OUTERMOST: union action mapping (agent -> union index -> local index)
    env = UnionActionWrapper(env, union_actions=union_actions)

    return env
