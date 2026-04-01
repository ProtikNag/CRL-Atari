"""
Classic control environment utilities for CRL.

Handles union action space computation, valid action mapping, and state
zero-padding across tasks with different observation dimensionalities.
CartPole has 4-dim states, Acrobot has 6-dim, and LunarLander has 8-dim.
All states are zero-padded to a common ``max_state_dim`` (8 by default)
so that a single MLP can process every task.
"""

import gymnasium as gym
import numpy as np
from typing import Dict, List


# ---------------------------------------------------------------------------
# Task metadata
# ---------------------------------------------------------------------------
# Each entry stores the native observation dimension and number of discrete
# actions for the corresponding Gymnasium environment.
TASK_INFO: Dict[str, Dict[str, int]] = {
    "CartPole-v1": {"state_dim": 4, "n_actions": 2},
    "Acrobot-v1": {"state_dim": 6, "n_actions": 3},
    "LunarLander-v2": {"state_dim": 8, "n_actions": 4},
    "LunarLander-v3": {"state_dim": 8, "n_actions": 4},
}


# ---------------------------------------------------------------------------
# Action space helpers
# ---------------------------------------------------------------------------
def compute_union_action_space_classic(task_sequence: List[str]) -> List[int]:
    """Compute the union of action spaces across classic control tasks.

    All supported environments use sequential integer actions starting from
    0, so the union is simply ``range(max_actions)``.

    Args:
        task_sequence: List of environment IDs,
            e.g. ``["CartPole-v1", "Acrobot-v1", "LunarLander-v2"]``.

    Returns:
        Sorted list of action indices covering every task in the sequence.
    """
    max_actions = max(TASK_INFO[env_id]["n_actions"] for env_id in task_sequence)
    return list(range(max_actions))


def get_valid_actions_classic(
    env_id: str, union_actions: List[int]
) -> List[int]:
    """Return indices into the union action list that are valid for *env_id*.

    For example, CartPole has 2 native actions (0 and 1). If the union is
    ``[0, 1, 2, 3]``, the valid indices are ``[0, 1]``.

    Args:
        env_id: Classic control environment ID.
        union_actions: The union action list produced by
            :func:`compute_union_action_space_classic`.

    Returns:
        List of indices ``i`` into *union_actions* where
        ``union_actions[i] < n_native_actions``.
    """
    n_actions = TASK_INFO[env_id]["n_actions"]
    return [i for i in range(len(union_actions)) if union_actions[i] < n_actions]


# ---------------------------------------------------------------------------
# State dimension helpers
# ---------------------------------------------------------------------------
def get_state_dim(env_id: str) -> int:
    """Return the native observation dimension for *env_id*.

    Args:
        env_id: Classic control environment ID.

    Returns:
        Integer dimensionality of the raw observation vector.
    """
    return TASK_INFO[env_id]["state_dim"]


def compute_max_state_dim(task_sequence: List[str]) -> int:
    """Return the maximum observation dimension across all tasks.

    This value is used as the zero-pad target so that every task produces
    observations of the same shape.

    Args:
        task_sequence: List of environment IDs.

    Returns:
        The largest ``state_dim`` found in the sequence.
    """
    return max(TASK_INFO[env_id]["state_dim"] for env_id in task_sequence)


# ---------------------------------------------------------------------------
# Environment wrappers
# ---------------------------------------------------------------------------
class ZeroPadWrapper(gym.ObservationWrapper):
    """Zero-pad observations to a fixed dimension.

    Converts ``Box(shape=(native_dim,))`` to ``Box(shape=(target_dim,))``.
    Observations shorter than *target_dim* are padded with zeros on the
    right; observations already at the target length are cast to float32
    and returned unchanged.

    Args:
        env: A Gymnasium environment whose observation space is a 1-D Box.
        target_dim: Desired observation length after padding.
    """

    def __init__(self, env: gym.Env, target_dim: int):
        super().__init__(env)
        self.target_dim = target_dim
        self.native_dim = env.observation_space.shape[0]
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(target_dim,), dtype=np.float32
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        """Pad *obs* to ``self.target_dim`` with trailing zeros.

        Args:
            obs: Raw observation from the inner environment.

        Returns:
            Float32 array of length ``target_dim``.
        """
        if self.native_dim == self.target_dim:
            return obs.astype(np.float32)
        padded = np.zeros(self.target_dim, dtype=np.float32)
        padded[: self.native_dim] = obs
        return padded


class UnionActionWrapperClassic(gym.Wrapper):
    """Map union action indices to environment-native actions.

    The agent selects actions in the union action space (size = max actions
    across all tasks). This wrapper translates each union index to the
    corresponding native action. If the union index exceeds the
    environment's native action count, action 0 is used as a safe
    fallback (analogous to NOOP in Atari).

    Args:
        env: A Gymnasium environment with a ``Discrete`` action space.
        union_actions: The union action list (sorted integers).
    """

    def __init__(self, env: gym.Env, union_actions: List[int]):
        super().__init__(env)
        self.union_actions = union_actions
        self.n_native = env.action_space.n
        # Override action space to union size
        self.action_space = gym.spaces.Discrete(len(union_actions))

    def step(self, action: int):
        """Translate *action* from union space and step the inner env.

        Args:
            action: Index into the union action list.

        Returns:
            Standard Gymnasium ``(obs, reward, terminated, truncated, info)``
            tuple.
        """
        native_action = self.union_actions[action]
        if native_action >= self.n_native:
            native_action = 0  # safe fallback for out-of-range actions
        return self.env.step(native_action)


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------
def make_classic_control_env(
    env_id: str,
    union_actions: List[int],
    max_state_dim: int = 8,
    seed: int = 42,
) -> gym.Env:
    """Create a fully-wrapped classic control environment.

    Wrapper order (inside-out)::

        gym.make -> ZeroPadWrapper -> UnionActionWrapperClassic

    Args:
        env_id: Gymnasium environment ID, e.g. ``"CartPole-v1"``.
        union_actions: Sorted union action list produced by
            :func:`compute_union_action_space_classic`.
        max_state_dim: Target observation dimension for zero-padding.
        seed: Random seed passed to the first ``env.reset()``.

    Returns:
        Wrapped environment with padded observations and union-space actions.
    """
    env = gym.make(env_id)
    env.reset(seed=seed)
    env = ZeroPadWrapper(env, target_dim=max_state_dim)
    env = UnionActionWrapperClassic(env, union_actions=union_actions)
    return env
