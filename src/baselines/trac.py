"""
TRAC: A Parameter-Free Optimizer for Lifelong Reinforcement Learning.
NeurIPS 2024. Adapted for DQN sequential training on classic control.

Wraps the base optimizer (Adam) with adaptive L2 regularization using
the Erfi potential function.  At each gradient step, TRAC computes a
scalar inner product h_t = <g_t, theta_t - theta_ref> and feeds it to
a bank of 1-D tuners (one per discount factor beta).  Each tuner
maintains a running variance and sum that together produce a scaling
factor via the erfi potential.  The total scaling S_{t+1} blends the
base optimizer's proposal toward or away from the reference point.

The reference point theta_ref is reset at the START of each new task
(set to the current parameter values), so TRAC naturally anchors
learning to the prior task's solution without requiring Fisher matrices
or explicit regularization weights.

Protocol (sequential continual learning on classic control):
    1.  Load expert checkpoint for Task 1 (skip training from scratch).
    2.  Set theta_ref = theta_1*.
    3.  For k = 2, ..., K:
        a.  Continue training on Task k with DQN loss.
        b.  At each gradient step, Adam proposes theta_base; TRAC scales
            the displacement (theta_base - theta_ref) by S_{t+1}.
        c.  Set theta_ref = current params for next task.
    4.  Save the final model checkpoint.
"""

import copy
import glob as glob_module
import math
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from src.models.mlp_dqn import MLPDQNNetwork
from src.data.vector_replay_buffer import VectorReplayBuffer
from src.data.classic_control_wrappers import (
    make_classic_control_env,
    get_valid_actions_classic,
    compute_max_state_dim,
)
from src.utils.logger import Logger


# ---------------------------------------------------------------------------
# Erfi helper
# ---------------------------------------------------------------------------
def erfi(x: float) -> float:
    """Imaginary error function: erfi(x) = -i * erf(ix).

    Equivalent to (2/sqrt(pi)) * integral_0^x exp(t^2) dt.
    Uses scipy when available; otherwise falls back to a Taylor series
    approximation that converges well for |x| <= 3.

    Args:
        x: Real-valued input.

    Returns:
        erfi(x) as a Python float.
    """
    try:
        from scipy.special import erfi as _erfi
        return float(_erfi(x))
    except ImportError:
        # Taylor series: erfi(x) = (2/sqrt(pi)) * sum_{n>=0} x^{2n+1} / (n! * (2n+1))
        result = 0.0
        term = x
        for n in range(50):
            result += term / (2 * n + 1)
            term *= x * x / (n + 1)
        return result * 2.0 / math.sqrt(math.pi)


# ---------------------------------------------------------------------------
# Single 1-D tuner
# ---------------------------------------------------------------------------
class TRACTuner:
    """Single 1-D TRAC tuner with discount factor beta.

    Maintains running variance v and sum sigma. At each step, given the
    scalar h_t = <g_t, theta_t - theta_ref>, produces a scaling factor
    s via the erfi potential.

    Args:
        beta: Discount factor in [0, 1).
        epsilon: Small constant for numerical stability.
    """

    def __init__(self, beta: float, epsilon: float = 1e-8):
        self.beta = beta
        self.epsilon = epsilon
        self.v = 0.0    # running variance
        self.sigma = 0.0  # running signed sum
        self._erfi_inv_sqrt2 = erfi(1.0 / math.sqrt(2.0))

    def step(self, h: float) -> float:
        """Process scalar h_t and return the scaling contribution.

        Args:
            h: Inner product <g_t, theta_t - theta_ref>.

        Returns:
            Scaling factor s_{t+1} for this tuner.
        """
        self.v = self.beta ** 2 * self.v + h ** 2
        self.sigma = self.beta * self.sigma - h

        denom = math.sqrt(2.0 * self.v) + self.epsilon
        arg = self.sigma / denom
        # Clamp to prevent overflow in erfi
        arg = max(-3.0, min(3.0, arg))

        s = (self.epsilon / self._erfi_inv_sqrt2) * erfi(arg)
        return s

    def reset(self) -> None:
        """Reset internal state (called at each new task)."""
        self.v = 0.0
        self.sigma = 0.0


# ---------------------------------------------------------------------------
# TRAC Trainer
# ---------------------------------------------------------------------------
class TRACTrainer:
    """Sequential DQN trainer with TRAC optimizer wrapping for classic control.

    For Task 1, loads an expert checkpoint (or trains from scratch).
    For Tasks 2+, trains DQN with TRAC wrapping Adam.  The reference
    point theta_ref is set to the parameter snapshot at the start of
    each new task.

    Args:
        config: Full configuration dictionary (loaded from classic_control.yaml).
        union_actions: Sorted list of union action indices.
        device: Torch device string.
        logger: Logger instance.
        tag: Experiment tag for checkpoint naming.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        union_actions: List[int],
        device: str,
        logger: Logger,
        tag: str = "default",
    ):
        self.config = config
        self.union_actions = union_actions
        self.device = device
        self.logger = logger
        self.tag = tag

        trac_cfg = config.get("trac", {})
        self.beta_grid: List[float] = trac_cfg.get(
            "beta_grid", [0.9, 0.99, 0.999, 0.9999]
        )
        self.epsilon: float = trac_cfg.get("epsilon", 1e-8)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def train_sequential(
        self,
        task_sequence: List[str],
        first_task_checkpoint: Optional[str] = None,
    ) -> MLPDQNNetwork:
        """Train the model sequentially on all tasks with TRAC.

        Args:
            task_sequence: List of env_ids
                (e.g. ``['CartPole-v1', 'Acrobot-v1', 'LunarLander-v2']``).
            first_task_checkpoint: Path to an existing expert checkpoint for
                Task 1.  If provided, training on Task 1 is skipped and the
                expert weights are loaded directly.

        Returns:
            The trained MLPDQNNetwork after all tasks.
        """
        train_cfg = self.config["training"]
        model_cfg = self.config["model"]
        debug_cfg = self.config.get("debug", {})
        debug_on = debug_cfg.get("enabled", False)

        total_timesteps = (
            debug_cfg.get("total_timesteps", train_cfg["total_timesteps"])
            if debug_on else train_cfg["total_timesteps"]
        )
        eval_freq = (
            debug_cfg.get("eval_freq", train_cfg["eval_freq"])
            if debug_on else train_cfg["eval_freq"]
        )
        eval_episodes = (
            debug_cfg.get("eval_episodes", train_cfg["eval_episodes"])
            if debug_on else train_cfg["eval_episodes"]
        )
        save_freq = (
            debug_cfg.get("save_freq", train_cfg["save_freq"])
            if debug_on else train_cfg["save_freq"]
        )
        buffer_size = (
            debug_cfg.get("buffer_size", train_cfg["buffer_size"])
            if debug_on else train_cfg["buffer_size"]
        )
        min_buffer = (
            debug_cfg.get("min_buffer_size", train_cfg["min_buffer_size"])
            if debug_on else train_cfg["min_buffer_size"]
        )
        train_freq = train_cfg["train_freq"]
        log_interval = self.config["logging"].get("log_interval", 500)

        max_state_dim = self.config.get(
            "max_state_dim",
            compute_max_state_dim(task_sequence),
        )

        checkpoint_dir = os.path.join(
            self.config["logging"]["checkpoint_dir"], self.tag
        )
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Build the single MLP model that will be trained across all tasks
        unified_action_dim = len(self.union_actions)
        model = MLPDQNNetwork(
            state_dim=max_state_dim,
            hidden_dims=model_cfg.get("hidden_dims", [128, 128]),
            unified_action_dim=unified_action_dim,
            dueling=model_cfg.get("dueling", False),
        ).to(self.device)

        # Target network
        target_net = copy.deepcopy(model)
        target_net.eval()

        self.logger.info("\n" + "=" * 60)
        self.logger.info("TRAC — Sequential Training (Classic Control)")
        self.logger.info("=" * 60)
        self.logger.info(f"Tasks: {task_sequence}")
        self.logger.info(f"Beta grid: {self.beta_grid}")
        self.logger.info(f"Epsilon: {self.epsilon}")

        for task_idx, env_id in enumerate(task_sequence):
            game_name = env_id.replace("-v1", "").replace("-v2", "")
            valid_actions = get_valid_actions_classic(env_id, self.union_actions)

            self.logger.info(f"\n{'=' * 60}")
            self.logger.info(
                f"TRAC Task {task_idx + 1}/{len(task_sequence)}: {env_id}"
            )
            self.logger.info(f"Valid actions: {valid_actions}")
            self.logger.info(f"{'=' * 60}")

            # ── Task 1: Load from expert checkpoint ─────────────────────
            if task_idx == 0 and first_task_checkpoint is not None:
                self.logger.info(
                    f"Loading Task 1 from expert: {first_task_checkpoint}"
                )
                ckpt = torch.load(
                    first_task_checkpoint,
                    map_location=self.device,
                    weights_only=False,
                )
                if isinstance(ckpt, dict) and "policy_net" in ckpt:
                    model.load_state_dict(ckpt["policy_net"])
                else:
                    model.load_state_dict(ckpt)
                target_net.load_state_dict(model.state_dict())

                self._save_task_checkpoint(
                    model, checkpoint_dir, game_name, task_idx
                )
                self.logger.info(
                    f"Task 1 ({game_name}) loaded from expert checkpoint."
                )
                continue

            # ── Tasks 2+ (or Task 1 without checkpoint): Train with TRAC ──

            # Snapshot reference point: theta_ref = current params
            theta_ref = {
                name: param.detach().clone()
                for name, param in model.named_parameters()
            }

            # Fresh tuners for this task
            tuners = [
                TRACTuner(beta=b, epsilon=self.epsilon) for b in self.beta_grid
            ]

            # Create environment
            env = make_classic_control_env(
                env_id=env_id,
                union_actions=self.union_actions,
                max_state_dim=max_state_dim,
                seed=self.config["seed"] + task_idx * 1000,
            )

            # Fresh optimizer
            optimizer = torch.optim.Adam(
                model.parameters(), lr=train_cfg["learning_rate"]
            )

            # Fresh replay buffer
            replay_buffer = VectorReplayBuffer(
                capacity=buffer_size,
                state_dim=max_state_dim,
                device=self.device,
            )

            # Action mask
            action_mask = torch.full(
                (unified_action_dim,), float("-inf"), device=self.device
            )
            action_mask[valid_actions] = 0.0

            # Exploration schedule
            eps_start = train_cfg["eps_start"]
            eps_end = train_cfg["eps_end"]
            eps_decay = train_cfg["eps_decay_steps"]
            gamma = train_cfg["gamma"]
            batch_size = train_cfg["batch_size"]
            target_update_freq = train_cfg["target_update_freq"]
            double_dqn = train_cfg.get("double_dqn", False)
            gradient_clip = train_cfg.get("gradient_clip", 10.0)

            state, _ = env.reset()
            episode_reward = 0.0
            episode_count = 0
            episode_rewards: List[float] = []
            best_eval_reward = float("-inf")
            train_steps = 0
            start_time = time.time()

            for step in range(1, total_timesteps + 1):
                # Epsilon-greedy action selection with masking
                eps = eps_start + min(step / eps_decay, 1.0) * (
                    eps_end - eps_start
                )
                if np.random.random() < eps:
                    action = np.random.choice(valid_actions)
                else:
                    with torch.no_grad():
                        s_t = (
                            torch.from_numpy(state)
                            .float()
                            .unsqueeze(0)
                            .to(self.device)
                        )
                        q = model(s_t) + action_mask.unsqueeze(0)
                        action = q.argmax(dim=1).item()

                next_state, reward, terminated, truncated, info = env.step(
                    action
                )
                done = terminated or truncated
                replay_buffer.push(state, action, reward, next_state, done)
                episode_reward += reward
                state = next_state

                # ── Training step with TRAC ─────────────────────────────
                if (
                    step % train_freq == 0
                    and len(replay_buffer) >= min_buffer
                ):
                    states_b, actions_b, rewards_b, next_b, dones_b = (
                        replay_buffer.sample(batch_size)
                    )

                    # Compute DQN loss
                    q_vals = model(states_b)
                    q_taken = q_vals.gather(
                        1, actions_b.unsqueeze(1)
                    ).squeeze(1)

                    with torch.no_grad():
                        if double_dqn:
                            next_q_policy = (
                                model(next_b) + action_mask.unsqueeze(0)
                            )
                            next_acts = next_q_policy.argmax(
                                dim=1, keepdim=True
                            )
                            next_q_target = target_net(next_b)
                            next_q_max = next_q_target.gather(
                                1, next_acts
                            ).squeeze(1)
                        else:
                            next_q = (
                                target_net(next_b) + action_mask.unsqueeze(0)
                            )
                            next_q_max = next_q.max(dim=1)[0]
                        target_vals = (
                            rewards_b + gamma * next_q_max * (1 - dones_b)
                        )

                    dqn_loss = nn.functional.smooth_l1_loss(
                        q_taken, target_vals
                    )

                    # Step 1: Compute gradient and let Adam propose update
                    optimizer.zero_grad()
                    dqn_loss.backward()
                    nn.utils.clip_grad_norm_(
                        model.parameters(), gradient_clip
                    )

                    # Flatten current gradient before Adam steps
                    grad_flat = torch.cat([
                        p.grad.detach().reshape(-1)
                        for p in model.parameters()
                        if p.grad is not None
                    ])

                    # Flatten (theta_t - theta_ref) before Adam steps
                    diff_flat = torch.cat([
                        (p.detach() - theta_ref[n]).reshape(-1)
                        for n, p in model.named_parameters()
                    ])

                    # Compute h_t = <g_t, theta_t - theta_ref>
                    h_t = float(torch.dot(grad_flat, diff_flat).item())

                    # Step 2: Let Adam propose theta_base
                    optimizer.step()

                    # Step 3: Update tuners and compute total scaling S
                    total_scaling = sum(tuner.step(h_t) for tuner in tuners)

                    # Step 4: Apply TRAC correction
                    # theta_{t+1} = theta_ref + (theta_base - theta_ref) * S
                    with torch.no_grad():
                        for name, param in model.named_parameters():
                            displacement = param.data - theta_ref[name]
                            param.data.copy_(
                                theta_ref[name] + displacement * total_scaling
                            )

                    train_steps += 1

                    # Target network update
                    if train_steps % target_update_freq == 0:
                        target_net.load_state_dict(model.state_dict())

                    # Logging
                    if step % log_interval == 0:
                        self.logger.log_scalars(
                            {
                                f"trac/{game_name}/dqn_loss": dqn_loss.item(),
                                f"trac/{game_name}/h_t": h_t,
                                f"trac/{game_name}/total_scaling": total_scaling,
                                f"trac/{game_name}/epsilon": eps,
                            },
                            step=step,
                        )

                # Episode end
                if done:
                    episode_count += 1
                    episode_rewards.append(episode_reward)
                    if step % log_interval == 0 or episode_count % 10 == 0:
                        recent_mean = np.mean(episode_rewards[-100:])
                        elapsed = time.time() - start_time
                        self.logger.info(
                            f"[TRAC-{game_name}] Step {step}/{total_timesteps}"
                            f" | Ep {episode_count}"
                            f" | Reward: {episode_reward:.1f}"
                            f" | Mean100: {recent_mean:.1f}"
                            f" | Eps: {eps:.3f}"
                            f" | S: {total_scaling if train_steps > 0 else 0:.4f}"
                            f" | Time: {elapsed:.0f}s"
                        )
                    episode_reward = 0.0
                    state, _ = env.reset()

                # Evaluate
                if step % eval_freq == 0:
                    eval_reward = self._evaluate_task(
                        model, env_id, valid_actions, max_state_dim,
                        eval_episodes,
                    )
                    self.logger.info(
                        f"[TRAC-{game_name}] Eval at step {step}: "
                        f"mean reward = {eval_reward:.2f}"
                    )
                    if eval_reward > best_eval_reward:
                        best_eval_reward = eval_reward
                        self.logger.info(
                            f"[TRAC-{game_name}] New best: {eval_reward:.2f}"
                        )

                # Periodic checkpoint
                if step % save_freq == 0:
                    ckpt_path = os.path.join(
                        checkpoint_dir,
                        f"trac_task{task_idx}_{game_name}_step{step}.pt",
                    )
                    torch.save(model.state_dict(), ckpt_path)

            env.close()

            # Final evaluation on current task
            final_eval = self._evaluate_task(
                model, env_id, valid_actions, max_state_dim, eval_episodes
            )
            self.logger.info(
                f"[TRAC-{game_name}] Training complete. "
                f"Final: {final_eval:.2f} | Best: {best_eval_reward:.2f}"
            )

            # Save post-task checkpoint
            self._save_task_checkpoint(
                model, checkpoint_dir, game_name, task_idx
            )

            # Cross-evaluate on all previously seen tasks
            self.logger.info("Cross-evaluation on prior tasks:")
            for prev_idx in range(task_idx + 1):
                prev_env_id = task_sequence[prev_idx]
                prev_game = prev_env_id.replace("-v1", "").replace("-v2", "")
                prev_valid = get_valid_actions_classic(
                    prev_env_id, self.union_actions
                )
                prev_reward = self._evaluate_task(
                    model, prev_env_id, prev_valid, max_state_dim,
                    eval_episodes,
                )
                self.logger.info(f"  {prev_game}: {prev_reward:.2f}")

            # Clean up step checkpoints
            step_ckpts = glob_module.glob(
                os.path.join(
                    checkpoint_dir,
                    f"trac_task{task_idx}_{game_name}_step*.pt",
                )
            )
            for ckpt_path in step_ckpts:
                os.remove(ckpt_path)

        # Save final consolidated model (bare state_dict)
        final_path = os.path.join(checkpoint_dir, "consolidated_trac.pt")
        torch.save(model.state_dict(), final_path)
        self.logger.info(f"Final TRAC model saved to {final_path}")

        return model

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def _evaluate_task(
        self,
        model: MLPDQNNetwork,
        env_id: str,
        valid_actions: List[int],
        max_state_dim: int,
        num_episodes: int = 20,
    ) -> float:
        """Evaluate the model on a single classic control task.

        Uses deterministic (greedy) action selection with no exploration.

        Args:
            model: The MLP DQN model to evaluate.
            env_id: Gymnasium environment ID.
            valid_actions: Indices into the union action space valid for this task.
            max_state_dim: Zero-pad target dimension.
            num_episodes: Number of evaluation episodes.

        Returns:
            Mean total reward across episodes.
        """
        env = make_classic_control_env(
            env_id=env_id,
            union_actions=self.union_actions,
            max_state_dim=max_state_dim,
            seed=self.config["seed"] + 3000,
        )

        model.eval()
        action_mask = torch.full(
            (model.unified_action_dim,), float("-inf"), device=self.device
        )
        action_mask[valid_actions] = 0.0

        rewards = []
        for _ in range(num_episodes):
            state, _ = env.reset()
            total_reward = 0.0
            done = False
            steps = 0
            while not done and steps < 10000:
                with torch.no_grad():
                    s_t = (
                        torch.from_numpy(state)
                        .float()
                        .unsqueeze(0)
                        .to(self.device)
                    )
                    q = model(s_t) + action_mask.unsqueeze(0)
                    action = q.argmax(dim=1).item()
                state, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                done = terminated or truncated
                steps += 1
            rewards.append(total_reward)

        env.close()
        model.train()
        return float(np.mean(rewards))

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------
    def _save_task_checkpoint(
        self,
        model: MLPDQNNetwork,
        checkpoint_dir: str,
        game_name: str,
        task_idx: int,
    ) -> None:
        """Save model state_dict after completing a task.

        Args:
            model: The model to save.
            checkpoint_dir: Directory for checkpoint files.
            game_name: Human-readable task name.
            task_idx: Zero-based task index.
        """
        path = os.path.join(
            checkpoint_dir, f"trac_after_task{task_idx}_{game_name}.pt"
        )
        torch.save(model.state_dict(), path)
        self.logger.info(f"Post-task checkpoint saved: {path}")
