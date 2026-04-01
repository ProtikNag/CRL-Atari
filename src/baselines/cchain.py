"""
C-CHAIN: Continual Churn Approximation Reduction.
ICML 2025. Adapted for DQN sequential training on classic control.

Adds a churn reduction regularizer to the standard DQN update to prevent
plasticity loss during sequential task training.  "Churn" refers to
unnecessary changes in the Q-function's action predictions for states
that were not part of the current training batch.  Accumulation of churn
across many updates degrades the effective rank of the NTK matrix,
which manifests as plasticity loss: the network's ability to learn new
tasks deteriorates over time.

The regularizer works as follows:
    1.  Sample a training batch B_train and a disjoint reference batch
        B_ref from the replay buffer.
    2.  Compute standard DQN (Huber) loss on B_train.
    3.  Compute Q-values on B_ref with the current (online) network and
        with the target network (detached).  The target network serves
        as a stable anchor: its Q-values represent what the "old" policy
        would have predicted.
    4.  The churn loss is 0.5 * mean((Q_online(B_ref) - Q_target(B_ref))^2)
        summed over valid actions only.
    5.  Total loss = DQN loss + churn_coeff * churn_loss.

Protocol (sequential continual learning on classic control):
    1.  Load expert checkpoint for Task 1 (skip training from scratch).
    2.  For k = 2, ..., K:
        a.  Continue training on Task k with DQN + churn regularization.
    3.  Save the final model checkpoint.
"""

import copy
import glob as glob_module
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.mlp_dqn import MLPDQNNetwork
from src.data.vector_replay_buffer import VectorReplayBuffer
from src.data.classic_control_wrappers import (
    make_classic_control_env,
    get_valid_actions_classic,
    compute_max_state_dim,
)
from src.utils.logger import Logger


class CChainTrainer:
    """Sequential DQN trainer with C-CHAIN churn reduction for classic control.

    Adds a churn regularization term that penalizes changes in Q-values
    for a reference batch of states, reducing the effective-rank collapse
    of the Neural Tangent Kernel that causes plasticity loss in sequential
    continual RL.

    For Task 1, loads an expert checkpoint.
    For Tasks 2+, trains DQN with the churn regularizer active.

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

        cchain_cfg = config.get("cchain", {})
        self.churn_coeff: float = cchain_cfg.get("churn_coeff", 0.1)
        self.ref_batch_size: int = cchain_cfg.get("ref_batch_size", 64)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def train_sequential(
        self,
        task_sequence: List[str],
        first_task_checkpoint: Optional[str] = None,
    ) -> MLPDQNNetwork:
        """Train the model sequentially on all tasks with C-CHAIN.

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
        self.logger.info("C-CHAIN — Sequential Training (Classic Control)")
        self.logger.info("=" * 60)
        self.logger.info(f"Tasks: {task_sequence}")
        self.logger.info(f"Churn coefficient: {self.churn_coeff}")
        self.logger.info(f"Reference batch size: {self.ref_batch_size}")

        for task_idx, env_id in enumerate(task_sequence):
            game_name = env_id.replace("-v1", "").replace("-v2", "")
            valid_actions = get_valid_actions_classic(env_id, self.union_actions)

            self.logger.info(f"\n{'=' * 60}")
            self.logger.info(
                f"C-CHAIN Task {task_idx + 1}/{len(task_sequence)}: {env_id}"
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

            # ── Tasks 2+ (or Task 1 without checkpoint): Train with C-CHAIN

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

            # Fresh replay buffer (needs to be large enough for both
            # train batch and reference batch sampling)
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

            # Minimum buffer fill must accommodate both batches
            effective_min_buffer = max(
                min_buffer, batch_size + self.ref_batch_size
            )

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

                # ── Training step with C-CHAIN ──────────────────────────
                if (
                    step % train_freq == 0
                    and len(replay_buffer) >= effective_min_buffer
                ):
                    # Sample training batch
                    states_b, actions_b, rewards_b, next_b, dones_b = (
                        replay_buffer.sample(batch_size)
                    )

                    # Compute standard DQN loss on B_train
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

                    dqn_loss = F.smooth_l1_loss(q_taken, target_vals)

                    # Sample disjoint reference batch for churn computation
                    ref_states, _, _, _, _ = replay_buffer.sample(
                        self.ref_batch_size
                    )

                    # Compute churn regularization on B_ref
                    # Current (online) Q-values on reference states
                    q_ref_online = model(ref_states)

                    # Stable Q-values from target network (detached anchor)
                    with torch.no_grad():
                        q_ref_target = target_net(ref_states)

                    # Churn = change in Q-values over valid actions only
                    # Shape: (ref_batch_size, |valid_actions|)
                    churn = (
                        q_ref_online[:, valid_actions]
                        - q_ref_target[:, valid_actions]
                    )

                    # L_churn = 0.5 * mean(churn^2) summed over actions
                    churn_loss = 0.5 * (churn ** 2).mean()

                    # Total loss
                    total_loss = dqn_loss + self.churn_coeff * churn_loss

                    optimizer.zero_grad()
                    total_loss.backward()
                    nn.utils.clip_grad_norm_(
                        model.parameters(), gradient_clip
                    )
                    optimizer.step()

                    train_steps += 1

                    # Target network update
                    if train_steps % target_update_freq == 0:
                        target_net.load_state_dict(model.state_dict())

                    # Logging
                    if step % log_interval == 0:
                        self.logger.log_scalars(
                            {
                                f"cchain/{game_name}/dqn_loss": dqn_loss.item(),
                                f"cchain/{game_name}/churn_loss": churn_loss.item(),
                                f"cchain/{game_name}/total_loss": total_loss.item(),
                                f"cchain/{game_name}/epsilon": eps,
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
                            f"[CCHAIN-{game_name}] Step {step}/{total_timesteps}"
                            f" | Ep {episode_count}"
                            f" | Reward: {episode_reward:.1f}"
                            f" | Mean100: {recent_mean:.1f}"
                            f" | Eps: {eps:.3f}"
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
                        f"[CCHAIN-{game_name}] Eval at step {step}: "
                        f"mean reward = {eval_reward:.2f}"
                    )
                    if eval_reward > best_eval_reward:
                        best_eval_reward = eval_reward
                        self.logger.info(
                            f"[CCHAIN-{game_name}] New best: {eval_reward:.2f}"
                        )

                # Periodic checkpoint
                if step % save_freq == 0:
                    ckpt_path = os.path.join(
                        checkpoint_dir,
                        f"cchain_task{task_idx}_{game_name}_step{step}.pt",
                    )
                    torch.save(model.state_dict(), ckpt_path)

            env.close()

            # Final evaluation on current task
            final_eval = self._evaluate_task(
                model, env_id, valid_actions, max_state_dim, eval_episodes
            )
            self.logger.info(
                f"[CCHAIN-{game_name}] Training complete. "
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
                    f"cchain_task{task_idx}_{game_name}_step*.pt",
                )
            )
            for ckpt_path in step_ckpts:
                os.remove(ckpt_path)

        # Save final consolidated model (bare state_dict)
        final_path = os.path.join(checkpoint_dir, "consolidated_cchain.pt")
        torch.save(model.state_dict(), final_path)
        self.logger.info(f"Final C-CHAIN model saved to {final_path}")

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
            checkpoint_dir, f"cchain_after_task{task_idx}_{game_name}.pt"
        )
        torch.save(model.state_dict(), path)
        self.logger.info(f"Post-task checkpoint saved: {path}")
