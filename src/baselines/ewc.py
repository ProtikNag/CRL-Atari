"""
Elastic Weight Consolidation (EWC) for Sequential Atari DQN.

Implements the EWC penalty from Kirkpatrick et al., "Overcoming catastrophic
forgetting in neural networks" (PNAS, 2017).  After each task the diagonal
Fisher information matrix is computed from a replay buffer of high-confidence
states, then stored alongside the parameter snapshot.  During training on
subsequent tasks the Fisher-weighted L2 penalty anchors parameters toward
all previous task solutions.

Protocol (sequential continual learning):
    1.  Load expert checkpoint for Task 1 (skip training from scratch).
    2.  Compute diagonal Fisher F_1 at theta_1*.
    3.  For k = 2, ..., K:
        a.  Continue training on Task k with DQN loss + EWC penalty.
        b.  Compute diagonal Fisher F_k at theta_k*.
    4.  Save the final model checkpoint.

The EWC penalty for the current parameters theta is:

    L_EWC(theta) = (lambda / 2) * sum_i sum_j  F_i^j * (theta_j - theta_i*_j)^2

where i ranges over all completed tasks and j over all parameters.

Online EWC variant (Schwarz et al., 2018) is also supported: instead of
storing per-task Fishers, a running average Fisher is maintained with
exponential decay gamma_ewc.
"""

import copy
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from src.agents.dqn_agent import DQNAgent
from src.data.atari_wrappers import make_atari_env, get_valid_actions
from src.models.dqn import DQNNetwork
from src.utils.logger import Logger


def compute_diagonal_fisher(
    model: DQNNetwork,
    replay_states: torch.Tensor,
    valid_actions: List[int],
    device: str,
    num_samples: int = 5000,
) -> Dict[str, torch.Tensor]:
    """Compute the diagonal Fisher information matrix from replay states.

    For each sampled state the Fisher is approximated as the squared gradient
    of the log-likelihood of the greedy action:

        F_j = E[ (d log pi(a*|s) / d theta_j)^2 ]

    where a* = argmax_a Q(s, a) over valid actions.

    Args:
        model: DQN network (placed in eval mode internally).
        replay_states: Tensor of shape (N, C, H, W), uint8 or float.
        valid_actions: Indices of valid actions for this task.
        device: Torch device string.
        num_samples: Number of states to sample for Fisher estimation.

    Returns:
        Dictionary mapping parameter name to diagonal Fisher tensor (same
        shape as the parameter).
    """
    model.eval()
    fisher: Dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        fisher[name] = torch.zeros_like(param, device=device)

    n = min(num_samples, replay_states.shape[0])
    indices = torch.randperm(replay_states.shape[0])[:n]

    action_mask = torch.full(
        (model.unified_action_dim,), float("-inf"), device=device
    )
    action_mask[valid_actions] = 0.0

    for idx in indices:
        state = replay_states[idx].unsqueeze(0).float().to(device) / 255.0
        model.zero_grad()

        q_values = model(state)
        masked_q = q_values + action_mask.unsqueeze(0)
        best_action = masked_q.argmax(dim=1)

        # Log-softmax over valid actions gives log pi(a|s)
        valid_q = q_values[0, valid_actions]
        log_probs = torch.log_softmax(valid_q, dim=0)

        # Find index of best_action within valid_actions
        best_in_valid = valid_actions.index(best_action.item())
        log_prob = log_probs[best_in_valid]
        log_prob.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                fisher[name] += param.grad.detach() ** 2

    # Average over samples
    for name in fisher:
        fisher[name] /= n

    model.train()
    return fisher


class EWCTrainer:
    """Sequential DQN trainer with EWC regularization.

    Trains a single DQN model across multiple Atari tasks sequentially.
    After each task, the diagonal Fisher and parameter snapshot are stored.
    The EWC penalty penalizes deviation from all prior task solutions.

    Args:
        config: Full configuration dictionary.
        union_actions: Sorted list of ALE action indices forming the union.
        device: Torch device string.
        logger: Logger instance.
        tag: Experiment tag for checkpoint naming.
        online_ewc: If True, use Online EWC (running Fisher average).
        gamma_ewc: Decay factor for Online EWC Fisher accumulation.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        union_actions: List[int],
        device: str,
        logger: Logger,
        tag: str = "default",
        online_ewc: bool = False,
        gamma_ewc: float = 0.95,
    ):
        self.config = config
        self.union_actions = union_actions
        self.device = device
        self.logger = logger
        self.tag = tag
        self.online_ewc = online_ewc
        self.gamma_ewc = gamma_ewc

        ewc_cfg = config.get("ewc", {})
        self.ewc_lambda = ewc_cfg.get("lambda", 5000.0)
        self.fisher_samples = ewc_cfg.get("fisher_samples", 5000)

        # Storage for completed tasks
        self.task_fishers: List[Dict[str, torch.Tensor]] = []
        self.task_params: List[Dict[str, torch.Tensor]] = []

        # For Online EWC: single accumulated Fisher
        self.online_fisher: Optional[Dict[str, torch.Tensor]] = None
        self.online_params: Optional[Dict[str, torch.Tensor]] = None

    def _ewc_penalty(self, model: DQNNetwork) -> torch.Tensor:
        """Compute the EWC penalty term for the current model parameters.

        For standard EWC:
            L = (lambda/2) * sum_task sum_param  F[p] * (theta[p] - theta*[p])^2

        For Online EWC:
            L = (lambda/2) * sum_param  F_online[p] * (theta[p] - theta_online*[p])^2

        Returns:
            Scalar tensor containing the EWC penalty.
        """
        penalty = torch.tensor(0.0, device=self.device)

        if self.online_ewc:
            if self.online_fisher is None:
                return penalty
            for name, param in model.named_parameters():
                if name in self.online_fisher:
                    diff = param - self.online_params[name]
                    penalty += (self.online_fisher[name] * diff ** 2).sum()
        else:
            for fisher, params_star in zip(self.task_fishers, self.task_params):
                for name, param in model.named_parameters():
                    if name in fisher:
                        diff = param - params_star[name]
                        penalty += (fisher[name] * diff ** 2).sum()

        return (self.ewc_lambda / 2.0) * penalty

    def _register_task(
        self, model: DQNNetwork, fisher: Dict[str, torch.Tensor]
    ) -> None:
        """Store Fisher and parameter snapshot after completing a task."""
        params_snapshot = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
        }

        if self.online_ewc:
            if self.online_fisher is None:
                self.online_fisher = {k: v.clone() for k, v in fisher.items()}
                self.online_params = params_snapshot
            else:
                for name in self.online_fisher:
                    self.online_fisher[name] = (
                        self.gamma_ewc * self.online_fisher[name] + fisher[name]
                    )
                self.online_params = params_snapshot
        else:
            self.task_fishers.append(fisher)
            self.task_params.append(params_snapshot)

    def train_sequential(
        self,
        task_sequence: List[str],
        first_task_checkpoint: Optional[str] = None,
    ) -> DQNNetwork:
        """Train the model sequentially on all tasks in the sequence.

        Args:
            task_sequence: List of env_ids (e.g., ['BreakoutNoFrameskip-v4', ...]).
            first_task_checkpoint: Path to an existing expert checkpoint for
                Task 1. If provided, training on Task 1 is skipped and the
                expert weights are loaded directly.

        Returns:
            The trained DQN model after all tasks.
        """
        train_cfg = self.config["training"]
        env_cfg = self.config["env"]
        ewc_cfg = self.config.get("ewc", {})
        total_timesteps = train_cfg["total_timesteps"]
        eval_freq = train_cfg["eval_freq"]
        eval_episodes = train_cfg["eval_episodes"]
        save_freq = train_cfg["save_freq"]
        train_freq = train_cfg["train_freq"]
        log_interval = self.config["logging"].get("log_interval", 1000)

        checkpoint_dir = os.path.join(
            self.config["logging"]["checkpoint_dir"], self.tag
        )
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Build the single model that will be trained across all tasks
        model_cfg = self.config["model"]
        model = DQNNetwork(
            in_channels=env_cfg["frame_stack"],
            conv_channels=model_cfg["conv_channels"],
            conv_kernels=model_cfg["conv_kernels"],
            conv_strides=model_cfg["conv_strides"],
            fc_hidden=model_cfg["fc_hidden"],
            unified_action_dim=model_cfg["unified_action_dim"],
            dueling=model_cfg.get("dueling", False),
        ).to(self.device)

        # Target network for DQN
        target_net = copy.deepcopy(model)
        target_net.eval()

        for task_idx, env_id in enumerate(task_sequence):
            game_name = env_id.replace("NoFrameskip-v4", "")
            valid_actions = get_valid_actions(env_id, self.union_actions)

            self.logger.info(f"\n{'='*60}")
            self.logger.info(
                f"EWC Task {task_idx + 1}/{len(task_sequence)}: {env_id}"
            )
            self.logger.info(f"{'='*60}")

            # ── Task 1: Load from expert checkpoint ──────────────────────
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

                # Compute Fisher for Task 1
                self.logger.info("Computing Fisher for Task 1...")
                fisher_states = self._collect_replay_states(
                    env_id, valid_actions, env_cfg,
                    num_states=ewc_cfg.get("buffer_size_per_task", 100_000),
                )
                fisher = compute_diagonal_fisher(
                    model, fisher_states, valid_actions,
                    self.device, self.fisher_samples,
                )
                self._register_task(model, fisher)

                # Save post-task checkpoint
                self._save_task_checkpoint(
                    model, checkpoint_dir, game_name, task_idx
                )
                self.logger.info(
                    f"Task 1 ({game_name}) loaded from expert. "
                    f"Fisher computed on {fisher_states.shape[0]} states."
                )
                continue

            # ── Tasks 2+: Train with EWC penalty ────────────────────────

            # Create environment
            env = make_atari_env(
                env_id=env_id,
                union_actions=self.union_actions,
                seed=self.config["seed"] + task_idx * 1000,
                frame_stack=env_cfg["frame_stack"],
                frame_skip=env_cfg["frame_skip"],
                screen_size=env_cfg["screen_size"],
                noop_max=env_cfg["noop_max"],
                episodic_life=env_cfg["episodic_life"],
                clip_reward=env_cfg["clip_reward"],
            )

            # Fresh optimizer for each task (standard in EWC literature)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=train_cfg["learning_rate"]
            )

            # Fresh replay buffer
            from src.data.replay_buffer import ReplayBuffer
            replay_buffer = ReplayBuffer(
                capacity=train_cfg["buffer_size"],
                frame_stack=env_cfg["frame_stack"],
                frame_shape=(env_cfg["screen_size"], env_cfg["screen_size"]),
                device=self.device,
            )

            # Action mask for this task
            action_mask = torch.full(
                (model_cfg["unified_action_dim"],), float("-inf"),
                device=self.device,
            )
            action_mask[valid_actions] = 0.0

            # Exploration schedule
            eps_start = train_cfg["eps_start"]
            eps_end = train_cfg["eps_end"]
            eps_decay = train_cfg["eps_decay_steps"]
            gamma = train_cfg["gamma"]
            batch_size = train_cfg["batch_size"]
            min_buffer = train_cfg["min_buffer_size"]
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
                # Epsilon-greedy action selection
                eps = eps_start + min(step / eps_decay, 1.0) * (eps_end - eps_start)
                if np.random.random() < eps:
                    action = np.random.choice(valid_actions)
                else:
                    with torch.no_grad():
                        s_t = (
                            torch.from_numpy(state).float()
                            .unsqueeze(0).to(self.device) / 255.0
                        )
                        q = model(s_t) + action_mask.unsqueeze(0)
                        action = q.argmax(dim=1).item()

                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                replay_buffer.push(state, action, reward, next_state, done)
                episode_reward += reward
                state = next_state

                # Train
                if step % train_freq == 0 and len(replay_buffer) >= min_buffer:
                    states_b, actions_b, rewards_b, next_b, dones_b = (
                        replay_buffer.sample(batch_size)
                    )

                    # DQN loss
                    q_vals = model(states_b)
                    q_taken = q_vals.gather(1, actions_b.unsqueeze(1)).squeeze(1)

                    with torch.no_grad():
                        if double_dqn:
                            next_q_policy = model(next_b) + action_mask.unsqueeze(0)
                            next_acts = next_q_policy.argmax(dim=1, keepdim=True)
                            next_q_target = target_net(next_b)
                            next_q_max = next_q_target.gather(
                                1, next_acts
                            ).squeeze(1)
                        else:
                            next_q = target_net(next_b) + action_mask.unsqueeze(0)
                            next_q_max = next_q.max(dim=1)[0]
                        target_vals = rewards_b + gamma * next_q_max * (1 - dones_b)

                    dqn_loss = nn.functional.smooth_l1_loss(q_taken, target_vals)

                    # EWC penalty
                    ewc_loss = self._ewc_penalty(model)
                    total_loss = dqn_loss + ewc_loss

                    optimizer.zero_grad()
                    total_loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                    optimizer.step()

                    train_steps += 1

                    # Target network update
                    if train_steps % target_update_freq == 0:
                        target_net.load_state_dict(model.state_dict())

                    # Logging
                    if step % log_interval == 0:
                        self.logger.log_scalars(
                            {
                                f"ewc/{game_name}/dqn_loss": dqn_loss.item(),
                                f"ewc/{game_name}/ewc_loss": ewc_loss.item(),
                                f"ewc/{game_name}/total_loss": total_loss.item(),
                                f"ewc/{game_name}/epsilon": eps,
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
                            f"[EWC-{game_name}] Step {step}/{total_timesteps} | "
                            f"Ep {episode_count} | Reward: {episode_reward:.1f} | "
                            f"Mean100: {recent_mean:.1f} | Eps: {eps:.3f} | "
                            f"Time: {elapsed:.0f}s"
                        )
                    episode_reward = 0.0
                    state, _ = env.reset()

                # Evaluate
                if step % eval_freq == 0:
                    eval_reward = self._evaluate_task(
                        model, env_id, valid_actions, env_cfg, eval_episodes
                    )
                    self.logger.info(
                        f"[EWC-{game_name}] Eval at step {step}: "
                        f"mean reward = {eval_reward:.2f}"
                    )
                    if eval_reward > best_eval_reward:
                        best_eval_reward = eval_reward
                        self.logger.info(
                            f"[EWC-{game_name}] New best: {eval_reward:.2f}"
                        )

                # Periodic checkpoint
                if step % save_freq == 0:
                    ckpt_path = os.path.join(
                        checkpoint_dir,
                        f"ewc_task{task_idx}_{game_name}_step{step}.pt",
                    )
                    torch.save(model.state_dict(), ckpt_path)

            env.close()

            # Final evaluation on current task
            final_eval = self._evaluate_task(
                model, env_id, valid_actions, env_cfg, eval_episodes
            )
            self.logger.info(
                f"[EWC-{game_name}] Training complete. "
                f"Final: {final_eval:.2f} | Best: {best_eval_reward:.2f}"
            )

            # Compute Fisher for this task
            self.logger.info(f"Computing Fisher for {game_name}...")
            fisher_states = self._collect_replay_states(
                env_id, valid_actions, env_cfg,
                num_states=ewc_cfg.get("buffer_size_per_task", 100_000),
            )
            fisher = compute_diagonal_fisher(
                model, fisher_states, valid_actions,
                self.device, self.fisher_samples,
            )
            self._register_task(model, fisher)

            # Save post-task checkpoint
            self._save_task_checkpoint(model, checkpoint_dir, game_name, task_idx)

            # Cross-evaluate on all previously seen tasks
            self.logger.info("Cross-evaluation on prior tasks:")
            for prev_idx in range(task_idx + 1):
                prev_env_id = task_sequence[prev_idx]
                prev_game = prev_env_id.replace("NoFrameskip-v4", "")
                prev_valid = get_valid_actions(prev_env_id, self.union_actions)
                prev_reward = self._evaluate_task(
                    model, prev_env_id, prev_valid, env_cfg, eval_episodes
                )
                self.logger.info(f"  {prev_game}: {prev_reward:.2f}")

            # Clean up step checkpoints
            import glob
            step_ckpts = glob.glob(
                os.path.join(checkpoint_dir, f"ewc_task{task_idx}_{game_name}_step*.pt")
            )
            for ckpt_path in step_ckpts:
                os.remove(ckpt_path)

        # Save final consolidated model (bare state_dict for evaluate.py)
        final_path = os.path.join(checkpoint_dir, "consolidated_ewc.pt")
        torch.save(model.state_dict(), final_path)
        self.logger.info(f"Final EWC model saved to {final_path}")

        return model

    def _collect_replay_states(
        self,
        env_id: str,
        valid_actions: List[int],
        env_cfg: Dict[str, Any],
        num_states: int = 100_000,
    ) -> torch.Tensor:
        """Collect observation frames by running the current policy.

        Returns:
            Tensor of shape (num_states, C, H, W), dtype uint8.
        """
        env = make_atari_env(
            env_id=env_id,
            union_actions=self.union_actions,
            seed=self.config["seed"] + 5000,
            frame_stack=env_cfg["frame_stack"],
            frame_skip=env_cfg["frame_skip"],
            screen_size=env_cfg["screen_size"],
            noop_max=env_cfg["noop_max"],
            episodic_life=env_cfg["episodic_life"],
            clip_reward=env_cfg["clip_reward"],
        )

        states = []
        state, _ = env.reset()
        for _ in range(num_states):
            states.append(state)
            action = np.random.choice(valid_actions)
            next_state, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                state, _ = env.reset()
            else:
                state = next_state

        env.close()
        return torch.from_numpy(np.array(states))

    def _evaluate_task(
        self,
        model: DQNNetwork,
        env_id: str,
        valid_actions: List[int],
        env_cfg: Dict[str, Any],
        num_episodes: int = 10,
    ) -> float:
        """Evaluate the model on a single task (deterministic, raw rewards)."""
        env = make_atari_env(
            env_id=env_id,
            union_actions=self.union_actions,
            seed=self.config["seed"] + 3000,
            frame_stack=env_cfg["frame_stack"],
            frame_skip=env_cfg["frame_skip"],
            screen_size=env_cfg["screen_size"],
            noop_max=env_cfg["noop_max"],
            episodic_life=False,
            clip_reward=False,
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
            while not done and steps < 27000:
                with torch.no_grad():
                    s_t = (
                        torch.from_numpy(state).float()
                        .unsqueeze(0).to(self.device) / 255.0
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

    def _save_task_checkpoint(
        self,
        model: DQNNetwork,
        checkpoint_dir: str,
        game_name: str,
        task_idx: int,
    ) -> None:
        """Save model after completing a task (bare state_dict)."""
        path = os.path.join(
            checkpoint_dir, f"ewc_after_task{task_idx}_{game_name}.pt"
        )
        torch.save(model.state_dict(), path)
        self.logger.info(f"Post-task checkpoint saved: {path}")
