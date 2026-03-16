"""
Multi-Task Joint Training Baseline for Atari DQN.

Trains a single DQN model on ALL tasks simultaneously by interleaving
environment steps across tasks in a round-robin fashion. This is the
standard "upper bound" baseline in continual learning: the model has
concurrent access to all task data, so catastrophic forgetting does
not apply.

Protocol:
    1.  Instantiate one environment and one replay buffer per task.
    2.  Round-robin: each global step advances one task's environment
        (cycling Breakout → SpaceInvaders → Pong → Breakout → ...).
    3.  Every ``train_freq`` steps, sample a mini-batch uniformly
        from a randomly chosen task's replay buffer, applying that
        task's action mask for both Q-value computation and target
        bootstrapping.
    4.  A single optimizer and target network are shared across tasks.

The default step budget is 5M total (≈1.67M per task), which usually
suffices because the shared gradient signal accelerates convergence.
The final model is saved as a bare ``state_dict`` compatible with
``scripts/evaluate.py``.
"""

import copy
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from src.data.atari_wrappers import make_atari_env, get_valid_actions
from src.data.replay_buffer import ReplayBuffer
from src.models.dqn import DQNNetwork
from src.utils.logger import Logger


class MultiTaskTrainer:
    """Joint DQN trainer that interleaves all Atari tasks simultaneously.

    At each global step the trainer advances one task's environment in a
    round-robin schedule, stores the transition in that task's replay
    buffer, then performs a training update sampled from one of the
    buffers that has enough data.

    Args:
        config: Full configuration dictionary.
        union_actions: Sorted list of ALE action indices forming the union.
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

        mt_cfg = config.get("multitask", {})
        self.total_timesteps = mt_cfg.get(
            "total_timesteps", 5_000_000
        )

    def train(
        self,
        task_sequence: List[str],
    ) -> DQNNetwork:
        """Train the model jointly on all tasks.

        Args:
            task_sequence: List of env_ids
                (e.g. ``['BreakoutNoFrameskip-v4', ...]``).

        Returns:
            The trained DQN model after all steps.
        """
        train_cfg = self.config["training"]
        env_cfg = self.config["env"]
        model_cfg = self.config["model"]
        mt_cfg = self.config.get("multitask", {})

        total_timesteps = self.total_timesteps
        train_freq = train_cfg["train_freq"]
        batch_size = train_cfg["batch_size"]
        gamma = train_cfg["gamma"]
        double_dqn = train_cfg.get("double_dqn", False)
        gradient_clip = train_cfg.get("gradient_clip", 10.0)
        target_update_freq = train_cfg["target_update_freq"]
        eval_freq = mt_cfg.get("eval_freq", train_cfg["eval_freq"])
        eval_episodes = train_cfg["eval_episodes"]
        save_freq = train_cfg["save_freq"]
        log_interval = self.config["logging"].get("log_interval", 1000)

        # Exploration schedule
        eps_start = train_cfg["eps_start"]
        eps_end = train_cfg["eps_end"]
        eps_decay = train_cfg["eps_decay_steps"]

        # Per-task buffer capacity: split total buffer among tasks.
        # Use a dedicated config key or fall back to equal split.
        buffer_per_task = mt_cfg.get(
            "buffer_per_task",
            train_cfg["buffer_size"] // len(task_sequence),
        )
        min_buffer = mt_cfg.get(
            "min_buffer_per_task",
            train_cfg["min_buffer_size"] // len(task_sequence),
        )

        checkpoint_dir = os.path.join(
            self.config["logging"]["checkpoint_dir"], self.tag
        )
        os.makedirs(checkpoint_dir, exist_ok=True)

        num_tasks = len(task_sequence)

        # ── Build model & target ─────────────────────────────────────
        model = DQNNetwork(
            in_channels=env_cfg["frame_stack"],
            conv_channels=model_cfg["conv_channels"],
            conv_kernels=model_cfg["conv_kernels"],
            conv_strides=model_cfg["conv_strides"],
            fc_hidden=model_cfg["fc_hidden"],
            unified_action_dim=model_cfg["unified_action_dim"],
            dueling=model_cfg.get("dueling", False),
        ).to(self.device)

        target_net = copy.deepcopy(model)
        target_net.eval()

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=train_cfg["learning_rate"]
        )

        # ── Per-task state: env, buffer, action mask, stats ──────────
        envs: List[Any] = []
        buffers: List[ReplayBuffer] = []
        action_masks: List[torch.Tensor] = []
        valid_actions_list: List[List[int]] = []
        game_names: List[str] = []

        # Per-task episode tracking
        states: List[np.ndarray] = []
        episode_rewards: List[float] = []
        episode_counts: List[int] = []
        recent_rewards: List[List[float]] = []
        best_eval_rewards: List[float] = []

        for task_idx, env_id in enumerate(task_sequence):
            game_name = env_id.replace("NoFrameskip-v4", "")
            game_names.append(game_name)
            valid_actions = get_valid_actions(env_id, self.union_actions)
            valid_actions_list.append(valid_actions)

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
            envs.append(env)

            buf = ReplayBuffer(
                capacity=buffer_per_task,
                frame_stack=env_cfg["frame_stack"],
                frame_shape=(env_cfg["screen_size"], env_cfg["screen_size"]),
                device=self.device,
            )
            buffers.append(buf)

            mask = torch.full(
                (model_cfg["unified_action_dim"],), float("-inf"),
                device=self.device,
            )
            mask[valid_actions] = 0.0
            action_masks.append(mask)

            state, _ = env.reset()
            states.append(state)
            episode_rewards.append(0.0)
            episode_counts.append(0)
            recent_rewards.append([])
            best_eval_rewards.append(float("-inf"))

        self.logger.info(f"\n{'='*60}")
        self.logger.info("Multi-Task Joint Training")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Tasks: {game_names}")
        self.logger.info(f"Total steps: {total_timesteps}")
        self.logger.info(
            f"Buffer per task: {buffer_per_task} "
            f"(min fill: {min_buffer})"
        )
        self.logger.info(f"Schedule: round-robin")

        train_steps = 0
        start_time = time.time()

        for step in range(1, total_timesteps + 1):
            # ── Round-robin task selection ────────────────────────────
            task_idx = (step - 1) % num_tasks
            env = envs[task_idx]
            buf = buffers[task_idx]
            valid_actions = valid_actions_list[task_idx]
            a_mask = action_masks[task_idx]
            state = states[task_idx]

            # ── Epsilon-greedy action ─────────────────────────────────
            eps = eps_start + min(step / eps_decay, 1.0) * (
                eps_end - eps_start
            )
            if np.random.random() < eps:
                action = np.random.choice(valid_actions)
            else:
                with torch.no_grad():
                    s_t = (
                        torch.from_numpy(state).float()
                        .unsqueeze(0).to(self.device) / 255.0
                    )
                    q = model(s_t) + a_mask.unsqueeze(0)
                    action = q.argmax(dim=1).item()

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            buf.push(state, action, reward, next_state, done)

            episode_rewards[task_idx] += reward
            states[task_idx] = next_state

            if done:
                episode_counts[task_idx] += 1
                recent_rewards[task_idx].append(episode_rewards[task_idx])
                episode_rewards[task_idx] = 0.0
                states[task_idx], _ = env.reset()

            # ── Training update ───────────────────────────────────────
            if step % train_freq == 0:
                # Pick a random task whose buffer has enough data
                ready = [
                    i for i in range(num_tasks) if len(buffers[i]) >= min_buffer
                ]
                if ready:
                    t_idx = ready[np.random.randint(len(ready))]
                    t_mask = action_masks[t_idx]

                    states_b, actions_b, rewards_b, next_b, dones_b = (
                        buffers[t_idx].sample(batch_size)
                    )

                    q_vals = model(states_b)
                    q_taken = q_vals.gather(
                        1, actions_b.unsqueeze(1)
                    ).squeeze(1)

                    with torch.no_grad():
                        if double_dqn:
                            next_q_policy = (
                                model(next_b) + t_mask.unsqueeze(0)
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
                                target_net(next_b) + t_mask.unsqueeze(0)
                            )
                            next_q_max = next_q.max(dim=1)[0]

                        target_vals = (
                            rewards_b + gamma * next_q_max * (1 - dones_b)
                        )

                    loss = nn.functional.smooth_l1_loss(q_taken, target_vals)

                    optimizer.zero_grad()
                    loss.backward()
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
                                f"multitask/{game_names[t_idx]}/loss": (
                                    loss.item()
                                ),
                                "multitask/epsilon": eps,
                            },
                            step=step,
                        )

            # ── Periodic console logging ──────────────────────────────
            if step % log_interval == 0:
                elapsed = time.time() - start_time
                parts = []
                for i in range(num_tasks):
                    r100 = (
                        np.mean(recent_rewards[i][-100:])
                        if recent_rewards[i] else 0.0
                    )
                    parts.append(
                        f"{game_names[i]}={r100:.1f}({episode_counts[i]}ep)"
                    )
                self.logger.info(
                    f"[MTL] Step {step}/{total_timesteps} | "
                    f"Eps: {eps:.3f} | {' | '.join(parts)} | "
                    f"{elapsed:.0f}s"
                )

            # ── Evaluation ────────────────────────────────────────────
            if step % eval_freq == 0:
                self.logger.info(
                    f"[MTL] Evaluation at step {step}:"
                )
                for i, env_id in enumerate(task_sequence):
                    eval_reward = self._evaluate_task(
                        model, env_id, valid_actions_list[i],
                        env_cfg, eval_episodes,
                    )
                    if eval_reward > best_eval_rewards[i]:
                        best_eval_rewards[i] = eval_reward
                    self.logger.info(
                        f"  {game_names[i]}: {eval_reward:.2f} "
                        f"(best: {best_eval_rewards[i]:.2f})"
                    )

            # ── Periodic checkpoint ───────────────────────────────────
            if step % save_freq == 0:
                ckpt_path = os.path.join(
                    checkpoint_dir, f"multitask_step{step}.pt"
                )
                torch.save(model.state_dict(), ckpt_path)

        # ── Cleanup environments ──────────────────────────────────────
        for env in envs:
            env.close()

        # ── Final evaluation ──────────────────────────────────────────
        self.logger.info(f"\n{'='*60}")
        self.logger.info("Final evaluation:")
        for i, env_id in enumerate(task_sequence):
            eval_reward = self._evaluate_task(
                model, env_id, valid_actions_list[i],
                env_cfg, eval_episodes,
            )
            self.logger.info(
                f"  {game_names[i]}: {eval_reward:.2f} "
                f"(best during training: {best_eval_rewards[i]:.2f})"
            )

        # ── Save final model ──────────────────────────────────────────
        final_path = os.path.join(checkpoint_dir, "consolidated_multitask.pt")
        torch.save(model.state_dict(), final_path)
        self.logger.info(f"Final multi-task model saved to {final_path}")

        # Clean up step checkpoints
        import glob
        step_ckpts = glob.glob(
            os.path.join(checkpoint_dir, "multitask_step*.pt")
        )
        for ckpt_path in step_ckpts:
            os.remove(ckpt_path)

        return model

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
