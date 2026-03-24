"""
Progress & Compress (P&C) continual learner.

Implements the two-phase continual learning algorithm from:
    Schwarz et al. (2018) "Progress & Compress: A scalable system for
    continual learning." ICML 2018.

For each task in sequence:
    Progress phase  -- Freeze KB; train a fresh Active Column (AC) on the
                       new task.  The AC has lateral connections from KB's
                       frozen CNN features, allowing it to reuse prior
                       knowledge without corrupting KB.
    Compress phase  -- Freeze AC; fine-tune KB via KD (KB -> AC) regularised
                       by Online EWC (Fisher accumulated across all seen
                       tasks) to protect previously learned behaviours.

The final output is only the KB -- no task identifier needed at test time.
"""

import copy
import glob
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.atari_wrappers import make_atari_env, get_valid_actions
from src.data.replay_buffer import ReplayBuffer
from src.models.dqn import DQNNetwork
from src.models.progressive import ProgressiveColumn
from src.utils.logger import Logger


class ProgressCompressTrainer:
    """Progress & Compress continual learner (Schwarz et al., ICML 2018).

    Runs two phases per task:
      1. Progress: a fresh ProgressiveColumn (AC) with lateral KB connections
         is trained via DQN on the new task. KB weights are frozen throughout.
      2. Compress: KB is fine-tuned to match the AC policy via KL distillation,
         with Online EWC regularisation to protect previously acquired skills.

    At test time only the KB is used; no task label is required.

    Args:
        config: Full configuration dictionary (loaded from base.yaml).
        union_actions: Sorted list of ALE action indices in the union space.
        device: Torch device string ('cuda', 'cpu', 'mps').
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

        pc_cfg = config.get("progress_compress", {})
        debug_cfg = config.get("debug", {})
        debug_on = debug_cfg.get("enabled", False)

        self.progress_steps: int = (
            debug_cfg.get("pc_progress_steps", 10000)
            if debug_on
            else pc_cfg.get("progress_steps", 3_000_000)
        )
        self.compress_epochs: int = (
            debug_cfg.get("pc_compress_epochs", 5)
            if debug_on
            else pc_cfg.get("compress_epochs", 10000)
        )
        self.compress_lr: float = pc_cfg.get("compress_lr", 5e-5)
        self.compress_batch_size: int = pc_cfg.get("compress_batch_size", 64)
        self.temperature: float = pc_cfg.get("temperature", 0.01)
        self.lambda_ewc: float = pc_cfg.get("lambda_ewc", 5000.0)
        self.gamma_ewc: float = pc_cfg.get("gamma_ewc", 0.95)
        self.fisher_samples: int = (
            debug_cfg.get("fisher_samples", 100)
            if debug_on
            else pc_cfg.get("fisher_samples", 5000)
        )
        self.buffer_size_per_task: int = (
            debug_cfg.get("buffer_size_per_task", 2000)
            if debug_on
            else pc_cfg.get("buffer_size_per_task", 100_000)
        )

        # Online EWC state: accumulated Fisher and parameter anchor
        self.fisher_accum: Optional[Dict[str, torch.Tensor]] = None
        self.kb_star: Optional[Dict[str, torch.Tensor]] = None

    # =========================================================================
    # Public API
    # =========================================================================

    def run(self, task_sequence: List[str]) -> DQNNetwork:
        """Run P&C across all tasks in sequence.

        Args:
            task_sequence: List of env_ids
                (e.g., ['BreakoutNoFrameskip-v4', ...]).

        Returns:
            The final KB DQNNetwork after all tasks.
        """
        env_cfg = self.config["env"]
        model_cfg = self.config["model"]

        checkpoint_dir = os.path.join(
            self.config["logging"]["checkpoint_dir"], self.tag
        )
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Build the Knowledge Base (KB): standard DQNNetwork, random init
        kb = DQNNetwork(
            in_channels=env_cfg["frame_stack"],
            conv_channels=model_cfg["conv_channels"],
            conv_kernels=model_cfg["conv_kernels"],
            conv_strides=model_cfg["conv_strides"],
            fc_hidden=model_cfg["fc_hidden"],
            unified_action_dim=model_cfg["unified_action_dim"],
            dueling=model_cfg.get("dueling", False),
        ).to(self.device)

        self.logger.info("\n" + "=" * 60)
        self.logger.info("Progress & Compress — Sequential Training")
        self.logger.info("=" * 60)
        self.logger.info(f"Tasks: {task_sequence}")
        self.logger.info(f"Progress steps/task: {self.progress_steps}")
        self.logger.info(f"Compress epochs/task: {self.compress_epochs}")
        self.logger.info(
            f"lambda_ewc={self.lambda_ewc}  gamma_ewc={self.gamma_ewc}"
        )

        for task_idx, env_id in enumerate(task_sequence):
            game_name = env_id.replace("NoFrameskip-v4", "")
            valid_actions = get_valid_actions(env_id, self.union_actions)

            self.logger.info(f"\n{'='*60}")
            self.logger.info(
                f"P&C Task {task_idx + 1}/{len(task_sequence)}: {env_id}"
            )
            self.logger.info(f"{'='*60}")

            # ── Progress phase ────────────────────────────────────────
            self.logger.info(
                f"[P&C-{game_name}] Progress phase "
                f"({self.progress_steps} steps) ..."
            )
            ac = self._progress(kb, env_id, valid_actions, task_idx)

            # ── Collect states for compress / Fisher ──────────────────
            self.logger.info(
                f"[P&C-{game_name}] Collecting {self.buffer_size_per_task} "
                f"states for compress / Fisher ..."
            )
            compress_states = self._collect_states(
                env_id, valid_actions, self.buffer_size_per_task
            )

            # ── Compress phase ────────────────────────────────────────
            self.logger.info(
                f"[P&C-{game_name}] Compress phase "
                f"({self.compress_epochs} epochs) ..."
            )
            kb = self._compress(
                kb, ac, compress_states, valid_actions, game_name
            )

            # Save KB after this task
            task_ckpt = os.path.join(
                checkpoint_dir,
                f"pc_kb_after_task{task_idx}_{game_name}.pt",
            )
            torch.save(kb.state_dict(), task_ckpt)
            self.logger.info(f"KB checkpoint saved: {task_ckpt}")

            # Cross-evaluate all seen tasks
            self.logger.info("Cross-evaluation on all seen tasks:")
            for prev_idx in range(task_idx + 1):
                prev_env_id = task_sequence[prev_idx]
                prev_game = prev_env_id.replace("NoFrameskip-v4", "")
                prev_valid = get_valid_actions(prev_env_id, self.union_actions)
                prev_reward = self._evaluate_task(
                    kb,
                    prev_env_id,
                    prev_valid,
                    self.config["env"],
                    self.config["training"]["eval_episodes"],
                )
                self.logger.info(f"  {prev_game}: {prev_reward:.2f}")

        # Save final consolidated KB
        final_path = os.path.join(checkpoint_dir, "consolidated_pc.pt")
        torch.save(kb.state_dict(), final_path)
        self.logger.info(f"Final P&C KB saved to {final_path}")

        return kb

    # =========================================================================
    # Progress phase
    # =========================================================================

    def _progress(
        self,
        kb: DQNNetwork,
        env_id: str,
        valid_actions: List[int],
        task_idx: int,
    ) -> ProgressiveColumn:
        """Train a fresh Active Column on env_id for progress_steps steps.

        KB is fully frozen during this phase.

        Args:
            kb: Knowledge Base (will be frozen internally).
            env_id: Gym environment id.
            valid_actions: Valid action indices for this game.
            task_idx: Index of current task (used for env seeding).

        Returns:
            Trained ProgressiveColumn.
        """
        train_cfg = self.config["training"]
        env_cfg = self.config["env"]
        model_cfg = self.config["model"]
        log_interval = self.config["logging"].get("log_interval", 1000)
        game_name = env_id.replace("NoFrameskip-v4", "")

        # Freeze KB
        for p in kb.parameters():
            p.requires_grad_(False)
        kb.eval()

        # Build AC
        ac = ProgressiveColumn(
            kb=kb,
            in_channels=env_cfg["frame_stack"],
            conv_channels=model_cfg["conv_channels"],
            conv_kernels=model_cfg["conv_kernels"],
            conv_strides=model_cfg["conv_strides"],
            fc_hidden=model_cfg["fc_hidden"],
            unified_action_dim=model_cfg["unified_action_dim"],
        ).to(self.device)

        # Target AC: deepcopy of AC, but sharing KB reference (not a KB copy)
        target_ac = copy.deepcopy(ac)
        object.__setattr__(target_ac, '_kb_ref', ac._kb_ref)
        target_ac.eval()

        optimizer = torch.optim.AdamW(
            ac.parameters(), lr=train_cfg["learning_rate"]
        )

        replay_buffer = ReplayBuffer(
            capacity=train_cfg["buffer_size"],
            frame_stack=env_cfg["frame_stack"],
            frame_shape=(env_cfg["screen_size"], env_cfg["screen_size"]),
            device=self.device,
        )

        # Action mask for epsilon-greedy and Q-target
        action_mask = torch.full(
            (model_cfg["unified_action_dim"],), float("-inf"),
            device=self.device,
        )
        action_mask[valid_actions] = 0.0

        eps_start = train_cfg["eps_start"]
        eps_end = train_cfg["eps_end"]
        eps_decay = train_cfg["eps_decay_steps"]
        gamma = train_cfg["gamma"]
        batch_size = train_cfg["batch_size"]
        min_buffer = train_cfg["min_buffer_size"]
        target_update_freq = train_cfg["target_update_freq"]
        double_dqn = train_cfg.get("double_dqn", False)
        gradient_clip = train_cfg.get("gradient_clip", 10.0)
        train_freq = train_cfg["train_freq"]
        eval_freq = train_cfg["eval_freq"]
        eval_episodes = train_cfg["eval_episodes"]

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

        state, _ = env.reset()
        episode_reward = 0.0
        episode_count = 0
        episode_rewards: List[float] = []
        best_eval_reward = float("-inf")
        train_steps = 0
        start_time = time.time()

        for step in range(1, self.progress_steps + 1):
            # Epsilon-greedy
            eps = eps_start + min(step / eps_decay, 1.0) * (eps_end - eps_start)
            if np.random.random() < eps:
                action = np.random.choice(valid_actions)
            else:
                with torch.no_grad():
                    s_t = (
                        torch.from_numpy(state).float()
                        .unsqueeze(0).to(self.device) / 255.0
                    )
                    q = ac(s_t) + action_mask.unsqueeze(0)
                    action = q.argmax(dim=1).item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            replay_buffer.push(state, action, reward, next_state, terminated)
            episode_reward += reward
            state = next_state

            # Train
            if step % train_freq == 0 and len(replay_buffer) >= min_buffer:
                states_b, actions_b, rewards_b, next_b, dones_b = (
                    replay_buffer.sample(batch_size)
                )

                q_vals = ac(states_b)
                q_taken = q_vals.gather(1, actions_b.unsqueeze(1)).squeeze(1)

                with torch.no_grad():
                    if double_dqn:
                        next_q_policy = ac(next_b) + action_mask.unsqueeze(0)
                        next_acts = next_q_policy.argmax(dim=1, keepdim=True)
                        next_q_target = target_ac(next_b)
                        next_q_max = next_q_target.gather(
                            1, next_acts
                        ).squeeze(1)
                    else:
                        next_q = target_ac(next_b) + action_mask.unsqueeze(0)
                        next_q_max = next_q.max(dim=1)[0]
                    target_vals = rewards_b + gamma * next_q_max * (1 - dones_b)

                dqn_loss = F.smooth_l1_loss(q_taken, target_vals)

                optimizer.zero_grad()
                dqn_loss.backward()
                nn.utils.clip_grad_norm_(ac.parameters(), gradient_clip)
                optimizer.step()

                train_steps += 1

                if train_steps % target_update_freq == 0:
                    # Sync only the AC-owned parameters (not KB)
                    target_ac.features.load_state_dict(ac.features.state_dict())
                    target_ac.ac_linear.load_state_dict(ac.ac_linear.state_dict())
                    target_ac.lateral.load_state_dict(ac.lateral.state_dict())
                    target_ac.head.load_state_dict(ac.head.state_dict())

                if step % log_interval == 0:
                    self.logger.log_scalars(
                        {
                            f"pc/{game_name}/progress_dqn_loss": dqn_loss.item(),
                            f"pc/{game_name}/progress_epsilon": eps,
                        },
                        step=step,
                    )

            if done:
                episode_count += 1
                episode_rewards.append(episode_reward)
                if step % log_interval == 0 or episode_count % 10 == 0:
                    recent_mean = np.mean(episode_rewards[-100:])
                    elapsed = time.time() - start_time
                    self.logger.info(
                        f"[PC-Progress-{game_name}] Step {step}/{self.progress_steps} | "
                        f"Ep {episode_count} | R: {episode_reward:.1f} | "
                        f"Mean100: {recent_mean:.1f} | Eps: {eps:.3f} | "
                        f"{elapsed:.0f}s"
                    )
                episode_reward = 0.0
                state, _ = env.reset()

            if step % eval_freq == 0:
                eval_reward = self._evaluate_task(
                    ac, env_id, valid_actions, env_cfg, eval_episodes
                )
                self.logger.info(
                    f"[PC-Progress-{game_name}] Eval at step {step}: "
                    f"mean reward = {eval_reward:.2f}"
                )
                if eval_reward > best_eval_reward:
                    best_eval_reward = eval_reward

        env.close()

        final_eval = self._evaluate_task(
            ac, env_id, valid_actions, env_cfg, eval_episodes
        )
        self.logger.info(
            f"[PC-Progress-{game_name}] Training complete. "
            f"Final: {final_eval:.2f} | Best: {best_eval_reward:.2f}"
        )

        # Unfreeze KB for compress phase
        for p in kb.parameters():
            p.requires_grad_(True)

        return ac

    # =========================================================================
    # Compress phase
    # =========================================================================

    def _compress(
        self,
        kb: DQNNetwork,
        ac: ProgressiveColumn,
        states: torch.Tensor,
        valid_actions: List[int],
        game_name: str,
    ) -> DQNNetwork:
        """Fine-tune KB to match AC policy with Online EWC regularisation.

        Loss = KL(AC(s)/T || KB(s)/T) + (lambda_ewc / 2) * EWC_penalty

        After training, computes the diagonal Fisher at the new KB optimum and
        accumulates it into the running Online EWC Fisher.

        Args:
            kb: Knowledge Base to update (parameters unfrozen by caller).
            ac: Trained Active Column (frozen here).
            states: Tensor (N, C, H, W) uint8 of states collected for this task.
            valid_actions: Valid action indices for this game.
            game_name: Human-readable game name for logging.

        Returns:
            Updated KB (same object, modified in-place; returned for clarity).
        """
        n_actions = kb.unified_action_dim
        n_states = states.shape[0]
        log_interval = self.config["logging"].get("log_interval", 1000)

        # Freeze AC
        for p in ac.parameters():
            p.requires_grad_(False)
        ac.eval()

        optimizer = torch.optim.AdamW(kb.parameters(), lr=self.compress_lr)

        kb.train()

        start_time = time.time()
        for epoch in range(1, self.compress_epochs + 1):
            # Sample a mini-batch from collected states
            idx = torch.randint(0, n_states, (self.compress_batch_size,))
            s_batch = states[idx].float().to(self.device) / 255.0

            # AC logits (teacher, frozen) — slice to valid actions only.
            # Computing KL over all 6 actions with -inf masking causes NaN on
            # GPU: softmax(-inf)=0 and log_softmax(-inf)=-inf, so the F.kl_div
            # kernel computes 0 * (-inf - -inf) = 0 * nan = nan (IEEE 754).
            with torch.no_grad():
                ac_logits = ac(s_batch)
                teacher_probs = torch.softmax(
                    ac_logits[:, valid_actions] / self.temperature, dim=1
                )

            # KB logits (student, trainable)
            kb_logits = kb(s_batch)
            student_log_probs = torch.log_softmax(
                kb_logits[:, valid_actions] / self.temperature, dim=1
            )

            # KL divergence: KL(teacher || student), valid actions only
            kl_loss = F.kl_div(
                student_log_probs,
                teacher_probs,
                reduction="batchmean",
            )

            # Online EWC penalty
            ewc_penalty = self._ewc_penalty(kb)
            total_loss = kl_loss + ewc_penalty

            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(kb.parameters(), 10.0)
            optimizer.step()

            if epoch % log_interval == 0 or epoch == self.compress_epochs:
                elapsed = time.time() - start_time
                self.logger.info(
                    f"[PC-Compress-{game_name}] Epoch {epoch}/{self.compress_epochs} | "
                    f"KL: {kl_loss.item():.4f} | "
                    f"EWC: {ewc_penalty.item():.4f} | "
                    f"Total: {total_loss.item():.4f} | "
                    f"{elapsed:.0f}s"
                )
                self.logger.log_scalars(
                    {
                        f"pc/{game_name}/compress_kl": kl_loss.item(),
                        f"pc/{game_name}/compress_ewc": ewc_penalty.item(),
                        f"pc/{game_name}/compress_total": total_loss.item(),
                    },
                    step=epoch,
                )

        # Compute Fisher at KB optimum and accumulate Online EWC
        self.logger.info(
            f"[PC-Compress-{game_name}] Computing Fisher "
            f"({self.fisher_samples} samples) ..."
        )
        fisher_new = self._compute_fisher(kb, states, valid_actions)
        self._accumulate_fisher(kb, fisher_new)

        # Unfreeze AC (not strictly necessary since AC is discarded, but clean)
        for p in ac.parameters():
            p.requires_grad_(True)

        return kb

    # =========================================================================
    # Fisher computation and Online EWC accumulation
    # =========================================================================

    def _compute_fisher(
        self,
        model: DQNNetwork,
        states: torch.Tensor,
        valid_actions: List[int],
    ) -> Dict[str, torch.Tensor]:
        """Diagonal empirical Fisher at model's current parameters.

        Follows the same approach as ewc.py: squared gradients of the
        log-probability of the greedy action, averaged over sampled states.

        Args:
            model: DQNNetwork to differentiate through.
            states: (N, C, H, W) uint8 state tensor.
            valid_actions: Valid action indices for the current task.

        Returns:
            Dict mapping parameter name to Fisher tensor (same shape).
        """
        model.eval()
        fisher: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            fisher[name] = torch.zeros_like(param, device=self.device)

        n = min(self.fisher_samples, states.shape[0])
        indices = torch.randperm(states.shape[0])[:n]

        action_mask = torch.full(
            (model.unified_action_dim,), float("-inf"), device=self.device
        )
        action_mask[valid_actions] = 0.0

        for idx_val in indices:
            state = states[idx_val].unsqueeze(0).float().to(self.device) / 255.0
            model.zero_grad()

            q_values = model(state)
            masked_q = q_values + action_mask.unsqueeze(0)
            best_action = masked_q.argmax(dim=1)

            valid_q = q_values[0, valid_actions]
            log_probs = torch.log_softmax(valid_q, dim=0)

            best_in_valid = valid_actions.index(best_action.item())
            log_prob = log_probs[best_in_valid]
            log_prob.backward()

            for name, param in model.named_parameters():
                if param.grad is not None:
                    fisher[name] += param.grad.detach() ** 2

        for name in fisher:
            fisher[name] /= n

        model.train()
        return fisher

    def _accumulate_fisher(
        self,
        kb: DQNNetwork,
        fisher_new: Dict[str, torch.Tensor],
    ) -> None:
        """Accumulate fisher_new into the running Online EWC Fisher.

        Online EWC update (Schwarz et al., 2018):
            F_accum[n] = gamma_ewc * F_accum[n] + F_new[n]

        Also saves the current KB parameters as the EWC anchor (kb_star).
        """
        if self.fisher_accum is None:
            self.fisher_accum = {k: v.clone() for k, v in fisher_new.items()}
        else:
            for name in self.fisher_accum:
                self.fisher_accum[name] = (
                    self.gamma_ewc * self.fisher_accum[name]
                    + fisher_new[name]
                )

        self.kb_star = {
            name: param.detach().clone()
            for name, param in kb.named_parameters()
        }

    def _ewc_penalty(self, kb: DQNNetwork) -> torch.Tensor:
        """Online EWC penalty: (lambda/2) * sum F[n] * (theta[n] - theta*[n])^2.

        Returns scalar 0.0 tensor if no prior tasks have been seen.
        """
        penalty = torch.tensor(0.0, device=self.device)
        if self.fisher_accum is None or self.kb_star is None:
            return penalty
        for name, param in kb.named_parameters():
            if name in self.fisher_accum:
                diff = param - self.kb_star[name]
                penalty += (self.fisher_accum[name] * diff ** 2).sum()
        return (self.lambda_ewc / 2.0) * penalty

    # =========================================================================
    # State collection
    # =========================================================================

    def _collect_states(
        self,
        env_id: str,
        valid_actions: List[int],
        num_states: int,
    ) -> torch.Tensor:
        """Roll out the environment with random actions to collect states.

        Args:
            env_id: Gym environment id.
            valid_actions: Valid action indices (used for random sampling).
            num_states: Number of states to collect.

        Returns:
            Tensor of shape (num_states, C, H, W), dtype uint8.
        """
        env_cfg = self.config["env"]
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

        collected = []
        state, _ = env.reset()
        for _ in range(num_states):
            collected.append(state)
            action = np.random.choice(valid_actions)
            next_state, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                state, _ = env.reset()
            else:
                state = next_state

        env.close()
        return torch.from_numpy(np.array(collected))

    # =========================================================================
    # Evaluation
    # =========================================================================

    def _evaluate_task(
        self,
        model: Any,
        env_id: str,
        valid_actions: List[int],
        env_cfg: Dict[str, Any],
        num_episodes: int = 10,
    ) -> float:
        """Evaluate a model on a single task (deterministic, raw rewards).

        Compatible with both DQNNetwork and ProgressiveColumn, both of which
        implement forward(x) -> Q-values of shape (batch, unified_action_dim).
        """
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
