"""
Expert Trainer: trains a single DQN agent on one Atari task.

Supports initialization from a global model checkpoint (for CRL pipeline).
"""

import os
import time
import numpy as np
import torch
from typing import Dict, Any, Optional, Tuple

from src.agents.dqn_agent import DQNAgent
from src.data.atari_wrappers import make_atari_env, get_valid_actions
from src.utils.logger import Logger


class ExpertTrainer:
    """Trains a DQN expert on a single Atari game.

    Args:
        config: Configuration dictionary.
        env_id: Gymnasium environment ID (e.g., 'PongNoFrameskip-v4').
        logger: Logger instance for metrics.
        device: Torch device string.
        global_weights: Optional global model weights for initialization.
        experiment_tag: Optional tag for checkpoint naming.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        env_id: str,
        logger: Logger,
        device: str = "cpu",
        global_weights: Optional[dict] = None,
        experiment_tag: str = "",
    ):
        self.config = config
        self.env_id = env_id
        self.logger = logger
        self.device = device
        self.experiment_tag = experiment_tag

        # Get valid actions for this game
        self.valid_actions = get_valid_actions(env_id)
        self.game_name = env_id.replace("NoFrameskip-v4", "")

        logger.info(
            f"Task: {env_id} | Valid actions: {self.valid_actions} "
            f"({len(self.valid_actions)} actions)"
        )

        # Create agent
        self.agent = DQNAgent(
            config=config,
            valid_actions=self.valid_actions,
            device=device,
        )

        # Initialize from global model if provided (HTCL-style)
        if global_weights is not None:
            logger.info("Initializing expert from global model weights.")
            self.agent.load_policy_weights(global_weights)

        # Create environment
        env_cfg = config["env"]
        self.env = make_atari_env(
            env_id=env_id,
            seed=config["seed"],
            frame_stack=env_cfg["frame_stack"],
            frame_skip=env_cfg["frame_skip"],
            screen_size=env_cfg["screen_size"],
            noop_max=env_cfg["noop_max"],
            episodic_life=env_cfg["episodic_life"],
            clip_reward=env_cfg["clip_reward"],
        )

    def train(self) -> Dict[str, Any]:
        """Run the full training loop.

        Returns:
            Dictionary with training results:
                - 'final_reward': mean reward over last eval
                - 'policy_state_dict': trained policy weights
                - 'valid_actions': valid action list
                - 'normalizer': normalizer state (if enabled)
                - 'replay_buffer': reference to replay buffer
        """
        train_cfg = self.config["training"]
        total_timesteps = train_cfg["total_timesteps"]
        eval_freq = train_cfg["eval_freq"]
        eval_episodes = train_cfg["eval_episodes"]
        save_freq = train_cfg["save_freq"]
        train_freq = train_cfg["train_freq"]
        log_interval = self.config["logging"].get("log_interval", 1000)
        checkpoint_dir = self.config["logging"]["checkpoint_dir"]

        state, _ = self.env.reset()
        episode_reward = 0.0
        episode_count = 0
        episode_rewards = []
        best_eval_reward = float("-inf")
        start_time = time.time()

        for step in range(1, total_timesteps + 1):
            # Select and execute action
            action = self.agent.select_action(state)
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            # Store transition
            self.agent.store_transition(state, action, reward, next_state, done)
            episode_reward += reward
            state = next_state

            # Train
            if step % train_freq == 0:
                loss = self.agent.train_step()
                if loss is not None and step % log_interval == 0:
                    self.logger.log_scalars(
                        {
                            f"{self.game_name}/train_loss": loss,
                            f"{self.game_name}/epsilon": self.agent.epsilon,
                        },
                        step=step,
                    )

            # Episode end
            if done:
                episode_count += 1
                episode_rewards.append(episode_reward)
                if step % log_interval == 0 or episode_count % 10 == 0:
                    recent_mean = np.mean(episode_rewards[-100:])
                    self.logger.log_scalar(
                        f"{self.game_name}/episode_reward", episode_reward, step
                    )
                    self.logger.log_scalar(
                        f"{self.game_name}/mean_reward_100", recent_mean, step
                    )
                    elapsed = time.time() - start_time
                    self.logger.info(
                        f"[{self.game_name}] Step {step}/{total_timesteps} | "
                        f"Ep {episode_count} | Reward: {episode_reward:.1f} | "
                        f"Mean100: {recent_mean:.1f} | Eps: {self.agent.epsilon:.3f} | "
                        f"Time: {elapsed:.0f}s"
                    )
                episode_reward = 0.0
                state, _ = self.env.reset()

            # Evaluate
            if step % eval_freq == 0:
                eval_reward = self.evaluate(eval_episodes)
                self.logger.log_scalar(
                    f"{self.game_name}/eval_reward", eval_reward, step
                )
                self.logger.info(
                    f"[{self.game_name}] Eval at step {step}: "
                    f"mean reward = {eval_reward:.2f}"
                )
                if eval_reward > best_eval_reward:
                    best_eval_reward = eval_reward
                    best_path = os.path.join(
                        checkpoint_dir,
                        self.experiment_tag,
                        f"expert_{self.game_name}_best.pt",
                    )
                    self.agent.save(best_path)
                    self.logger.info(
                        f"[{self.game_name}] New best model saved: {eval_reward:.2f}"
                    )

            # Periodic checkpoint
            if step % save_freq == 0:
                ckpt_path = os.path.join(
                    checkpoint_dir,
                    self.experiment_tag,
                    f"expert_{self.game_name}_step{step}.pt",
                )
                self.agent.save(ckpt_path)

        # Final save
        final_path = os.path.join(
            checkpoint_dir,
            self.experiment_tag,
            f"expert_{self.game_name}_final.pt",
        )
        self.agent.save(final_path)

        final_eval = self.evaluate(eval_episodes)
        self.logger.info(
            f"[{self.game_name}] Training complete. "
            f"Final eval: {final_eval:.2f} | Best eval: {best_eval_reward:.2f}"
        )

        result = {
            "final_reward": final_eval,
            "best_reward": best_eval_reward,
            "policy_state_dict": self.agent.get_policy_state_dict(),
            "valid_actions": self.valid_actions,
            "replay_buffer": self.agent.replay_buffer,
            "game_name": self.game_name,
            "env_id": self.env_id,
        }
        if self.agent.normalizer is not None:
            result["normalizer"] = self.agent.normalizer.state_dict()

        return result

    def evaluate(self, num_episodes: int = 10) -> float:
        """Evaluate the agent deterministically.

        Args:
            num_episodes: Number of evaluation episodes.

        Returns:
            Mean episode reward.
        """
        env_cfg = self.config["env"]
        eval_env = make_atari_env(
            env_id=self.env_id,
            seed=self.config["seed"] + 1000,
            frame_stack=env_cfg["frame_stack"],
            frame_skip=env_cfg["frame_skip"],
            screen_size=env_cfg["screen_size"],
            noop_max=env_cfg["noop_max"],
            episodic_life=False,  # Full episodes for eval
            clip_reward=False,  # Raw rewards for eval
        )

        rewards = []
        for _ in range(num_episodes):
            state, _ = eval_env.reset()
            total_reward = 0.0
            done = False
            while not done:
                action = self.agent.select_action(state, deterministic=True)
                state, reward, terminated, truncated, _ = eval_env.step(action)
                total_reward += reward
                done = terminated or truncated
            rewards.append(total_reward)

        eval_env.close()
        return float(np.mean(rewards))
