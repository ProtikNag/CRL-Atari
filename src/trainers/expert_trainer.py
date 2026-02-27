"""
Expert Trainer: trains a single DQN agent on one Atari task.

Each expert starts from random initialization and trains independently.
"""

import json
import os
import time
import numpy as np
import torch
from typing import Dict, Any, List, Optional, Tuple

from src.agents.dqn_agent import DQNAgent
from src.data.atari_wrappers import make_atari_env, get_valid_actions
from src.utils.logger import Logger

# ── Visualization palette ─────────────────────────────────────────────────────

_PALETTE = {
    "pastel_blue": "#A8D8EA",
    "pastel_orange": "#FFD8B1",
    "pastel_green": "#B5EAD7",
}
_EDGE_COLOR = "#1a1a1a"
_GAME_STYLE = {
    "Pong":          {"color": _PALETTE["pastel_blue"],   "marker": "o"},
    "Breakout":      {"color": _PALETTE["pastel_orange"], "marker": "s"},
    "SpaceInvaders": {"color": _PALETTE["pastel_green"],  "marker": "^"},
}


def _setup_mpl_style() -> None:
    """Apply publication-quality matplotlib style."""
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.edgecolor": _EDGE_COLOR,
        "axes.linewidth": 1.2,
        "axes.facecolor": "#FAFAFA",
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#E0E0E0",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.7,
        "figure.facecolor": "white",
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "legend.frameon": True,
        "legend.edgecolor": _EDGE_COLOR,
        "legend.fancybox": False,
        "legend.framealpha": 0.9,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
    })


def _smooth(values: np.ndarray, window: int = 5) -> np.ndarray:
    """Simple moving-average smoothing."""
    if window <= 1 or len(values) < window:
        return values
    kernel = np.ones(window) / window
    padded = np.concatenate([
        np.full(window // 2, values[0]),
        values,
        np.full(window // 2, values[-1]),
    ])
    return np.convolve(padded, kernel, mode="valid")[:len(values)]


class ExpertTrainer:
    """Trains a DQN expert on a single Atari game.

    Args:
        config: Configuration dictionary.
        env_id: Gymnasium environment ID (e.g., 'PongNoFrameskip-v4').
        union_actions: Sorted list of ALE action indices forming the union.
        logger: Logger instance for metrics.
        device: Torch device string.
        experiment_tag: Optional tag for checkpoint naming.
        resume_checkpoint: Optional path to resume training from.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        env_id: str,
        union_actions: List[int],
        logger: Logger,
        device: str = "cpu",
        experiment_tag: str = "",
        resume_checkpoint: Optional[str] = None,
    ):
        self.config = config
        self.env_id = env_id
        self.logger = logger
        self.device = device
        self.experiment_tag = experiment_tag
        self.union_actions = union_actions

        # Get valid actions for this game (indices into union)
        self.valid_actions = get_valid_actions(env_id, union_actions)
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

        # Collect eval rewards for live reward curve plotting
        self.eval_steps: List[int] = []
        self.eval_rewards: List[float] = []

        # Resume from checkpoint if provided
        if resume_checkpoint is not None and os.path.exists(resume_checkpoint):
            logger.info(f"Resuming expert from checkpoint: {resume_checkpoint}")
            self.agent.load(resume_checkpoint)
            # Load reward history from JSON sidecar (enables continuous curves)
            self._load_reward_history()
            logger.info(
                f"Resuming from env_step={self.agent.env_steps} "
                f"with {len(self.eval_steps)} prior eval points"
            )

        # Create environment
        env_cfg = config["env"]
        self.env = make_atari_env(
            env_id=env_id,
            union_actions=union_actions,
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

        # Resume-aware: start from where the agent left off
        start_step = self.agent.env_steps + 1
        if start_step > 1:
            self.logger.info(
                f"[{self.game_name}] Resuming training from step {start_step}"
            )

        for step in range(start_step, total_timesteps + 1):
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
                self.eval_steps.append(step)
                self.eval_rewards.append(eval_reward)
                # Persist reward history for resumable training
                self._save_reward_history()
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
                    # Clean up periodic step checkpoints (keep only best)
                    import glob
                    step_ckpts = glob.glob(
                        os.path.join(
                            checkpoint_dir, self.experiment_tag,
                            f"expert_{self.game_name}_step*.pt"
                        )
                    )
                    for ckpt in step_ckpts:
                        os.remove(ckpt)

            # Periodic checkpoint
            if step % save_freq == 0:
                ckpt_path = os.path.join(
                    checkpoint_dir,
                    self.experiment_tag,
                    f"expert_{self.game_name}_step{step}.pt",
                )
                self.agent.save(ckpt_path)

        final_eval = self.evaluate(eval_episodes)
        self.logger.info(
            f"[{self.game_name}] Training complete. "
            f"Final eval: {final_eval:.2f} | Best eval: {best_eval_reward:.2f}"
        )

        # Persist final reward history (ensures last eval point is captured)
        self._save_reward_history()

        # Generate reward curve
        figure_dir = self.config["logging"].get("figure_dir", "results/figures")
        self._save_reward_curve(figure_dir)

        result = {
            "final_reward": final_eval,
            "best_reward": best_eval_reward,
            "eval_steps": self.eval_steps,
            "eval_rewards": self.eval_rewards,
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
            union_actions=self.union_actions,
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

    # ── Reward history persistence (for resumable training) ───────────────────

    def _reward_history_path(self) -> str:
        """Return the path to the JSON sidecar file for this game's eval history."""
        checkpoint_dir = self.config["logging"]["checkpoint_dir"]
        return os.path.join(
            checkpoint_dir, self.experiment_tag,
            f"expert_{self.game_name}_history.json",
        )

    def _save_reward_history(self) -> None:
        """Persist eval_steps and eval_rewards to a lightweight JSON file.

        Called after every evaluation so the history survives interruption.
        Typical size: ~1-2 KB per game (20 eval points at 50K freq / 1M steps).
        """
        path = self._reward_history_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            "game_name": self.game_name,
            "env_id": self.env_id,
            "eval_steps": self.eval_steps,
            "eval_rewards": self.eval_rewards,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def _load_reward_history(self) -> None:
        """Load previously saved eval history from the JSON sidecar.

        Populates self.eval_steps and self.eval_rewards so that reward curves
        generated after resumed training span the full 0 → final_step range.
        """
        path = self._reward_history_path()
        if not os.path.exists(path):
            return
        try:
            with open(path, "r") as f:
                data = json.load(f)
            self.eval_steps = data.get("eval_steps", [])
            self.eval_rewards = data.get("eval_rewards", [])
        except (json.JSONDecodeError, KeyError) as exc:
            self.logger.info(
                f"[{self.game_name}] Could not load reward history: {exc}"
            )

    # ── Reward curve plotting ─────────────────────────────────────────────────

    def _save_reward_curve(self, figure_dir: str, smooth_window: int = 5) -> None:
        """Save a reward curve figure for this expert (PNG + SVG)."""
        if len(self.eval_steps) < 2:
            return

        try:
            import matplotlib
            matplotlib.use("Agg")  # non-interactive backend
            import matplotlib.pyplot as plt
        except ImportError:
            self.logger.info("matplotlib not available, skipping reward curve.")
            return

        _setup_mpl_style()

        steps = np.array(self.eval_steps)
        rewards = np.array(self.eval_rewards)
        smoothed = _smooth(rewards, smooth_window)
        best_idx = int(np.argmax(rewards))

        style = _GAME_STYLE.get(
            self.game_name, {"color": "#A8D8EA", "marker": "o"}
        )

        fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)

        # Raw data (faded)
        ax.plot(
            steps, rewards,
            color=style["color"], alpha=0.3, linewidth=0.8,
            label="Raw",
        )
        # Smoothed curve
        ax.plot(
            steps, smoothed,
            color=style["color"], marker=style["marker"],
            markeredgecolor=_EDGE_COLOR, markeredgewidth=0.8,
            markersize=4, linewidth=2.0,
            markevery=max(1, len(steps) // 15),
            label=f"Smoothed (w={smooth_window})",
        )
        # Best point
        ax.plot(
            steps[best_idx], rewards[best_idx],
            marker="*", color="#FF4444", markersize=14,
            markeredgecolor=_EDGE_COLOR, markeredgewidth=1.0,
            zorder=10, label=f"Best: {rewards[best_idx]:.1f}",
        )

        ax.set_xlabel("Environment Steps")
        ax.set_ylabel("Mean Evaluation Reward")
        ax.set_title(f"{self.game_name} Expert \u2014 Training Reward Curve")
        ax.legend(loc="best")
        ax.xaxis.set_major_formatter(
            plt.FuncFormatter(
                lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}k"
            )
        )

        for fmt in ("png", "svg"):
            out_dir = os.path.join(figure_dir, fmt)
            os.makedirs(out_dir, exist_ok=True)
            fig.savefig(
                os.path.join(out_dir, f"expert_{self.game_name}_reward_curve.{fmt}"),
                dpi=300, bbox_inches="tight", facecolor="white",
            )
        plt.close(fig)
        self.logger.info(
            f"[{self.game_name}] Reward curve saved to {figure_dir}/{{png,svg}}/"
        )

    @staticmethod
    def save_combined_reward_curves(
        all_results: List[Dict[str, Any]],
        figure_dir: str,
        smooth_window: int = 5,
    ) -> None:
        """Generate a combined multi-panel reward curve for all experts.

        Call this after all experts have been trained, passing the list of
        result dicts returned by each ExpertTrainer.train().
        """
        results_with_data = [
            r for r in all_results
            if len(r.get("eval_steps", [])) >= 2
        ]
        if not results_with_data:
            return

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return

        _setup_mpl_style()

        n = len(results_with_data)
        fig, axes = plt.subplots(
            1, n, figsize=(5.5 * n, 4.5), constrained_layout=True,
        )
        if n == 1:
            axes = [axes]

        for ax, r in zip(axes, results_with_data):
            game = r["game_name"]
            steps = np.array(r["eval_steps"])
            rewards = np.array(r["eval_rewards"])
            smoothed = _smooth(rewards, smooth_window)
            best_idx = int(np.argmax(rewards))
            style = _GAME_STYLE.get(game, {"color": "#A8D8EA", "marker": "o"})

            ax.plot(steps, rewards, color=style["color"], alpha=0.3, linewidth=0.8)
            ax.plot(
                steps, smoothed,
                color=style["color"], marker=style["marker"],
                markeredgecolor=_EDGE_COLOR, markeredgewidth=0.8,
                markersize=4, linewidth=2.0,
                markevery=max(1, len(steps) // 12),
            )
            ax.plot(
                steps[best_idx], rewards[best_idx],
                marker="*", color="#FF4444", markersize=14,
                markeredgecolor=_EDGE_COLOR, markeredgewidth=1.0,
                zorder=10,
            )
            ax.set_xlabel("Environment Steps")
            ax.set_ylabel("Mean Eval Reward")
            ax.set_title(f"{game} (Best: {rewards[best_idx]:.1f})")
            ax.xaxis.set_major_formatter(
                plt.FuncFormatter(
                    lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}k"
                )
            )

        fig.suptitle(
            "Expert DQN Training \u2014 Evaluation Reward Curves",
            fontsize=14, fontweight="bold", y=1.02,
        )

        for fmt in ("png", "svg"):
            out_dir = os.path.join(figure_dir, fmt)
            os.makedirs(out_dir, exist_ok=True)
            fig.savefig(
                os.path.join(out_dir, f"expert_all_reward_curves.{fmt}"),
                dpi=300, bbox_inches="tight", facecolor="white",
            )
        plt.close(fig)
