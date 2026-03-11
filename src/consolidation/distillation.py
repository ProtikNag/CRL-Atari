"""
Knowledge Distillation consolidation for continual RL.

Trains a student (global) model to mimic the Q-value distributions of
multiple teacher (expert) models across all tasks.

Follows the L_KL loss from Rusu et al. (2016) "Policy Distillation":
    L_KL = Σ_i softmax(q_T / τ) · ln[ softmax(q_T / τ) / softmax(q_S) ]
where τ is applied ONLY to the teacher (sharpening; default τ=0.01).
The student uses softmax with no temperature (effectively τ=1).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import os
import numpy as np
from typing import Dict, Any, List, Optional

from src.models.dqn import DQNNetwork
from src.data.replay_buffer import ReplayBuffer
from src.utils.logger import Logger


class DistillationConsolidator:
    """Knowledge distillation for consolidating expert models.

    The student model is trained to match the soft Q-value distributions
    of each expert on their respective replay data, with Q-value
    normalization across tasks.

    Args:
        config: Configuration dictionary.
        device: Torch device string.
        logger: Logger instance.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        device: str = "cpu",
        logger: Optional[Logger] = None,
    ):
        self.config = config
        self.device = device
        self.logger = logger
        self.distill_cfg = config["distillation"]

        self.temperature = self.distill_cfg["temperature"]
        self.alpha = self.distill_cfg["alpha"]
        self.distill_epochs = self.distill_cfg["distill_epochs"]
        self.distill_lr = self.distill_cfg["distill_lr"]
        self.batch_size = self.distill_cfg["distill_batch_size"]

    def consolidate(
        self,
        global_model: DQNNetwork,
        expert_results: List[Dict[str, Any]],
    ) -> DQNNetwork:
        """Consolidate expert knowledge via knowledge distillation.

        For each epoch, iterates over all tasks. For each task, samples a
        batch from the task's replay buffer, computes teacher (expert) soft
        Q-values, and trains the student to match them.

        Q-values are normalized per-task before distillation to ensure scale
        comparability.

        Args:
            global_model: The current global model (student).
            expert_results: List of expert results with state dicts and buffers.

        Returns:
            The consolidated (distilled) model.
        """
        if self.logger:
            self.logger.info("Starting Knowledge Distillation consolidation...")

        student = copy.deepcopy(global_model).to(self.device)
        student.train()

        # Build teacher models (frozen)
        teachers: List[DQNNetwork] = []
        for result in expert_results:
            teacher = copy.deepcopy(global_model).to(self.device)
            teacher.load_state_dict(result["policy_state_dict"])
            teacher.eval()
            teachers.append(teacher)

        optimizer = torch.optim.AdamW(student.parameters(), lr=self.distill_lr)
        buffer_size_per_task = self.distill_cfg.get("buffer_size_per_task", 10_000)

        # Precompute per-task Q-value statistics for normalization
        task_q_stats = []
        for idx, (teacher, result) in enumerate(zip(teachers, expert_results)):
            buffer = result["replay_buffer"]
            valid_actions = result["valid_actions"]
            states = buffer.sample_states(min(buffer_size_per_task, len(buffer)))
            with torch.no_grad():
                q_vals = teacher(states)
                # Only consider valid actions
                valid_q = q_vals[:, valid_actions]
                q_mean = valid_q.mean().item()
                q_std = max(valid_q.std().item(), 1e-6)
            task_q_stats.append({"mean": q_mean, "std": q_std})
            if self.logger:
                self.logger.info(
                    f"Task {result['game_name']}: Q-value stats "
                    f"mean={q_mean:.3f}, std={q_std:.3f}"
                )

        # Training loop
        for epoch in range(self.distill_epochs):
            epoch_loss = 0.0
            total_batches = 0

            for task_idx, (teacher, result) in enumerate(
                zip(teachers, expert_results)
            ):
                buffer = result["replay_buffer"]
                valid_actions = result["valid_actions"]
                q_mean = task_q_stats[task_idx]["mean"]
                q_std = task_q_stats[task_idx]["std"]

                num_batches = max(1, min(len(buffer), buffer_size_per_task) // self.batch_size)

                for _ in range(num_batches):
                    states = buffer.sample_states(self.batch_size)

                    # Teacher Q-values: normalize then sharpen with
                    # temperature τ (Rusu et al. 2016, Section 3.2).
                    with torch.no_grad():
                        teacher_q = teacher(states)
                        teacher_q_norm = (teacher_q - q_mean) / q_std
                        teacher_soft = F.softmax(
                            teacher_q_norm / self.temperature, dim=1
                        )

                    # Student Q-values: softmax with NO temperature
                    # (τ_student = 1, per the paper's L_KL formulation).
                    student_q = student(states)
                    student_q_norm = (student_q - q_mean) / q_std
                    student_log_soft = F.log_softmax(
                        student_q_norm, dim=1
                    )

                    # KL(teacher || student) — no T² scaling since
                    # temperature is only applied to the teacher side.
                    loss = F.kl_div(
                        student_log_soft,
                        teacher_soft,
                        reduction="batchmean",
                    )

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                    optimizer.step()

                    epoch_loss += loss.item()
                    total_batches += 1

            avg_loss = epoch_loss / max(total_batches, 1)
            if self.logger and (epoch + 1) % max(1, self.distill_epochs // 10) == 0:
                self.logger.info(
                    f"Distillation epoch {epoch + 1}/{self.distill_epochs} | "
                    f"Avg loss: {avg_loss:.6f}"
                )

        student.eval()
        if self.logger:
            self.logger.info("Knowledge Distillation consolidation complete.")
        return student
