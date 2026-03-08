"""
Hybrid Consolidation with Knowledge Distillation.

Two-phase approach:
    Phase 1 — Multi-round iterative HTCL consolidation produces a model
              that is structurally close to every expert via second-order
              Taylor updates.
    Phase 2 — Knowledge distillation refinement: the Phase-1 model serves
              as the student initialisation, and expert teachers further
              refine it by matching soft Q-value distributions.

The rationale is that HTCL provides a principled, closed-form starting
point that already respects the curvature of each task's loss landscape,
while KD adds a gradient-based fine-tuning phase that can recover subtle
policy details the Taylor approximation may have missed.
"""

import copy
import json
import os
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.consolidation.htcl import HTCLConsolidator
from src.consolidation.distillation import DistillationConsolidator
from src.models.dqn import DQNNetwork
from src.utils.logger import Logger


class HybridConsolidator:
    """Hybrid Consolidation: HTCL followed by Knowledge Distillation.

    Phase 1 uses multi-round iterative HTCL to produce a good initial
    model.  Phase 2 refines it via knowledge distillation with reduced
    learning rate and fewer epochs (since the starting point is already
    close to optimal).

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

        hybrid_cfg = config.get("hybrid", {})
        htcl_cfg = config.get("htcl", {})

        # Phase 1 (HTCL) parameters
        self.lambda_val = hybrid_cfg.get("lambda_htcl", htcl_cfg.get("lambda_htcl", 100.0))
        self.eta = hybrid_cfg.get("eta", htcl_cfg.get("eta", 0.9))
        self.num_passes = hybrid_cfg.get("num_passes", htcl_cfg.get("num_passes", 3))
        self.joint_refinement = hybrid_cfg.get(
            "joint_refinement", htcl_cfg.get("joint_refinement", True),
        )

        # Phase 2 (KD refinement) parameters
        dist_cfg = config.get("distillation", {})
        self.kd_epochs = hybrid_cfg.get("kd_epochs", max(1, dist_cfg.get("distill_epochs", 50) // 2))
        self.kd_lr = hybrid_cfg.get("kd_lr", dist_cfg.get("distill_lr", 5e-5) * 0.5)
        self.kd_temperature = hybrid_cfg.get("temperature", dist_cfg.get("temperature", 2.0))
        self.kd_batch_size = hybrid_cfg.get("kd_batch_size", dist_cfg.get("distill_batch_size", 64))

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def consolidate(
        self,
        global_model: DQNNetwork,
        expert_results: List[Dict[str, Any]],
        filtered_states_list: List[torch.Tensor],
        expert_models: Optional[List[DQNNetwork]] = None,
        lambda_override: Optional[float] = None,
    ) -> DQNNetwork:
        """Run hybrid two-phase consolidation.

        Args:
            global_model: Global model initialised from first expert.
            expert_results: List of expert result dicts (all tasks).
            filtered_states_list: High-confidence states per expert.
            expert_models: Frozen expert DQN models.
            lambda_override: Override default lambda for Phase 1.

        Returns:
            Consolidated DQNNetwork.
        """
        lam = lambda_override if lambda_override is not None else self.lambda_val

        if self.logger:
            self.logger.info(
                f"Starting Hybrid Consolidation "
                f"(Phase 1: HTCL λ={lam}, passes={self.num_passes} | "
                f"Phase 2: KD epochs={self.kd_epochs}, lr={self.kd_lr:.1e})..."
            )

        # ── Phase 1: Multi-round iterative HTCL ──
        if self.logger:
            self.logger.info("\n── Phase 1: Multi-Round Iterative HTCL ──")

        htcl_consolidator = HTCLConsolidator(
            self.config, device=self.device, logger=self.logger,
        )

        model_phase1 = copy.deepcopy(global_model).to(self.device)

        htcl_consolidator.register_initial_task(
            model_phase1,
            expert_results[0]["valid_actions"],
            filtered_states_list[0],
            expert_results[0]["game_name"],
        )

        htcl_result = htcl_consolidator.consolidate(
            model_phase1,
            expert_results,
            filtered_states_list=filtered_states_list,
            expert_models=expert_models,
            lambda_override=lam,
            num_passes=self.num_passes,
            joint_refinement=self.joint_refinement,
        )

        if self.logger:
            p1_norm = sum(
                p.data.norm().item() ** 2 for p in htcl_result.parameters()
            ) ** 0.5
            self.logger.info(
                f"  Phase 1 complete | param norm = {p1_norm:.4f}"
            )

        # ── Phase 2: Knowledge Distillation refinement ──
        if self.logger:
            self.logger.info("\n── Phase 2: Knowledge Distillation Refinement ──")

        student = copy.deepcopy(htcl_result).to(self.device)
        student.train()

        # Build teacher models (frozen)
        teachers: List[DQNNetwork] = []
        for result in expert_results:
            teacher = copy.deepcopy(global_model).to(self.device)
            teacher.load_state_dict(result["policy_state_dict"])
            teacher.eval()
            for p in teacher.parameters():
                p.requires_grad_(False)
            teachers.append(teacher)

        optimizer = torch.optim.AdamW(student.parameters(), lr=self.kd_lr)

        # Precompute per-task Q-value statistics for normalization
        task_q_stats = []
        buffer_size_per_task = self.config.get("distillation", {}).get(
            "buffer_size_per_task", 10_000,
        )
        for teacher, result in zip(teachers, expert_results):
            buffer = result["replay_buffer"]
            valid_actions = result["valid_actions"]
            states = buffer.sample_states(min(buffer_size_per_task, len(buffer)))
            with torch.no_grad():
                q_vals = teacher(states)
                valid_q = q_vals[:, valid_actions]
                q_mean = valid_q.mean().item()
                q_std = max(valid_q.std().item(), 1e-6)
            task_q_stats.append({"mean": q_mean, "std": q_std})

        # KD training loop
        for epoch in range(self.kd_epochs):
            epoch_loss = 0.0
            total_batches = 0

            for task_idx, (teacher, result) in enumerate(
                zip(teachers, expert_results)
            ):
                buffer = result["replay_buffer"]
                q_mean = task_q_stats[task_idx]["mean"]
                q_std = task_q_stats[task_idx]["std"]

                num_batches = max(
                    1,
                    min(len(buffer), buffer_size_per_task) // self.kd_batch_size,
                )

                for _ in range(num_batches):
                    states = buffer.sample_states(self.kd_batch_size)

                    with torch.no_grad():
                        teacher_q = teacher(states)
                        teacher_q_norm = (teacher_q - q_mean) / q_std
                        teacher_soft = F.softmax(
                            teacher_q_norm / self.kd_temperature, dim=1,
                        )

                    student_q = student(states)
                    student_q_norm = (student_q - q_mean) / q_std
                    student_log_soft = F.log_softmax(
                        student_q_norm / self.kd_temperature, dim=1,
                    )

                    kl_loss = F.kl_div(
                        student_log_soft,
                        teacher_soft,
                        reduction="batchmean",
                    )
                    loss = kl_loss * (self.kd_temperature ** 2)

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                    optimizer.step()

                    epoch_loss += loss.item()
                    total_batches += 1

            avg_loss = epoch_loss / max(total_batches, 1)
            if self.logger and (epoch + 1) % max(1, self.kd_epochs // 10) == 0:
                self.logger.info(
                    f"  KD refinement epoch {epoch + 1}/{self.kd_epochs} | "
                    f"Avg loss: {avg_loss:.6f}"
                )

        student.eval()

        if self.logger:
            p2_norm = sum(
                p.data.norm().item() ** 2 for p in student.parameters()
            ) ** 0.5
            self.logger.info(
                f"  Phase 2 complete | param norm = {p2_norm:.4f}"
            )
            self.logger.info("Hybrid Consolidation complete.")

        # Keep fisher log reference
        self._fisher_log = htcl_consolidator.fisher_log

        return student

    def save_fisher_log(self, path: str) -> None:
        """Save Fisher statistics log from Phase 1."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(getattr(self, "_fisher_log", []), f, indent=2)
        if self.logger:
            self.logger.info(f"Hybrid: Fisher log saved to {path}")
