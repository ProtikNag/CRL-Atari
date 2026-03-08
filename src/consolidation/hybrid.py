"""
Hybrid Consolidation with Knowledge Distillation (CRL.pdf, Section 5, Algorithm 2).

Two-phase approach:
    Phase 0 -- Anchor initialisation: w_bar = (1/N) Sigma_i w_e^(i); w_g = w_bar.
    Phase 1 -- Multi-round joint Taylor consolidation (Algorithm 1 / Section 4).
               Each round re-expands at w_g, computes averaged Fisher and gradient
               over all experts jointly, and applies a single Taylor update with
               step size eta_0 * gamma^k.
    Phase 2 -- Knowledge-distillation fine-tuning:
               L_KD = (1/N) Sigma_i T^2 D_KL(sigma(Q_expert(B_i)/T) || sigma(Q_student(B_i)/T))
               w_g <- w_g - eta_KD grad_{w_g} L_KD   for T distillation epochs.

The rationale (Theorem 5.2, Corollary 5.4) is that Phase 1 provides a warm
start with advantage phi, reducing KD epochs by a factor ~1 - 1/phi.
"""

import copy
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.consolidation.htcl import HTCLConsolidator
from src.models.dqn import DQNNetwork
from src.utils.logger import Logger


class HybridConsolidator:
    """Hybrid Consolidation: Joint Taylor + KD  (Algorithm 2).

    Phase 1 runs multi-round joint Taylor consolidation (Algorithm 1)
    at the ensemble-mean anchor.  Phase 2 refines via knowledge
    distillation with reduced learning rate and fewer epochs (since the
    starting point is already close to optimal).

    The caller MUST initialize the global model to the ensemble mean
    w_bar = (1/N) Sigma_i w_e^(i)  before calling ``consolidate()``.

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

        # Phase 1 (Joint Taylor) parameters
        self.lambda_val: float = hybrid_cfg.get(
            "lambda_htcl", htcl_cfg.get("lambda_htcl", 100.0),
        )
        self.eta_0: float = hybrid_cfg.get("eta", htcl_cfg.get("eta", 0.9))
        self.gamma: float = hybrid_cfg.get("gamma", 0.5)
        self.num_rounds: int = hybrid_cfg.get(
            "num_rounds", hybrid_cfg.get("num_passes", htcl_cfg.get("num_passes", 3)),
        )
        self.recompute_fisher: bool = hybrid_cfg.get("recompute_fisher", False)

        # Phase 2 (KD refinement) parameters
        dist_cfg = config.get("distillation", {})
        self.kd_epochs: int = hybrid_cfg.get(
            "kd_epochs", max(1, dist_cfg.get("distill_epochs", 50) // 2),
        )
        self.kd_lr: float = hybrid_cfg.get(
            "kd_lr", dist_cfg.get("distill_lr", 5e-5) * 0.5,
        )
        self.kd_temperature: float = hybrid_cfg.get(
            "temperature", dist_cfg.get("temperature", 2.0),
        )
        self.kd_batch_size: int = hybrid_cfg.get(
            "kd_batch_size", dist_cfg.get("distill_batch_size", 64),
        )

        # Internal HTCL helper
        self._htcl = HTCLConsolidator(config, device=device, logger=logger)

    # ------------------------------------------------------------------
    # Fisher / gradient helper
    # ------------------------------------------------------------------

    def _compute_avg_fisher_gradient(
        self,
        consolidated: DQNNetwork,
        expert_results: List[Dict[str, Any]],
        filtered_states_list: List[torch.Tensor],
        global_sd: Dict[str, torch.Tensor],
        num_tasks: int,
        round_idx: int = 0,
    ) -> tuple:
        """Compute averaged Fisher and gradient at current model position."""
        avg_fisher: Dict[str, torch.Tensor] = {
            name: torch.zeros_like(param)
            for name, param in consolidated.named_parameters()
            if param.requires_grad
        }
        avg_gradient: Dict[str, torch.Tensor] = {
            name: torch.zeros_like(param)
            for name, param in consolidated.named_parameters()
            if param.requires_grad
        }

        for task_idx, (result, filt_states) in enumerate(
            zip(expert_results, filtered_states_list)
        ):
            game_name = result["game_name"]
            valid_actions = result["valid_actions"]

            if self.logger:
                self.logger.info(
                    f"  [{game_name}] Computing Fisher/gradient..."
                )

            consolidated.load_state_dict(global_sd)

            task_fisher = self._htcl.compute_diagonal_fisher(
                consolidated, valid_actions, states=filt_states,
            )
            task_gradient = self._htcl.compute_gradient(
                consolidated, valid_actions, states=filt_states,
            )

            for name in task_fisher:
                avg_fisher[name] += task_fisher[name] / num_tasks
                avg_gradient[name] += task_gradient[name] / num_tasks

        return avg_fisher, avg_gradient

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
        """Run hybrid two-phase consolidation (Algorithm 2).

        The caller MUST initialize *global_model* to the ensemble mean
        w_bar = (1/N) Sigma_i w_e^(i)  before calling this method.

        Args:
            global_model: Model initialised at the ensemble-mean anchor.
            expert_results: List of expert result dicts (all tasks).
            filtered_states_list: High-confidence states per expert.
            expert_models: Frozen expert DQN models.
            lambda_override: Override default lambda for Phase 1.

        Returns:
            Consolidated DQNNetwork.
        """
        lam = lambda_override if lambda_override is not None else self.lambda_val
        num_tasks = len(expert_results)
        K = self.num_rounds

        if self.logger:
            self.logger.info(
                f"Starting Hybrid Consolidation "
                f"(Phase 1: Joint Taylor K={K}, lambda={lam}, "
                f"eta_0={self.eta_0}, gamma={self.gamma}, "
                f"recompute_fisher={self.recompute_fisher} | "
                f"Phase 2: KD epochs={self.kd_epochs}, lr={self.kd_lr:.1e})..."
            )

        # ══════════════════════════════════════════════════════════════
        # Phase 1: Multi-round joint Taylor consolidation (Algorithm 1)
        # ══════════════════════════════════════════════════════════════
        if self.logger:
            self.logger.info(
                "\n============================================"
                "\n Phase 1: Multi-Round Joint Taylor"
                "\n============================================"
            )

        consolidated = copy.deepcopy(global_model).to(self.device)
        global_sd = {
            name: param.clone()
            for name, param in consolidated.state_dict().items()
        }

        # Head-layer names for action masking (Remark 7.1)
        n_actions = consolidated.unified_action_dim
        head_weight_names = {
            n for n in global_sd
            if n.endswith(".weight")
            and global_sd[n].shape[0] == n_actions
            and ("fc." in n or "advantage_stream." in n)
        }
        head_bias_names = {
            n for n in global_sd
            if n.endswith(".bias")
            and global_sd[n].shape[0] == n_actions
            and ("fc." in n or "advantage_stream." in n)
        }
        head_names = head_weight_names | head_bias_names

        # Pre-compute Fisher when recompute_fisher=False
        cached_fisher = None
        cached_gradient = None
        cached_lam = None

        if not self.recompute_fisher:
            if self.logger:
                self.logger.info(
                    "Computing Fisher/gradient ONCE at initial anchor "
                    "(recompute_fisher=false; each task seen once)..."
                )
            consolidated.load_state_dict(global_sd)
            cached_fisher, cached_gradient = (
                self._compute_avg_fisher_gradient(
                    consolidated, expert_results, filtered_states_list,
                    global_sd, num_tasks, round_idx=0,
                )
            )
            cached_lam = self._htcl._ensure_lambda_constraint(
                cached_fisher, lam,
            )

        for round_k in range(K):
            eta_k = self.eta_0 * (self.gamma ** round_k)

            if self.logger:
                self.logger.info(
                    f"\n--- Round {round_k + 1}/{K} "
                    f"(eta_k={eta_k:.6f}) ---"
                )

            # Fisher/gradient (re-computed or cached)
            if self.recompute_fisher:
                consolidated.load_state_dict(global_sd)
                avg_fisher, avg_gradient = (
                    self._compute_avg_fisher_gradient(
                        consolidated, expert_results, filtered_states_list,
                        global_sd, num_tasks, round_idx=round_k,
                    )
                )
                eff_lam = self._htcl._ensure_lambda_constraint(
                    avg_fisher, lam,
                )
            else:
                avg_fisher = cached_fisher
                avg_gradient = cached_gradient
                eff_lam = cached_lam

            # Per-expert corrections averaged = u* with d*
            avg_correction: Dict[str, torch.Tensor] = {
                name: torch.zeros_like(param)
                for name, param in consolidated.named_parameters()
                if param.requires_grad
            }

            for result in expert_results:
                local_sd = result["policy_state_dict"]
                task_valid = result["valid_actions"]
                unused = (
                    sorted(set(range(n_actions)) - set(task_valid))
                    if len(task_valid) < n_actions
                    else []
                )

                for name in avg_fisher:
                    w_global = global_sd[name].to(self.device)
                    w_local = local_sd[name].to(self.device)
                    h_diag = avg_fisher[name]
                    g = avg_gradient[name]

                    delta_d = w_local - w_global

                    if unused and name in head_names:
                        if name in head_weight_names:
                            delta_d[unused, :] = 0.0
                        else:
                            delta_d[unused] = 0.0

                    numerator = eff_lam * delta_d - g
                    denominator = h_diag + eff_lam
                    correction = numerator / (denominator + 1e-8)
                    avg_correction[name] += correction / num_tasks

            # Apply: w_g^(k+1) = w_bar + eta_k * u*
            updated_sd = {}
            for name in global_sd:
                if name in avg_correction:
                    updated_sd[name] = (
                        global_sd[name].to(self.device)
                        + eta_k * avg_correction[name]
                    )
                else:
                    updated_sd[name] = global_sd[name]

            if self.logger:
                update_norm = sum(
                    (updated_sd[n] - global_sd[n].to(self.device))
                    .norm().item() ** 2
                    for n in avg_fisher
                ) ** 0.5
                self.logger.info(
                    f"  Round {round_k + 1} | update_norm={update_norm:.4f} "
                    f"| effective lambda={eff_lam:.4f}"
                )

            global_sd = updated_sd

        # Phase 1 complete
        consolidated.load_state_dict(global_sd)

        if self.logger:
            p1_norm = sum(
                p.data.norm().item() ** 2 for p in consolidated.parameters()
            ) ** 0.5
            self.logger.info(
                f"  Phase 1 complete | param norm = {p1_norm:.4f}"
            )

        # ══════════════════════════════════════════════════════════════
        # Phase 2: Knowledge Distillation refinement (Algorithm 2, lines 16-19)
        # L_KD = (1/N) Sigma_i T^2 D_KL(sigma(Q_e(B_i)/T) || sigma(Q_g(B_i)/T))
        # ══════════════════════════════════════════════════════════════
        if self.logger:
            self.logger.info(
                "\n============================================"
                "\n Phase 2: Knowledge Distillation Refinement"
                "\n============================================"
            )

        student = copy.deepcopy(consolidated).to(self.device)
        student.train()

        # Build frozen teacher models
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
        buffer_size_per_task = self.config.get("distillation", {}).get(
            "buffer_size_per_task", 10_000,
        )
        task_q_stats = []
        for teacher, result in zip(teachers, expert_results):
            buffer = result["replay_buffer"]
            valid_actions = result["valid_actions"]
            states = buffer.sample_states(
                min(buffer_size_per_task, len(buffer)),
            )
            with torch.no_grad():
                q_vals = teacher(states)
                valid_q = q_vals[:, valid_actions]
                q_mean = valid_q.mean().item()
                q_std = max(valid_q.std().item(), 1e-6)
            task_q_stats.append({"mean": q_mean, "std": q_std})

        # KD training loop (Algorithm 2, lines 16-19)
        T = self.kd_temperature
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
                        teacher_soft = F.softmax(teacher_q_norm / T, dim=1)

                    student_q = student(states)
                    student_q_norm = (student_q - q_mean) / q_std
                    student_log_soft = F.log_softmax(
                        student_q_norm / T, dim=1,
                    )

                    # D_KL(teacher || student)  scaled by T^2
                    kl_loss = F.kl_div(
                        student_log_soft,
                        teacher_soft,
                        reduction="batchmean",
                    )
                    loss = kl_loss * (T ** 2)

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                    optimizer.step()

                    epoch_loss += loss.item()
                    total_batches += 1

            avg_loss = epoch_loss / max(total_batches, 1)
            if self.logger and (
                (epoch + 1) % max(1, self.kd_epochs // 10) == 0
            ):
                self.logger.info(
                    f"  KD epoch {epoch + 1}/{self.kd_epochs} | "
                    f"avg loss: {avg_loss:.6f}"
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

        return student
