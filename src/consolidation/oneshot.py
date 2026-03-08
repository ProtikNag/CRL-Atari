"""
One-Shot Joint Consolidation.

All experts are considered simultaneously: Fisher and gradient are computed
over every task at the global model position, then a single averaged Taylor
correction is applied.  Unlike the iterative approach, no task sees a
different Fisher landscape than any other because every computation happens
at the same w_global.

Algorithm:
    1.  Initialise w_global from the first expert.
    2.  At w_global, compute Fisher_k and gradient_k for each task k.
    3.  Form joint Fisher H = Σ_k Fisher_k  and  joint gradient g = Σ_k g_k.
    4.  For each expert k:
            δ_k = w_expert_k − w_global   (drift, action-masked for heads)
            c_k = (H + λI)^{−1} (λ δ_k − g)
    5.  Average correction  c = (1/K) Σ_k c_k.
    6.  w_new = w_global + η · c.
"""

import copy
import json
import os
from typing import Any, Dict, List, Optional

import torch

from src.consolidation.htcl import HTCLConsolidator
from src.models.dqn import DQNNetwork
from src.utils.logger import Logger


class OneShotConsolidator:
    """One-Shot Joint Consolidation via second-order Taylor expansion.

    Computes Fisher and gradient jointly over all tasks at the initial
    global model position, then applies a single averaged Taylor correction
    toward every expert simultaneously.

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

        oneshot_cfg = config.get("oneshot", config.get("htcl", {}))
        self.lambda_val = oneshot_cfg.get("lambda_htcl", 100.0)
        self.eta = oneshot_cfg.get("eta", 0.9)
        self.fisher_samples = oneshot_cfg.get("fisher_samples", 5000)

        # Internal HTCL instance provides Fisher/gradient/Taylor helpers
        self._htcl = HTCLConsolidator(config, device=device, logger=logger)

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
        """Run one-shot joint consolidation.

        Args:
            global_model: Global model initialised from first expert.
            expert_results: List of expert result dicts (all tasks).
            filtered_states_list: High-confidence states per expert.
            expert_models: Frozen expert DQN models (for KL logging).
            lambda_override: Override default lambda.

        Returns:
            Consolidated DQNNetwork.
        """
        lam = lambda_override if lambda_override is not None else self.lambda_val

        if self.logger:
            self.logger.info(
                f"Starting One-Shot Joint Consolidation "
                f"(λ={lam}, η={self.eta})..."
            )

        consolidated = copy.deepcopy(global_model).to(self.device)
        global_sd = {
            name: param.clone()
            for name, param in consolidated.state_dict().items()
        }

        # ── Step 1: Compute joint Fisher and joint gradient ──
        joint_fisher: Dict[str, torch.Tensor] = {
            name: torch.zeros_like(param)
            for name, param in consolidated.named_parameters()
            if param.requires_grad
        }
        joint_gradient: Dict[str, torch.Tensor] = {
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
                    f"  Computing Fisher/gradient for {game_name} "
                    f"at global position..."
                )

            consolidated.load_state_dict(global_sd)

            task_fisher = self._htcl.compute_diagonal_fisher(
                consolidated, valid_actions, states=filt_states,
            )
            task_gradient = self._htcl.compute_gradient(
                consolidated, valid_actions, states=filt_states,
            )

            for name in task_fisher:
                joint_fisher[name] += task_fisher[name]
                joint_gradient[name] += task_gradient[name]

            # Log per-task Fisher statistics
            self._htcl._log_fisher_statistics(
                task_fisher, task_idx, game_name,
                prefix="oneshot", is_cumulative=False,
            )

        # Log joint Fisher statistics
        self._htcl._log_fisher_statistics(
            joint_fisher, len(expert_results), "joint",
            prefix="oneshot", is_cumulative=True,
        )

        # ── Step 2: Validate lambda against joint Fisher ──
        eff_lam = self._htcl._ensure_lambda_constraint(joint_fisher, lam)

        if self.logger:
            self.logger.info(
                f"  Joint Fisher computed | effective λ = {eff_lam:.4f}"
            )

        # ── Step 3: Compute per-expert Taylor corrections and average ──
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

        avg_correction: Dict[str, torch.Tensor] = {
            name: torch.zeros_like(param)
            for name, param in consolidated.named_parameters()
            if param.requires_grad
        }
        num_tasks = len(expert_results)

        for result in expert_results:
            local_sd = result["policy_state_dict"]
            game_name = result["game_name"]
            task_valid = result["valid_actions"]
            unused = (
                sorted(set(range(n_actions)) - set(task_valid))
                if len(task_valid) < n_actions
                else []
            )

            for name in joint_fisher:
                w_global = global_sd[name].to(self.device)
                w_local = local_sd[name].to(self.device)
                h_diag = joint_fisher[name]
                g = joint_gradient[name]

                delta_d = w_local - w_global

                # Action mask: zero drift for untrained action rows
                if unused and name in head_names:
                    if name in head_weight_names:
                        delta_d[unused, :] = 0.0
                    else:
                        delta_d[unused] = 0.0

                numerator = eff_lam * delta_d - g
                denominator = h_diag + eff_lam
                correction = numerator / (denominator + 1e-8)
                avg_correction[name] += correction / num_tasks

        # ── Step 4: Apply averaged correction ──
        updated_sd = {}
        for name in global_sd:
            if name in avg_correction:
                updated_sd[name] = (
                    global_sd[name].to(self.device)
                    + self.eta * avg_correction[name]
                )
            else:
                # Non-trainable params: keep global
                updated_sd[name] = global_sd[name]

        # Log update norm
        if self.logger:
            update_norm = sum(
                (updated_sd[n] - global_sd[n].to(self.device))
                .norm().item() ** 2
                for n in joint_fisher
            ) ** 0.5
            self.logger.info(
                f"  One-shot update_norm = {update_norm:.4f}"
            )

        consolidated.load_state_dict(updated_sd)
        consolidated.eval()

        if self.logger:
            self.logger.info("One-Shot Joint Consolidation complete.")

        return consolidated

    def save_fisher_log(self, path: str) -> None:
        """Delegate to internal HTCL consolidator."""
        self._htcl.save_fisher_log(path)
