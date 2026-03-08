"""
One-Shot Joint Consolidation  (CRL.pdf, Section 3, Theorem 3.4).

All experts are considered simultaneously: Fisher and gradient are computed
at the ensemble-mean anchor, then a single joint Taylor correction is
applied.

Algorithm  (Theorem 3.4  /  Remark 3.6  /  Eq. 12):
    1.  Anchor:  w_bar = (1/N) Sigma_i w_e^(i).
    2.  At w_bar, compute per-task Fisher F_i and gradient g_i.
    3.  Average:  F_bar = (1/N) Sigma_i F_i,   g_bar = (1/N) Sigma_i g_i.
    4.  Drift per expert:  Delta_d^(i) = w_e^(i) - w_bar  (action-masked
        for head layers per Remark 7.1).
    5.  Weighted drift centroid (uniform lambda):
            d* = (1/N) Sigma_i Delta_d^(i).
        NOTE: with anchor = mean, d* = 0 (Remark 3.6).
    6.  Update:  u* = (F_bar + lambda I)^{-1} [lambda d* - g_bar].
    7.  Result:  w_g = w_bar + u*   (no step-size for one-shot; Eq. 12).

With uniform lambda and anchor = ensemble mean, d* = 0, so the update
simplifies to a Newton step:  u* = -(F_bar + lambda I)^{-1} g_bar.
"""

import copy
from typing import Any, Dict, List, Optional

import torch

from src.consolidation.htcl import HTCLConsolidator
from src.models.dqn import DQNNetwork
from src.utils.logger import Logger


class OneShotConsolidator:
    """One-Shot Joint Consolidation (Theorem 3.4).

    Computes averaged Fisher and averaged gradient over all tasks at
    the ensemble-mean anchor, then applies a single closed-form Taylor
    correction.

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

        oneshot_cfg = config.get("oneshot", config.get("htcl", {}))
        self.lambda_val: float = oneshot_cfg.get("lambda_htcl", 100.0)
        self.fisher_samples: int = oneshot_cfg.get("fisher_samples", 5000)

        # Internal HTCL instance provides Fisher/gradient/logging helpers
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
        """Run one-shot joint consolidation (Theorem 3.4).

        The caller MUST initialize *global_model* to the ensemble mean
        w_bar = (1/N) Sigma_i w_e^(i)  before calling this method.

        Args:
            global_model: Model initialised at the ensemble-mean anchor.
            expert_results: List of expert result dicts (all tasks).
            filtered_states_list: High-confidence states per expert.
            expert_models: Frozen expert DQN models (unused, for API compat).
            lambda_override: Override default lambda.

        Returns:
            Consolidated DQNNetwork.
        """
        lam = lambda_override if lambda_override is not None else self.lambda_val
        num_tasks = len(expert_results)

        if self.logger:
            self.logger.info(
                f"Starting One-Shot Joint Consolidation "
                f"(lambda={lam}, N={num_tasks})..."
            )

        consolidated = copy.deepcopy(global_model).to(self.device)
        global_sd = {
            name: param.clone()
            for name, param in consolidated.state_dict().items()
        }

        # ── Step 1: Compute AVERAGED Fisher and gradient ──
        # F_bar = (1/N) Sigma_i F_i,   g_bar = (1/N) Sigma_i g_i
        # (paper Eq. 8)
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
                    f"  Computing Fisher/gradient for {game_name} "
                    f"at anchor position..."
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

        # ── Step 2: Validate lambda (Lemma 3.7) ──
        # With diagonal Fisher (F_bar >= 0), any lambda > 0 suffices.
        eff_lam = self._htcl._ensure_lambda_constraint(avg_fisher, lam)

        if self.logger:
            self.logger.info(
                f"  Averaged Fisher computed | effective lambda = {eff_lam:.4f}"
            )

        # ── Step 3: Compute per-expert Taylor corrections and average ──
        # Per-expert:  c_k = (F_bar + lambda I)^{-1} [lambda Delta_d^(k) - g_bar]
        # Average:     (1/N) Sigma_k c_k = (F_bar + lambda I)^{-1} [lambda d* - g_bar]
        #              which equals u* from Theorem 3.4.
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

        for result in expert_results:
            local_sd = result["policy_state_dict"]
            game_name = result["game_name"]
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

                # Action mask (Remark 7.1): zero drift for untrained rows
                if unused and name in head_names:
                    if name in head_weight_names:
                        delta_d[unused, :] = 0.0
                    else:
                        delta_d[unused] = 0.0

                numerator = eff_lam * delta_d - g
                denominator = h_diag + eff_lam
                correction = numerator / (denominator + 1e-8)
                avg_correction[name] += correction / num_tasks

        # ── Step 4: w_g = w_bar + u*  (Eq. 12, no step size) ──
        updated_sd = {}
        for name in global_sd:
            if name in avg_correction:
                updated_sd[name] = (
                    global_sd[name].to(self.device) + avg_correction[name]
                )
            else:
                # Non-trainable params: keep anchor value
                updated_sd[name] = global_sd[name]

        # Log update norm
        if self.logger:
            update_norm = sum(
                (updated_sd[n] - global_sd[n].to(self.device))
                .norm().item() ** 2
                for n in avg_fisher
            ) ** 0.5
            self.logger.info(
                f"  One-shot update_norm = {update_norm:.4f}"
            )

        consolidated.load_state_dict(updated_sd)
        consolidated.eval()

        if self.logger:
            self.logger.info("One-Shot Joint Consolidation complete.")

        return consolidated
