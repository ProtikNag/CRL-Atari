"""
Multi-Round Iterative Consolidation  (CRL.pdf, Section 4, Algorithm 1).

Each round re-expands around the current global model, computes averaged
Fisher and gradient over ALL experts jointly, then applies a single
combined Taylor correction.  Repeating for K rounds progressively reduces
the effective drift (Theorem 4.2).

Algorithm 1 (Paper):
    1.  w_g^(0) <- (1/N) Sigma_i w_e^(i)              [ensemble-mean anchor]
    2.  for k = 0, ..., K-1 do
    3.      w_bar <- w_g^(k)                            [re-expand]
    4.      for i = 1, ..., N do
    5.          g_i <- grad L_i(w_bar)
    6.          H_i <- Hessian L_i(w_bar)  (diagonal Fisher)
    7.          Delta_d^(i) <- w_e^(i) - w_bar          [action-masked]
    8.      end for
    9.      g_bar <- (1/N) Sigma_i g_i;  H_bar <- (1/N) Sigma_i H_i
   10.      d* <- weighted mean of Delta_d^(i)
   11.      u* <- (H_bar + lambda_bar I)^{-1} [lambda_bar d* - g_bar]
   12.      w_g^(k+1) <- w_bar + eta_k u*
   13.  end for
   14.  return w_g^(K)

Step-size schedule: eta_k = eta_0 * gamma^k  (Algorithm 2, line 13).
"""

import copy
from typing import Any, Dict, List, Optional

import torch

from src.consolidation.htcl import HTCLConsolidator
from src.models.dqn import DQNNetwork
from src.utils.logger import Logger


class IterativeConsolidator:
    """Multi-Round Joint Iterative Consolidation (Algorithm 1).

    In each round, Fisher and gradient are computed for every expert at
    the current global model position, averaged, and a single joint
    Taylor update is applied.  This is fundamentally different from
    sequential HTCL, where each expert is integrated one-at-a-time.

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

        iter_cfg = config.get("iterative", config.get("htcl", {}))
        self.lambda_val: float = iter_cfg.get("lambda_htcl", 100.0)
        self.eta_0: float = iter_cfg.get("eta", 0.9)
        self.gamma: float = iter_cfg.get("gamma", 0.5)
        self.num_rounds: int = iter_cfg.get("num_rounds", iter_cfg.get("num_passes", 3))
        self.fisher_samples: int = iter_cfg.get("fisher_samples", 5000)
        self.recompute_fisher: bool = iter_cfg.get("recompute_fisher", False)

        # Internal HTCL helper for Fisher/gradient computation
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
        """Compute averaged Fisher and gradient at current model position.

        Called once (with caching) or per-round depending on
        ``recompute_fisher``.
        """
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
        """Run multi-round joint iterative consolidation (Algorithm 1).

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
        K = self.num_rounds

        if self.logger:
            self.logger.info(
                f"Starting Multi-Round Joint Consolidation "
                f"(lambda={lam}, eta_0={self.eta_0}, gamma={self.gamma}, "
                f"K={K}, N={num_tasks}, "
                f"recompute_fisher={self.recompute_fisher})..."
            )

        consolidated = copy.deepcopy(global_model).to(self.device)
        global_sd = {
            name: param.clone()
            for name, param in consolidated.state_dict().items()
        }

        # Pre-compute head-layer names for action masking (Remark 7.1)
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

        # ── Pre-compute Fisher when recompute_fisher=False ──
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

        # ── Multi-round loop (Algorithm 1, lines 2-13) ──
        for round_k in range(K):
            eta_k = self.eta_0 * (self.gamma ** round_k)

            if self.logger:
                self.logger.info(
                    f"\n--- Round {round_k + 1}/{K} "
                    f"(eta_k={eta_k:.6f}) ---"
                )

            # Steps 3-9: Fisher/gradient (re-computed or cached)
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

            if self.logger:
                self.logger.info(
                    f"  Round {round_k + 1} | "
                    f"effective lambda = {eff_lam:.4f}"
                )

            # Steps 10-11: Compute per-expert corrections and average
            # (equivalent to using d* when lambda is uniform)
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

                    # Action mask (Remark 7.1)
                    if unused and name in head_names:
                        if name in head_weight_names:
                            delta_d[unused, :] = 0.0
                        else:
                            delta_d[unused] = 0.0

                    numerator = eff_lam * delta_d - g
                    denominator = h_diag + eff_lam
                    correction = numerator / (denominator + 1e-8)
                    avg_correction[name] += correction / num_tasks

            # Step 12: w_g^(k+1) = w_bar + eta_k * u*
            updated_sd = {}
            for name in global_sd:
                if name in avg_correction:
                    updated_sd[name] = (
                        global_sd[name].to(self.device)
                        + eta_k * avg_correction[name]
                    )
                else:
                    updated_sd[name] = global_sd[name]

            # Log round diagnostics
            if self.logger:
                update_norm = sum(
                    (updated_sd[n] - global_sd[n].to(self.device))
                    .norm().item() ** 2
                    for n in avg_fisher
                ) ** 0.5
                drift = max(
                    sum(
                        (result["policy_state_dict"][n].to(self.device)
                         - global_sd[n].to(self.device))
                        .norm().item() ** 2
                        for n in avg_fisher
                    ) ** 0.5
                    for result in expert_results
                )
                self.logger.info(
                    f"  Round {round_k + 1} | update_norm={update_norm:.4f} "
                    f"| max_drift={drift:.4f}"
                )
                self.logger.log_scalar(
                    "iterative/update_norm", update_norm, round_k,
                )
                self.logger.log_scalar(
                    "iterative/max_drift", drift, round_k,
                )

            global_sd = updated_sd

        # Load final weights
        consolidated.load_state_dict(global_sd)
        consolidated.eval()

        if self.logger:
            self.logger.info("Multi-Round Joint Consolidation complete.")

        return consolidated
