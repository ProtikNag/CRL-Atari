"""
Weighted Hessian Consolidation (WHC).

Implements the closed-form Hessian-weighted consolidation from:
"Reconstructing the Loss Landscape of a Generalized Continual Learner
from Expert Models via Weighted Hessian Consolidation" (Nag, 2026).

Core idea:
    Approximate each expert loss L_i(w) via second-order Taylor expansion
    at its own optimum w_i^*, forming a surrogate:

        L_tilde(w) = C + (1/2) w^T H_agg w  -  w^T b_agg

    where:
        H_agg = sum_i alpha_i * H_i          (aggregated Hessian)
        b_agg = sum_i alpha_i * H_i * w_i^*  (Hessian-weighted centroid)

    The closed-form minimiser (Eq. 11) is:

        w_hat = H_agg^{-1} b_agg

    With Tikhonov regularization for numerical stability (Eq. 13):

        w_hat_lambda = (H_agg + lambda I)^{-1} b_agg

    Diagonal Fisher Information Matrix at each expert's own optimum
    is used as the Hessian approximation.  This placement (at the expert
    optimum rather than a shared anchor) is critical for minimizing
    |L(w) - L_tilde(w)|, since the Taylor remainder is smallest near
    the expansion point.

Key differences from One-Shot / HTCL:
    - Fisher computed at each expert's OWN optimum w_i^* (not a shared anchor)
    - No gradient term needed (gradient vanishes at expert optima: nabla L_i(w_i^*)=0)
    - Closed-form weighted combination (no iterative updates, no step size)
    - Action masking on head layers to prevent Q-corruption from invalid actions
"""

import copy
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from src.models.dqn import DQNNetwork
from src.utils.logger import Logger


class WHCConsolidator:
    """Weighted Hessian Consolidation (Theorem 5.2 / Eq. 11-13).

    Computes per-expert diagonal Fisher at each expert's own optimum,
    then produces the closed-form Hessian-weighted consolidation.

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

        whc_cfg = config.get("whc", {})
        self.lambda_reg: float = whc_cfg.get("lambda_reg", 1.0)
        self.fisher_samples: int = whc_cfg.get("fisher_samples", 20000)
        self.alpha_weights: Optional[List[float]] = whc_cfg.get(
            "alpha_weights", None,
        )

    def _log(self, msg: str) -> None:
        if self.logger:
            self.logger.info(msg)

    def _compute_diagonal_fisher(
        self,
        model: DQNNetwork,
        valid_actions: List[int],
        states: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute diagonal Fisher Information Matrix at the model's current parameters.

        Uses the empirical Fisher: E[grad(log pi(a|s))^2].

        Args:
            model: Network positioned at expert optimum w_i^*.
            valid_actions: Valid action indices for this task.
            states: High-confidence states for Fisher estimation.

        Returns:
            Dictionary of parameter name -> diagonal Fisher tensor.
        """
        total = len(states)
        model.eval()
        fisher: Dict[str, torch.Tensor] = {
            name: torch.zeros_like(param)
            for name, param in model.named_parameters()
            if param.requires_grad
        }

        batch_size = 64
        for i in range(0, total, batch_size):
            batch = states[i : i + batch_size].to(self.device)
            model.zero_grad()

            q_values = model(batch)
            mask = torch.full(
                (model.unified_action_dim,), float("-inf"), device=self.device,
            )
            mask[valid_actions] = 0.0
            masked_q = q_values + mask.unsqueeze(0)

            probs = F.softmax(masked_q, dim=1)
            sampled_actions = torch.multinomial(probs, num_samples=1).squeeze(1)

            log_probs = F.log_softmax(masked_q, dim=1)
            selected_log_probs = log_probs.gather(
                1, sampled_actions.unsqueeze(1),
            ).squeeze(1)

            for j in range(len(batch)):
                model.zero_grad()
                selected_log_probs[j].backward(
                    retain_graph=(j < len(batch) - 1),
                )
                for name, param in model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        fisher[name] += (param.grad.data ** 2) / total

        model.train()
        return fisher

    def consolidate(
        self,
        global_model: DQNNetwork,
        expert_results: List[Dict[str, Any]],
        filtered_states_list: List[torch.Tensor],
        expert_models: Optional[List[DQNNetwork]] = None,
        lambda_override: Optional[float] = None,
    ) -> DQNNetwork:
        """Run Weighted Hessian Consolidation (Eq. 11/13).

        Computes Fisher at each expert's own optimum, then solves:
            w_hat = (H_agg + lambda I)^{-1} b_agg

        Args:
            global_model: Model template (used only for architecture/device).
            expert_results: List of expert result dicts containing
                policy_state_dict, valid_actions, game_name.
            filtered_states_list: High-confidence states per expert.
            expert_models: Frozen expert models for Fisher computation.
            lambda_override: Override regularization strength.

        Returns:
            Consolidated DQNNetwork.
        """
        lam = lambda_override if lambda_override is not None else self.lambda_reg
        num_tasks = len(expert_results)

        # Task weights alpha_i (uniform by default, must sum to 1)
        if self.alpha_weights is not None and len(self.alpha_weights) == num_tasks:
            alphas = self.alpha_weights
        else:
            alphas = [1.0 / num_tasks] * num_tasks

        self._log(
            f"Starting Weighted Hessian Consolidation "
            f"(lambda={lam:.4f}, alphas={alphas}, N={num_tasks})..."
        )

        # Identify head layer names for action masking
        ref_sd = global_model.state_dict()
        n_actions = global_model.unified_action_dim
        head_weight_names = {
            n for n in ref_sd
            if n.endswith(".weight")
            and ref_sd[n].shape[0] == n_actions
            and ("fc." in n or "advantage_stream." in n)
        }
        head_bias_names = {
            n for n in ref_sd
            if n.endswith(".bias")
            and ref_sd[n].shape[0] == n_actions
            and ("fc." in n or "advantage_stream." in n)
        }
        head_names = head_weight_names | head_bias_names

        # Initialize accumulators: H_agg and b_agg  (Eq. 7, 8)
        trainable_names = [
            name for name, param in global_model.named_parameters()
            if param.requires_grad
        ]
        h_agg: Dict[str, torch.Tensor] = {
            name: torch.zeros_like(ref_sd[name], device=self.device)
            for name in trainable_names
        }
        b_agg: Dict[str, torch.Tensor] = {
            name: torch.zeros_like(ref_sd[name], device=self.device)
            for name in trainable_names
        }

        # Compute per-expert Fisher at expert's OWN optimum and accumulate
        for task_idx, (result, filt_states) in enumerate(
            zip(expert_results, filtered_states_list)
        ):
            game_name = result["game_name"]
            valid_actions = result["valid_actions"]
            alpha_i = alphas[task_idx]

            self._log(
                f"  [{task_idx+1}/{num_tasks}] Computing Fisher for {game_name} "
                f"at expert optimum (alpha={alpha_i:.3f})..."
            )

            # Build a fresh model at expert's own optimum w_i^*
            expert_model = copy.deepcopy(global_model).to(self.device)
            expert_model.load_state_dict(result["policy_state_dict"])

            # Compute Fisher H_i at w_i^* (the expert's own optimum)
            fisher_i = self._compute_diagonal_fisher(
                expert_model, valid_actions, filt_states,
            )

            # Fisher statistics for logging
            fisher_norms = {
                n: fisher_i[n].norm().item()
                for n in list(fisher_i.keys())[:3]
            }
            self._log(f"    Fisher norms (sample): {fisher_norms}")

            # Expert parameters w_i^*
            expert_sd = result["policy_state_dict"]

            # Determine unused actions for masking
            unused = (
                sorted(set(range(n_actions)) - set(valid_actions))
                if len(valid_actions) < n_actions
                else []
            )

            # Accumulate H_agg and b_agg
            for name in trainable_names:
                h_i = fisher_i[name]
                w_i = expert_sd[name].to(self.device).float()

                # Action masking: zero out Fisher and weight contribution
                # for actions this expert was never trained on
                if unused and name in head_names:
                    if name in head_weight_names:
                        h_i[unused, :] = 0.0
                        w_i_masked = w_i.clone()
                        w_i_masked[unused, :] = 0.0
                    else:  # bias
                        h_i[unused] = 0.0
                        w_i_masked = w_i.clone()
                        w_i_masked[unused] = 0.0
                else:
                    w_i_masked = w_i

                h_agg[name] += alpha_i * h_i
                b_agg[name] += alpha_i * h_i * w_i_masked

            del expert_model

        # Solve: w_hat = (H_agg + lambda I)^{-1} (b_agg + lambda * w_bar)
        # The lambda * w_bar term regularizes toward the ensemble mean
        # rather than toward zero, ensuring the solution remains in a
        # sensible region of parameter space when lambda is large.
        self._log(f"  Solving closed-form consolidation (lambda={lam})...")

        # Compute ensemble mean for regularization anchor
        ensemble_mean = {
            name: torch.stack(
                [r["policy_state_dict"][name].float().to(self.device)
                 for r in expert_results]
            ).mean(dim=0)
            for name in trainable_names
        }

        consolidated_sd = {}
        for name in ref_sd:
            if name in h_agg:
                denominator = h_agg[name] + lam
                numerator = b_agg[name] + lam * ensemble_mean[name]
                consolidated_sd[name] = numerator / (denominator + 1e-12)
            else:
                # Non-trainable parameters: average across experts
                consolidated_sd[name] = torch.stack(
                    [r["policy_state_dict"][name].float().to(self.device)
                     for r in expert_results]
                ).mean(dim=0)

        # Log consolidation statistics
        if self.logger:
            dist_from_mean = sum(
                (consolidated_sd[n] - ensemble_mean[n]).norm().item() ** 2
                for n in trainable_names
            ) ** 0.5
            self._log(f"  Distance from ensemble mean: {dist_from_mean:.4f}")

            # Per-expert distance
            for task_idx, result in enumerate(expert_results):
                dist = sum(
                    (consolidated_sd[n] - result["policy_state_dict"][n].float().to(self.device)).norm().item() ** 2
                    for n in trainable_names
                ) ** 0.5
                self._log(
                    f"  Distance from {result['game_name']} expert: {dist:.4f}"
                )

        consolidated = copy.deepcopy(global_model).to(self.device)
        consolidated.load_state_dict(consolidated_sd)
        consolidated.eval()

        self._log("Weighted Hessian Consolidation complete.")
        return consolidated
