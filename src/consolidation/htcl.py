"""
Hierarchical Taylor-based Continual Learning (HTCL) Consolidation.

Implements the second-order Taylor expansion consolidation mechanism from:
"Mitigating Task-Order Sensitivity and Forgetting via Hierarchical Second-Order
Consolidation" (Nag, Raghavan, Narayanan, 2026).

Key properties:
  - Global model initialized from the first expert (not averaged)
  - Diagonal Fisher approximation of the Hessian on high-confidence states
  - Lambda selected via validation-based grid search (KL proxy)
  - Catch-up iterations with geometrically decaying step size
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import json
import os
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

from src.models.dqn import DQNNetwork
from src.data.replay_buffer import ReplayBuffer
from src.utils.logger import Logger


class HTCLConsolidator:
    """HTCL consolidation for merging expert models into a global model.

    Uses a second-order Taylor expansion around the current global parameters
    to find an optimal update direction that balances stability (preserving
    past knowledge, captured by the Fisher/Hessian) with plasticity (absorbing
    new expert knowledge).

    The closed-form update is::

        w_global^(t) = w_global^(t-1)
            + eta * (H + lambda * I)^{-1} [lambda * delta_d - g]

    where:
        delta_d = w_local^(t) - w_global^(t-1)  (expert drift)
        g = gradient of past loss at w_global     (gradient on old tasks)
        H = diagonal Fisher approximation          (Hessian on old tasks)

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
        self.htcl_cfg = config["htcl"]

        self.lambda_htcl = self.htcl_cfg["lambda_htcl"]
        self.lambda_auto = self.htcl_cfg.get("lambda_auto", False)
        self.lambda_margin = self.htcl_cfg.get("lambda_margin", 0.1)
        self.fisher_samples = self.htcl_cfg["fisher_samples"]
        self.catch_up_iterations = self.htcl_cfg["catch_up_iterations"]
        self.eta = self.htcl_cfg.get("eta", 0.9)
        self.eta_decay = self.htcl_cfg.get("eta_decay", 0.5)
        self.diagonal_fisher = self.htcl_cfg.get("diagonal_fisher", True)

        # Lambda grid search config
        self.lambda_grid_search = self.htcl_cfg.get("lambda_grid_search", False)
        self.lambda_candidates = self.htcl_cfg.get(
            "lambda_candidates", [0.1, 1.0, 10.0, 100.0, 1000.0]
        )

        # Storage for cumulative Fisher (diagonal) across all registered tasks
        self.cumulative_fisher: Optional[Dict[str, torch.Tensor]] = None
        # Cumulative gradient at global params
        self.cumulative_gradient: Optional[Dict[str, torch.Tensor]] = None
        # Number of tasks registered
        self.num_registered_tasks = 0
        # Fisher statistics log (saved alongside checkpoint for later plotting)
        self.fisher_log: List[Dict[str, Any]] = []
        # Lambda grid search results (for visualization)
        self.lambda_grid_results: List[Dict[str, Any]] = []

    # ── Core computation ─────────────────────────────────────────────────────

    def compute_diagonal_fisher(
        self,
        model: DQNNetwork,
        valid_actions: List[int],
        states: Optional[torch.Tensor] = None,
        replay_buffer: Optional[ReplayBuffer] = None,
        num_samples: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute diagonal Fisher Information Matrix.

        Uses the empirical Fisher: E[grad(log pi(a|s))^2] as a diagonal
        approximation of the Hessian.  Accepts either pre-filtered *states*
        or a *replay_buffer* from which states are sampled.

        Args:
            model: Network to compute Fisher for.
            valid_actions: Valid action indices.
            states: Pre-filtered state tensor (preferred).
            replay_buffer: Fallback replay data.
            num_samples: Samples to draw from replay_buffer (ignored when
                states is given).

        Returns:
            Dictionary of parameter name -> diagonal Fisher tensor.
        """
        if states is None:
            if replay_buffer is None:
                raise ValueError(
                    "Must provide either `states` or `replay_buffer`."
                )
            if num_samples is None:
                num_samples = self.fisher_samples
            states = replay_buffer.sample_states(num_samples)

        total = len(states)
        model.eval()
        fisher = {
            name: torch.zeros_like(param)
            for name, param in model.named_parameters()
            if param.requires_grad
        }

        batch_size = 64
        for i in range(0, total, batch_size):
            batch = states[i : i + batch_size]
            model.zero_grad()

            q_values = model(batch)
            mask = torch.full(
                (model.unified_action_dim,), float("-inf"), device=self.device
            )
            mask[valid_actions] = 0.0
            masked_q = q_values + mask.unsqueeze(0)

            # Sample actions from softmax policy (not argmax) for proper
            # empirical Fisher.
            probs = F.softmax(masked_q, dim=1)
            sampled_actions = torch.multinomial(probs, num_samples=1).squeeze(1)

            log_probs = F.log_softmax(masked_q, dim=1)
            selected_log_probs = log_probs.gather(
                1, sampled_actions.unsqueeze(1)
            ).squeeze(1)

            # Compute Fisher per sample to avoid gradient cancellation
            for j in range(len(batch)):
                model.zero_grad()
                selected_log_probs[j].backward(
                    retain_graph=(j < len(batch) - 1)
                )
                for name, param in model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        fisher[name] += (param.grad.data ** 2) / total

        model.train()
        return fisher

    def compute_gradient(
        self,
        model: DQNNetwork,
        valid_actions: List[int],
        states: Optional[torch.Tensor] = None,
        replay_buffer: Optional[ReplayBuffer] = None,
        num_samples: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute gradient of loss at current model parameters.

        This represents g^{(t-1)} = nabla J(w_global^{(t-1)}) evaluated
        on replay data from past tasks.

        Args:
            model: Global model.
            valid_actions: Valid actions.
            states: Pre-filtered state tensor (preferred).
            replay_buffer: Fallback replay data.
            num_samples: Samples to draw (ignored when states is given).

        Returns:
            Dictionary of parameter name -> gradient tensor.
        """
        if states is None:
            if replay_buffer is None:
                raise ValueError(
                    "Must provide either `states` or `replay_buffer`."
                )
            if num_samples is None:
                num_samples = self.fisher_samples
            states = replay_buffer.sample_states(num_samples)

        total = len(states)
        model.eval()
        gradient = {
            name: torch.zeros_like(param)
            for name, param in model.named_parameters()
            if param.requires_grad
        }

        batch_size = 64
        for i in range(0, total, batch_size):
            batch = states[i : i + batch_size]
            model.zero_grad()

            q_values = model(batch)
            mask = torch.full(
                (model.unified_action_dim,), float("-inf"), device=self.device
            )
            mask[valid_actions] = 0.0
            masked_q = q_values + mask.unsqueeze(0)

            # Negative log-likelihood loss
            log_probs = F.log_softmax(masked_q, dim=1)
            actions = masked_q.argmax(dim=1)
            loss = -log_probs.gather(1, actions.unsqueeze(1)).squeeze(1).mean()
            loss.backward()

            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    gradient[name] += param.grad.data * (len(batch) / total)

        model.train()
        return gradient

    # ── Lambda constraint ────────────────────────────────────────────────────

    def _ensure_lambda_constraint(
        self,
        fisher: Dict[str, torch.Tensor],
        lambda_val: Optional[float] = None,
    ) -> float:
        """Ensure lambda > -mu_min(H) for positive definiteness.

        Args:
            fisher: Diagonal Fisher dictionary.
            lambda_val: Specific lambda to validate (default: self.lambda_htcl).

        Returns:
            Valid lambda value satisfying the constraint.
        """
        if lambda_val is None:
            lambda_val = self.lambda_htcl

        mu_min = float("inf")
        for name, f_diag in fisher.items():
            local_min = f_diag.min().item()
            if local_min < mu_min:
                mu_min = local_min

        lambda_lower_bound = -mu_min + self.lambda_margin

        if self.lambda_auto:
            effective_lambda = max(lambda_val, lambda_lower_bound)
        else:
            effective_lambda = lambda_val
            if effective_lambda <= -mu_min:
                if self.logger:
                    self.logger.warning(
                        f"HTCL: lambda={effective_lambda:.4f} violates "
                        f"constraint lambda > -mu_min(H) = {-mu_min:.4f}."
                    )

        if self.logger:
            self.logger.info(
                f"HTCL: mu_min(H) = {mu_min:.6f} | "
                f"Constraint: lambda > {-mu_min:.6f} | "
                f"Using lambda = {effective_lambda:.6f}"
            )

        return effective_lambda

    # ── Fisher / Hessian diagnostic logging ──────────────────────────────────

    def _log_fisher_statistics(
        self,
        fisher: Dict[str, torch.Tensor],
        task_idx: int,
        game_name: str,
        prefix: str = "htcl",
        is_cumulative: bool = False,
    ) -> Dict[str, Any]:
        """Log detailed per-layer Fisher (Hessian diagonal) statistics.

        Args:
            fisher: Diagonal Fisher dict (param_name -> tensor).
            task_idx: 0-based task index.
            game_name: Name of the current game/task.
            prefix: TensorBoard tag prefix.
            is_cumulative: Whether this is the cumulative Fisher.

        Returns:
            Summary dict with per-layer and global statistics.
        """
        kind = "cumulative" if is_cumulative else "task"
        tag_prefix = f"{prefix}/fisher_{kind}"

        layer_stats: Dict[str, Dict[str, float]] = {}
        all_vals = []

        for name, f_diag in fisher.items():
            vals = f_diag.detach().cpu().float()
            all_vals.append(vals.flatten())

            stats = {
                "min": vals.min().item(),
                "max": vals.max().item(),
                "mean": vals.mean().item(),
                "std": vals.std().item(),
                "median": vals.median().item(),
                "nonzero_frac": (vals > 1e-10).float().mean().item(),
                "numel": vals.numel(),
            }
            layer_stats[name] = stats

            if self.logger:
                for stat_name, stat_val in stats.items():
                    if stat_name == "numel":
                        continue
                    self.logger.log_scalar(
                        f"{tag_prefix}/{name}/{stat_name}",
                        stat_val,
                        task_idx + 1,
                    )

        all_flat = torch.cat(all_vals)
        global_stats = {
            "min": all_flat.min().item(),
            "max": all_flat.max().item(),
            "mean": all_flat.mean().item(),
            "std": all_flat.std().item(),
            "median": all_flat.median().item(),
            "nonzero_frac": (all_flat > 1e-10).float().mean().item(),
            "total_params": all_flat.numel(),
        }

        if self.logger:
            for stat_name, stat_val in global_stats.items():
                if stat_name == "total_params":
                    continue
                self.logger.log_scalar(
                    f"{tag_prefix}/global/{stat_name}",
                    stat_val,
                    task_idx + 1,
                )
            self.logger.info(
                f"HTCL: Fisher ({kind}) after {game_name} | "
                f"global min={global_stats['min']:.6f}, "
                f"max={global_stats['max']:.4f}, "
                f"mean={global_stats['mean']:.6f}, "
                f"std={global_stats['std']:.6f}, "
                f"nonzero={global_stats['nonzero_frac']*100:.1f}%"
            )

        top_layers = sorted(
            layer_stats.items(), key=lambda kv: kv[1]["mean"], reverse=True
        )[:5]
        if self.logger:
            for rank, (lname, lstats) in enumerate(top_layers, 1):
                self.logger.info(
                    f"  Top-{rank} Fisher layer: {lname} | "
                    f"mean={lstats['mean']:.6f}, max={lstats['max']:.4f}"
                )

        summary = {
            "task_idx": task_idx,
            "game_name": game_name,
            "kind": kind,
            "global": global_stats,
            "per_layer": layer_stats,
        }
        self.fisher_log.append(summary)
        return summary

    def save_fisher_log(self, path: str) -> None:
        """Save accumulated Fisher statistics to JSON for offline plotting."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.fisher_log, f, indent=2)
        if self.logger:
            self.logger.info(f"HTCL: Fisher log saved to {path}")

    def save_lambda_grid_log(self, path: str) -> None:
        """Save lambda grid search results to JSON for visualization."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.lambda_grid_results, f, indent=2)
        if self.logger:
            self.logger.info(f"HTCL: Lambda grid log saved to {path}")

    # ── Taylor update ────────────────────────────────────────────────────────

    def _taylor_update(
        self,
        global_state_dict: Dict[str, torch.Tensor],
        local_state_dict: Dict[str, torch.Tensor],
        fisher: Dict[str, torch.Tensor],
        gradient: Dict[str, torch.Tensor],
        lambda_val: float,
        eta_override: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute the HTCL closed-form parameter update.

        w_new = w_global + eta * (H + lambda*I)^{-1} * [lambda * delta_d - g]

        Args:
            global_state_dict: Current global model parameters.
            local_state_dict: Expert (local) model parameters.
            fisher: Diagonal Fisher / Hessian approximation.
            gradient: Gradient of past loss at global params.
            lambda_val: Validated lambda value.
            eta_override: If given, overrides self.eta (used for catch-up).

        Returns:
            Updated state dict.
        """
        eta = eta_override if eta_override is not None else self.eta
        updated = {}
        for name in global_state_dict:
            if name in fisher:
                w_global = global_state_dict[name].to(self.device)
                w_local = local_state_dict[name].to(self.device)
                h_diag = fisher[name].to(self.device)
                g = gradient[name].to(self.device)

                delta_d = w_local - w_global
                numerator = lambda_val * delta_d - g
                denominator = h_diag + lambda_val
                update = numerator / (denominator + 1e-8)

                updated[name] = w_global + eta * update
            else:
                # Non-trainable params (e.g., BN stats): average
                updated[name] = (
                    global_state_dict[name].to(self.device)
                    + local_state_dict[name].to(self.device)
                ) / 2.0

        return updated

    # ── KL divergence proxy for lambda selection ─────────────────────────────

    def _compute_kl_to_experts(
        self,
        model: DQNNetwork,
        expert_models: List[DQNNetwork],
        expert_results: List[Dict[str, Any]],
        filtered_states_list: List[torch.Tensor],
    ) -> Dict[str, float]:
        """Compute mean KL(expert || consolidated) per task.

        Low KL means the consolidated model closely matches the expert's
        policy on high-confidence states.

        Args:
            model: Consolidated model to evaluate.
            expert_models: List of expert DQN models (frozen).
            expert_results: Expert result dicts (for valid_actions / game_name).
            filtered_states_list: High-confidence states per task.

        Returns:
            Dict mapping game_name -> mean KL divergence.
        """
        model.eval()
        kl_per_task = {}

        for expert_model, result, states in zip(
            expert_models, expert_results, filtered_states_list,
        ):
            valid_actions = result["valid_actions"]
            game_name = result["game_name"]

            mask = torch.full(
                (model.unified_action_dim,), float("-inf"), device=self.device,
            )
            mask[valid_actions] = 0.0

            batch_size = 256
            total_kl = 0.0
            n_batches = 0

            with torch.no_grad():
                for start in range(0, len(states), batch_size):
                    batch = states[start : start + batch_size]

                    # Expert policy
                    expert_q = expert_model(batch)
                    expert_masked = expert_q + mask.unsqueeze(0)
                    expert_probs = F.softmax(expert_masked, dim=1)

                    # Consolidated model policy
                    consol_q = model(batch)
                    consol_masked = consol_q + mask.unsqueeze(0)
                    consol_log_probs = F.log_softmax(consol_masked, dim=1)

                    # KL(expert || consolidated)
                    kl = F.kl_div(
                        consol_log_probs, expert_probs,
                        reduction="batchmean",
                    )
                    total_kl += kl.item()
                    n_batches += 1

            kl_per_task[game_name] = total_kl / max(n_batches, 1)

        return kl_per_task

    # ── Initial task registration ────────────────────────────────────────────

    def register_initial_task(
        self,
        model: DQNNetwork,
        valid_actions: List[int],
        filtered_states: torch.Tensor,
        game_name: str,
    ) -> None:
        """Register the first expert as the global model baseline.

        Computes and stores Fisher and gradient for the initial task so that
        subsequent consolidation steps properly protect task-1 knowledge.

        Args:
            model: First expert model (= initial global model).
            valid_actions: Valid actions for the first task.
            filtered_states: High-confidence states from task 1.
            game_name: Name of the first game.
        """
        if self.logger:
            self.logger.info(
                f"HTCL: Registering initial task ({game_name}) — "
                f"computing Fisher & gradient baseline..."
            )

        self.cumulative_fisher = self.compute_diagonal_fisher(
            model, valid_actions, states=filtered_states,
        )
        self.cumulative_gradient = self.compute_gradient(
            model, valid_actions, states=filtered_states,
        )
        self.num_registered_tasks = 1

        # Log Fisher statistics for task 0
        self._log_fisher_statistics(
            self.cumulative_fisher, 0, game_name,
            prefix="htcl", is_cumulative=False,
        )
        self._log_fisher_statistics(
            self.cumulative_fisher, 0, game_name,
            prefix="htcl", is_cumulative=True,
        )

        if self.logger:
            self.logger.info(
                f"HTCL: Initial task ({game_name}) registered. "
                f"cumulative_fisher has "
                f"{len(self.cumulative_fisher)} parameter groups."
            )

    # ── Lambda grid search ───────────────────────────────────────────────────

    def grid_search_lambda(
        self,
        global_sd: Dict[str, torch.Tensor],
        expert_results: List[Dict[str, Any]],
        expert_models: List[DQNNetwork],
        filtered_states_list: List[torch.Tensor],
        model_template: DQNNetwork,
    ) -> float:
        """Find the best lambda via grid search using KL-divergence proxy.

        For each candidate lambda, runs the full consolidation loop (without
        catch-up to keep it cheap) and evaluates how well the resulting model
        preserves each expert's policy on high-confidence states.

        Args:
            global_sd: Current global state dict (expert-1 weights).
            expert_results: All expert results (including task 1 for KL eval).
            expert_models: Frozen expert models for KL computation.
            filtered_states_list: High-confidence states per task.
            model_template: Template DQN for weight loading.

        Returns:
            Best lambda value.
        """
        candidates = self.lambda_candidates
        if self.logger:
            self.logger.info(
                f"HTCL: Lambda grid search over {candidates}..."
            )

        # Tasks to consolidate (skip task 1 — it's the initial global)
        consolidation_results = expert_results[1:]
        consolidation_states = filtered_states_list[1:]

        best_lambda = candidates[0]
        best_avg_kl = float("inf")

        for lam in candidates:
            # Run lightweight consolidation with this lambda (no catch-up)
            trial_sd = {k: v.clone() for k, v in global_sd.items()}
            trial_fisher = {
                k: v.clone() for k, v in self.cumulative_fisher.items()
            }
            trial_gradient = {
                k: v.clone() for k, v in self.cumulative_gradient.items()
            }

            trial_model = copy.deepcopy(model_template).to(self.device)

            for task_idx, (result, filt_states) in enumerate(
                zip(consolidation_results, consolidation_states)
            ):
                local_sd = result["policy_state_dict"]
                valid_actions = result["valid_actions"]

                # Compute task Fisher and gradient at trial position
                trial_model.load_state_dict(trial_sd)
                task_fisher = self.compute_diagonal_fisher(
                    trial_model, valid_actions, states=filt_states,
                )
                task_gradient = self.compute_gradient(
                    trial_model, valid_actions, states=filt_states,
                )

                # Accumulate
                for name in task_fisher:
                    trial_fisher[name] = (
                        trial_fisher[name] + task_fisher[name]
                    )
                    trial_gradient[name] = (
                        trial_gradient[name] + task_gradient[name]
                    )

                # Validate lambda
                eff_lam = self._ensure_lambda_constraint(trial_fisher, lam)

                # Single Taylor update (no catch-up for speed)
                trial_sd = self._taylor_update(
                    trial_sd, local_sd,
                    trial_fisher, trial_gradient, eff_lam,
                )

            # Evaluate KL divergence to ALL experts
            trial_model.load_state_dict(trial_sd)
            kl_scores = self._compute_kl_to_experts(
                trial_model, expert_models,
                expert_results, filtered_states_list,
            )
            avg_kl = float(np.mean(list(kl_scores.values())))

            grid_entry = {
                "lambda": lam,
                "kl_per_task": kl_scores,
                "avg_kl": avg_kl,
            }
            self.lambda_grid_results.append(grid_entry)

            if self.logger:
                kl_str = ", ".join(
                    f"{g}={v:.4f}" for g, v in kl_scores.items()
                )
                self.logger.info(
                    f"  lambda={lam:>10.1f} | "
                    f"avg_kl={avg_kl:.4f} | {kl_str}"
                )

            if avg_kl < best_avg_kl:
                best_avg_kl = avg_kl
                best_lambda = lam

        if self.logger:
            self.logger.info(
                f"HTCL: Best lambda = {best_lambda} "
                f"(avg KL = {best_avg_kl:.4f})"
            )

        return best_lambda

    # ── Main consolidation ───────────────────────────────────────────────────

    def consolidate(
        self,
        global_model: DQNNetwork,
        expert_results: List[Dict[str, Any]],
        filtered_states_list: Optional[List[torch.Tensor]] = None,
        expert_models: Optional[List[DQNNetwork]] = None,
    ) -> DQNNetwork:
        """Consolidate expert models into global model via HTCL.

        Expects that ``register_initial_task()`` has already been called
        so that task-1 Fisher/gradient are available.  Consolidates
        remaining tasks (expert_results[1:]) sequentially.

        If ``lambda_grid_search`` is enabled, runs the grid search first
        to pick the best lambda.

        Args:
            global_model: Current global model (initialized to expert 1).
            expert_results: List of ALL expert results (task 1 included for
                KL evaluation during grid search).
            filtered_states_list: Pre-filtered high-confidence states per
                expert (one tensor per expert, same order as expert_results).
            expert_models: Frozen expert DQN models (needed for KL-based
                lambda grid search).

        Returns:
            The consolidated global model.
        """
        if self.logger:
            self.logger.info("Starting HTCL consolidation...")

        consolidated = copy.deepcopy(global_model).to(self.device)
        global_sd = {
            name: param.clone()
            for name, param in consolidated.state_dict().items()
        }

        # ── Lambda grid search (if enabled) ──
        effective_lambda = self.lambda_htcl
        if self.lambda_grid_search and expert_models is not None:
            effective_lambda = self.grid_search_lambda(
                global_sd, expert_results, expert_models,
                filtered_states_list or [],
                consolidated,
            )
            # Reset cumulative state after grid search trials
            first_result = expert_results[0]
            first_states = (
                filtered_states_list[0] if filtered_states_list else None
            )
            consolidated.load_state_dict(global_sd)
            self.cumulative_fisher = self.compute_diagonal_fisher(
                consolidated,
                first_result["valid_actions"],
                states=first_states,
            )
            self.cumulative_gradient = self.compute_gradient(
                consolidated,
                first_result["valid_actions"],
                states=first_states,
            )
            self.num_registered_tasks = 1

        if self.logger:
            self.logger.info(
                f"HTCL: Using lambda = {effective_lambda:.4f} "
                f"for consolidation."
            )

        # ── Consolidate remaining tasks (skip task 1) ──
        tasks_to_consolidate = expert_results[1:]
        states_to_use = (
            filtered_states_list[1:]
            if filtered_states_list
            else [None] * len(tasks_to_consolidate)
        )

        for rel_idx, (result, filt_states) in enumerate(
            zip(tasks_to_consolidate, states_to_use)
        ):
            # Global task index (0-based, task 1 was index 0)
            task_idx = rel_idx + 1

            game_name = result["game_name"]
            local_sd = result["policy_state_dict"]
            valid_actions = result["valid_actions"]

            if self.logger:
                self.logger.info(
                    f"HTCL: Consolidating task {task_idx + 1}/"
                    f"{len(expert_results)} ({game_name})..."
                )

            # Compute Fisher at current global params on this task
            consolidated.load_state_dict(global_sd)
            task_fisher = self.compute_diagonal_fisher(
                consolidated, valid_actions, states=filt_states,
            )

            # Accumulate Fisher
            for name in task_fisher:
                self.cumulative_fisher[name] = (
                    self.cumulative_fisher[name] + task_fisher[name]
                )

            # Log Fisher diagnostics
            self._log_fisher_statistics(
                task_fisher, task_idx, game_name,
                prefix="htcl", is_cumulative=False,
            )
            self._log_fisher_statistics(
                self.cumulative_fisher, task_idx, game_name,
                prefix="htcl", is_cumulative=True,
            )

            # Compute gradient at global params on this task
            task_gradient = self.compute_gradient(
                consolidated, valid_actions, states=filt_states,
            )

            # Accumulate gradient
            for name in task_gradient:
                self.cumulative_gradient[name] = (
                    self.cumulative_gradient[name] + task_gradient[name]
                )

            # Validate lambda
            eff_lam = self._ensure_lambda_constraint(
                self.cumulative_fisher, effective_lambda,
            )

            if self.logger:
                self.logger.info(
                    f"HTCL: Task {task_idx + 1} ({game_name}) | "
                    f"effective_lambda = {eff_lam:.4f}"
                )
                self.logger.log_scalar(
                    "htcl/effective_lambda", eff_lam, task_idx + 1,
                )
                self.logger.log_scalar(
                    "htcl/num_tasks_consolidated",
                    task_idx + 1, task_idx + 1,
                )

            # ── Main Taylor update ──
            updated_sd = self._taylor_update(
                global_sd, local_sd,
                self.cumulative_fisher, self.cumulative_gradient,
                eff_lam,
            )

            # Log drift and update norms
            if self.logger:
                drift_norm = sum(
                    (local_sd[n].to(self.device)
                     - global_sd[n].to(self.device))
                    .norm().item() ** 2
                    for n in self.cumulative_fisher
                ) ** 0.5
                update_norm = sum(
                    (updated_sd[n]
                     - global_sd[n].to(self.device))
                    .norm().item() ** 2
                    for n in self.cumulative_fisher
                ) ** 0.5
                self.logger.info(
                    f"HTCL: {game_name} | "
                    f"expert_drift_norm = {drift_norm:.4f} | "
                    f"update_norm = {update_norm:.4f}"
                )
                self.logger.log_scalar(
                    f"htcl/{game_name}/expert_drift_norm",
                    drift_norm, task_idx + 1,
                )
                self.logger.log_scalar(
                    f"htcl/{game_name}/update_norm",
                    update_norm, task_idx + 1,
                )

            global_sd = updated_sd

            # ── Catch-up iterations with decaying eta ──
            eta_catch = self.eta
            for catch_iter in range(self.catch_up_iterations):
                eta_catch *= self.eta_decay  # Decay before use

                consolidated.load_state_dict(global_sd)
                refined_gradient = self.compute_gradient(
                    consolidated, valid_actions, states=filt_states,
                )

                global_sd = self._taylor_update(
                    global_sd, local_sd,
                    self.cumulative_fisher, refined_gradient,
                    eff_lam, eta_override=eta_catch,
                )

                if self.logger:
                    self.logger.info(
                        f"HTCL: Catch-up {catch_iter + 1}/"
                        f"{self.catch_up_iterations} for {game_name}"
                        f" | eta={eta_catch:.4f}"
                    )

            self.num_registered_tasks += 1

        # Load final consolidated weights
        consolidated.load_state_dict(global_sd)
        consolidated.eval()

        if self.logger:
            self.logger.info("HTCL consolidation complete.")
        return consolidated

    # ── Utility methods ──────────────────────────────────────────────────────

    def get_global_weights(self, model: DQNNetwork) -> dict:
        """Get a copy of the global model weights."""
        return copy.deepcopy(model.state_dict())

    def save(self, path: str) -> None:
        """Save HTCL state."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "cumulative_fisher": self.cumulative_fisher,
                "cumulative_gradient": self.cumulative_gradient,
                "num_registered_tasks": self.num_registered_tasks,
                "lambda_htcl": self.lambda_htcl,
            },
            path,
        )

    def load(self, path: str) -> None:
        """Load HTCL state."""
        data = torch.load(
            path, map_location=self.device, weights_only=False,
        )
        self.cumulative_fisher = data["cumulative_fisher"]
        self.cumulative_gradient = data["cumulative_gradient"]
        self.num_registered_tasks = data["num_registered_tasks"]
        self.lambda_htcl = data["lambda_htcl"]
