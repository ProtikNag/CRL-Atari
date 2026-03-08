"""
Multi-Round Iterative Consolidation.

Experts are absorbed one at a time.  After each task, the cumulative Fisher
grows, protecting previously integrated knowledge from being overwritten by
later tasks.  The full sequence is repeated for *num_passes* rounds with
geometrically decaying step size so that corrections shrink and converge.
A final joint-refinement step re-computes Fisher/gradient over all tasks
simultaneously and applies a small correction.

Algorithm (single pass):
    1.  Initialise w_global from first expert; compute Fisher_1, g_1.
    2.  For k = 2 … K:
            a.  Compute Fisher_k, g_k at w_global.
            b.  H ← H + Fisher_k ;  g ← g + g_k.
            c.  δ_k = w_expert_k − w_global  (action-masked for heads).
            d.  w_global ← w_global + η (H + λI)^{−1} (λ δ_k − g).
    3.  (Repeat for additional passes with η ← η × decay.)
    4.  Joint refinement: fresh Fisher/gradient over all tasks, small step.
"""

import copy
import json
import os
from typing import Any, Dict, List, Optional

import torch

from src.consolidation.htcl import HTCLConsolidator
from src.models.dqn import DQNNetwork
from src.utils.logger import Logger


class IterativeConsolidator:
    """Multi-Round Iterative Consolidation via sequential Taylor updates.

    Wraps the HTCLConsolidator's multi-pass consolidation loop with a
    fixed lambda (no grid search).  Provides a clean interface matching
    the other consolidation methods.

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
        self.lambda_val = iter_cfg.get("lambda_htcl", 100.0)
        self.eta = iter_cfg.get("eta", 0.9)
        self.num_passes = iter_cfg.get("num_passes", 3)
        self.joint_refinement = iter_cfg.get("joint_refinement", True)
        self.fisher_samples = iter_cfg.get("fisher_samples", 5000)

        # Internal HTCL instance provides the full iterative loop
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
        """Run multi-round iterative consolidation.

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
                f"Starting Multi-Round Iterative Consolidation "
                f"(λ={lam}, η={self.eta}, passes={self.num_passes}, "
                f"joint_refine={self.joint_refinement})..."
            )

        # Fresh consolidator for this run (so cumulative Fisher is clean)
        consolidator = HTCLConsolidator(
            self.config, device=self.device, logger=self.logger,
        )

        # Register first task (initialise cumulative Fisher/gradient)
        model = copy.deepcopy(global_model).to(self.device)

        consolidator.register_initial_task(
            model,
            expert_results[0]["valid_actions"],
            filtered_states_list[0],
            expert_results[0]["game_name"],
        )

        # Run multi-pass sequential consolidation with joint refinement
        consolidated = consolidator.consolidate(
            model,
            expert_results,
            filtered_states_list=filtered_states_list,
            expert_models=expert_models,
            lambda_override=lam,
            num_passes=self.num_passes,
            joint_refinement=self.joint_refinement,
        )

        if self.logger:
            self.logger.info("Multi-Round Iterative Consolidation complete.")

        # Keep fisher log reference
        self._fisher_log = consolidator.fisher_log

        return consolidated

    def save_fisher_log(self, path: str) -> None:
        """Save Fisher statistics log."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(getattr(self, "_fisher_log", []), f, indent=2)
        if self.logger:
            self.logger.info(f"Iterative: Fisher log saved to {path}")
