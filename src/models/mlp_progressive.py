"""
MLP Active Column (MLPProgressiveColumn) for Progress & Compress on Classic Control.

MLP analogue of ProgressiveColumn (src/models/progressive.py).  Implements the
lateral connection architecture from Schwarz et al. (2018) "Progress & Compress:
A scalable system for continual learning" (ICML 2018), adapted for
low-dimensional state inputs instead of image observations.

Architecture:
    h_AC = ReLU( W_AC * trunk_AC(state) + U * trunk_KB(state) )
    Q    = head(h_AC)

where ``trunk_KB`` is the frozen Knowledge Base's shared hidden trunk, ``U`` is
a learned linear adapter (no bias), and ``trunk_AC`` is the Active Column's own
hidden trunk (same architecture, fresh initialization).

After the compress phase the AC is discarded; only the KB persists at test time,
so no task identifier is required at inference.
"""

import torch
import torch.nn as nn
from typing import List, Optional

from src.models.mlp_dqn import MLPDQNNetwork


class MLPProgressiveColumn(nn.Module):
    """Active Column with lateral connection from a frozen MLP Knowledge Base.

    Args:
        kb: Frozen ``MLPDQNNetwork`` Knowledge Base.  Its ``trunk`` is queried
            during forward to provide the lateral signal.  The KB must be
            frozen externally (all ``requires_grad = False``) before
            instantiating this class.
        state_dim: Dimensionality of the (zero-padded) input state vector.
        hidden_dims: List of hidden layer widths.  Must match the KB
            architecture for the trunk layers.
        unified_action_dim: Union action space dimension.
    """

    def __init__(
        self,
        kb: MLPDQNNetwork,
        state_dim: int = 8,
        hidden_dims: Optional[List[int]] = None,
        unified_action_dim: int = 4,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [128, 128]

        self.unified_action_dim = unified_action_dim

        # Store KB as a non-module attribute so deepcopy does not duplicate it.
        # The caller is responsible for freezing KB before instantiation.
        object.__setattr__(self, '_kb_ref', kb)

        # ── AC's own trunk (same architecture, random init) ─────────────
        trunk_layers: List[nn.Module] = []
        prev_dim = state_dim
        for h_dim in hidden_dims[:-1]:
            trunk_layers.append(nn.Linear(prev_dim, h_dim))
            trunk_layers.append(nn.ReLU(inplace=True))
            prev_dim = h_dim
        self.trunk = nn.Sequential(*trunk_layers)

        # Trunk output dimension (input to the last hidden layer)
        self._ac_trunk_dim: int = prev_dim
        self._kb_trunk_dim: int = prev_dim  # same architecture -> same size

        last_hidden = hidden_dims[-1]

        # W_AC: AC trunk features -> last_hidden (standard linear)
        self.ac_linear = nn.Linear(self._ac_trunk_dim, last_hidden)

        # U: KB trunk features -> last_hidden (no bias, as in the paper)
        self.lateral = nn.Linear(self._kb_trunk_dim, last_hidden, bias=False)

        # Output head
        self.head = nn.Linear(last_hidden, unified_action_dim)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward combining AC trunk features and frozen KB lateral features.

        Args:
            x: Float state tensor of shape ``(batch, state_dim)``.
               No pixel normalization is applied.

        Returns:
            Q-values of shape ``(batch, unified_action_dim)``.
        """
        # AC path (trainable)
        ac_features = self.trunk(x)

        # KB lateral path (frozen, no gradients)
        kb: MLPDQNNetwork = object.__getattribute__(self, '_kb_ref')
        with torch.no_grad():
            kb_features = kb.hidden_features(x)

        # Lateral combination: h = ReLU(W_AC * ac + U * kb)
        h = torch.relu(self.ac_linear(ac_features) + self.lateral(kb_features))

        return self.head(h)
