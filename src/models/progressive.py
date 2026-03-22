"""
Active Column (ProgressiveColumn) for Progress & Compress.

Implements the column architecture from:
    Schwarz et al. (2018) "Progress & Compress: A scalable system for
    continual learning." ICML 2018.

The Active Column (AC) mirrors the KB CNN backbone but receives an
additional lateral input from the frozen KB's flattened CNN features
at the first FC layer (Eq. 2 of the paper):

    h_FC^AC = ReLU( W_AC · flat_AC  +  U · flat_KB )

where U is a learned linear adapter (no bias) that maps the KB's feature
space to the AC's FC hidden size.  This lets the AC reuse the KB's
already-learned representations while training on a new task without
modifying the KB.

After the compress phase the AC is discarded; only the KB persists at
test time, so no task identifier is required at inference.
"""

import torch
import torch.nn as nn
from typing import List, Optional

from src.models.dqn import DQNNetwork


class ProgressiveColumn(nn.Module):
    """Active Column with lateral connection from a frozen Knowledge Base.

    Args:
        kb: Frozen Knowledge Base DQNNetwork. Its CNN backbone
            (``kb.features``) is queried during forward to provide the
            lateral signal.  The KB must be frozen externally (all
            ``requires_grad = False``) before instantiating this class.
        in_channels: Number of stacked frames (must match KB).
        conv_channels: Conv output channels per layer.
        conv_kernels: Conv kernel sizes per layer.
        conv_strides: Conv strides per layer.
        fc_hidden: FC hidden layer size.
        unified_action_dim: Union action space dimension.
    """

    def __init__(
        self,
        kb: DQNNetwork,
        in_channels: int = 4,
        conv_channels: Optional[List[int]] = None,
        conv_kernels: Optional[List[int]] = None,
        conv_strides: Optional[List[int]] = None,
        fc_hidden: int = 512,
        unified_action_dim: int = 6,
    ):
        super().__init__()

        if conv_channels is None:
            conv_channels = [32, 64, 64]
        if conv_kernels is None:
            conv_kernels = [8, 4, 3]
        if conv_strides is None:
            conv_strides = [4, 2, 1]

        self.unified_action_dim = unified_action_dim
        # Store as a non-module attribute so deepcopy does not duplicate KB.
        # The caller is responsible for freezing KB before instantiation.
        object.__setattr__(self, '_kb_ref', kb)

        # AC's own CNN backbone (same architecture, random init)
        layers: List[nn.Module] = []
        prev_ch = in_channels
        for out_ch, k, s in zip(conv_channels, conv_kernels, conv_strides):
            layers.append(nn.Conv2d(prev_ch, out_ch, kernel_size=k, stride=s))
            layers.append(nn.ReLU(inplace=True))
            prev_ch = out_ch
        self.features = nn.Sequential(*layers)

        self._ac_flat: int = self._flat_size(in_channels)
        self._kb_flat: int = kb._feature_size   # same architecture -> same size

        # W_AC : AC features -> fc_hidden  (standard linear)
        self.ac_linear = nn.Linear(self._ac_flat, fc_hidden)
        # U : KB features -> fc_hidden  (no bias, as in the paper)
        self.lateral = nn.Linear(self._kb_flat, fc_hidden, bias=False)
        # Output head
        self.head = nn.Linear(fc_hidden, unified_action_dim)

    # ------------------------------------------------------------------
    def _flat_size(self, in_channels: int) -> int:
        dummy = torch.zeros(1, in_channels, 84, 84)
        with torch.no_grad():
            out = self.features(dummy)
        return int(out.view(1, -1).size(1))

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward combining AC CNN features and frozen KB lateral features.

        Args:
            x: (batch, channels, 84, 84), uint8 [0,255] or float [0,1].

        Returns:
            Q-values (batch, unified_action_dim).
        """
        if x.dtype == torch.uint8:
            x = x.float() / 255.0

        # AC path (trainable)
        ac_flat = self.features(x).view(x.size(0), -1)

        # KB lateral path (frozen, no gradients)
        kb = object.__getattribute__(self, '_kb_ref')
        with torch.no_grad():
            kb_flat = kb.features(x).view(x.size(0), -1)

        # Lateral combination  h = ReLU(W_AC*ac + U*kb)
        h = torch.relu(self.ac_linear(ac_flat) + self.lateral(kb_flat))
        return self.head(h)
