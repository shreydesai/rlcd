"""MLP calibration policy: one forward pass -> a single Bernoulli probability.

10 -> 64 -> 64 -> 1, ReLU hidden activations, sigmoid head. No sampling of
network internals; the only stochasticity used anywhere is the exploration
jitter applied externally to the emitted probability during RLCD training.
"""

import torch
import torch.nn as nn


class CalibrationMLP(nn.Module):
    def __init__(self, in_dim=10, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def logits(self, x):
        return self.net(x).squeeze(-1)

    def forward(self, x):
        """Return calibrated probability p in (0, 1), one pass, no sampling."""
        return torch.sigmoid(self.logits(x))
