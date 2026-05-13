from __future__ import annotations

import torch
import torch.nn as nn


class FiLM(nn.Module):
    # Feature-wise Linear Modulation: scale + shift features by a condition vector.

    def __init__(self, feature_dim: int, condition_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.gamma_net = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
        )
        self.beta_net = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
        )
        # Zero-init the last layer of each branch so FiLM starts as identity.
        for net in (self.gamma_net, self.beta_net):
            last = net[-1]
            if isinstance(last, nn.Linear):
                nn.init.zeros_(last.weight)
                nn.init.zeros_(last.bias)

    def forward(self, features: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        if condition.dim() == 1:
            condition = condition.unsqueeze(0)
        if condition.shape[0] == 1 and features.shape[0] > 1:
            condition = condition.expand(features.shape[0], -1)
        if condition.shape[0] != features.shape[0]:
            raise ValueError(
                f"Batch size mismatch: features {features.shape}, condition {condition.shape}"
            )

        gamma = self.gamma_net(condition)
        beta = self.beta_net(condition)
        return (1.0 + gamma) * features + beta
