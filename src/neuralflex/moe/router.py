from __future__ import annotations

import torch
from torch import nn


class TopKRouter(nn.Module):
    """Top-k router stub with load balancing hooks."""

    def __init__(self, hidden_size: int, num_experts: int, top_k: int) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Returns indices, scores, and routing metrics."""
        raise NotImplementedError("Routing dispatch is scheduled for Milestone 2.")
