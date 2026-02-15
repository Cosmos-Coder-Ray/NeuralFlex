from __future__ import annotations

from torch import nn


class SwiGLUExpert(nn.Module):
    """Single expert FFN stub."""

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w2 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w3 = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        raise NotImplementedError("Expert forward path will be implemented in Milestone 1.")
