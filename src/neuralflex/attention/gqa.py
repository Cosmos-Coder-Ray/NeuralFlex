from __future__ import annotations

import torch
from torch import nn


class GroupedQueryAttention(nn.Module):
    """Stub for GQA attention with optional flash/sliding-window backends."""

    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int, head_dim: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

    def forward(self, x: torch.Tensor, *, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        raise NotImplementedError("GQA execution backend will be implemented in Milestone 1.")
