from __future__ import annotations

import math

import torch
from torch import nn


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).reshape_as(x)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return (x * cos) + (_rotate_half(x) * sin)


class GroupedQueryAttention(nn.Module):
    """Grouped-query attention with RoPE and optional KV cache."""

    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int, head_dim: int, rope_theta: float = 10000.0) -> None:
        super().__init__()
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads for GQA")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.q_per_kv = num_heads // num_kv_heads
        self.rope_theta = rope_theta

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

    def _rope_cache(self, positions: torch.Tensor, *, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = 1.0 / (self.rope_theta ** (torch.arange(0, self.head_dim, 2, device=positions.device).float() / self.head_dim))
        freqs = torch.outer(positions.float(), inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1).to(dtype=dtype)
        return emb.cos()[None, None, :, :], emb.sin()[None, None, :, :]

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        if self.q_per_kv == 1:
            return x
        b, h, t, d = x.shape
        return x[:, :, None, :, :].expand(b, h, self.q_per_kv, t, d).reshape(b, h * self.q_per_kv, t, d)

    def forward(
        self,
        x: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        bsz, seq_len, _ = x.shape
        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if position_ids is None:
            start = 0 if past_key_value is None else past_key_value[0].shape[2]
            position_ids = torch.arange(start, start + seq_len, device=x.device)
        else:
            position_ids = position_ids[0] if position_ids.ndim == 2 else position_ids

        cos, sin = self._rope_cache(position_ids, dtype=q.dtype)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        present = (k, v) if use_cache else None

        k_full = self._repeat_kv(k)
        v_full = self._repeat_kv(v)

        scores = torch.matmul(q, k_full.transpose(-2, -1)) / math.sqrt(self.head_dim)
        key_len = k_full.shape[-2]

        causal = torch.ones(seq_len, key_len, device=x.device, dtype=torch.bool).triu(diagonal=1 + key_len - seq_len)
        scores = scores.masked_fill(causal[None, None, :, :], torch.finfo(scores.dtype).min)

        if attention_mask is not None:
            mask = attention_mask[:, None, None, :key_len].to(torch.bool)
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

        probs = torch.softmax(scores.float(), dim=-1).to(dtype=q.dtype)
        out = torch.matmul(probs, v_full)
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.num_heads * self.head_dim)
        return self.o_proj(out), present
