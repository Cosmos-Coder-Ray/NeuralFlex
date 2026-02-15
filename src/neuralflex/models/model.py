from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from neuralflex.attention.gqa import GroupedQueryAttention
from neuralflex.config.schemas import ModelConfig


@dataclass
class ModelOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None
    past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return norm * self.weight


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        attn = config.attention
        self.attn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn = GroupedQueryAttention(
            config.hidden_size,
            attn.num_heads,
            attn.num_kv_heads,
            attn.head_dim,
            rope_theta=attn.rope_theta,
        )
        intermediate = config.intermediate_size or int((8 * config.hidden_size) / 3)
        self.mlp_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = SwiGLU(config.hidden_size, intermediate)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor | None,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None,
        use_cache: bool,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        attn_out, present = self.attn(
            self.attn_norm(x),
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.mlp(self.mlp_norm(x)))
        return x, present


class NeuralFlexMoEModel(nn.Module):
    """Dense decoder-only transformer baseline (MoE intentionally disabled)."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([DecoderBlock(config) for _ in range(config.num_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        use_cache: bool = False,
    ) -> ModelOutput:
        x = self.embed_tokens(input_ids)
        next_kv: list[tuple[torch.Tensor, torch.Tensor]] = [] if use_cache else None

        for layer_idx, layer in enumerate(self.layers):
            past = None if past_key_values is None else past_key_values[layer_idx]
            x, present = layer(
                x,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past,
                use_cache=use_cache,
            )
            if use_cache and present is not None:
                next_kv.append(present)

        logits = self.lm_head(self.norm(x))
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100)

        return ModelOutput(logits=logits, loss=loss, past_key_values=next_kv)

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, *, max_new_tokens: int, temperature: float = 1.0) -> torch.Tensor:
        self.eval()
        out = input_ids
        past_key_values = None
        for _ in range(max_new_tokens):
            model_input = out[:, -1:] if past_key_values is not None else out
            result = self.forward(model_input, past_key_values=past_key_values, use_cache=True)
            past_key_values = result.past_key_values
            logits = result.logits[:, -1, :] / max(temperature, 1e-5)
            next_id = torch.argmax(logits, dim=-1, keepdim=True)
            out = torch.cat([out, next_id], dim=-1)
        return out
