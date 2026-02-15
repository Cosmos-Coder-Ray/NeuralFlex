from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from neuralflex.config.schemas import ModelConfig


@dataclass
class ModelOutput:
    logits: torch.Tensor


class NeuralFlexMoEModel(nn.Module):
    """Top-level model skeleton for NeuralFlex SLM-MoE."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, *, attention_mask: torch.Tensor | None = None) -> ModelOutput:
        raise NotImplementedError("Core transformer stack is planned for Milestone 1.")
