from __future__ import annotations

import torch

from neuralflex.config.schemas import AttentionConfig, ModelConfig
from neuralflex.models.model import NeuralFlexMoEModel


def _config() -> ModelConfig:
    return ModelConfig(
        vocab_size=128,
        hidden_size=64,
        num_layers=2,
        max_position_embeddings=64,
        intermediate_size=128,
        attention=AttentionConfig(num_heads=4, num_kv_heads=2, head_dim=16),
    )


def test_dense_forward_and_loss() -> None:
    model = NeuralFlexMoEModel(_config())
    input_ids = torch.randint(0, 128, (2, 16))
    labels = torch.randint(0, 128, (2, 16))
    out = model(input_ids, labels=labels)
    assert out.logits.shape == (2, 16, 128)
    assert out.loss is not None


def test_generate_with_kv_cache() -> None:
    model = NeuralFlexMoEModel(_config())
    input_ids = torch.randint(0, 128, (1, 8))
    generated = model.generate(input_ids, max_new_tokens=4)
    assert generated.shape == (1, 12)
