from dataclasses import dataclass, field


@dataclass
class AttentionConfig:
    num_heads: int = 16
    num_kv_heads: int = 4
    head_dim: int = 128
    sliding_window: int | None = None
    use_flash: bool = True


@dataclass
class MoEConfig:
    num_experts: int = 8
    top_k: int = 2
    capacity_factor: float = 1.25
    aux_loss_coef: float = 1e-2


@dataclass
class ModelConfig:
    vocab_size: int = 64000
    hidden_size: int = 2048
    num_layers: int = 24
    max_position_embeddings: int = 4096
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    moe: MoEConfig = field(default_factory=MoEConfig)
