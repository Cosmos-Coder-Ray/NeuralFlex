from dataclasses import dataclass, field


@dataclass
class AttentionConfig:
    num_heads: int = 16
    num_kv_heads: int = 4
    head_dim: int = 128
    rope_theta: float = 10000.0
    use_flash: bool = False


@dataclass
class ModelConfig:
    vocab_size: int = 64000
    hidden_size: int = 2048
    num_layers: int = 24
    max_position_embeddings: int = 4096
    intermediate_size: int | None = None
    rms_norm_eps: float = 1e-5
    dropout: float = 0.0
    attention: AttentionConfig = field(default_factory=AttentionConfig)


@dataclass
class TrainingConfig:
    train_path: str
    train_format: str = "txt"
    text_key: str = "text"
    output_dir: str = "artifacts/dense_baseline"
    sequence_length: int = 256
    batch_size: int = 8
    max_steps: int = 2000
    grad_accum_steps: int = 1
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    warmup_steps: int = 100
    weight_decay: float = 0.1
    grad_clip_norm: float = 1.0
    precision: str = "bf16"
    seed: int = 42
    log_every: int = 10
    save_every: int = 200
    shuffle_buffer_size: int = 2048
