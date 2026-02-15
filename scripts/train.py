from __future__ import annotations

import argparse
from dataclasses import asdict

import yaml

from neuralflex.config.schemas import AttentionConfig, ModelConfig, TrainingConfig
from neuralflex.training.trainer import Trainer


def _load_configs(path: str) -> tuple[ModelConfig, TrainingConfig]:
    with open(path, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)

    attn_cfg = AttentionConfig(**payload.get("model", {}).get("attention", {}))
    model_cfg = ModelConfig(**{k: v for k, v in payload.get("model", {}).items() if k != "attention"}, attention=attn_cfg)
    train_cfg = TrainingConfig(**payload.get("train", {}))
    return model_cfg, train_cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Train NeuralFlex dense transformer baseline")
    parser.add_argument("--config", type=str, required=True, help="Path to yaml config")
    args = parser.parse_args()

    model_cfg, train_cfg = _load_configs(args.config)
    trainer = Trainer(model_cfg, train_cfg)
    print("Training with config:", {"model": asdict(model_cfg), "train": asdict(train_cfg)})
    trainer.train()


if __name__ == "__main__":
    main()
