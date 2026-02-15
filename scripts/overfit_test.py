from __future__ import annotations

__test__ = False

import argparse
from pathlib import Path

import torch

from neuralflex.config.schemas import AttentionConfig, ModelConfig, TrainingConfig
from neuralflex.training.trainer import Trainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Overfit validation for dense baseline")
    parser.add_argument("--train-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="artifacts/overfit_test")
    parser.add_argument("--max-steps", type=int, default=800)
    args = parser.parse_args()

    model_cfg = ModelConfig(
        vocab_size=1024,
        hidden_size=256,
        num_layers=4,
        max_position_embeddings=256,
        intermediate_size=768,
        attention=AttentionConfig(num_heads=8, num_kv_heads=2, head_dim=32),
    )
    train_cfg = TrainingConfig(
        train_path=args.train_path,
        output_dir=args.output_dir,
        sequence_length=128,
        batch_size=8,
        max_steps=args.max_steps,
        grad_accum_steps=2,
        learning_rate=5e-4,
        min_learning_rate=1e-4,
        warmup_steps=20,
        log_every=10,
        save_every=200,
        precision="bf16",
    )
    trainer = Trainer(model_cfg, train_cfg)
    trainer.train()

    ckpt = Path(args.output_dir) / "final.pt"
    trainer.load_checkpoint(str(ckpt))
    model = trainer.model.eval()

    prompt = "neuralflex"
    ids = trainer.tokenizer.encode(prompt, add_bos=True)
    input_ids = torch.tensor([ids], dtype=torch.long, device=trainer.device)
    out = model.generate(input_ids, max_new_tokens=24)
    text = trainer.tokenizer.decode(out[0].tolist())
    print("PROMPT:", prompt)
    print("GENERATED:", text)


if __name__ == "__main__":
    main()
