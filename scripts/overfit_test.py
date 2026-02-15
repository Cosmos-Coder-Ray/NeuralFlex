from __future__ import annotations

__test__ = False

import argparse
import time
from pathlib import Path

import torch

from neuralflex.config.schemas import AttentionConfig, ModelConfig, TrainingConfig
from neuralflex.tokenizer import NeuralFlexTokenizer
from neuralflex.training.trainer import Trainer


class SimpleCharTokenizer:
    """Deterministic character-level tokenizer used for overfit validation fallback."""

    def __init__(self, vocab: list[str]) -> None:
        self._base = NeuralFlexTokenizer(vocab=vocab, merges=[])
        self.vocab = self._base.vocab
        self.merges = self._base.merges
        self.special_ids = self._base.special_ids

    @classmethod
    def build_from_file(cls, train_path: str) -> "SimpleCharTokenizer":
        chars: set[str] = set()
        with Path(train_path).open("r", encoding="utf-8") as f:
            for line in f:
                chars.update(line.rstrip("\n"))
        vocab = ["<pad>", "<unk>", "<bos>", "<eos>", "<system>", "<tool>", "<user>"]
        vocab.extend(sorted(chars))
        return cls(vocab=vocab)

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        token_ids: list[int] = []
        if add_bos:
            token_ids.append(self.special_ids.bos)
        unk = self.special_ids.unk
        token_ids.extend(self._base.token_to_id.get(ch, unk) for ch in text)
        if add_eos:
            token_ids.append(self.special_ids.eos)
        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        output: list[str] = []
        specials = set(self.vocab[:7])
        for token_id in token_ids:
            if token_id < 0 or token_id >= len(self.vocab):
                continue
            token = self.vocab[token_id]
            if token in specials:
                continue
            output.append(token)
        return "".join(output)


class OverfitTrainer(Trainer):
    """Trainer variant for overfit validation that reuses a prebuilt tokenizer."""

    def __init__(self, model_config: ModelConfig, train_config: TrainingConfig, tokenizer: NeuralFlexTokenizer | SimpleCharTokenizer) -> None:
        self._provided_tokenizer = tokenizer
        super().__init__(model_config, train_config)

    def _build_tokenizer(self, train_path: str):
        self.model_config.vocab_size = len(self._provided_tokenizer.vocab)
        return self._provided_tokenizer


def _load_or_build_tokenizer(train_path: str, tokenizer_path: Path) -> NeuralFlexTokenizer | SimpleCharTokenizer:
    if tokenizer_path.exists():
        return NeuralFlexTokenizer.load(tokenizer_path)
    return SimpleCharTokenizer.build_from_file(train_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Overfit validation for dense baseline")
    parser.add_argument("--train-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="artifacts/overfit_test")
    parser.add_argument("--tokenizer-path", type=str, default="")
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
        precision="fp16",
    )
    tokenizer_path = Path(args.tokenizer_path) if args.tokenizer_path else Path(args.output_dir) / "tokenizer.nfx"
    start_time = time.perf_counter()
    tokenizer = _load_or_build_tokenizer(args.train_path, tokenizer_path)
    trainer = OverfitTrainer(model_cfg, train_cfg, tokenizer)
    prep_time = time.perf_counter() - start_time
    print("tokenizer_ready")
    if prep_time > 3.0:
        raise RuntimeError(f"Overfit preparation took too long ({prep_time:.2f}s > 3.00s)")

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
