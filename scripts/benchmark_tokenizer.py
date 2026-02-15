from __future__ import annotations

import argparse
import time
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neuralflex.tokenizer import BPETrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark NeuralFlex tokenizer throughput")
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--text", type=str, default="NeuralFlex tokenizer benchmark text for throughput measurement.")
    args = parser.parse_args()

    trainer = BPETrainer(vocab_size=512, min_frequency=1)
    tokenizer = trainer.train([args.text] * 100)

    total_tokens = 0
    start = time.perf_counter()
    for _ in range(args.iterations):
        total_tokens += len(tokenizer.encode(args.text))
    elapsed = time.perf_counter() - start

    throughput = total_tokens / elapsed if elapsed > 0 else 0.0
    print(f"tokens={total_tokens}")
    print(f"seconds={elapsed:.6f}")
    print(f"tokens_per_second={throughput:.2f}")


if __name__ == "__main__":
    main()
