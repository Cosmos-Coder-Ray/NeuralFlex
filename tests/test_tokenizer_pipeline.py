from __future__ import annotations

import json

from neuralflex.data.pipeline import DataPipeline, DatasetSource
from neuralflex.tokenizer import BPETrainer, NeuralFlexTokenizer


def _build_tokenizer() -> NeuralFlexTokenizer:
    texts = [
        "Hello world from NeuralFlex",
        "Hello tokenizer world",
        "Deterministic tokenization is important",
    ]
    trainer = BPETrainer(vocab_size=128, min_frequency=1)
    return trainer.train(texts)


def test_tokenizer_determinism_and_reversibility(tmp_path) -> None:
    tokenizer = _build_tokenizer()

    text = " Hello   world from NeuralFlex "
    ids_a = tokenizer.encode(text, add_bos=True, add_eos=True)
    ids_b = tokenizer.encode(text, add_bos=True, add_eos=True)
    assert ids_a == ids_b

    decoded = tokenizer.decode(ids_a)
    assert decoded == "Hello world from NeuralFlex"

    path = tmp_path / "tokenizer.nfx"
    tokenizer.save(path)
    mmap_tokenizer = NeuralFlexTokenizer.load(path)
    assert mmap_tokenizer.encode("Hello world") == tokenizer.encode("Hello world")
    assert mmap_tokenizer.decode(tokenizer.encode("Hello world")) == "Hello world"


def test_dataset_pipeline_is_deterministic(tmp_path) -> None:
    tokenizer = _build_tokenizer()
    jsonl_path = tmp_path / "data.jsonl"
    payloads = [
        {"text": "alpha beta gamma"},
        {"text": "delta epsilon zeta"},
        {"text": "eta theta iota"},
    ]
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in payloads:
            f.write(json.dumps(row) + "\n")

    source = DatasetSource(path=str(jsonl_path), format="jsonl")
    pipeline_a = DataPipeline(
        sources=[source],
        tokenizer=tokenizer,
        sequence_length=8,
        batch_size=2,
        shuffle_buffer_size=2,
        seed=42,
        sliding_window_stride=4,
    )
    pipeline_b = DataPipeline(
        sources=[source],
        tokenizer=tokenizer,
        sequence_length=8,
        batch_size=2,
        shuffle_buffer_size=2,
        seed=42,
        sliding_window_stride=4,
    )

    batches_a = list(pipeline_a.build(epoch=0))
    batches_b = list(pipeline_b.build(epoch=0))

    assert len(batches_a) == len(batches_b)
    for batch_a, batch_b in zip(batches_a, batches_b):
        assert batch_a.input_ids.tolist() == batch_b.input_ids.tolist()
        assert batch_a.labels.tolist() == batch_b.labels.tolist()
        assert batch_a.attention_mask.tolist() == batch_b.attention_mask.tolist()
        assert batch_a.position_ids.tolist() == batch_b.position_ids.tolist()
