from __future__ import annotations

import json
import multiprocessing as mp
import random
from dataclasses import dataclass
from pathlib import Path
from queue import Empty
from typing import Iterable, Iterator

import numpy as np

from neuralflex.tokenizer.tokenizer import NeuralFlexTokenizer


@dataclass(frozen=True)
class DatasetSource:
    path: str
    format: str
    text_key: str = "text"


@dataclass(frozen=True)
class Batch:
    input_ids: np.ndarray
    labels: np.ndarray
    attention_mask: np.ndarray
    position_ids: np.ndarray


class DataPipeline:
    """Streaming token dataset with deterministic shuffle and sequence packing."""

    def __init__(
        self,
        *,
        sources: list[DatasetSource],
        tokenizer: NeuralFlexTokenizer,
        sequence_length: int,
        batch_size: int,
        shuffle_buffer_size: int = 4096,
        seed: int = 0,
        sliding_window_stride: int | None = None,
    ) -> None:
        self.sources = sources
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.shuffle_buffer_size = shuffle_buffer_size
        self.seed = seed
        self.sliding_window_stride = sliding_window_stride or sequence_length

    def build(self, *, epoch: int = 0) -> Iterator[Batch]:
        documents = self._iter_documents(epoch=epoch)
        token_sequences = self._iter_token_sequences(documents)
        return self._batch_sequences(token_sequences)

    def _iter_documents(self, *, epoch: int) -> Iterator[str]:
        rng = random.Random(self.seed + epoch)
        source_indices = list(range(len(self.sources)))
        rng.shuffle(source_indices)

        stream = self._stream_sources(source_indices)
        yield from self._buffered_shuffle(stream, rng=rng)

    def _stream_sources(self, source_indices: list[int]) -> Iterator[str]:
        for idx in source_indices:
            source = self.sources[idx]
            path = Path(source.path)
            fmt = source.format.lower()
            if fmt == "jsonl":
                with path.open("r", encoding="utf-8") as f:
                    for line in f:
                        payload = json.loads(line)
                        text = payload.get(source.text_key, "")
                        if text:
                            yield str(text)
            elif fmt == "txt":
                with path.open("r", encoding="utf-8") as f:
                    for line in f:
                        text = line.strip()
                        if text:
                            yield text
            elif fmt == "parquet":
                yield from self._read_parquet(path, source.text_key)
            else:
                raise ValueError(f"Unsupported source format: {source.format}")

    @staticmethod
    def _read_parquet(path: Path, text_key: str) -> Iterator[str]:
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("Parquet support requires pyarrow") from exc

        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(columns=[text_key], batch_size=1024):
            for value in batch.column(0).to_pylist():
                if value:
                    yield str(value)

    def _buffered_shuffle(self, documents: Iterable[str], *, rng: random.Random) -> Iterator[str]:
        buffer: list[str] = []
        for document in documents:
            if len(buffer) < self.shuffle_buffer_size:
                buffer.append(document)
                continue
            swap_index = rng.randrange(len(buffer))
            yield buffer[swap_index]
            buffer[swap_index] = document
        while buffer:
            idx = rng.randrange(len(buffer))
            yield buffer.pop(idx)

    def _iter_token_sequences(self, documents: Iterable[str]) -> Iterator[np.ndarray]:
        stream: list[int] = []
        stride = self.sliding_window_stride

        for doc in documents:
            encoded = self.tokenizer.encode(doc, add_eos=True)
            stream.extend(encoded)
            while len(stream) >= self.sequence_length + 1:
                window = np.asarray(stream[: self.sequence_length + 1], dtype=np.int64)
                yield window
                if stride >= self.sequence_length:
                    stream = stream[self.sequence_length :]
                else:
                    stream = stream[stride:]

    def _batch_sequences(self, sequences: Iterable[np.ndarray]) -> Iterator[Batch]:
        batch: list[np.ndarray] = []
        for seq in sequences:
            batch.append(seq)
            if len(batch) == self.batch_size:
                yield self._to_batch(batch)
                batch = []

    def _to_batch(self, examples: list[np.ndarray]) -> Batch:
        stacked = np.stack(examples)
        input_ids = stacked[:, :-1]
        labels = stacked[:, 1:]
        attention_mask = np.ones_like(input_ids, dtype=np.int64)
        position_ids = np.arange(input_ids.shape[1], dtype=np.int64)[None, :].repeat(input_ids.shape[0], axis=0)
        return Batch(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )


class PrefetchDataLoader:
    """Multiprocess prefetch wrapper around DataPipeline."""

    def __init__(self, pipeline: DataPipeline, *, epoch: int = 0, prefetch_batches: int = 16) -> None:
        self.pipeline = pipeline
        self.epoch = epoch
        self.prefetch_batches = prefetch_batches
        self._queue: mp.Queue[Batch | None] = mp.Queue(maxsize=prefetch_batches)
        self._proc: mp.Process | None = None

    def __iter__(self) -> Iterator[Batch]:
        if self._proc is None:
            self._proc = mp.Process(target=self._worker, daemon=True)
            self._proc.start()

        while True:
            try:
                item = self._queue.get(timeout=5)
            except Empty:
                if self._proc is not None and not self._proc.is_alive():
                    break
                continue
            if item is None:
                break
            yield item

        if self._proc is not None:
            self._proc.join(timeout=1)
            self._proc = None

    def _worker(self) -> None:
        try:
            for batch in self.pipeline.build(epoch=self.epoch):
                self._queue.put(batch)
        finally:
            self._queue.put(None)
