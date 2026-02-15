from __future__ import annotations

import json
import mmap
import struct
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterable, Iterator

MAGIC = b"NFXTK1\x00\x00"
_VERSION = 1

SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>", "<system>", "<tool>", "<user>"]


@dataclass(frozen=True)
class SpecialTokenIds:
    pad: int
    unk: int
    bos: int
    eos: int
    system: int
    tool: int
    user: int


class DeterministicNormalizer:
    """Deterministic text normalization used by training and runtime."""

    def normalize(self, text: str) -> str:
        normalized = unicodedata.normalize("NFKC", text)
        normalized = " ".join(normalized.strip().split())
        return normalized


class BPETrainer:
    """Simple deterministic BPE trainer implemented fully in-repo."""

    def __init__(self, *, vocab_size: int, min_frequency: int = 2, normalizer: DeterministicNormalizer | None = None) -> None:
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.normalizer = normalizer or DeterministicNormalizer()

    def train(self, texts: Iterable[str]) -> "NeuralFlexTokenizer":
        word_counts: Counter[tuple[str, ...]] = Counter()
        for text in texts:
            normalized = self.normalizer.normalize(text)
            if not normalized:
                continue
            for word in normalized.split(" "):
                chars = tuple(list(word) + ["</w>"])
                word_counts[chars] += 1

        merges: list[tuple[str, str]] = []
        symbols = set(SPECIAL_TOKENS)
        for word in word_counts:
            symbols.update(word)

        target_merges = max(self.vocab_size - len(symbols), 0)
        for _ in range(target_merges):
            pair_counts: Counter[tuple[str, str]] = Counter()
            for word, freq in word_counts.items():
                for i in range(len(word) - 1):
                    pair_counts[(word[i], word[i + 1])] += freq
            if not pair_counts:
                break
            (best_left, best_right), best_freq = max(pair_counts.items(), key=lambda item: (item[1], item[0]))
            if best_freq < self.min_frequency:
                break

            merged = f"{best_left}{best_right}"
            merges.append((best_left, best_right))
            symbols.add(merged)

            updated_counts: Counter[tuple[str, ...]] = Counter()
            for word, freq in word_counts.items():
                updated_counts[self._merge_word(word, best_left, best_right)] += freq
            word_counts = updated_counts

        vocab = list(SPECIAL_TOKENS)
        vocab.extend(sorted(s for s in symbols if s not in SPECIAL_TOKENS))

        return NeuralFlexTokenizer(vocab=vocab, merges=merges, normalizer=self.normalizer)

    @staticmethod
    def _merge_word(word: tuple[str, ...], left: str, right: str) -> tuple[str, ...]:
        merged = f"{left}{right}"
        output: list[str] = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == left and word[i + 1] == right:
                output.append(merged)
                i += 2
            else:
                output.append(word[i])
                i += 1
        return tuple(output)


class NeuralFlexTokenizer:
    """In-repo BPE tokenizer with compact binary export and mmap loading."""

    def __init__(self, *, vocab: list[str], merges: list[tuple[str, str]], normalizer: DeterministicNormalizer | None = None) -> None:
        self.normalizer = normalizer or DeterministicNormalizer()
        self.vocab = vocab
        self.merges = merges
        self.token_to_id = {token: i for i, token in enumerate(vocab)}
        self.merge_ranks = {merge: i for i, merge in enumerate(merges)}
        self.special_ids = SpecialTokenIds(
            pad=self.token_to_id["<pad>"],
            unk=self.token_to_id["<unk>"],
            bos=self.token_to_id["<bos>"],
            eos=self.token_to_id["<eos>"],
            system=self.token_to_id["<system>"],
            tool=self.token_to_id["<tool>"],
            user=self.token_to_id["<user>"],
        )

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        normalized = self.normalizer.normalize(text)
        token_ids: list[int] = []
        if add_bos:
            token_ids.append(self.special_ids.bos)
        if normalized:
            for word in normalized.split(" "):
                token_ids.extend(self._encode_word(word))
        if add_eos:
            token_ids.append(self.special_ids.eos)
        return token_ids

    def encode_stream(self, chunks: Iterable[str]) -> Iterator[int]:
        remainder = ""
        for chunk in chunks:
            if not chunk:
                continue
            text = remainder + chunk
            split = text.rsplit(" ", 1)
            complete = split[0] if len(split) == 2 else ""
            remainder = split[1] if len(split) == 2 else text
            if complete:
                for tid in self.encode(complete):
                    yield tid
        if remainder:
            for tid in self.encode(remainder):
                yield tid

    def decode(self, token_ids: Iterable[int]) -> str:
        words: list[str] = []
        current = ""
        for token_id in token_ids:
            if token_id < 0 or token_id >= len(self.vocab):
                token = "<unk>"
            else:
                token = self.vocab[token_id]
            if token in SPECIAL_TOKENS:
                continue
            if token.endswith("</w>"):
                current += token[:-4]
                words.append(current)
                current = ""
            else:
                current += token
        if current:
            words.append(current)
        return " ".join(words)

    def decode_incremental(self, token_id: int) -> str:
        if token_id < 0 or token_id >= len(self.vocab):
            return ""
        token = self.vocab[token_id]
        if token in SPECIAL_TOKENS:
            return ""
        if token.endswith("</w>"):
            return token[:-4] + " "
        return token

    def _encode_word(self, word: str) -> list[int]:
        symbols = list(word) + ["</w>"]
        while len(symbols) > 1:
            pairs = [(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)]
            ranked = [(self.merge_ranks[p], p) for p in pairs if p in self.merge_ranks]
            if not ranked:
                break
            _, best = min(ranked)
            merged: list[str] = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == best[0] and symbols[i + 1] == best[1]:
                    merged.append(symbols[i] + symbols[i + 1])
                    i += 2
                else:
                    merged.append(symbols[i])
                    i += 1
            symbols = merged
        return [self.token_to_id.get(sym, self.special_ids.unk) for sym in symbols]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        token_blob = "\n".join(self.vocab).encode("utf-8")
        offsets: list[int] = []
        cursor = 0
        for token in self.vocab:
            offsets.append(cursor)
            cursor += len(token.encode("utf-8")) + 1
        merge_blob = json.dumps(self.merges, separators=(",", ":")).encode("utf-8")

        with path.open("wb") as f:
            self._write_header(f, token_blob_len=len(token_blob), merge_blob_len=len(merge_blob), offsets_len=len(offsets))
            f.write(struct.pack(f"<{len(offsets)}I", *offsets))
            f.write(token_blob)
            f.write(merge_blob)

    @classmethod
    def load(cls, path: str | Path) -> "MMapNeuralFlexTokenizer":
        return MMapNeuralFlexTokenizer(Path(path))

    def _write_header(self, f: BinaryIO, *, token_blob_len: int, merge_blob_len: int, offsets_len: int) -> None:
        header = struct.pack(
            "<8sI5I7I",
            MAGIC,
            _VERSION,
            len(self.vocab),
            len(self.merges),
            token_blob_len,
            merge_blob_len,
            offsets_len,
            self.special_ids.pad,
            self.special_ids.unk,
            self.special_ids.bos,
            self.special_ids.eos,
            self.special_ids.system,
            self.special_ids.tool,
            self.special_ids.user,
        )
        f.write(header)


class MMapNeuralFlexTokenizer(NeuralFlexTokenizer):
    """Tokenizer with mmap-backed vocabulary payload for fast startup."""

    def __init__(self, path: Path) -> None:
        self._fd = path.open("rb")
        self._mm = mmap.mmap(self._fd.fileno(), 0, access=mmap.ACCESS_READ)

        header_size = struct.calcsize("<8sI5I7I")
        header = struct.unpack("<8sI5I7I", self._mm[:header_size])
        magic, version = header[0], header[1]
        if magic != MAGIC or version != _VERSION:
            raise ValueError("Invalid tokenizer binary format")

        vocab_size, _merge_count, token_blob_len, merge_blob_len, offsets_len = header[2:7]
        specials = header[7:14]
        special_ids = SpecialTokenIds(*specials)

        cursor = header_size
        offsets_end = cursor + offsets_len * 4
        offsets = struct.unpack(f"<{offsets_len}I", self._mm[cursor:offsets_end])
        cursor = offsets_end
        token_blob = self._mm[cursor : cursor + token_blob_len]
        cursor += token_blob_len
        merge_blob = self._mm[cursor : cursor + merge_blob_len]

        vocab = token_blob.decode("utf-8").split("\n") if vocab_size else []
        if len(vocab) != vocab_size:
            raise ValueError("Corrupt vocabulary table")

        self._offsets = offsets
        self._token_blob = token_blob
        merges = [tuple(pair) for pair in json.loads(merge_blob.decode("utf-8"))]

        super().__init__(vocab=vocab, merges=merges)
        self.special_ids = special_ids

    def close(self) -> None:
        self._mm.close()
        self._fd.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
