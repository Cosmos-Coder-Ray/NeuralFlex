# NeuralFlex Tokenizer Format and Pipeline

## Overview

NeuralFlex ships an in-repo deterministic BPE tokenizer with:
- train-time BPE merge learning,
- runtime streaming encode/decode,
- compact binary serialization,
- memory-mapped loading for low startup overhead.

Special tokens:
- `<pad>`
- `<unk>`
- `<bos>`
- `<eos>`
- `<system>`
- `<tool>`
- `<user>`

## Deterministic normalization

Normalization is fixed and deterministic:
1. Unicode NFKC normalization
2. leading/trailing trim
3. collapse internal whitespace runs to single spaces

## Binary format (`.nfx`)

Little-endian layout:

1. Header (`<8sI5I7I`)
   - magic: `NFXTK1\0`
   - version: `1`
   - `vocab_size`
   - `merge_count`
   - `token_blob_len`
   - `merge_blob_len`
   - `offsets_len`
   - special token ids in order: pad, unk, bos, eos, system, tool, user
2. Vocabulary offsets table (`offsets_len` x `uint32`)
3. Vocabulary token blob (`\n`-joined UTF-8 token bytes)
4. Merge blob (UTF-8 JSON list of token pairs)

The loader uses `mmap` to map the full file and decode tables without eager copying the on-disk artifact.

## Streaming behavior

- `encode_stream(chunks)` accepts text chunks and only flushes completed words.
- `decode_incremental(token_id)` emits immediate text pieces for autoregressive generation.

## Dataset pipeline integration

`DataPipeline` supports `jsonl`, `txt`, and `parquet` sources, deterministic buffered shuffle, sliding-window sequence generation, and batched tensors (`input_ids`, `labels`, `attention_mask`, `position_ids`).

`PrefetchDataLoader` wraps the pipeline with a multiprocessing prefetch queue for better GPU feed behavior.
