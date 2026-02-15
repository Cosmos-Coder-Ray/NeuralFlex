# NeuralFlex SLM-MoE Architecture and Implementation Roadmap

## 1) Product Objectives and Constraints

### Objective
Deliver a production-grade Small Language Model (SLM) with Mixture-of-Experts (MoE) sparsity that reaches **~1B-4B dense-equivalent capability** while remaining deployable on:
- 8GB-16GB VRAM consumer GPUs (RTX 3060/4060-class and above)
- Modern desktop/laptop CPUs (AVX2/AVX-512 where available)

### Hard Deployment Constraints
- Offline local inference, no runtime cloud dependency.
- Mixed precision support (FP16/BF16 inference + training path).
- Batch-size=1 latency optimized serving mode.
- Quantized inference path required.
- Avoid framework lock-in (PyTorch-first, ONNX path, Rust kernel hooks).

### Design Philosophy
- Prioritize practical systems choices over novelty.
- Treat throughput, memory footprint, and stability as first-class constraints.
- Keep architecture modular for future agentic extensions (tools, memory, planners).

---

## 2) System Architecture Decision

## 2.1 Backbone Transformer
- Decoder-only transformer optimized for autoregressive generation.
- **Pre-Norm RMSNorm** for stability at smaller scale and mixed precision.
- **SwiGLU/GEGLU** activations for better parameter efficiency than ReLU.
- Rotary Position Embeddings (RoPE) with configurable scaling for longer contexts.

## 2.2 Attention Strategy (Local Efficiency First)
Chosen stack:
1. **Grouped-Query Attention (GQA)** to reduce KV cache size.
2. **Flash Attention path** when CUDA backend supports it.
3. **Sliding-window attention mode** for long-context low-memory inference.

Rationale:
- GQA directly lowers memory pressure and token latency under batch=1.
- Flash path provides major speedups on consumer NVIDIA GPUs.
- Sliding window bounds quadratic memory growth for CPU and low-VRAM cases.

Default initial settings:
- hidden size: 2048
- query heads: 16
- KV heads: 4 (GQA ratio 4:1)
- head dim: 128
- default context: 4k tokens; optional 8k with windowed mode

## 2.3 MoE Block Design
- Replace dense FFN with **Top-2 routed MoE FFN** in selected layers.
- Expert FFN uses SwiGLU.
- Shared expert optional (one always-on expert) to improve robustness.
- Capacity factor and token dropping enabled for predictable memory/latency.

Proposed v1 layout:
- 24 layers total
- MoE in 16 layers (interleaved with dense FFN layers)
- 8 experts per MoE layer
- top-k = 2

This offers strong sparse capacity while limiting routing overhead.

## 2.4 Routing Algorithm
Router:
- Linear gating projection from hidden state to expert logits.
- **Top-k selection with straight-through dispatch**.
- **Auxiliary load-balance loss** + z-loss/logit regularization.
- **Router jitter/noise during training** to reduce collapse.

Expert collapse prevention:
- Capacity-aware dispatch per expert.
- Load-balancing coefficient schedule (higher warmup, taper later).
- Monitoring metrics in training loop (entropy, utilization, dropped tokens).

---

## 3) Parameter Scale Targets

## 3.1 Target Model Families

### NeuralFlex-SLM-MoE-1 (primary first milestone)
- hidden: 2048
- layers: 24
- experts/layer: 8 (on MoE layers)
- top-k: 2
- dense-equivalent estimate: ~2.0B-2.8B
- active parameters/token: ~0.9B-1.2B equivalent compute

### NeuralFlex-SLM-MoE-2 (second milestone)
- hidden: 2560
- layers: 28
- experts/layer: 8-12
- top-k: 2
- dense-equivalent estimate: ~3B-4B+
- designed for 12-16GB VRAM quantized inference

## 3.2 Memory Budget Targets
- FP16/BF16 inference (batch=1, 4k context): fit in <=16GB.
- 4-bit quantized inference target: <=8GB for primary model.
- KV cache with GQA and paged cache options to control memory growth.

---

## 4) Training Strategy

## 4.1 Training Phases
1. **Phase A (stabilization pretrain)**
   - start with shorter context (2k) and reduced MoE intensity.
   - router warmup with stronger balance loss.
2. **Phase B (main pretraining)**
   - full context 4k, production MoE config.
   - curriculum over sequence length and data mix.
3. **Phase C (alignment/finetune readiness)**
   - instruction-tuning compatible checkpoints and tokenizer freeze.

## 4.2 Optimization
- Optimizer: AdamW (fused where available).
- LR schedule: cosine decay with warmup.
- Gradient clipping mandatory.
- Mixed precision: BF16 preferred, FP16 fallback.
- Gradient checkpointing + activation recompute to reduce VRAM.
- ZeRO/FSDP optional plugin; single-node DDP baseline.

## 4.3 Stability Controls
- RMSNorm everywhere.
- Router z-loss and entropy tracking.
- Expert capacity factor tuning (start 1.25-1.5).
- Automatic loss scaling for FP16.
- NaN/Inf watchdog and checkpoint rollback hooks.

---

## 5) Dataset Pipeline

## 5.1 Data Sources (curated local/offline friendly)
- permissive web + code + books + QA mixtures.
- explicit provenance metadata stored per shard.

## 5.2 Data Processing Stages
1. Ingestion (jsonl/parquet/text)
2. Deduplication (document-level + near-duplicate MinHash/SimHash)
3. Quality filtering (language, toxicity, malformed text, boilerplate)
4. Tokenization to packed sequences
5. Sharding into streaming-ready mmap/arrow format

## 5.3 Tokenizer Integration
- In-repo tokenizer package with:
  - trainable BPE/Unigram training script
  - runtime pure-Python + optional Rust accelerated implementation
  - serialized vocab/merges shipped with checkpoints
- No mandatory network runtime dependency.

---

## 6) Evaluation Framework

## 6.1 Core Quality Metrics
- Perplexity on held-out corpora.
- Task suite (commonsense, reasoning-lite, coding-lite, factual QA).
- Instruction following score for aligned checkpoints.

## 6.2 MoE Health Metrics
- Expert utilization histogram.
- Router entropy.
- Dropped-token rate due to capacity.
- Load-balance auxiliary loss trends.

## 6.3 Systems Metrics (must-pass)
- Tokens/sec generation at batch=1.
- First-token latency.
- Peak VRAM / RAM usage.
- Quantized-vs-fp quality delta.

---

## 7) Inference Optimization Plan (Local Device Focus)

## 7.1 Runtime Modes
- **Latency mode** (batch=1, max responsiveness):
  - fused kernels, static cache buffers, minimal scheduler overhead.
- **Throughput mode** (small batch multi-stream):
  - dynamic batching optional.

## 7.2 Quantization Path
- PTQ baseline: 8-bit weights + 16-bit activations.
- Target path: 4-bit weight-only quantization for linear/expert projections.
- Calibration dataset integrated in tooling.
- Quantization-aware checkpoints compatibility hooks for future QAT.

## 7.3 Cache + Attention Optimizations
- GQA KV compression by design.
- Paged KV cache abstraction for long sessions.
- Sliding-window decode option to cap memory.

## 7.4 Backend Portability
- PyTorch eager + torch.compile baseline.
- ONNX export with graph simplification + provider-specific kernels.
- Rust kernel crate reserved for critical ops (router dispatch, quant GEMM, cache ops).

---

## 8) Modular Extension Plan (Agentic-Ready)

The architecture separates:
- Core model definition
- Inference runtime/session manager
- Tool-augmented orchestration adapters

Future agent extensions can attach via:
- structured generation APIs
- function-call schema layers
- memory/tool routers without changing core transformer code.

---

## 9) Project Structure

```text
NeuralFlex/
  configs/
    model/
    train/
    inference/
  docs/
    architecture.md
    roadmap.md
  src/neuralflex/
    config/
    models/
    moe/
    attention/
    training/
    data/
    tokenizer/
    inference/
    export/
    eval/
    utils/
  rust/neuralflex-kernels/
    src/
  scripts/
  tests/
```

---

## 10) Implementation Roadmap

## Milestone 0: Repo Scaffolding (current)
- Create modular package layout.
- Add config schemas and interface stubs.
- Add architecture document and roadmap baseline.

## Milestone 1: Core Model Skeleton
- Implement decoder blocks with pluggable attention and FFN/MoE layers.
- Implement RoPE + RMSNorm + SwiGLU modules.
- Add config-driven model builder.

## Milestone 2: MoE Routing and Stability
- Implement top-k router, dispatch/combine, capacity and aux losses.
- Add expert utilization telemetry.
- Add unit tests for routing correctness and collapse prevention signals.

## Milestone 3: Training Stack
- Build dataloader + packed sequence pipeline.
- Add trainer loop with AMP, grad accumulation, checkpointing.
- Add distributed-ready abstractions.

## Milestone 4: Inference Runtime
- Implement streaming generation APIs and KV-cache manager.
- Add latency-optimized decode loop.
- Add quantized inference integration and benchmarks.

## Milestone 5: Export + Deployment
- ONNX export pipeline with validation.
- Runtime compatibility matrix (CUDA, CPU).
- CLI/server local serving package.

## Milestone 6: Rust Kernel Acceleration
- Implement high-impact kernels in Rust (router dispatch/quant matmul/cache).
- Bindings and fallback logic.
- Benchmark and A/B test against PyTorch baseline.

---

## 11) Risks and Mitigations

- **Expert collapse** -> stronger aux loss warmup + router noise + health dashboards.
- **Low-VRAM instability** -> strict memory budget tests + activation checkpointing.
- **Quantization quality drop** -> calibration curation + per-layer sensitivity fallback.
- **Backend divergence** -> golden tests for PyTorch vs ONNX outputs.

---

## 12) Definition of Done (Phase 1)

A checkpoint is Phase-1 production-candidate when:
- Runs offline local inference on 8-16GB targets.
- Supports streaming decode and batch=1 latency mode.
- Supports mixed precision and quantized path.
- Exposes ONNX export path with correctness tests.
- Demonstrates stable MoE utilization without sustained collapse.

