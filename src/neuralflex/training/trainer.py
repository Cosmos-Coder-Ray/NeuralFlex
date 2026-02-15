from __future__ import annotations

import json
import math
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW

from neuralflex.config.schemas import ModelConfig, TrainingConfig
from neuralflex.data.pipeline import DataPipeline, DatasetSource
from neuralflex.models.model import NeuralFlexMoEModel
from neuralflex.tokenizer import BPETrainer


class Trainer:
    """Dense LM trainer with deterministic controls and checkpointing."""

    def __init__(self, model_config: ModelConfig, train_config: TrainingConfig) -> None:
        self.model_config = model_config
        self.train_config = train_config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = self._resolve_dtype(train_config.precision)

        self.output_dir = Path(train_config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.output_dir / "metrics.jsonl"

        self._set_seed(train_config.seed)
        self.tokenizer = self._build_tokenizer(train_config.train_path)
        self.model = NeuralFlexMoEModel(model_config).to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=train_config.learning_rate, weight_decay=train_config.weight_decay)
        self.scaler = torch.amp.GradScaler(enabled=self._use_scaler)
        self.global_step = 0

    @property
    def _use_scaler(self) -> bool:
        return self.device.type == "cuda" and self.dtype == torch.float16

    def _resolve_dtype(self, precision: str) -> torch.dtype:
        prec = precision.lower()
        if prec == "bf16":
            return torch.bfloat16
        if prec == "fp16":
            return torch.float16
        return torch.float32

    def _set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _build_tokenizer(self, train_path: str):
        texts: list[str] = []
        with Path(train_path).open("r", encoding="utf-8") as f:
            for line in f:
                text = line.strip()
                if text:
                    texts.append(text)
        trainer = BPETrainer(vocab_size=min(self.model_config.vocab_size, 2048), min_frequency=1)
        tokenizer = trainer.train(texts)
        self.model_config.vocab_size = len(tokenizer.vocab)
        return tokenizer

    def _autocast_context(self):
        enabled = self.device.type == "cuda" and self.dtype in {torch.float16, torch.bfloat16}
        return torch.autocast(device_type=self.device.type, dtype=self.dtype, enabled=enabled)

    def _lr_for_step(self, step: int) -> float:
        cfg = self.train_config
        if step < cfg.warmup_steps:
            return cfg.learning_rate * float(step + 1) / float(max(cfg.warmup_steps, 1))
        progress = (step - cfg.warmup_steps) / float(max(cfg.max_steps - cfg.warmup_steps, 1))
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
        return cfg.min_learning_rate + cosine * (cfg.learning_rate - cfg.min_learning_rate)

    def _build_loader(self, epoch: int):
        source = DatasetSource(path=self.train_config.train_path, format=self.train_config.train_format, text_key=self.train_config.text_key)
        pipeline = DataPipeline(
            sources=[source],
            tokenizer=self.tokenizer,
            sequence_length=self.train_config.sequence_length,
            batch_size=self.train_config.batch_size,
            shuffle_buffer_size=self.train_config.shuffle_buffer_size,
            seed=self.train_config.seed,
        )
        return pipeline.build(epoch=epoch)

    def save_checkpoint(self, name: str) -> None:
        path = self.output_dir / name
        ckpt = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": self.global_step,
            "model_config": asdict(self.model_config),
            "train_config": asdict(self.train_config),
            "tokenizer_vocab": self.tokenizer.vocab,
            "tokenizer_merges": self.tokenizer.merges,
        }
        torch.save(ckpt, path)

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.global_step = ckpt["step"]

    def _log_metrics(self, payload: dict[str, float | int]) -> None:
        with self.metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")

    def train(self) -> None:
        cfg = self.train_config
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        epoch = 0
        running_loss = 0.0
        accumulation = 0
        while self.global_step < cfg.max_steps:
            for batch in self._build_loader(epoch):
                input_ids = torch.from_numpy(batch.input_ids).to(self.device)
                labels = torch.from_numpy(batch.labels).to(self.device)
                attention_mask = torch.from_numpy(batch.attention_mask).to(self.device)

                lr = self._lr_for_step(self.global_step)
                for group in self.optimizer.param_groups:
                    group["lr"] = lr

                with self._autocast_context():
                    out = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                    assert out.loss is not None
                    loss = out.loss / cfg.grad_accum_steps

                if self._use_scaler:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                running_loss += float(loss.item())
                accumulation += 1

                if accumulation < cfg.grad_accum_steps:
                    continue

                if self._use_scaler:
                    self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip_norm)

                if self._use_scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

                self.global_step += 1
                accumulation = 0

                if self.global_step % cfg.log_every == 0:
                    avg_loss = running_loss / cfg.log_every
                    running_loss = 0.0
                    self._log_metrics({"step": self.global_step, "loss": avg_loss, "lr": lr})

                if self.global_step % cfg.save_every == 0:
                    self.save_checkpoint(f"checkpoint_step_{self.global_step}.pt")

                if self.global_step >= cfg.max_steps:
                    break
            epoch += 1

        self.save_checkpoint("final.pt")
