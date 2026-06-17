"""
Sparse Trainer — Core training loop for ASFT sparse fine-tuning.
Applies a SparseSelectionMask, runs the training loop, and saves
delta checkpoints separately from the base model.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    epoch: int = 0
    step: int = 0
    train_loss: float = 0.0
    eval_loss: float = 0.0
    learning_rate: float = 0.0
    trainable_params: int = 0
    total_params: int = 0
    sparsity: float = 0.0
    duration_seconds: float = 0.0
    history: List[Dict[str, Any]] = field(default_factory=list)

    def log_step(self) -> None:
        entry = {
            "step": self.step,
            "train_loss": round(self.train_loss, 6),
            "eval_loss": round(self.eval_loss, 6),
            "lr": self.learning_rate,
        }
        self.history.append(entry)

    def summary(self) -> str:
        return (
            f"Step {self.step} | loss={self.train_loss:.4f} | eval_loss={self.eval_loss:.4f} "
            f"| sparsity={self.sparsity:.2%} | time={self.duration_seconds:.1f}s"
        )


class SparseTrainer:
    """
    Trains only the selected sparse subset of parameters.

    Workflow:
      1. Apply SparseSelectionMask (freeze all others)
      2. Run standard AdamW training loop with gradient accumulation
      3. Save only the delta (changed params) to delta_output_dir
      4. Optionally evaluate every eval_steps
    """

    def __init__(
        self,
        model: nn.Module,
        config,  # SparseTrainingConfig
        mask=None,  # SparseSelectionMask
        device: Optional[str] = None,
    ):
        self._model = model
        self._config = config
        self._mask = mask
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device)

        # Snapshot base weights for delta computation
        self._base_weights: Dict[str, torch.Tensor] = {}

    def _apply_mask(self) -> None:
        if self._mask is None:
            logger.info("No mask provided — training all parameters")
            return
        for name, param in self._model.named_parameters():
            should_train = any(name.startswith(t) or t in name
                               for t in self._mask.trainable_params)
            param.requires_grad_(should_train)

    def _snapshot_base(self) -> None:
        """Save base weights for delta computation after training."""
        self._base_weights = {
            name: param.data.clone().cpu()
            for name, param in self._model.named_parameters()
            if param.requires_grad
        }

    def _build_optimizer(self, lr: float) -> torch.optim.Optimizer:
        trainable = [p for p in self._model.parameters() if p.requires_grad]
        return torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)

    def _build_scheduler(self, optimizer, num_training_steps: int):
        from transformers import get_cosine_schedule_with_warmup
        warmup = int(num_training_steps * self._config.warmup_ratio)
        return get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup,
            num_training_steps=num_training_steps,
        )

    def train(
        self,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
    ) -> TrainingMetrics:
        """Run the sparse training loop. Returns final metrics."""
        self._apply_mask()
        self._snapshot_base()

        trainable_params = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self._model.parameters())
        sparsity = 1.0 - trainable_params / max(1, total_params)

        logger.info(
            "SparseTrainer: trainable=%d/%d (%.2f%% frozen) | device=%s",
            trainable_params, total_params, sparsity * 100, self._device
        )

        cfg = self._config
        optimizer = self._build_optimizer(cfg.learning_rate)
        total_steps = min(cfg.max_steps, len(train_dataloader) * 100)
        scheduler = self._build_scheduler(optimizer, total_steps)

        metrics = TrainingMetrics(
            trainable_params=trainable_params,
            total_params=total_params,
            sparsity=sparsity,
        )

        start_time = time.time()
        self._model.train()
        global_step = 0
        accum_loss = 0.0

        progress = tqdm(total=total_steps, desc="ASFT Sparse Training")

        for batch in train_dataloader:
            if global_step >= total_steps:
                break

            # Move batch to device
            if isinstance(batch, dict):
                batch = {
                    k: v.to(self._device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                outputs = self._model(**batch)
                loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
            else:
                inputs = batch[0].to(self._device)
                labels = batch[1].to(self._device) if len(batch) > 1 else inputs
                outputs = self._model(inputs, labels=labels)
                loss = outputs[0]

            loss = loss / cfg.gradient_accumulation_steps
            loss.backward()
            accum_loss += float(loss.item())

            if (global_step + 1) % cfg.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    self._model.parameters(), cfg.max_grad_norm
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                metrics.step = global_step
                metrics.train_loss = accum_loss
                metrics.learning_rate = scheduler.get_last_lr()[0]
                accum_loss = 0.0

                progress.set_postfix(loss=f"{metrics.train_loss:.4f}")
                progress.update(1)

                if global_step % cfg.eval_steps == 0 and eval_dataloader:
                    metrics.eval_loss = self._evaluate(eval_dataloader)
                    logger.info(metrics.summary())

                if global_step % cfg.save_steps == 0:
                    self._save_delta(global_step)

                metrics.log_step()

            global_step += 1

        progress.close()
        metrics.duration_seconds = round(time.time() - start_time, 2)

        # Final save
        self._save_delta("final")
        logger.info("Training complete: %s", metrics.summary())
        return metrics

    def _evaluate(self, eval_dataloader: DataLoader) -> float:
        self._model.eval()
        total_loss = 0.0
        n = 0
        with torch.no_grad():
            for batch in eval_dataloader:
                if n >= 20:
                    break
                if isinstance(batch, dict):
                    batch = {
                        k: v.to(self._device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()
                    }
                    outputs = self._model(**batch)
                    loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
                else:
                    inputs = batch[0].to(self._device)
                    outputs = self._model(inputs)
                    loss = outputs[0]
                total_loss += float(loss.item())
                n += 1
        self._model.train()
        return total_loss / max(1, n)

    def _save_delta(self, step) -> None:
        """Save only changed parameters as a sparse delta checkpoint."""
        if not self._base_weights:
            return

        output_dir = Path(self._config.delta_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        delta_path = output_dir / f"delta_step_{step}.pt"

        delta: Dict[str, torch.Tensor] = {}
        for name, param in self._model.named_parameters():
            if name in self._base_weights:
                diff = param.data.cpu() - self._base_weights[name]
                # Only store if there's actual change (norm > threshold)
                if diff.norm() > 1e-8:
                    delta[name] = diff

        torch.save(delta, delta_path)
        logger.info("Delta saved: %s (%d tensors)", delta_path, len(delta))

    def load_delta(self, delta_path: str) -> None:
        """Apply a previously saved delta onto the current model weights."""
        delta = torch.load(delta_path, map_location=self._device)
        with torch.no_grad():
            for name, param in self._model.named_parameters():
                if name in delta:
                    param.data.add_(delta[name].to(self._device))
        logger.info("Delta loaded from %s (%d tensors applied)", delta_path, len(delta))
