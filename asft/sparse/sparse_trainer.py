"""
⚠️  EXPERIMENTAL — DO NOT USE IN PRODUCTION ⚠️

SparseTrainer — Gradient masking based "sparse" trainer.

STATUS: DEPRECATED, MOVED TO experimental/

WHY THIS DOES NOT PROVIDE REAL SPEEDUP:
    This trainer applies a binary mask to gradients before the optimizer step.
    However, the FORWARD PASS still computes dense matrix multiplications for
    ALL weights, including "masked" ones. Setting a gradient to zero does NOT
    prevent the forward computation that produced it.

    On GPU/CPU:
    - Dense matrix multiply (GEMM) computes ALL rows/columns
    - PyTorch does not skip zero-gradient rows in the backward pass
    - The "sparsity" is cosmetic: memory and compute are unchanged

    The original README claimed 80–90% training speedup from this approach.
    This claim cannot be reproduced because it is mathematically impossible
    without sparse kernel support (cuSPARSE, structured pruning, or sparse hardware).

WHAT TO USE INSTEAD:
    - LoRA / QLoRA via asft.training.peft_trainer — real parameter reduction
    - Dynamic sparse training: asft.sparse.dynamic_sparse.DynamicSparseTrainer
    - Parameter selection: asft.selection.parameter_selector.ParameterSelector

KEPT FOR:
    - Reference comparison in benchmarks
    - Research experimentation
    - Historical documentation

This file is intentionally NOT imported by any production module.
"""

import logging
from typing import Any

import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class SparseTrainer:
    """
    Experimental Sparse Trainer.

    NOTE: The original implementation claimed 80-90% training time reduction by setting
    `requires_grad = False` dynamically. This does NOT actually reduce FLOPs in standard
    dense matrix multiplications without custom sparse CUDA kernels.

    This class is retained for compatibility and research purposes, but the critical
    step calculation bug has been fixed.
    """

    def __init__(self, model, optimizer, config: dict[str, Any]):
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.sparsity_ratio = config.get("sparsity_ratio", 0.95)

    def _compute_masks(self, threshold: float) -> dict[str, torch.Tensor]:
        """Compute boolean masks for gradients based on magnitude."""
        masks = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                mask = torch.abs(param.grad) > threshold
                masks[name] = mask
        return masks

    def _apply_masks(self, masks: dict[str, torch.Tensor]):
        """Zero out gradients where mask is False."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and name in masks:
                param.grad.data.mul_(masks[name])

    def train(self, dataloader: DataLoader, max_steps: int = 500) -> dict[str, Any]:
        """
        Run the training loop with experimental sparsity.

        BUGFIX: The original code calculated `total_steps = min(max_steps, len(dataloader) * 100)`
        which falsely inflated the steps by 100x. It is now corrected.
        """
        self.model.train()
        self.model.to(self.device)

        # FIXED: Remove the * 100 multiplier
        total_steps = min(max_steps, len(dataloader))
        logger.info(
            "Starting sparse training run: steps=%d, sparsity=%.2f",
            total_steps,
            self.sparsity_ratio,
        )

        losses: list[float] = []
        step = 0

        # We need an iterator to manually step through
        dl_iter = iter(dataloader)

        while step < total_steps:
            try:
                batch = next(dl_iter)
            except StopIteration:
                # Refresh dataloader if we exhaust it before max_steps
                dl_iter = iter(dataloader)
                batch = next(dl_iter)

            # Move batch to device
            if isinstance(batch, dict):
                batch = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in batch.items()}
            elif isinstance(batch, (list, tuple)):
                batch = [v.to(self.device) if hasattr(v, "to") else v for v in batch]
            else:
                batch = batch.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(**batch) if isinstance(batch, dict) else self.model(*batch)
            loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]

            # Backward pass
            loss.backward()

            # Apply sparsity mask to gradients
            # Note: This does not save compute during forward/backward pass,
            # it only zeros out updates, which is why original claims were misleading.
            if self.sparsity_ratio > 0.0:
                all_grads = torch.cat(
                    [
                        p.grad.flatten()
                        for p in self.model.parameters()
                        if p.requires_grad and p.grad is not None
                    ]
                )
                if len(all_grads) > 0:
                    threshold = torch.quantile(torch.abs(all_grads), self.sparsity_ratio).item()
                    masks = self._compute_masks(threshold)
                    self._apply_masks(masks)

            self.optimizer.step()

            losses.append(loss.item())
            step += 1

            if step % 10 == 0:
                logger.debug("Step %d/%d - loss: %.4f", step, total_steps, loss.item())

        final_loss = losses[-1] if losses else 0.0
        avg_loss = sum(losses) / len(losses) if losses else 0.0

        return {
            "status": "completed",
            "steps": step,
            "final_loss": final_loss,
            "average_loss": avg_loss,
        }
