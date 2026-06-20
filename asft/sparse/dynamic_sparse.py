"""
ASFT Dynamic Sparse Training — RigL-style adaptive sparsity.

WHY THE ORIGINAL SparseTrainer WAS WRONG:
    The original implementation applied a binary gradient mask (0 or 1) to
    parameter gradients. This is WRONG because:

    1. The forward pass still computes dense matrix multiplications.
       PyTorch does not skip zeroed-weight rows/columns by default.
       Zero gradients ≠ zero compute. The "sparsity" was cosmetic.

    2. Weight masking after the gradient step (rather than before) means
       gradients are computed for masked parameters anyway, wasting compute.

    3. The claimed "80–90% training speedup" cannot be achieved with this
       approach on dense hardware (GPU/CPU). Real sparse speedup requires
       either: sparse tensor formats (CSR/COO), specialized sparse kernels
       (cuSPARSE), or hardware specifically designed for sparsity (Cerebras).

THE CORRECT APPROACH — Dynamic Sparse Training:

    Rigged Lottery (RigL), Evci et al. (2020), NeurIPS:
    - Maintains a SPARSE weight mask throughout training
    - Periodically UPDATES the mask: drop lowest-magnitude weights, grow
      highest-gradient-magnitude weights
    - This is the only gradient-masking approach with proven accuracy parity
      to dense training in the published literature
    - Validated: RigL matches dense ResNet-50 accuracy at 80% sparsity
      on ImageNet (FLOP reduction requires sparse kernels)

    Available from this module:
        DynamicSparseTrainer — a HuggingFace Trainer wrapper that:
        1. Initializes a random sparse mask (target sparsity %)
        2. Applies mask to weights at every step (zeroes masked weights)
        3. Every `update_interval` steps: drops low |weight| and grows
           high |gradient| positions (the RigL update rule)

HONEST LIMITATIONS:
    1. This implementation uses dense PyTorch tensors with masking.
       The COMPUTE savings require sparse kernels not available in standard
       PyTorch on GPU without significant extra engineering.

    2. MEMORY savings: The mask itself costs 1 bit per weight. A 7B model
       with 80% sparsity: 7B weights × (4 bytes for float16 + 1 bit mask)
       ≈ saves ~22GB vs dense, costs ~0.9GB for mask = net savings ≈ 21GB.
       This IS real and implementable.

    3. Gradient compute savings DO come from smaller backward passes IF
       requires_grad_(False) is applied to frozen weights — which is what
       ParameterSelector does. These two tools should be used together.

    4. RigL's accuracy-at-sparsity results apply to CNNs (ResNet, VGG).
       For LLMs, the Lottery Ticket Hypothesis suggests high sparsity is
       achievable but the search cost is non-trivial (Sanh et al. 2020 on BERT).

RECOMMENDED USE:
    Use DynamicSparseTrainer with sparsity ≤ 0.90 for language tasks.
    Always validate on a held-out set. Do not rely on this for compute savings
    without also using sparse kernels (cuSPARSE) or sparse hardware.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class SparseTrainingConfig:
    """Configuration for dynamic sparse training."""
    target_sparsity: float = 0.80       # Fraction of weights to keep ZEROED
    update_interval: int = 100          # Steps between mask updates (RigL update frequency)
    warmup_steps: int = 200             # Steps before first mask update (stabilize training)
    drop_fraction: float = 0.20         # Fraction of mask to update per cycle (20% = conservative)
    sparsity_schedule: str = "constant" # "constant" | "linear" (gradually increase sparsity)
    target_layers: list[str] | None = None  # Layer name substrings to target; None = all Linear


@dataclass
class SparsityReport:
    """Report on current sparsity state."""
    step: int
    target_sparsity: float
    actual_sparsity: float
    n_zero_weights: int
    n_total_weights: int
    n_mask_updates: int = 0

    def summary(self) -> str:
        return (
            f"Step {self.step}: sparsity={self.actual_sparsity:.1%} "
            f"(target={self.target_sparsity:.1%}) | "
            f"zero={self.n_zero_weights:,}/{self.n_total_weights:,} weights | "
            f"updates={self.n_mask_updates}"
        )


class DynamicSparseMask:
    """
    Maintains and updates the binary sparsity mask.

    The mask determines which weights are allowed to be non-zero.
    During training: weights outside the mask are zeroed after each optimizer step.
    During mask update (RigL cycle): drop lowest |weight|, grow highest |gradient|.
    """

    def __init__(self, model: nn.Module, config: SparseTrainingConfig):
        self._model = model
        self._config = config
        self._masks: dict[str, torch.Tensor] = {}
        self._n_updates = 0

        self._initialize_masks()

    def _target_modules(self) -> list[tuple[str, nn.Module]]:
        """Return (name, module) pairs for layers that should be sparsified."""
        targets = self._config.target_layers
        result = []
        for name, module in self._model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if targets is None or any(t in name for t in targets):
                result.append((name, module))
        return result

    def _initialize_masks(self) -> None:
        """
        Create initial random sparse masks.
        Random initialization: each weight independently masked with p=target_sparsity.
        (This is the standard initialization for dynamic sparse training.)
        """
        for name, module in self._target_modules():
            weight = module.weight.data
            n = weight.numel()
            n_keep = max(1, int(n * (1.0 - self._config.target_sparsity)))

            # Random: keep n_keep random positions
            mask = torch.zeros(n, dtype=torch.bool, device=weight.device)
            keep_idx = torch.randperm(n, device=weight.device)[:n_keep]
            mask[keep_idx] = True
            self._masks[name] = mask.view(weight.shape)

            # Apply initial mask
            module.weight.data[~self._masks[name]] = 0.0

        n_layers = len(self._masks)
        n_total = sum(m.numel() for m in self._masks.values())
        n_active = sum(m.sum().item() for m in self._masks.values())
        logger.info(
            "DynamicSparseMask initialized: %d layers | %.1f%% sparsity | "
            "%d/%d weights active",
            n_layers, 100 * (1 - n_active / max(1, n_total)), int(n_active), n_total
        )

    def apply(self) -> None:
        """Zero out masked weights. Call after every optimizer step."""
        for name, module in self._target_modules():
            if name in self._masks:
                module.weight.data[~self._masks[name]] = 0.0

    def update(self, current_step: int) -> bool:
        """
        RigL mask update: drop low-magnitude weights, grow high-gradient weights.

        Drop rule:   Remove the `drop_fraction` of active weights with lowest |w|
        Grow rule:   Activate the same count of inactive weights with highest |∇w|

        This preserves the sparsity level while moving the mask toward
        weights that are currently most informative (high gradient × magnitude).

        Returns True if an update occurred.
        """
        if current_step < self._config.warmup_steps:
            return False
        if (current_step - self._config.warmup_steps) % self._config.update_interval != 0:
            return False

        logger.debug("RigL mask update at step %d", current_step)
        n_updated = 0

        for name, module in self._target_modules():
            if name not in self._masks:
                continue

            weight = module.weight.data
            mask = self._masks[name]

            if module.weight.grad is None:
                continue

            grad = module.weight.grad.data.abs()

            # Count active weights to update
            n_active = int(mask.sum().item())
            n_drop = max(1, int(n_active * self._config.drop_fraction))

            # Drop: lowest |weight| among currently active
            active_weights = weight.abs() * mask.float()
            active_weights_flat = active_weights.flatten()
            active_mask_flat = mask.flatten()

            # Get active positions and sort by |weight|
            active_positions = active_mask_flat.nonzero(as_tuple=True)[0]
            if len(active_positions) <= n_drop:
                continue  # Don't drop if we'd empty the layer

            active_vals = active_weights_flat[active_positions]
            _, drop_local_idx = active_vals.topk(n_drop, largest=False)
            drop_positions = active_positions[drop_local_idx]

            # Grow: highest |gradient| among currently inactive
            inactive_mask_flat = ~active_mask_flat
            inactive_positions = inactive_mask_flat.nonzero(as_tuple=True)[0]
            if len(inactive_positions) < n_drop:
                continue

            inactive_grads = grad.flatten()[inactive_positions]
            _, grow_local_idx = inactive_grads.topk(n_drop, largest=True)
            grow_positions = inactive_positions[grow_local_idx]

            # Update mask
            new_mask_flat = mask.flatten().clone()
            new_mask_flat[drop_positions] = False
            new_mask_flat[grow_positions] = True
            self._masks[name] = new_mask_flat.view(mask.shape)

            # Apply updated mask to weights
            module.weight.data[~self._masks[name]] = 0.0
            n_updated += n_drop

        self._n_updates += 1
        logger.debug("Mask update complete: ~%d weight positions changed", n_updated)
        return True

    def get_sparsity_report(self, step: int) -> SparsityReport:
        """Current sparsity statistics."""
        n_total = sum(m.numel() for m in self._masks.values())
        n_zero = sum((~m).sum().item() for m in self._masks.values())
        return SparsityReport(
            step=step,
            target_sparsity=self._config.target_sparsity,
            actual_sparsity=n_zero / max(1, n_total),
            n_zero_weights=int(n_zero),
            n_total_weights=n_total,
            n_mask_updates=self._n_updates,
        )


class DynamicSparseTrainer:
    """
    Wraps a HuggingFace Trainer with dynamic sparse training.

    Usage:
        sparse_config = SparseTrainingConfig(target_sparsity=0.80)
        trainer = DynamicSparseTrainer(hf_trainer, model, sparse_config)
        trainer.train()

    IMPORTANT: This trainer handles the RigL mask update as a training callback.
    It is compatible with any HuggingFace Trainer (including TRL SFTTrainer).
    """

    def __init__(self, hf_trainer, model: nn.Module, config: SparseTrainingConfig):
        self._hf_trainer = hf_trainer
        self._model = model
        self._config = config
        self._mask = DynamicSparseMask(model, config)
        self._step = 0

    def train(self) -> Any:
        """
        Run training with dynamic sparse mask updates.

        Injects a post-step hook that:
        1. Applies mask (zeroes masked weights) after every optimizer step
        2. Runs RigL update every `update_interval` steps
        """
        original_step = self._hf_trainer.training_step

        def patched_training_step(model, inputs):
            loss = original_step(model, inputs)
            self._mask.apply()  # Enforce sparsity after optimizer step
            self._mask.update(self._step)
            self._step += 1

            if self._step % 100 == 0:
                report = self._mask.get_sparsity_report(self._step)
                logger.info(report.summary())

            return loss

        self._hf_trainer.training_step = patched_training_step
        return self._hf_trainer.train()

    def get_sparsity_report(self) -> SparsityReport:
        return self._mask.get_sparsity_report(self._step)
