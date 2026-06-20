"""
ASFT Elastic Weight Consolidation — Continual Learning Without Forgetting.

PROBLEM: CATASTROPHIC FORGETTING
    When a neural network is fine-tuned on Task B after training on Task A,
    it rapidly forgets Task A. This is because gradient descent on Task B
    pushes weights to a completely new minimum, moving away from the
    Task A minimum. This is called "catastrophic forgetting" (McCloskey 1989).

    For ASFT, this means: if you fine-tune a model on your domain dataset,
    it may forget general capabilities from pretraining (language, reasoning).

THE EWC SOLUTION (Kirkpatrick et al. 2017, PNAS):
    Elastic Weight Consolidation adds a regularization term to the loss that
    penalizes changes to parameters that were IMPORTANT for previous tasks:

    L_EWC = L_B(θ)  +  λ/2 · Σ_i F_i (θ_i - θ*_i)²

    Where:
        L_B(θ)       = loss on new task B
        θ*_i         = parameter values after Task A (anchor point)
        F_i          = Fisher Information for parameter i (from Task A)
        λ            = EWC regularization strength (hyperparameter)

    F_i (Fisher Information diagonal) measures how IMPORTANT parameter i
    was for Task A. High F_i → this parameter is critical for Task A →
    penalize changes to it heavily.

FISHER INFORMATION DIAGONAL:
    F_i ≈ E[(∂ log p(y|x, θ)/∂θ_i)²]
         ≈ mean(grad_i²) over the Task A dataset

    This is the same quantity used in ParameterSelector for importance ranking.
    We compute it by running several forward-backward passes on Task A data
    and averaging the squared gradients.

VALIDATED RESULTS (Kirkpatrick et al. 2017):
    - Permuted MNIST: EWC retains Task A accuracy at 97% while learning Task B
      (vs. 50% without EWC after Task B training)
    - Atari games: EWC retains performance on 10 Atari tasks simultaneously
    - BERT continual learning (Ke et al. 2021): EWC achieves competitive
      performance vs task-specific fine-tuning on sequential NLP tasks

WHEN IT FAILS:
    - Very large λ: over-constrains the network, preventing Task B learning
    - Very small λ: insufficient regularization, forgetting occurs anyway
    - Many sequential tasks: Fisher information accumulates across all previous
      tasks; memory grows linearly with task count (use online EWC for long sequences)
    - Capacity bottleneck: if the model is too small to hold both tasks,
      EWC cannot help — the trade-off is fundamental

WHEN TO USE EWC vs. REPLAY vs. LoRA ADAPTERS:
    - EWC: best for sequential task learning with same model architecture
    - Experience Replay: simpler, often equally effective if Task A data available
    - LoRA per-task adapters: best isolation; can switch between tasks without
      forgetting (see replay_buffer.py for adapter-based approach)

HYPERPARAMETER GUIDANCE:
    λ (ewc_lambda):
        - Text classification: λ ∈ [100, 1000]
        - Language generation: λ ∈ [1000, 5000]
        - Start with 1000 and tune based on Task A performance after Task B training
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class EWCConfig:
    """Configuration for EWC continual learning."""

    ewc_lambda: float = 1000.0  # Regularization strength λ
    n_fisher_samples: int = 200  # Batches to estimate Fisher Information
    online_ewc: bool = False  # Online EWC (single running Fisher) vs standard
    fisher_estimation_batch_size: int = 4


class EWCRegularizer:
    """
    Computes and applies the EWC regularization penalty.

    Workflow:
        1. After completing Task A, call compute_fisher(task_a_dataloader)
        2. During Task B training, add ewc_loss() to the standard loss
        3. Optionally, after Task B, update_after_task() to add Task B to the
           "protected" set (for sequential multi-task learning)

    Usage:
        ewc = EWCRegularizer(model, EWCConfig(ewc_lambda=1000))
        ewc.compute_fisher(task_a_dataloader)  # After task A

        # During task B training loop:
        loss = cross_entropy_loss + ewc.ewc_loss()
        loss.backward()
    """

    def __init__(self, model: nn.Module, config: EWCConfig):
        self._model = model
        self._config = config
        self._fisher: dict[str, torch.Tensor] = {}
        self._anchors: dict[str, torch.Tensor] = {}
        self._task_count = 0

    def compute_fisher(
        self,
        dataloader,
        task_name: str = "task_0",
    ) -> dict[str, float]:
        """
        Estimate the Fisher Information diagonal for the current task.
        Must be called AFTER training on Task A is complete.

        The model parameters at this point become the "anchor" θ*
        for EWC regularization on future tasks.

        Args:
            dataloader:  DataLoader with Task A data (used for Fisher estimation).
            task_name:   Optional name for logging.

        Returns:
            Dict of {param_name: mean_fisher_value} for inspection.
        """
        logger.info("Computing Fisher Information for '%s'...", task_name)
        self._model.eval()

        # Initialize Fisher accumulator
        fisher_accum: dict[str, torch.Tensor] = {
            name: torch.zeros_like(param)
            for name, param in self._model.named_parameters()
            if param.requires_grad
        }

        n_batches = 0
        max_batches = self._config.n_fisher_samples

        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break
            try:
                self._model.zero_grad()
                loss = self._compute_loss(batch)
                loss.backward()

                for name, param in self._model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        fisher_accum[name] += param.grad.pow(2)

                n_batches += 1
            except Exception as e:
                logger.warning("Fisher batch %d failed: %s", i, e)
                self._model.zero_grad()

        if n_batches == 0:
            logger.error("Fisher estimation failed: no valid batches")
            return {}

        # Normalize by number of batches
        for name in fisher_accum:
            fisher_accum[name] /= n_batches

        # Save anchor parameters and Fisher values
        if self._config.online_ewc and self._task_count > 0:
            # Online EWC: update running Fisher with γ = 0.9
            for name in self._fisher:
                self._fisher[name] = 0.9 * self._fisher[name] + 0.1 * fisher_accum[name]
        else:
            self._fisher = fisher_accum

        self._anchors = {
            name: param.detach().clone()
            for name, param in self._model.named_parameters()
            if param.requires_grad
        }

        self._task_count += 1

        # Compute summary statistics for logging
        mean_fisher = {name: float(f.mean()) for name, f in self._fisher.items()}
        max_fisher_param = max(mean_fisher, key=mean_fisher.get) if mean_fisher else "none"
        logger.info(
            "Fisher computed: %d params | n_batches=%d | " "top param: %s (%.4f)",
            len(self._fisher),
            n_batches,
            max_fisher_param,
            mean_fisher.get(max_fisher_param, 0),
        )
        return mean_fisher

    def ewc_loss(self) -> torch.Tensor:
        """
        Compute the EWC penalty term.

        L_EWC = λ/2 · Σ_i F_i (θ_i - θ*_i)²

        This should be ADDED to the task-specific loss during Task B training.
        The penalty grows when parameters deviate from Task A values,
        weighted by their Fisher importance.

        Returns:
            Scalar tensor (0.0 if Fisher not yet computed).
        """
        if not self._fisher:
            return torch.tensor(0.0)

        penalty = torch.tensor(0.0, device=next(self._model.parameters()).device)

        for name, param in self._model.named_parameters():
            if name not in self._fisher or name not in self._anchors:
                continue
            F_i = self._fisher[name].to(param.device)
            anchor = self._anchors[name].to(param.device)
            penalty = penalty + (F_i * (param - anchor).pow(2)).sum()

        return self._config.ewc_lambda / 2.0 * penalty

    def has_fisher(self) -> bool:
        """True if Fisher has been computed for at least one task."""
        return len(self._fisher) > 0

    def save(self, path: str) -> None:
        """Save Fisher and anchors to disk for resuming."""
        out = {
            "fisher": {k: v.cpu() for k, v in self._fisher.items()},
            "anchors": {k: v.cpu() for k, v in self._anchors.items()},
            "task_count": self._task_count,
        }
        torch.save(out, path)
        logger.info("EWC state saved to %s", path)

    def load(self, path: str) -> None:
        """Load Fisher and anchors from disk."""
        state = torch.load(path, map_location="cpu")
        self._fisher = state["fisher"]
        self._anchors = state["anchors"]
        self._task_count = state.get("task_count", 1)
        logger.info("EWC state loaded: %d tasks, %d params", self._task_count, len(self._fisher))

    def _compute_loss(self, batch) -> torch.Tensor:
        """Run a forward pass and return the loss."""
        device = next(self._model.parameters()).device
        if isinstance(batch, dict):
            inputs = {
                k: v.to(device)
                for k, v in batch.items()
                if k in ("input_ids", "attention_mask", "labels")
            }
            out = self._model(**inputs)
            return out.loss
        elif isinstance(batch, (list, tuple)):
            out = self._model(batch[0].to(device))
            return out[0] if isinstance(out, tuple) else out
        else:
            raise ValueError(f"Unsupported batch type: {type(batch)}")
