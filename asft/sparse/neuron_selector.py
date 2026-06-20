"""
Neuron Selector — Identifies which parameters to train based on importance.
Implements: Magnitude, Gradient, Fisher Information, and Activation-based selection.
Produces a SparseSelectionMask that freezes all non-selected parameters.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class SparseSelectionMask:
    """
    Defines which parameters should be trained (True) vs frozen (False).
    Also tracks sparsity metrics.
    """
    trainable_params: set[str] = field(default_factory=set)
    frozen_params: set[str] = field(default_factory=set)
    sparsity_ratio: float = 0.95
    method: str = "activation"
    total_params: int = 0
    trainable_param_count: int = 0

    @property
    def actual_sparsity(self) -> float:
        if self.total_params == 0:
            return 0.0
        return 1.0 - (self.trainable_param_count / self.total_params)

    def summary(self) -> str:
        return (
            f"SparseSelectionMask | method={self.method} | "
            f"trainable={self.trainable_param_count:,} / {self.total_params:,} params | "
            f"sparsity={self.actual_sparsity:.2%}"
        )


class NeuronSelector:
    """
    Selects the minimal set of parameters to train while freezing the rest.

    Methods:
      - magnitude:   Select params with highest absolute weight values
      - gradient:    Select params with highest gradient norms after a forward-backward pass
      - fisher:      Approximate Fisher Information diagonal for importance
      - activation:  Use output of ActivationAnalyzer to select important layers
    """

    def __init__(self, model: nn.Module, sparsity_ratio: float = 0.95,
                 method: str = "activation", dynamic: bool = True):
        self._model = model
        self._sparsity_ratio = sparsity_ratio
        self._method = method
        self._dynamic = dynamic  # Adjust ratio based on task complexity

    def select(self, activation_report=None, dataloader=None,
               task_complexity: float = 0.5) -> SparseSelectionMask:
        """
        Run selection and return a SparseSelectionMask.
        `task_complexity`: 0.0 (trivial) → 1.0 (very complex).
        Higher complexity → train more params.
        """
        ratio = self._sparsity_ratio
        if self._dynamic:
            # Adjust: complex tasks get 20% more trainable params
            ratio = max(0.5, ratio - task_complexity * 0.2)

        logger.info("NeuronSelector: method=%s sparsity=%.2f", self._method, ratio)

        if self._method == "activation" and activation_report is not None:
            mask = self._select_by_activation(activation_report, ratio)
        elif self._method == "magnitude":
            mask = self._select_by_magnitude(ratio)
        elif self._method == "gradient" and dataloader is not None:
            mask = self._select_by_gradient(dataloader, ratio)
        elif self._method == "fisher" and dataloader is not None:
            mask = self._select_by_fisher(dataloader, ratio)
        else:
            logger.warning("Falling back to magnitude selection")
            mask = self._select_by_magnitude(ratio)

        mask.method = self._method
        mask.sparsity_ratio = ratio
        self._count_params(mask)
        logger.info(mask.summary())
        return mask

    def apply_mask(self, mask: SparseSelectionMask) -> None:
        """Apply the mask: freeze frozen params, unfreeze trainable params."""
        for name, param in self._model.named_parameters():
            # Match by layer name prefix
            should_train = any(name.startswith(t) or t in name for t in mask.trainable_params)
            param.requires_grad_(should_train)

        trainable = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self._model.parameters())
        logger.info(
            "Mask applied: trainable=%d / %d params (%.2f%%)",
            trainable, total, 100 * trainable / max(1, total)
        )

    # ------------------------------------------------------------------
    # Selection strategies
    # ------------------------------------------------------------------

    def _select_by_activation(self, report, ratio: float) -> SparseSelectionMask:
        """Use activation report's top_layers to select trainable params."""
        top_layers = set(report.top_layers)
        trainable: set[str] = set()
        frozen: set[str] = set()

        for name, _ in self._model.named_parameters():
            # Check if this param belongs to a top layer
            matched = any(layer in name for layer in top_layers)
            if matched:
                trainable.add(name)
            else:
                frozen.add(name)

        return SparseSelectionMask(trainable_params=trainable, frozen_params=frozen)

    def _select_by_magnitude(self, ratio: float) -> SparseSelectionMask:
        """Select parameters with highest absolute weight magnitudes."""
        scores: dict[str, float] = {}
        for name, param in self._model.named_parameters():
            if param.requires_grad:
                scores[name] = float(param.data.abs().mean())

        return self._threshold_select(scores, ratio)

    def _select_by_gradient(self, dataloader, ratio: float) -> SparseSelectionMask:
        """Run a forward-backward pass to collect gradient norms."""
        self._model.train()
        scores: dict[str, float] = {}

        for i, batch in enumerate(dataloader):
            if i >= 3:
                break
            if isinstance(batch, dict):
                inputs = {k: v for k, v in batch.items()
                          if k in ("input_ids", "attention_mask", "labels")}
                outputs = self._model(**inputs)
                loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
            else:
                outputs = self._model(batch[0])
                loss = outputs[0]
            loss.backward()
            break

        for name, param in self._model.named_parameters():
            if param.grad is not None:
                scores[name] = float(param.grad.norm(2))
            else:
                scores[name] = 0.0

        self._model.zero_grad()
        return self._threshold_select(scores, ratio)

    def _select_by_fisher(self, dataloader, ratio: float) -> SparseSelectionMask:
        """Approximate Fisher Information using squared gradients."""
        self._model.train()
        fisher: dict[str, float] = {
            name: 0.0 for name, p in self._model.named_parameters() if p.requires_grad
        }
        n_samples = 0

        for i, batch in enumerate(dataloader):
            if i >= 5:
                break
            if isinstance(batch, dict):
                inputs = {k: v for k, v in batch.items()
                          if k in ("input_ids", "attention_mask", "labels")}
                outputs = self._model(**inputs)
                loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
            else:
                outputs = self._model(batch[0])
                loss = outputs[0]
            loss.backward()

            for name, param in self._model.named_parameters():
                if param.grad is not None:
                    fisher[name] += float(param.grad.pow(2).mean())
            self._model.zero_grad()
            n_samples += 1

        # Normalize
        if n_samples > 0:
            fisher = {k: v / n_samples for k, v in fisher.items()}

        return self._threshold_select(fisher, ratio)

    def _threshold_select(self, scores: dict[str, float], ratio: float) -> SparseSelectionMask:
        """Select top (1-ratio) fraction of params by score."""
        if not scores:
            return SparseSelectionMask()
        sorted_params = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        n_trainable = max(1, int(len(sorted_params) * (1.0 - ratio)))
        trainable = {name for name, _ in sorted_params[:n_trainable]}
        frozen = {name for name, _ in sorted_params[n_trainable:]}
        return SparseSelectionMask(trainable_params=trainable, frozen_params=frozen)

    def _count_params(self, mask: SparseSelectionMask) -> None:
        param_map: dict[str, int] = {
            name: p.numel() for name, p in self._model.named_parameters()
        }
        mask.total_params = sum(param_map.values())
        mask.trainable_param_count = sum(
            param_map.get(name, 0) for name in mask.trainable_params
        )
