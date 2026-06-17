"""
Activation Analyzer — Hooks into transformer forward passes to collect
activation statistics for neuron/layer importance ranking.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class LayerActivationStats:
    layer_name: str
    mean_activation: float = 0.0
    std_activation: float = 0.0
    max_activation: float = 0.0
    sparsity: float = 0.0        # fraction of zero activations
    gradient_norm: float = 0.0   # L2 norm of gradients (if available)
    num_samples: int = 0

    @property
    def importance_score(self) -> float:
        """Combined importance: high activation + low sparsity + gradient norm."""
        act_score = self.mean_activation * (1.0 - self.sparsity)
        return float(act_score + 0.3 * self.gradient_norm)


@dataclass
class ActivationReport:
    layer_stats: Dict[str, LayerActivationStats] = field(default_factory=dict)
    ranked_layers: List[Tuple[str, float]] = field(default_factory=list)
    top_layers: List[str] = field(default_factory=list)

    def summary(self, top_n: int = 10) -> str:
        lines = [f"=== Activation Analysis (top {top_n} layers) ==="]
        for i, (name, score) in enumerate(self.ranked_layers[:top_n], 1):
            stats = self.layer_stats[name]
            lines.append(
                f"  {i:2d}. {name:<50s} score={score:.4f} "
                f"mean={stats.mean_activation:.4f} sparsity={stats.sparsity:.2%}"
            )
        return "\n".join(lines)


class ActivationAnalyzer:
    """
    Attaches forward hooks to a model and collects activation statistics
    during a forward pass. Used to identify the most important layers/neurons
    for sparse training.
    """

    def __init__(self, model: nn.Module, target_module_types: Optional[Tuple] = None):
        self._model = model
        self._hooks: List[torch.utils.hooks.RemovableHook] = []
        self._stats: Dict[str, LayerActivationStats] = defaultdict(
            lambda: LayerActivationStats(layer_name="")
        )
        self._target_types = target_module_types or (nn.Linear, nn.LayerNorm)
        self._attached = False

    def attach(self) -> None:
        """Attach forward hooks to all target layers."""
        if self._attached:
            return

        for name, module in self._model.named_modules():
            if isinstance(module, self._target_types):
                hook = module.register_forward_hook(self._make_hook(name))
                self._hooks.append(hook)

        logger.info("ActivationAnalyzer: attached hooks to %d layers", len(self._hooks))
        self._attached = True

    def detach(self) -> None:
        """Remove all hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._attached = False
        logger.debug("ActivationAnalyzer: hooks removed")

    def _make_hook(self, layer_name: str) -> Callable:
        def hook(module, input, output):
            with torch.no_grad():
                if isinstance(output, tuple):
                    out = output[0]
                else:
                    out = output

                if not isinstance(out, torch.Tensor):
                    return

                flat = out.detach().float().abs().flatten()
                n = flat.numel()
                if n == 0:
                    return

                stats = self._stats[layer_name]
                stats.layer_name = layer_name
                stats.mean_activation = float(flat.mean())
                stats.std_activation = float(flat.std())
                stats.max_activation = float(flat.max())
                stats.sparsity = float((flat == 0).float().mean())
                stats.num_samples += 1
        return hook

    def collect_gradient_norms(self) -> None:
        """After backward(), collect gradient norms per layer."""
        for name, module in self._model.named_modules():
            if name in self._stats and hasattr(module, "weight"):
                if module.weight.grad is not None:
                    grad_norm = float(module.weight.grad.norm(2))
                    self._stats[name].gradient_norm = grad_norm

    def analyze(self, sparsity_target: float = 0.95) -> ActivationReport:
        """
        Produce a ranked activation report.
        `sparsity_target`: fraction of layers to exclude from training.
        """
        if not self._stats:
            logger.warning("No activation statistics collected — did you run forward passes?")
            return ActivationReport()

        # Rank layers by importance score
        ranked = sorted(
            [(name, stats.importance_score) for name, stats in self._stats.items()],
            key=lambda x: x[1],
            reverse=True,
        )

        # Select top (1 - sparsity_target) fraction as "active"
        total = len(ranked)
        top_n = max(1, int(total * (1.0 - sparsity_target)))
        top_layers = [name for name, _ in ranked[:top_n]]

        return ActivationReport(
            layer_stats=dict(self._stats),
            ranked_layers=ranked,
            top_layers=top_layers,
        )

    def reset(self) -> None:
        """Clear all collected statistics."""
        self._stats.clear()

    def __enter__(self):
        self.attach()
        return self

    def __exit__(self, *args):
        self.detach()


def analyze_model_activations(
    model: nn.Module,
    dataloader,
    num_batches: int = 10,
    sparsity_target: float = 0.95,
    device: str = "cpu",
) -> ActivationReport:
    """
    Convenience function: run forward passes and return activation report.
    """
    analyzer = ActivationAnalyzer(model)
    model.eval()
    with analyzer:
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= num_batches:
                    break
                # Support dict batches (HuggingFace style)
                if isinstance(batch, dict):
                    inputs = {
                        k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()
                        if k in ("input_ids", "attention_mask")
                    }
                    model(**inputs)
                elif isinstance(batch, (list, tuple)):
                    model(batch[0].to(device))

    return analyzer.analyze(sparsity_target=sparsity_target)
