"""
ASFT Parameter Selector — Train only the most important parameters.

MISSION:
    Reduce trainable parameters by 70–99% while maintaining accuracy.
    This module extends the existing NeuronSelector with proven importance
    scoring methods backed by peer-reviewed research.

METHODS AND THEIR EVIDENCE:

    1. magnitude (fastest, weakest):
       - Select params with highest |weight| values
       - Basis: LeCun et al. (1989) "Optimal Brain Damage" — large weights
         contribute more to the output, so their updates matter more
       - When it fails: post-ReLU activations; sparse models where large weights
         already correspond to near-dead neurons
       - Cost: O(N) time, O(1) memory overhead

    2. gradient (medium cost, stronger signal):
       - Select params with highest ||∇_θ L||_2 after one backward pass
       - Basis: gradient norm reflects how much this parameter is currently
         being pushed to change. High gradient = high learning pressure.
       - When it fails: saturated gradients (vanishing/exploding); early in
         training when gradients are noisy
       - Cost: 1 full forward+backward pass

    3. fisher (most principled, higher cost):
       - Fisher Information diagonal: F_i ≈ E[∂²L/∂θ_i²] ≈ E[(∂L/∂θ_i)²]
       - High Fisher = this parameter has high curvature = changing it
         would significantly change the loss = it matters
       - Basis: Kirkpatrick et al. (2017) EWC; Lecam (1960) original FI theory
       - Used in: network pruning (Molchanov et al. 2017), continual learning
       - When it fails: Fisher diagonal approximation breaks down for
         highly correlated parameters (off-diagonal terms ignored)
       - Cost: 3–5 forward+backward passes

    4. taylor (best accuracy/cost tradeoff for pruning):
       - Taylor expansion approximation: importance_i = |g_i · θ_i|
       - This approximates the change in loss if θ_i were set to zero:
         ΔL ≈ |∂L/∂θ_i · θ_i|
       - Basis: Molchanov et al. (2017) "Pruning CNNs for Resource-Efficient Inference"
       - When it works best: identifying which LAYERS to train (not individual neurons)
       - Cost: 1 forward+backward pass

    5. activation (fast, uses ActivationAnalyzer output):
       - Already implemented in NeuronSelector from activation_analyzer.py
       - Basis: empirical observation that layers with high activation variance
         are more task-relevant (Frankle et al. 2019 Lottery Ticket)

ACHIEVABLE REDUCTION (honest):
    - LoRA already achieves 95–99.9% parameter reduction mathematically
    - Selective layer training (activation/gradient/taylor): 70–95%
    - These methods complement LoRA: use them to identify WHICH layers
      to apply LoRA adapters to, not as a replacement for LoRA

FAILURE MODES:
    - At extreme sparsity (>99%), accuracy degrades precipitously on complex tasks
    - Layer pruning without FLOP-awareness: pruning a small but computationally
      critical bottleneck layer gains almost nothing
    - Random seed sensitivity: gradient and fisher scores are noisy across runs
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ParameterImportanceReport:
    """Result of parameter importance analysis."""

    method: str
    total_params: int
    selected_params: int
    trainable_fraction: float
    layer_scores: dict[str, float] = field(default_factory=dict)
    top_layers: list[str] = field(default_factory=list)
    selected_param_names: set[str] = field(default_factory=set)
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"ParameterImportance | method={self.method} | "
            f"trainable={self.selected_params:,}/{self.total_params:,} "
            f"({self.trainable_fraction:.1%})"
        )


class ParameterSelector:
    """
    Identifies which parameters have the highest importance for a given task,
    enabling selective fine-tuning instead of updating all parameters.

    This wraps and extends the existing NeuronSelector with additional methods
    and explicit research citations.

    Args:
        model:         PyTorch model to analyze.
        sparsity:      Fraction of parameters to FREEZE (0.95 = train only 5%).
        method:        Importance scoring method.
        n_probe_steps: For gradient/fisher/taylor: number of calibration steps.
    """

    def __init__(
        self,
        model,
        sparsity: float = 0.95,
        method: str = "taylor",
        n_probe_steps: int = 10,
    ):
        self._model = model
        self._sparsity = sparsity
        self._method = method
        self._n_probe_steps = n_probe_steps
        self._param_count = sum(p.numel() for p in model.parameters())

    def analyze(
        self,
        dataloader=None,
        activation_report=None,
        task_complexity: float = 0.5,
    ) -> ParameterImportanceReport:
        """
        Run importance analysis and return a structured report.

        Args:
            dataloader:        Required for gradient, fisher, taylor methods.
            activation_report: Required for activation method.
            task_complexity:   0–1; higher = train more parameters adaptively.

        Returns:
            ParameterImportanceReport with selected parameter names.
        """
        # Adjust sparsity based on task complexity (simpler tasks need fewer params)
        adjusted_sparsity = max(0.50, self._sparsity - task_complexity * 0.20)
        n_keep_fraction = 1.0 - adjusted_sparsity

        logger.info(
            "ParameterSelector: method=%s sparsity=%.2f adjusted_sparsity=%.2f",
            self._method,
            self._sparsity,
            adjusted_sparsity,
        )

        warnings = []

        if self._method == "taylor" and dataloader is not None:
            scores = self._score_taylor(dataloader)
        elif self._method == "fisher" and dataloader is not None:
            scores = self._score_fisher(dataloader)
        elif self._method == "gradient" and dataloader is not None:
            scores = self._score_gradient(dataloader)
        elif self._method == "activation" and activation_report is not None:
            return self._from_activation_report(activation_report, adjusted_sparsity)
        else:
            if dataloader is None and self._method != "magnitude":
                warnings.append(
                    f"Method '{self._method}' requires a dataloader. Falling back to magnitude."
                )
            scores = self._score_magnitude()

        return self._build_report(scores, n_keep_fraction, warnings)

    def apply(self, report: ParameterImportanceReport) -> None:
        """
        Apply the selection: freeze frozen params, enable gradients on selected.
        Call this BEFORE creating the optimizer.
        """
        for name, param in self._model.named_parameters():
            should_train = name in report.selected_param_names
            param.requires_grad_(should_train)

        actual_trainable = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
        logger.info(
            "ParameterSelector applied: %d/%d params trainable (%.2f%%)",
            actual_trainable,
            self._param_count,
            100 * actual_trainable / max(1, self._param_count),
        )

    # ------------------------------------------------------------------
    # Scoring strategies
    # ------------------------------------------------------------------

    def _score_magnitude(self) -> dict[str, float]:
        """Magnitude-based importance: |weight|.mean(). O(N), no data needed."""
        return {
            name: float(param.data.abs().mean())
            for name, param in self._model.named_parameters()
            if param.requires_grad
        }

    def _score_gradient(self, dataloader) -> dict[str, float]:
        """Single-batch gradient norm scoring."""
        return self._run_backward_scoring(dataloader, n_steps=1, score_fn="grad_norm")

    def _score_fisher(self, dataloader) -> dict[str, float]:
        """
        Fisher Information diagonal approximation.
        F_i ≈ mean(g_i²) over n_probe_steps mini-batches.
        Evidence: Kirkpatrick et al. 2017 (EWC).
        """
        return self._run_backward_scoring(
            dataloader, n_steps=self._n_probe_steps, score_fn="fisher"
        )

    def _score_taylor(self, dataloader) -> dict[str, float]:
        """
        Taylor expansion importance: importance_i = |g_i · θ_i|.
        Approximates the change in loss if parameter θ_i were zeroed.
        Evidence: Molchanov et al. 2017.
        """
        return self._run_backward_scoring(
            dataloader, n_steps=self._n_probe_steps, score_fn="taylor"
        )

    def _run_backward_scoring(
        self,
        dataloader,
        n_steps: int,
        score_fn: str,
    ) -> dict[str, float]:
        """Shared backward-pass scoring loop."""

        self._model.train()
        accum: dict[str, float] = {
            name: 0.0 for name, p in self._model.named_parameters() if p.requires_grad
        }
        n_batches = 0

        for i, batch in enumerate(dataloader):
            if i >= n_steps:
                break
            try:
                if isinstance(batch, dict):
                    inputs = {
                        k: v
                        for k, v in batch.items()
                        if k in ("input_ids", "attention_mask", "labels")
                    }
                    out = self._model(**inputs)
                    loss = out.loss
                else:
                    out = self._model(batch[0])
                    loss = out[0] if isinstance(out, tuple) else out

                loss.backward()

                for name, param in self._model.named_parameters():
                    if param.grad is None:
                        continue
                    if score_fn == "grad_norm":
                        accum[name] += float(param.grad.norm(2))
                    elif score_fn == "fisher":
                        accum[name] += float(param.grad.pow(2).mean())
                    elif score_fn == "taylor":
                        accum[name] += float((param.grad * param.data).abs().mean())

                self._model.zero_grad()
                n_batches += 1
            except Exception as e:
                logger.warning("Scoring batch failed: %s", e)
                self._model.zero_grad()

        if n_batches > 0:
            accum = {k: v / n_batches for k, v in accum.items()}

        return accum

    def _from_activation_report(
        self, activation_report: Any, adjusted_sparsity: float
    ) -> ParameterImportanceReport:
        """Build report from ActivationAnalyzer output."""
        top_layers = set(getattr(activation_report, "top_layers", []))
        selected: set[str] = set()
        for name, _ in self._model.named_parameters():
            if any(layer in name for layer in top_layers):
                selected.add(name)

        return self._build_report_from_selection(
            selected=selected,
            layer_scores={l: 1.0 for l in top_layers},  # noqa: E741
            method="activation",
            warnings=[],
        )

    def _build_report(
        self,
        scores: dict[str, float],
        keep_fraction: float,
        warnings: list[str],
    ) -> ParameterImportanceReport:
        """Convert scores to a selection report."""
        if not scores:
            return ParameterImportanceReport(
                method=self._method,
                total_params=self._param_count,
                selected_params=0,
                trainable_fraction=0.0,
                warnings=warnings,
            )

        sorted_params = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        n_keep = max(1, int(len(sorted_params) * keep_fraction))
        selected = {name for name, _ in sorted_params[:n_keep]}

        return self._build_report_from_selection(
            selected=selected,
            layer_scores=scores,
            method=self._method,
            warnings=warnings,
        )

    def _build_report_from_selection(
        self,
        selected: set[str],
        layer_scores: dict[str, float],
        method: str,
        warnings: list[str],
    ) -> ParameterImportanceReport:
        param_sizes = {name: p.numel() for name, p in self._model.named_parameters()}
        selected_count = sum(param_sizes.get(name, 0) for name in selected)
        top_layers = sorted(layer_scores, key=lambda k: layer_scores[k], reverse=True)[:10]

        report = ParameterImportanceReport(
            method=method,
            total_params=self._param_count,
            selected_params=selected_count,
            trainable_fraction=selected_count / max(1, self._param_count),
            layer_scores=layer_scores,
            top_layers=top_layers,
            selected_param_names=selected,
            warnings=warnings,
        )
        logger.info(report.summary())
        return report
