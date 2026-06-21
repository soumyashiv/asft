"""Fine-tuning cost and accuracy estimation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from asft.analysis.rag_analyzer import RAGEvaluationResult


@dataclass
class FinetuneEstimateResult:
    """Typed result from the fine-tuning estimation stage."""

    method: str = "fine_tuning"
    expected_score: float = 0.0     # normalised 0-100
    estimated_cost: float = 0.0     # USD
    estimated_hours: float = 0.0    # GPU hours

    def as_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "expected_score": round(self.expected_score / 100, 2),
            "estimated_cost": round(self.estimated_cost, 2),
            "estimated_hours": round(self.estimated_hours, 1),
        }


class FinetuneEstimator:
    """Estimates the potential accuracy improvement and compute cost of fine-tuning.

    Heuristics here are intentionally conservative.  Swap the body of
    ``estimate()`` with a real scaling-law calculator when available.

    Parameters
    ----------
    hourly_gpu_rate:
        Price per GPU-hour in USD (default ≈ A100 spot price).
    ft_threshold_small:
        Expected accuracy gain (pp) for tasks where RAG is strong.
    ft_threshold_large:
        Expected accuracy gain (pp) for tasks where RAG adds little.
    """

    DEFAULT_RATE = 55.55  # USD / GPU-hour  (~$500 for 9 h)

    def __init__(
        self,
        hourly_gpu_rate: float = DEFAULT_RATE,
        ft_threshold_small: float = 1.0,
        ft_threshold_large: float = 15.0,
    ) -> None:
        self.hourly_gpu_rate = hourly_gpu_rate
        self.ft_threshold_small = ft_threshold_small
        self.ft_threshold_large = ft_threshold_large

    def estimate(
        self,
        task_config: Dict[str, Any],
        rag_result: RAGEvaluationResult,
    ) -> FinetuneEstimateResult:
        """Return a :class:`FinetuneEstimateResult` for the given task and RAG baseline."""
        model = task_config.get("model", "").lower()
        task_name = task_config.get("task_name", "").lower()

        current_best = rag_result.score

        # Tasks with strong RAG support see diminishing FT returns
        if rag_result.retrieval_available and "support" in task_name:
            gain = self.ft_threshold_small
        else:
            gain = self.ft_threshold_large

        expected_score = min(current_best + gain, 99.0)

        # Rough GPU-hour estimate based on model scale
        gpu_hours = 40.0 if "70b" in model else 9.0
        cost = gpu_hours * self.hourly_gpu_rate

        return FinetuneEstimateResult(
            method="fine_tuning",
            expected_score=expected_score,
            estimated_cost=cost,
            estimated_hours=gpu_hours,
        )
