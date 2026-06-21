"""Decision recommendation engine."""

from __future__ import annotations

from dataclasses import dataclass

from asft.analysis.evaluator import PromptEvaluationResult
from asft.analysis.finetune_estimator import FinetuneEstimateResult
from asft.analysis.rag_analyzer import RAGEvaluationResult


@dataclass
class RecommendationResult:
    """Typed output from the recommendation engine."""

    method: str  # "PROMPTING" | "RAG" | "FINE-TUNING"
    confidence: float  # 0-100
    reason: str
    savings_usd: float  # 0 if fine-tuning is the recommendation
    savings_hours: float  # 0 if fine-tuning is the recommendation


class DecisionRecommender:
    """Pure-logic engine that maps evaluation results to an actionable recommendation.

    All reason strings are generated dynamically from the actual numbers —
    no task-name heuristics, no hardcoded strings.

    Parameters
    ----------
    rag_threshold:
        Minimum RAG-over-prompt improvement (percentage points) to prefer RAG.
    ft_threshold:
        Minimum FT-over-best improvement (pp) to prefer fine-tuning despite cost.
    """

    def __init__(self, rag_threshold: float = 10.0, ft_threshold: float = 5.0) -> None:
        self.rag_threshold = rag_threshold
        self.ft_threshold = ft_threshold

    def recommend(
        self,
        prompt: PromptEvaluationResult,
        rag: RAGEvaluationResult,
        ft: FinetuneEstimateResult,
    ) -> RecommendationResult:
        """Evaluate trade-offs and return a :class:`RecommendationResult`."""
        rag_gain = rag.score - prompt.score
        ft_gain = ft.expected_score - max(rag.score, prompt.score)

        # ── Fine-tuning wins ──────────────────────────────────────────────────
        if ft_gain >= self.ft_threshold:
            conf = min(80.0 + ft_gain, 99.0)
            reason = (
                f"Fine-tuning adds {ft_gain:.1f} pp over the current best "
                f"({max(rag.score, prompt.score):.0f}%), justifying the "
                f"${ft.estimated_cost:.0f} / {ft.estimated_hours:.0f} h investment."
            )
            return RecommendationResult(
                method="FINE-TUNING",
                confidence=conf,
                reason=reason,
                savings_usd=0.0,
                savings_hours=0.0,
            )

        # ── RAG wins ─────────────────────────────────────────────────────────
        if rag_gain >= self.rag_threshold:
            conf = min(80.0 + rag_gain, 99.0)
            reason = (
                f"RAG improves accuracy by {rag_gain:.1f} pp over prompting "
                f"({prompt.score:.0f}% -> {rag.score:.0f}%). "
                f"Fine-tuning adds only {ft_gain:.1f} pp more while costing "
                f"${ft.estimated_cost:.0f} and {ft.estimated_hours:.0f} GPU hours. "
                f"Avoid unnecessary fine-tuning."
            )
            return RecommendationResult(
                method="RAG",
                confidence=conf,
                reason=reason,
                savings_usd=ft.estimated_cost,
                savings_hours=ft.estimated_hours,
            )

        # ── Prompting wins ────────────────────────────────────────────────────
        conf = 85.0
        reason = (
            f"Prompt engineering ({prompt.score:.0f}%) is competitive with RAG "
            f"({rag.score:.0f}%) and fine-tuning ({ft.expected_score:.0f}%). "
            f"Additional complexity is not justified."
        )
        return RecommendationResult(
            method="PROMPTING",
            confidence=conf,
            reason=reason,
            savings_usd=ft.estimated_cost,
            savings_hours=ft.estimated_hours,
        )
