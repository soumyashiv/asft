"""RAG (Retrieval-Augmented Generation) evaluation interface and mock implementation."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict

from asft.analysis.evaluator import PromptEvaluationResult


@dataclass
class RAGEvaluationResult:
    """Typed result from a RAG evaluation stage."""

    method: str = "rag"
    score: float = 0.0  # normalised 0-100
    retrieval_available: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {"method": self.method, "score": round(self.score / 100, 2)}


class RAGAnalyzer(ABC):
    """Abstract base class for RAG evaluation.

    Implement this class to plug in a real retrieval pipeline
    (e.g. LangChain, LlamaIndex, custom embeddings + vector store).
    """

    @abstractmethod
    def evaluate_rag(
        self,
        task_config: Dict[str, Any],
        baseline: PromptEvaluationResult,
    ) -> RAGEvaluationResult:
        """Run RAG against the task and return the retrieval-augmented score."""
        ...


class MockRAGAnalyzer(RAGAnalyzer):
    """Demonstration RAG analyzer — returns realistic-looking scores without running retrieval.

    Replace with a real :class:`RAGAnalyzer` subclass when you have embeddings set up.
    """

    def evaluate_rag(
        self,
        task_config: Dict[str, Any],
        baseline: PromptEvaluationResult,
    ) -> RAGEvaluationResult:
        has_docs = bool(task_config.get("documents")) or bool(task_config.get("dataset"))
        task_name = task_config.get("task_name", "").lower()

        if not has_docs:
            return RAGEvaluationResult(
                method="rag", score=baseline.score, retrieval_available=False
            )

        boost = 17.0 if "support" in task_name else 5.0
        return RAGEvaluationResult(
            method="rag",
            score=min(baseline.score + boost, 99.0),
            retrieval_available=True,
        )
