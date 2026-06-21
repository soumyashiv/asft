"""Prompt baseline evaluation interface and mock implementation."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class PromptEvaluationResult:
    """Typed result from a prompt-baseline evaluation stage."""

    method: str = "prompt"
    score: float = 0.0  # normalised 0-100

    def as_dict(self) -> Dict[str, Any]:
        return {"method": self.method, "score": round(self.score / 100, 2)}


class PromptEvaluator(ABC):
    """Abstract base class for zero-shot / few-shot baseline evaluation.

    Implement this class to plug in a real benchmark runner
    (e.g. lm-eval-harness, OpenAI Evals, custom test set).
    """

    @abstractmethod
    def evaluate_baseline(self, task_config: Dict[str, Any]) -> PromptEvaluationResult:
        """Run the base model against the task and return the baseline score."""
        ...


class MockPromptEvaluator(PromptEvaluator):
    """Demonstration evaluator — returns realistic-looking scores without running a model.

    Replace with a real :class:`PromptEvaluator` subclass when you have GPU access.
    """

    def evaluate_baseline(self, task_config: Dict[str, Any]) -> PromptEvaluationResult:
        task_name = task_config.get("task_name", "").lower()
        score = 72.0 if "support" in task_name else 65.0
        return PromptEvaluationResult(method="prompt", score=score)
