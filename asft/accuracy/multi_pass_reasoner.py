"""
Multi-Pass Reasoner — Generates K candidate solutions, scores each,
and returns the highest-confidence answer. Implements self-consistency reasoning.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from asft.accuracy.confidence_scorer import ConfidenceScorer, ConfidenceScore

logger = logging.getLogger(__name__)


@dataclass
class ReasoningResult:
    best_output: str
    best_score: ConfidenceScore
    all_outputs: List[str] = field(default_factory=list)
    all_scores: List[ConfidenceScore] = field(default_factory=list)
    consensus_output: Optional[str] = None
    passes_used: int = 1
    task_type: str = "general"

    def summary(self) -> str:
        return (
            f"MultiPassReasoning: {self.passes_used} passes | "
            f"best={self.best_score.composite:.3f} | "
            f"task={self.task_type}"
        )


class MultiPassReasoner:
    """
    Generates multiple candidate solutions and selects the best one.

    Strategies:
      - best_of_k:       Generate K outputs, pick highest-confidence
      - self_consistency: Pick output with highest agreement across candidates
      - escalating:       Start with 1 pass, escalate if confidence is low
    """

    def __init__(
        self,
        k: int = 3,
        min_confidence: float = 0.7,
        strategy: str = "best_of_k",
    ):
        self._k = k
        self._min_confidence = min_confidence
        self._strategy = strategy
        self._scorer = ConfidenceScorer(threshold=min_confidence)

    def reason(
        self,
        generate_fn: Callable[[int], List[str]],
        task_type: str = "general",
    ) -> ReasoningResult:
        """
        Run multi-pass reasoning.

        Args:
            generate_fn: callable(n_samples) → List[str] — model generation function
            task_type: domain hint for scoring

        Returns:
            ReasoningResult with best output and all candidates
        """
        if self._strategy == "escalating":
            return self._escalating_pass(generate_fn, task_type)
        elif self._strategy == "self_consistency":
            return self._self_consistency(generate_fn, task_type)
        else:
            return self._best_of_k(generate_fn, task_type)

    def _best_of_k(self, generate_fn, task_type: str) -> ReasoningResult:
        """Generate K outputs, score all, return highest-confidence."""
        outputs = generate_fn(self._k)
        if not outputs:
            return ReasoningResult(best_output="", best_score=ConfidenceScore(0, 0, 0, 0, ["no_output"]))

        scores = self._scorer.batch_score(outputs, task_type)
        best_idx = max(range(len(scores)), key=lambda i: scores[i].composite)

        logger.debug(
            "BestOfK[%d]: scores=%s best=%.3f",
            self._k, [round(s.composite, 3) for s in scores], scores[best_idx].composite
        )

        return ReasoningResult(
            best_output=outputs[best_idx],
            best_score=scores[best_idx],
            all_outputs=outputs,
            all_scores=scores,
            passes_used=len(outputs),
            task_type=task_type,
        )

    def _self_consistency(self, generate_fn, task_type: str) -> ReasoningResult:
        """Generate K outputs, find the one with most agreement."""
        outputs = generate_fn(self._k)
        if not outputs:
            return ReasoningResult(best_output="", best_score=ConfidenceScore(0, 0, 0, 0, ["no_output"]))

        scores = self._scorer.batch_score(outputs, task_type)

        # Find consensus: output most similar (by token overlap) to others
        consensus_idx = self._find_consensus(outputs)
        consensus = outputs[consensus_idx]

        best_idx = max(range(len(scores)), key=lambda i: scores[i].composite)

        return ReasoningResult(
            best_output=outputs[best_idx],
            best_score=scores[best_idx],
            all_outputs=outputs,
            all_scores=scores,
            consensus_output=consensus,
            passes_used=len(outputs),
            task_type=task_type,
        )

    def _escalating_pass(self, generate_fn, task_type: str) -> ReasoningResult:
        """
        Start with 1 pass. If confidence is below threshold, escalate to K.
        Minimizes compute for easy tasks.
        """
        # First pass
        outputs = generate_fn(1)
        if not outputs:
            return ReasoningResult(best_output="", best_score=ConfidenceScore(0, 0, 0, 0, ["no_output"]))

        score = self._scorer.score(outputs[0], task_type)
        if score.composite >= self._min_confidence:
            logger.debug("Escalating: single pass sufficient (score=%.3f)", score.composite)
            return ReasoningResult(
                best_output=outputs[0], best_score=score,
                all_outputs=outputs, all_scores=[score], passes_used=1, task_type=task_type
            )

        # Escalate: generate K-1 more
        logger.debug("Escalating: low confidence (%.3f) — generating %d more", score.composite, self._k - 1)
        extra = generate_fn(self._k - 1)
        all_outputs = outputs + extra
        all_scores = self._scorer.batch_score(all_outputs, task_type)
        best_idx = max(range(len(all_scores)), key=lambda i: all_scores[i].composite)

        return ReasoningResult(
            best_output=all_outputs[best_idx],
            best_score=all_scores[best_idx],
            all_outputs=all_outputs,
            all_scores=all_scores,
            passes_used=len(all_outputs),
            task_type=task_type,
        )

    def _find_consensus(self, outputs: List[str]) -> int:
        """Find the output with highest token overlap to all others."""
        def token_overlap(a: str, b: str) -> float:
            ta, tb = set(a.lower().split()), set(b.lower().split())
            return len(ta & tb) / max(1, len(ta | tb))

        scores = []
        for i, out in enumerate(outputs):
            total = sum(token_overlap(out, outputs[j]) for j in range(len(outputs)) if j != i)
            scores.append(total)

        return int(max(range(len(scores)), key=lambda i: scores[i]))
