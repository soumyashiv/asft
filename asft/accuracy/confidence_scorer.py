"""
Confidence Scorer — Assigns confidence, reliability, and verification scores
to every model output. Low scores trigger additional reasoning passes.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ConfidenceScore:
    confidence: float       # 0–1: overall output confidence
    reliability: float      # 0–1: factual reliability estimate
    verification: float     # 0–1: how well the output can be verified
    composite: float        # weighted combination
    flags: List[str]        # detected quality issues

    @property
    def needs_extra_pass(self) -> bool:
        return self.composite < 0.7

    @property
    def label(self) -> str:
        if self.composite >= 0.85:
            return "HIGH"
        elif self.composite >= 0.65:
            return "MEDIUM"
        else:
            return "LOW"

    def __str__(self) -> str:
        return (
            f"ConfidenceScore [{self.label}] "
            f"conf={self.confidence:.2f} rel={self.reliability:.2f} "
            f"ver={self.verification:.2f} composite={self.composite:.2f}"
            + (f" flags={self.flags}" if self.flags else "")
        )


# Patterns that indicate uncertainty or low confidence
_UNCERTAINTY_PATTERNS = [
    r"\bi (think|believe|guess|assume)\b",
    r"\b(probably|possibly|maybe|perhaps|might|could be)\b",
    r"\b(i('m| am) not sure|i don't know|uncertain|unclear)\b",
    r"\b(approximately|roughly|around|about)\b",
]

# Patterns that indicate hallucination risk
_HALLUCINATION_PATTERNS = [
    r"\bspecific(ally)? (in|on|at) \d{4}\b",  # specific years without context
    r"\baccording to (a|the) (recent|new) study\b",
    r"\bexperts (say|agree|claim)\b",
    r"\bit('s| is) (widely|commonly|generally) known\b",
]

# Patterns that indicate verifiable content
_VERIFIABLE_PATTERNS = [
    r"\d+(\.\d+)?",        # numbers
    r"```[\s\S]+?```",     # code blocks
    r"https?://\S+",       # URLs
    r"\b\d{4}\b",          # years
]


class ConfidenceScorer:
    """
    Multi-dimensional output quality scorer.

    Analyzes:
      - Linguistic uncertainty markers
      - Hallucination risk patterns
      - Verifiability signals
      - Output length and structure
      - Task-specific quality indicators
    """

    def __init__(self, threshold: float = 0.7):
        self._threshold = threshold
        self._uncertainty_re = [re.compile(p, re.IGNORECASE) for p in _UNCERTAINTY_PATTERNS]
        self._hallucination_re = [re.compile(p, re.IGNORECASE) for p in _HALLUCINATION_PATTERNS]
        self._verifiable_re = [re.compile(p) for p in _VERIFIABLE_PATTERNS]

    def score(self, output: str, task_type: str = "general",
              model_logprobs: Optional[List[float]] = None) -> ConfidenceScore:
        """Score a model output."""
        flags: List[str] = []

        if not output or len(output.strip()) < 5:
            return ConfidenceScore(0.1, 0.1, 0.1, 0.1, ["empty_output"])

        # 1. Confidence: based on uncertainty language
        confidence = self._score_confidence(output, flags)

        # 2. Reliability: based on hallucination risk patterns
        reliability = self._score_reliability(output, flags)

        # 3. Verification: how verifiable is the output
        verification = self._score_verification(output)

        # 4. Length/structure bonus
        length_bonus = min(0.1, len(output) / 2000)
        has_structure = any(c in output for c in ["\n", "##", "- ", "1."])
        structure_bonus = 0.05 if has_structure else 0.0

        # 5. Log-prob bonus if available
        logprob_bonus = 0.0
        if model_logprobs:
            avg_logprob = sum(model_logprobs) / len(model_logprobs)
            logprob_bonus = min(0.1, max(0.0, (avg_logprob + 5) / 50))

        # Composite (weighted)
        composite = (
            0.40 * confidence
            + 0.30 * reliability
            + 0.20 * verification
            + length_bonus
            + structure_bonus
            + logprob_bonus
        )
        composite = min(1.0, max(0.0, composite))

        return ConfidenceScore(
            confidence=round(confidence, 3),
            reliability=round(reliability, 3),
            verification=round(verification, 3),
            composite=round(composite, 3),
            flags=flags,
        )

    def _score_confidence(self, text: str, flags: List[str]) -> float:
        uncertainty_hits = sum(1 for p in self._uncertainty_re if p.search(text))
        if uncertainty_hits >= 3:
            flags.append("high_uncertainty")
            return 0.4
        elif uncertainty_hits >= 1:
            flags.append("some_uncertainty")
            return 0.7
        return 0.9

    def _score_reliability(self, text: str, flags: List[str]) -> float:
        hallucination_hits = sum(1 for p in self._hallucination_re if p.search(text))
        if hallucination_hits >= 2:
            flags.append("hallucination_risk")
            return 0.35
        elif hallucination_hits >= 1:
            flags.append("minor_hallucination_risk")
            return 0.65
        return 0.85

    def _score_verification(self, text: str) -> float:
        verifiable_hits = sum(1 for p in self._verifiable_re if p.search(text))
        return min(1.0, 0.3 + verifiable_hits * 0.15)

    def batch_score(self, outputs: List[str], task_type: str = "general") -> List[ConfidenceScore]:
        return [self.score(o, task_type) for o in outputs]

    def best_output(self, outputs: List[str], task_type: str = "general") -> tuple:
        """Select the highest-confidence output from a list."""
        scores = self.batch_score(outputs, task_type)
        best_idx = max(range(len(scores)), key=lambda i: scores[i].composite)
        return outputs[best_idx], scores[best_idx]
