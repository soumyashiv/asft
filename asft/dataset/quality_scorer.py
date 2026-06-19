"""
Dataset Quality Scorer — Evaluates individual dataset samples for noise and quality.
Used to aggressively prune low-quality, toxic, or poorly formatted training data.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class QualityScore:
    """Quality and noise assessment for a single sample."""
    score: float          # 0.0 to 1.0 (higher is better)
    is_rejected: bool
    rejection_reason: Optional[str] = None
    flags: List[str] = None

    def __post_init__(self):
        if self.flags is None:
            self.flags = []


class QualityScorer:
    """
    Heuristic quality scorer for dataset pruning.
    Evaluates formatting, length, repetition, and noise.
    """

    def __init__(self, min_length: int = 20, max_length: int = 8000, 
                 min_score_threshold: float = 0.5):
        self._min_length = min_length
        self._max_length = max_length
        self._threshold = min_score_threshold

    def score(self, text: str) -> QualityScore:
        """
        Evaluate the quality of a single text sample.
        """
        if not text or not text.strip():
            return QualityScore(0.0, True, "empty_text")

        text_len = len(text)
        if text_len < self._min_length:
            return QualityScore(0.1, True, "too_short")
        if text_len > self._max_length:
            return QualityScore(0.2, True, "too_long")

        score = 1.0
        flags = []

        # 1. Repetition penalty (N-gram repetition)
        if self._has_extreme_repetition(text):
            return QualityScore(0.1, True, "extreme_repetition")

        # 2. Formatting & noise penalties
        if not re.search(r'[a-zA-Z]', text):
            return QualityScore(0.1, True, "no_alphabetic_chars")

        # High ratio of special characters or numbers usually means noise/logs
        alpha_count = sum(c.isalpha() for c in text)
        if alpha_count / max(1, text_len) < 0.3:
            score -= 0.4
            flags.append("low_alpha_ratio")

        # Check for trailing/unclosed markdown or broken tags
        if text.count("```") % 2 != 0:
            score -= 0.2
            flags.append("unclosed_code_block")

        # Basic casing checks (all caps or all lowercase without punctuation)
        if text.isupper():
            score -= 0.3
            flags.append("all_caps")

        # Deduct score for excessive line breaks relative to text length
        if text.count("\n") > text_len / 20:
            score -= 0.2
            flags.append("excessive_newlines")

        score = max(0.0, min(1.0, score))
        is_rejected = score < self._threshold

        return QualityScore(
            score=score,
            is_rejected=is_rejected,
            rejection_reason="low_quality_score" if is_rejected else None,
            flags=flags
        )

    def _has_extreme_repetition(self, text: str) -> bool:
        """Detect extreme N-gram repetition indicative of model collapse or bad scraping."""
        words = text.split()
        if len(words) < 10:
            return False
            
        # Check for single word repeated excessively
        if len(set(words)) / len(words) < 0.1:
            return True
            
        # Check for phrase repetition (e.g., 3-grams)
        trigrams = [" ".join(words[i:i+3]) for i in range(len(words)-2)]
        if trigrams and len(set(trigrams)) / len(trigrams) < 0.2:
            return True
            
        return False

    def filter_dataset(self, texts: List[str]) -> Tuple[List[str], Dict[str, int]]:
        """
        Filter an entire dataset, returning passing texts and stats.
        """
        passed = []
        stats = {
            "total": len(texts),
            "passed": 0,
            "rejected": 0,
            "reasons": {}
        }

        for text in texts:
            result = self.score(text)
            if not result.is_rejected:
                passed.append(text)
                stats["passed"] += 1
            else:
                stats["rejected"] += 1
                reason = result.rejection_reason or "unknown"
                stats["reasons"][reason] = stats["reasons"].get(reason, 0) + 1

        return passed, stats
