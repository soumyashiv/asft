"""
Self-Critique Engine — Reviews model outputs before finalizing.
Detects logical errors, contradictions, hallucinations, and weak reasoning.
Automatically revises answers when issues are detected.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CritiqueResult:
    original_output: str
    revised_output: str
    issues_found: list[str]
    was_revised: bool
    critique_rounds: int
    quality_improvement: float  # estimated improvement 0–1

    @property
    def is_clean(self) -> bool:
        return len(self.issues_found) == 0


class SelfCritiqueEngine:
    """
    Reviews and revises model outputs to detect and fix:
      - Logical errors and contradictions
      - Hallucination indicators
      - Missing information
      - Weak or unsupported reasoning
      - Inconsistent statements

    Uses a generate_fn (model inference callable) to produce revisions.
    """

    def __init__(self, max_rounds: int = 2):
        self._max_rounds = max_rounds
        self._issue_detectors = [
            self._detect_contradictions,
            self._detect_hallucination_markers,
            self._detect_logical_gaps,
            self._detect_incomplete_response,
        ]

    def critique(
        self,
        output: str,
        original_task: str,
        generate_fn: Callable[[str], str] | None = None,
    ) -> CritiqueResult:
        """
        Critique an output and optionally revise it.

        Args:
            output: the original model output to critique
            original_task: the original task/question
            generate_fn: optional callable(prompt) → str for generating revisions

        Returns:
            CritiqueResult with original, revised, and issue list
        """
        current = output
        all_issues: list[str] = []
        rounds = 0
        was_revised = False

        for round_num in range(self._max_rounds):
            issues = self._detect_issues(current)
            if not issues:
                logger.debug("SelfCritique round %d: no issues found", round_num)
                break

            all_issues.extend(issues)
            logger.info(
                "SelfCritique round %d: %d issues found: %s", round_num, len(issues), issues
            )

            if generate_fn is None:
                # No model available — mark issues but cannot revise
                break

            # Generate a revised version
            revision_prompt = self._build_revision_prompt(
                original_task=original_task,
                current_output=current,
                issues=issues,
            )
            revised = generate_fn(revision_prompt)
            if revised and revised.strip() and revised != current:
                current = revised
                was_revised = True
                rounds += 1
            else:
                break

        quality_improvement = min(0.3, 0.1 * rounds) if was_revised else 0.0

        return CritiqueResult(
            original_output=output,
            revised_output=current,
            issues_found=list(set(all_issues)),
            was_revised=was_revised,
            critique_rounds=rounds,
            quality_improvement=quality_improvement,
        )

    def _detect_issues(self, text: str) -> list[str]:
        issues = []
        for detector in self._issue_detectors:
            found = detector(text)
            if found:
                issues.extend(found)
        return issues

    def _detect_contradictions(self, text: str) -> list[str]:
        """Detect obvious contradictions using negation patterns."""
        sentences = [s.strip() for s in re.split(r"[.!?]", text) if len(s.strip()) > 10]
        issues = []
        for i in range(len(sentences)):
            for j in range(i + 1, min(i + 5, len(sentences))):
                a, b = sentences[i].lower(), sentences[j].lower()
                # Simple: check if one sentence negates a claim from another
                a_words = set(a.split())
                b_words = set(b.split())
                shared = a_words & b_words
                # Check if "not" appears near shared words
                if len(shared) > 3 and ("not" in b_words) != ("not" in a_words):
                    issues.append(f"potential_contradiction: '{sentences[i][:40]}...'")
                    break
        return issues[:2]  # Cap at 2 to avoid noise

    def _detect_hallucination_markers(self, text: str) -> list[str]:
        """Detect patterns commonly associated with hallucinations."""
        patterns = [
            (r"\brecently published\b.*\bstudy\b", "unsourced_study_claim"),
            (r"\bexperts agree\b", "vague_expert_claim"),
            (r"\b(all|every|always|never)\b.*\b(are|is|do|does)\b", "absolute_claim"),
            (r"\bscientifically proven\b", "unverified_scientific_claim"),
        ]
        issues = []
        for pattern, label in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                issues.append(label)
        return issues

    def _detect_logical_gaps(self, text: str) -> list[str]:
        """Detect reasoning gaps — conclusions without supporting reasoning."""
        issues = []
        has_conclusion = any(
            w in text.lower() for w in ["therefore", "thus", "hence", "so", "conclusion"]
        )
        has_reasoning = any(
            w in text.lower() for w in ["because", "since", "given that", "due to", "as a result"]
        )
        if has_conclusion and not has_reasoning and len(text) > 100:
            issues.append("conclusion_without_reasoning")
        return issues

    def _detect_incomplete_response(self, text: str) -> list[str]:
        """Detect suspiciously short or abruptly cut-off responses."""
        issues = []
        if len(text.strip()) < 20:
            issues.append("response_too_short")
        if text.strip() and text.strip()[-1] not in ".!?\"'`":
            if len(text) > 50:  # Not just a very short answer
                issues.append("response_appears_truncated")
        return issues

    def _build_revision_prompt(
        self, original_task: str, current_output: str, issues: list[str]
    ) -> str:
        issues_str = "\n".join(f"  - {issue}" for issue in issues)
        return (
            f"Review and improve this response. The following issues were detected:\n"
            f"{issues_str}\n\n"
            f"Original task: {original_task}\n\n"
            f"Current response:\n{current_output}\n\n"
            f"Provide an improved response that fixes the identified issues. "
            f"Be accurate, clear, and complete."
        )
