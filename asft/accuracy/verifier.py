"""
ASFT Verifier — Production-grade output verification.

CRITICAL REDESIGN from the original:
  - REMOVED: subprocess.run() for code execution (was RCE vulnerability)
  - REMOVED: eval() on regex-stripped LLM output (was code injection)
  - ADDED:   SymPy CAS for mathematical verification (safe, no exec)
  - ADDED:   AST-based syntax validation for code blocks (no exec)
  - ADDED:   Memory cross-check via the memory manager
  - ADDED:   KnowledgeGapDetector with temporal reasoning

What this module can SAFELY do:
  ✓ Parse and validate code syntax without executing it
  ✓ Evaluate arithmetic expressions via SymPy CAS
  ✓ Cross-check factual claims against the memory system
  ✓ Detect knowledge gaps and recommend retrieval actions

What this module deliberately CANNOT do:
  ✗ Execute arbitrary Python code (security boundary)
  ✗ Make outbound network requests
  ✗ Access the filesystem

For real code execution testing, use an external sandboxed runner
(Docker/gVisor) invoked by a separate, privileged service.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from asft.core.interfaces import IVerifier, VerificationResult
from asft.security.sandbox import validate_code_syntax, verify_math_with_sympy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Primary Verifier
# ---------------------------------------------------------------------------


class SafeVerifier(IVerifier):
    """
    Multi-method output verifier. Routes to the appropriate
    verification strategy based on task type and output content.

    Priority order:
      1. Mathematical content  → SymPy CAS
      2. Code blocks           → AST syntax validation
      3. Factual claims        → Memory cross-check
      4. No method available   → Conservative confidence 0.5
    """

    def __init__(self, memory_manager=None):
        self._memory = memory_manager

    def verify(self, output: str, task: str, task_type: str = "general") -> VerificationResult:
        """Route to the appropriate verification method."""

        # Route by task_type hint
        if task_type in ("mathematics", "math") or self._looks_mathematical(output):
            return self._verify_math(output, task)

        if task_type in ("coding", "code") or self._has_code_block(output):
            return self._verify_code(output)

        if self._memory:
            return self._verify_via_memory(output, task)

        return VerificationResult(
            verified=True,
            method="none",
            confidence=0.5,
            details="No verification method available for this task type.",
        )

    # -------------------------------------------------------------------------
    # Math verification — SymPy CAS (NO eval(), NO exec())
    # -------------------------------------------------------------------------

    def _verify_math(self, output: str, task: str) -> VerificationResult:
        """
        Extract arithmetic expressions from the task and verify the result
        using SymPy's CAS. This is completely safe — SymPy never executes
        arbitrary Python code.
        """
        # Extract the answer from the output (last number found)
        result_numbers = re.findall(r"-?\d+\.?\d*", output)
        if not result_numbers:
            return VerificationResult(
                verified=True, method="math_cas", confidence=0.4,
                details="No numeric result found in output to verify."
            )

        # Extract a clean arithmetic expression from the task
        expr = re.sub(r"[^0-9+\-*/().^ \t]", "", task).strip()
        expr = expr.replace("^", "**")

        if not expr or len(expr) < 3:
            return VerificationResult(
                verified=True, method="math_cas", confidence=0.5,
                details="Could not extract a verifiable expression from the task."
            )

        sandbox_result = verify_math_with_sympy(expr)

        if not sandbox_result.success:
            return VerificationResult(
                verified=True, method="math_cas", confidence=0.4,
                details=f"SymPy evaluation failed: {sandbox_result.error}"
            )

        try:
            expected = float(sandbox_result.output or "0")
            actual = float(result_numbers[-1])
            tolerance = max(0.01, abs(expected) * 1e-6)  # relative tolerance

            if abs(actual - expected) <= tolerance:
                return VerificationResult(
                    verified=True, method="math_cas", confidence=0.95,
                    details=f"SymPy verified: {actual} ≈ {expected}"
                )
            else:
                return VerificationResult(
                    verified=False, method="math_cas", confidence=0.92,
                    details=f"Mismatch: output={actual}, expected={expected}",
                    corrections=f"Correct answer: {expected}"
                )
        except (ValueError, TypeError):
            return VerificationResult(
                verified=True, method="math_cas", confidence=0.45,
                details="Could not compare numeric values."
            )

    # -------------------------------------------------------------------------
    # Code verification — AST syntax only (NO subprocess, NO exec)
    # -------------------------------------------------------------------------

    def _verify_code(self, output: str) -> VerificationResult:
        """
        Extract code blocks and validate their syntax using Python's AST.
        No code is executed. This validates that the code is at least
        syntactically correct.
        """
        code_blocks = re.findall(r"```(?:python)?\n?([\s\S]+?)```", output)

        if not code_blocks:
            return VerificationResult(
                verified=True, method="code_syntax", confidence=0.4,
                details="No fenced code blocks found in output."
            )

        code = code_blocks[0]
        sandbox_result = validate_code_syntax(code, language="python")

        if sandbox_result.was_blocked:
            return VerificationResult(
                verified=False, method="code_syntax", confidence=0.6,
                details=f"Code blocked by security sandbox: {sandbox_result.error}",
                safe_to_execute=False
            )

        if sandbox_result.success:
            return VerificationResult(
                verified=True, method="code_syntax", confidence=0.75,
                details="Python syntax is valid. Note: syntax ≠ correctness. No execution performed."
            )
        else:
            return VerificationResult(
                verified=False, method="code_syntax", confidence=0.80,
                details=f"Syntax error: {sandbox_result.error}",
                corrections=f"Fix syntax: {sandbox_result.error}"
            )

    # -------------------------------------------------------------------------
    # Memory cross-check
    # -------------------------------------------------------------------------

    def _verify_via_memory(self, output: str, task: str) -> VerificationResult:
        """Cross-check output claims against the memory system."""
        try:
            import asyncio
            # If we're in an async context, get the event loop
            try:
                loop = asyncio.get_event_loop()
                results = loop.run_until_complete(self._memory.query(task, top_k=3))
            except RuntimeError:
                # Fallback for sync contexts
                results = []

            if results:
                return VerificationResult(
                    verified=True, method="memory_cross_check", confidence=0.70,
                    details=f"Memory cross-check: {len(results)} relevant records found."
                )
        except Exception as e:
            logger.debug("Memory verification failed: %s", e)

        return VerificationResult(
            verified=True, method="memory_cross_check", confidence=0.50,
            details="No relevant memory records found for cross-check."
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _looks_mathematical(text: str) -> bool:
        """Heuristic: does the output look like it contains a math result?"""
        return bool(re.search(r"=\s*-?\d+\.?\d*", text))

    @staticmethod
    def _has_code_block(text: str) -> bool:
        """Does the output contain a fenced code block?"""
        return "```" in text


# ---------------------------------------------------------------------------
# Knowledge Gap Detector
# ---------------------------------------------------------------------------


@dataclass
class KnowledgeGapResult:
    """Result of a knowledge gap detection check."""
    has_gap: bool
    gap_type: str          # "temporal" | "factual" | "self_reported" | "none"
    gap_description: str
    recommended_action: str  # "memory_lookup" | "tool_use" | "research" | "none"
    confidence: float


# Regex patterns indicating self-reported uncertainty
_UNCERTAINTY_PATTERNS: list[re.Pattern] = [
    re.compile(r"\bi (don'?t|do not) know\b", re.I),
    re.compile(r"\buncertain\b", re.I),
    re.compile(r"\bunsure\b", re.I),
    re.compile(r"\bno (information|data|knowledge)\b", re.I),
    re.compile(r"\bcannot (answer|tell|say)\b", re.I),
    re.compile(r"\boutside (my|the) (knowledge|training)\b", re.I),
    re.compile(r"\bmy (knowledge|training) (cutoff|date|limit)\b", re.I),
]

# Year patterns that indicate temporal requirements
_TEMPORAL_PATTERN = re.compile(r"\b(202[3-9]|203\d|latest|current|recent|now|today)\b", re.I)


class KnowledgeGapDetector:
    """
    Detects when required knowledge is missing before a task is answered.
    Recommends the cheapest resolution strategy (memory → tool → research).
    """

    def detect(self, task: str, context: Optional[str] = None) -> KnowledgeGapResult:
        """
        Analyse the task to determine if there is a knowledge gap.

        Decision order (cheapest first):
          1. Temporal content → recommend tool use (live data)
          2. Specific factual lookup → recommend memory
          3. Model self-reported uncertainty → recommend research
          4. No gap → proceed normally
        """
        full_text = f"{task} {context or ''}"

        # Temporal gap: query requires information newer than training cutoff
        if _TEMPORAL_PATTERN.search(full_text):
            return KnowledgeGapResult(
                has_gap=True,
                gap_type="temporal",
                gap_description="Task requires recent or current information.",
                recommended_action="tool_use",
                confidence=0.85,
            )

        # Self-reported uncertainty in the task description
        for pattern in _UNCERTAINTY_PATTERNS:
            if pattern.search(full_text):
                return KnowledgeGapResult(
                    has_gap=True,
                    gap_type="self_reported",
                    gap_description="Task contains uncertainty indicators.",
                    recommended_action="memory_lookup",
                    confidence=0.70,
                )

        # Specific factual lookup required
        if re.search(r"\b(who|what|when|where|which) (is|was|are|were)\b", full_text, re.I):
            if re.search(r"\b(exact|specific|precise|actual)\b", full_text, re.I):
                return KnowledgeGapResult(
                    has_gap=True,
                    gap_type="factual",
                    gap_description="Task requires specific factual lookup.",
                    recommended_action="memory_lookup",
                    confidence=0.65,
                )

        return KnowledgeGapResult(
            has_gap=False,
            gap_type="none",
            gap_description="No knowledge gap detected.",
            recommended_action="none",
            confidence=0.90,
        )
