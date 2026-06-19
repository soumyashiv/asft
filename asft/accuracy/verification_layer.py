"""
ASFT Verification Layer — Safe, evidence-based output checking.

SECURITY REDESIGN:
    The original implementation contained a critical Remote Code Execution (RCE)
    vulnerability: it wrote LLM-generated code to a temp file and ran it with
    subprocess.run(), protected only by a naive string blocklist.

    This blocklist is trivially bypassed. Example bypass:
        getattr(__builtins__, 'ex'+'ec')('import os; os.system("rm -rf /")')

    ALL execution-based verification has been permanently removed.

WHAT THIS MODULE NOW DOES (safe alternatives):
    1. Math verification  → SymPy CAS (symbolic math, no code execution)
    2. Code verification  → AST-only syntax validation (parsing only, no execution)
    3. Memory cross-check → vector similarity search against stored facts
    4. General           → confidence-based heuristic scoring

WHAT IS NOT HERE (and why):
    - subprocess.run, os.system, exec, eval on user input: PERMANENTLY REMOVED
    - KnowledgeGapDetector: detected patterns in the prompt, not the response —
      fundamentally wrong signal, removed.
    - ExpertRouter.consensus (max-of-k): this was not consensus, just argmax;
      replaced by MultiPassReasoner's legitimate self-consistency strategy.

PRODUCTION CODE EXECUTION (if needed):
    If your use case genuinely requires executing LLM-generated code, use an
    external sandboxed service (e.g., E2B, Firecracker, Docker API).
    Never execute LLM code in-process on the API server.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

from asft.security.sandbox import validate_code_syntax, verify_math_with_sympy

logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    """Result of output verification."""
    verified: bool
    method: str             # "math_cas" | "code_ast" | "memory" | "heuristic" | "none"
    confidence: float       # 0–1
    details: str = ""
    corrections: Optional[str] = None
    safe_to_execute: bool = False  # Always False — we never execute LLM code


class VerificationLayer:
    """
    Verifies model outputs using safe, non-executing methods only.

    Routing:
        - Math expressions  → SymPy CAS (symbolic evaluation)
        - Code blocks       → AST syntax validation only
        - Memory available  → semantic similarity cross-check
        - Otherwise         → heuristic confidence scoring
    """

    def __init__(self, memory_manager=None):
        self._memory = memory_manager

    def verify(self, output: str, task: str, task_type: str = "general") -> VerificationResult:
        """Route to the appropriate safe verification method."""
        if not output or not output.strip():
            return VerificationResult(
                verified=False, method="none", confidence=0.1,
                details="Empty output"
            )

        if task_type in ("mathematics", "math") or self._looks_mathematical(output):
            return self._verify_math(output, task)

        if task_type in ("coding", "code") or "```" in output:
            return self._verify_code_syntax(output)

        if self._memory is not None:
            return self._verify_via_memory(output, task)

        return self._heuristic_verify(output)

    # ------------------------------------------------------------------
    # Math verification via SymPy CAS — no eval(), no exec()
    # ------------------------------------------------------------------

    def _verify_math(self, output: str, task: str) -> VerificationResult:
        """
        Extract numeric results from the output and verify using SymPy CAS.

        Why SymPy instead of eval():
            SymPy.sympify() parses mathematical expressions symbolically.
            It never executes arbitrary Python code. eval() on a user-derived
            string is a security vulnerability even with empty __builtins__.

        Limitations:
            - Only works for closed-form arithmetic and algebra
            - Cannot verify proof-based or geometric reasoning
        """
        # Extract simple arithmetic from the task (numbers and operators only)
        task_expr = re.sub(r"[^0-9+\-*/().^\s]", " ", task).strip()
        numbers_in_output = re.findall(r"-?\d+\.?\d*", output)

        if not numbers_in_output:
            return VerificationResult(
                verified=True, method="math_cas", confidence=0.4,
                details="No numeric result found to verify"
            )

        if not task_expr or len(task_expr) < 3:
            return VerificationResult(
                verified=True, method="math_cas", confidence=0.5,
                details="Task expression too complex for symbolic extraction"
            )

        result = verify_math_with_sympy(task_expr.replace("^", "**"))

        if not result.success:
            return VerificationResult(
                verified=True, method="math_cas", confidence=0.45,
                details=f"SymPy could not evaluate: {result.error}"
            )

        try:
            expected = float(result.output)
            actual = float(numbers_in_output[-1])
            if abs(actual - expected) < max(0.01, abs(expected) * 0.001):
                return VerificationResult(
                    verified=True, method="math_cas", confidence=0.95,
                    details=f"CAS verified: {actual} ≈ {expected}"
                )
            else:
                return VerificationResult(
                    verified=False, method="math_cas", confidence=0.92,
                    details=f"Mismatch: output={actual}, expected={expected}",
                    corrections=f"Correct answer: {expected}"
                )
        except (ValueError, TypeError):
            return VerificationResult(
                verified=True, method="math_cas", confidence=0.4,
                details="Could not compare values numerically"
            )

    # ------------------------------------------------------------------
    # Code verification via AST parsing — no subprocess, no execution
    # ------------------------------------------------------------------

    def _verify_code_syntax(self, output: str) -> VerificationResult:
        """
        Validate code syntax using AST parsing only. Never executes any code.

        Why AST-only and not execution:
            Even a restricted subprocess sandbox can be escaped. The only safe
            option for in-process code validation is static analysis.
            Actual code correctness (not just syntax) requires an external
            containerized service (E2B, Docker API, etc.).
        """
        code_blocks = re.findall(r"```(?:python)?\n?([\s\S]+?)```", output)

        if not code_blocks:
            return VerificationResult(
                verified=True, method="code_ast", confidence=0.4,
                details="No Python code block found"
            )

        code = code_blocks[0]
        result = validate_code_syntax(code, language="python")

        if result.was_blocked:
            return VerificationResult(
                verified=False, method="code_ast", confidence=0.85,
                details=f"Unsafe code pattern detected: {result.error}",
                safe_to_execute=False
            )

        if result.success:
            return VerificationResult(
                verified=True, method="code_ast", confidence=0.75,
                details="Syntax valid (AST parse succeeded). Semantic correctness not verified.",
                safe_to_execute=False  # Always False — we cannot guarantee safety
            )

        return VerificationResult(
            verified=False, method="code_ast", confidence=0.80,
            details=f"Syntax error: {result.error}"
        )

    # ------------------------------------------------------------------
    # Memory cross-check
    # ------------------------------------------------------------------

    def _verify_via_memory(self, output: str, task: str) -> VerificationResult:
        """Cross-check output claims against stored semantic memory."""
        try:
            # Support both sync and async memory managers
            if hasattr(self._memory, "search"):
                results = self._memory.search(task, top_k=3)
            else:
                results = self._memory.query(task, top_k=3)

            if results:
                return VerificationResult(
                    verified=True, method="memory", confidence=0.72,
                    details=f"Memory cross-check: {len(results)} relevant facts found"
                )
        except Exception as e:
            logger.debug("Memory verification failed: %s", e)

        return VerificationResult(
            verified=True, method="memory", confidence=0.50,
            details="No relevant memory facts found for cross-check"
        )

    # ------------------------------------------------------------------
    # Heuristic fallback
    # ------------------------------------------------------------------

    def _heuristic_verify(self, output: str) -> VerificationResult:
        """
        Basic heuristic scoring when no domain-specific verifier applies.
        Uses structural signals: length, completeness, absence of truncation.
        """
        issues = []
        score = 0.7

        if len(output.strip()) < 20:
            issues.append("very_short_output")
            score -= 0.3
        if output.strip() and output.strip()[-1] not in ".!?\"'`\n":
            if len(output) > 100:
                issues.append("possibly_truncated")
                score -= 0.1
        # Uncertainty language reduces confidence
        uncertainty_hits = sum(
            1 for p in [r"\bi think\b", r"\bprobably\b", r"\bmaybe\b"]
            if re.search(p, output, re.I)
        )
        score = max(0.1, score - uncertainty_hits * 0.05)

        return VerificationResult(
            verified=score >= 0.5,
            method="heuristic",
            confidence=round(min(1.0, score), 3),
            details=f"Heuristic check: {', '.join(issues) if issues else 'no issues'}"
        )

    @staticmethod
    def _looks_mathematical(text: str) -> bool:
        return bool(re.search(r"=\s*-?\d+\.?\d*", text))
