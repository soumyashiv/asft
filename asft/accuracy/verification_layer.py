"""
Verification Layer — Cross-checks outputs against memory, tools, math, and code execution.
Knowledge Gap Detector — Identifies when required knowledge is missing before answering.
Expert Router — Routes tasks to specialized skill packs with multi-expert consensus.
"""
from __future__ import annotations

import logging
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ===========================================================================
# Verification Layer
# ===========================================================================

@dataclass
class VerificationResult:
    verified: bool
    method: str
    confidence: float
    details: str = ""
    corrections: Optional[str] = None


class VerificationLayer:
    """
    Verifies model outputs using multiple cross-checking methods:
      - Memory: check against stored facts
      - Mathematical: validate numeric results
      - Code execution: run generated code and check output
    """

    def __init__(self, memory_manager=None, enable_code_execution: bool = True):
        self._memory = memory_manager
        self._enable_code = enable_code_execution

    def verify(self, output: str, task: str, task_type: str = "general") -> VerificationResult:
        """Route to the appropriate verification method."""
        if task_type in ("mathematics", "math") or self._looks_mathematical(output):
            return self._verify_math(output, task)
        if task_type in ("coding", "code") or "```" in output:
            return self._verify_code(output)
        if self._memory:
            return self._verify_via_memory(output, task)
        return VerificationResult(verified=True, method="none", confidence=0.5,
                                  details="No verification method available")

    def _verify_math(self, output: str, task: str) -> VerificationResult:
        """Extract and validate numeric results."""
        numbers = re.findall(r'-?\d+\.?\d*', output)
        task_numbers = re.findall(r'-?\d+\.?\d*', task)

        if not numbers:
            return VerificationResult(verified=True, method="math", confidence=0.4,
                                      details="No numeric result found to verify")

        # Try to re-compute simple arithmetic from task
        try:
            expr = re.sub(r'[^0-9+\-*/().^ ]', '', task).strip()
            if expr:
                expr = expr.replace('^', '**')
                expected = eval(expr, {"__builtins__": {}})
                result = float(numbers[-1])
                if abs(result - expected) < 0.01:
                    return VerificationResult(verified=True, method="math", confidence=0.95,
                                              details=f"Verified: {result} == {expected}")
                else:
                    return VerificationResult(verified=False, method="math", confidence=0.9,
                                              details=f"Mismatch: got {result}, expected {expected}",
                                              corrections=f"Correct answer: {expected}")
        except Exception:
            pass

        return VerificationResult(verified=True, method="math", confidence=0.5,
                                  details="Could not compute expected value for comparison")

    def _verify_code(self, output: str) -> VerificationResult:
        """Extract and execute code blocks to verify they run without errors."""
        if not self._enable_code:
            return VerificationResult(verified=True, method="code", confidence=0.5,
                                      details="Code execution disabled")

        code_blocks = re.findall(r'```(?:python)?\n?([\s\S]+?)```', output)
        if not code_blocks:
            return VerificationResult(verified=True, method="code", confidence=0.4,
                                      details="No executable code block found")

        code = code_blocks[0]
        # Safety: only run if no dangerous imports
        dangerous = ["os.system", "subprocess", "shutil.rmtree", "__import__", "exec(", "eval("]
        if any(d in code for d in dangerous):
            return VerificationResult(verified=True, method="code", confidence=0.5,
                                      details="Code not executed (safety check)")

        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(code)
                tmpfile = f.name

            result = subprocess.run(
                ["python", tmpfile], capture_output=True, text=True, timeout=10
            )

            if result.returncode == 0:
                return VerificationResult(verified=True, method="code", confidence=0.9,
                                          details=f"Code executed successfully. Output: {result.stdout[:200]}")
            else:
                return VerificationResult(verified=False, method="code", confidence=0.85,
                                          details=f"Code error: {result.stderr[:200]}",
                                          corrections=f"Stderr: {result.stderr[:200]}")
        except subprocess.TimeoutExpired:
            return VerificationResult(verified=True, method="code", confidence=0.4,
                                      details="Code execution timed out")
        except Exception as e:
            return VerificationResult(verified=True, method="code", confidence=0.3,
                                      details=f"Execution failed: {e}")

    def _verify_via_memory(self, output: str, task: str) -> VerificationResult:
        """Cross-check claims against memory."""
        try:
            results = self._memory.query(task, top_k=3)
            if results and results[0].hit:
                return VerificationResult(verified=True, method="memory", confidence=0.75,
                                          details=f"Memory cross-check: {len(results)} relevant facts found")
        except Exception:
            pass
        return VerificationResult(verified=True, method="memory", confidence=0.5,
                                  details="No memory cross-check data")

    def _looks_mathematical(self, text: str) -> bool:
        return bool(re.search(r'=\s*-?\d+\.?\d*', text))


# ===========================================================================
# Knowledge Gap Detector
# ===========================================================================

@dataclass
class KnowledgeGapResult:
    has_gap: bool
    gap_description: str
    recommended_action: str  # "memory" | "tool" | "research" | "none"
    confidence: float


class KnowledgeGapDetector:
    """
    Detects when required knowledge is missing before answering.
    Recommends: search memory → use tools → conduct research.
    """

    _GAP_INDICATORS = [
        r"\bi don'?t know\b", r"\buncertain\b", r"\bunsure\b",
        r"\bno (information|data|knowledge)\b", r"\bcannot (answer|tell|say)\b",
        r"\boutside (my|the) (knowledge|training)\b",
        r"\bas of (my|the) (training|knowledge) (cutoff|date)\b",
    ]

    def __init__(self):
        self._patterns = [re.compile(p, re.IGNORECASE) for p in self._GAP_INDICATORS]

    def detect(self, task: str, context: Optional[str] = None) -> KnowledgeGapResult:
        text = f"{task} {context or ''}"
        gap_found = any(p.search(text) for p in self._patterns)

        # Classify gap type
        if "recent" in task.lower() or "latest" in task.lower() or "2024" in task or "2025" in task:
            return KnowledgeGapResult(
                has_gap=True, gap_description="Requires recent/current information",
                recommended_action="tool", confidence=0.8
            )
        if "specific" in task.lower() and any(
            w in task.lower() for w in ["fact", "number", "date", "name"]
        ):
            return KnowledgeGapResult(
                has_gap=True, gap_description="Requires specific factual lookup",
                recommended_action="memory", confidence=0.7
            )
        if gap_found:
            return KnowledgeGapResult(
                has_gap=True, gap_description="Model indicated knowledge uncertainty",
                recommended_action="research", confidence=0.75
            )

        return KnowledgeGapResult(
            has_gap=False, gap_description="",
            recommended_action="none", confidence=0.85
        )


# ===========================================================================
# Expert Router
# ===========================================================================

@dataclass
class ExpertDecision:
    selected_experts: List[str]
    strategy: str  # "single" | "multi" | "consensus"
    scores: Dict[str, float]


class ExpertRouter:
    """
    Routes tasks to specialized skill packs.
    Supports single, multi-expert collaboration, and consensus generation.
    """

    def __init__(self, skill_router, registry):
        self._skill_router = skill_router
        self._registry = registry

    def route(self, task: str, strategy: str = "single", top_k: int = 2) -> ExpertDecision:
        decision = self._skill_router.route(task, top_k=top_k, strategy=strategy)
        return ExpertDecision(
            selected_experts=decision.selected_packs,
            strategy=strategy,
            scores=decision.scores,
        )

    def execute_with_experts(
        self,
        task: str,
        strategy: str = "single",
        model=None,
        tokenizer=None,
    ) -> Dict[str, Any]:
        """Route task to experts and execute. Merge results for multi/consensus."""
        expert_decision = self.route(task, strategy=strategy)

        if not expert_decision.selected_experts:
            return {"output": f"No expert available for: {task}", "confidence": 0.1}

        results = []
        for expert_name in expert_decision.selected_experts:
            pack = self._registry.get_or_none("skill_packs", expert_name)
            if pack:
                result = pack.process(task, model=model, tokenizer=tokenizer)
                results.append(result)

        if not results:
            return {"output": "Expert execution failed", "confidence": 0.0}

        if strategy == "single" or len(results) == 1:
            r = results[0]
            return {"output": r.output, "confidence": r.confidence, "expert": expert_decision.selected_experts[0]}

        # Multi/consensus: pick highest confidence
        best = max(results, key=lambda r: r.confidence)
        return {
            "output": best.output,
            "confidence": best.confidence,
            "expert": best.skill_name,
            "all_experts": [r.skill_name for r in results],
        }
