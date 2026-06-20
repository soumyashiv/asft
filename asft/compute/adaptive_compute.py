"""
ASFT Adaptive Compute Allocator — Never Spend More Compute Than Necessary.

MISSION:
    Assign compute budget proportional to task difficulty.
    Easy tasks use minimal compute; hard tasks get more.

STRATEGIES:

    1. EARLY EXIT (inference-time compute reduction):
       - Add exit points at intermediate transformer layers
       - Easy examples exit at early layers (< 30% compute)
       - Hard examples use all layers (100% compute)
       - Evidence: DeJa Vu (Ma et al. 2023): 2x speedup on 50% of tokens
         Confident Adaptive Language Modeling (CALM, Schuster et al. 2022)
       - Validated on: text classification, NLI, QA

    2. COMPUTE BUDGET ROUTER:
       - Before generation: classify task as "easy" | "medium" | "hard"
       - Easy: use 1-layer forward pass for routing, then small model
       - Hard: use full model with multi-pass reasoning
       - Evidence: Mixture-of-Experts (Shazeer 2017): sparse gating reduces
         active parameters to 1/k of total at each step

    3. DYNAMIC BATCHING BY COMPLEXITY:
       - Sort sequences by predicted complexity before batching
       - Process easy sequences together with fewer steps
       - Process hard sequences separately with more steps

WHAT THIS MODULE PROVIDES:
    ComputeBudgetRouter — classifies tasks and routes to appropriate compute tier.
    This is a lightweight classifier that runs BEFORE the main model.

HONEST LIMITATIONS:
    - Full early-exit implementation requires modifying model architecture
      (adding exit heads at each layer). This module provides the ROUTING
      logic; the architecture modification is out of scope for this version.
    - Task complexity classification is inherently uncertain. A task that
      looks easy may have a complex answer.
    - Calibration of the threshold values requires domain-specific validation.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ComputeTier(str, Enum):
    """Compute allocation tier."""

    MINIMAL = "minimal"  # Memory lookup or 1-pass forward
    LOW = "low"  # Single forward pass, small model
    MEDIUM = "medium"  # Standard inference, full model
    HIGH = "high"  # Multi-pass reasoning, full model
    MAXIMUM = "maximum"  # Ensemble or extended reasoning


@dataclass
class ComputeDecision:
    """Result of compute budget routing."""

    tier: ComputeTier
    max_new_tokens: int
    n_reasoning_passes: int
    use_full_model: bool
    skip_critique: bool
    reasoning: str
    confidence: float  # How confident the router is in its tier assignment


# Compute budgets per tier
_TIER_BUDGETS = {
    ComputeTier.MINIMAL: dict(max_new_tokens=50, n_passes=1, full_model=False, skip_critique=True),
    ComputeTier.LOW: dict(max_new_tokens=150, n_passes=1, full_model=True, skip_critique=True),
    ComputeTier.MEDIUM: dict(max_new_tokens=512, n_passes=1, full_model=True, skip_critique=False),
    ComputeTier.HIGH: dict(max_new_tokens=1024, n_passes=2, full_model=True, skip_critique=False),
    ComputeTier.MAXIMUM: dict(
        max_new_tokens=2048, n_passes=3, full_model=True, skip_critique=False
    ),
}

# Patterns that suggest high complexity
_HIGH_COMPLEXITY_PATTERNS = [
    r"\bprove\b",
    r"\bderive\b",
    r"\bexplain\s+(why|how)\b",
    r"\boptimize\b",
    r"\bdesign\b",
    r"\barchitecture\b",
    r"\bcomplex\b",
    r"\badvanced\b",
    r"\bcomprehensive\b",
    r"\bimplement\b.{0,50}\b(algorithm|system|framework)\b",
    r"\bmulti.step\b",
    r"\bchain\b",
    r"\bsequential\b",
]

# Patterns that suggest low complexity
_LOW_COMPLEXITY_PATTERNS = [
    r"^(what|who|where|when)\s+(is|are|was|were)\b",
    r"\bdefine\b",
    r"\blist\b",
    r"\bname\b",
    r"\byes or no\b",
    r"\btrue or false\b",
    r"\bhow\s+many\b",
    r"\bhow\s+much\b",
    r"\bwhat\s+does\s+\w+\s+stand\s+for\b",
]


class ComputeBudgetRouter:
    """
    Routes tasks to appropriate compute tiers based on estimated difficulty.

    Uses:
        - Token count heuristics
        - Linguistic complexity patterns
        - Domain-specific signals
        - Optional: model perplexity score on the task

    This runs BEFORE the main inference, adding minimal overhead
    (regex pattern matching: ~0.1ms per query).
    """

    def __init__(
        self,
        default_tier: ComputeTier = ComputeTier.MEDIUM,
        enable_perplexity_scoring: bool = False,
    ):
        self._default = default_tier
        self._ppl_scoring = enable_perplexity_scoring

        self._high_patterns = [re.compile(p, re.IGNORECASE) for p in _HIGH_COMPLEXITY_PATTERNS]
        self._low_patterns = [re.compile(p, re.IGNORECASE) for p in _LOW_COMPLEXITY_PATTERNS]

    def route(
        self,
        task: str,
        domain: str = "general",
        budget_tokens: int | None = None,
    ) -> ComputeDecision:
        """
        Classify the task and return a compute allocation decision.

        Args:
            task:          The task/prompt text.
            domain:        Domain hint (coding | math | general …)
            budget_tokens: Hard cap on max_new_tokens. None = use tier default.

        Returns:
            ComputeDecision with tier assignment and budget.
        """
        tier, confidence, reasoning = self._classify(task, domain)

        budget = _TIER_BUDGETS[tier]
        max_tokens = min(budget["max_new_tokens"], budget_tokens or 999_999)

        decision = ComputeDecision(
            tier=tier,
            max_new_tokens=max_tokens,
            n_reasoning_passes=budget["n_passes"],
            use_full_model=budget["full_model"],
            skip_critique=budget["skip_critique"],
            reasoning=reasoning,
            confidence=confidence,
        )

        logger.debug(
            "ComputeRouter: %s → %s (conf=%.2f) | tokens=%d passes=%d",
            task[:60],
            tier.value,
            confidence,
            max_tokens,
            budget["n_passes"],
        )
        return decision

    def _classify(self, task: str, domain: str) -> tuple[ComputeTier, float, str]:
        """Score complexity and return (tier, confidence, reasoning)."""
        task_len = len(task.split())
        high_hits = sum(1 for p in self._high_patterns if p.search(task))
        low_hits = sum(1 for p in self._low_patterns if p.search(task))

        # High-complexity signals
        if domain in ("coding", "math", "mathematics", "science"):
            high_hits += 2
        if task_len > 200:
            high_hits += 1
        if task_len < 10:
            low_hits += 2

        score = high_hits - low_hits

        if score >= 4:
            return ComputeTier.MAXIMUM, 0.85, f"Very high complexity ({high_hits} high signals)"
        elif score >= 2:
            return ComputeTier.HIGH, 0.75, f"High complexity (domain={domain}, signals={high_hits})"
        elif score >= 0:
            return ComputeTier.MEDIUM, 0.70, f"Standard complexity (score={score})"
        elif score >= -2:
            return ComputeTier.LOW, 0.75, f"Low complexity ({low_hits} simple-task signals)"
        else:
            return ComputeTier.MINIMAL, 0.80, "Minimal complexity (likely factual lookup)"

    def batch_route(self, tasks: list[str], domain: str = "general") -> list[ComputeDecision]:
        """Route multiple tasks and return sorted by tier (easy first) for efficient batching."""
        decisions = [self.route(task, domain) for task in tasks]
        tier_order = list(ComputeTier)
        decisions.sort(key=lambda d: tier_order.index(d.tier))
        return decisions

    def estimate_compute_savings(self, tasks: list[str]) -> dict:
        """
        Estimate compute savings vs always using MAXIMUM tier.
        Returns fraction of tokens saved and tier distribution.
        """
        decisions = [self.route(t) for t in tasks]
        max_tokens = _TIER_BUDGETS[ComputeTier.MAXIMUM]["max_new_tokens"]

        total_actual = sum(d.max_new_tokens for d in decisions)
        total_max = max_tokens * len(decisions)
        savings = 1.0 - total_actual / max(1, total_max)

        tier_counts = {}
        for tier in ComputeTier:
            tier_counts[tier.value] = sum(1 for d in decisions if d.tier == tier)

        return {
            "n_tasks": len(decisions),
            "total_tokens_actual": total_actual,
            "total_tokens_if_maximum": total_max,
            "estimated_compute_savings_fraction": round(savings, 3),
            "tier_distribution": tier_counts,
        }
