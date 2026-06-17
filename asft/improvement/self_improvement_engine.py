"""
Self-Improvement Engine — Analyzes completed tasks, detects failures and
inefficiencies, and improves prompts, workflows, and planning strategies.
Prefers workflow optimization over model retraining.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ===========================================================================
# Task Analyzer
# ===========================================================================

@dataclass
class TaskAnalysisReport:
    total_tasks: int
    failure_rate: float
    bottlenecks: List[str]
    repeated_mistakes: List[str]
    recommendations: List[str]
    timestamp: float = field(default_factory=time.time)


class TaskAnalyzer:
    """Analyzes episodic memory to detect failure patterns and bottlenecks."""

    def __init__(self, episodic_memory):
        self._episodic = episodic_memory

    def analyze(self, window_hours: float = 72.0, min_count: int = 3) -> TaskAnalysisReport:
        since = time.time() - window_hours * 3600
        events = self._episodic.query(since_timestamp=since, limit=2000)
        total = len(events)
        if total == 0:
            return TaskAnalysisReport(0, 0.0, [], [], [])

        failures = [e for e in events if not e.get("success", True)]
        failure_rate = len(failures) / total

        # Detect repeated mistakes by event type
        from collections import Counter
        failure_types = Counter(e["event_type"] for e in failures)
        repeated = [t for t, c in failure_types.most_common() if c >= min_count]

        # Detect bottlenecks by duration
        slow_events = sorted(events, key=lambda e: e.get("duration_seconds", 0), reverse=True)
        bottlenecks = [e["event_type"] for e in slow_events[:5]]

        # Generate recommendations
        recommendations = []
        if failure_rate > 0.3:
            recommendations.append("High failure rate — consider prompt optimization")
        if repeated:
            recommendations.append(f"Repeated failures in: {repeated[:3]} — update skill packs")
        if bottlenecks:
            recommendations.append(f"Slow tasks: {bottlenecks[:3]} — consider workflow optimization")
        if failure_rate < 0.05:
            recommendations.append("Performance is healthy — no immediate action required")

        return TaskAnalysisReport(
            total_tasks=total,
            failure_rate=failure_rate,
            bottlenecks=bottlenecks,
            repeated_mistakes=repeated,
            recommendations=recommendations,
        )


# ===========================================================================
# Prompt Optimizer
# ===========================================================================

@dataclass
class PromptOptimizationResult:
    original_prompt: str
    best_prompt: str
    best_score: float
    all_candidates: List[Dict[str, Any]]
    improvement: float


class PromptOptimizer:
    """
    Evolutionary prompt improvement.
    Generates N variant prompts, benchmarks them, keeps the best.
    """

    def __init__(self, n_variants: int = 5, eval_samples: int = 10):
        self._n_variants = n_variants
        self._eval_samples = eval_samples

    def optimize(self, prompt_template: str, eval_fn, score_fn) -> PromptOptimizationResult:
        """
        Optimize a prompt template.
        eval_fn(prompt) → str: generates output from prompt
        score_fn(output) → float: scores the output quality
        """
        # Generate variants
        variants = self._generate_variants(prompt_template)
        candidates = []

        for variant in variants:
            try:
                output = eval_fn(variant)
                score = score_fn(output)
                candidates.append({"prompt": variant, "score": score, "output": output})
            except Exception as e:
                logger.warning("Variant failed: %s", e)

        if not candidates:
            return PromptOptimizationResult(prompt_template, prompt_template, 0.0, [], 0.0)

        candidates.sort(key=lambda c: c["score"], reverse=True)
        best = candidates[0]

        # Try original too
        try:
            orig_output = eval_fn(prompt_template)
            orig_score = score_fn(orig_output)
        except Exception:
            orig_score = 0.0

        improvement = best["score"] - orig_score

        return PromptOptimizationResult(
            original_prompt=prompt_template,
            best_prompt=best["prompt"],
            best_score=best["score"],
            all_candidates=candidates,
            improvement=improvement,
        )

    def _generate_variants(self, template: str) -> List[str]:
        """Generate prompt variants through simple transformations."""
        variants = [template]
        # Variant 1: Add "Think step by step" prefix
        variants.append(f"Think step by step. {template}")
        # Variant 2: Add explicit output format request
        variants.append(f"{template}\n\nProvide a clear, structured, comprehensive answer.")
        # Variant 3: Role-play framing
        variants.append(f"As an expert in this domain, {template.lower()}")
        # Variant 4: Chain-of-thought
        variants.append(f"{template}\n\nReasoning: Let me think through this carefully.")
        return variants[:self._n_variants]


# ===========================================================================
# Workflow Optimizer
# ===========================================================================

@dataclass
class WorkflowOptimizationResult:
    original_workflow: Dict[str, Any]
    optimized_workflow: Dict[str, Any]
    changes_made: List[str]
    estimated_improvement: str


class WorkflowOptimizer:
    """
    Analyzes task execution graphs to identify and remove inefficient steps.
    Rewrites pipelines for speed and reliability.
    """

    def optimize(self, workflow: Dict[str, Any],
                 performance_data: Optional[Dict] = None) -> WorkflowOptimizationResult:
        changes = []
        optimized = dict(workflow)

        steps = workflow.get("steps", [])
        if not steps:
            return WorkflowOptimizationResult(workflow, optimized, [], "No steps to optimize")

        # Rule 1: Remove redundant sequential memory queries
        new_steps = self._remove_redundant_steps(steps, changes)

        # Rule 2: Parallelize independent steps
        new_steps = self._suggest_parallelization(new_steps, changes)

        # Rule 3: Cache expensive operations
        new_steps = self._add_caching_hints(new_steps, changes)

        optimized["steps"] = new_steps
        estimated = f"{len(changes)} optimizations applied"
        return WorkflowOptimizationResult(workflow, optimized, changes, estimated)

    def _remove_redundant_steps(self, steps: List[Dict], changes: List[str]) -> List[Dict]:
        seen_types = set()
        new_steps = []
        for step in steps:
            stype = step.get("type", "unknown")
            if stype in seen_types and step.get("cache_ok", True):
                changes.append(f"Removed redundant step: {stype}")
                continue
            seen_types.add(stype)
            new_steps.append(step)
        return new_steps

    def _suggest_parallelization(self, steps: List[Dict], changes: List[str]) -> List[Dict]:
        for i, step in enumerate(steps):
            if step.get("depends_on") is None and i > 0:
                step["parallel_candidate"] = True
                changes.append(f"Step {i} ({step.get('type')}) can be parallelized")
        return steps

    def _add_caching_hints(self, steps: List[Dict], changes: List[str]) -> List[Dict]:
        expensive = {"vector_search", "embedding", "model_inference"}
        for step in steps:
            if step.get("type") in expensive and not step.get("cached"):
                step["cache_result"] = True
                changes.append(f"Added caching to {step.get('type')} step")
        return steps


# ===========================================================================
# Self-Improvement Engine (Orchestrator)
# ===========================================================================

@dataclass
class ImprovementDecision:
    action: str  # "memory" | "workflow" | "prompt" | "skill_pack" | "train" | "none"
    reason: str
    priority: str  # "high" | "medium" | "low"
    estimated_gain: str


class SelfImprovementEngine:
    """
    Master orchestrator. Decides what to improve and how.
    Prefers: memory → workflow → prompt → skill pack → training.
    """

    def __init__(self, episodic_memory, long_term_memory, config=None):
        self._episodic = episodic_memory
        self._long_term = long_term_memory
        self._analyzer = TaskAnalyzer(episodic_memory)
        self._prompt_optimizer = PromptOptimizer()
        self._workflow_optimizer = WorkflowOptimizer()
        self._improvement_log: List[Dict] = []
        self._output_dir = Path(getattr(config, "data_dir", "./asft_data") if config else "./asft_data") / "improvements"
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def analyze_and_decide(self) -> ImprovementDecision:
        """Analyze current state and decide on the best improvement action."""
        report = self._analyzer.analyze()
        logger.info("Improvement analysis: failure_rate=%.2f", report.failure_rate)

        if report.total_tasks < 10:
            return ImprovementDecision("none", "Insufficient data", "low", "Collect more task data")

        if report.failure_rate > 0.5:
            return ImprovementDecision(
                "prompt", "High failure rate — prompt optimization is fastest fix",
                "high", "Expected 20–40% failure rate reduction"
            )
        if report.failure_rate > 0.2:
            return ImprovementDecision(
                "workflow", "Moderate failures — workflow can be optimized",
                "medium", "Expected 10–20% improvement"
            )
        if report.repeated_mistakes:
            return ImprovementDecision(
                "skill_pack", f"Repeated mistakes in {report.repeated_mistakes[:2]}",
                "medium", "Targeted skill pack update expected"
            )
        if report.bottlenecks:
            return ImprovementDecision(
                "workflow", f"Bottlenecks detected in {report.bottlenecks[:2]}",
                "low", "Expected speed improvement"
            )

        return ImprovementDecision("none", "System performing well", "low", "No action needed")

    def run_improvement_cycle(self) -> Dict[str, Any]:
        """Run one full improvement cycle."""
        decision = self.analyze_and_decide()
        result = {
            "decision": decision.action,
            "reason": decision.reason,
            "priority": decision.priority,
            "timestamp": time.time(),
        }

        logger.info("Improvement decision: %s — %s", decision.action, decision.reason)

        # Log improvement
        self._improvement_log.append(result)
        log_path = self._output_dir / "improvement_log.jsonl"
        with open(log_path, "a") as f:
            f.write(json.dumps(result) + "\n")

        return result

    def get_improvement_history(self) -> List[Dict]:
        return list(self._improvement_log)
