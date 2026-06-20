"""
Skill Router — Automatically routes tasks to the best skill pack(s).
Uses embedding similarity + learned routing weights for expert selection.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RoutingDecision:
    task_input: str
    selected_packs: list[str]
    scores: dict[str, float]
    strategy: str  # "single" | "multi" | "consensus"
    reason: str = ""


# Keyword-based domain hints for fast routing
_DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "coding": [
        "code",
        "python",
        "javascript",
        "function",
        "class",
        "bug",
        "debug",
        "algorithm",
        "implement",
        "script",
        "api",
        "compile",
        "syntax",
        "error",
        "program",
        "software",
        "github",
        "git",
        "sql",
        "query",
    ],
    "research": [
        "research",
        "study",
        "analyze",
        "paper",
        "literature",
        "find",
        "search",
        "investigate",
        "explore",
        "discover",
        "survey",
        "evidence",
        "data",
        "statistics",
        "hypothesis",
        "experiment",
    ],
    "planning": [
        "plan",
        "schedule",
        "roadmap",
        "steps",
        "strategy",
        "milestone",
        "task",
        "project",
        "timeline",
        "organize",
        "priority",
        "goal",
        "objective",
        "workflow",
        "process",
        "breakdown",
    ],
    "mathematics": [
        "calculate",
        "compute",
        "math",
        "equation",
        "formula",
        "integral",
        "derivative",
        "algebra",
        "geometry",
        "statistics",
        "probability",
        "linear",
        "matrix",
        "solve",
        "proof",
        "theorem",
    ],
    "trading": [
        "trade",
        "stock",
        "market",
        "price",
        "portfolio",
        "investment",
        "forex",
        "crypto",
        "option",
        "futures",
        "technical",
        "fundamental",
        "chart",
        "trend",
        "indicator",
        "signal",
        "risk",
    ],
    "automation": [
        "automate",
        "script",
        "workflow",
        "trigger",
        "schedule",
        "repeat",
        "batch",
        "pipeline",
        "task",
        "cron",
        "bot",
        "agent",
        "tool",
        "process",
        "deploy",
    ],
}


class SkillRouter:
    """
    Routes an incoming task to one or more skill packs.

    Routing strategy:
      1. Keyword/domain scoring (fast, no ML)
      2. Embedding similarity (accurate, requires sentence-transformer)
      3. Learned routing weights (from performance history)

    Combines all three for final decision.
    """

    def __init__(self, registry, config=None, embedding_model=None):
        self._registry = registry
        self._config = config
        self._embedder = embedding_model  # Optional EmbeddingModel
        # FIX F10: bounded deque prevents unbounded memory growth
        self._routing_history: deque = deque(maxlen=10_000)
        self._skill_embeddings: dict[str, list[float]] = {}

    def route(
        self,
        task_input: str,
        top_k: int = 1,
        strategy: str = "single",
    ) -> RoutingDecision:
        """
        Route a task to the best skill pack(s).
        strategy: "single" | "multi" | "consensus"
        """
        available_skills = self._registry.list("skill_packs")
        if not available_skills:
            logger.warning("No skill packs registered.")
            return RoutingDecision(
                task_input=task_input,
                selected_packs=[],
                scores={},
                strategy=strategy,
                reason="No skill packs available",
            )

        # Step 1: keyword scoring
        keyword_scores = self._keyword_score(task_input, available_skills)

        # Step 2: embedding similarity (if embedder available)
        emb_scores = self._embedding_score(task_input, available_skills)

        # Step 3: performance-weighted routing
        perf_scores = self._performance_score(available_skills)

        # Combine scores
        combined: dict[str, float] = {}
        for skill in available_skills:
            combined[skill] = (
                0.4 * keyword_scores.get(skill, 0.0)
                + 0.4 * emb_scores.get(skill, 0.0)
                + 0.2 * perf_scores.get(skill, 0.0)
            )

        sorted_skills = sorted(combined.items(), key=lambda x: x[1], reverse=True)

        if strategy == "single":
            selected = [sorted_skills[0][0]] if sorted_skills else []
        elif strategy == "multi":
            selected = [name for name, _ in sorted_skills[:top_k]]
        else:  # consensus
            # Select all with score above half of max
            max_score = sorted_skills[0][1] if sorted_skills else 0
            threshold = max_score * 0.5
            selected = [name for name, score in sorted_skills if score >= threshold]

        decision = RoutingDecision(
            task_input=task_input,
            selected_packs=selected,
            scores=combined,
            strategy=strategy,
            reason=f"keyword={keyword_scores}, embedding={emb_scores}",
        )
        self._routing_history.append({"decision": decision, "input_snippet": task_input[:100]})
        logger.info(
            "Routed to: %s (scores=%s)", selected, {k: round(v, 3) for k, v in combined.items()}
        )
        return decision

    def _keyword_score(self, text: str, skills: list[str]) -> dict[str, float]:
        text_lower = text.lower()
        scores: dict[str, float] = {}
        for skill in skills:
            domain = skill.split(".")[-1] if "." in skill else skill
            keywords = _DOMAIN_KEYWORDS.get(domain, [])
            hit_count = sum(1 for kw in keywords if kw in text_lower)
            scores[skill] = min(1.0, hit_count / max(1, len(keywords) * 0.3))
        return scores

    def _embedding_score(self, text: str, skills: list[str]) -> dict[str, float]:
        if self._embedder is None:
            return {s: 0.0 for s in skills}
        try:
            import numpy as np

            query_emb = self._embedder.encode_one(text)

            scores: dict[str, float] = {}
            for skill in skills:
                if skill not in self._skill_embeddings:
                    # Build description from domain keywords
                    domain = skill.split(".")[-1] if "." in skill else skill
                    desc = " ".join(_DOMAIN_KEYWORDS.get(domain, [domain]))
                    self._skill_embeddings[skill] = self._embedder.encode_one(desc)

                skill_emb = self._skill_embeddings[skill]
                q = np.array(query_emb)
                s = np.array(skill_emb)
                cosine = float(np.dot(q, s) / (np.linalg.norm(q) * np.linalg.norm(s) + 1e-9))
                scores[skill] = max(0.0, cosine)

            return scores
        except Exception as e:
            logger.debug("Embedding routing failed: %s", e)
            return {s: 0.0 for s in skills}

    def _performance_score(self, skills: list[str]) -> dict[str, float]:
        scores: dict[str, float] = {}
        for skill in skills:
            pack = self._registry.get_or_none("skill_packs", skill)
            if pack and hasattr(pack, "meta"):
                scores[skill] = pack.meta.performance_score
            else:
                scores[skill] = 0.5  # neutral default
        return scores

    def record_feedback(self, skill_name: str, success: bool, score: float) -> None:
        """Update skill pack performance from routing feedback."""
        pack = self._registry.get_or_none("skill_packs", skill_name)
        if pack and hasattr(pack, "record_usage"):
            pack.record_usage(success=success, score=score)

    def get_top_skills(self, task: str, top_k: int = 3) -> tuple[list[str], list[float]]:
        """
        Return top-k skills and their scores for a given task.
        Used by AutoOptimizer to check skill coverage before recommending training.
        """
        available_skills = self._registry.list("skill_packs") if self._registry else []
        if not available_skills:
            return [], []
        keyword_scores = self._keyword_score(task, available_skills)
        emb_scores = self._embedding_score(task, available_skills)
        combined = {
            s: 0.6 * keyword_scores.get(s, 0.0) + 0.4 * emb_scores.get(s, 0.0)
            for s in available_skills
        }
        sorted_skills = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        top = sorted_skills[:top_k]
        return [name for name, _ in top], [score for _, score in top]

    def routing_stats(self) -> dict[str, Any]:
        skill_counts: dict[str, int] = {}
        for entry in self._routing_history:
            for skill in entry["decision"].selected_packs:
                skill_counts[skill] = skill_counts.get(skill, 0) + 1
        return {
            "total_routes": len(self._routing_history),
            "skill_distribution": skill_counts,
        }
