"""
Skill Pack — Base class for all ASFT modular skill packs.
Each skill pack is independently trainable, loadable, removable, and mergeable.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SkillPackMeta:
    name: str
    version: str = "0.1.0"
    description: str = ""
    domain: str = "general"
    tags: List[str] = field(default_factory=list)
    author: str = "asft"
    created_at: float = field(default_factory=time.time)
    performance_score: float = 0.0  # 0–1 overall quality
    usage_count: int = 0
    last_used: Optional[float] = None


@dataclass
class SkillResult:
    skill_name: str
    output: Any
    confidence: float = 1.0
    duration_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error: Optional[str] = None


class SkillPack(ABC):
    """
    Base class for ASFT skill packs.

    A skill pack encapsulates:
      - A trained task-specific adapter (LoRA delta or sparse delta)
      - A set of prompt templates
      - Task-specific preprocessing/postprocessing
      - Evaluation logic

    Skill packs do NOT hold the base model — they hold only deltas.
    The base model is loaded once and skill packs are hot-swapped.
    """

    def __init__(self, name: str, pack_dir: Optional[str] = None):
        self.meta = SkillPackMeta(name=name)
        self._pack_dir = Path(pack_dir) if pack_dir else Path("./asft_data/skill_packs") / name
        self._loaded = False
        self._adapter = None  # LoRA weights or sparse delta

    # ------------------------------------------------------------------
    # Abstract interface — subclasses must implement
    # ------------------------------------------------------------------

    @abstractmethod
    def process(self, task_input: str, model=None, tokenizer=None, **kwargs) -> SkillResult:
        """Execute the skill on a task input."""
        ...

    @abstractmethod
    def get_prompt_template(self) -> str:
        """Return the skill's system prompt template."""
        ...

    @abstractmethod
    def evaluate(self, sample_input: str, sample_output: str) -> float:
        """Evaluate output quality. Returns 0–1 score."""
        ...

    # ------------------------------------------------------------------
    # Lifecycle methods
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load skill pack delta from disk."""
        meta_path = self._pack_dir / "meta.json"
        adapter_path = self._pack_dir / "adapter"

        if meta_path.exists():
            with open(meta_path) as f:
                data = json.load(f)
            self.meta.performance_score = data.get("performance_score", 0.0)
            self.meta.usage_count = data.get("usage_count", 0)
            logger.info("SkillPack loaded: %s (score=%.2f)", self.meta.name, self.meta.performance_score)
        self._loaded = True

    def save(self) -> None:
        """Save skill pack meta and adapter to disk."""
        self._pack_dir.mkdir(parents=True, exist_ok=True)
        meta_path = self._pack_dir / "meta.json"
        with open(meta_path, "w") as f:
            json.dump({
                "name": self.meta.name,
                "version": self.meta.version,
                "description": self.meta.description,
                "domain": self.meta.domain,
                "tags": self.meta.tags,
                "performance_score": self.meta.performance_score,
                "usage_count": self.meta.usage_count,
            }, f, indent=2)
        logger.info("SkillPack saved: %s", self._pack_dir)

    def unload(self) -> None:
        """Release loaded resources."""
        self._adapter = None
        self._loaded = False
        logger.debug("SkillPack unloaded: %s", self.meta.name)

    def record_usage(self, success: bool = True, score: Optional[float] = None) -> None:
        """Update usage statistics."""
        self.meta.usage_count += 1
        self.meta.last_used = time.time()
        if score is not None:
            # Exponential moving average of performance score
            alpha = 0.1
            self.meta.performance_score = (
                alpha * score + (1 - alpha) * self.meta.performance_score
            )

    # ------------------------------------------------------------------
    # Merging
    # ------------------------------------------------------------------

    def merge_with(self, other: "SkillPack", weight: float = 0.5) -> "MergedSkillPack":
        """Merge this skill pack with another using weighted averaging."""
        return MergedSkillPack([self, other], weights=[weight, 1.0 - weight])

    def __repr__(self) -> str:
        return (
            f"<SkillPack name={self.meta.name!r} "
            f"score={self.meta.performance_score:.2f} "
            f"loaded={self._loaded}>"
        )


class MergedSkillPack(SkillPack):
    """A composite skill pack that merges results from multiple packs."""

    def __init__(self, packs: List[SkillPack], weights: Optional[List[float]] = None):
        name = "_".join(p.meta.name for p in packs)
        super().__init__(name=f"merged_{name}")
        self._packs = packs
        self._weights = weights or [1.0 / len(packs)] * len(packs)
        assert abs(sum(self._weights) - 1.0) < 0.01, "Weights must sum to 1"

    def process(self, task_input: str, model=None, tokenizer=None, **kwargs) -> SkillResult:
        results = [
            p.process(task_input, model=model, tokenizer=tokenizer, **kwargs)
            for p in self._packs
        ]
        # Weighted confidence aggregation
        total_conf = sum(r.confidence * w for r, w in zip(results, self._weights))
        best_result = max(results, key=lambda r: r.confidence)
        best_result.confidence = total_conf
        best_result.skill_name = self.meta.name
        return best_result

    def get_prompt_template(self) -> str:
        return self._packs[0].get_prompt_template() if self._packs else ""

    def evaluate(self, sample_input: str, sample_output: str) -> float:
        scores = [p.evaluate(sample_input, sample_output) for p in self._packs]
        return sum(s * w for s, w in zip(scores, self._weights))
