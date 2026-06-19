"""
ASFT AutoOptimizer — Training-Only-If-Necessary Decision Engine.

CORE QUESTION: "What is the cheapest way to achieve the required capability?"

DECISION HIERARCHY (cheapest to most expensive):
    1. Working memory      → O(1) lookup, zero cost
    2. Episodic memory     → O(log n) FTS5 search, near-zero cost
    3. Semantic memory     → O(log n) fact retrieval, near-zero cost
    4. Skill pack          → forward pass only, cents/query
    5. RAG / retrieval     → embedding + vector search, cents/query
    6. Knowledge distill   → train small student from large teacher
    7. LoRA fine-tune      → partial parameter update, dollars
    8. QLoRA fine-tune     → 4-bit quantized LoRA, fewer dollars
    9. Dynamic sparse      → selective layer training, moderate cost
   10. Full fine-tune      → all parameters, most expensive, rarely needed

The optimizer evaluates each option in order and returns the cheapest one
whose expected accuracy meets the target.

RESEARCH BASIS:
    - The principle of "cheapest sufficient capability" is standard in
      resource-constrained ML (Strubell et al., 2019; Schwartz et al., 2020).
    - Retrieval-Augmented Generation as alternative to fine-tuning:
      Lewis et al. 2020, Shi et al. 2023 — RAG often matches FT accuracy
      on knowledge-intensive tasks at a fraction of the cost.
    - LoRA efficiency: Hu et al. 2021 — near full-FT accuracy with 0.1–1%
      of trainable parameters.
    - QLoRA: Dettmers et al. 2023 — 4-bit quantization with <2% accuracy drop.

LIMITATIONS (honest):
    - Memory/skill coverage checks are heuristic (embedding similarity).
      A high similarity score does not guarantee a correct answer.
    - Accuracy estimates at each tier are empirical medians, not guarantees.
    - For novel domains with no existing data, this optimizer may under-estimate
      the value of fine-tuning.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# Action tokens — canonical names for each decision
ACTION_USE_WORKING_MEMORY = "use_working_memory"
ACTION_USE_EPISODIC_MEMORY = "use_episodic_memory"
ACTION_USE_SEMANTIC_MEMORY = "use_semantic_memory"
ACTION_USE_SKILL = "use_skill"
ACTION_USE_RAG = "use_rag"
ACTION_DISTILL = "distill"
ACTION_USE_LORA = "use_lora"
ACTION_USE_QLORA = "use_qlora"
ACTION_SPARSE_TUNE = "sparse_tune"
ACTION_FULL_FINETUNE = "full_finetune"
ACTION_REJECT = "reject"  # budget exceeded or dataset too small

# Expected accuracy per tier (median empirical, task-type-agnostic)
# Source: ASFT benchmark suite + literature review
_TIER_ACCURACY: Dict[str, float] = {
    ACTION_USE_WORKING_MEMORY: 0.95,   # If it's in memory, recall is near-perfect
    ACTION_USE_EPISODIC_MEMORY: 0.80,
    ACTION_USE_SEMANTIC_MEMORY: 0.78,
    ACTION_USE_SKILL: 0.75,
    ACTION_USE_RAG: 0.72,              # Lewis et al. 2020: ~72% on NQ, TriviaQA
    ACTION_DISTILL: 0.80,              # DistilBERT: 97% of BERT on GLUE = ~0.80 abs
    ACTION_USE_LORA: 0.85,             # Hu et al. 2021: near-FT accuracy
    ACTION_USE_QLORA: 0.84,            # Dettmers et al. 2023: <2% drop vs LoRA
    ACTION_SPARSE_TUNE: 0.80,          # Selective layer training: variable
    ACTION_FULL_FINETUNE: 0.90,
}

# Cost in USD per 1000 queries for inference-time options
_TIER_COST_PER_K: Dict[str, float] = {
    ACTION_USE_WORKING_MEMORY: 0.00,
    ACTION_USE_EPISODIC_MEMORY: 0.001,
    ACTION_USE_SEMANTIC_MEMORY: 0.001,
    ACTION_USE_SKILL: 0.01,
    ACTION_USE_RAG: 0.05,
    ACTION_DISTILL: 200.0,    # One-time training cost (amortized)
    ACTION_USE_LORA: 50.0,    # One-time LoRA training
    ACTION_USE_QLORA: 25.0,   # Cheaper than LoRA due to 4-bit
    ACTION_SPARSE_TUNE: 75.0,
    ACTION_FULL_FINETUNE: 500.0,
}


@dataclass
class OptimizerDecision:
    """The AutoOptimizer's recommendation."""
    action: str
    reasoning: str
    estimated_cost_usd: float
    expected_accuracy: float
    alternatives: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class AutoOptimizer:
    """
    Determines the cheapest path to the required task accuracy.

    Evaluates the full capability hierarchy in order, from zero-cost
    memory lookup to expensive full fine-tuning, returning the first
    option whose expected accuracy meets the target.

    Args:
        registry:       ComponentRegistry for accessing memory, skills, etc.
        cost_estimator: CostEstimator for training cost projection.
    """

    def __init__(self, registry=None, cost_estimator=None):
        self._registry = registry
        if cost_estimator is None:
            from asft.optimizer.cost_estimator import CostEstimator
            cost_estimator = CostEstimator()
        self._estimator = cost_estimator

    def decide(
        self,
        task: str,
        domain: str = "general",
        target_accuracy: float = 0.80,
        budget_usd: Optional[float] = None,
        model_name: str = "Qwen/Qwen2-7B",
        dataset_size: int = 1_000,
        allow_training: bool = True,
    ) -> OptimizerDecision:
        """
        Evaluate all capability tiers and return the cheapest sufficient option.

        Args:
            task:            The task description (used for memory/skill lookup).
            domain:          Domain hint (coding | math | science | general …).
            target_accuracy: Minimum acceptable accuracy (0.0–1.0).
            budget_usd:      Maximum spend in USD. None = no limit.
            model_name:      Model to use if training is recommended.
            dataset_size:    Available training samples.
            allow_training:  If False, will never recommend training.

        Returns:
            OptimizerDecision with recommended action and reasoning.
        """
        logger.info(
            "AutoOptimizer.decide | domain=%s target_acc=%.2f budget=%s",
            domain, target_accuracy, f"${budget_usd:.2f}" if budget_usd else "unlimited"
        )

        warnings: List[str] = []
        alternatives: List[Dict[str, Any]] = []

        # ----------------------------------------------------------------
        # Tier 1–3: Memory tiers (zero cost)
        # ----------------------------------------------------------------
        if self._registry is not None:
            memory_coverage = self._estimate_memory_coverage(task)
            logger.debug("Memory coverage score: %.3f", memory_coverage)

            if memory_coverage > 0.85:
                return self._make_decision(
                    action=ACTION_USE_WORKING_MEMORY,
                    reasoning=(
                        f"Working/episodic memory has a {memory_coverage:.0%} coverage score "
                        f"for this task. Retrieving from memory is always the cheapest option. "
                        f"No training required."
                    ),
                    cost=0.0,
                    accuracy=_TIER_ACCURACY[ACTION_USE_WORKING_MEMORY],
                    target=target_accuracy,
                    budget=budget_usd,
                    alternatives=alternatives,
                    warnings=warnings,
                )

            if memory_coverage > 0.60:
                alternatives.append({
                    "action": ACTION_USE_EPISODIC_MEMORY,
                    "estimated_accuracy": _TIER_ACCURACY[ACTION_USE_EPISODIC_MEMORY],
                    "cost_usd": 0.001,
                    "note": "Partial memory coverage — augment with RAG for higher accuracy"
                })

        # ----------------------------------------------------------------
        # Tier 4: Skill pack
        # ----------------------------------------------------------------
        skill_match = self._estimate_skill_coverage(task, domain)
        if skill_match > 0.70:
            if _TIER_ACCURACY[ACTION_USE_SKILL] >= target_accuracy:
                return self._make_decision(
                    action=ACTION_USE_SKILL,
                    reasoning=(
                        f"A domain skill pack for '{domain}' matches this task with "
                        f"{skill_match:.0%} confidence. Skill packs require no training — "
                        f"only a forward pass. Expected accuracy: "
                        f"{_TIER_ACCURACY[ACTION_USE_SKILL]:.0%}."
                    ),
                    cost=_TIER_COST_PER_K[ACTION_USE_SKILL] / 1000,
                    accuracy=_TIER_ACCURACY[ACTION_USE_SKILL],
                    target=target_accuracy,
                    budget=budget_usd,
                    alternatives=alternatives,
                    warnings=warnings,
                )
            else:
                alternatives.append({
                    "action": ACTION_USE_SKILL,
                    "estimated_accuracy": _TIER_ACCURACY[ACTION_USE_SKILL],
                    "cost_usd": 0.0,
                    "note": f"Skill pack accuracy {_TIER_ACCURACY[ACTION_USE_SKILL]:.0%} < target {target_accuracy:.0%}"
                })

        # ----------------------------------------------------------------
        # Tier 5: RAG
        # ----------------------------------------------------------------
        rag_accuracy = _TIER_ACCURACY[ACTION_USE_RAG]
        if rag_accuracy >= target_accuracy:
            return self._make_decision(
                action=ACTION_USE_RAG,
                reasoning=(
                    "Retrieval-Augmented Generation (RAG) often matches fine-tuning accuracy "
                    "on knowledge-intensive tasks at ~1% of the training cost. "
                    f"Expected accuracy: {rag_accuracy:.0%} (Lewis et al. 2020). "
                    "Build a vector index from your dataset before committing to training."
                ),
                cost=_TIER_COST_PER_K[ACTION_USE_RAG] / 1000,
                accuracy=rag_accuracy,
                target=target_accuracy,
                budget=budget_usd,
                alternatives=alternatives,
                warnings=warnings,
            )
        else:
            alternatives.append({
                "action": ACTION_USE_RAG,
                "estimated_accuracy": rag_accuracy,
                "cost_usd": 0.05,
                "note": "RAG insufficient for target accuracy — consider augmenting with fine-tuning"
            })

        # ----------------------------------------------------------------
        # Training gate: refuse if not allowed
        # ----------------------------------------------------------------
        if not allow_training:
            warnings.append(
                "Training is disabled. No memory, skill, or RAG solution meets the "
                f"target accuracy of {target_accuracy:.0%}. Enable training to proceed."
            )
            return OptimizerDecision(
                action=ACTION_REJECT,
                reasoning="Training disabled. No non-training option meets target accuracy.",
                estimated_cost_usd=0.0,
                expected_accuracy=0.0,
                alternatives=alternatives,
                warnings=warnings,
            )

        # ----------------------------------------------------------------
        # Tier 6–10: Training tiers
        # ----------------------------------------------------------------
        training_tiers = [
            (ACTION_DISTILL, "qlora"),
            (ACTION_USE_QLORA, "qlora"),
            (ACTION_USE_LORA, "peft_lora"),
            (ACTION_SPARSE_TUNE, "sparse"),
            (ACTION_FULL_FINETUNE, "full"),
        ]

        for action, method in training_tiers:
            if action == ACTION_FULL_FINETUNE and method == "full":
                # Only recommend full FT if LoRA accuracy is demonstrably insufficient
                if _TIER_ACCURACY[ACTION_USE_QLORA] >= target_accuracy:
                    continue  # QLoRA already sufficient — no need for full FT

            estimate = self._estimator.estimate(
                model_name=model_name,
                dataset_size=dataset_size,
                method=method,
            )
            expected_acc = _TIER_ACCURACY[action]
            cost = estimate.cost_usd

            # Budget check
            if budget_usd is not None and cost > budget_usd:
                alternatives.append({
                    "action": action,
                    "estimated_accuracy": expected_acc,
                    "cost_usd": cost,
                    "note": f"Exceeds budget (${cost:.2f} > ${budget_usd:.2f})"
                })
                warnings.append(f"{action} exceeds budget: ${cost:.2f}")
                continue

            if expected_acc >= target_accuracy:
                reasoning = self._build_training_reasoning(
                    action, method, estimate, target_accuracy, dataset_size
                )
                return self._make_decision(
                    action=action,
                    reasoning=reasoning,
                    cost=cost,
                    accuracy=expected_acc,
                    target=target_accuracy,
                    budget=budget_usd,
                    alternatives=alternatives,
                    warnings=warnings + estimate.warnings,
                )
            else:
                alternatives.append({
                    "action": action,
                    "estimated_accuracy": expected_acc,
                    "cost_usd": cost,
                    "note": f"Accuracy {expected_acc:.0%} < target {target_accuracy:.0%}"
                })

        # Nothing meets the target
        return OptimizerDecision(
            action=ACTION_REJECT,
            reasoning=(
                f"No available option meets the target accuracy of {target_accuracy:.0%}. "
                "Consider: (1) lowering the accuracy target, (2) collecting more training data, "
                "or (3) increasing the budget."
            ),
            estimated_cost_usd=0.0,
            expected_accuracy=0.0,
            alternatives=alternatives,
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _estimate_memory_coverage(self, task: str) -> float:
        """
        Estimate how well the memory system can answer this task.
        Returns 0–1: 1.0 = definitely in memory, 0.0 = definitely not.
        """
        if self._registry is None:
            return 0.0
        try:
            memory = self._registry.get("memory", "manager")
            if memory and hasattr(memory, "search"):
                results = memory.search(task, top_k=3)
                if results:
                    # Use the top similarity score as coverage
                    top_score = results[0].get("similarity", 0.5) if isinstance(results[0], dict) else 0.5
                    return float(top_score)
        except Exception as e:
            logger.debug("Memory coverage check failed: %s", e)
        return 0.0

    def _estimate_skill_coverage(self, task: str, domain: str) -> float:
        """
        Estimate how well an existing skill pack covers this task.
        Returns 0–1.
        """
        if self._registry is None:
            return 0.0
        try:
            router = self._registry.get("skills", "router")
            if router and hasattr(router, "get_top_skills"):
                skills, scores = router.get_top_skills(task, top_k=1)
                if skills and scores:
                    return float(scores[0])
        except Exception as e:
            logger.debug("Skill coverage check failed: %s", e)
        # Domain keyword heuristic fallback
        domain_skills = {"coding", "math", "mathematics", "science", "planning", "research"}
        if domain.lower() in domain_skills:
            return 0.60
        return 0.20

    @staticmethod
    def _make_decision(
        action: str, reasoning: str, cost: float, accuracy: float,
        target: float, budget: Optional[float],
        alternatives: List, warnings: List,
    ) -> OptimizerDecision:
        """Create a decision, checking budget constraint."""
        if budget is not None and cost > budget and cost > 0.01:
            warnings.append(f"Recommended action ({action}) costs ${cost:.2f} which exceeds budget ${budget:.2f}")
        return OptimizerDecision(
            action=action,
            reasoning=reasoning,
            estimated_cost_usd=cost,
            expected_accuracy=accuracy,
            alternatives=alternatives,
            warnings=warnings,
        )

    @staticmethod
    def _build_training_reasoning(
        action: str, method: str, estimate: Any, target: float, dataset_size: int
    ) -> str:
        """Build a human-readable explanation for a training recommendation."""
        method_evidence = {
            "qlora": (
                "QLoRA (Dettmers et al. 2023): 4-bit quantized LoRA achieves within 2% "
                "of full fine-tuning accuracy while reducing memory by 65–75%."
            ),
            "peft_lora": (
                "LoRA (Hu et al. 2021): trains only 0.1–1% of parameters while reaching "
                "near-full fine-tuning accuracy."
            ),
            "sparse": (
                "Sparse fine-tuning: selectively updates only the most important layers "
                "identified by Taylor-expansion importance scoring."
            ),
            "full": (
                "Full fine-tuning: all parameters updated. Only recommended when PEFT "
                "methods cannot achieve the required accuracy."
            ),
        }
        evidence = method_evidence.get(method, "")
        return (
            f"{action.replace('_', ' ').title()} recommended.\n"
            f"Evidence: {evidence}\n"
            f"Estimate: ${estimate.cost_usd:.2f}, {estimate.gpu_hours:.2f} GPU-hours, "
            f"{estimate.wall_time_minutes:.0f} min wall-time.\n"
            f"Expected accuracy: {_TIER_ACCURACY[action]:.0%} (target: {target:.0%}).\n"
            f"Dataset: {dataset_size} samples."
        )
