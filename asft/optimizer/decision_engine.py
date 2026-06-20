import logging
import random
import time
import uuid
from typing import Any

from sqlalchemy.orm import Session

from asft.db.database import SessionLocal
from asft.db.models import RoutingHistory

logger = logging.getLogger(__name__)

# Allowed strategies
STRATEGIES = [
    "memory_rag",
    "skill_pack",
    "lora",
    "qlora",
    "sparse_training",
    "distillation",
    "do_nothing",
]


class CostEstimator:
    """
    Estimates the computational and time cost of a given strategy for a workload.
    Phase 5: Dynamic Cost Estimator
    """

    def estimate(
        self, strategy: str, dataset_size: int, model_size_params: int, task_hash: str
    ) -> float:
        with SessionLocal() as db:
            results = (
                db.query(RoutingHistory)
                .filter(
                    RoutingHistory.task_hash == task_hash,
                    RoutingHistory.strategy_selected == strategy,
                    RoutingHistory.actual_cost.isnot(None),
                )
                .all()
            )

            if results and len(results) >= 5:
                # Use empirical estimates
                total_cost = sum(r.actual_cost for r in results)
                return total_cost / len(results)

        # Fallback: Rule-based estimates
        base_costs = {
            "do_nothing": 0.0,
            "skill_pack": 0.1,
            "memory_rag": 0.5,
            "qlora": 5.0,
            "lora": 10.0,
            "sparse_training": 15.0,
            "distillation": 50.0,
        }

        multiplier = 1.0
        if dataset_size > 10_000:
            multiplier *= 2.0
        if model_size_params > 7_000_000_000:
            multiplier *= 2.0

        return base_costs.get(strategy, 10.0) * multiplier


class MultiArmedBanditRouter:
    """
    Implements Epsilon-Greedy selection strategy based on historical outcomes.
    """

    def __init__(
        self, epsilon_initial: float = 0.50, epsilon_min: float = 0.05, alpha: float = 0.995
    ):
        self.epsilon_initial = epsilon_initial
        self.epsilon_min = epsilon_min
        self.alpha = alpha

    def _get_historical_utilities(
        self, db: Session, task_hash: str
    ) -> tuple[dict[str, float], int]:
        """
        Calculates expected utility for each strategy based on (Accuracy / Cost)
        Returns (utilities_dict, total_samples)
        """
        from asft.db.models import StrategyOutcome

        utilities = {s: 1.0 for s in STRATEGIES}

        results = (
            db.query(RoutingHistory)
            .filter(RoutingHistory.task_hash == task_hash, RoutingHistory.success)
            .all()
        )
        outcomes = db.query(StrategyOutcome).filter(StrategyOutcome.task_hash == task_hash).all()

        total_samples = len(results) + sum(o.sample_count for o in outcomes)

        stats = {s: {"acc": [], "cost": []} for s in STRATEGIES}
        for r in results:
            if r.actual_accuracy and r.actual_cost and r.actual_cost > 0:
                stats[r.strategy_selected]["acc"].append(r.actual_accuracy)
                stats[r.strategy_selected]["cost"].append(r.actual_cost)

        for o in outcomes:
            if o.avg_accuracy and o.avg_cost and o.avg_cost > 0:
                # Weight the aggregated outcomes by their sample count
                for _ in range(o.sample_count):
                    stats[o.strategy]["acc"].append(o.avg_accuracy)
                    stats[o.strategy]["cost"].append(o.avg_cost)

        for s in STRATEGIES:
            if stats[s]["acc"]:
                avg_acc = sum(stats[s]["acc"]) / len(stats[s]["acc"])
                avg_cost = sum(stats[s]["cost"]) / len(stats[s]["cost"])
                utilities[s] = avg_acc / max(0.1, avg_cost)

        return utilities, total_samples

    def select_strategy(
        self, task_hash: str, available_strategies: list[str] = None
    ) -> tuple[str, bool]:
        """
        Returns (selected_strategy, is_exploration)
        """
        strategies = available_strategies or STRATEGIES

        with SessionLocal() as db:
            utilities, total_samples = self._get_historical_utilities(db, task_hash)

        # Phase 9: Cold Start Protection
        if total_samples < 50:
            # Rule Engine Dominates
            # Force safe initial routing: try tools/memory before training
            safe_strategies = ["memory_rag", "skill_pack"]
            available_safe = [s for s in safe_strategies if s in strategies]
            if available_safe:
                return random.choice(available_safe), False
            else:
                return "qlora", False

        # Phase 6: Decaying Epsilon
        current_epsilon = max(self.epsilon_min, self.epsilon_initial * (self.alpha**total_samples))

        if total_samples >= 50 and total_samples <= 500:
            # Hybrid Rule + Bandit: boost exploration
            current_epsilon = max(current_epsilon, 0.20)

        if random.random() < current_epsilon:
            # Explore
            return random.choice(strategies), True

        # Exploit
        # Filter for available
        valid_utilities = {k: v for k, v in utilities.items() if k in strategies}
        if not valid_utilities:
            return "do_nothing", False

        best_strategy = max(valid_utilities.items(), key=lambda x: x[1])[0]
        return best_strategy, False


class OptimizerDecisionEngine:
    """
    Central brain of ASFT V3.
    Determines the cheapest, fastest, and most effective way to improve the AI system.
    """

    def __init__(self):
        self.estimator = CostEstimator()
        self.bandit = MultiArmedBanditRouter(epsilon=0.10)  # 10% explore, 90% exploit

    def evaluate_workload(
        self, task_type: str, dataset_size: int, model_size: int
    ) -> dict[str, Any]:
        """
        Main entry point for determining how to process a new learning objective.
        """
        task_hash = f"{task_type}_{dataset_size // 1000}k"

        logger.info(f"Optimizer evaluating workload: {task_hash}")

        from asft.memory.backends.secure_qdrant import SecureQdrantAdapter

        adapter = SecureQdrantAdapter()
        available_strategies = list(STRATEGIES)
        if not adapter.is_healthy() and "memory_rag" in available_strategies:
            logger.warning(
                "SecureQdrantAdapter health check failed. Removing memory_rag from optimizer strategies."
            )
            available_strategies.remove("memory_rag")

        strategy, is_explore = self.bandit.select_strategy(task_hash, available_strategies)
        expected_cost = self.estimator.estimate(strategy, dataset_size, model_size, task_hash)

        # We record the routing decision for future RLHF/Outcome tracking
        record_id = self._record_decision(task_hash, strategy, is_explore, expected_cost)

        return {
            "routing_id": record_id,
            "selected_strategy": strategy,
            "exploration_mode": is_explore,
            "expected_cost_units": expected_cost,
            "v4_rlhf_ready": True,
        }

    def _record_decision(
        self, task_hash: str, strategy: str, is_explore: bool, expected_cost: float
    ) -> str:
        record_id = str(uuid.uuid4())
        with SessionLocal() as db:
            record = RoutingHistory(
                id=record_id,
                task_hash=task_hash,
                strategy_selected=strategy,
                exploration_mode=is_explore,
                expected_cost=expected_cost,
                timestamp=time.time(),
            )
            db.add(record)
            db.commit()

        return record_id

    # ---------------------------------------------------------
    # V4 ROADMAP PLACEHOLDERS (DO NOT IMPLEMENT)
    # ---------------------------------------------------------
    def _v4_contextual_bandit_hook(self):
        """V4 Extension Point: Contextual Bandit Implementation"""
        raise NotImplementedError("Scheduled for V4")

    def _v4_dpo_reward_hook(self):
        """V4 Extension Point: Direct Preference Optimization"""
        raise NotImplementedError("Scheduled for V4")

    def _v4_rlhf_policy_hook(self):
        """V4 Extension Point: RLHF Policy Update"""
        raise NotImplementedError("Scheduled for V4")
