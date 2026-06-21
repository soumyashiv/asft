"""
Tests for the Training Acceleration subsystems:
    - CostEstimator: scaling law calculations
    - AutoOptimizer: decision hierarchy
    - AdaptiveSampleSelector: sample scoring
    - ComputeBudgetRouter: task complexity classification
    - EWCRegularizer: Fisher computation and penalty
"""

from __future__ import annotations

import pytest

# ============================================================================
# CostEstimator Tests
# ============================================================================


class TestCostEstimator:

    @pytest.fixture
    def estimator(self):
        from asft.optimizer.cost_estimator import CostEstimator

        return CostEstimator()

    def test_estimate_returns_positive_values(self, estimator):
        est = estimator.estimate("Qwen/Qwen2-7B", dataset_size=1000, method="qlora")
        assert est.gpu_hours > 0
        assert est.cost_usd >= 0
        assert est.trainable_fraction > 0

    def test_qlora_cheaper_than_full_ft(self, estimator):
        qlora = estimator.estimate("Qwen/Qwen2-7B", dataset_size=1000, method="qlora")
        full = estimator.estimate("Qwen/Qwen2-7B", dataset_size=1000, method="full")
        # QLoRA should always be cheaper due to fewer trainable params
        assert qlora.cost_usd < full.cost_usd

    def test_trainable_fraction_qlora(self, estimator):
        est = estimator.estimate("Qwen/Qwen2-7B", dataset_size=1000, method="qlora")
        # LoRA should train ~0.5% of params
        assert est.trainable_fraction < 0.02
        assert est.trainable_fraction > 0.0001

    def test_larger_dataset_costs_more(self, estimator):
        small = estimator.estimate("Qwen/Qwen2-7B", dataset_size=100, method="qlora")
        large = estimator.estimate("Qwen/Qwen2-7B", dataset_size=10_000, method="qlora")
        assert large.cost_usd >= small.cost_usd

    def test_unknown_model_defaults_to_7b(self, estimator):
        est = estimator.estimate("unknown/my-custom-model", dataset_size=500, method="qlora")
        assert est.n_params_billions == pytest.approx(7.0)

    def test_small_dataset_warns(self, estimator):
        est = estimator.estimate("Qwen/Qwen2-7B", dataset_size=10, method="qlora")
        assert len(est.warnings) > 0
        assert est.recommendation in ("retrieve", "proceed_cheap", "proceed")

    def test_compare_methods_returns_all(self, estimator):
        comparison = estimator.compare_methods("Qwen/Qwen2-7B", dataset_size=500)
        assert "qlora" in comparison
        assert "peft_lora" in comparison
        assert "full" in comparison

    def test_roi_score_positive(self, estimator):
        est = estimator.estimate("Qwen/Qwen2-7B", dataset_size=1000, method="qlora")
        assert est.roi_score > 0

    @pytest.mark.parametrize(
        "model,expected_billions",
        [
            ("Qwen/Qwen2-7B", 7.0),
            ("Qwen/Qwen2-0.5B", 0.5),
            ("meta-llama/Meta-Llama-3-8B", 8.0),
            ("mistralai/Mistral-7B-v0.1", 7.0),
        ],
    )
    def test_model_param_lookup(self, estimator, model, expected_billions):
        est = estimator.estimate(model, dataset_size=100)
        assert est.n_params_billions == pytest.approx(expected_billions, abs=0.1)


# ============================================================================
# AutoOptimizer Tests
# ============================================================================


class TestAutoOptimizer:

    @pytest.fixture
    def optimizer_no_registry(self):
        from asft.optimizer.auto_optimizer import AutoOptimizer

        return AutoOptimizer(registry=None)

    def test_recommends_rag_for_low_accuracy_target(self, optimizer_no_registry):
        decision = optimizer_no_registry.decide(
            task="What is the capital of France?",
            target_accuracy=0.70,
        )
        # RAG accuracy is 0.72, should be sufficient for 0.70 target
        assert decision.action in (
            "use_rag",
            "use_skill",
            "use_episodic_memory",
            "use_semantic_memory",
            "use_working_memory",
        )

    def test_recommends_training_for_high_accuracy(self, optimizer_no_registry):
        decision = optimizer_no_registry.decide(
            task="Fine-tune on domain data",
            target_accuracy=0.88,
            allow_training=True,
        )
        assert decision.action in ("use_lora", "use_qlora", "distill", "full_finetune")

    def test_rejects_when_training_disabled_and_target_too_high(self, optimizer_no_registry):
        from asft.optimizer.auto_optimizer import ACTION_REJECT

        decision = optimizer_no_registry.decide(
            task="Complex specialized task",
            target_accuracy=0.98,
            allow_training=False,
        )
        assert decision.action == ACTION_REJECT

    def test_respects_budget_constraint(self, optimizer_no_registry):
        decision = optimizer_no_registry.decide(
            task="Train on huge dataset",
            target_accuracy=0.85,
            budget_usd=0.001,  # Essentially zero budget
            allow_training=True,
        )
        # With near-zero budget, should either reject or pick cheapest option
        assert decision.estimated_cost_usd <= 10.0 or decision.action in (
            "use_rag",
            "use_skill",
            "use_working_memory",
            "reject",
        )

    def test_has_alternatives(self, optimizer_no_registry):
        decision = optimizer_no_registry.decide(
            task="Some ML task",
            target_accuracy=0.80,
        )
        # Should always provide alternatives for context
        assert isinstance(decision.alternatives, list)

    def test_decision_has_reasoning(self, optimizer_no_registry):
        decision = optimizer_no_registry.decide(
            task="Write a Python function",
            domain="coding",
            target_accuracy=0.75,
        )
        assert len(decision.reasoning) > 20


# ============================================================================
# AdaptiveSampleSelector Tests
# ============================================================================


class TestAdaptiveSampleSelector:

    @pytest.fixture
    def selector_random(self):
        from asft.selection.sample_selector import AdaptiveSampleSelector

        return AdaptiveSampleSelector(
            model=None, tokenizer=None, keep_fraction=0.5, method="random"
        )

    def test_random_selection_keeps_fraction(self, selector_random):
        samples = [f"Sample {i}" for i in range(100)]
        selected, report = selector_random.select(samples)
        assert len(selected) == 50
        assert report.original_count == 100
        assert report.selected_count == 50

    def test_empty_dataset_returns_empty(self, selector_random):
        selected, report = selector_random.select([])
        assert len(selected) == 0
        assert report.original_count == 0

    def test_tiny_dataset_returns_all(self, selector_random):
        """Datasets < 50 samples should not be pruned."""
        samples = [f"s{i}" for i in range(10)]
        selected, report = selector_random.select(samples)
        assert len(selected) == 10  # All returned

    def test_report_reduction_percent(self, selector_random):
        samples = [f"s{i}" for i in range(100)]
        _, report = selector_random.select(samples)
        assert abs(report.reduction_percent - 50.0) < 5.0  # ~50% reduction

    def test_dict_samples_supported(self, selector_random):
        samples = [{"text": f"Sample {i}", "label": i} for i in range(100)]
        selected, report = selector_random.select(samples, text_field="text")
        assert len(selected) == 50


# ============================================================================
# ComputeBudgetRouter Tests
# ============================================================================


class TestComputeBudgetRouter:

    @pytest.fixture
    def router(self):
        from asft.compute.adaptive_compute import ComputeBudgetRouter

        return ComputeBudgetRouter()

    def test_simple_factual_gets_low_tier(self, router):
        from asft.compute.adaptive_compute import ComputeTier

        decision = router.route("What is the capital of France?")
        assert decision.tier in (ComputeTier.MINIMAL, ComputeTier.LOW, ComputeTier.MEDIUM)

    def test_complex_coding_gets_high_tier(self, router):
        from asft.compute.adaptive_compute import ComputeTier

        decision = router.route(
            "Design and implement a distributed caching system with consistent hashing",
            domain="coding",
        )
        assert decision.tier in (ComputeTier.HIGH, ComputeTier.MAXIMUM)

    def test_budget_cap_respected(self, router):
        decision = router.route("complex question", budget_tokens=100)
        assert decision.max_new_tokens <= 100

    def test_compute_savings_estimation(self, router):
        tasks = [
            "What is 2+2?",  # Easy
            "Implement a full transformer architecture from scratch in PyTorch",  # Hard
            "What is the capital of Japan?",  # Easy
        ]
        savings = router.estimate_compute_savings(tasks)
        assert savings["estimated_compute_savings_fraction"] > 0
        assert savings["n_tasks"] == 3

    def test_batch_route_sorted_easy_first(self, router):
        from asft.compute.adaptive_compute import ComputeTier

        tasks = [
            "Design a distributed ML training system",
            "What is 1+1?",
        ]
        decisions = router.batch_route(tasks)
        tier_order = list(ComputeTier)
        # Easy should be first
        assert tier_order.index(decisions[0].tier) <= tier_order.index(decisions[1].tier)


# ============================================================================
# EWCRegularizer Tests
# ============================================================================


class TestEWCRegularizer:

    def _make_tiny_model(self):
        """Create a tiny 2-layer linear model for testing."""
        import torch.nn as nn

        model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 3))
        return model

    def _make_dataloader(self, model):
        """Create a minimal dataloader."""
        import torch

        x = torch.randn(20, 10)
        labels = torch.zeros(20, dtype=torch.long)

        # Wrap as HF-style dict batches
        class DictDataset:
            def __init__(self):
                self.data = [
                    ({"input_ids": x[i : i + 1], "labels": labels[i : i + 1]}) for i in range(20)
                ]

            def __len__(self):
                return len(self.data)

            def __getitem__(self, i):
                return self.data[i]

        return DictDataset()

    def test_ewc_loss_zero_before_compute(self):
        from asft.continual.ewc_trainer import EWCConfig, EWCRegularizer

        model = self._make_tiny_model()
        ewc = EWCRegularizer(model, EWCConfig(ewc_lambda=1000.0))
        assert not ewc.has_fisher()
        loss = ewc.ewc_loss()
        assert float(loss) == pytest.approx(0.0)

    def test_ewc_loss_positive_after_weight_change(self):
        import torch

        from asft.continual.ewc_trainer import EWCConfig, EWCRegularizer

        model = self._make_tiny_model()
        ewc = EWCRegularizer(model, EWCConfig(ewc_lambda=1000.0))

        # Manually set up Fisher and anchors (bypass dataloader requirement for unit test)
        ewc._fisher = {
            name: torch.ones_like(param)
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        ewc._anchors = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

        # Before weight change: penalty should be ~0
        initial_penalty = float(ewc.ewc_loss())
        assert initial_penalty == pytest.approx(0.0, abs=1e-5)

        # After weight change: penalty should be > 0
        with torch.no_grad():
            for param in model.parameters():
                param.add_(torch.ones_like(param))  # Shift all params by 1

        changed_penalty = float(ewc.ewc_loss())
        assert changed_penalty > 0, "EWC penalty must increase when params change from anchor"

    def test_ewc_lambda_scales_penalty(self):
        import torch

        from asft.continual.ewc_trainer import EWCConfig, EWCRegularizer

        def make_ewc(lam):
            model = self._make_tiny_model()
            ewc = EWCRegularizer(model, EWCConfig(ewc_lambda=lam))
            ewc._fisher = {
                n: torch.ones_like(p) for n, p in model.named_parameters() if p.requires_grad
            }
            ewc._anchors = {
                n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad
            }
            return ewc, model

        ewc_low, _ = make_ewc(100.0)
        ewc_high, _ = make_ewc(10000.0)

        assert float(ewc_high.ewc_loss()) > float(ewc_low.ewc_loss())
