"""Tests for asft.analysis — ASFT LLM Decision Pipeline."""
from __future__ import annotations

import pytest

from asft.analysis.analyzer import Analyzer
from asft.analysis.evaluator import MockPromptEvaluator, PromptEvaluationResult
from asft.analysis.finetune_estimator import FinetuneEstimateResult, FinetuneEstimator
from asft.analysis.rag_analyzer import MockRAGAnalyzer, RAGEvaluationResult
from asft.analysis.recommender import DecisionRecommender


# ── Helpers ───────────────────────────────────────────────────────────────────

def _prompt(score: float) -> PromptEvaluationResult:
    return PromptEvaluationResult(method="prompt", score=score)

def _rag(score: float, available: bool = True) -> RAGEvaluationResult:
    return RAGEvaluationResult(method="rag", score=score, retrieval_available=available)

def _ft(score: float, cost: float = 500.0, hours: float = 9.0) -> FinetuneEstimateResult:
    return FinetuneEstimateResult(
        method="fine_tuning", expected_score=score,
        estimated_cost=cost, estimated_hours=hours
    )


# ── DecisionRecommender ───────────────────────────────────────────────────────

class TestDecisionRecommender:
    def setup_method(self):
        self.rec = DecisionRecommender(rag_threshold=10.0, ft_threshold=5.0)

    def test_prompting_recommended_when_gains_are_small(self):
        result = self.rec.recommend(_prompt(85.0), _rag(87.0), _ft(88.0))
        assert result.method == "PROMPTING"
        assert result.savings_usd > 0  # savings shown vs fine-tuning

    def test_rag_recommended_when_retrieval_is_strong(self):
        result = self.rec.recommend(_prompt(60.0), _rag(85.0), _ft(86.0))
        assert result.method == "RAG"
        assert result.savings_usd > 0
        assert result.savings_hours > 0

    def test_fine_tuning_recommended_when_gain_is_large(self):
        result = self.rec.recommend(_prompt(60.0), _rag(62.0), _ft(85.0))
        assert result.method == "FINE-TUNING"
        assert result.savings_usd == 0.0  # no savings — FT is recommended

    def test_reason_is_dynamically_generated(self):
        """Reason must be built from the actual numbers — not a hardcoded string."""
        result_a = self.rec.recommend(_prompt(60.0), _rag(85.0), _ft(86.0))
        result_b = self.rec.recommend(_prompt(70.0), _rag(95.0), _ft(96.0))
        # Both are RAG recommendations but with different numbers — reasons must differ
        assert result_a.reason != result_b.reason

    def test_confidence_increases_with_improvement(self):
        low_gain  = self.rec.recommend(_prompt(72.0), _rag(83.0), _ft(84.0))
        high_gain = self.rec.recommend(_prompt(60.0), _rag(85.0), _ft(86.0))
        # Both are RAG, but high_gain has a larger rag_improvement
        assert high_gain.confidence > low_gain.confidence


# ── FinetuneEstimator ─────────────────────────────────────────────────────────

class TestFinetuneEstimator:
    def test_support_task_with_rag_gets_small_ft_gain(self):
        estimator = FinetuneEstimator()
        task = {"task_name": "customer support chatbot", "model": "meta-llama/Llama-3-8B"}
        rag = _rag(89.0, available=True)
        result = estimator.estimate(task, rag)
        assert result.expected_score == 90.0   # +1 pp
        assert result.estimated_hours == 9.0

    def test_70b_model_costs_more(self):
        estimator = FinetuneEstimator()
        task = {"task_name": "code generation", "model": "meta-llama/Llama-3-70B"}
        rag = _rag(70.0, available=False)
        result = estimator.estimate(task, rag)
        assert result.estimated_hours == 40.0
        assert result.estimated_cost > 1000.0

    def test_non_support_task_gets_larger_gain(self):
        estimator = FinetuneEstimator()
        task = {"task_name": "code generation", "model": "meta-llama/Llama-3-8B"}
        rag = _rag(70.0, available=False)
        result = estimator.estimate(task, rag)
        assert result.expected_score > 80.0   # +15 pp


# ── Mock Evaluators ───────────────────────────────────────────────────────────

class TestMockEvaluators:
    def test_mock_prompt_support_task(self):
        ev = MockPromptEvaluator()
        result = ev.evaluate_baseline({"task_name": "customer support chatbot"})
        assert isinstance(result, PromptEvaluationResult)
        assert result.score == 72.0

    def test_mock_prompt_generic_task(self):
        ev = MockPromptEvaluator()
        result = ev.evaluate_baseline({"task_name": "code generation"})
        assert result.score == 65.0

    def test_mock_rag_with_documents(self):
        ra = MockRAGAnalyzer()
        baseline = _prompt(72.0)
        result = ra.evaluate_rag({"task_name": "support", "documents": "./docs"}, baseline)
        assert isinstance(result, RAGEvaluationResult)
        assert result.retrieval_available is True
        assert result.score == 89.0

    def test_mock_rag_without_documents(self):
        ra = MockRAGAnalyzer()
        baseline = _prompt(65.0)
        result = ra.evaluate_rag({"task_name": "code generation"}, baseline)
        assert result.retrieval_available is False
        assert result.score == 65.0  # unchanged


# ── Analyzer (public API) ─────────────────────────────────────────────────────

class TestAnalyzer:
    def test_from_config_returns_report(self):
        analyzer = Analyzer.from_config({
            "task_name": "customer support chatbot",
            "model": "meta-llama/Llama-3",
            "documents": "./docs",
        })
        report = analyzer.run()
        assert report.task_name == "Customer Support Chatbot"
        assert report.prompt.score > 0
        assert report.rag.score >= report.prompt.score
        assert report.recommendation.method in {"PROMPTING", "RAG", "FINE-TUNING"}

    def test_from_config_support_recommends_rag(self):
        analyzer = Analyzer.from_config({
            "task_name": "customer support chatbot",
            "model": "meta-llama/Llama-3",
            "documents": "./docs",
        })
        report = analyzer.run()
        assert report.recommendation.method == "RAG"

    def test_analyzer_report_has_nonzero_savings_for_rag(self):
        analyzer = Analyzer.from_config({
            "task_name": "customer support chatbot",
            "model": "meta-llama/Llama-3",
            "documents": "./docs",
        })
        report = analyzer.run()
        assert report.recommendation.savings_usd > 0
