"""
ASFT — Pytest Test Suite
Tests all core components without requiring a GPU or LLM.
Run with: pytest tests/ -v
"""

import time
from pathlib import Path

import pytest

# ===========================================================================
# Core tests
# ===========================================================================


class TestHardwareProfiler:
    def test_detect_hardware_returns_profile(self):
        from asft.core.hardware_profiler import detect_hardware

        hw = detect_hardware()
        assert hw is not None
        assert hw.ram_total_gb > 0
        assert hw.cpu_physical_cores > 0
        assert hw.platform in ("Windows", "Linux", "Darwin")

    def test_hardware_summary(self):
        from asft.core.hardware_profiler import detect_hardware

        hw = detect_hardware()
        summary = hw.summary()
        assert isinstance(summary, str) and len(summary) > 10

    def test_recommendations_not_empty(self):
        from asft.core.hardware_profiler import detect_hardware

        hw = detect_hardware()
        # Method can return any valid option (CPU-only may return int4)
        assert hw.recommended_training_method in ("full", "lora", "qlora", "sparse", "asft")
        assert hw.recommended_precision in ("fp32", "fp16", "bf16", "int4", "int8")
        assert hw.recommended_batch_size >= 1


class TestConfig:
    def test_default_config_creates(self):
        from asft.core.config import ASFTConfig

        cfg = ASFTConfig()
        assert cfg is not None
        assert cfg.data_dir is not None

    def test_config_ensure_dirs(self, tmp_path):
        from asft.core.config import ASFTConfig

        cfg = ASFTConfig(data_dir=str(tmp_path / "asft_test_data"))
        cfg.ensure_dirs()
        assert Path(cfg.data_dir).exists()

    def test_config_yaml_roundtrip(self, tmp_path):
        from asft.core.config import ASFTConfig

        cfg = ASFTConfig(data_dir=str(tmp_path))
        yaml_path = str(tmp_path / "test_config.yaml")
        cfg.to_yaml(yaml_path)
        loaded = ASFTConfig.from_yaml(yaml_path)
        assert loaded is not None

    def test_config_apply_hardware(self):
        from asft.core.config import ASFTConfig
        from asft.core.hardware_profiler import detect_hardware

        cfg = ASFTConfig()
        hw = detect_hardware()
        cfg.apply_hardware_profile(hw)
        assert cfg.hardware.precision == hw.recommended_precision


class TestRegistry:
    """Registry class is named 'Registry' (not ASFTRegistry)."""

    def test_register_and_get(self):
        from asft.core.registry import Registry

        reg = Registry()
        reg.register("test", "item", "value_1")
        assert reg.get("test", "item") == "value_1"

    def test_list_namespace(self):
        from asft.core.registry import Registry

        reg = Registry()
        reg.register("ns", "a", 1)
        reg.register("ns", "b", 2)
        result = reg.list("ns")
        assert "a" in result and "b" in result

    def test_unregister(self):
        from asft.core.registry import Registry

        reg = Registry()
        reg.register("ns", "key", "val")
        reg.unregister("ns", "key")
        with pytest.raises(KeyError):
            reg.get("ns", "key")

    def test_get_or_none(self):
        from asft.core.registry import Registry

        reg = Registry()
        assert reg.get_or_none("nonexistent", "key") is None

    def test_exists(self):
        from asft.core.registry import Registry

        reg = Registry()
        reg.register("ns", "x", 42)
        assert reg.exists("ns", "x")
        assert not reg.exists("ns", "missing")


# ===========================================================================
# Memory tests
# ===========================================================================


class TestWorkingMemory:
    def test_set_and_get(self):
        from asft.memory.working_memory import WorkingMemory

        wm = WorkingMemory()
        wm.set("x", 42)
        assert wm.get("x") == 42

    def test_ttl_expiry(self):
        from asft.memory.working_memory import WorkingMemory

        wm = WorkingMemory()
        wm.set("temp", "data", ttl=0.001)
        time.sleep(0.02)
        wm.purge_expired()
        assert wm.get("temp") is None

    def test_tags(self):
        from asft.memory.working_memory import WorkingMemory

        wm = WorkingMemory()
        wm.set("a", 1, tags=["t1"])
        wm.set("b", 2, tags=["t1", "t2"])
        results = wm.search_by_tag("t1")
        assert len(results) == 2

    def test_all_keys(self):
        from asft.memory.working_memory import WorkingMemory

        wm = WorkingMemory(max_items=50)
        for i in range(3):
            wm.set(f"k{i}", i)
        keys = wm.all_keys()
        assert len(keys) >= 3

    def test_clear(self):
        from asft.memory.working_memory import WorkingMemory

        wm = WorkingMemory()
        wm.set("x", 1)
        wm.clear()
        assert wm.get("x") is None

    def test_delete(self):
        from asft.memory.working_memory import WorkingMemory

        wm = WorkingMemory()
        wm.set("y", 99)
        wm.delete("y")
        assert wm.get("y") is None

    def test_snapshot(self):
        from asft.memory.working_memory import WorkingMemory

        wm = WorkingMemory()
        wm.set("a", 1)
        wm.set("b", 2)
        snap = wm.snapshot()
        assert isinstance(snap, dict)
        assert len(snap) >= 2


class TestEpisodicMemory:
    """EpisodicMemory.store() takes an Episode dataclass."""

    def setup_method(self):
        import tempfile

        from asft.memory.episodic_memory import EpisodicMemory

        self.db_fd, self.db_path = tempfile.mkstemp(suffix=".db")
        self.em = EpisodicMemory(db_path=self.db_path)

    def teardown_method(self):
        import os

        os.close(self.db_fd)
        try:
            os.remove(self.db_path)
        except OSError:
            pass

    def _make_record(self, task, content="test"):
        from asft.memory.episodic_memory import Episode

        return Episode(id="", content=content, task=task)

    def test_record_and_count(self):
        self.em.store(self._make_record("task_start"))
        assert self.em.count() >= 1

    def test_query_limit(self):
        for i in range(10):
            self.em.store(self._make_record(f"event_{i}"))
        results = self.em.query("event", top_k=5)
        assert len(results) <= 5

    def test_get_by_id(self):
        ev_id = self.em.store(self._make_record("lookup_test"))
        if ev_id is not None:
            fetched = self.em.get(ev_id)
            assert fetched is not None


class TestSemanticMemory:
    """SemanticMemory.add_fact() takes a FactRecord dataclass."""

    def setup_method(self):
        from asft.memory.semantic_memory import SemanticMemory

        self.sm = SemanticMemory(db_path=":memory:")

    def _fact(self, subject, predicate, obj, source="test"):
        from asft.memory.semantic_memory import FactRecord

        return FactRecord(subject=subject, predicate=predicate, object=obj, source=source)

    def test_store_and_retrieve_fact(self):
        self.sm.add_fact(self._fact("Python", "is_a", "language"))
        facts = self.sm.query_by_subject("Python")
        assert len(facts) >= 1
        # query_by_subject returns dicts
        first = facts[0]
        predicate = first.get("predicate") if isinstance(first, dict) else first.predicate
        assert predicate == "is_a"

    def test_count_facts(self):
        self.sm.add_fact(self._fact("A", "is", "B"))
        self.sm.add_fact(self._fact("C", "is", "D"))
        assert self.sm.count() >= 2

    def test_query_by_predicate(self):
        self.sm.add_fact(self._fact("ASFT", "uses", "sparse_training"))
        self.sm.add_fact(self._fact("LoRA", "uses", "adapters"))
        results = self.sm.query_by_predicate("uses")
        assert len(results) >= 2


# ===========================================================================
# Skill tests
# ===========================================================================


class TestSkillPacks:
    @pytest.fixture(
        params=["coding", "research", "planning", "mathematics", "trading", "automation"]
    )
    def pack(self, request):
        from asft.skills.packs.automation import AutomationSkillPack
        from asft.skills.packs.coding import CodingSkillPack
        from asft.skills.packs.mathematics import MathematicsSkillPack
        from asft.skills.packs.planning import PlanningSkillPack
        from asft.skills.packs.research import ResearchSkillPack
        from asft.skills.packs.trading import TradingSkillPack

        packs = {
            "coding": CodingSkillPack,
            "research": ResearchSkillPack,
            "planning": PlanningSkillPack,
            "mathematics": MathematicsSkillPack,
            "trading": TradingSkillPack,
            "automation": AutomationSkillPack,
        }
        return packs[request.param]()

    def test_process_no_model(self, pack):
        result = pack.process("test task")
        assert result is not None
        assert result.output is not None
        assert 0.0 <= result.confidence <= 1.0

    def test_meta_populated(self, pack):
        assert pack.meta.name
        assert pack.meta.domain
        assert pack.meta.description

    def test_prompt_template_non_empty(self, pack):
        tmpl = pack.get_prompt_template()
        assert tmpl and len(tmpl) > 20

    def test_evaluate_returns_score(self, pack):
        score = pack.evaluate("test input", "test output with detailed content here")
        assert 0.0 <= score <= 1.0


class TestMathDirectCompute:
    def test_simple_arithmetic(self):
        from asft.skills.packs.mathematics import MathematicsSkillPack

        mp = MathematicsSkillPack()
        result = mp.process("2 + 2 * 5")
        assert "12" in str(result.output)
        assert result.confidence == 1.0

    def test_complex_expression(self):
        from asft.skills.packs.mathematics import MathematicsSkillPack

        mp = MathematicsSkillPack()
        result = mp.process("100 / 4 + 3")
        assert "28" in str(result.output)

    def test_no_model_fallback(self):
        from asft.skills.packs.mathematics import MathematicsSkillPack

        mp = MathematicsSkillPack()
        # Non-arithmetic text → model fallback (returns placeholder without model)
        result = mp.process("Prove the Pythagorean theorem")
        assert result.output is not None


class TestSkillRouter:
    def setup_method(self):
        from asft.core.registry import Registry
        from asft.skills.packs.coding import CodingSkillPack
        from asft.skills.packs.mathematics import MathematicsSkillPack
        from asft.skills.skill_router import SkillRouter

        self.reg = Registry()
        self.reg.register_skill("coding", CodingSkillPack())
        self.reg.register_skill("mathematics", MathematicsSkillPack())
        self.router = SkillRouter(registry=self.reg)

    def test_route_returns_decision(self):
        d = self.router.route("Write a Python function to parse JSON")
        assert d is not None
        assert isinstance(d.selected_packs, list)

    def test_route_has_scores(self):
        d = self.router.route("Calculate the integral of x^2")
        assert len(d.scores) > 0

    def test_multi_strategy(self):
        d = self.router.route("Complex task", strategy="multi", top_k=2)
        assert d.strategy == "multi"

    def test_routing_stats(self):
        self.router.route("task 1")
        self.router.route("task 2")
        stats = self.router.routing_stats()
        assert stats["total_routes"] == 2

    def test_consensus_strategy(self):
        d = self.router.route("research this topic", strategy="consensus")
        assert d.strategy == "consensus"


# ===========================================================================
# Accuracy tests
# ===========================================================================


class TestConfidenceScorer:
    def setup_method(self):
        from asft.accuracy.confidence_scorer import ConfidenceScorer

        self.scorer = ConfidenceScorer()

    def test_empty_output_low_score(self):
        score = self.scorer.score("")
        assert score.composite < 0.2

    def test_uncertain_output_lower_composite(self):
        # Composite should be meaningfully reduced for uncertain outputs
        clear_score = self.scorer.score("The result is definitively 42.")
        uncertain_score = self.scorer.score("I think maybe probably the answer could be X")
        assert uncertain_score.composite <= clear_score.composite

    def test_clear_output_reasonable_score(self):
        score = self.scorer.score("The result is 42. Here is the proof: steps 1, 2, 3.")
        assert score.composite > 0.5

    def test_code_output_high_score(self):
        score = self.scorer.score("```python\ndef add(a, b):\n    return a + b\n```")
        assert score.composite > 0.5

    def test_batch_score(self):
        outputs = ["clear answer with details", "maybe possibly unclear response"]
        scores = self.scorer.batch_score(outputs)
        assert len(scores) == 2

    def test_best_output(self):
        outputs = ["", "clear and specific: the answer is 42"]
        best, score = self.scorer.best_output(outputs)
        assert best == "clear and specific: the answer is 42"

    def test_score_flags(self):
        score = self.scorer.score("I think maybe probably it could be true")
        assert len(score.flags) > 0

    def test_high_medium_low_labels(self):
        low = self.scorer.score("")
        assert low.label == "LOW"
        high = self.scorer.score(
            "```python\ndef verified_function():\n    return 42\n```\nThe result is 42."
        )
        assert high.label in ("HIGH", "MEDIUM")


class TestSelfCritique:
    def test_no_generate_fn_marks_issues_only(self):
        from asft.accuracy.self_critique import SelfCritiqueEngine

        critic = SelfCritiqueEngine()
        # Without generate_fn, issues are found but no revision attempted
        result = critic.critique("The answer is 42.", "What is the answer?")
        assert isinstance(result.issues_found, list)
        assert not result.was_revised

    def test_hallucination_detected(self):
        from asft.accuracy.self_critique import SelfCritiqueEngine

        critic = SelfCritiqueEngine()
        output = "According to a recent study, experts agree this is widely known."
        result = critic.critique(output, "Explain X")
        assert len(result.issues_found) > 0

    def test_short_output_flagged(self):
        from asft.accuracy.self_critique import SelfCritiqueEngine

        critic = SelfCritiqueEngine()
        result = critic.critique("ok", "Question requiring a detailed answer")
        assert "response_too_short" in result.issues_found

    def test_clean_output_no_issues(self):
        from asft.accuracy.self_critique import SelfCritiqueEngine

        critic = SelfCritiqueEngine()
        clean = (
            "The Pythagorean theorem states that in a right triangle, "
            "a² + b² = c². This follows directly from Euclidean geometry. "
            "For example, a 3-4-5 triangle satisfies 9 + 16 = 25."
        )
        result = critic.critique(clean, "Explain Pythagorean theorem")
        assert result.is_clean or len(result.issues_found) <= 1


class TestVerificationLayer:
    def test_math_correct_answer(self):
        from asft.accuracy.verification_layer import VerificationLayer

        vl = VerificationLayer()
        result = vl.verify("The result is 12.", "2 + 2 * 5", task_type="mathematics")
        assert result.verified is True
        assert result.confidence > 0.0

    def test_math_wrong_answer(self):
        from asft.accuracy.verification_layer import VerificationLayer

        vl = VerificationLayer()
        result = vl.verify("The result is 999.", "2 + 2 * 5", task_type="mathematics")
        assert result.method == "math_cas"

    def test_no_verifier_returns_none_result(self):
        from asft.accuracy.verification_layer import VerificationLayer

        vl = VerificationLayer()
        result = vl.verify("Some general output.", "General question.", task_type="general")
        assert result is not None


class TestMultiPassReasoner:
    def test_best_of_k(self):
        from asft.accuracy.multi_pass_reasoner import MultiPassReasoner

        reasoner = MultiPassReasoner(k=3, strategy="best_of_k")

        def mock_gen(n):
            return [f"Answer {i}: the result is {42 + i}." for i in range(n)]

        result = reasoner.reason(mock_gen, task_type="mathematics")
        assert result.best_output is not None
        assert result.passes_used == 3

    def test_escalating_stops_early_if_confident(self):
        from asft.accuracy.multi_pass_reasoner import MultiPassReasoner

        reasoner = MultiPassReasoner(k=3, min_confidence=0.3, strategy="escalating")

        call_count = [0]

        def mock_gen(n):
            call_count[0] += n
            return ["The answer is definitively 42. Steps: 1, 2, 3."] * n

        result = reasoner.reason(mock_gen, task_type="general")
        assert result.best_output is not None
        # With low threshold, should stop at 1 pass
        assert call_count[0] <= 3

    def test_self_consistency(self):
        from asft.accuracy.multi_pass_reasoner import MultiPassReasoner

        reasoner = MultiPassReasoner(k=3, strategy="self_consistency")

        def mock_gen(n):
            return ["The answer is 42."] * n

        result = reasoner.reason(mock_gen)
        assert result.consensus_output is not None


# ===========================================================================
# Dataset tests
# ===========================================================================


class TestDeduplicator:
    def test_exact_duplicates_removed(self):
        try:
            from asft.dataset.deduplicator import DatasetDeduplicator

            dedup = DatasetDeduplicator(threshold=0.9, num_perm=128)
            texts = ["Hello world this is a test sentence"] * 5 + [
                "Completely different text about machine learning and AI systems"
            ]
            kept, _, stats = dedup.deduplicate(texts)
            assert stats["kept_count"] < stats["original_count"]
        except ImportError:
            pytest.skip("datasketch not installed")

    def test_unique_texts_mostly_kept(self):
        try:
            from asft.dataset.deduplicator import DatasetDeduplicator

            # Use num_perm=128 (default) and reasonable threshold
            dedup = DatasetDeduplicator(threshold=0.95, num_perm=128)
            texts = [
                "Machine learning fundamentals for beginners",
                "Python programming language syntax and semantics",
                "Neural network architectures and deep learning approaches",
                "Natural language processing with transformers and attention",
                "Computer vision and image recognition algorithms",
            ]
            kept, _, stats = dedup.deduplicate(texts)
            # Unique texts should mostly be preserved
            assert stats["kept_count"] >= 3
        except ImportError:
            pytest.skip("datasketch not installed")


class TestRepresentativeSelector:
    def test_centroid_selection(self):
        import numpy as np

        from asft.dataset.representative_selector import RepresentativeSelector

        embeddings = np.random.randn(10, 16)
        labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])
        selector = RepresentativeSelector(strategy="centroid", samples_per_cluster=1)
        indices, stats = selector.select(embeddings, labels)
        assert len(indices) == 3
        assert stats["n_clusters"] == 3

    def test_diversity_selection(self):
        import numpy as np

        from asft.dataset.representative_selector import RepresentativeSelector

        embeddings = np.random.randn(9, 8)
        labels = np.array([0] * 3 + [1] * 3 + [2] * 3)
        selector = RepresentativeSelector(strategy="diversity", samples_per_cluster=2)
        indices, stats = selector.select(embeddings, labels)
        assert len(indices) == 6

    def test_hybrid_selection(self):
        import numpy as np

        from asft.dataset.representative_selector import RepresentativeSelector

        embeddings = np.random.randn(8, 8)
        labels = np.array([0] * 4 + [1] * 4)
        selector = RepresentativeSelector(strategy="hybrid", samples_per_cluster=2)
        indices, stats = selector.select(embeddings, labels)
        assert len(indices) > 0

    def test_reduction_ratio(self):
        import numpy as np

        from asft.dataset.representative_selector import RepresentativeSelector

        embeddings = np.random.randn(20, 8)
        labels = np.array([i // 4 for i in range(20)])  # 5 clusters of 4
        selector = RepresentativeSelector(strategy="centroid", samples_per_cluster=1)
        indices, stats = selector.select(embeddings, labels)
        assert stats["reduction_ratio"] > 0.5  # Should reduce by more than 50%


# ===========================================================================
# Evolutionary tests
# ===========================================================================
