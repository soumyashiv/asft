"""Pytest configuration and shared fixtures for ASFT test suite."""
import pytest


@pytest.fixture(scope="session")
def hardware():
    from asft.core.hardware_profiler import detect_hardware
    return detect_hardware()


@pytest.fixture(scope="session")
def default_config(tmp_path_factory):
    from asft.core.config import ASFTConfig
    data_dir = str(tmp_path_factory.mktemp("asft_test_data"))
    cfg = ASFTConfig(data_dir=data_dir)
    cfg.ensure_dirs()
    return cfg


@pytest.fixture
def working_memory():
    from asft.memory.working_memory import WorkingMemory
    return WorkingMemory(max_items=50)


@pytest.fixture
def episodic_memory():
    from asft.memory.episodic_memory import EpisodicMemory
    return EpisodicMemory(db_path=":memory:")


@pytest.fixture
def semantic_memory():
    from asft.memory.semantic_memory import SemanticMemory
    return SemanticMemory(db_path=":memory:")


@pytest.fixture
def skill_registry():
    from asft.core.registry import ASFTRegistry
    from asft.skills.packs.coding import CodingSkillPack
    from asft.skills.packs.mathematics import MathematicsSkillPack
    from asft.skills.packs.research import ResearchSkillPack
    reg = ASFTRegistry()
    for Pack in [CodingSkillPack, MathematicsSkillPack, ResearchSkillPack]:
        p = Pack()
        reg.register_skill(p.meta.name, p)
    return reg


@pytest.fixture
def confidence_scorer():
    from asft.accuracy.confidence_scorer import ConfidenceScorer
    return ConfidenceScorer()
