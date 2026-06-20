"""
ASFT Base Skill Pack — Refactored for typed JSON contracts.

REPLACES the original `process()` which returned simple prompt strings.
Now returns structured `SkillOutput` and expects typed `SkillInput`
from the API schemas.
"""
import logging
import time

from asft.core.interfaces import ISkillPack, SkillInput, SkillOutput

logger = logging.getLogger(__name__)


class BaseSkillPack(ISkillPack):
    """
    Base implementation for all skill packs.
    Enforces typed input/output and tracks execution duration.
    """
    
    @property
    def name(self) -> str:
        return self.__class__.__name__.replace("Skill", "").lower()
        
    @property
    def description(self) -> str:
        return "Base skill pack."
        
    @property
    def tags(self) -> list[str]:
        return ["general"]
        
    def process(self, skill_input: SkillInput, model=None, tokenizer=None) -> SkillOutput:
        """
        Execute the skill. This should be overridden by subclasses.
        """
        start = time.time()
        
        # Default behavior: Echo the task
        output_text = f"Processed: {skill_input.task}"
        
        duration = round(time.time() - start, 3)
        return SkillOutput(
            skill_name=self.name,
            output=output_text,
            confidence=1.0,
            duration_seconds=duration,
        )
        
    def evaluate(self, sample_input: str, sample_output: str) -> float:
        """Score output quality. Returns 0-1."""
        return 0.8

# --- LEGACY COMPATIBILITY LAYER ---
from dataclasses import dataclass, field
from typing import Any

@dataclass
class SkillMeta:
    name: str = ""
    domain: str = ""
    description: str = ""
    tags: list[str] = field(default_factory=list)
    performance_score: float = 1.0

@dataclass
class SkillResult:
    skill_name: str
    output: str
    confidence: float
    duration_seconds: float
    metadata: dict[str, Any] = field(default_factory=dict)

class SkillPack:
    def __init__(self, name: str, pack_dir=None):
        self.meta = SkillMeta(name=name)
        self.pack_dir = pack_dir
        
    def process(self, task_input: str, model=None, tokenizer=None, **kwargs) -> SkillResult:
        raise NotImplementedError

    def evaluate(self, sample_input: str, sample_output: str) -> float:
        return 0.0

    def record_usage(self, success: bool, score: float):
        pass
        
    def get_prompt_template(self) -> str:
        return ""

