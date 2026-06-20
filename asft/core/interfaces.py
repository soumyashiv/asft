"""
ASFT Abstract Interfaces — All subsystem contracts defined as ABCs.

Every concrete implementation (memory backend, trainer, skill pack, verifier)
MUST implement the corresponding interface. This enforces the dependency
inversion principle and allows clean unit testing via mocks.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Memory Interfaces
# ---------------------------------------------------------------------------


@dataclass
class MemoryQueryResult:
    """A single result returned from a memory query."""

    source: str  # which memory tier returned this
    content: Any  # the retrieved content
    confidence: float  # relevance score 0–1
    metadata: dict[str, Any] = field(default_factory=dict)


class IMemoryStore(abc.ABC):
    """Abstract contract for all vector memory backends (Qdrant, FAISS, etc.)."""

    @abc.abstractmethod
    async def add(self, content: str, metadata: dict | None = None) -> str:
        """Persist a single item. Returns a unique item ID."""

    @abc.abstractmethod
    async def update(self, item_id: str, content: str, metadata: dict | None = None) -> bool:
        """Update an existing item."""

    @abc.abstractmethod
    async def delete(self, item_id: str) -> bool:
        """Delete an item by ID. Returns True if deleted."""

    @abc.abstractmethod
    async def search(self, query_vector: list[float], top_k: int = 5) -> list[MemoryQueryResult]:
        """Retrieve top-k items most relevant to the query vector."""

    @abc.abstractmethod
    async def batch_insert(
        self, contents: list[str], metadatas: list[dict] | None = None
    ) -> list[str]:
        """Insert multiple items efficiently. Returns a list of IDs."""

    @abc.abstractmethod
    async def health_check(self) -> bool:
        """Return True if the backend is healthy and reachable."""


# ---------------------------------------------------------------------------
# Training Interfaces
# ---------------------------------------------------------------------------


@dataclass
class TrainingConfig:
    """Validated, hardware-aware training configuration."""

    model_name: str
    dataset_path: str
    method: str = "peft_lora"  # peft_lora | qlora | sparse
    max_steps: int = 500
    learning_rate: float = 2e-4
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    quantization: str = "4bit"  # none | 4bit | 8bit
    output_dir: str = "./asft_data/checkpoints"
    eval_steps: int = 50
    save_steps: int = 100
    sparsity_ratio: float = 0.95  # only used by sparse method
    fsdp: str | None = None  # e.g., "full_shard auto_wrap"
    deepspeed: str | None = None  # e.g., "ds_config.json"


@dataclass
class TrainingResult:
    """Outcome of a completed training job."""

    job_id: str
    status: str  # completed | failed | cancelled
    method: str
    final_loss: float | None = None
    eval_loss: float | None = None
    steps_completed: int = 0
    duration_seconds: float = 0.0
    checkpoint_path: str | None = None
    error_message: str | None = None


class ITrainer(abc.ABC):
    """Abstract contract for all training backends."""

    @abc.abstractmethod
    def train(self, config: TrainingConfig) -> TrainingResult:
        """Run training and return a result. Blocking."""

    @abc.abstractmethod
    def supports_method(self, method: str) -> bool:
        """Return True if this trainer handles the given method."""


# ---------------------------------------------------------------------------
# Skill Pack Interfaces
# ---------------------------------------------------------------------------


@dataclass
class SkillInput:
    """Validated, typed input for a skill pack."""

    task: str
    context: str | None = None
    max_tokens: int = 512
    temperature: float = 0.7
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SkillOutput:
    """Typed, auditable output from a skill pack."""

    skill_name: str
    output: str
    confidence: float  # 0–1 calibrated confidence
    duration_seconds: float
    requires_disclaimer: bool = False
    disclaimer: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ISkillPack(abc.ABC):
    """Abstract contract for all skill packs."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Unique skill pack identifier."""

    @property
    @abc.abstractmethod
    def description(self) -> str:
        """Human-readable description."""

    @property
    @abc.abstractmethod
    def tags(self) -> list[str]:
        """Domain tags used for routing."""

    @abc.abstractmethod
    def process(self, skill_input: SkillInput, model=None, tokenizer=None) -> SkillOutput:
        """Execute the skill on the validated input."""

    @abc.abstractmethod
    def evaluate(self, sample_input: str, sample_output: str) -> float:
        """Score output quality. Returns 0–1."""

    def health_check(self) -> bool:
        """Override to add custom health checks. Default: always healthy."""
        return True


# ---------------------------------------------------------------------------
# Verifier Interfaces
# ---------------------------------------------------------------------------


@dataclass
class VerificationResult:
    """Result of output verification."""

    verified: bool
    method: str  # "math_cas" | "code_sandbox" | "memory" | "none"
    confidence: float  # 0–1
    details: str = ""
    corrections: str | None = None
    safe_to_execute: bool = True  # set False if code deemed unsafe


class IVerifier(abc.ABC):
    """Abstract contract for output verifiers."""

    @abc.abstractmethod
    def verify(self, output: str, task: str, task_type: str = "general") -> VerificationResult:
        """Verify the output. Returns a VerificationResult."""


# ---------------------------------------------------------------------------
# Job Store Interfaces
# ---------------------------------------------------------------------------


@dataclass
class JobRecord:
    """Persistent record of a background job."""

    job_id: str
    job_type: str  # "training" | "compression"
    status: str  # "queued" | "running" | "completed" | "failed" | "cancelled"
    created_at: float
    updated_at: float
    payload: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] | None = None
    error: str | None = None


class IJobStore(abc.ABC):
    """Abstract contract for persistent job state storage."""

    @abc.abstractmethod
    async def create(self, job_id: str, job_type: str, payload: dict[str, Any]) -> JobRecord:
        """Create a new job record."""

    @abc.abstractmethod
    async def get(self, job_id: str) -> JobRecord | None:
        """Retrieve a job record by ID. Returns None if not found."""

    @abc.abstractmethod
    async def update_status(
        self, job_id: str, status: str, result: dict | None = None, error: str | None = None
    ) -> None:
        """Atomically update job status and result."""

    @abc.abstractmethod
    async def list_jobs(
        self, job_type: str | None = None, status: str | None = None, limit: int = 50
    ) -> list[JobRecord]:
        """List jobs, optionally filtered by type and status."""


# ---------------------------------------------------------------------------
# Observability Interfaces
# ---------------------------------------------------------------------------


class IMetricsCollector(abc.ABC):
    """Abstract contract for metrics collection."""

    @abc.abstractmethod
    def increment(
        self, name: str, value: float = 1.0, labels: dict[str, str] | None = None
    ) -> None:
        """Increment a counter metric."""

    @abc.abstractmethod
    def gauge(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        """Set an absolute gauge value."""

    @abc.abstractmethod
    def histogram(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        """Record a histogram observation (e.g., latency)."""


# ---------------------------------------------------------------------------
# Optimization & Distillation Interfaces
# ---------------------------------------------------------------------------


@dataclass
class OptimizationDecision:
    """Decision produced by the AutoOptimizer."""

    action: str  # "train" | "distill" | "rag" | "skill" | "reject"
    method: str  # "full" | "qlora" | "sparse"
    reasoning: str  # Explanation of why this action was chosen
    estimated_cost: float  # Estimated cost in USD
    projected_accuracy: float  # Projected final accuracy


class IOptimizer(abc.ABC):
    """Abstract contract for the training decision engine."""

    @abc.abstractmethod
    def decide(
        self, task: str, domain: str, target_accuracy: float, budget_usd: float
    ) -> OptimizationDecision:
        """Evaluate alternatives and return a cost-aware training decision."""


@dataclass
class DistillationConfig:
    """Configuration for Knowledge Distillation."""

    teacher_model: str
    student_model: str
    dataset_path: str
    temperature: float = 2.0
    alpha_ce: float = 0.5
    alpha_distill: float = 0.5
    output_dir: str = "./asft_data/distilled"


class IDistiller(abc.ABC):
    """Abstract contract for Knowledge Distillation."""

    @abc.abstractmethod
    def distill(self, config: DistillationConfig) -> TrainingResult:
        """Run the distillation process."""


# ---------------------------------------------------------------------------
# Sandbox Interfaces
# ---------------------------------------------------------------------------


class ISandbox(abc.ABC):
    """Abstract contract for secure code execution sandbox."""

    @abc.abstractmethod
    async def execute(self, code: str, timeout: int = 5) -> str:
        """Execute code securely and return the output."""

    @abc.abstractmethod
    async def terminate(self) -> None:
        """Force terminate the sandbox and clean up resources."""

    @abc.abstractmethod
    async def health_check(self) -> bool:
        """Return True if the sandbox environment is healthy."""


# ---------------------------------------------------------------------------
# Evaluation Interfaces
# ---------------------------------------------------------------------------


class IEvaluationHarness(abc.ABC):
    """Abstract contract for model evaluation harness (e.g. lm-evaluation-harness)."""

    @abc.abstractmethod
    async def evaluate(self, model_path: str, tasks: list[str]) -> dict[str, Any]:
        """Run benchmark tasks on a model and return metrics."""
