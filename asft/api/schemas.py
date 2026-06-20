"""
ASFT API Schemas — All Pydantic v2 request and response models.

Centralising all schemas here enforces consistent validation across
all endpoints and makes API contract changes easy to find and audit.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


class TrainRequest(BaseModel):
    model_name: str = Field(
        "Qwen/Qwen2-0.5B",
        description="HuggingFace model identifier or local path",
        min_length=2,
        max_length=256,
    )
    dataset_path: str = Field(
        ...,
        description="Path to a JSONL training dataset",
        min_length=1,
        max_length=512,
    )
    method: str = Field(
        "peft_lora",
        description="Training method: peft_lora | qlora | sparse",
    )
    max_steps: int = Field(500, ge=1, le=100_000)
    learning_rate: float = Field(2e-4, gt=0.0, le=1.0)
    batch_size: int = Field(1, ge=1, le=64)
    lora_r: int = Field(8, ge=1, le=256)
    quantization: str = Field("4bit", description="none | 4bit | 8bit")

    @field_validator("method")
    @classmethod
    def validate_method(cls, v: str) -> str:
        allowed = {"peft_lora", "qlora", "lora", "sparse"}
        if v not in allowed:
            raise ValueError(f"method must be one of {allowed}")
        return v

    @field_validator("quantization")
    @classmethod
    def validate_quantization(cls, v: str) -> str:
        allowed = {"none", "4bit", "8bit"}
        if v not in allowed:
            raise ValueError(f"quantization must be one of {allowed}")
        return v


class TrainResponse(BaseModel):
    job_id: str
    status: str
    message: str = "Training job queued"


class JobStatusResponse(BaseModel):
    job_id: str
    job_type: str
    status: str
    created_at: float
    updated_at: float
    payload: dict[str, Any] = {}
    result: dict[str, Any] | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# Skills
# ---------------------------------------------------------------------------


class SkillRouteRequest(BaseModel):
    task: str = Field(..., min_length=1, max_length=8000)
    strategy: str = Field("single", description="single | multi | consensus")
    top_k: int = Field(1, ge=1, le=6)

    @field_validator("strategy")
    @classmethod
    def validate_strategy(cls, v: str) -> str:
        allowed = {"single", "multi", "consensus"}
        if v not in allowed:
            raise ValueError(f"strategy must be one of {allowed}")
        return v


class SkillRouteResponse(BaseModel):
    selected: list[str]
    scores: dict[str, float]
    strategy: str


class SkillProcessRequest(BaseModel):
    task: str = Field(..., min_length=1, max_length=8000)
    context: str | None = Field(None, max_length=32000)
    max_tokens: int = Field(512, ge=1, le=4096)
    temperature: float = Field(0.7, ge=0.0, le=2.0)


class SkillProcessResponse(BaseModel):
    skill_name: str
    output: str
    confidence: float
    duration_seconds: float
    requires_disclaimer: bool = False
    disclaimer: str | None = None


class SkillInfoResponse(BaseModel):
    name: str
    description: str
    tags: list[str]
    performance_score: float


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------


class MemoryQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(5, ge=1, le=50)


class MemoryQueryResult(BaseModel):
    source: str
    content: str
    confidence: float


class MemoryQueryResponse(BaseModel):
    can_answer: bool
    results: list[MemoryQueryResult]
    total_found: int


class FactRequest(BaseModel):
    subject: str = Field(..., min_length=1, max_length=500)
    predicate: str = Field(..., min_length=1, max_length=200)
    object: str = Field(..., min_length=1, max_length=500)
    source: str = Field("api", max_length=100)
    confidence: float = Field(1.0, ge=0.0, le=1.0)


class FactResponse(BaseModel):
    fact_id: str
    subject: str
    predicate: str
    object: str


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class CompressRequest(BaseModel):
    dataset_path: str = Field(..., min_length=1, max_length=512)
    text_field: str = Field("text", min_length=1, max_length=64)
    output_name: str = Field("compressed", min_length=1, max_length=128)
    embedding_model: str = Field(
        "all-MiniLM-L6-v2",
        description="SentenceTransformer model for clustering"
    )


class CompressResponse(BaseModel):
    job_id: str
    status: str
    message: str = "Compression job queued"


# ---------------------------------------------------------------------------
# System
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    uptime_seconds: float


class HardwareGPU(BaseModel):
    index: int
    name: str
    vram_total_gb: float
    vram_free_gb: float


class HardwareRecommendations(BaseModel):
    precision: str
    quantization: str
    method: str
    batch_size: int
    gradient_checkpointing: bool
    flash_attention: bool


class HardwareResponse(BaseModel):
    platform: str
    cpu_brand: str
    cpu_cores: int
    ram_total_gb: float
    ram_available_gb: float
    has_cuda: bool
    has_mps: bool
    gpus: list[HardwareGPU]
    recommendations: HardwareRecommendations


class StatusResponse(BaseModel):
    status: str = "ok"
    version: str
    uptime_seconds: float
    skills_registered: list[str]
    memory_available: bool
    active_jobs: int
    hardware_summary: dict[str, Any]


# ---------------------------------------------------------------------------
# Error
# ---------------------------------------------------------------------------


class ErrorResponse(BaseModel):
    error: str
    message: str
    request_id: str | None = None


# ---------------------------------------------------------------------------
# Training Acceleration — Estimator & AutoOptimizer
# ---------------------------------------------------------------------------


class EstimateRequest(BaseModel):
    """Request a cost/time estimate before committing to a training job."""
    model_name: str = Field(..., min_length=2, max_length=256,
                            description="HuggingFace model identifier")
    dataset_size: int = Field(..., ge=1, le=100_000_000,
                              description="Number of training samples")
    method: str = Field("qlora", description="Training method: peft_lora | qlora")
    target_accuracy_gain: float = Field(
        0.05, ge=0.0, le=1.0,
        description="Desired accuracy improvement (0.05 = 5%)"
    )


class EstimateResponse(BaseModel):
    """Cost and time projection for a proposed training job."""
    estimated_gpu_hours: float
    estimated_cost_usd: float
    estimated_accuracy_gain: float
    recommendation: str          # "proceed" | "use_qlora" | "retrieve" | "use_skill" | "reject"
    reasoning: str
    roi_score: float             # expected accuracy gain per dollar


class OptimizeRequest(BaseModel):
    """Ask the AutoOptimizer what the cheapest path to capability is."""
    task: str = Field(..., min_length=1, max_length=8_000)
    domain: str = Field("general", max_length=64)
    target_accuracy: float = Field(0.8, ge=0.0, le=1.0)
    budget_usd: float | None = Field(None, ge=0.0,
                                        description="Max spend in USD. None = no budget limit.")
    allow_training: bool = Field(True, description="Allow training as a fallback option.")


class OptimizeResponse(BaseModel):
    """AutoOptimizer decision result."""
    recommended_action: str   # "use_memory" | "use_skill" | "use_qlora" | "use_lora" | "distill"
    reasoning: str
    estimated_cost_usd: float
    alternatives: list[dict[str, Any]]


# ---------------------------------------------------------------------------
# Sample Selection
# ---------------------------------------------------------------------------


class SampleSelectRequest(BaseModel):
    """Request adaptive sample selection on a dataset."""
    dataset_path: str = Field(..., min_length=1, max_length=512)
    model_name: str = Field(..., min_length=2, max_length=256)
    method: str = Field(
        "perplexity",
        description="Selection method: perplexity | el2n | random"
    )
    keep_fraction: float = Field(0.3, ge=0.05, le=1.0,
                                 description="Fraction of samples to keep (0.3 = keep 30%)")
    text_field: str = Field("text", max_length=64)


class SampleSelectResponse(BaseModel):
    job_id: str
    status: str
    message: str = "Sample selection job queued"


# ---------------------------------------------------------------------------
# Knowledge Distillation
# ---------------------------------------------------------------------------


class DistillRequest(BaseModel):
    """Request knowledge distillation from a teacher to a student model."""
    teacher_model: str = Field(..., min_length=2, max_length=256)
    student_model: str = Field(..., min_length=2, max_length=256)
    dataset_path: str = Field(..., min_length=1, max_length=512)
    temperature: float = Field(4.0, ge=1.0, le=20.0,
                               description="Distillation temperature (Hinton 2015). Higher = softer targets.")
    alpha: float = Field(0.5, ge=0.0, le=1.0,
                         description="Weight between hard labels (1.0) and soft targets (0.0)")
    max_steps: int = Field(500, ge=1, le=50_000)


class DistillResponse(BaseModel):
    job_id: str
    status: str
    message: str = "Distillation job queued"

