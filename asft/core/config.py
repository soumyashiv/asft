"""
ASFT Central Configuration
===========================
Single source of truth for all framework settings.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings


class MemoryConfig(BaseModel):
    vector_backend: Literal["chromadb", "faiss", "qdrant"] = "chromadb"
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_device: str = "cpu"
    chroma_persist_dir: str = "./asft_data/chroma"
    chroma_host: str | None = None
    chroma_port: int = 8000
    faiss_index_path: str = "./asft_data/faiss.index"
    faiss_index_type: str = "Flat"
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "asft_memory"
    sqlite_path: str = "./asft_data/memory.db"
    working_memory_max_tokens: int = 8192
    episodic_max_events: int = 10_000
    vector_top_k: int = 10
    consolidation_interval_hours: float = 24.0
    consolidation_min_events: int = 100


class SparseTrainingConfig(BaseModel):
    sparsity_ratio: float = Field(0.95, ge=0.0, le=1.0)
    selection_method: Literal["magnitude", "gradient", "fisher", "activation"] = "activation"
    save_delta_separately: bool = True
    delta_output_dir: str = "./asft_data/deltas"
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.03
    learning_rate: float = 2e-4
    max_steps: int = 500
    eval_steps: int = 50
    save_steps: int = 100
    dynamic_sparsity: bool = True


class LoRAConfig(BaseModel):
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    target_modules: list[str] | None = None


class HardwareConfig(BaseModel):
    precision: Literal["fp32", "fp16", "bf16", "int8", "int4"] | None = None
    quantization: Literal["none", "8bit", "4bit"] | None = None
    training_method: Literal["full", "lora", "qlora", "sparse", "asft"] | None = None
    batch_size: int | None = None
    gradient_checkpointing: bool | None = None
    cpu_offload: bool | None = None
    num_workers: int | None = None
    max_model_size_gb: float | None = None


class DatasetConfig(BaseModel):
    dedup_threshold: float = Field(0.85, ge=0.0, le=1.0)
    dedup_num_perm: int = 128
    cluster_method: Literal["kmeans", "dbscan"] = "kmeans"
    cluster_reduction_ratio: float = Field(0.3, ge=0.0, le=1.0)
    min_cluster_size: int = 3
    compressed_output_dir: str = "./asft_data/datasets"


class EvolutionaryConfig(BaseModel):
    population_size: int = 20
    elite_fraction: float = 0.2
    mutation_rate: float = 0.3
    max_generations: int = 50
    convergence_threshold: float = 0.001
    fitness_eval_samples: int = 50


class AccuracyConfig(BaseModel):
    multi_pass_k: int = 3
    min_confidence_threshold: float = 0.7
    max_critique_rounds: int = 2
    enable_code_execution: bool = True
    enable_web_search: bool = False


class BenchmarkConfig(BaseModel):
    output_dir: str = "./asft_data/benchmarks"
    methods_to_compare: list[str] = ["full_ft", "lora", "qlora", "sparse", "asft"]
    eval_samples: int = 100
    timeout_seconds: int = 300


class SkillPackConfig(BaseModel):
    skill_pack_dir: str = "./asft_data/skill_packs"
    auto_route: bool = True
    routing_top_k: int = 3
    merge_strategy: Literal["weighted_avg", "voting", "cascade"] = "weighted_avg"


class APIConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8080
    workers: int = 1
    reload: bool = False
    log_level: str = "info"


class ModelConfig(BaseModel):
    model_name_or_path: str = "Qwen/Qwen2-0.5B"
    tokenizer_name: str | None = None
    cache_dir: str = "./asft_data/model_cache"
    trust_remote_code: bool = True
    max_sequence_length: int = 2048


class ASFTConfig(BaseSettings):
    """Master ASFT configuration. Env vars: ASFT_<FIELD> or ASFT_<SUB>__<FIELD>."""

    model_config = {"env_prefix": "ASFT_", "env_nested_delimiter": "__"}

    data_dir: str = "./asft_data"
    log_level: str = "INFO"
    log_dir: str = "./asft_data/logs"

    model: ModelConfig = Field(default_factory=ModelConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    sparse: SparseTrainingConfig = Field(default_factory=SparseTrainingConfig)
    lora: LoRAConfig = Field(default_factory=LoRAConfig)
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    evolutionary: EvolutionaryConfig = Field(default_factory=EvolutionaryConfig)
    accuracy: AccuracyConfig = Field(default_factory=AccuracyConfig)
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)
    skills: SkillPackConfig = Field(default_factory=SkillPackConfig)
    api: APIConfig = Field(default_factory=APIConfig)

    @field_validator("data_dir", mode="before")
    @classmethod
    def expand_data_dir(cls, v: str) -> str:
        return str(Path(v).expanduser().resolve())

    def ensure_dirs(self) -> None:
        dirs = [
            self.data_dir, self.log_dir,
            self.memory.chroma_persist_dir,
            self.sparse.delta_output_dir,
            self.dataset.compressed_output_dir,
            self.benchmark.output_dir,
            self.skills.skill_pack_dir,
            self.model.cache_dir,
        ]
        for d in dirs:
            Path(d).mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_yaml(cls, path: str | Path) -> ASFTConfig:
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: str | Path) -> None:
        import yaml
        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)

    def apply_hardware_profile(self, profile) -> None:
        """Override unset hardware settings from auto-detected profile."""
        hw = self.hardware
        if hw.precision is None:
            hw.precision = profile.recommended_precision
        if hw.quantization is None:
            hw.quantization = profile.recommended_quantization
        if hw.training_method is None:
            hw.training_method = profile.recommended_training_method
        if hw.batch_size is None:
            hw.batch_size = profile.recommended_batch_size
        if hw.gradient_checkpointing is None:
            hw.gradient_checkpointing = profile.recommended_gradient_checkpointing
        if hw.cpu_offload is None:
            hw.cpu_offload = profile.offload_to_cpu
        if hw.num_workers is None:
            hw.num_workers = profile.num_workers
        if hw.max_model_size_gb is None:
            hw.max_model_size_gb = profile.max_trainable_model_gb
