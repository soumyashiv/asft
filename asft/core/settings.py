"""
ASFT Central Settings — Pydantic-Settings based configuration.

All values can be overridden via environment variables.
Prefix: ASFT_

Examples:
    ASFT_API_KEYS=key1,key2
    ASFT_ALLOWED_ORIGINS=https://myapp.com,https://staging.myapp.com
    ASFT_MAX_TRAINING_WORKERS=2
    ASFT_DB_PATH=/data/asft/episodic.db
"""
from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Any

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class ASFTSettings(BaseSettings):
    """
    Master configuration for the ASFT framework.
    All fields are overridable via ASFT_<FIELD_NAME> environment variables.
    """

    model_config = SettingsConfigDict(
        env_prefix="ASFT_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # API / Server
    # ------------------------------------------------------------------
    api_keys_str: str = Field(
        default="",
        validation_alias="ASFT_API_KEYS",
        description="Comma-separated list of valid API keys. Empty = auth disabled (dev only).",
    )
    allowed_origins_str: str = Field(
        default="http://localhost:3000,http://localhost:8080",
        validation_alias="ASFT_ALLOWED_ORIGINS",
        description="Allowed CORS origins. Never use ['*'] in production.",
    )

    @property
    def api_keys(self) -> List[str]:
        return [k.strip() for k in self.api_keys_str.split(",") if k.strip()]

    @property
    def allowed_origins(self) -> List[str]:
        return [o.strip() for o in self.allowed_origins_str.split(",") if o.strip()]
    rate_limit_per_minute: int = Field(default=120, ge=1, le=10_000)
    rate_limit_burst: int = Field(default=30, ge=1, le=1_000)
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000, ge=1, le=65535)

    # ------------------------------------------------------------------
    # Training Workers
    # ------------------------------------------------------------------
    max_training_workers: int = Field(
        default=1,
        ge=1,
        le=8,
        description="Number of parallel training worker processes.",
    )
    training_timeout_seconds: int = Field(
        default=86_400,  # 24h
        ge=60,
        description="Maximum wall-clock time for a single training job.",
    )

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------
    data_dir: Path = Field(
        default=Path("./asft_data"),
        description="Root directory for all persisted ASFT data.",
    )

    @property
    def episodic_db_path(self) -> Path:
        return self.data_dir / "episodic.db"

    @property
    def semantic_db_path(self) -> Path:
        return self.data_dir / "semantic.db"

    @property
    def checkpoint_dir(self) -> Path:
        return self.data_dir / "checkpoints"

    @property
    def benchmark_dir(self) -> Path:
        return self.data_dir / "benchmarks"

    @property
    def dataset_dir(self) -> Path:
        return self.data_dir / "datasets"

    # ------------------------------------------------------------------
    # Training Defaults
    # ------------------------------------------------------------------
    default_training_method: str = Field(
        default="qlora",
        description="Default method: peft_lora | qlora | auto",
    )
    default_quantization: str = Field(
        default="4bit",
        description="Default quantization: none | 4bit | 8bit",
    )
    default_lora_r: int = Field(default=8, ge=1, le=256)
    default_lora_alpha: int = Field(default=16, ge=1, le=512)
    default_max_steps: int = Field(default=500, ge=1)

    # ------------------------------------------------------------------
    # Training Acceleration
    # ------------------------------------------------------------------
    # AutoOptimizer: refuse training if estimated ROI is below this
    min_training_roi: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Minimum expected accuracy gain per training dollar to approve a job.",
    )
    # Dataset compression target
    dataset_compression_target: float = Field(
        default=0.5,
        ge=0.05,
        le=0.99,
        description="Target dataset size as fraction of original (0.5 = 50% reduction).",
    )
    # Sample selection: EL2N batch size for importance scoring
    sample_selection_probe_steps: int = Field(
        default=50,
        ge=10,
        le=500,
        description="Number of gradient steps used to score sample importance.",
    )

    # ------------------------------------------------------------------
    # Memory System
    # ------------------------------------------------------------------
    episodic_memory_max_items: int = Field(default=100_000, ge=1_000)
    semantic_memory_max_facts: int = Field(default=50_000, ge=1_000)
    working_memory_max_items: int = Field(default=1_000, ge=100)
    vector_memory_backend: str = Field(
        default="chroma",
        description="Vector backend: chroma | qdrant | faiss",
    )

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------
    log_level: str = Field(default="INFO")
    log_format: str = Field(default="json", description="json | text")
    enable_metrics: bool = Field(default=True)
    metrics_port: int = Field(default=9090, ge=1024, le=65535)

    # ------------------------------------------------------------------
    # Security
    # ------------------------------------------------------------------
    enable_prompt_injection_check: bool = Field(default=True)
    max_task_length: int = Field(default=8_000, ge=100, le=100_000)
    max_context_length: int = Field(default=32_000, ge=1_000, le=500_000)

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v):
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid:
            raise ValueError(f"log_level must be one of {valid}")
        return v.upper()

    def ensure_dirs(self) -> None:
        """Create all required data directories."""
        for d in [
            self.data_dir,
            self.checkpoint_dir,
            self.benchmark_dir,
            self.dataset_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)

    def is_auth_enabled(self) -> bool:
        return len(self.api_keys) > 0

    def __repr__(self) -> str:
        # Never log secrets
        return (
            f"ASFTSettings(host={self.host}, port={self.port}, "
            f"auth={'enabled' if self.is_auth_enabled() else 'DISABLED'}, "
            f"workers={self.max_training_workers}, "
            f"vector_backend={self.vector_memory_backend})"
        )


@lru_cache(maxsize=1)
def get_settings() -> ASFTSettings:
    """
    Return the global settings singleton.
    Cached after first call — call get_settings.cache_clear() in tests.
    """
    settings = ASFTSettings()
    settings.ensure_dirs()
    logger.info("ASFT Settings loaded: %s", settings)
    return settings
