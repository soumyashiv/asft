"""
ASFT Exception Hierarchy — Typed, structured exceptions for all subsystems.

Design principles:
  - Every exception carries a machine-readable `code` for API serialization.
  - HTTP-mapped exceptions include a `status_code` for FastAPI error handlers.
  - Security exceptions never leak internal details to callers.
"""
from __future__ import annotations

from typing import Any, Optional


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


class ASFTError(Exception):
    """Root of the ASFT exception tree."""

    code: str = "asft_error"
    status_code: int = 500

    def __init__(self, message: str, *, detail: Optional[Any] = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail

    def to_dict(self) -> dict:
        return {"error": self.code, "message": self.message}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class ConfigurationError(ASFTError):
    """Raised when required configuration is missing or invalid."""
    code = "configuration_error"
    status_code = 500


# ---------------------------------------------------------------------------
# Security
# ---------------------------------------------------------------------------


class SecurityError(ASFTError):
    """Base for all security-related exceptions. Never leak internal detail."""
    code = "security_error"
    status_code = 403


class AuthenticationError(SecurityError):
    """Invalid or missing API credentials."""
    code = "authentication_error"
    status_code = 401


class AuthorizationError(SecurityError):
    """Caller lacks permission for this operation."""
    code = "authorization_error"
    status_code = 403


class RateLimitError(SecurityError):
    """Request rate limit exceeded."""
    code = "rate_limit_error"
    status_code = 429


class SandboxViolationError(SecurityError):
    """Code execution was blocked by the security sandbox."""
    code = "sandbox_violation"
    status_code = 400


class InputValidationError(ASFTError):
    """Input failed schema or safety validation."""
    code = "input_validation_error"
    status_code = 422


class PromptInjectionError(SecurityError):
    """Detected prompt injection attempt in user input."""
    code = "prompt_injection_detected"
    status_code = 400


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


class TrainingError(ASFTError):
    """Base for training pipeline errors."""
    code = "training_error"
    status_code = 500


class ModelNotFoundError(TrainingError):
    """Model identifier could not be resolved."""
    code = "model_not_found"
    status_code = 404


class InsufficientResourcesError(TrainingError):
    """Not enough GPU/CPU/RAM for the requested operation."""
    code = "insufficient_resources"
    status_code = 400


class JobNotFoundError(TrainingError):
    """Training job ID does not exist."""
    code = "job_not_found"
    status_code = 404


class CheckpointError(TrainingError):
    """Failed to save or load a checkpoint."""
    code = "checkpoint_error"
    status_code = 500


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------


class MemoryError(ASFTError):
    """Base for memory subsystem errors."""
    code = "memory_error"
    status_code = 500


class MemoryBackendError(MemoryError):
    """Vector database or storage backend failure."""
    code = "memory_backend_error"
    status_code = 503


class MemoryPoisoningError(SecurityError):
    """Detected attempt to inject malicious content into memory."""
    code = "memory_poisoning_detected"
    status_code = 400


# ---------------------------------------------------------------------------
# Skills
# ---------------------------------------------------------------------------


class SkillError(ASFTError):
    """Base for skill pack errors."""
    code = "skill_error"
    status_code = 500


class SkillNotFoundError(SkillError):
    """Requested skill pack is not registered."""
    code = "skill_not_found"
    status_code = 404


class SkillExecutionError(SkillError):
    """Skill pack failed during task execution."""
    code = "skill_execution_error"
    status_code = 500


class SkillRoutingError(SkillError):
    """No suitable skill pack could be selected for the task."""
    code = "skill_routing_error"
    status_code = 400


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class DatasetError(ASFTError):
    """Base for dataset pipeline errors."""
    code = "dataset_error"
    status_code = 500


class DatasetNotFoundError(DatasetError):
    """Dataset file or path does not exist."""
    code = "dataset_not_found"
    status_code = 404


class DatasetCompressionError(DatasetError):
    """Dataset compression pipeline failed."""
    code = "dataset_compression_error"
    status_code = 500


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


class VerificationError(ASFTError):
    """Output verification subsystem failure."""
    code = "verification_error"
    status_code = 500
