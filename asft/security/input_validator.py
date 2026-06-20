"""
ASFT Input Validator — Schema-validated, safety-checked input layer.

Every API endpoint and skill pack MUST route all user input through
this module before processing. This prevents:
  - Prompt injection attacks
  - Oversized inputs causing OOM
  - Unicode smuggling attacks
  - Null-byte injection
  - Control character injection
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass

from asft.core.exceptions import InputValidationError, PromptInjectionError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Limits
# ---------------------------------------------------------------------------

MAX_TASK_LENGTH: int = 8_000  # characters
MAX_CONTEXT_LENGTH: int = 32_000  # characters
MAX_QUERY_LENGTH: int = 2_000  # characters
MAX_FACT_LENGTH: int = 1_000  # characters
MAX_DATASET_PATH_LENGTH: int = 512

# ---------------------------------------------------------------------------
# Prompt injection patterns — common attack signatures
# ---------------------------------------------------------------------------

_INJECTION_PATTERNS: list[re.Pattern] = [
    # Classic role-override
    re.compile(r"ignore (all |previous |above )?(instructions|rules|prompts?)", re.I),
    re.compile(r"you are now", re.I),
    re.compile(r"disregard (your |the )?(previous |prior )?(instructions?|rules?)", re.I),
    re.compile(r"forget (everything |all )?(above|previous|instructions?)", re.I),
    # System prompt leaking
    re.compile(
        r"(reveal|show|print|output|repeat|display) (your |the )?(system )?(prompt|instructions?)",
        re.I,
    ),
    re.compile(r"(what (are|were) your (instructions?|rules?|system prompt))", re.I),
    # Jailbreak keywords
    re.compile(r"\bDAN\b"),  # "Do Anything Now" jailbreak
    re.compile(r"jailbreak", re.I),
    re.compile(r"developer mode", re.I),
    re.compile(r"act as (a |an )?(different|new|unrestricted|unfiltered)", re.I),
    # Code injection attempts via prompt
    re.compile(r"```.*?(exec|eval|import os|subprocess|system\()", re.I | re.DOTALL),
    re.compile(r"<script", re.I),
    # Multi-line instruction overrides
    re.compile(r"---+\s*(system|instructions?|prompt)\s*---+", re.I),
]


# ---------------------------------------------------------------------------
# Sanitisation helpers
# ---------------------------------------------------------------------------


def _strip_control_chars(text: str) -> str:
    """
    Remove ASCII control characters (except newline/tab which are legitimate).
    Normalise Unicode to NFC form to prevent homoglyph attacks.
    """
    text = unicodedata.normalize("NFC", text)
    # Remove control chars except \n (0x0A), \r (0x0D), \t (0x09)
    return re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)


def _check_injection(text: str) -> str | None:
    """
    Scan text for prompt injection patterns.
    Returns the first matched pattern label, or None if clean.
    """
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(text):
            return pattern.pattern
    return None


# ---------------------------------------------------------------------------
# Public validation functions
# ---------------------------------------------------------------------------


@dataclass
class ValidatedInput:
    """A sanitised, validated input string."""

    value: str
    original_length: int
    sanitised: bool  # True if any chars were stripped


def validate_task(task: str, max_length: int = MAX_TASK_LENGTH) -> ValidatedInput:
    """
    Validate and sanitise a task/prompt string.

    Raises:
        InputValidationError: if the input is empty or exceeds max length.
        PromptInjectionError: if injection patterns are detected.
    """
    if not task or not task.strip():
        raise InputValidationError("Task input must not be empty.")

    original_length = len(task)

    if original_length > max_length:
        raise InputValidationError(
            f"Task exceeds maximum length of {max_length} characters "
            f"(received {original_length})."
        )

    cleaned = _strip_control_chars(task)
    sanitised = cleaned != task

    # Injection check on the cleaned text
    matched_pattern = _check_injection(cleaned)
    if matched_pattern:
        logger.warning(
            "Prompt injection detected | pattern=%s | snippet=%s", matched_pattern, cleaned[:80]
        )
        raise PromptInjectionError(
            "Input contains patterns associated with prompt injection attacks."
        )

    return ValidatedInput(value=cleaned, original_length=original_length, sanitised=sanitised)


def validate_query(query: str) -> ValidatedInput:
    """Validate a memory/search query string."""
    if not query or not query.strip():
        raise InputValidationError("Query must not be empty.")
    if len(query) > MAX_QUERY_LENGTH:
        raise InputValidationError(
            f"Query exceeds maximum length of {MAX_QUERY_LENGTH} characters."
        )
    cleaned = _strip_control_chars(query)
    return ValidatedInput(value=cleaned, original_length=len(query), sanitised=cleaned != query)


def validate_fact(subject: str, predicate: str, obj: str) -> tuple[str, str, str]:
    """
    Validate semantic fact triple components.
    Returns sanitised (subject, predicate, object).
    """
    parts = []
    for label, val in [("subject", subject), ("predicate", predicate), ("object", obj)]:
        if not val or not val.strip():
            raise InputValidationError(f"Fact {label} must not be empty.")
        if len(val) > MAX_FACT_LENGTH:
            raise InputValidationError(
                f"Fact {label} exceeds maximum length of {MAX_FACT_LENGTH} characters."
            )
        parts.append(_strip_control_chars(val))
    return tuple(parts)  # type: ignore[return-value]


def validate_dataset_path(path: str) -> str:
    """
    Validate a dataset file path.
    Prevents path traversal attacks.
    """
    if not path:
        raise InputValidationError("Dataset path must not be empty.")
    if len(path) > MAX_DATASET_PATH_LENGTH:
        raise InputValidationError("Dataset path is too long.")

    # Prevent path traversal
    if ".." in path.replace("\\", "/"):
        raise InputValidationError(
            "Dataset path must not contain directory traversal sequences ('..')."
        )

    # Only allow safe characters in paths
    if not re.match(r"^[\w\s\-_./\\:]+$", path):
        raise InputValidationError("Dataset path contains disallowed characters.")

    return path
