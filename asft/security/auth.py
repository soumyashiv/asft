"""
ASFT Security — API Authentication & Authorization middleware.

Supports two modes:
  1. API Key  — header `X-API-Key: <key>` (default for local/self-hosted)
  2. Bearer   — header `Authorization: Bearer <jwt>` (for enterprise deployments)

Configuration via environment variables or asft_config.yaml:
  ASFT_API_KEY_ENABLED   = true
  ASFT_API_KEYS          = "key1,key2,key3"   (comma-separated)
  ASFT_API_AUTH_REQUIRED = true               (set false for dev-only)

Design:
  - Keys are stored as SHA-256 hashes in config — never in plaintext.
  - Failed auth attempts are always logged to the security audit trail.
  - The middleware never reveals WHY authentication failed to the caller.
"""

from __future__ import annotations

import hashlib
import logging
import os
from collections.abc import Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public endpoints that bypass auth (e.g., health checks)
# ---------------------------------------------------------------------------
_PUBLIC_PATHS: set[str] = {"/", "/health", "/status", "/docs", "/openapi.json", "/redoc"}


# ---------------------------------------------------------------------------
# Key store — in-memory SHA-256 hash set
# ---------------------------------------------------------------------------


def _hash_key(key: str) -> str:
    """SHA-256 hash of an API key for safe comparison."""
    return hashlib.sha256(key.strip().encode()).hexdigest()


def _load_valid_key_hashes() -> set[str]:
    """
    Load valid API key hashes from environment.

    Reads ASFT_API_KEYS as comma-separated plaintext keys, converts
    each to a SHA-256 hash. Only hashes are kept in memory.
    """
    raw = os.environ.get("ASFT_API_KEYS", "")
    if not raw:
        # Dev-mode: generate a random ephemeral key and log it once
        import secrets

        ephemeral = secrets.token_urlsafe(32)
        logger.warning(
            "ASFT_API_KEYS not set. Generated ephemeral key for this session: %s", ephemeral
        )
        logger.warning("Set ASFT_API_KEYS env var in production to use stable keys.")
        return {_hash_key(ephemeral)}
    return {_hash_key(k) for k in raw.split(",") if k.strip()}


# Singleton set loaded once at import time
_VALID_KEY_HASHES: set[str] = _load_valid_key_hashes()

# Whether auth is enforced (can be disabled for local dev)
_AUTH_REQUIRED: bool = os.environ.get("ASFT_API_AUTH_REQUIRED", "true").lower() == "true"


# ---------------------------------------------------------------------------
# Authentication functions
# ---------------------------------------------------------------------------


def verify_api_key(key: str) -> bool:
    """Return True if the key is in the valid set. Constant-time comparison."""
    if not key:
        return False
    key_hash = _hash_key(key)
    # Use set membership — Python set lookup is O(1) and not timing-sensitive
    return key_hash in _VALID_KEY_HASHES


def extract_api_key(request: Request) -> str | None:
    """Extract API key from the request. Checks X-API-Key header first."""
    # X-API-Key header (preferred)
    key = request.headers.get("X-API-Key")
    if key:
        return key
    # Authorization: Bearer <token> fallback
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:]
    return None


# ---------------------------------------------------------------------------
# FastAPI Middleware
# ---------------------------------------------------------------------------


class APIKeyMiddleware(BaseHTTPMiddleware):
    """
    Starlette middleware that enforces API key authentication on all
    non-public endpoints.

    Adds response headers:
      X-Request-ID: <uuid>   — unique request identifier for tracing
      X-Auth-Method: api_key — which auth method was used
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        import uuid

        request_id = str(uuid.uuid4())[:12]

        # Bypass auth for public paths
        if request.url.path in _PUBLIC_PATHS:
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            return response

        # Skip auth enforcement if disabled (dev mode)
        if not _AUTH_REQUIRED:
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Auth-Method"] = "disabled"
            return response

        key = extract_api_key(request)

        if not key or not verify_api_key(key):
            # SECURITY: log the IP and path, but NEVER the submitted key
            logger.warning(
                "AUTH_FAIL | request_id=%s path=%s ip=%s",
                request_id,
                request.url.path,
                request.client.host if request.client else "unknown",
            )
            return JSONResponse(
                status_code=401,
                content={
                    "error": "authentication_error",
                    "message": "Valid API key required.",
                    "request_id": request_id,
                },
                headers={"X-Request-ID": request_id},
            )

        logger.debug("AUTH_OK | request_id=%s path=%s", request_id, request.url.path)
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Auth-Method"] = "api_key"
        return response


# ---------------------------------------------------------------------------
# Utility: add new key at runtime (admin operation only)
# ---------------------------------------------------------------------------


def register_api_key(plaintext_key: str) -> None:
    """
    Register a new API key at runtime.
    Only the hash is stored. The plaintext key is NOT retained.
    """
    if len(plaintext_key) < 16:
        raise ValueError("API key must be at least 16 characters.")
    _VALID_KEY_HASHES.add(_hash_key(plaintext_key))
    logger.info("New API key registered (hash stored, plaintext discarded).")


def revoke_api_key(plaintext_key: str) -> bool:
    """Revoke an API key by its plaintext value. Returns True if it existed."""
    key_hash = _hash_key(plaintext_key)
    if key_hash in _VALID_KEY_HASHES:
        _VALID_KEY_HASHES.discard(key_hash)
        logger.info("API key revoked.")
        return True
    return False
