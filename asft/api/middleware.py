"""
ASFT API Middleware — Rate limiting, request ID injection, structured logging.

Middleware stack (applied in order, outermost first):
  1. APIKeyMiddleware  — authentication (from security/auth.py)
  2. RateLimitMiddleware — per-IP request throttling
  3. RequestLoggingMiddleware — structured access log

All middleware adds X-Request-ID to responses for distributed tracing.
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict
from collections.abc import Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rate Limiter — sliding window, per-IP
# ---------------------------------------------------------------------------


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Sliding window rate limiter.

    Defaults: 60 requests per minute per IP address.
    Configure via constructor arguments.

    The sliding window is stored in-process memory. For distributed
    deployments, replace with a Redis-backed rate limiter.
    """

    def __init__(
        self,
        app,
        requests_per_minute: int = 60,
        burst_limit: int = 20,   # max requests in a 10-second window
    ) -> None:
        super().__init__(app)
        self._rpm = requests_per_minute
        self._burst = burst_limit
        # {ip: [(timestamp, count), ...]}
        self._windows: dict[str, list] = defaultdict(list)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        ip = request.client.host if request.client else "unknown"
        now = time.time()

        # Clean events older than 60 seconds
        window = self._windows[ip]
        window[:] = [(ts, c) for ts, c in window if now - ts < 60]

        # Count requests in current window
        total_in_window = sum(c for _, c in window)

        # Count requests in last 10 seconds (burst window)
        burst_count = sum(c for ts, c in window if now - ts < 10)

        if total_in_window >= self._rpm:
            logger.warning("RATE_LIMIT | ip=%s requests_in_window=%d", ip, total_in_window)
            return JSONResponse(
                status_code=429,
                content={
                    "error": "rate_limit_error",
                    "message": f"Rate limit exceeded: max {self._rpm} requests per minute.",
                },
            )

        if burst_count >= self._burst:
            logger.warning("BURST_LIMIT | ip=%s burst_count=%d", ip, burst_count)
            return JSONResponse(
                status_code=429,
                content={
                    "error": "rate_limit_error",
                    "message": f"Burst limit exceeded: max {self._burst} requests per 10 seconds.",
                },
            )

        window.append((now, 1))
        return await call_next(request)


# ---------------------------------------------------------------------------
# Request Logging Middleware — structured access log
# ---------------------------------------------------------------------------


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Structured access log for every request.

    Logs: method, path, status_code, duration_ms, ip, request_id.
    All fields are machine-parseable (use JSON log handler in production).
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = round((time.perf_counter() - start) * 1000, 2)

        request_id = response.headers.get("X-Request-ID", "—")
        ip = request.client.host if request.client else "unknown"

        logger.info(
            "ACCESS | method=%s path=%s status=%d duration_ms=%s ip=%s request_id=%s",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
            ip,
            request_id,
        )
        response.headers["X-Response-Time-Ms"] = str(duration_ms)
        return response


# ---------------------------------------------------------------------------
# Security Headers Middleware
# ---------------------------------------------------------------------------


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Add standard security response headers to every response.

    Headers added:
      X-Content-Type-Options: nosniff
      X-Frame-Options: DENY
      X-XSS-Protection: 1; mode=block
      Referrer-Policy: strict-origin-when-cross-origin
      Content-Security-Policy: default-src 'self'
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        # Allow Swagger UI to load its own resources
        if request.url.path in ("/docs", "/redoc"):
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; script-src 'self' 'unsafe-inline' cdn.jsdelivr.net; "
                "style-src 'self' 'unsafe-inline' cdn.jsdelivr.net"
            )
        else:
            response.headers["Content-Security-Policy"] = "default-src 'self'"
        return response
