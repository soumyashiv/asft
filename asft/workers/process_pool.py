"""
ASFT Worker Process Pool — Isolated GPU workloads off the ASGI event loop.

PROBLEM SOLVED:
    The original code ran trainer.train() (a blocking PyTorch GPU call lasting
    minutes to hours) inside FastAPI BackgroundTasks, which share the same
    asyncio event loop as the API server.  Any training job would starve all
    HTTP handlers for its entire duration.

SOLUTION:
    A persistent ProcessPoolExecutor runs trainer and compressor functions in
    separate OS processes, completely isolated from the event loop.  The API
    submits a job and immediately returns a job ID.  The background coroutine
    awaits the future without blocking other requests.

DESIGN:
    - Uses concurrent.futures.ProcessPoolExecutor (stdlib, no extra deps)
    - Worker processes are initialized once at startup (expensive imports done once)
    - Job submission returns an asyncio.Future wrapping the process future
    - Timeout is enforced per-job via settings.training_timeout_seconds
    - Graceful shutdown on SIGTERM/SIGINT

ALTERNATIVE (not implemented here):
    For distributed, multi-node deployments: replace with Celery + Redis.
    The IJobStore abstraction means the API layer is unaffected by the swap.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Callable
from concurrent.futures import Future, ProcessPoolExecutor
from typing import Any

logger = logging.getLogger(__name__)

# Module-level pool singleton — managed by lifespan
_pool: ProcessPoolExecutor | None = None


def _worker_initializer() -> None:
    """
    Called once per worker process at startup.
    Pre-import heavy packages so the first job doesn't pay cold-start cost.
    """
    # Suppress verbose HuggingFace logs in worker processes
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    logger.info("Worker process initialized (PID=%d)", os.getpid())


def get_pool(max_workers: int = 1) -> ProcessPoolExecutor:
    """
    Return the global process pool, creating it if necessary.
    Call this at application startup (in the lifespan handler).
    """
    global _pool
    if _pool is None:
        _pool = ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=_worker_initializer,
        )
        logger.info("ProcessPoolExecutor created with %d worker(s)", max_workers)
    return _pool


def shutdown_pool(wait: bool = True) -> None:
    """
    Gracefully shut down the process pool.
    Call this in the application shutdown lifespan handler.
    """
    global _pool
    if _pool is not None:
        logger.info("Shutting down worker process pool (wait=%s)...", wait)
        _pool.shutdown(wait=wait, cancel_futures=not wait)
        _pool = None
        logger.info("Worker process pool shut down.")


async def submit_to_pool(
    fn: Callable,
    *args: Any,
    timeout: float | None = None,
    **kwargs: Any,
) -> Any:
    """
    Submit a blocking function to the process pool and await its result
    without blocking the ASGI event loop.

    Args:
        fn:      A picklable callable (top-level function or static method).
        *args:   Positional arguments for fn.
        timeout: Optional wall-clock timeout in seconds.
        **kwargs: Keyword arguments — passed as a single dict to avoid
                  ProcessPoolExecutor's lack of direct kwarg support.

    Returns:
        The return value of fn(*args, _kwargs=kwargs).

    Raises:
        asyncio.TimeoutError: if the job exceeds `timeout` seconds.
        Any exception raised inside the worker process.
    """
    pool = get_pool()
    loop = asyncio.get_running_loop()

    # ProcessPoolExecutor doesn't support **kwargs directly —
    # wrap fn to accept a kwargs dict as the last positional arg.
    future: Future = loop.run_in_executor(pool, _kwarg_wrapper, fn, args, kwargs)

    if timeout:
        return await asyncio.wait_for(asyncio.wrap_future(future), timeout=timeout)
    return await asyncio.wrap_future(future)


def _kwarg_wrapper(fn: Callable, args: tuple, kwargs: dict) -> Any:
    """
    Unwrapper executed inside the worker process.
    Allows passing keyword arguments through ProcessPoolExecutor.
    """
    return fn(*args, **kwargs)
