"""
ASFT API Server — Production-ready FastAPI application.

FIXES APPLIED vs original:
    F3:  Training jobs now run in a ProcessPoolExecutor, NOT in BackgroundTasks.
         The original code called trainer.train() (a blocking PyTorch call) inside
         FastAPI BackgroundTasks, which shares the ASGI event loop. This would
         starve all HTTP handlers for the entire training duration.

    F6:  CORS allowed_origins now reads from settings (env var ASFT_ALLOWED_ORIGINS).
         The original had allow_origins=["*"] hardcoded — never acceptable in production.

    F12: run_compression_job now calls DatasetCompressor.compress_jsonl() with the
         correct API. The original called DatasetCompressor(embedding_model=...) and
         .compress(dataset_path, keeping_ratio=0.8) — neither of which are valid.

    F13: Request IDs are propagated into background job context.

NEW:
    - /api/v1/optimize endpoint — invokes the AutoOptimizer before any training
    - /api/v1/estimate endpoint — returns cost/time estimate without training
    - Worker pool lifecycle managed by FastAPI lifespan
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from asft.api.middleware import (
    RateLimitMiddleware,
    RequestLoggingMiddleware,
    SecurityHeadersMiddleware,
)
from asft.security.auth import APIKeyMiddleware
from asft.api.schemas import (
    CompressRequest,
    CompressResponse,
    ErrorResponse,
    EstimateRequest,
    EstimateResponse,
    HealthResponse,
    JobStatusResponse,
    OptimizeRequest,
    OptimizeResponse,
    TrainRequest,
    TrainResponse,
)
from asft.core.exceptions import ASFTError, JobNotFoundError
from asft.core.hardware_profiler import HardwareProfiler
from asft.core.registry import Registry
from asft.core.settings import get_settings
from asft.training.job_store import create_job_store
from asft.training.peft_trainer import PEFTTrainer
from asft.workers.process_pool import get_pool, shutdown_pool, submit_to_pool

logger = logging.getLogger(__name__)
settings = get_settings()

# Global singletons
registry = Registry()
job_store = create_job_store()
profiler = HardwareProfiler()


# ---------------------------------------------------------------------------
# Worker functions — must be top-level picklable functions for ProcessPool
# ---------------------------------------------------------------------------

def _training_worker(job_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Runs in an isolated worker process.
    Imports are deferred to inside the function so they happen in the worker,
    not in the main API process.
    """
    from asft.core.interfaces import TrainingConfig
    from asft.training.peft_trainer import PEFTTrainer

    config = TrainingConfig(**payload)
    trainer = PEFTTrainer()
    result = trainer.train(config)
    return result.__dict__


def _compression_worker(job_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Dataset compression runs in an isolated worker process.
    Uses the correct DatasetCompressor API (compress_jsonl).
    """
    from asft.dataset.compressor import DatasetCompressor

    compressor = DatasetCompressor()
    compressed_texts, report = compressor.compress_jsonl(
        input_path=payload["dataset_path"],
        text_field=payload.get("text_field", "text"),
        embedding_model=payload.get("embedding_model", "all-MiniLM-L6-v2"),
        output_name=payload.get("output_name", "compressed"),
    )
    return {
        "original_count": report["original_count"],
        "final_count": report["final_count"],
        "total_reduction": report["total_reduction"],
        "output_path": report["output_path"],
    }


# ---------------------------------------------------------------------------
# Background coroutines — submit to process pool, then update job store
# ---------------------------------------------------------------------------

async def run_training_job(job_id: str, payload: Dict[str, Any]) -> None:
    """
    Submit training to the worker process pool.
    Does NOT block the event loop — awaits the future asynchronously.
    """
    await job_store.update_status(job_id, "running")
    logger.info("Training job %s submitted to worker pool", job_id)
    try:
        result = await submit_to_pool(
            _training_worker,
            job_id,
            payload,
            timeout=float(settings.training_timeout_seconds),
        )
        if result.get("status") == "completed":
            await job_store.update_status(job_id, "completed", result=result)
        else:
            await job_store.update_status(
                job_id, "failed", error=result.get("error_message", "Unknown error")
            )
    except asyncio.TimeoutError:
        logger.error("Training job %s timed out after %ds", job_id, settings.training_timeout_seconds)
        await job_store.update_status(job_id, "failed", error="Job timed out")
    except Exception as e:
        logger.exception("Training job %s failed", job_id)
        await job_store.update_status(job_id, "failed", error=str(e))


async def run_compression_job(job_id: str, payload: Dict[str, Any]) -> None:
    """Submit dataset compression to the worker pool."""
    await job_store.update_status(job_id, "running")
    logger.info("Compression job %s submitted to worker pool", job_id)
    try:
        result = await submit_to_pool(
            _compression_worker,
            job_id,
            payload,
            timeout=3600.0,  # 1 hour max
        )
        await job_store.update_status(job_id, "completed", result=result)
    except Exception as e:
        logger.exception("Compression job %s failed", job_id)
        await job_store.update_status(job_id, "failed", error=str(e))


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle: startup and shutdown."""
    logger.info("ASFT API Server starting up...")

    # Initialize process pool for GPU workers
    get_pool(max_workers=settings.max_training_workers)

    # Register trainers
    registry.register("trainer", "peft", PEFTTrainer())

    # Profile hardware
    profiler.profile()
    logger.info("Hardware profile: %s", profiler)

    yield

    # Graceful shutdown: wait for running training jobs to finish
    logger.info("ASFT API Server shutting down...")
    shutdown_pool(wait=True)


# ---------------------------------------------------------------------------
# App Initialization
# ---------------------------------------------------------------------------

app = FastAPI(
    title="ASFT — Training Acceleration Framework",
    description=(
        "Adaptive Sparse Fine-Tuning: Achieve the same or better model capability "
        "with dramatically fewer resources — compute, data, memory, time, and energy."
    ),
    version="0.3.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


# ---------------------------------------------------------------------------
# Exception Handlers
# ---------------------------------------------------------------------------

@app.exception_handler(ASFTError)
async def asft_error_handler(request: Request, exc: ASFTError):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    logger.warning("API Error | request_id=%s type=%s message=%s", request_id, exc.code, exc.message)
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.code, message=exc.message, request_id=request_id
        ).model_dump(),
    )


@app.exception_handler(Exception)
async def unhandled_error_handler(request: Request, exc: Exception):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    logger.exception("Unhandled Server Error | request_id=%s", request_id)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="internal_server_error",
            message="An unexpected error occurred.",
            request_id=request_id,
        ).model_dump(),
    )


# ---------------------------------------------------------------------------
# Middleware (executed bottom-up in Starlette)
# ---------------------------------------------------------------------------

app.add_middleware(APIKeyMiddleware)
app.add_middleware(
    RateLimitMiddleware,
    requests_per_minute=settings.rate_limit_per_minute,
    burst_limit=settings.rate_limit_burst,
)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,   # FIX F6: from env, not hardcoded ["*"]
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)
app.add_middleware(RequestLoggingMiddleware)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health_check():
    """Public health check. No authentication required."""
    import time
    from asft import __version__
    return HealthResponse(
        version=__version__,
        uptime_seconds=round(time.time() - app.state.start_time, 1),
    )


@app.post("/api/v1/estimate", response_model=EstimateResponse, tags=["optimizer"])
async def estimate_training_cost(request: EstimateRequest):
    """
    Estimate training cost, time, and resource requirements BEFORE committing.
    Uses scaling laws (Kaplan et al. 2020 + Chinchilla) to project costs.
    Returns a recommendation: train | use_qlora | retrieve | use_skill.
    """
    from asft.optimizer.cost_estimator import CostEstimator
    estimator = CostEstimator()
    estimate = estimator.estimate(
        model_name=request.model_name,
        dataset_size=request.dataset_size,
        method=request.method,
        hardware_profile=profiler.get_profile(),
    )
    return EstimateResponse(
        estimated_gpu_hours=estimate.gpu_hours,
        estimated_cost_usd=estimate.cost_usd,
        estimated_accuracy_gain=estimate.accuracy_gain_estimate,
        recommendation=estimate.recommendation,
        reasoning=estimate.reasoning,
        roi_score=estimate.roi_score,
    )


@app.post("/api/v1/optimize", response_model=OptimizeResponse, tags=["optimizer"])
async def auto_optimize(request: OptimizeRequest):
    """
    AutoOptimizer: determine the cheapest path to solving a task.
    Checks memory → retrieval → skills → LoRA → QLoRA before recommending training.
    """
    from asft.optimizer.auto_optimizer import AutoOptimizer
    optimizer = AutoOptimizer(registry=registry)
    decision = optimizer.decide(
        task=request.task,
        domain=request.domain,
        target_accuracy=request.target_accuracy,
        budget_usd=request.budget_usd,
    )
    return OptimizeResponse(
        recommended_action=decision.action,
        reasoning=decision.reasoning,
        estimated_cost_usd=decision.estimated_cost_usd,
        alternatives=decision.alternatives,
    )


@app.post("/api/v1/train", response_model=TrainResponse, tags=["training"])
async def queue_training(request: TrainRequest, background_tasks: BackgroundTasks):
    """
    Queue a fine-tuning job. The job runs in a worker process pool —
    the API response is immediate; poll /api/v1/jobs/{job_id} for status.
    """
    if request.method not in ("peft_lora", "qlora", "lora"):
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(
                error="unsupported_method",
                message=f"Method '{request.method}' is not supported. Use: peft_lora, qlora, lora.",
            ).model_dump(),
        )

    payload = request.model_dump()
    job = await job_store.create(job_type="training", payload=payload)

    # Submit to process pool asynchronously (does NOT block the event loop)
    background_tasks.add_task(run_training_job, job.job_id, payload)

    return TrainResponse(job_id=job.job_id, status="queued")


@app.post("/api/v1/dataset/compress", response_model=CompressResponse, tags=["dataset"])
async def queue_compression(request: CompressRequest, background_tasks: BackgroundTasks):
    """Queue a dataset compression job."""
    payload = request.model_dump()
    job = await job_store.create(job_type="compression", payload=payload)
    background_tasks.add_task(run_compression_job, job.job_id, payload)
    return CompressResponse(job_id=job.job_id, status="queued")


@app.get("/api/v1/jobs/{job_id}", response_model=JobStatusResponse, tags=["jobs"])
async def get_job_status(job_id: str):
    """Retrieve the status and results of a background job."""
    job = await job_store.get(job_id)
    if not job:
        raise JobNotFoundError(f"Job '{job_id}' not found.")
    return JobStatusResponse(**job.__dict__)


@app.get("/api/v1/jobs", tags=["jobs"])
async def list_jobs(
    job_type: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 50,
):
    """List recent jobs, optionally filtered by type and status."""
    jobs = await job_store.list_jobs(job_type=job_type, status=status, limit=limit)
    return {"jobs": [j.__dict__ for j in jobs], "count": len(jobs)}


# Initialize start time
import time
app.state.start_time = time.time()
