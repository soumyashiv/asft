"""
ASFT FastAPI Server — REST API exposing all framework capabilities.
"""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from asft.core.config import ASFTConfig
from asft.core.hardware_profiler import detect_hardware
from asft.core.registry import registry

logger = logging.getLogger(__name__)

_config: Optional[ASFTConfig] = None
_hardware = None
_memory_manager = None
_start_time: float = time.time()
_jobs: Dict[str, Dict] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _config, _hardware, _memory_manager
    _config = ASFTConfig()
    _config.ensure_dirs()
    _hardware = detect_hardware()
    _config.apply_hardware_profile(_hardware)
    try:
        from asft.memory.memory_manager import MemoryManager
        _memory_manager = MemoryManager(config=_config)
    except Exception as e:
        logger.warning("Memory init failed: %s", e)
    _register_skills()
    logger.info("ASFT API ready")
    yield
    logger.info("ASFT API shutdown")


def _register_skills():
    try:
        from asft.skills.packs.coding import CodingSkillPack
        from asft.skills.packs.research import ResearchSkillPack
        from asft.skills.packs.planning import PlanningSkillPack
        from asft.skills.packs.mathematics import MathematicsSkillPack
        from asft.skills.packs.trading import TradingSkillPack
        from asft.skills.packs.automation import AutomationSkillPack
        for Pack in [CodingSkillPack, ResearchSkillPack, PlanningSkillPack,
                     MathematicsSkillPack, TradingSkillPack, AutomationSkillPack]:
            p = Pack()
            registry.register_skill(p.meta.name, p)
    except Exception as e:
        logger.warning("Skill registration failed: %s", e)


app = FastAPI(title="ASFT API", version="0.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class TrainRequest(BaseModel):
    model_name: str = "Qwen/Qwen2-0.5B"
    dataset_path: str
    max_steps: int = Field(100, ge=1)
    sparsity_ratio: float = Field(0.95, ge=0.0, le=1.0)
    method: str = "asft"


class MemoryQueryRequest(BaseModel):
    query: str
    top_k: int = 5


class FactRequest(BaseModel):
    subject: str
    predicate: str
    object: str
    source: str = "api"
    confidence: float = 1.0


class SkillRouteRequest(BaseModel):
    task: str
    strategy: str = "single"
    top_k: int = 2


class CompressRequest(BaseModel):
    dataset_path: str
    text_field: str = "text"
    output_name: str = "compressed"


@app.get("/")
async def root():
    return {"name": "ASFT API", "version": "0.1.0"}


@app.get("/status")
async def status():
    return JSONResponse({
        "uptime_seconds": round(time.time() - _start_time, 1),
        "hardware": {
            "has_cuda": getattr(_hardware, "has_cuda", False),
            "method": getattr(_hardware, "recommended_training_method", "unknown"),
        },
        "memory": _memory_manager.stats() if _memory_manager else {},
        "skills": registry.list("skill_packs"),
        "jobs": len(_jobs),
    })


@app.get("/hardware")
async def hardware_info():
    if not _hardware:
        raise HTTPException(503, "Hardware not detected")
    return JSONResponse({
        "cpu": _hardware.cpu_brand,
        "ram_gb": _hardware.ram_available_gb,
        "gpus": [{"name": g.name, "vram_gb": g.vram_total_gb} for g in _hardware.gpus],
        "recommendations": {
            "precision": _hardware.recommended_precision,
            "quantization": _hardware.recommended_quantization,
            "method": _hardware.recommended_training_method,
            "batch_size": _hardware.recommended_batch_size,
        },
    })


@app.post("/train", status_code=202)
async def train(req: TrainRequest, bg: BackgroundTasks):
    job_id = f"train_{int(time.time())}"
    _jobs[job_id] = {"status": "queued", "method": req.method}

    def _run():
        try:
            _jobs[job_id]["status"] = "running"
            logger.info("Training job %s started: %s", job_id, req.method)
            time.sleep(1)  # Placeholder — real training wired via SparseTrainer
            _jobs[job_id]["status"] = "completed"
        except Exception as e:
            _jobs[job_id] = {"status": "failed", "error": str(e)}

    bg.add_task(_run)
    return {"job_id": job_id, "status": "queued"}


@app.get("/train/{job_id}")
async def job_status(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(404, "Job not found")
    return _jobs[job_id]


@app.get("/skills")
async def list_skills():
    skills = []
    for name in registry.list("skill_packs"):
        p = registry.get_or_none("skill_packs", name)
        if p and hasattr(p, "meta"):
            skills.append({
                "name": p.meta.name, "description": p.meta.description,
                "domain": p.meta.domain, "score": p.meta.performance_score,
            })
    return {"skills": skills}


@app.post("/skills/route")
async def route_skill(req: SkillRouteRequest):
    from asft.skills.skill_router import SkillRouter
    router = SkillRouter(registry=registry)
    d = router.route(req.task, top_k=req.top_k, strategy=req.strategy)
    return {"selected": d.selected_packs, "scores": {k: round(v, 4) for k, v in d.scores.items()}}


@app.post("/skills/{skill_name}/process")
async def process_skill(skill_name: str, body: Dict[str, Any]):
    p = registry.get_or_none("skill_packs", skill_name)
    if not p:
        raise HTTPException(404, f"Skill '{skill_name}' not found")
    task = body.get("task", "")
    if not task:
        raise HTTPException(400, "Missing 'task'")
    r = p.process(task)
    return {"output": str(r.output), "confidence": r.confidence, "duration_seconds": r.duration_seconds}


@app.post("/memory/query")
async def query_memory(req: MemoryQueryRequest):
    if not _memory_manager:
        raise HTTPException(503, "Memory not initialized")
    results = _memory_manager.query(req.query, top_k=req.top_k)
    return {
        "can_answer": _memory_manager.can_answer_from_memory(req.query),
        "results": [{"source": r.source, "content": str(r.content)[:300], "confidence": r.confidence} for r in results],
    }


@app.post("/memory/facts")
async def add_fact(req: FactRequest):
    if not _memory_manager:
        raise HTTPException(503, "Memory not initialized")
    fid = _memory_manager.learn_fact(req.subject, req.predicate, req.object, req.source, req.confidence)
    return {"fact_id": fid}


@app.get("/memory/stats")
async def memory_stats():
    if not _memory_manager:
        raise HTTPException(503, "Memory not initialized")
    return _memory_manager.stats()


@app.post("/dataset/compress", status_code=202)
async def compress_dataset(req: CompressRequest, bg: BackgroundTasks):
    job_id = f"compress_{int(time.time())}"
    _jobs[job_id] = {"status": "queued"}

    def _run():
        try:
            _jobs[job_id]["status"] = "running"
            from asft.dataset.compressor import DatasetCompressor
            c = DatasetCompressor(_config.dataset if _config else None)
            _, report = c.compress_jsonl(req.dataset_path, text_field=req.text_field, output_name=req.output_name)
            _jobs[job_id] = {"status": "completed", "report": report}
        except Exception as e:
            _jobs[job_id] = {"status": "failed", "error": str(e)}

    bg.add_task(_run)
    return {"job_id": job_id}


def run_server(host: str = "0.0.0.0", port: int = 8080, reload: bool = False):
    import uvicorn
    uvicorn.run("asft.api.server:app", host=host, port=port, reload=reload)
