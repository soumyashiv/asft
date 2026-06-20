import asyncio
import logging
import time
import uuid
from typing import Any

import psutil

from asft.core.hardware_profiler import detect_hardware
from asft.db.database import SessionLocal
from asft.db.models import BenchmarkResult
from asft.evaluation.harness import LmEvalHarnessAdapter

logger = logging.getLogger(__name__)

class BenchmarkManager:
    """
    Orchestrates empirical validation of ASFT core claims.
    """
    
    def __init__(self):
        self.harness = LmEvalHarnessAdapter()

    def evaluate_accuracy(self, model_path: str, tasks: list[str] = None) -> dict[str, Any]:
        """
        Validates reasoning and hallucination claims using lm-evaluation-harness.
        """
        if tasks is None:
            tasks = ["mmlu", "gsm8k", "truthfulqa_gen"]
            
        logger.info(f"Starting Accuracy Benchmark on {model_path} for tasks {tasks}")
        
        start_time = time.time()
        try:
            results = asyncio.run(self.harness.evaluate(model_path=model_path, tasks=tasks))
            # `results` is a dict of task_name -> metrics
        except Exception as e:
            logger.error(f"Accuracy evaluation failed: {e}")
            results = {"error": str(e)}

        execution_time = time.time() - start_time
        
        metrics = {
            "tasks": tasks,
            "execution_time_seconds": execution_time,
            "results": results
        }
        
        self._save_result("accuracy", model_path, metrics)
        return metrics

    def evaluate_resources(self, model_path: str) -> dict[str, Any]:
        """
        Validates the Training Time and Resource Reduction claims.
        Runs a mocked/minimal training profile to measure VRAM and execution speed.
        """
        logger.info(f"Starting Resource Benchmark on {model_path}")
        
        # Profile hardware before
        hw = detect_hardware()
        start_vram = hw.gpus[0].vram_free_gb if hw.has_cuda else 0
        
        start_time = time.time()
        
        # Simulate loading the model to measure peak VRAM
        peak_vram = 0
        try:
            # We mock the actual training loop by sleeping or instantiating a dummy,
            # but ideally we would run a 1-step PEFT training run here.
            # For the benchmark suite, we do a dry-run instantiation.
            from asft.sparse.lora_adapter import load_quantized_model
            
            model = load_quantized_model(model_path, quantization="4bit")
            # Profile after load
            hw_after = detect_hardware()
            peak_vram = start_vram - (hw_after.gpus[0].vram_free_gb if hw_after.has_cuda else 0)
            
            # Mock 10-step dummy forward pass
            import torch
            if hw.has_cuda:
                inputs = torch.randint(0, 1000, (1, 128)).cuda()
                for _ in range(10):
                    _ = model(inputs)
                    
            del model
            if hw.has_cuda:
                torch.cuda.empty_cache()
                
        except Exception as e:
            logger.error(f"Resource benchmark failed: {e}")
            return {"error": str(e)}
            
        execution_time = time.time() - start_time
        
        metrics = {
            "peak_vram_gb_used": max(0, peak_vram),
            "execution_time_seconds": execution_time,
            "cpu_usage_percent": psutil.cpu_percent(),
            "ram_used_gb": psutil.virtual_memory().used / (1024 ** 3)
        }
        
        self._save_result("resources", model_path, metrics)
        return metrics

    def _save_result(self, claim_type: str, model_name: str, metrics: dict[str, Any]):
        """
        Saves the benchmark metrics to the SQLite/Postgres database.
        """
        with SessionLocal() as db:
            result = BenchmarkResult(
                id=str(uuid.uuid4()),
                claim_type=claim_type,
                model_name=model_name,
                metrics=metrics,
                timestamp=time.time()
            )
            db.add(result)
            db.commit()
            logger.info(f"Saved benchmark result {result.id} for {claim_type}")
