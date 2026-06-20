import json
import logging
import time
from typing import Any

import redis
from celery import shared_task

from asft.core.interfaces import TrainingConfig
from asft.core.settings import get_settings
from asft.db.database import SessionLocal
from asft.db.models import Job, RoutingHistory
from asft.training.peft_trainer import PEFTTrainer

logger = logging.getLogger(__name__)


def _publish_event(job_id: str, status: str, payload: dict[str, Any] = None):
    try:
        settings = get_settings()
        r = redis.from_url(settings.celery_broker_url)
        event = {"job_id": job_id, "status": status, "payload": payload or {}}
        r.publish("job_events", json.dumps(event))
    except Exception as e:
        logger.error("Failed to publish event for job %s: %s", job_id, e)


def _update_job_status(job_id: str, status: str, result: dict[str, Any] = None, error: str = None):
    db = SessionLocal()
    try:
        job = db.query(Job).filter(Job.id == job_id).first()
        if job:
            job.status = status
            job.updated_at = time.time()
            if result:
                job.result = result
            if error:
                job.error = error
            db.commit()
    except Exception as e:
        logger.error("Failed to update job %s in db: %s", job_id, e)
    finally:
        db.close()

    _publish_event(job_id, status, {"result": result, "error": error})


@shared_task(bind=True, name="asft.workers.tasks.run_training_job")
def run_training_job(
    self, config_dict: dict[str, Any], job_id: str = None, routing_id: str = None
) -> dict[str, Any]:
    """
    Celery task that executes a PEFT training job.
    The config_dict is a serialized TrainingConfig.
    """
    job_id = job_id or self.request.id
    logger.info("Starting Celery training job %s with config: %s", job_id, config_dict)

    _update_job_status(job_id, "running")

    # Reconstruct config
    config = TrainingConfig(**config_dict)

    # Ensure the trainer runs
    trainer = PEFTTrainer()

    # PEFTTrainer blocks here
    try:
        result = trainer.train(config, job_id=job_id)

        # Format the TrainingResult as dict for JSON serialization
        result_dict = {
            "job_id": job_id,  # Override with celery task ID
            "status": result.status,
            "method": result.method,
            "final_loss": result.final_loss,
            "eval_loss": result.eval_loss,
            "steps_completed": result.steps_completed,
            "duration_seconds": result.duration_seconds,
            "checkpoint_path": result.checkpoint_path,
            "error_message": result.error_message,
        }

        # Phase 3: Optimizer Feedback Loop for Training Costs
        if routing_id:
            try:
                db = SessionLocal()
                history_record = (
                    db.query(RoutingHistory).filter(RoutingHistory.id == routing_id).first()
                )
                if history_record:
                    history_record.actual_runtime = result.duration_seconds

                    # Estimate cost dynamically based on runtime
                    # 1 GPU second ~ 0.001 cost units (placeholder heuristic)
                    history_record.actual_cost = result.duration_seconds * 0.001

                    db.commit()
            except Exception as db_err:
                logger.error(f"Failed to update RoutingHistory {routing_id}: {db_err}")
            finally:
                db.close()

        _update_job_status(job_id, result.status, result=result_dict, error=result.error_message)
        return result_dict
    except Exception as e:
        logger.exception("Task %s failed during training", job_id)
        _update_job_status(job_id, "failed", error=str(e))
        return {"job_id": job_id, "status": "failed", "error_message": str(e)}


@shared_task(bind=True, name="asft.workers.tasks.run_compression_job")
def run_compression_job(self, payload: dict[str, Any], job_id: str = None) -> dict[str, Any]:
    """
    Celery task that executes dataset compression.
    """
    job_id = job_id or self.request.id
    logger.info("Starting Celery compression job %s", job_id)

    _update_job_status(job_id, "running")

    from asft.dataset.streaming_compressor import StreamingCompressor

    compressor = StreamingCompressor()

    try:
        compressed_texts, report = compressor.compress_stream(
            dataset_path=payload["dataset_path"],
            text_field=payload.get("text_field", "text"),
            embedding_model=payload.get("embedding_model", "all-MiniLM-L6-v2"),
            output_name=payload.get("output_name", "compressed"),
        )

        result_dict = {
            "job_id": job_id,
            "original_count": report["original_count"],
            "final_count": report["final_count"],
            "total_reduction": report["total_reduction"],
            "output_path": report["output_path"],
        }

        _update_job_status(job_id, "completed", result=result_dict)
        return result_dict
    except Exception as e:
        logger.exception("Task %s failed during compression", job_id)
        _update_job_status(job_id, "failed", error=str(e))
        return {"job_id": job_id, "status": "failed", "error_message": str(e)}


@shared_task(bind=True, name="asft.workers.tasks.run_benchmark_task")
def run_benchmark_task(
    self,
    claim_type: str,
    model_path: str,
    kwargs: dict[str, Any] = None,
    job_id: str = None,
    routing_id: str = None,
) -> dict[str, Any]:
    """
    Celery task that executes a benchmark validation job.
    Updates the RoutingHistory if a routing_id is provided, forming the Optimizer feedback loop.
    """
    job_id = job_id or self.request.id
    logger.info("Starting Celery benchmark job %s for %s", job_id, claim_type)

    _update_job_status(job_id, "running")
    kwargs = kwargs or {}
    start_time = time.time()

    try:
        from asft.evaluation.benchmark_manager import BenchmarkManager

        manager = BenchmarkManager()

        if claim_type == "accuracy":
            tasks = kwargs.get("tasks", ["mmlu", "gsm8k", "truthfulqa_gen"])
            result = manager.evaluate_accuracy(model_path=model_path, tasks=tasks)
        elif claim_type == "resources":
            result = manager.evaluate_resources(model_path=model_path)
        else:
            raise ValueError(f"Unknown claim_type: {claim_type}")

        runtime = time.time() - start_time

        # Phase 3: Optimizer Feedback Loop
        if routing_id:
            try:
                db = SessionLocal()
                history_record = (
                    db.query(RoutingHistory).filter(RoutingHistory.id == routing_id).first()
                )
                if history_record:
                    history_record.actual_runtime = runtime
                    history_record.success = True

                    if claim_type == "accuracy":
                        # Assume metrics contains an 'overall_accuracy' or average it
                        avg_acc = sum(result.values()) / max(len(result), 1) if result else 0.0
                        history_record.actual_accuracy = avg_acc

                        # Calculate reward_score: e.g. accuracy / cost
                        cost = history_record.actual_cost or history_record.expected_cost or 1.0
                        history_record.reward_score = avg_acc / max(0.1, cost)

                    db.commit()
            except Exception as db_err:
                logger.error(f"Failed to update RoutingHistory {routing_id}: {db_err}")
            finally:
                db.close()

        result_dict = {
            "job_id": job_id,
            "claim_type": claim_type,
            "metrics": result,
            "routing_id": routing_id,
            "runtime": runtime,
        }

        _update_job_status(job_id, "completed", result=result_dict)
        return result_dict
    except Exception as e:
        logger.exception("Task %s failed during benchmark", job_id)
        _update_job_status(job_id, "failed", error=str(e))
        return {"job_id": job_id, "status": "failed", "error_message": str(e)}
